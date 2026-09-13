# Vanus: a transformer in Java and Scala

Pure Java numerical kernels and autograd, with Scala training commands. Both
languages target and run on Java 21; there is no Python, PyTorch, native BLAS, or
pretrained weights. This is a small educational ML core, not a complete PyTorch
implementation or a claim that JVM
transformers have never existed.

## Is this a real LLM?

Architecturally, yes: causal self-attention, RoPE, RMSNorm, SwiGLU, backprop
through an autograd tape, AdamW, and autoregressive next-byte prediction via
cross-entropy are the same pieces production transformers use (see
Architecture below). Nothing special-cases words or does string matching;
every response comes out of matrix multiplications over learned weights.

What it is not is large. `tiny` has 18,656 parameters, 1 attention block, and
trains on dozens of examples with a byte-level (not subword) tokenizer, versus
billions of parameters and trillions of tokens for something like GPT or
Llama. At this scale, the easiest thing for gradient descent to learn is
memorizing exact training byte sequences, not generalizing across novel
phrasing, so a heavily trained tiny model behaves like a lookup table for
prompts it saw verbatim and produces noise for anything else. It is a genuine,
fully-inspectable implementation of how an LLM works mechanically, not a
smaller version of a production model's capabilities.

## Run

Use JDK 21 exactly and sbt. Vanus checks the runtime version at startup and exits
when launched with another Java release. Set `JAVA_HOME` to a JDK 21 installation
before running the commands:

```sh
sbt test
sbt 'run info'
sbt 'run demo'
sbt 'run chat checkpoints/greeting.vanus Hello, how are you doing'
sbt 'run train tiny data/greetings.tsv 300 checkpoints/custom.vanus'
sbt 'run train 20m data/greetings.tsv 10 checkpoints/vanus20m.vanus'
```

The demo trains from random weights for 200 steps on Pride and Prejudice sentence
pairs (`data/pride-and-prejudice.tsv`), filtered down to whichever pairs fit the
tiny model's byte context. This is memorization on a tiny model, not evidence of
conversational generalization. Inference generates from learned logits; it
contains no answer lookup. Optional demo arguments are
`demo <steps> <checkpoint>`. Custom data is UTF-8 prompt/answer pairs separated by
one literal tab, one pair per line. Empty fields and overlong examples fail.

`data/dictionary-raw.tsv` is a small word/definition/example source (one
`word<TAB>definition<TAB>example` entry per line). `data/dictionary_to_tsv.py`
turns it into `data/dictionary.tsv` prompt/answer pairs ("What does X mean?" and
"Use X in a sentence.") the same way `data/book_to_tsv.py` turns a raw novel into
sentence-continuation pairs. `demo2dict` trains on `data/dictionary2.tsv`
(generated from `data/dictionary-raw2.tsv`, a smaller word list) so it stays
inside `tiny`'s context and converges faster than the full dictionary set:

```sh
python3 data/dictionary_to_tsv.py data/dictionary-raw.tsv data/dictionary.tsv
python3 data/dictionary_to_tsv.py data/dictionary-raw2.tsv data/dictionary2.tsv
sbt 'run train tiny data/dictionary.tsv 300 checkpoints/dictionary.vanus'
sbt 'run demo2dict'
```

The 20M command is a smoke run, not sufficient training. Start with tiny: scalar
CPU loops at full size are slow. `chat` is single-turn greedy completion. The Java
generation API also supports temperature and top-k sampling. Generation stops
at EOS, the token limit, or the context boundary; no KV cache or sliding window.

## Architecture

Inspired by the [LLaMA paper](https://arxiv.org/abs/2302.13971): pre-RMSNorm,
rotary positions, causal multi-head attention, and SwiGLU. Vanus uses its own
implementation, dimensions, byte tokenizer, and tied embedding/output weights.
It is not compatible with LLaMA checkpoints.

| Setting | Tiny | Vanus 20M |
| --- | ---: | ---: |
| Vocabulary | 260 | 260 |
| Width | 32 | 448 |
| FFN width | 64 | 1280 |
| Blocks | 1 | 8 |
| Heads | 4 | 8 |
| Context (byte/special tokens) | 128 | 256 |
| Parameters | 18,656 | 20,309,184 |

Parameter count: `V*d + L*(4*d*d + 3*d*f + 2*d) + d`. No linear biases.
UTF-8 byte IDs 0..255 plus BOS/EOS/user/assistant IDs 256..259 avoid a tokenizer
dependency. This is not BPE. Complete input round trips losslessly; invalid
generated UTF-8 decodes with replacement characters. Context is measured in
bytes, so multilingual text can fill it quickly.

Full-size weights, gradients and Adam moments occupy about 310 MiB before
activations and JVM overhead. sbt run uses a 2 GiB heap. Inference disables the
graph but still allocates parameter gradient arrays.

## Learning Path

Read `src/main/java/org/berlin/vanus/Tensor.java`, then `ByteTokenizer.java`,
`Transformer.java`, `AdamW.java`, and finally `src/main/scala/Main.scala`.

```text
IDs [T] -> embeddings [T,d]
  -> repeated blocks:
       RMSNorm -> Q,K,V [T,d] -> RoPE on Q,K
       causal softmax(Q K^T / sqrt(headWidth)) V per head
       output projection -> residual addition
       RMSNorm -> SiLU(gate) * up [T,f] -> down [T,d] -> residual
  -> RMSNorm -> tied output projection -> logits [T,V]
  -> next-token cross entropy -> scalar loss
  -> backward -> global gradient clipping -> AdamW
```

Attention only reads positions `s <= t`. Softmax subtracts the maximum for
stability. Autograd records tensor operations, not individual scalar operations.
Attention and RMSNorm register their explicit backward formulas on that tape.
The trainer samples examples with replacement at batch size one, masks prompt
targets with -1, and supervises the first answer token through EOS. Learning rate
uses ten-step warmup and cosine decay. Gradients accumulate until `zeroGrad`;
do not mutate weights between forward and backward.

Checkpoints store versioned configuration and named weights. Optimizer moments
and RNG state are not saved, so loading supports inference, not exact resume.

## PyTorch Checklist Mapping

| Concepts | Implemented equivalent / limit |
| --- | --- |
| Tensor, creation, shape/dtype/device | `of`, `zeros`, `parameter`, `shape`; contiguous 2-D float32 CPU |
| matmul, transpose, elementwise | `matmul`, `transpose`, `add`, `multiply`; exact shapes |
| autograd, requires_grad, backward, grad | Construction flag, reverse tape, `backward`, `grad` |
| no_grad, detach | `noGrad`, `detach` (copy) |
| Module, Parameter, parameters, state_dict | `Transformer`, trainable tensors, named registry, `save`/`load` |
| Linear, Embedding | `matmul`, `embedding` with scatter-add gradients |
| Positions, normalization, activations | RoPE, RMSNorm, SiLU/SwiGLU; no LayerNorm/GELU/dropout |
| Sequential, ModuleList, residual blocks | Explicit Java block list and `forward` |
| Softmax, mask, Q/K/V, heads | Fused differentiable causal `attention` |
| Logits, shifted targets, cross entropy | `forward`, Scala target shifting, stable `crossEntropy` |
| Vocabulary, IDs, tokenizer | UTF-8 byte vocabulary; no BPE |
| AdamW, clipping, zero_grad, scheduler | `AdamW`, global norm clipping, `zeroGrad`, Scala schedule |
| Argmax, temperature, top-k | `generate`; temperature zero is greedy |
| Dataset, DataLoader, batching | In-memory TSV pairs, random sampling, batch size one |
| Validation loss, perplexity | Not automated; held-out loss via `noGrad`, perplexity = exp(mean loss) |
| NumPy, general tensor utilities | Java arrays; no arbitrary ranks, broadcasting, views, reshape, permute, slicing, cat/stack, or general public reductions |

Next steps for broader language learning: held-out evaluation, a larger licensed
corpus, streaming batches, optimizer resume, BPE, and faster matrix kernels.
The current TSV trainer is supervised response training, not a large-corpus
pretraining pipeline. General tensor APIs can be added as models need them.

Tests cover finite-difference gradients across parameter groups, causal
invariance, gradient accumulation, tokenizer and checkpoint round trips, loss
reduction, and invalid inputs. These verify mechanics, not language quality.
