# Vanus: a transformer in Java and Scala

Pure Java numerical kernels and autograd, with Scala training commands. Both
languages target and run on Java 21; there is no Python, PyTorch, native BLAS, or
pretrained weights. This is a small educational ML core, not a complete PyTorch
implementation or a claim that JVM
transformers have never existed.

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

The demo trains from random weights for 200 steps on one greeting pair. This is
memorization, not evidence of conversational generalization. Inference generates
from learned logits; it contains no answer lookup. Optional demo arguments are
`demo <steps> <checkpoint>`. Custom data is UTF-8 prompt/answer pairs separated by
one literal tab, one pair per line. Empty fields and overlong examples fail.

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
