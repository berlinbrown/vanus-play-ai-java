# Vanus: a transformer in Java and Scala

Vanus is a small educational decoder-only transformer. Java implements float32
tensor operations, autograd, attention, the transformer, checkpointing, and
Swing visualization. Scala implements command parsing, TSV loading, training,
and evaluation. Use JDK 21 or newer.

## Recommended model with Swing

```sh
sbt 'run demo2greet'
```

This uses the configuration and dataset with the best verified results. It opens
Swing after loading `checkpoints/conversation.vanus`. If the checkpoint is
missing, it trains for 2,000 steps first. Try `Hellow`, `How are you doing`, or
`Good morning`.

To intentionally retrain from random weights and replace the checkpoint:

```sh
sbt 'run demo2greet 2000'
```

For a larger but unverified capacity experiment:

```sh
sbt 'run demo2daily200k 10000'
```

The 200K command trains on the 300-pair DailyDialog starter set and then opens
Swing. A larger parameter count does not guarantee better conversation.

## Swing controls

- **Send** generates one deterministic, greedy reply from the typed prompt.
- The dropdown copies a known training prompt into the input field.
- **Talk to itself** sends `hello` immediately. Every two seconds it randomly
  selects words from the previous reply, preserves their original order, and
  sends them as the next prompt.
- **Stop self-talk** prevents future automatic turns after the current reply.
- The transcript shows numbered and timestamped self-talk turns.
- The live trace shows prompt token IDs, context usage, selected tokens, the five
  most likely next-token candidates, and the stopping reason.

Generation runs in a background `SwingWorker`; UI changes and the timer run on
Swing's event thread. A timer tick is skipped while generation is busy, so
generations never overlap. Empty or nonword replies fall back to `hello`.
Self-talk changes the next input but does not train or modify model weights.

## Complete application command reference

Run commands from this directory as `sbt 'run <command> <arguments>'`.
Square brackets mark optional arguments; angle brackets mark required ones.
Do not type the brackets.

| Command | Behavior | Default steps / batch | Checkpoint |
|---|---|---:|---|
| `info` | Print configurations, commands, and argument rules | — | — |
| `greetings [steps]` | Load/train greetings and evaluate in the terminal | 2,000 / 4 | `checkpoints/conversation.vanus` |
| `demo2greet [steps]` | Same greeting evaluation, then open Swing | 2,000 / 4 | `checkpoints/conversation.vanus` |
| `demo2daily [steps]` | Tiny model on the 300-pair DailyDialog starter, then Swing | 10,000 / 1 | `checkpoints/dailydialog-starter.vanus` |
| `demo2daily200k [steps]` | 200K model on the same starter set, then Swing | 10,000 / 1 | `checkpoints/dailydialog-200k.vanus` |
| `demo [steps] [checkpoint]` | Always train tiny on short Pride and Prejudice pairs; terminal only | 200 / 1 | Default: `checkpoints/greeting.vanus` |
| `demo2 [steps]` | Load/train tiny on Pride and Prejudice, then Swing | 200 / 1 | `checkpoints/demo2.vanus` |
| `demo2dict [steps]` | Load/train the dictionary preset on `dictionary2.tsv`, then Swing | 200 / 8 | `checkpoints/demo2-dictionary.vanus` |
| `weights <checkpoint>` | Inspect real tensors from one checkpoint in Swing | No training | Required input path |
| `weights <current> <reference>` | Compare matching checkpoints and visualize `current − reference` | No training | Two required input paths |
| `train <tiny\|200k\|1m\|20m> <pairs.tsv> <steps> <checkpoint> [batch]` | Train prompt/reply data with automatic light typo augmentation | Required / 1 | Required output path |
| `pretrain <tiny\|200k\|1m\|20m> <text> <steps> <checkpoint> [batch] [bpe-vocab]` | Learn BPE and continuously predict raw text | Required / 1 / 512 | Required output path |
| `continue <checkpoint> <pairs.tsv> <steps> <output> [batch]` | Continue saved training with automatic light typo augmentation | Required / 1 | Required input and output paths |
| `continue-pretrain <checkpoint> <text> <steps> <output> [batch]` | Continue raw-text prediction with the saved tokenizer and AdamW state | Required / 1 | Required input and output paths |
| `gui <checkpoint> [pairs.tsv]` | Open any saved model in Swing without training; optional TSV fills the prompt dropdown | No training | Required input path |
| `server <checkpoint> <pairs.tsv> [server options]` | Run continuous headless self-talk, persist it to SQLite, and expose two authenticated JSON routes | No training | Required checkpoint and TSV |
| `chat <checkpoint> <prompt...>` | Load weights and generate one terminal reply | No training | Required input path |
| `eval <checkpoint> <pairs.tsv>` | Print replies, exact matches, answer-token accuracy, and loss | No training | Required input path |

## Headless AI server

The server loads an existing checkpoint, starts the same style of automatic
self-talk as the Swing application, and runs until interrupted. It generates a
turn about every 30 seconds and injects either `hello` or `goodbye` every five
minutes. Both the prompt and response are appended to `<dir>/vanus.db`; an
existing database is reused. SQLite uses WAL mode, a 5-second busy timeout, and
foreign-key enforcement. The newest 1,000 messages are retained.

```sh
sbt 'run server checkpoints/for-scatty-1m-2k.vanus data/scatty-language/scatty.tsv -p 8086 -d /Users/berlinbrown/Documents/Github/vanus-play-framework/core/server/vanusplay/db --rate-limit 200 -h 127.0.0.1'

./scripts/launch-local-vanus-ai.sh -p 8086 -d /Users/berlinbrown/Documents/Github/vanus-play-framework/core/server/vanusplay/db --rate-limit 200 -h 127.0.0.1
```

The script defaults to the checkpoint and dataset shown above. Override them
with `--checkpoint <path>` and `--data <path>`. The `-d` directory is created if
needed and contains `vanus.db`. The AI server serves REST routes only; unrelated
paths return 404.

The launcher can be called from any directory using its full path. Its default
checkpoint, dataset, and database directory are resolved to full paths from the
script's location:

```sh
/Users/berlinbrown/gitmain5/git_dir/Github/vanus-play-ai-java/core/vanus-ai-java/scripts/build-deployment.sh

/Users/berlinbrown/gitmain5/git_dir/Github/vanus-play-ai-java/core/vanus-ai-java/scripts/launch-vanus-ai.sh
```

`build-deployment.sh` creates the self-contained executable JAR at
`target/scala-3.9.0/vanus-ai-java.jar`. The production launcher runs this JAR
directly and requires Java 21 or newer; it does not require sbt at runtime. Both
Java and Scala compilation target Java 21 bytecode, even when the build itself
runs under a newer JDK. Set `JAVA_BIN` to a specific Java 21 executable when
needed.

The deliberately simple local access token is:

```text
dmFudXMtc2NhdHR5LW9wcy0yMDI2
```

```sh
curl 'http://127.0.0.1:8086/_vanus-ops-manage/_vanus-bot-status-info?token=dmFudXMtc2NhdHR5LW9wcy0yMDI2'

curl 'http://127.0.0.1:8086/_vanus-ops-manage/_vanus-bot-messages?token=dmFudXMtc2NhdHR5LW9wcy0yMDI2'
```

The status operation returns the number currently stored. The messages
operation returns at most 140 rows, newest first, with `id`, UTC `timestamp`,
`role`, and `message`. API responses use `application/json` and `Cache-Control:
no-store`. Keep the default loopback host because the hard-coded token is only a
small guard for local experimentation.

Bare `sbt run` behaves like `sbt 'run info'`. Step counts must be positive whole
numbers. Supplying `[steps]` to a demo always starts over from random weights and
replaces its checkpoint. Without `[steps]`, the command loads its checkpoint if
present or trains with the listed default if absent. The plain `demo` command is
the exception: it always trains.

Current training checkpoints contain the model configuration, tokenizer,
weights, AdamW moments, and completed update count. `continue` restores them.
Its sampling RNG and learning-rate schedule restart for the new run, so it is a
practical continuation rather than a bit-exact uninterrupted run. Version 1
checkpoints still load for inference but cannot continue. The GUI launchers still read their TSV files for dataset counts
and the prompt dropdown; generation itself does not look up answers in the TSV.

## Model configurations

| Preset | Parameters | Width | FFN width | Layers | Heads | Context |
|---|---:|---:|---:|---:|---:|---:|
| `tiny` | 18,656 | 32 | 64 | 1 | 4 | 128 |
| `dictionary` | 98,880 | 64 | 128 | 2 | 4 | 256 |
| `200k` | 200,544 | 96 | 176 | 2 | 4 | 128 |
| `1m` | 1,067,040 | 160 | 320 | 4 | 8 | 256 |
| `20m` | 20,309,184 | 448 | 1,280 | 8 | 8 | 256 |

The generic `train` command accepts `tiny`, `200k`, `1m`, and `20m`. The `dictionary`
preset is currently available only through `demo2dict`. The 1M preset is the
practical intermediate option for longer answers. The 20M model is very
slow in this scalar CPU implementation; a few steps are useful as a smoke test,
not meaningful language training.

Parameter count is:

```text
vocabulary*width
+ layers*(4*width*width + 3*width*ffnWidth + 2*width)
+ width
```

The count follows from the architecture; it is not a count of words or facts.
The table uses the 260-token byte vocabulary. A BPE checkpoint adds
`(bpeVocabulary - 260) * width` embedding parameters.
The 20M parameters require about 310 MiB for weights, gradients, and Adam's two
moment buffers before activations and JVM overhead. `sbt run` uses a 2 GiB heap.

## Useful command examples

```sh
# Verified greeting GUI; load saved weights or train defaults if absent
sbt 'run demo2greet'

# Explicitly retrain greetings, evaluate, then open Swing
sbt 'run demo2greet 2000'

# Train the 200K experiment and open Swing
sbt 'run demo2daily200k 10000'

# Reopen that saved checkpoint without retraining
sbt 'run demo2daily200k'

# Train without a window, then evaluate and chat
sbt 'run train 200k data/dailydialog/starter.tsv 10000 checkpoints/dailydialog-200k.vanus 4'
sbt 'run gui checkpoints/dailydialog-200k.vanus data/dailydialog/starter.tsv'
sbt 'run eval checkpoints/dailydialog-200k.vanus data/dailydialog/starter.tsv'
sbt 'run eval checkpoints/dailydialog-200k.vanus data/dailydialog/validation.tsv'
sbt 'run chat checkpoints/dailydialog-200k.vanus Good morning'

# Focused greeting evaluation
sbt 'run greetings 2000'
sbt 'run eval checkpoints/conversation.vanus data/conversation-unseen.tsv'
sbt 'run chat checkpoints/conversation.vanus Hellow'

# Learn BPE from continuous book text, then adapt the checkpoint to dialogue
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 10000 checkpoints/book-bpe.vanus 4 512'
sbt 'run continue-pretrain checkpoints/book-bpe.vanus data/pride-prejudice-raw-public.txt 5000 checkpoints/book-bpe-more.vanus 4'
sbt 'run continue checkpoints/book-bpe.vanus data/dailydialog/starter.tsv 5000 checkpoints/book-chat.vanus 4'
sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'

# Later, reopen that GUI without doing any training
sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'

# Other Swing demonstrations
sbt 'run demo2 100000'
sbt 'run demo2dict 10000'

# Inspect one checkpoint's real weight tensors
sbt 'run weights checkpoints/dailydialog-200k.vanus'
```

Full DailyDialog evaluation generates thousands of replies and can take time.
Use validation while choosing settings and reserve `data/dailydialog/test.tsv`
for a final assessment.

## Training data

A TSV file contains one UTF-8 `prompt<TAB>answer` pair per line. The TSV is the
file format; the collection is a training dataset or corpus; one row is a
training example. The raw dictionary is source data, while its converted
prompt/answer rows are supervised training data.

For byte-tokenized 128-position models, keep the combined prompt and answer at
no more than 125 UTF-8 bytes. Three special tokens are also used. BPE can fit
the same text into fewer positions. Overlong examples fail in
generic training; fixed demos filter their source pairs to their context limit.

Data preparation commands:

```sh
python3 data/dictionary_to_tsv.py data/dictionary-raw2.tsv data/dictionary-tiny.tsv 125
python3 data/book_to_tsv.py data/pride-prejudice-raw-public.txt data/book-custom.tsv 125
python3 data/prepare_dailydialog.py
```

`dictionary_to_tsv.py` accepts raw `word<TAB>definition<TAB>example` entries and
creates question/answer phrasings. `book_to_tsv.py` creates consecutive-sentence
pairs. `prepare_dailydialog.py` rebuilds the derived splits from the downloaded
ZIP archives using only the Python standard library.

DailyDialog is stored under `data/dailydialog` with 45,530 short training pairs,
3,411 validation pairs, 3,095 test pairs, and a 300-pair starter set containing
256 seeded DailyDialog pairs plus 44 local greeting examples. The directory is
about 6.2 MB including source ZIPs and derived files. See the
[dataset README](data/dailydialog/README.md) for provenance, hashes, preparation,
limitations, and its CC BY-NC-SA 4.0 license.

## What training and checkpoints mean

A fresh supervised model has the transformer architecture and byte tokenizer, but no
learned language knowledge. Most weights begin as small random values;
normalization gains begin at one. Training adjusts those values. After training,
the checkpoint is sufficient for inference, so `chat` does not need to search
the original TSV for an answer.

More steps provide more sampled examples, but do not guarantee better replies.
For 300 examples at batch size one, 10,000 steps average about 33 sampled
exposures per example. Larger batches average gradients across examples. Model
capacity, data clarity and variety, context, and optimization all affect results.
Training recall measures memorization; held-out data measures limited
generalization. Natural dialogue also permits valid replies that exact matching
will count as failures.

Measured diagnostic results:

| Experiment | Result |
|---|---|
| Tiny greetings, 2,000 steps, batch 4 | 44/44 exact training replies and 5/5 small unseen variations |
| Tiny DailyDialog starter, 10,000 steps, batch 1 | 14/300 exact training replies (4.7%), 48.3% teacher-forced answer-byte accuracy, 0/5 unseen greeting variations |
| 200K preset | Compilation, finite-gradient test, and one-step training smoke test pass; full reply quality has not been measured |

These are small diagnostics, not general chatbot benchmarks. The focused
greeting checkpoint remains the verified choice for the three example greetings.

## Architecture

Vanus uses pre-RMSNorm, rotary position embeddings (RoPE), causal multi-head
self-attention, SwiGLU feed-forward layers, residual connections, and tied input
embedding/output weights. These are real transformer components, but Vanus uses
its own dimensions and checkpoint format and is not compatible with LLaMA.

```text
token IDs [T]
  -> embedding lookup E[token]                         [T,d]
  -> repeat L decoder blocks:
       a = RMSNorm(x)
       q = RoPE(a Wq), k = RoPE(a Wk), v = a Wv        [T,d]
       attention = causalSoftmax(q k^T / sqrt(d/H)) v  [T,d]
       r = x + attention Wo                             [T,d]
       f = RMSNorm(r)
       x = r + (SiLU(f Wgate) * (f Wup)) Wdown          [T,d]
  -> final RMSNorm                                     [T,d]
  -> logits = x E^T                                    [T,V]
```

Here `T` is sequence length, `V` is vocabulary size, `d` is model width,
`H` is the number of heads, and each head has width `d/H`. The feed-forward
intermediate has width `f`. Each block has four `d×d` attention matrices, three
feed-forward matrices (`d×f`, `d×f`, and `f×d`), and two learned RMSNorm gain
vectors. A final RMSNorm gain follows the blocks. There are no linear biases.
The embedding table `E[V,d]` is reused transposed for output logits, so there is
no separate language-model head.

RMSNorm operates independently on each sequence row, does not subtract a mean,
uses epsilon `1e-5`, and multiplies by a learned gain. RoPE uses base 10,000 and
rotates adjacent feature pairs in Q and K only. V is not rotated. Attention is
split into contiguous heads, scaled by `1/sqrt(headWidth)`, and causally masked:
position `t` can attend only to positions `0..t`. Softmax subtracts its maximum
for numerical stability. Training caches an `H×T×T` attention-probability array
for the explicit backward pass.

The base vocabulary consists of UTF-8 byte IDs 0–255 plus `BOS=256`, `EOS=257`,
`USER=258`, and `ASSISTANT=259`. The default tokenizer stops there. `pretrain`
learns deterministic byte-pair merges from the supplied text and assigns them
IDs from 260 upward. The merge table is stored in the checkpoint, so all later
training and inference use the same vocabulary. A prompt is encoded as
`BOS USER <text tokens> ASSISTANT`; supervised training appends answer tokens
and EOS. Input and target sequences are shifted by one position, and prompt
targets are masked with `-1`, so mean cross-entropy supervises answer tokens
through EOS. Context is measured in token positions. With byte tokenization,
multilingual text can fill it quickly. Invalid generated UTF-8 displays
replacement characters.

All trainable matrices start from Gaussian values with standard deviation 0.02;
RMSNorm gains start at one. Backpropagation uses a reverse topological tape of
tensor operations. Gradients accumulate in parameter buffers until `zeroGrad`.
AdamW uses betas 0.9 and 0.999, epsilon `1e-8`, bias correction, global-norm
clipping at one, and weight decay 0.01 on matrices; one-row RMSNorm gains are
excluded from decay.

Supervised training samples examples with replacement using deterministic seed
42. To improve tolerance for ordinary typing differences without adding CLI
arguments, 20% of sampled prompts receive one temporary small change: a missing
internal letter, swapped adjacent letters, removed comma, or removed ending
punctuation. The other 80% remain clean, and dataset files are never modified.
Training masks prompt targets, supervises answer tokens through EOS, and uses a base learning
rate of 0.003 multiplied by a ten-step warmup and a cosine schedule that reaches
a 10% floor at the final step. Gradients are averaged across the command-specific
batch before the optimizer step. Continuous-text pretraining instead samples
windows from one encoded raw-text stream and predicts every next token. Training
logs at the first step, approximately every four seconds, and at the final step.
Each report is a compact four-line block. It shows `elapsed=HH:MM:SS`, percentage,
segment and cumulative steps, recent average loss, batch size, estimated time,
the learning objective, tokenizer, vocabulary, context, AdamW learning rate,
pre-clipping gradient norm, and throughput. It also confirms that every
trainable tensor is updating and reports parameter/tensor counts, weight RMS,
maximum absolute weight, gradient RMS, architecture dimensions, and the trained
weight groups: embedding/tied output, attention Q/K/V/O, SwiGLU gate/up/down,
and RMSNorm gains. The time interval keeps long jobs visible without producing
a line for every update. The save message reports total elapsed time; hours may
exceed 24 for multi-day runs.

CLI and GUI generation use temperature zero and top-k one, which is greedy and
deterministic. The Java API also supports temperature and top-k sampling. During
generation all text-token IDs and EOS are eligible; BOS, USER, and ASSISTANT cannot be
sampled. Each generated token reruns the full forward pass over the growing
sequence. Generation stops at EOS, its token limit, or the context boundary.
There is no KV cache, sliding context window, GPU backend, or native matrix
library.

## Evaluation and visualization

`eval` runs free generation for exact-reply comparison and also reports
teacher-forced answer-token accuracy and mean cross-entropy loss. Token accuracy
uses the expected answer prefix; free generation uses its own prior predictions,
so a single error can compound. Perplexity is not printed, but for the reported
mean loss it is `exp(loss)`.

The Swing network is a schematic of embedding, decoder blocks, and output; its
connection lines are not learned attention weights. The live decoding table is
based on actual model logits. Candidate probabilities are normalized across all
eligible text tokens plus EOS before temperature/top-k filtering. Byte-tokenized
nonprintable values appear in hex.

### Checkpoint weight inspector

The `weights` command opens a separate Swing application for the actual float32
values stored in a `.vanus` checkpoint:

```sh
# Inspect one saved model
sbt 'run weights checkpoints/dailydialog-200k.vanus'

# Compare two checkpoints with the same architecture
sbt 'run weights checkpoints/dailydialog-200k-100000.vanus checkpoints/dailydialog-200k-100.vanus'
```

Choose a tensor such as `embedding`, `blocks.0.q`, or `blocks.1.down`. Orange
connections are positive and blue connections are negative; thickness and
opacity represent magnitude. A large matrix is sampled evenly at up to 24×24
connections so the display remains readable. The statistics table uses every
weight, not the sample. Norm gain vectors are shown as signed bars.

With two checkpoints, select **Difference: current − reference** to draw weight
changes. The table reports full-tensor delta RMS, maximum absolute delta, exact
changed-value count, and cosine similarity. Both checkpoints must have identical
model configurations and tensor shapes.

The normal demo checkpoint is replaced when you retrain it, so preserve distinct
files when comparing iteration counts. For example:

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 100 checkpoints/dailydialog-200k-100.vanus'
sbt 'run train 200k data/dailydialog/starter.tsv 100000 checkpoints/dailydialog-200k-100000.vanus'
sbt 'run weights checkpoints/dailydialog-200k-100000.vanus checkpoints/dailydialog-200k-100.vanus'
```

These runs use the same initialization seed and begin with the same sampled
example sequence, but both start from scratch. The cosine schedule also depends
on the requested total steps, so the comparison reflects both training duration
and a different learning-rate trajectory; it is not a resumed 100-to-100,000
step experiment. Use `continue` to retain the earlier checkpoint's optimizer
moments when extending a run.

## Build, tests, and code tour

```sh
sbt compile
sbt test
sbt clean
```

`sbt clean` removes build output, not checkpoints. Tests cover tokenizer UTF-8
round trips, autograd accumulation, numerical gradients, causal attention,
no-grad behavior, checkpoint validation and round trips, parameter counts,
AdamW behavior, generation diagnostics, nonfinite predictions, the 200K preset,
and self-talk prompt selection. They verify mechanics rather than language quality.

Suggested reading order:

1. `src/main/java/org/berlin/vanus/Tensor.java`
2. `src/main/java/org/berlin/vanus/ByteTokenizer.java` and `BpeTokenizer.java`
3. `src/main/java/org/berlin/vanus/Transformer.java`
4. `src/main/java/org/berlin/vanus/AdamW.java`
5. `src/main/java/org/berlin/vanus/TransformerVisualizerApp.java`
6. `src/main/scala/Main.scala`

The project demonstrates transformer mechanics and small, narrow learned tasks.
It is not a general-purpose assistant. Likely next improvements include faster
matrix kernels, streaming datasets, a production tokenizer, KV caching,
attention inspection, and more systematic held-out evaluation.
