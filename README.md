# vanus-play-ai-java

An educational decoder-only transformer implemented with Java numerical kernels,
Java autograd, and a small Scala command-line trainer. It uses no PyTorch,
pretrained weights, native BLAS, or external tokenizer. It includes its own raw
byte and educational BPE tokenizers.

## Is this a real LLM?

Architecturally, yes: causal self-attention, RoPE, RMSNorm, SwiGLU, backprop through an autograd tape, AdamW, and autoregressive next-token prediction via cross-entropy are the same pieces production transformers use (see Architecture below). Nothing special-cases words or does string matching; every response comes out of matrix multiplications over learned weights.

## ELI5: how Vanus works

Think of Vanus as a tiny child learning to talk.

### Before training

Vanus knows nothing. Its **weights** are thousands of random number knobs, so
its answers are initially random noise.

### Training

We show it examples:

```text
Person: Hello
Answer: Hello! Nice to meet you.
```

Vanus tries to guess the answer. When it guesses incorrectly, the code turns
its number knobs slightly. It repeats this process thousands of times:

```text
Look → Guess → Check → Adjust
```

### The dataset

The TSV file is its picture book:

```text
Hello    Hello! Nice to meet you.
Goodbye  See you later.
```

More examples teach more situations, but a tiny brain can only remember so
much.

### The checkpoint

After training, the adjusted knobs are saved in a `.vanus` checkpoint file.
That file is the model's learned memory.

### Running the GUI

The GUI loads those saved knobs. When you enter `Hello`, Vanus changes the text
into small tokens and asks:

```text
What token should come next?
```

It adds the chosen token and guesses again:

```text
Hello
Hello!
Hello! Nice
Hello! Nice to
```

It stops when it chooses the special finished token.

### What the Java code does

The Java code builds the little brain, performs the guesses, calculates its
mistakes, adjusts the knobs, saves them, and loads them later. The Swing code
provides a window for talking to that brain.

The complete process is:

```text
TSV examples
     ↓
Training changes the number knobs
     ↓
Checkpoint saves the knobs
     ↓
GUI loads the checkpoint
     ↓
Your prompt enters the model
     ↓
The model guesses one token at a time
     ↓
You see its answer
```

The model does not search the TSV when you talk to it. Training has already
squeezed patterns from the TSV into the saved numbers.

## Recommended GUI demo

Use JDK 21 or newer. From the model directory:

```sh
cd core/vanus-ai-java
sbt 'run demo2greet'
```

This loads the focused greeting checkpoint and opens the Swing GUI. If the
checkpoint does not exist, it trains the greeting model for 2,000 steps first.
This configuration and dataset produced the best verified results in the project.
Try `Hellow`, `How are you doing`, or `Good morning`.

Click **Talk to itself** to send `hello` immediately and then feed selected words
from each reply back into the model every two seconds. This is an autonomous
inference loop; it does not update the model's weights.

To deliberately retrain the greeting model from random weights:

```sh
sbt 'run demo2greet 2000'
```

For the larger, experimental 200K-parameter model with DailyDialog data:

```sh
sbt 'run demo2daily200k 10000'
```

The greeting model is the most reliable current demo. The 200K model is a
capacity experiment whose conversational quality has not yet been measured.
See [Best and most interesting GUI runs](EXAMPLES.md#best-and-most-interesting-gui-runs)
for a ranked set of GUI experiments, including raw BPE text generation,
dialogue tuning, self-talk, and checkpoint-weight comparison.

## Command reference

Run these commands from `core/vanus-ai-java`. Square brackets mean optional
arguments; do not type the brackets.

| Command | What it does | Trains? |
|---|---|---|
| `info` | Lists model sizes, commands, and argument rules | No |
| `greetings [steps]` | Loads or trains the focused greeting model and evaluates it in the terminal | Sometimes |
| `demo2greet [steps]` | Greeting model evaluation followed by Swing | Sometimes |
| `demo2daily [steps]` | Tiny DailyDialog starter model with Swing | Sometimes |
| `demo2daily200k [steps]` | 200K DailyDialog starter model with Swing | Sometimes |
| `demo [steps] [checkpoint]` | Trains the tiny novel-continuation model in the terminal | Yes |
| `demo2 [steps]` | Loads or trains the tiny novel model and opens Swing | Sometimes |
| `demo2dict [steps]` | Loads or trains the dictionary model and opens Swing | Sometimes |
| `train <size> <pairs.tsv> <steps> <checkpoint> [batch]` | Starts supervised training with automatic light typo augmentation | Yes |
| `continue <checkpoint> <pairs.tsv> <steps> <output> [batch]` | Continues supervised training with automatic light typo augmentation | Yes |
| `pretrain <size> <text> <steps> <checkpoint> [batch] [bpe-vocab]` | Learns BPE and starts continuous-text pretraining | Yes |
| `continue-pretrain <checkpoint> <text> <steps> <output> [batch]` | Continues continuous-text pretraining with the saved tokenizer and optimizer | Yes |
| `gui <checkpoint> [pairs.tsv]` | Opens an existing checkpoint in Swing; the optional TSV fills the prompt dropdown | No |
| `server <checkpoint> <pairs.tsv> [server options]` | Runs persistent headless self-talk with SQLite and a local JSON API | No |
| `chat <checkpoint> <prompt...>` | Generates one reply in the terminal | No |
| `eval <checkpoint> <pairs.tsv>` | Reports generated replies, exact matches, token accuracy, and loss | No |
| `weights <checkpoint>` | Visualizes one checkpoint's real weights | No |
| `weights <current> <reference>` | Visualizes differences between compatible checkpoints | No |

`<size>` is `tiny`, `200k`, `1m`, or `20m`. Batch size defaults to 1. The BPE
vocabulary defaults to 512. The `20m` model is very slow with this scalar CPU
implementation.

## Scenarios and examples

### Use the best existing GUI demonstration

```sh
sbt 'run demo2greet'
```

If `checkpoints/conversation.vanus` exists, this loads it. Otherwise it trains
the default 2,000 steps first. Supplying a number always deliberately retrains:

```sh
sbt 'run demo2greet 5000'
```

### Train a dialogue model, then open it

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 10000 checkpoints/chat.vanus 4'
sbt 'run gui checkpoints/chat.vanus data/dailydialog/starter.tsv'
```

The first command starts from random weights. The second command only loads the
saved weights and opens Swing. Later, reopen it without training:

```sh
sbt 'run gui checkpoints/chat.vanus data/dailydialog/starter.tsv'
```

The TSV argument is optional and only supplies the GUI's example dropdown:

```sh
sbt 'run gui checkpoints/chat.vanus'
```

### Continue an existing dialogue checkpoint

```sh
sbt 'run continue checkpoints/chat.vanus data/dailydialog/starter.tsv 5000 checkpoints/chat-more.vanus 4'
sbt 'run gui checkpoints/chat-more.vanus data/dailydialog/starter.tsv'
```

This restores the model weights, tokenizer, AdamW moment buffers, and completed
step count. Using a new output filename preserves the earlier checkpoint for
comparison. You may use the same input and output path if you intend to replace
the checkpoint.

### Pretrain on raw text with BPE, then adapt to dialogue

```sh
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 10000 checkpoints/book-bpe.vanus 4 512'
sbt 'run continue checkpoints/book-bpe.vanus data/dailydialog/starter.tsv 5000 checkpoints/book-chat.vanus 4'
sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'
```

`pretrain` learns a 512-token byte-pair vocabulary from the book and predicts
successive tokens from continuous passages. `continue` then uses prompt/reply
examples to adapt those weights for the chat format.

### Continue raw-text pretraining

```sh
sbt 'run continue-pretrain checkpoints/book-bpe.vanus data/pride-prejudice-raw-public.txt 5000 checkpoints/book-bpe-more.vanus 4'
```

The saved BPE vocabulary is reused. It is not learned again.

### Evaluate and use a model without training

```sh
sbt 'run eval checkpoints/chat.vanus data/dailydialog/validation.tsv'
sbt 'run chat checkpoints/chat.vanus Who are you?'
sbt 'run gui checkpoints/chat.vanus data/dailydialog/starter.tsv'
```

All three commands are inference-only. They never update weights. `gui` and
`chat` fail if the requested checkpoint does not exist.

### Compare a short and long training run

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 100 checkpoints/chat-100.vanus 4'
sbt 'run train 200k data/dailydialog/starter.tsv 100000 checkpoints/chat-100000.vanus 4'
sbt 'run weights checkpoints/chat-100000.vanus checkpoints/chat-100.vanus'
```

These are separate runs from the same deterministic initialization. To extend
the 100-step model instead, use `continue`.

### Build and test

```sh
sbt compile
sbt test
sbt clean
sbt 'run info'
```

Training checkpoints save optimizer state. Older version 1 checkpoints still
load for inference, but they cannot resume because they do not contain AdamW
moments.

See the [complete model guide](core/vanus-ai-java/README.md) for every command,
model dimensions, datasets, measured results, architecture, and learning notes.
The [command examples and workflows](EXAMPLES.md) page enumerates every command
form, quick and long training settings, estimated runtimes, and linked sequences.
The [cat-language dataset](core/vanus-ai-java/data/cat-language/README.md) teaches
English-to-cat, cat-to-English, and direct cat-noise conversations.

### Common Scenarios and Commands

```text
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 10000 checkpoints/book-bpe.vanus 4 512'

sbt 'run continue checkpoints/book-bpe.vanus data/dailydialog/starter.tsv 5000 checkpoints/book-chat.vanus 4'

sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'
```
