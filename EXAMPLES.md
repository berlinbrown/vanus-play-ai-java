# Vanus command examples and workflows

This page enumerates every application command and shows how commands fit
together. Run everything below from the model directory:

```sh
cd core/vanus-ai-java
```

The general form is:

```sh
sbt 'run <command> <arguments>'
```

Square brackets in the documentation mean optional arguments. Do not type the
brackets. Paths under `checkpoints/` are examples and may be renamed.

## Runtime guide

On this machine, the following command takes about 30 minutes:

```sh
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 10000 checkpoints/book-bpe.vanus 4 512'
```

Use that measurement as the local baseline. Approximate times for the same
`200k`, batch 4, context 128, and 512-token BPE setup are:

| Training level | Steps | Approximate time | Intended use |
|---|---:|---:|---|
| Smoke | 1–10 | Seconds plus startup | Check that the command and checkpoint work |
| Quick | 100 | Roughly 20 seconds to 1 minute | Inspect learning mechanics |
| Short | 1,000 | Roughly 3 minutes | Early quality experiment |
| High | 10,000 | About 30 minutes | Substantial educational run |
| Very high | 100,000 | Roughly 5 hours | Long experiment; save distinct checkpoints |

These are estimates rather than guarantees. Runtime grows with steps, batch
size, sequence length, model size, and BPE vocabulary size. A batch of 4 does
about four forward/backward passes per optimizer step. Continuous pretraining
usually fills the whole context, while short TSV examples can run faster. The
`20m` model is more than 100 times larger than `200k` and may take many hours or
days in this scalar CPU implementation. BPE learning also adds startup time
before the first training step.

While training, Vanus prints a compact four-line progress block at startup,
approximately every four seconds, and at completion. It shows elapsed time,
completion percentage, current and cumulative steps, recent average loss,
gradient norm, batch size, learning rate, steps and samples per second, and
estimated time remaining. It identifies the objective and tokenizer and reports
weight RMS, maximum absolute weight, gradient RMS, parameter/tensor counts,
architecture dimensions, and the weight groups being trained. This is status
logging only; it does not change the training calculation.

For a quick end-to-end check, use:

```sh
sbt 'run pretrain tiny data/pride-prejudice-raw-public.txt 100 checkpoints/quick-bpe.vanus 1 320'
sbt 'run gui checkpoints/quick-bpe.vanus'
```

## Best and most interesting GUI runs

These are the recommended GUI experiments, ordered by what they demonstrate.

### 1. Best working conversation demo

```sh
sbt 'run demo2greet'
```

- **Data:** `data/conversation.tsv` with 44 focused greeting examples; the
  command also evaluates 5 held-out phrasings from
  `data/conversation-unseen.tsv`.
- **Model:** `tiny`, 18,656 parameters, byte tokenizer.
- **Training:** None when `checkpoints/conversation.vanus` already exists.
  Otherwise it automatically trains 2,000 steps with batch 4.
- **Why run it:** This is the most reliable model in the repository for
  `Hellow`, `How are you doing`, `Good morning`, and similar greetings.
- **GUI experiment:** Click **Talk to itself** to watch the output feed back into
  the next prompt every two seconds.

Use this first when you want the GUI to produce recognizable answers.

### 2. Most interesting modern training pipeline

If `checkpoints/book-bpe.vanus` is the checkpoint you already trained, adapt it
to the dialogue format and open it:

```sh
sbt 'run continue checkpoints/book-bpe.vanus data/dailydialog/starter.tsv 5000 checkpoints/book-chat.vanus 4'
sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'
```

- **Pretraining data already in the checkpoint:** Continuous public-domain
  *Pride and Prejudice* text from `data/pride-prejudice-raw-public.txt`.
- **Dialogue-tuning data:** 300 prompt/reply rows from
  `data/dailydialog/starter.tsv`.
- **Model:** `200k` architecture with the BPE vocabulary stored in
  `book-bpe.vanus`.
- **Training time:** A 5,000-step continuation may take roughly 15 minutes at
  the measured 10,000-steps-per-30-minutes rate. Short dialogue sequences may
  make it faster.
- **Why run it:** This demonstrates a simplified modern sequence: raw-text
  pretraining followed by supervised dialogue tuning, then inference.

This is the most educational GUI run, although it is not guaranteed to answer
as reliably as the narrowly trained greeting model.

### 3. Inspect the raw BPE-pretrained model

```sh
sbt 'run gui checkpoints/book-bpe.vanus'
```

- **Data used by this GUI invocation:** None. It only loads the checkpoint.
- **What the checkpoint learned from:** Continuous
  `data/pride-prejudice-raw-public.txt` text.
- **Expected behavior:** Prose-like continuation rather than question answering.
- **Prompts to try:** `It is a truth`, `Elizabeth`, `Mr. Darcy`, or a short
  sentence opening from the novel.
- **Why run it:** It shows the difference between next-token text pretraining
  and a chat model. Questions such as `Who are you?` may produce poor results
  because that response format was not the pretraining objective.

### 4. Larger dialogue model from random weights

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 10000 checkpoints/daily-chat-10k.vanus 4'
sbt 'run gui checkpoints/daily-chat-10k.vanus data/dailydialog/starter.tsv'
```

- **Data:** 300 rows from `data/dailydialog/starter.tsv`.
- **Model:** `200k`, 200,544 parameters with the default byte tokenizer.
- **Training time:** Plan for up to roughly 30 minutes using the observed local
  baseline, though its shorter TSV examples may finish sooner than continuous
  pretraining.
- **Why run it:** It has more capacity and dialogue variety than the focused
  greeting demo. It may also be less consistent because 300 varied examples are
  much harder than 44 focused examples.

For a quick preview before committing to the full run:

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 1000 checkpoints/daily-chat-1k.vanus 4'
sbt 'run gui checkpoints/daily-chat-1k.vanus data/dailydialog/starter.tsv'
```

### 5. Compare what training changed

Create or preserve checkpoints at different stages, then open the weight viewer:

```sh
sbt 'run weights checkpoints/chat-10k.vanus checkpoints/chat-1k.vanus'
```

- **Data during this command:** None; it reads two checkpoint files.
- **Why run it:** The separate Swing weight viewer shows actual tensor values
  and `10k - 1k` changes. This is more useful for understanding optimization
  than the schematic network lines in the chat GUI.
- **Requirement:** Both checkpoints must have identical architecture and
  vocabulary. A 260-token byte checkpoint cannot be compared with a 512-token
  BPE checkpoint.

### Recommended order

Run these in this order for the clearest demonstration:

```sh
# 1. Reliable narrow behavior
sbt 'run demo2greet'

# 2. Raw language pretraining behavior; no additional training
sbt 'run gui checkpoints/book-bpe.vanus'

# 3. Adapt those pretrained weights to dialogue, then inspect the result
sbt 'run continue checkpoints/book-bpe.vanus data/dailydialog/starter.tsv 5000 checkpoints/book-chat.vanus 4'
sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'

# 4. Compare with a dialogue model trained from random weights
sbt 'run train 200k data/dailydialog/starter.tsv 10000 checkpoints/daily-chat-10k.vanus 4'
sbt 'run gui checkpoints/daily-chat-10k.vanus data/dailydialog/starter.tsv'
```

The most useful comparison is between steps 3 and 4: both end on the same
dialogue dataset, but one begins with continuous-text BPE pretraining and the
other begins from random weights. They use different tokenizers, so compare
their generated replies and evaluation scores rather than using the weight
difference viewer between them.

## Training data and file paths

All command examples assume the current directory is `core/vanus-ai-java`, so a
command path such as `data/dailydialog/starter.tsv` refers to the repository file
`core/vanus-ai-java/data/dailydialog/starter.tsv`.

### Data formats

Supervised training uses UTF-8 TSV files with exactly two nonempty columns:

```text
prompt<TAB>answer
```

Each line is one independent training example. The model does not query the TSV
while answering; training converts its examples into learned weights. The
`pretrain` commands instead accept an ordinary UTF-8 text file and learn to
predict each next token in continuous passages.

### Included datasets

| File path from `core/vanus-ai-java` | Contents and origin | Typical use |
|---|---|---|
| [`data/conversation.tsv`](core/vanus-ai-java/data/conversation.tsv) | 44 locally authored greeting and small-talk prompt/reply pairs, including spelling and punctuation variants such as `Hello`, `Hellow`, and `How are you?` | Focused greeting training used by `greetings` and `demo2greet` |
| [`data/conversation-unseen.tsv`](core/vanus-ai-java/data/conversation-unseen.tsv) | 5 locally authored greeting phrasings excluded from focused training | Small held-out generalization check used automatically by the greeting demos |
| [`data/dailydialog/starter.tsv`](core/vanus-ai-java/data/dailydialog/starter.tsv) | 300 pairs: all 44 local greeting pairs plus 256 deterministically sampled DailyDialog training pairs | Bounded dialogue experiments and the `demo2daily*` commands |
| [`data/dailydialog/train.tsv`](core/vanus-ai-java/data/dailydialog/train.tsv) | 45,530 retained adjacent-turn pairs from the DailyDialog training split | Larger supervised dialogue training |
| [`data/dailydialog/validation.tsv`](core/vanus-ai-java/data/dailydialog/validation.tsv) | 3,411 retained DailyDialog validation pairs whose exact prompt strings do not occur in training | Model selection and progress checks |
| [`data/dailydialog/test.tsv`](core/vanus-ai-java/data/dailydialog/test.tsv) | 3,095 retained DailyDialog test pairs whose exact prompt strings do not occur in training or validation | Final evaluation after settings are chosen |
| [`data/pride-prejudice-raw-public.txt`](core/vanus-ai-java/data/pride-prejudice-raw-public.txt) | 888,930-byte public-domain *Pride and Prejudice* OCR text. The stored source contains scan/OCR artifacts and page material | Raw continuous text for BPE `pretrain` and `continue-pretrain` |
| [`data/pride-and-prejudice.tsv`](core/vanus-ai-java/data/pride-and-prejudice.tsv) | 1,986 consecutive-sentence prompt/reply pairs derived from the cleaned novel text | Book-continuation `demo` and `demo2` training |
| [`data/dictionary-raw2.tsv`](core/vanus-ai-java/data/dictionary-raw2.tsv) | 9 local `word<TAB>definition<TAB>example` source entries | Input to `dictionary_to_tsv.py`; not directly accepted by model training because it has three columns |
| [`data/dictionary2.tsv`](core/vanus-ai-java/data/dictionary2.tsv) | 162 prompt/reply variations generated from `dictionary-raw2.tsv`, including definition questions, partial-word prompts, and example requests | `demo2dict` training |
| [`data/dictionary-raw.tsv`](core/vanus-ai-java/data/dictionary-raw.tsv) | 30 older local dictionary source entries in three-column form | Optional source for generating another dictionary TSV |
| [`data/dictionary.tsv`](core/vanus-ai-java/data/dictionary.tsv) | 60 older derived dictionary prompt/reply pairs | Optional supervised experiments |
| [`data/greetings.tsv`](core/vanus-ai-java/data/greetings.tsv) | 25 older local question/answer, fact, and phrase-completion examples | Optional small supervised experiments; not the main greeting demo dataset |
| [`data/source-training-example.txt`](core/vanus-ai-java/data/source-training-example.txt) | Explanatory Java-code training examples and notes | Documentation only; `Main.scala` never loads it |
| [`data/cat-language/train.tsv`](core/vanus-ai-java/data/cat-language/train.tsv) | 1,762 generated rows teaching consistent English-to-cat, cat-to-English, and cat-noise conversations | Train the bilingual invented cat language |
| [`data/cat-language/test.tsv`](core/vanus-ai-java/data/cat-language/test.tsv) | 265 held-out cat-language combinations with no exact prompt overlap with training | Test whether the model learned reusable cat grammar |

DailyDialog is a manually labelled multi-turn dialogue dataset introduced by
Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu at IJCNLP
2017. This repository downloaded its split archives from the Hugging Face
`roskoN/dailydialog` mirror. The preparation script converts adjacent dialogue
turns into prompt/reply rows, normalizes spacing, removes duplicate pairs, and
filters examples to fit the 128-position byte-token model. DailyDialog is
licensed CC BY-NC-SA 4.0, so its derived files are not covered by the software
license.

Detailed DailyDialog provenance, hashes, filtering rules, and limitations are
in `core/vanus-ai-java/data/dailydialog/README.md`. Machine-readable counts and
source archive hashes are in
`core/vanus-ai-java/data/dailydialog/manifest.json`. The downloaded archives are
retained under `core/vanus-ai-java/data/dailydialog/source/`.

### Data preparation files

Rebuild the derived DailyDialog files:

```sh
python3 data/prepare_dailydialog.py
```

Create sentence-continuation pairs from another raw book:

```sh
python3 data/book_to_tsv.py data/my-book.txt data/my-book.tsv 125
```

Create prompt/reply variants from three-column dictionary source data:

```sh
python3 data/dictionary_to_tsv.py data/dictionary-raw2.tsv data/my-dictionary.tsv 125
```

The final number is the maximum combined prompt-and-answer size in UTF-8 bytes.
For a 128-position byte-tokenized model, 125 leaves room for the three chat
control tokens. BPE may encode the same text into fewer positions, but generic
supervised training still rejects any encoded example that exceeds the model's
context.

### Choosing the appropriate data

- Use `data/conversation.tsv` to reproduce the narrow, reliable greeting demo.
- Use `data/dailydialog/starter.tsv` for faster dialogue experiments.
- Use `data/dailydialog/train.tsv` when you want much more dialogue variety and
  accept longer training runs.
- Use `validation.tsv` while comparing settings and `test.tsv` only for the
  final evaluation.
- Use `pride-prejudice-raw-public.txt` for continuous-text BPE pretraining.
- Use `pride-and-prejudice.tsv` when teaching direct sentence-to-sentence
  continuation with the supervised `train` command.
- Use `dictionary2.tsv` for a narrow word-definition demonstration.
- Use `data/cat-language/train.tsv` to teach English and cat speech in both
  directions, then evaluate with `data/cat-language/test.tsv`.

### Cat-language workflow

```sh
# Train English → cat, cat → English, and cat-noise conversations
sbt 'run train 1m data/cat-language/train.tsv 14000 checkpoints/cat-language-1m.vanus 4'

# Open the bilingual cat model
sbt 'run gui checkpoints/cat-language-1m.vanus data/cat-language/train.tsv'

# Evaluate combinations excluded from training
sbt 'run eval checkpoints/cat-language-1m.vanus data/cat-language/test.tsv'
```

Try both English and cat-noise inputs:

```text
Hello
Meooow
Hi Meooow
Meoow Moewwww
I am happy.
Meow purr.
I want food.
Meow yowl nom.
Translate to cat: You are curious.
What does "Mewmau trill." mean?
```

The dataset and generator are documented in
[`data/cat-language/README.md`](core/vanus-ai-java/data/cat-language/README.md).

## Which data each command uses

| Command | Training or evaluation data | Checkpoint behavior |
|---|---|---|
| `sbt run` | No dataset. It only prints the same help as `info`. | Does not read or write a checkpoint |
| `sbt 'run info'` | No dataset. It prints command and model configuration information. | Does not read or write a checkpoint |
| `sbt 'run greetings [steps]'` | Trains/evaluates on `data/conversation.tsv`, then evaluates `data/conversation-unseen.tsv` | Loads `checkpoints/conversation.vanus` when no steps are supplied and it exists; otherwise writes it |
| `sbt 'run demo2greet [steps]'` | Same two greeting files as `greetings`; `conversation.tsv` also fills the Swing dropdown | Loads or writes `checkpoints/conversation.vanus` |
| `sbt 'run demo2daily [steps]'` | Trains on `data/dailydialog/starter.tsv`; the same file fills the Swing dropdown | Loads or writes `checkpoints/dailydialog-starter.vanus` |
| `sbt 'run demo2daily200k [steps]'` | Trains on `data/dailydialog/starter.tsv`; the same file fills the Swing dropdown | Loads or writes `checkpoints/dailydialog-200k.vanus` |
| `sbt 'run demo [steps] [checkpoint]'` | Trains on context-fitting rows from `data/pride-and-prejudice.tsv` | Always trains and writes the supplied checkpoint or `checkpoints/greeting.vanus` |
| `sbt 'run demo2 [steps]'` | Trains on context-fitting rows from `data/pride-and-prejudice.tsv`; the same rows fill the Swing dropdown | Loads or writes `checkpoints/demo2.vanus` |
| `sbt 'run demo2dict [steps]'` | Trains on context-fitting rows from `data/dictionary2.tsv`; the same rows fill the Swing dropdown | Loads or writes `checkpoints/demo2-dictionary.vanus` |
| `sbt 'run train … <pairs.tsv> …'` | Trains on the two-column TSV path supplied in the command | Starts from random weights and writes the supplied output checkpoint |
| `sbt 'run continue <checkpoint> <pairs.tsv> …'` | Trains on the supplied two-column TSV | Reads model and optimizer state from the input checkpoint and writes the requested output checkpoint |
| `sbt 'run pretrain … <text> …'` | Learns BPE from and trains on the supplied ordinary UTF-8 text file | Starts from random weights and writes the supplied output checkpoint |
| `sbt 'run continue-pretrain <checkpoint> <text> …'` | Continues next-token training on the supplied text using the checkpoint's saved tokenizer | Reads model and optimizer state and writes the requested output checkpoint |
| `sbt 'run gui <checkpoint> [pairs.tsv]'` | Does not train. The optional TSV only provides example prompts and a displayed dataset count. | Reads the supplied checkpoint without changing it |
| `sbt 'run chat <checkpoint> <prompt…>'` | No dataset. It generates from the literal prompt typed in the command. | Reads the supplied checkpoint without changing it |
| `sbt 'run eval <checkpoint> <pairs.tsv>'` | Evaluates against every row of the supplied TSV | Reads the supplied checkpoint without changing it |
| `sbt 'run weights <checkpoint>'` | No dataset. It reads tensors stored in the checkpoint. | Reads one checkpoint without changing it |
| `sbt 'run weights <current> <reference>'` | No dataset. It compares tensors from two checkpoints. | Reads both checkpoints without changing them |

The important distinction is that a dataset passed to `gui` is not training
data for that invocation. It only supplies convenient prompts. Likewise,
`eval` measures a model but does not teach it from the evaluation rows.

## Commands that never train

These commands only inspect or use existing state.

### Show help and model configurations

Both forms are equivalent:

```sh
sbt run          # no data; print help only
sbt 'run info'   # no data; print help only
```

### Open a checkpoint in Swing

Without a TSV prompt list:

```sh
sbt 'run gui checkpoints/chat.vanus' # checkpoint inference; no dataset
```

With prompts loaded into the GUI dropdown:

```sh
sbt 'run gui checkpoints/chat.vanus data/dailydialog/starter.tsv' # TSV supplies dropdown prompts only
```

`gui` never trains and never changes the checkpoint. It fails if the checkpoint
does not exist. The optional TSV only supplies dataset information and dropdown
prompts; generation still comes from the model weights.

### Generate one terminal response

```sh
sbt 'run chat checkpoints/chat.vanus Who are you?' # literal prompt; no dataset
sbt 'run chat checkpoints/chat.vanus Good morning' # literal prompt; no dataset
```

Every argument after the checkpoint is joined into one prompt. `chat` performs
inference only.

### Evaluate a checkpoint

```sh
sbt 'run eval checkpoints/chat.vanus data/dailydialog/starter.tsv'    # evaluate starter rows
sbt 'run eval checkpoints/chat.vanus data/dailydialog/validation.tsv' # evaluate validation rows
sbt 'run eval checkpoints/chat.vanus data/dailydialog/test.tsv'       # final test rows
```

Evaluation prints generated replies, exact-match results, teacher-forced token
accuracy, and loss. It does not update weights. Validation is useful while
choosing settings; reserve the test split for a final comparison.

### Inspect checkpoint weights

Inspect one checkpoint:

```sh
sbt 'run weights checkpoints/chat.vanus'
```

Compare two checkpoints with identical model configurations:

```sh
sbt 'run weights checkpoints/chat-more.vanus checkpoints/chat.vanus'
```

The comparison displays `current - reference`. It cannot compare checkpoints
whose vocabulary or architecture differs.

## Demo commands

Demo commands have fixed data, model sizes, batches, and checkpoint names.
When a demo is run without a step count, it loads its checkpoint if present. If
the checkpoint is missing, it trains the default number of steps. Supplying a
step count always starts that demo again from random weights.

### Focused greeting terminal demo

Load if available, otherwise train 2,000 steps with batch 4, then evaluate:

```sh
sbt 'run greetings'
```

Force fresh training for a chosen number of steps:

```sh
sbt 'run greetings 100'
sbt 'run greetings 2000'
sbt 'run greetings 10000'
```

### Focused greeting Swing demo

Load if available, otherwise train 2,000 steps with batch 4, evaluate, and open
Swing:

```sh
sbt 'run demo2greet'
```

Force fresh quick or high training, then open Swing:

```sh
sbt 'run demo2greet 100'
sbt 'run demo2greet 10000'
```

This is the most reliable narrow conversational demonstration.

### Tiny DailyDialog Swing demo

Load if available, otherwise train 10,000 steps with batch 1:

```sh
sbt 'run demo2daily'
```

Force fresh training:

```sh
sbt 'run demo2daily 100'
sbt 'run demo2daily 10000'
```

### 200K DailyDialog Swing demo

Load if available, otherwise train 10,000 steps with batch 1:

```sh
sbt 'run demo2daily200k'
```

Force fresh quick or high training:

```sh
sbt 'run demo2daily200k 100'
sbt 'run demo2daily200k 10000'
```

Ten thousand steps can take a while. It is reasonable to leave this running
for tens of minutes on this machine.

### Tiny book-continuation terminal demo

This command is the exception among demos: it always trains. Its default is 200
steps and its default checkpoint is `checkpoints/greeting.vanus`:

```sh
sbt 'run demo'
sbt 'run demo 1000'
sbt 'run demo 10000 checkpoints/book-demo.vanus'
```

The second positional argument cannot be used without the step argument.

### Tiny book-continuation Swing demo

Load if available, otherwise train the default 200 steps:

```sh
sbt 'run demo2'
```

Force fresh training and then open Swing:

```sh
sbt 'run demo2 100'
sbt 'run demo2 10000'
```

### Dictionary Swing demo

Load if available, otherwise train the default 200 steps with batch 8:

```sh
sbt 'run demo2dict'
```

Force fresh training:

```sh
sbt 'run demo2dict 100'
sbt 'run demo2dict 10000'
```

Batch 8 makes each step substantially more expensive than batch 1.

## Generic supervised training

The complete syntax is:

```text
train <tiny|200k|1m|20m> <pairs.tsv> <steps> <checkpoint> [batch]
```

All model-size enumerations are:

```sh
sbt 'run train tiny data/dailydialog/starter.tsv 100 checkpoints/chat-tiny.vanus'
sbt 'run train 200k data/dailydialog/starter.tsv 100 checkpoints/chat-200k.vanus'
sbt 'run train 1m data/dailydialog/starter.tsv 100 checkpoints/chat-1m.vanus'
sbt 'run train 20m data/dailydialog/starter.tsv 100 checkpoints/chat-20m.vanus'
```

Omitting the final batch uses batch 1. Supplying it sets the number of examples
whose gradients are averaged per optimizer update:

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 100 checkpoints/chat-quick.vanus 1'
sbt 'run train 200k data/dailydialog/starter.tsv 1000 checkpoints/chat-short.vanus 4'
sbt 'run train 200k data/dailydialog/starter.tsv 10000 checkpoints/chat-high.vanus 4'
```

`train` always starts from random weights and writes a checkpoint containing
the weights, tokenizer, AdamW state, and completed step count. Supervised
training automatically applies a missing letter, swapped adjacent letters,
removed comma, or removed ending punctuation to 20% of sampled prompts while
retaining the same expected answer:

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 30000 checkpoints/robust-typos.vanus 4'
sbt 'run gui checkpoints/robust-typos.vanus data/dailydialog/starter.tsv'
```

The other 80% remain exact. Variants exist only while training; the original
DailyDialog TSV, validation data, and test data remain unchanged.

For the intermediate 1M model on the starter dataset:

```sh
# Benchmark 100 steps first to estimate runtime on your machine
sbt 'run train 1m data/dailydialog/starter.tsv 100 checkpoints/starter-1m-smoke.vanus 4'

# Approximately one-hour run at the measured local rate
sbt 'run train 1m data/dailydialog/starter.tsv 8000 checkpoints/starter-1m-8k.vanus 4'

# Open the completed checkpoint without training
sbt 'run gui checkpoints/starter-1m-8k.vanus data/dailydialog/starter.tsv'
```

The 1M preset has 1,067,040 parameters, four transformer layers, width 160,
eight attention heads, and a 256-token context. It is about 5.3 times larger
than `200k`. A local batch-4 smoke benchmark reached roughly 2.2 steps per
second after startup, making 8,000 steps about one hour and 30,000 steps roughly
four hours. Actual speed depends on sampled sequence lengths and the machine.

## Continue supervised training

The two possible forms are:

```sh
# Default batch 1
sbt 'run continue checkpoints/chat.vanus data/dailydialog/starter.tsv 5000 checkpoints/chat-more.vanus'

# Explicit batch 4; light prompt augmentation is automatic
sbt 'run continue checkpoints/chat.vanus data/dailydialog/starter.tsv 5000 checkpoints/chat-more.vanus 4'
```

`continue` restores the checkpoint's model, tokenizer, optimizer moments, and
step count, then trains on the supplied TSV. A distinct output filename keeps
the original. This form deliberately replaces it:

```sh
sbt 'run continue checkpoints/chat.vanus data/dailydialog/starter.tsv 5000 checkpoints/chat.vanus 4'
```

Version 1 checkpoints can be loaded for inference but cannot continue because
they do not contain optimizer state.

## Continuous-text BPE pretraining

The complete syntax is:

```text
pretrain <tiny|200k|1m|20m> <text> <steps> <checkpoint> [batch] [bpe-vocab]
```

Every optional-argument form is:

```sh
# Defaults: batch 1 and requested BPE vocabulary 512
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 1000 checkpoints/book-default.vanus'

# Explicit batch; BPE vocabulary remains 512
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 1000 checkpoints/book-batch4.vanus 4'

# Explicit batch and requested BPE vocabulary
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 1000 checkpoints/book-bpe.vanus 4 512'
```

All size choices are valid:

```sh
sbt 'run pretrain tiny data/pride-prejudice-raw-public.txt 100 checkpoints/book-tiny.vanus 1 320'
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 10000 checkpoints/book-200k.vanus 4 512'
sbt 'run pretrain 1m data/pride-prejudice-raw-public.txt 100 checkpoints/book-1m.vanus 1 512'
sbt 'run pretrain 20m data/pride-prejudice-raw-public.txt 100 checkpoints/book-20m.vanus 1 512'
```

The BPE vocabulary is learned before model training and stored inside the
checkpoint. The requested vocabulary must be between 260 and 65,536. It can end
up smaller if the text does not contain enough repeated pairs. Larger
vocabularies increase BPE setup work, embedding parameters, checkpoint size,
and next-token output work.

## Continue continuous-text pretraining

The two forms are:

```sh
# Default batch 1
sbt 'run continue-pretrain checkpoints/book-bpe.vanus data/pride-prejudice-raw-public.txt 5000 checkpoints/book-more.vanus'

# Explicit batch 4
sbt 'run continue-pretrain checkpoints/book-bpe.vanus data/pride-prejudice-raw-public.txt 5000 checkpoints/book-more.vanus 4'
```

This reuses the BPE tokenizer saved in the checkpoint. It does not relearn the
vocabulary. The supplied raw text may be the original corpus or another text.

## Complete command sequences

### Sequence A: quick supervised model and GUI

```sh
sbt 'run train tiny data/dailydialog/starter.tsv 100 checkpoints/quick-chat.vanus 1'
sbt 'run eval checkpoints/quick-chat.vanus data/dailydialog/validation.tsv'
sbt 'run gui checkpoints/quick-chat.vanus data/dailydialog/starter.tsv'
```

This verifies the pipeline quickly. One hundred steps are unlikely to produce a
good conversational model.

### Sequence B: higher-quality supervised experiment

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 10000 checkpoints/chat-10k.vanus 4'
sbt 'run eval checkpoints/chat-10k.vanus data/dailydialog/validation.tsv'
sbt 'run gui checkpoints/chat-10k.vanus data/dailydialog/starter.tsv'
```

Plan for roughly 30 minutes if it runs at the same speed as the measured 10,000
step pretraining command. TSV training may be faster because examples can be
shorter than the full context.

### Sequence C: train, inspect, continue, and compare

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 1000 checkpoints/chat-1k.vanus 4'
sbt 'run weights checkpoints/chat-1k.vanus'
sbt 'run continue checkpoints/chat-1k.vanus data/dailydialog/starter.tsv 9000 checkpoints/chat-10k.vanus 4'
sbt 'run eval checkpoints/chat-10k.vanus data/dailydialog/validation.tsv'
sbt 'run weights checkpoints/chat-10k.vanus checkpoints/chat-1k.vanus'
sbt 'run gui checkpoints/chat-10k.vanus data/dailydialog/starter.tsv'
```

The learning-rate schedule restarts for the continuation segment, and the data
sampling RNG restarts at seed 42. Optimizer moments and the completed step count
are preserved, but this is not bit-for-bit identical to one uninterrupted
10,000-step run.

### Sequence D: BPE pretrain, continue pretraining, then dialogue-tune

```sh
sbt 'run pretrain 200k data/pride-prejudice-raw-public.txt 10000 checkpoints/book-10k.vanus 4 512'
sbt 'run continue-pretrain checkpoints/book-10k.vanus data/pride-prejudice-raw-public.txt 10000 checkpoints/book-20k.vanus 4'
sbt 'run continue checkpoints/book-20k.vanus data/dailydialog/starter.tsv 5000 checkpoints/book-chat.vanus 4'
sbt 'run eval checkpoints/book-chat.vanus data/dailydialog/validation.tsv'
sbt 'run gui checkpoints/book-chat.vanus data/dailydialog/starter.tsv'
```

At the measured speed, each 10,000-step pretraining segment is about 30 minutes,
so the two pretraining segments alone may take about an hour. Dialogue tuning
adds more time.

### Sequence E: separate 100-step and 100,000-step runs

```sh
sbt 'run train 200k data/dailydialog/starter.tsv 100 checkpoints/chat-100.vanus 4'
sbt 'run train 200k data/dailydialog/starter.tsv 100000 checkpoints/chat-100000.vanus 4'
sbt 'run weights checkpoints/chat-100000.vanus checkpoints/chat-100.vanus'
```

These runs share the deterministic initialization and early sampled-example
sequence, but both start from scratch and use schedules based on their requested
step totals. At the 30-minute-per-10,000-step baseline, 100,000 steps is roughly
five hours and should be treated as a long run.

## Build commands

```sh
sbt compile
sbt test
sbt clean
```

`clean` removes compiled build output. It does not delete checkpoints.
