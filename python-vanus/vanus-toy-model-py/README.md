# Vanus Toy Language Model

Vanus is a small decoder-only language model implemented directly in PyTorch.
The project covers the complete language-model pipeline: collecting text,
cleaning and splitting documents, training a byte-level BPE tokenizer,
converting text to token IDs, training a Transformer from scratch, saving
checkpoints, and probing those checkpoints with fixed prompts.

The model code does not load a pretrained architecture or pretrained weights.
Its attention, rotary position encoding, normalization, feed-forward network,
loss calculation, and autoregressive generation are implemented in
`src/vanus_toy_model_py/model/`.

## What the project does

The pipeline has six main stages:

```text
Hugging Face datasets / personal text
                  |
                  v
          data/raw/*.txt
                  |
        clean + deduplicate + split
                  |
                  v
    train.txt and validation.txt
                  |
          train byte-level BPE
                  |
                  v
       tokenizer/tokenizer.json
                  |
       encode text as uint16 IDs
                  |
                  v
      train.bin and validation.bin
                  |
       train decoder-only Transformer
                  |
                  v
       checkpoints/step-NNNNNN
                  |
          run repeatable probes
                  |
                  v
       reports/probes/*.json
```

Each text document is separated by `<|endoftext|>`. During training, the model
receives a sequence of token IDs and learns to predict the token one position
ahead. For example:

```text
input:   [The, cat, is, happy]
target:  [cat, is, happy, .]
```

Cross-entropy loss measures how well the output logits predict those target
tokens. Backpropagation and AdamW then update the model weights.

## Model architecture

The default Vanus model is a compact, Llama-style causal Transformer with
17,308,032 trainable parameters.

| Setting | Default |
| --- | ---: |
| Vocabulary size | 8,192 |
| Context length | 256 tokens |
| Transformer blocks | 8 |
| Hidden size | 384 |
| Attention heads | 6 |
| Dimension per head | 64 |
| SwiGLU intermediate size | 1,024 |
| Dropout | 0.0 |
| RoPE base (`theta`) | 10,000 |
| Input/output embeddings | Tied |

The forward path is:

```text
token IDs [batch, sequence]
            |
            v
token embedding [batch, sequence, 384]
            |
            v
  +---------------------------------------+
  | Transformer block, repeated 8 times  |
  |                                       |
  | RMSNorm                               |
  |    -> Q, K, V projections             |
  |    -> RoPE on queries and keys        |
  |    -> causal multi-head attention     |
  |    -> output projection               |
  |    -> residual connection             |
  |                                       |
  | RMSNorm                               |
  |    -> SwiGLU feed-forward network     |
  |    -> residual connection             |
  +---------------------------------------+
            |
            v
        final RMSNorm
            |
            v
tied vocabulary projection
            |
            v
logits [batch, sequence, 8192]
```

Important implementation details:

- **Causal self-attention** prevents a token from seeing later tokens. It uses
  PyTorch's optimized scaled-dot-product attention with `is_causal=True`.
- **RoPE** rotates query and key vector pairs according to token position, so
  attention knows the order of the sequence without learned position vectors.
- **RMSNorm** normalizes vector magnitude before each attention and feed-forward
  sublayer.
- **SwiGLU** computes `SiLU(gate(x)) * up(x)` and projects the result back to the
  hidden size.
- **Pre-normalization and residual connections** help gradients flow through
  the eight Transformer blocks.
- **Weight tying** shares the token-embedding weights with the final vocabulary
  projection, reducing the parameter count.
- **Generation** is autoregressive and supports greedy decoding, temperature,
  top-k sampling, and an end-of-text stopping token. It currently recomputes
  the active context for every generated token; there is no KV cache yet.

## Setup

The project requires Python 3.12 or newer and uses
[uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync
```

Run the commands below from the repository root because data, tokenizer,
checkpoint, and report paths are relative to the current working directory.

## Prepare training data

### 1. Download the example datasets

```bash
uv run python -m vanus_toy_model_py.download_data
```

This streams WikiText-103 and TinyStories and creates:

```text
data/raw/wikitext.txt
data/raw/tinystories.txt
```

The script stops after approximately 100 million WikiText characters and
35 million TinyStories characters rather than downloading each complete
dataset into the output files.

### 2. Optionally import personal text

Place UTF-8 `.txt` files anywhere under `data/pre-raw/`, then run:

```bash
uv run python -m vanus_toy_model_py.import_pre_raw
```

The importer cleans those files and combines them into
`data/raw/blog1.txt`, with an end-of-text marker between documents.

### 3. Clean, deduplicate, and split

```bash
uv run python -m vanus_toy_model_py.prepare_text
```

This stage:

- normalizes whitespace and removes null characters;
- ignores documents shorter than 100 characters;
- deduplicates documents using a normalized SHA-256 fingerprint;
- shuffles documents with seed `1337`;
- reserves 1% of the documents for validation; and
- writes `data/processed/train.txt` and
  `data/processed/validation.txt`.

### 4. Train the tokenizer

```bash
uv run python -m vanus_toy_model_py.train_tokenizer
```

This trains an 8,192-token byte-level BPE tokenizer with NFC normalization.
Its special tokens are:

```text
<|pad|> <|unk|> <|bos|> <|eos|> <|endoftext|>
```

The resulting tokenizer is saved to `tokenizer/tokenizer.json`.

To perform a simple encode/decode check, run:

```bash
uv run python -m vanus_toy_model_py.test_tokenizer
```

### 5. Convert text to binary token IDs

```bash
uv run python -m vanus_toy_model_py.tokenize_data
```

The tokenizer encodes every document, appends the end-of-text token, and writes
flat NumPy-compatible arrays:

```text
data/processed/train.bin
data/processed/validation.bin
```

Token IDs use unsigned 16-bit integers because the 8,192-token vocabulary fits
within that data type. Training memory-maps these files instead of loading the
complete token corpus into RAM.

## Test the model implementation

Before a full training run, this smoke test checks tensor shapes, loss,
backpropagation, embedding gradients, and generation:

```bash
uv run python -m vanus_toy_model_py.test_model
```

## Train the model

Start training with the defaults from `training_config.py`:

```bash
uv run python -m vanus_toy_model_py.train
```

Useful command-line overrides include:

```bash
uv run python -m vanus_toy_model_py.train \
  --max-steps 20 \
  --micro-batch-size 8 \
  --evaluation-interval 5 \
  --checkpoint-interval 10
```

The default effective batch contains:

```text
8 micro-batch sequences
* 256 tokens per sequence
* 8 gradient-accumulation passes
= 16,384 token presentations per optimizer step
```

Training uses:

- AdamW with weight decay on matrix-shaped weights but not one-dimensional
  normalization parameters;
- linear learning-rate warmup followed by cosine decay;
- gradient accumulation to simulate a larger batch;
- global gradient-norm clipping at 1.0;
- periodic validation and perplexity calculation;
- TensorBoard logging under `reports/tensorboard/`; and
- SafeTensors model checkpoints under `checkpoints/`.

The training loop chooses Apple MPS first when available, then CUDA, then CPU.

### Resume training

Pass a complete checkpoint directory, not just the `checkpoints` parent:

```bash
uv run python -m vanus_toy_model_py.train \
  --max-steps 2500 \
  --resume-from checkpoints/step-000020
```

`--max-steps` is the final optimizer-step target. If a checkpoint completed
step 20, the resumed loop begins at step 21.

Each checkpoint contains:

```text
model.safetensors     model weights
optimizer.pt          AdamW state needed to resume training
model_config.json     architecture used by the checkpoint
training_config.json  training settings used by the checkpoint
training_state.json   step, tokens seen, and latest losses
tokenizer.json        tokenizer snapshot, when available
```

To inspect training curves, run:

```bash
uv run tensorboard --logdir reports/tensorboard
```

## Probe a checkpoint

`probe_model.py` runs the same prompts against a saved checkpoint. For every
prompt it records the highest-probability next tokens, a greedy continuation,
and a sampled continuation. This makes it easier to compare model behavior at
different training steps.

```bash
uv run python -m vanus_toy_model_py.probe_model \
  --checkpoint checkpoints/step-000020
```

By default, prompts come from `data/probe_prompts.json`. Sampling uses 40 new
tokens, temperature `0.8`, and top-k `50`. These can be changed with
`--prompts`, `--max-new-tokens`, `--temperature`, `--top-k`, and
`--top-predictions`.

Console output shows the top next-token predictions and both continuations. A
complete machine-readable report is saved as:

```text
reports/probes/step-000020.json
```

Early checkpoints generally produce weak or incoherent text. That is expected:
the model starts with random weights and must learn token patterns entirely
from this project's training corpus.

## Project layout

```text
data/
  pre-raw/                 optional personal source text
  raw/                     downloaded/imported document text
  processed/               cleaned text and binary token IDs
  probe_prompts.json       fixed qualitative evaluation prompts
tokenizer/
  tokenizer.json           trained byte-level BPE tokenizer
checkpoints/
  step-NNNNNN/             resumable model snapshots
reports/
  tensorboard/             scalar training logs
  probes/                  JSON checkpoint-probe reports
src/vanus_toy_model_py/
  model/config.py          model dimensions and validation
  model/model.py           Transformer and generation implementation
  download_data.py         streamed dataset acquisition
  import_pre_raw.py        personal-text importer
  prepare_text.py          cleaning, deduplication, and splitting
  train_tokenizer.py       BPE tokenizer training
  tokenize_data.py         text-to-binary token conversion
  training_config.py       optimizer and training defaults
  train.py                 training, evaluation, resume, and checkpoints
  probe_model.py           repeatable checkpoint inspection
```
