# DailyDialog for Vanus

Downloaded 2026-09-14 from https://huggingface.co/datasets/roskoN/dailydialog
(original ZIP splits, retained in `source/`). The original author's download
URL now returns a parked-domain page rather than a ZIP.

Authors: Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu.
DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset, IJCNLP 2017.
Paper: https://aclanthology.org/I17-1099/
Dataset card: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
Dataset license: CC BY-NC-SA 4.0 (noncommercial, attribution, share-alike):
https://creativecommons.org/licenses/by-nc-sa/4.0/
These derived data files retain that license; they are not covered by the
repository's software license.

## Preparation

From core/vanus-ai-java, run `python3 data/prepare_dailydialog.py`.
The converter reads text members directly from ZIPs; no third-party code or
Python packages are needed. It pairs adjacent turns, normalizes whitespace and
punctuation spacing, removes duplicate pairs, and keeps only examples with
prompt + answer <= 125 UTF-8 bytes (three role tokens bring the input to 128).
It discards overlong pairs, rather than truncating replies mid-sentence.

Original dialogue splits are preserved. Validation excludes training prompt
strings, and test excludes both training and validation prompt strings.
The local greeting prompts are also excluded from evaluation splits.
This is exact normalized-string separation, not semantic deduplication.
Different replies to the same prompt within one split are retained.

`starter.tsv` contains 256 seeded samples from the training split plus all
44 local greeting examples. `manifest.json` records counts and archive hashes.
Everyday dialogues can need earlier context; these isolated pairs lose it.
This corpus has varied responses and some awkward wording. Exact-match scores
can penalize valid alternative replies, so inspect generated replies as well.

## Use

Start with the bounded starter set:

```sh
sbt 'run train tiny data/dailydialog/starter.tsv 10000 checkpoints/dailydialog-starter.vanus'
sbt 'run chat checkpoints/dailydialog-starter.vanus Good morning'
sbt 'run eval checkpoints/dailydialog-starter.vanus data/dailydialog/starter.tsv'
```

For a larger experiment, substitute `data/dailydialog/train.tsv`. Evaluate on
`data/dailydialog/validation.tsv`; reserve `test.tsv` for a final assessment.
Full evaluation generates thousands of replies and takes longer. Training
steps sample individual examples, not whole epochs. No reply-quality claim
is made for this dataset with the 18,656-parameter model. The existing greeting
checkpoint is still the verified option for reliable greetings.

## Measured starter baseline (2026-09-14)

With `tiny`, seed 42, batch size 1, and 10,000 steps, exact training
reply recall was **14/300 (4.7%)**, teacher-forced byte accuracy **48.3%**,
and loss **1.7491**. The five local unseen greeting variants scored **0/5**
exact replies. This baseline is not a useful general conversation model.
The focused greeting checkpoint remains the better greeting demonstration.

Open the saved experiment in Swing with `sbt 'run demo2daily'`.
Use `sbt 'run demo2daily 100000'` to restart training for 100,000 steps and
then open Swing. Higher-iteration quality has not yet been measured.
