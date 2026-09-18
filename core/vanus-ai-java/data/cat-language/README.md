# Vanus cat language

This dataset teaches a small invented cat language in both directions:

- English prompt to cat-language response
- Cat-language prompt to English meaning
- English and cat-noise conversations
- Reusable subject, feeling, action, and object grammar

The vocabulary is deliberately consistent. For example:

| Cat word | Meaning |
|---|---|
| `Meow` | I |
| `Mew` | you |
| `Mau` | the cat |
| `purr` | happy |
| `hiss` | angry |
| `yowl` | want |
| `nom` | food |
| `lap` | water |
| `nyaaap` | sleepy |
| `trill` | playful |

Requested conversational examples are included:

```text
Hi Meooow	Meooow! Purr purr.
Meoow Moewwww	Hello! I am very excited.
```

Regenerate the TSV files:

```sh
python3 data/cat-language/generate_cat_language.py
```

Train the 1M model:

```sh
sbt 'run train 1m data/cat-language/train.tsv 14000 checkpoints/cat-language-1m.vanus 4'
```

Open Swing:

```sh
sbt 'run gui checkpoints/cat-language-1m.vanus data/cat-language/train.tsv'
```

Evaluate held-out combinations:

```sh
sbt 'run eval checkpoints/cat-language-1m.vanus data/cat-language/test.tsv'
```

Useful prompts:

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

`test.tsv` holds out complete semantic combinations and has no exact prompt
overlap with `train.tsv`. This makes it more useful for checking whether the
model learned reusable patterns rather than only memorizing rows.
