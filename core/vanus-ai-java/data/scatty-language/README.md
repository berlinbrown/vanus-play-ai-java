# Scatty: a small controlled language

Scatty uses 146 distinct words across 1,590 training rows and 149 held-out test
rows. Each TSV row is `prompt<TAB>response`. Replies use short sentences, with explanations for fact checks and warmer greetings.
The word count ignores case and punctuation; it is not the tokenizer vocabulary
(Vanus still uses byte tokens for the normal `train` command).

## What it teaches

| Task | Input | Expected reply |
| --- | --- | --- |
| Greeting | `hello` | `hello friend. it is good to see you` |
| Identity | `who are you` | `i am scatty. i can talk about simple facts` |
| Form a sentence | `cat happy` | `cat is happy` |
| Check a fact | `cat is happy. is cat happy?` | `yes, cat is happy` |
| Check a negative | `cat is not happy. is cat happy?` | `no, cat is not happy` |
| Find a place | `sam is at home. where is sam?` | `you can find sam at home` |
| Find an owner | `alex has the key. who has the key?` | `alex has the key` |
| Read a color | `the ball is red. what color is the ball?` | `the ball is red` |
| Missing information | `where is cat?` | `i do not know where cat is` |
| Simple suggestion | `i am hungry` | `take a break and eat some food` |
| English to cat | `say in cat: hello` | `meow` |
| Cat to English | `translate to english: purr` | `yes, cat is happy` |

This is a toy world with explicit rules: each subject has one current state and
one location in a prompt. A different state or location answers `no` followed by the supplied fact; missing
information produces an explicit “i do not know” reply. Real life allows multiple states, but this
exercise deliberately uses a simpler rule. Negative facts only teach rejection
of the explicitly denied property. Suggestions are canned practice responses.

Include facts and the question in the same input. These rows do not teach memory
between chat turns, arbitrary reasoning, or executing actions. Cat translation
always has an explicit request so it cannot conflict with a normal greeting.
Start with the lowercase forms and punctuation shown above.

## Train, open the GUI, and evaluate

Run from `core/vanus-ai-java`:

```sh
# Quick experiment: 2,000 updates; quality may still be limited
sbt 'run train 1m data/scatty-language/scatty.tsv 2000 checkpoints/scatty-controlled-quick.vanus 4'

# Longer experiment: start a fresh model with this focused dataset
sbt 'run train 1m data/scatty-language/scatty.tsv 14000 checkpoints/scatty-controlled.vanus 4'

# Open the longer-run checkpoint without training
sbt 'run gui checkpoints/scatty-controlled.vanus data/scatty-language/scatty.tsv'

# Check combinations withheld from training
sbt 'run eval checkpoints/scatty-controlled.vanus data/scatty-language/test.tsv'
```

Choose one training command; both start from scratch. For the quick checkpoint,
use its filename in the GUI and eval commands. At your earlier 2.69 updates/sec,
14,000 updates would take about 87 minutes; these shorter examples may run faster.
Longer training does not guarantee generalization. Compare generated answers on
both training and test rows. Many yes/no questions have `no` answers, so inspect
`yes` cases, unknown cases, and text answers rather than trusting accuracy alone.

Test splits hold out subject/state, subject/location, subject/object and
object/color combinations, plus new wording for suggestions. These test rows
come from the same templates, so success is a narrow compositional test, not
evidence of unrestricted language understanding.

## Files and regeneration

- `scatty.tsv`: training pairs, replacing the mixed dialogue dataset.
- `test.tsv`: unseen test prompts; do not train on these when measuring progress.
- `vocabulary.txt`: the 146 words used across both splits.
- `scatty-original.tsv`: a copy of your dataset before this replacement.
- `generate_scatty.py`: deterministic dataset generator, not the AI model.

```sh
python3 data/scatty-language/generate_scatty.py
```

Regeneration overwrites `scatty.tsv`, `test.tsv`, and `vocabulary.txt`. Edit the
generator to keep customizations reproducible. It verifies unique prompts,
no train/test prompt overlap, at most 200 words, and examples fitting even a
128-token byte context. The original backup is not overwritten by the generator.

## Self-talk phrases

The dataset also teaches a playful conversation chain:

```text
hello
hello friend. it is good to see you
good it is
it is good
yes, so good
a good day to talk
let us talk friend
what is on your mind?
a quiet day at home
```

Each line can prompt the next reply. These are fictional conversational phrases,
not observations of a real home or garden. Existing prompts retain their original
answers. The generator adds full phrases, capitalized versions, punctuation-free
versions, and one-to-three-word fragments in their original order to support the
Swing self-talk feature. For overlapping fragments, the first defined answer wins.

Swing randomly selects words from replies, so the actual conversation can jump
between phrases or repeat. Longer fragments and unfamiliar combinations are not
all covered, and training does not guarantee every response. This adds training
examples, not live learning or new GUI behavior. The held-out tests still measure
the original fact tasks; the self-talk chains are training examples.
