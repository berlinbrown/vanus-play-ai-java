#!/usr/bin/env python3
"""Generate consistent English <-> cat-language prompt/reply training pairs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
train: dict[str, str] = {}
test: dict[str, str] = {}


def add(rows: dict[str, str], prompt: str, answer: str) -> None:
    previous = rows.get(prompt)
    if previous is not None and previous != answer:
        raise ValueError(f"Conflicting answer for {prompt!r}: {previous!r} vs {answer!r}")
    rows[prompt] = answer


def bilingual(english: str, cat: str, holdout: bool = False) -> None:
    """Add direct and explicitly requested translations in both directions."""
    rows = test if holdout else train
    add(rows, english, cat)
    add(rows, f"Say in cat: {english}", cat)
    add(rows, f"Translate to cat: {english}", cat)
    add(rows, cat, english)
    add(rows, f'What does "{cat}" mean?', english)
    add(rows, f"Translate to English: {cat}", english)


# Core phrases are deliberately repetitive so a small model can learn them.
core = [
    ("Hello!", "Meooow!"),
    ("Hi!", "Mew-oow!"),
    ("Good morning!", "Meooow sun-purr!"),
    ("Good night!", "Meooow moon-nap!"),
    ("Goodbye!", "Mew-naro!"),
    ("Yes.", "Purr."),
    ("No.", "Hiss."),
    ("Please.", "Mrr-please."),
    ("Thank you.", "Purr-thanks."),
    ("You are welcome.", "Mew-welcome."),
    ("I love you.", "Meow heart-purr."),
    ("I am a cat.", "Meow mau."),
    ("You are a cat.", "Mew mau."),
    ("We are cats.", "Meowmew maumau."),
    ("My name is Vanus.", "Meow name Vanus."),
    ("What is your name?", "Mew name?"),
    ("Who are you?", "Mew who?"),
    ("I understand.", "Meow know-purr."),
    ("I do not understand.", "Meow no-know."),
    ("Come here.", "Mew come-mrrp."),
    ("Go away.", "Mew go-hiss."),
    ("Help me.", "Mew help-meow."),
    ("Look at me.", "Mew look-meow."),
    ("Listen to me.", "Mew hear-meow."),
    ("I am very excited!", "Meoow Moewwww!"),
]
for english, cat in core:
    bilingual(english, cat)

# Short conversational noise-to-noise and mixed-language exchanges requested by
# the project owner. These teach replies rather than literal translations.
conversations = [
    ("Hello", "Meooow!"),
    ("Hi", "Mew-oow!"),
    ("Meooow", "Hello!"),
    ("Mew-oow", "Hi!"),
    ("Hi Meooow", "Meooow! Purr purr."),
    ("Meoow Moewwww", "Hello! I am very excited."),
    ("Meooow Meooow", "Hello, cat friend!"),
    ("Purr purr", "I am happy."),
    ("Hiss hiss", "I am angry. Please give me space."),
    ("Mrrp?", "What do you need?"),
    ("Mew?", "Yes?"),
    ("Nom nom", "I want food."),
    ("Lap lap", "I want water."),
    ("Nyaaap", "I want to sleep."),
    ("Chirp chirp", "I see something interesting."),
    ("Trill meow", "Let us play!"),
]
for prompt, answer in conversations:
    add(train, prompt, answer)

subjects = [
    ("I", "Meow"),
    ("you", "Mew"),
    ("the cat", "Mau"),
    ("the kitten", "Mewmau"),
    ("we", "Meowmew"),
]
states = [
    ("happy", "purr"),
    ("sad", "mrrp"),
    ("hungry", "nomnom"),
    ("thirsty", "laplap"),
    ("sleepy", "nyaaap"),
    ("angry", "hiss"),
    ("curious", "chirp"),
    ("playful", "trill"),
    ("scared", "eekmew"),
    ("comfortable", "softpurr"),
]
objects = [
    ("food", "nom"),
    ("water", "lap"),
    ("milk", "milkmew"),
    ("fish", "fishnom"),
    ("a warm bed", "warmnap"),
    ("a toy", "playmew"),
    ("a box", "boxmrrp"),
    ("a sunny window", "sunwindow"),
    ("a blanket", "softblanket"),
    ("a friend", "friendpurr"),
]
actions = [
    ("see", "chirp-see"),
    ("hear", "mrrp-hear"),
    ("find", "mew-find"),
    ("follow", "paw-follow"),
    ("protect", "hiss-guard"),
    ("visit", "purr-visit"),
]

# Hold out every seventh semantic combination. All phrasings of that combination
# go to test, preventing exact prompt leakage from train to test.
semantic_index = 0
for english_subject, cat_subject in subjects:
    verb = "am" if english_subject == "I" else "are" if english_subject in ("you", "we") else "is"
    for english_state, cat_state in states:
        holdout = semantic_index % 7 == 0
        bilingual(f"{english_subject.capitalize()} {verb} {english_state}.",
                  f"{cat_subject} {cat_state}.", holdout)
        semantic_index += 1

for english_subject, cat_subject in subjects:
    for english_object, cat_object in objects:
        holdout = semantic_index % 7 == 0
        bilingual(f"{english_subject.capitalize()} want {english_object}.",
                  f"{cat_subject} yowl {cat_object}.", holdout)
        semantic_index += 1
        holdout = semantic_index % 7 == 0
        bilingual(f"{english_subject.capitalize()} like {english_object}.",
                  f"{cat_subject} purr-like {cat_object}.", holdout)
        semantic_index += 1

for english_subject, cat_subject in subjects:
    for english_action, cat_action in actions:
        for english_object, cat_object in objects[:5]:
            holdout = semantic_index % 7 == 0
            bilingual(f"{english_subject.capitalize()} {english_action} {english_object}.",
                      f"{cat_subject} {cat_action} {cat_object}.", holdout)
            semantic_index += 1

# Questions and natural replies teach actual conversation in both languages.
for english_state, cat_state in states:
    add(train, f"Are you {english_state}?", f"Mew {cat_state}?")
    add(train, f"Mew {cat_state}?", f"Are you {english_state}?")
    add(train, f"How does a happy cat say {english_state}?", f"Meow {cat_state}.")

for english_object, cat_object in objects:
    add(train, f"Do you want {english_object}?", f"Mew yowl {cat_object}?")
    add(train, f"Mew yowl {cat_object}?", f"Do you want {english_object}?")
    add(train, f"The cat wants {english_object}.", f"Mau yowl {cat_object}.")

# A few held-out conversational combinations for meaningful evaluation.
held_out = [
    ("Hello, cat friend!", "Meooow friendpurr!"),
    ("I want fish.", "Meow yowl fishnom."),
    ("The kitten is playful.", "Mewmau trill."),
    ("We are sleepy.", "Meowmew nyaaap."),
    ("Mau purr-like warmnap.", "The cat likes a warm bed."),
    ("Meow chirp-see fishnom.", "I see fish."),
    ("Mew hiss?", "Are you angry?"),
    ("Translate to cat: You are curious.", "Mew chirp."),
]
for prompt, answer in held_out:
    train.pop(prompt, None)
    add(test, prompt, answer)

# Ensure no prompt appears in both splits.
overlap = set(train) & set(test)
if overlap:
    raise ValueError(f"Train/test prompt overlap: {sorted(overlap)[:5]}")


def write(path: Path, rows: dict[str, str]) -> None:
    path.write_text("".join(f"{prompt}\t{answer}\n" for prompt, answer in rows.items()), encoding="utf-8")


write(ROOT / "train.tsv", train)
write(ROOT / "test.tsv", test)
print(f"Wrote {len(train)} training rows and {len(test)} held-out test rows")
