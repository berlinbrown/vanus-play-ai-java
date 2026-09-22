"""Build a small controlled-language corpus; no model training occurs here."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
train, test = {}, {}


def add(prompt, reply, held=False):
    assert prompt and reply and '\t' not in prompt + reply
    assert '\n' not in prompt + reply
    assert prompt not in train and prompt not in test, prompt
    (test if held else train)[prompt] = reply


for prompts, reply in [
    (["hello", "hi", "hey", "hello there", "hellow"], "hello friend. it is good to see you"),
    (["good morning", "morning"], "good morning friend. let us start a good day"),
    (["good evening"], "good evening friend. how was your day?"),
    (["good night"], "good night sleep well"),
    (["bye", "goodbye", "see you later"], "goodbye friend. come back and talk soon"),
    (["thanks", "thank you"], "you are welcome. i am happy to help"),
    (["how are you", "how are you doing", "how are you doin"], "i am fine thank you"),
    (["who are you", "what is your name"], "i am scatty. i can talk about simple facts"),
    (["help", "what can you do"], "ask about a fact or say in cat"),
    (["peace"], "peace to you"),
    (["yes"], "yes indeed"),
    (["no"], "no indeed"),
]:
    for prompt in prompts:
        for variant in (prompt, prompt.capitalize(), prompt + "!"):
            add(variant, reply)

# A fact is supplied in every question: the model need not remember prior turns.
subjects = ["cat", "dog", "bird", "fish", "alex", "sam", "teacher", "friend"]
states = ["happy", "sad", "hungry", "tired", "calm", "ready"]
places = ["home", "school", "park", "shop", "garden", "room"]
objects = ["key", "book", "ball", "cup", "bag", "box"]
for i, subject in enumerate(subjects):
    for j, state in enumerate(states):
        held = (i + j) % 7 == 0
        add(f"{subject} {state}", f"{subject} is {state}", held)
        for k, asked in enumerate(states):
            # Hold out entire fact/property combinations, not punctuation variants.
            add(f"{subject} is {state}. is {subject} {asked}?",
                f"yes, {subject} is {state}" if state == asked else f"no, {subject} is {state}", held)
        add(f"{subject} is not {state}. is {subject} {state}?", f"no, {subject} is not {state}", held)
        add(f"{subject} is {state}. how is {subject}?", f"{subject} is {state}", held)
    for j, place in enumerate(places):
        held = (i + j) % 7 == 0
        add(f"{subject} is at {place}. where is {subject}?", f"you can find {subject} at {place}", held)
        for asked in places:
            add(f"{subject} is at {place}. is {subject} at {asked}?",
                f"yes, {subject} is at {place}" if place == asked else f"no, {subject} is at {place}", held)
    for j, obj in enumerate(objects):
        held = (i + j) % 7 == 0
        add(f"{subject} has the {obj}. what does {subject} have?", f"{subject} has the {obj}", held)
        add(f"{subject} has the {obj}. who has the {obj}?", f"{subject} has the {obj}", held)
    for other in subjects:
        if subject != other:
            add(f"{subject} is happy. how is {other}?", f"i do not know. tell me about {other}", (i + subjects.index(other)) % 7 == 0)
    for state in states:
        add(f"is {subject} {state}?", f"i do not know. tell me about {subject}")
    add(f"where is {subject}?", f"i do not know where {subject} is")

# Counterbalanced yes/no examples prevent always answering "no" from succeeding.
for i, obj in enumerate(objects):
    for j, color in enumerate(["red", "blue", "green", "white"]):
        held = (i + j) % 7 == 0
        add(f"the {obj} is {color}. what color is the {obj}?", f"the {obj} is {color}", held)
        add(f"the {obj} is {color}. is the {obj} {color}?", f"yes, the {obj} is {color}", held)
        add(f"the {obj} is not {color}. is the {obj} {color}?", f"no, the {obj} is not {color}", held)

for state, action in [("hungry", "take a break and eat some food"),
                      ("thirsty", "take a break and drink some water"),
                      ("tired", "find a quiet place and rest"),
                      ("cold", "wear a warm coat and go inside"),
                      ("lost", "ask for help to find your way home"),
                      ("sad", "talk to a friend. you are not alone")]:
    add(f"i am {state}", action)
    add(f"i am {state}. what can i do?", action)
    add(f"i feel {state}", action, True)

for english, cat in [("hello", "meow"), ("goodbye", "mew"), ("yes", "purr"),
                     ("no", "hiss"), ("thank you", "purr purr"),
                     ("good morning", "meow sun"), ("good night", "meow moon")]:
    add(f"say in cat: {english}", cat)
    add(f"translate to cat: {english}", cat)
    add(f"translate to english: {cat}", english)
    add(f"what does {cat} mean?", english)

# Short reply chains plus ordered fragments for Swing's random-word self-talk.
# Existing examples always win; never add a second answer for an existing prompt.
from itertools import combinations

self_talk = [
    ("hello friend. it is good to see you", "good it is"),
    ("good it is", "it is good"),
    ("it is good", "yes, so good"),
    ("yes, so good", "a good day to talk"),
    ("a good day to talk", "let us talk friend"),
    ("let us talk friend", "what is on your mind?"),
    ("what is on your mind?", "a quiet day at home"),
    ("a quiet day at home", "home is a good place"),
    ("home is a good place", "a good place to rest"),
    ("a good place to rest", "rest and dream a little"),
    ("rest and dream a little", "i dream of a green garden"),
    ("i dream of a green garden", "a bird sings in the garden"),
    ("a bird sings in the garden", "a little song brings peace"),
    ("a little song brings peace", "peace to you my friend"),
    ("peace to you my friend", "thank you for the kind words"),
    ("thank you for the kind words", "kind words make a good day"),
    ("kind words make a good day", "good it is"),
    ("yes indeed", "it is good"),
    ("no indeed", "let us talk friend"),
    ("peace to you", "peace to you my friend"),
    ("i am fine thank you", "a good day to talk"),
    ("you are welcome. i am happy to help", "thank you for the kind words"),
]

# Full phrases have priority over overlapping fragments.
for prompt, reply in self_talk:
    for variant in dict.fromkeys((prompt, prompt.capitalize(),
                                  " ".join(re.findall(r"[a-z]+", prompt)))):
        if variant not in train and variant not in test:
            add(variant, reply)
for prompt, reply in self_talk:
    words = re.findall(r"[a-z]+", prompt)
    for size in range(1, min(3, len(words)) + 1):
        for indices in combinations(range(len(words)), size):
            fragment = " ".join(words[index] for index in indices)
            if fragment not in train and fragment not in test:
                add(fragment, reply)

vocabulary = sorted(set(re.findall(r"[a-z]+", " ".join(
    text.lower() for pairs in (train, test) for pair in pairs.items() for text in pair))))
assert len(vocabulary) <= 200
assert not train.keys() & test.keys()
for pairs in (train, test):
    assert max(len((p + a).encode()) + 4 for p, a in pairs.items()) <= 128
for name, pairs in [("scatty.tsv", train), ("test.tsv", test)]:
    (ROOT / name).write_text("".join(f"{p}\t{a}\n" for p, a in pairs.items()))
(ROOT / "vocabulary.txt").write_text("\n".join(vocabulary) + "\n")
print(f"Training: {len(train)} rows; test: {len(test)} rows; vocabulary: {len(vocabulary)} words")
