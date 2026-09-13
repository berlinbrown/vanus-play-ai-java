#!/usr/bin/env python3
"""Convert a word/definition/example dictionary into prompt<TAB>answer training pairs.

Input is UTF-8 text with one entry per line: word<TAB>definition<TAB>example
sentence. Each entry expands into several worded variants of the same question
(the model only memorizes exact byte sequences, so more accepted phrasings has
to mean more literal training pairs, not smarter generalization), matching the
format Main.scala's `train` command expects.

Usage: python3 dictionary_to_tsv.py <input.tsv> <output.tsv> [max_pair_bytes]
"""
import sys


def parse_entries(lines: list[str]) -> list[tuple[str, str, str]]:
    entries = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not all(f.strip() for f in fields):
            raise ValueError(f"Line {lineno}: expected word<TAB>definition<TAB>example")
        word, definition, example = (f.strip() for f in fields)
        entries.append((word, definition, example))
    return entries


def build_pairs(entries: list[tuple[str, str, str]], max_pair_bytes: int) -> list[tuple[str, str]]:
    pairs = []
    for word, definition, example in entries:
        # Every one of these phrasings is a distinct byte sequence the model must see during
        # training to answer it; there is no shortcut that lets it infer the rest on its own.
        definition_prompts = [
            f"What does {word} mean?",
            f"What does {word} mean",
            f"What does {word}",
            f"what does {word} mean?",
            word,
        ]
        example_prompts = [
            f"Use {word} in a sentence.",
            f"Use {word} in a sentence",
            f"{word} in a sentence",
        ]
        candidates = [(p, definition) for p in definition_prompts] + [(p, example) for p in example_prompts]
        for prompt, answer in candidates:
            if len((prompt + answer).encode("utf-8")) <= max_pair_bytes:
                pairs.append((prompt, answer))
    return pairs


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Usage: dictionary_to_tsv.py <input.tsv> <output.tsv> [max_pair_bytes]")
    input_path, output_path = sys.argv[1], sys.argv[2]
    max_pair_bytes = int(sys.argv[3]) if len(sys.argv) == 4 else 220

    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()
    entries = parse_entries(lines)
    pairs = build_pairs(entries, max_pair_bytes)

    with open(output_path, "w", encoding="utf-8") as f:
        for prompt, answer in pairs:
            f.write(f"{prompt}\t{answer}\n")
    print(f"Wrote {len(pairs)} pairs from {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
