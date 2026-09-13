#!/usr/bin/env python3
"""Convert a raw OCR'd public-domain novel into prompt<TAB>answer training pairs.

Pairs consecutive sentences within the same paragraph so the model learns to
continue prose, matching the format Main.scala's `train` command expects.

Usage: python3 book_to_tsv.py <input.txt> <output.tsv> [max_pair_bytes]
"""
import re
import sys

CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+\s*$")
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
RUNNING_HEADER_RE = re.compile(r"^[A-Z' ]{4,40}$")
END_MARKERS = ("FINIS", "THE END", "NOTES", "QUESTIONS")
# OCR inserts stray margin/page line-numbers as bare digit tokens; this prose never uses bare digits.
MARGIN_NUMBER_RE = re.compile(r"\b\d{1,4}\b")
ABBREVIATIONS = ["Mr.", "Mrs.", "Dr.", "St.", "Mme.", "Mlle."]


def extract_body(lines: list[str]) -> list[str]:
    start = next((i for i, l in enumerate(lines) if CHAPTER_RE.match(l.strip())), 0)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() in END_MARKERS:
            end = i
            break
    return lines[start:end]


def clean_paragraphs(body_lines: list[str]) -> list[str]:
    paragraphs, current = [], ""
    for raw in body_lines:
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if CHAPTER_RE.match(line) or PAGE_NUMBER_RE.match(line) or RUNNING_HEADER_RE.match(line):
            continue
        # A trailing hyphen means the word was split across a line wrap; rejoin without a space.
        if current.endswith("-") and re.match(r"^[a-z]", line):
            current = current[:-1] + line
        else:
            current = f"{current} {line}" if current else line
    if current:
        paragraphs.append(current)
    cleaned = []
    for p in paragraphs:
        p = MARGIN_NUMBER_RE.sub(" ", p)
        p = p.replace("''", '"').replace("\\", "")
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    return cleaned


def split_sentences(paragraph: str) -> list[str]:
    guarded = paragraph
    for abbreviation in ABBREVIATIONS:
        guarded = guarded.replace(abbreviation, abbreviation.replace(".", "\u0000"))
    pieces = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c\u2018\u2019])', guarded)
    return [piece.replace("\u0000", ".").strip() for piece in pieces if piece.strip()]


def is_usable(sentence: str) -> bool:
    letters = sum(c.isalpha() for c in sentence)
    return len(sentence) >= 3 and letters >= max(3, len(sentence) // 3)


def build_pairs(paragraphs: list[str], max_pair_bytes: int) -> list[tuple[str, str]]:
    pairs = []
    for paragraph in paragraphs:
        sentences = [s for s in split_sentences(paragraph) if is_usable(s)]
        for a, b in zip(sentences, sentences[1:]):
            if len((a + b).encode("utf-8")) <= max_pair_bytes:
                pairs.append((a, b))
    return pairs


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Usage: book_to_tsv.py <input.txt> <output.tsv> [max_pair_bytes]")
    input_path, output_path = sys.argv[1], sys.argv[2]
    max_pair_bytes = int(sys.argv[3]) if len(sys.argv) == 4 else 220

    with open(input_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    paragraphs = clean_paragraphs(extract_body(lines))
    pairs = build_pairs(paragraphs, max_pair_bytes)

    with open(output_path, "w", encoding="utf-8") as f:
        for prompt, answer in pairs:
            f.write(f"{prompt}\t{answer}\n")
    print(f"Wrote {len(pairs)} pairs from {len(paragraphs)} paragraphs to {output_path}")


if __name__ == "__main__":
    main()
