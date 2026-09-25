# Prepare raw text for training by cleaning, deduplicating, and splitting into training and validation sets.
import hashlib
import random
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def clean_document(text: str) -> str:
    """Remove unwanted characters and normalize spacing in one document."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fingerprint(text: str) -> str:
    """Create a stable content hash used to identify duplicate documents."""
    normalized = re.sub(r"\s+", " ", text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def save_documents(path: Path, values: list[str]) -> None:
    """Write documents to a UTF-8 file separated by end-of-text markers."""
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(value)
            output.write("\n<|endoftext|>\n")


def main() -> None:
    """Clean, deduplicate, split, and save all raw text documents."""
    random.seed(1337)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = []
    seen = set()

    for path in RAW_DIR.glob("*.txt"):
        contents = path.read_text(encoding="utf-8", errors="ignore")

        for document in contents.split("<|endoftext|>"):
            document = clean_document(document)

            if len(document) < 100:
                continue

            digest = fingerprint(document)

            if digest in seen:
                continue

            seen.add(digest)
            documents.append(document)

    random.shuffle(documents)

    validation_size = max(1, int(len(documents) * 0.01))
    validation_documents = documents[:validation_size]
    training_documents = documents[validation_size:]

    save_documents(OUTPUT_DIR / "train.txt", training_documents)
    save_documents(OUTPUT_DIR / "validation.txt", validation_documents)

    print(f"Training documents:   {len(training_documents):,}")
    print(f"Validation documents: {len(validation_documents):,}")


if __name__ == "__main__":
    main()
