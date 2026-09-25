# Download datasets
# Download salesforce wikitext and TinyStories datasets
# Run with:
#  uv run python -m vanus_toy_model_py.download_data

from pathlib import Path
from datasets import load_dataset

# Target download path
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Stop after 100 million characters
TARGET_WIKITEXT_CHARS = 100_000_000

# Stop after 35 million characters
TARGET_STORY_CHARS = 35_000_000

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()

def write_limited(dataset, output_path: Path, max_chars: int):
    chars_written = 0
    documents_written = 0

    with output_path.open("w", encoding="utf-8") as output:
        for row in dataset:
            text = normalize(row["text"])

            if len(text) < 100:
                continue

            output.write(text)
            output.write("\n<|endoftext|>\n")

            chars_written += len(text)
            documents_written += 1

            if chars_written >= max_chars:
                break

    print(
        f"{output_path}: "
        f"{documents_written:,} documents, "
        f"{chars_written:,} characters"
    )

def main():
    wiki = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )

    stories = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    write_limited(
        wiki,
        RAW_DIR / "wikitext.txt",
        TARGET_WIKITEXT_CHARS,
    )

    write_limited(
        stories,
        RAW_DIR / "tinystories.txt",
        TARGET_STORY_CHARS,
    )

if '__main__' == __name__:
    print("==========")
    print("Downloading datasets...")
    print("==========")

    main()
    print("==========")
    print("End Run")
    print("==========")