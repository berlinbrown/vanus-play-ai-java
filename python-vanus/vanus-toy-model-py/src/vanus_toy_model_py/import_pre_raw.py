# Import pre-raw text, plain text
# And convert to raw format
import re
from pathlib import Path

SOURCE_DIR = Path("data/pre-raw")
OUTPUT_FILE = Path("data/raw/blog1.txt")

def clean_text(text: str) -> str:
    # Normalize different operating-system line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove spaces and tabs at the end of lines.
    text = "\n".join(line.rstrip() for line in text.splitlines())

    # Replace three or more blank lines with one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace surrounding the complete document.
    return text.strip()


def main():
    source_files = sorted(SOURCE_DIR.rglob("*.txt"))

    if not source_files:
        raise RuntimeError(f"No text files found under {SOURCE_DIR}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    documents_written = 0
    characters_written = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for source_file in source_files:
            text = source_file.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            text = clean_text(text)

            if len(text) < 50:
                print(f"Skipping short document: {source_file}")
                continue

            output.write(text)
            output.write("\n<|endoftext|>\n")

            documents_written += 1
            characters_written += len(text)

            print(f"Imported: {source_file}")

    print(f"Documents written: {documents_written:,}")
    print(f"Characters written: {characters_written:,}")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()