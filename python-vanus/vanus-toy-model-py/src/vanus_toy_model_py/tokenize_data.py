# Convert everything once into integer token IDs.
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
eos_id = tokenizer.token_to_id("<|endoftext|>")


def tokenize_file(input_path: str, output_path: str):
    text = Path(input_path).read_text(encoding="utf-8")

    documents = text.split("<|endoftext|>")
    all_tokens = []

    for number, document in enumerate(documents, start=1):
        document = document.strip()

        if not document:
            continue

        token_ids = tokenizer.encode(document).ids
        all_tokens.extend(token_ids)
        all_tokens.append(eos_id)

        if number % 10_000 == 0:
            print(f"Processed {number:,} documents")

    # 8,192-token vocabulary fits comfortably in unsigned 16-bit values.
    array = np.asarray(all_tokens, dtype=np.uint16)
    array.tofile(output_path)

    print(f"{output_path}: {len(array):,} tokens")


tokenize_file(
    "data/processed/train.txt",
    "data/processed/train.bin",
)

tokenize_file(
    "data/processed/validation.txt",
    "data/processed/validation.bin",
)