from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")


# The unusual character represents a space before the token.
# It is a display convention used by byte-level BPE tokenizers.
tests = [
    "The cat is happy.",
    "SQLite supports multiple concurrent readers.",
    "A transformer uses self-attention.",
    "Nuclear energy powers the hover ship.",
]

for text in tests:
    encoding = tokenizer.encode(text)

    print()
    print("Text:  ", text)
    print("Tokens:", encoding.tokens)
    print("IDs:   ", encoding.ids)
    print("Back:  ", tokenizer.decode(encoding.ids))