from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

OUTPUT_DIR = Path("tokenizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8192,
    min_frequency=2,
    special_tokens=[
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|endoftext|>",
    ],
)

tokenizer.train(
    files=["data/processed/train.txt"],
    trainer=trainer,
)

tokenizer.save("tokenizer/tokenizer.json")

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")