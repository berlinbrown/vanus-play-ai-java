# config.py for Vanus model configuration
from dataclasses import asdict, dataclass

@dataclass
class VanusConfig:
    # Tokenizer vocabulary size.
    vocab_size: int = 8192

    # Maximum number of tokens processed at once.
    context_length: int = 256

    # Number of Transformer blocks.
    num_layers: int = 8

    # Size of each token's internal representation.
    hidden_size: int = 384

    # Number of parallel attention heads.
    num_heads: int = 6

    # Size of the SwiGLU feed-forward layer.
    intermediate_size: int = 1024

    # No dropout initially, matching modern Llama-style configurations.
    dropout: float = 0.0

    # Base frequency used by rotary positional embeddings.
    rope_theta: float = 10_000.0

    # Small constant used by RMSNorm.
    rms_norm_epsilon: float = 1e-5

    # Standard deviation for initial model weights.
    initializer_range: float = 0.02

    # Share the input embedding matrix with the output projection.
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive"
            )

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)