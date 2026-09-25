# The effective number of tokens per optimizer update is:
from dataclasses import asdict, dataclass

@dataclass
class TrainingConfig:
    # Input files
    train_data_path: str = "data/processed/train.bin"
    validation_data_path: str = "data/processed/validation.bin"

    # Output directories
    checkpoint_directory: str = "checkpoints"
    tensorboard_directory: str = "reports/tensorboard"

    # Reproducibility
    random_seed: int = 1337

    # Batch dimensions
    sequence_length: int = 256
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8

    # Training duration
    maximum_steps: int = 2_500

    # AdamW
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8

    # Learning-rate warmup
    warmup_steps: int = 200

    # Stability
    maximum_gradient_norm: float = 1.0

    # Evaluation
    evaluation_interval: int = 250
    evaluation_batches: int = 20

    # Checkpointing
    checkpoint_interval: int = 500

    # Console and TensorBoard logging
    logging_interval: int = 10

    @property
    def tokens_per_micro_batch(self) -> int:
        return (
            self.micro_batch_size
            * self.sequence_length
        )

    @property
    def tokens_per_optimizer_step(self) -> int:
        return (
            self.tokens_per_micro_batch
            * self.gradient_accumulation_steps
        )

    def to_dict(self) -> dict:
        return asdict(self)