import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from torch.utils.tensorboard import SummaryWriter

from vanus_toy_model_py.model import (
    VanusConfig,
    VanusModel,
)
from vanus_toy_model_py.training_config import (
    TrainingConfig,
)


# Read optional training settings supplied on the command line.
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line options that override the default training settings.

    Returns the parsed arguments so main() can apply only the values that the
    user explicitly supplied.
    """
    parser = argparse.ArgumentParser(
        description="Train the Vanus toy language model."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Override the configured maximum number "
            "of optimizer steps."
        ),
    )

    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Override the configured micro-batch size.",
    )

    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=None,
        help="Override the validation interval.",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override the checkpoint interval.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint directory from which to resume training.",
    )

    return parser.parse_args()


# Select the best computation device available on this machine.
def select_device() -> torch.device:
    """
    Choose the device that will perform the model calculations.

    Apple Silicon MPS is preferred, followed by an NVIDIA CUDA GPU, with the
    CPU used as the fallback.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# Initialize every random-number generator used during training.
def set_random_seeds(seed: int) -> None:
    """
    Apply the same seed to Python, NumPy, and PyTorch.

    Using a shared seed makes data sampling and model initialization more
    repeatable between training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Open a tokenized dataset while keeping its contents on disk.
def load_token_data(path: str) -> np.memmap:
    """
    Memory-map a binary file containing unsigned 16-bit token IDs.

    Memory mapping lets training read portions of a large dataset without
    loading the complete file into RAM.
    """
    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Token data does not exist: {data_path}"
        )

    data = np.memmap(
        data_path,
        dtype=np.uint16,
        mode="r",
    )

    print(
        f"Loaded {data_path}: "
        f"{len(data):,} tokens"
    )

    return data


# Build a random batch of inputs and their next-token targets.
def get_batch(
    data: np.memmap,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select random token sequences.

    For each starting position:

        input  = tokens[start : start + sequence_length]
        target = tokens[start + 1 : start + sequence_length + 1]
    """

    maximum_start = len(data) - sequence_length - 1

    if maximum_start <= 0:
        raise ValueError(
            "Dataset is smaller than the sequence length"
        )

    starting_positions = np.random.randint(
        low=0,
        high=maximum_start,
        size=batch_size,
    )

    input_batch = np.stack(
        [
            np.asarray(
                data[
                    position:
                    position + sequence_length
                ],
                dtype=np.int64,
            )
            for position in starting_positions
        ]
    )

    target_batch = np.stack(
        [
            np.asarray(
                data[
                    position + 1:
                    position + sequence_length + 1
                ],
                dtype=np.int64,
            )
            for position in starting_positions
        ]
    )

    input_tensor = torch.from_numpy(
        input_batch
    ).to(
        device=device,
        dtype=torch.long,
    )

    target_tensor = torch.from_numpy(
        target_batch
    ).to(
        device=device,
        dtype=torch.long,
    )

    return input_tensor, target_tensor


# Create AdamW groups with weight decay applied where appropriate.
def configure_optimizer(
    model: VanusModel,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """
    Apply weight decay to matrix-shaped weights, but not to
    one-dimensional normalization parameters.
    """

    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )

    print(
        "Parameters with weight decay:",
        sum(
            parameter.numel()
            for parameter in decay_parameters
        ),
    )

    print(
        "Parameters without weight decay:",
        sum(
            parameter.numel()
            for parameter in no_decay_parameters
        ),
    )

    return optimizer


# Compute the learning rate for a particular optimizer step.
def calculate_learning_rate(
    step: int,
    config: TrainingConfig,
) -> float:
    """
    Linear warmup followed by cosine decay.
    """

    if step < config.warmup_steps:
        warmup_fraction = (
            (step + 1)
            / config.warmup_steps
        )

        return (
            config.learning_rate
            * warmup_fraction
        )

    if step >= config.maximum_steps:
        return config.minimum_learning_rate

    decay_progress = (
        (step - config.warmup_steps)
        / (
            config.maximum_steps
            - config.warmup_steps
        )
    )

    cosine_value = 0.5 * (
        1.0 + math.cos(
            math.pi * decay_progress
        )
    )

    return (
        config.minimum_learning_rate
        + cosine_value
        * (
            config.learning_rate
            - config.minimum_learning_rate
        )
    )


# Apply the scheduled learning rate to every optimizer group.
def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """
    Update every optimizer parameter group to use the supplied learning rate.

    Both the decayed and non-decayed parameter groups must follow the same
    warmup and cosine-decay schedule.
    """
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


# Measure validation loss without calculating or storing gradients.
@torch.no_grad()
def evaluate(
    model: VanusModel,
    validation_data: np.memmap,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    """
    Calculate average validation loss without updating model weights.
    """

    was_training = model.training
    model.eval()

    losses = []

    for _ in range(config.evaluation_batches):
        input_ids, targets = get_batch(
            data=validation_data,
            batch_size=config.micro_batch_size,
            sequence_length=config.sequence_length,
            device=device,
        )

        output = model(
            input_ids=input_ids,
            targets=targets,
        )

        losses.append(output.loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses)


# Restore model and optimizer state from an earlier checkpoint.
def load_checkpoint(
    checkpoint_path: str,
    model: VanusModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """
    Load saved weights, optimizer values, and the completed step number.

    The returned step tells the training loop where the resumed run should
    continue.
    """
    checkpoint = Path(checkpoint_path)

    model_file = checkpoint / "model.safetensors"
    optimizer_file = checkpoint / "optimizer.pt"
    state_file = checkpoint / "training_state.json"

    for required_file in [
        model_file,
        optimizer_file,
        state_file,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing checkpoint file: {required_file}"
            )

    load_model(
        model,
        str(model_file),
        strict=True,
    )

    optimizer_state = torch.load(
        optimizer_file,
        map_location=device,
        weights_only=True,
    )

    optimizer.load_state_dict(optimizer_state)

    with state_file.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        training_state = json.load(input_file)

    completed_step = int(training_state["step"])

    print(
        f"Resumed checkpoint {checkpoint} "
        f"at step {completed_step}"
    )

    return completed_step


# Store everything needed to inspect or resume a training run.
def save_checkpoint(
    model: VanusModel,
    optimizer: torch.optim.Optimizer,
    model_config: VanusConfig,
    training_config: TrainingConfig,
    step: int,
    training_loss: float,
    validation_loss: float | None,
) -> Path:
    """
    Save model weights, optimizer state, configuration, and training metrics.

    A tokenizer copy is also included when the tokenizer file is available.
    """
    checkpoint_root = Path(
        training_config.checkpoint_directory
    )

    checkpoint_path = (
        checkpoint_root / f"step-{step:06d}"
    )

    checkpoint_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = checkpoint_path / "model.safetensors"

    # save_model understands that our input embeddings and output
    # projection share the same underlying Parameter.
    save_model(
        model,
        str(model_path),
    )

    torch.save(
        optimizer.state_dict(),
        checkpoint_path / "optimizer.pt",
    )

    training_state = {
        "step": step,
        "tokens_seen": (
            step
            * training_config.tokens_per_optimizer_step
        ),
        "training_loss": training_loss,
        "validation_loss": validation_loss,
    }

    with (
        checkpoint_path / "training_state.json"
    ).open("w", encoding="utf-8") as output:
        json.dump(
            training_state,
            output,
            indent=2,
        )

    with (
        checkpoint_path / "model_config.json"
    ).open("w", encoding="utf-8") as output:
        json.dump(
            model_config.to_dict(),
            output,
            indent=2,
        )

    with (
        checkpoint_path / "training_config.json"
    ).open("w", encoding="utf-8") as output:
        json.dump(
            training_config.to_dict(),
            output,
            indent=2,
        )

    tokenizer_path = Path(
        "tokenizer/tokenizer.json"
    )

    if tokenizer_path.exists():
        shutil.copy2(
            tokenizer_path,
            checkpoint_path / "tokenizer.json",
        )

    print(f"Saved checkpoint: {checkpoint_path}")

    return checkpoint_path


# Execute the complete model-training loop.
def train(
    model_config: VanusConfig,
    training_config: TrainingConfig,
    resume_from: str | None = None,
) -> None:
    """
    Train the language model using the supplied model and training settings.

    This coordinates data loading, gradient updates, validation, TensorBoard
    logging, checkpoint creation, and optional checkpoint resumption.
    """
    set_random_seeds(training_config.random_seed)

    device = select_device()

    print()
    print("Vanus model training")
    print("====================")
    print(f"Device: {device}")
    print(
        "Tokens per optimizer step:",
        f"{training_config.tokens_per_optimizer_step:,}",
    )
    print(
        "Maximum optimizer steps:",
        f"{training_config.maximum_steps:,}",
    )
    print(
        "Maximum token presentations:",
        f"{training_config.maximum_steps * training_config.tokens_per_optimizer_step:,}",
    )

    train_data = load_token_data(
        training_config.train_data_path
    )

    validation_data = load_token_data(
        training_config.validation_data_path
    )

    model = VanusModel(model_config)
    model = model.to(device)

    print(
        "Model parameters:",
        f"{model.count_parameters():,}",
    )

    optimizer = configure_optimizer(
        model=model,
        config=training_config,
    )

    start_step = 0

    if resume_from is not None:
        start_step = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    checkpoint_directory = Path(
        training_config.checkpoint_directory
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensorboard_directory = Path(
        training_config.tensorboard_directory
    )

    tensorboard_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(
        log_dir=str(tensorboard_directory)
    )

    # Evaluate the randomly initialized model before training.
    initial_validation_loss = evaluate(
        model=model,
        validation_data=validation_data,
        config=training_config,
        device=device,
    )

    print(
        "Initial validation loss:",
        f"{initial_validation_loss:.4f}",
    )

    writer.add_scalar(
        "loss/validation",
        initial_validation_loss,
        0,
    )

    writer.add_scalar(
        "perplexity/validation",
        math.exp(
            min(initial_validation_loss, 20.0)
        ),
        0,
    )

    model.train()

    total_start_time = time.perf_counter()
    interval_start_time = total_start_time
    interval_tokens = 0

    latest_training_loss = float("nan")
    latest_validation_loss = (
        initial_validation_loss
    )

    try:
        # Each outer-loop iteration performs one optimizer update. That update
        # combines gradients from several smaller micro-batches to reduce the
        # amount of memory required at one time.
        for step in range(
            start_step + 1,
            training_config.maximum_steps + 1,
        ):
            step_learning_rate = calculate_learning_rate(
                step=step - 1,
                config=training_config,
            )

            set_learning_rate(
                optimizer,
                step_learning_rate,
            )

            # Discard gradients left over from the previous optimizer step.
            optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0

            for _ in range(
                training_config.gradient_accumulation_steps
            ):
                input_ids, targets = get_batch(
                    data=train_data,
                    batch_size=training_config.micro_batch_size,
                    sequence_length=training_config.sequence_length,
                    device=device,
                )

                output = model(
                    input_ids=input_ids,
                    targets=targets,
                )

                # Dividing the loss makes accumulated gradients equal
                # to the average across the micro-batches.
                scaled_loss = (
                    output.loss
                    / training_config.gradient_accumulation_steps
                )

                scaled_loss.backward()

                accumulated_loss += output.loss.item()

            latest_training_loss = (
                accumulated_loss
                / training_config.gradient_accumulation_steps
            )

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=(
                        training_config.maximum_gradient_norm
                    ),
                )
            )

            # AdamW applies the accumulated, clipped gradients to the weights.
            optimizer.step()

            interval_tokens += (
                training_config.tokens_per_optimizer_step
            )

            writer.add_scalar(
                "loss/train",
                latest_training_loss,
                step,
            )

            writer.add_scalar(
                "learning_rate",
                step_learning_rate,
                step,
            )

            writer.add_scalar(
                "gradients/global_norm",
                gradient_norm.item(),
                step,
            )

            writer.add_scalar(
                "tokens/processed",
                (
                    step
                    * training_config.tokens_per_optimizer_step
                ),
                step,
            )

            if (
                step
                % training_config.logging_interval
                == 0
            ):
                # Measure throughput over the latest logging interval rather
                # than averaging it over the entire training run.
                current_time = time.perf_counter()

                interval_seconds = (
                    current_time - interval_start_time
                )

                tokens_per_second = (
                    interval_tokens / interval_seconds
                )

                elapsed_seconds = (
                    current_time - total_start_time
                )

                print(
                    f"step={step:5d} "
                    f"loss={latest_training_loss:.4f} "
                    f"lr={step_learning_rate:.6f} "
                    f"grad={gradient_norm.item():.4f} "
                    f"tokens/s={tokens_per_second:,.0f} "
                    f"elapsed={elapsed_seconds / 60:.1f}m"
                )

                writer.add_scalar(
                    "performance/tokens_per_second",
                    tokens_per_second,
                    step,
                )

                interval_start_time = current_time
                interval_tokens = 0

            if (
                step
                % training_config.evaluation_interval
                == 0
            ):
                # Evaluation temporarily switches the model to inference mode;
                # evaluate() restores training mode before returning.
                latest_validation_loss = evaluate(
                    model=model,
                    validation_data=validation_data,
                    config=training_config,
                    device=device,
                )

                validation_perplexity = math.exp(
                    min(latest_validation_loss, 20.0)
                )

                print(
                    f"validation step={step} "
                    f"loss={latest_validation_loss:.4f} "
                    f"perplexity={validation_perplexity:.2f}"
                )

                writer.add_scalar(
                    "loss/validation",
                    latest_validation_loss,
                    step,
                )

                writer.add_scalar(
                    "perplexity/validation",
                    validation_perplexity,
                    step,
                )

                writer.flush()

            if (
                step
                % training_config.checkpoint_interval
                == 0
            ):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    model_config=model_config,
                    training_config=training_config,
                    step=step,
                    training_loss=latest_training_loss,
                    validation_loss=latest_validation_loss,
                )

        # Ensure the final model is saved even when maximum_steps does
        # not fall exactly on the checkpoint interval.
        if (
            training_config.maximum_steps
            % training_config.checkpoint_interval
            != 0
        ):
            latest_validation_loss = evaluate(
                model=model,
                validation_data=validation_data,
                config=training_config,
                device=device,
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training_config=training_config,
                step=training_config.maximum_steps,
                training_loss=latest_training_loss,
                validation_loss=latest_validation_loss,
            )

    finally:
        writer.flush()
        writer.close()

    total_seconds = (
        time.perf_counter() - total_start_time
    )

    print()
    print("Training complete")
    print("=================")
    print(
        "Elapsed time:",
        f"{total_seconds / 60:.1f} minutes",
    )
    print(
        "Final training loss:",
        f"{latest_training_loss:.4f}",
    )
    print(
        "Latest validation loss:",
        f"{latest_validation_loss:.4f}",
    )


# Configure and launch training from the command line.
def main() -> None:
    """
    Create the default configurations and apply command-line overrides.

    Once configuration is complete, pass it to train() to begin the run.
    """
    arguments = parse_arguments()

    model_config = VanusConfig()
    training_config = TrainingConfig()

    if arguments.max_steps is not None:
        training_config.maximum_steps = (
            arguments.max_steps
        )

        # Avoid a warmup longer than a small test run.
        training_config.warmup_steps = min(
            training_config.warmup_steps,
            max(1, arguments.max_steps // 10),
        )

    if arguments.micro_batch_size is not None:
        training_config.micro_batch_size = (
            arguments.micro_batch_size
        )

    if arguments.evaluation_interval is not None:
        training_config.evaluation_interval = (
            arguments.evaluation_interval
        )

    if arguments.checkpoint_interval is not None:
        training_config.checkpoint_interval = (
            arguments.checkpoint_interval
        )

    train(
        model_config=model_config,
        training_config=training_config,
        resume_from=arguments.resume_from,
    )

if __name__ == "__main__":
    main()
