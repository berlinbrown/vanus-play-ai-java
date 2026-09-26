# Chatbot - final step
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model
from tokenizers import Tokenizer

from vanus_toy_model_py.model import (
    VanusConfig,
    VanusModel,
)

DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a trained Vanus model."
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint directory to load. If omitted, "
            "the latest valid checkpoint is selected."
        ),
    )

    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Directory containing step-* checkpoint directories.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of tokens generated per response.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "Sampling temperature. Use 0 for deterministic "
            "greedy generation."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Restrict sampling to the top-k predicted tokens.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used during sampling.",
    )

    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def validate_checkpoint(
    checkpoint: Path,
) -> tuple[bool, list[Path]]:
    required_files = [
        checkpoint / "model.safetensors",
        checkpoint / "model_config.json",
        checkpoint / "tokenizer.json",
        checkpoint / "training_state.json",
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    return len(missing_files) == 0, missing_files


def checkpoint_step(checkpoint: Path) -> int | None:
    """
    Convert a directory name such as step-000500 into integer 500.
    """

    if not checkpoint.name.startswith("step-"):
        return None

    step_text = checkpoint.name.removeprefix("step-")

    try:
        return int(step_text)
    except ValueError:
        return None


def find_latest_checkpoint(
    checkpoint_root: Path,
) -> Path:
    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Checkpoint root does not exist: {checkpoint_root}"
        )

    candidates: list[tuple[int, Path]] = []

    for path in checkpoint_root.glob("step-*"):
        if not path.is_dir():
            continue

        step = checkpoint_step(path)

        if step is None:
            continue

        is_valid, _ = validate_checkpoint(path)

        if not is_valid:
            continue

        candidates.append((step, path))

    if not candidates:
        raise FileNotFoundError(
            f"No valid checkpoints found in {checkpoint_root}"
        )

    candidates.sort(
        key=lambda candidate: candidate[0]
    )

    _, latest_checkpoint = candidates[-1]

    return latest_checkpoint


def resolve_checkpoint(
    explicit_checkpoint: str | None,
    checkpoint_root: str,
) -> Path:
    if explicit_checkpoint is not None:
        checkpoint = Path(explicit_checkpoint)

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint}"
            )

        is_valid, missing_files = validate_checkpoint(
            checkpoint
        )

        if not is_valid:
            missing_text = "\n".join(
                f"  - {path}"
                for path in missing_files
            )

            raise FileNotFoundError(
                "Checkpoint is incomplete. Missing:\n"
                f"{missing_text}"
            )

        return checkpoint

    return find_latest_checkpoint(
        Path(checkpoint_root)
    )


def load_checkpoint(
    checkpoint_directory: Path,
    device: torch.device,
) -> tuple[VanusModel, Tokenizer, dict]:
    model_path = (
        checkpoint_directory / "model.safetensors"
    )

    model_config_path = (
        checkpoint_directory / "model_config.json"
    )

    tokenizer_path = (
        checkpoint_directory / "tokenizer.json"
    )

    training_state_path = (
        checkpoint_directory / "training_state.json"
    )

    model_config_values = read_json(
        model_config_path
    )

    model_config = VanusConfig(
        **model_config_values
    )

    model = VanusModel(model_config)
    model = model.to(device)

    load_model(
        model,
        str(model_path),
        strict=True,
    )

    model.eval()

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    training_state = read_json(
        training_state_path
    )

    return model, tokenizer, training_state


def clean_response(text: str) -> str:
    """
    Remove generated conversation boundaries and document markers.
    """

    stopping_markers = [
        "<|endoftext|>",
        "<|eos|>",
        "\nUser:",
        "\nYou:",
        "\nHuman:",
        "\nAssistant:",
        "\nVanus:",
    ]

    earliest_position = None

    for marker in stopping_markers:
        position = text.find(marker)

        if position == -1:
            continue

        if (
            earliest_position is None
            or position < earliest_position
        ):
            earliest_position = position

    if earliest_position is not None:
        text = text[:earliest_position]

    return text.strip()


def build_transcript(
    conversation: list[tuple[str, str]],
    current_message: str,
) -> str:
    sections = []

    for user_message, assistant_message in conversation:
        sections.append(
            f"User: {user_message}\n"
            f"Assistant: {assistant_message}"
        )

    sections.append(
        f"User: {current_message}\nAssistant:"
    )

    return "\n\n".join(sections)


@torch.no_grad()
def generate_response(
    model: VanusModel,
    tokenizer: Tokenizer,
    transcript: str,
    device: torch.device,
    maximum_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    prompt_ids = tokenizer.encode(
        transcript
    ).ids

    # Reserve part of the context window for the generated answer.
    maximum_generation_length = min(
        maximum_new_tokens,
        model.config.context_length - 1,
    )

    maximum_prompt_length = (
        model.config.context_length
        - maximum_generation_length
    )

    # When conversation history becomes too long, retain its newest
    # tokens because they contain the current question.
    prompt_ids = prompt_ids[
        -maximum_prompt_length:
    ]

    if not prompt_ids:
        raise ValueError(
            "The encoded prompt contains no tokens"
        )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    eos_token_id = tokenizer.token_to_id(
        "<|endoftext|>"
    )

    generated = model.generate(
        input_ids=input_ids,
        maximum_new_tokens=maximum_generation_length,
        temperature=temperature,
        top_k=top_k if temperature > 0 else None,
        eos_token_id=eos_token_id,
    )

    all_ids = generated[0].tolist()

    response_ids = all_ids[
        len(prompt_ids):
    ]

    response_text = tokenizer.decode(
        response_ids,
        skip_special_tokens=False,
    )

    return clean_response(response_text)


def print_model_information(
    checkpoint_directory: Path,
    model: VanusModel,
    training_state: dict,
    device: torch.device,
) -> None:
    print()
    print("Vanus Toy Chat")
    print("==============")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_directory}")
    print(
        "Parameters:",
        f"{model.count_parameters():,}",
    )
    print(
        "Training step:",
        f"{training_state.get('step', 'unknown')}",
    )
    print(
        "Tokens seen:",
        f"{training_state.get('tokens_seen', 0):,}",
    )
    print(
        "Training loss:",
        training_state.get(
            "training_loss",
            "unknown",
        ),
    )
    print(
        "Validation loss:",
        training_state.get(
            "validation_loss",
            "unknown",
        ),
    )
    print()
    print("Commands:")
    print("  /clear  Clear conversation history")
    print("  /info   Display model information")
    print("  /quit   Exit")
    print()


def main() -> None:
    arguments = parse_arguments()

    if arguments.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be positive"
        )

    if arguments.temperature < 0:
        raise ValueError(
            "--temperature cannot be negative"
        )

    if arguments.top_k <= 0:
        raise ValueError(
            "--top-k must be positive"
        )

    torch.manual_seed(arguments.seed)

    device = select_device()

    checkpoint_directory = resolve_checkpoint(
        explicit_checkpoint=arguments.checkpoint,
        checkpoint_root=arguments.checkpoint_root,
    )

    model, tokenizer, training_state = load_checkpoint(
        checkpoint_directory=checkpoint_directory,
        device=device,
    )

    print_model_information(
        checkpoint_directory=checkpoint_directory,
        model=model,
        training_state=training_state,
        device=device,
    )

    conversation: list[tuple[str, str]] = []

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_message:
            continue

        command = user_message.lower()

        if command in {
            "/quit",
            "/exit",
            "quit",
            "exit",
        }:
            print("Goodbye.")
            break

        if command == "/clear":
            conversation.clear()
            print("Conversation cleared.")
            print()
            continue

        if command == "/info":
            print_model_information(
                checkpoint_directory=checkpoint_directory,
                model=model,
                training_state=training_state,
                device=device,
            )
            continue

        transcript = build_transcript(
            conversation=conversation,
            current_message=user_message,
        )

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            transcript=transcript,
            device=device,
            maximum_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
        )

        if not response:
            response = (
                "[The model generated no visible response.]"
            )

        print(f"Vanus: {response}")
        print()

        conversation.append(
            (user_message, response)
        )

if __name__ == "__main__":
    main()