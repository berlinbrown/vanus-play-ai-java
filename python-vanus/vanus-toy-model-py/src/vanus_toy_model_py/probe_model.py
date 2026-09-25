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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed probes against a Vanus checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory to inspect.",
    )

    parser.add_argument(
        "--prompts",
        default="data/probe_prompts.json",
        help="JSON file containing probe prompts.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Number of tokens to generate per prompt.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampled generation.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Limit sampling to the top-k tokens.",
    )

    parser.add_argument(
        "--top-predictions",
        type=int,
        default=10,
        help="Number of next-token predictions to display.",
    )

    return parser.parse_args()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


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

    training_state_path = (
        checkpoint_directory / "training_state.json"
    )

    tokenizer_path = (
        checkpoint_directory / "tokenizer.json"
    )

    required_files = [
        model_path,
        model_config_path,
        training_state_path,
        tokenizer_path,
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing checkpoint file: {required_file}"
            )

    model_config_values = load_json(
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

    training_state = load_json(
        training_state_path
    )

    return model, tokenizer, training_state


@torch.no_grad()
def get_top_predictions(
    model: VanusModel,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    count: int,
) -> list[dict]:
    output = model(input_ids=input_ids)

    next_token_logits = output.logits[0, -1, :]

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    top_probabilities, top_ids = torch.topk(
        probabilities,
        k=count,
    )

    results = []

    for probability, token_id in zip(
        top_probabilities.tolist(),
        top_ids.tolist(),
    ):
        token_text = tokenizer.decode(
            [token_id]
        )

        token_name = tokenizer.id_to_token(
            token_id
        )

        results.append(
            {
                "id": token_id,
                "token": token_name,
                "decoded": token_text,
                "probability": probability,
            }
        )

    return results


@torch.no_grad()
def generate_continuation(
    model: VanusModel,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    maximum_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_token_id: int | None,
) -> dict:
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        maximum_new_tokens=maximum_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )

    complete_ids = generated_ids[0].tolist()

    continuation_ids = complete_ids[
        len(prompt_ids):
    ]

    return {
        "complete_text": tokenizer.decode(
            complete_ids
        ),
        "continuation": tokenizer.decode(
            continuation_ids
        ),
        "generated_token_ids": continuation_ids,
    }


def main() -> None:
    arguments = parse_arguments()

    torch.manual_seed(1337)

    checkpoint_directory = Path(
        arguments.checkpoint
    )

    prompt_path = Path(arguments.prompts)

    prompts = load_json(prompt_path)

    if not isinstance(prompts, list):
        raise ValueError(
            "The prompt file must contain a JSON list"
        )

    if not prompts:
        raise ValueError(
            "The prompt file contains no prompts"
        )

    device = select_device()

    model, tokenizer, training_state = (
        load_checkpoint(
            checkpoint_directory=checkpoint_directory,
            device=device,
        )
    )

    eos_token_id = tokenizer.token_to_id(
        "<|endoftext|>"
    )

    print()
    print("Vanus checkpoint probe")
    print("======================")
    print(f"Device: {device}")
    print(
        "Checkpoint:",
        checkpoint_directory,
    )
    print(
        "Training step:",
        training_state["step"],
    )
    print(
        "Tokens seen:",
        f"{training_state['tokens_seen']:,}",
    )

    report = {
        "checkpoint": str(checkpoint_directory),
        "training_state": training_state,
        "prompts": [],
    }

    for prompt_number, prompt in enumerate(
        prompts,
        start=1,
    ):
        if not isinstance(prompt, str):
            raise ValueError(
                f"Prompt {prompt_number} is not a string"
            )

        prompt_ids = tokenizer.encode(
            prompt
        ).ids

        if len(prompt_ids) > model.config.context_length:
            prompt_ids = prompt_ids[
                -model.config.context_length:
            ]

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        top_predictions = get_top_predictions(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            count=arguments.top_predictions,
        )

        # Temperature 0 selects the most likely token every time.
        greedy_result = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            maximum_new_tokens=arguments.max_new_tokens,
            temperature=0.0,
            top_k=None,
            eos_token_id=eos_token_id,
        )

        sampled_result = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            device=device,
            maximum_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
            eos_token_id=eos_token_id,
        )

        prompt_report = {
            "prompt": prompt,
            "prompt_token_ids": prompt_ids,
            "top_next_tokens": top_predictions,
            "greedy": greedy_result,
            "sampled": sampled_result,
        }

        report["prompts"].append(
            prompt_report
        )

        print()
        print(f"Prompt {prompt_number}")
        print("--------------------")
        print(prompt)

        print()
        print("Top next-token predictions:")

        for prediction in top_predictions[:5]:
            print(
                f"  {prediction['probability']:.4%} "
                f"id={prediction['id']:4d} "
                f"token={prediction['token']!r} "
                f"decoded={prediction['decoded']!r}"
            )

        print()
        print("Greedy continuation:")
        print(greedy_result["continuation"])

        print()
        print("Sampled continuation:")
        print(sampled_result["continuation"])

    report_directory = Path(
        "reports/probes"
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_name = checkpoint_directory.name

    report_path = (
        report_directory
        / f"{checkpoint_name}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            report,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved probe report: {report_path}")


if __name__ == "__main__":
    main()