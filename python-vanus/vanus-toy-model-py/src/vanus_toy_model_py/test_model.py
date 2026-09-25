import torch

from vanus_toy_model_py.model import (
    VanusConfig,
    VanusModel,
)

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    torch.manual_seed(1337)

    device = select_device()

    config = VanusConfig()

    model = VanusModel(config)
    model = model.to(device)

    print(f"Device: {device}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Layers: {config.num_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention heads: {config.num_heads}")
    print(f"Head size: {config.head_size}")
    print(f"Context length: {config.context_length}")
    print(f"Vocabulary size: {config.vocab_size}")

    batch_size = 2
    sequence_length = 32

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )

    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )

    output = model(
        input_ids=input_ids,
        targets=targets,
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Logit shape: {output.logits.shape}")
    print(f"Initial loss: {output.loss.item():.4f}")

    output.loss.backward()

    embedding_gradient = (
        model.token_embedding.weight.grad
    )

    print(
        "Embedding gradient exists:",
        embedding_gradient is not None,
    )

    print(
        "Embedding gradient norm:",
        embedding_gradient.norm().item(),
    )

    generated = model.generate(
        input_ids=input_ids[:1, :5],
        maximum_new_tokens=10,
        temperature=0.8,
        top_k=50,
    )

    print(f"Generated shape: {generated.shape}")
    print(f"Generated IDs: {generated[0].tolist()}")

if __name__ == "__main__":
    main()