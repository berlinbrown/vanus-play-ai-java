# model.py for the Vanus model implementation
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VanusConfig

@dataclass
class VanusOutput:
    """
    Output returned by the model.

    logits:
        Raw next-token scores with shape:
        [batch_size, sequence_length, vocabulary_size]

    loss:
        Cross-entropy loss when training targets are supplied.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike standard LayerNorm, RMSNorm does not subtract the mean.
    It rescales each vector based on its root-mean-square magnitude.
    """

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.epsilon = epsilon

        # One learned scaling value for each hidden dimension.
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate in float32 for numerical stability.
        original_dtype = x.dtype

        x_float = x.float()

        variance = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        normalized = x_float * torch.rsqrt(
            variance + self.epsilon
        )

        return (
            normalized.to(original_dtype)
            * self.weight
        )


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings, commonly called RoPE.

    RoPE rotates pairs of query/key dimensions according to their
    token positions. This gives attention information about token
    order without adding a separate positional embedding vector.
    """

    def __init__(
        self,
        head_size: int,
        maximum_length: int,
        theta: float = 10_000.0,
    ):
        super().__init__()

        if head_size % 2 != 0:
            raise ValueError(
                "RoPE requires an even attention head size"
            )

        # Frequencies for half of the head dimensions.
        dimension_indices = torch.arange(
            0,
            head_size,
            2,
            dtype=torch.float32,
        )

        inverse_frequencies = 1.0 / (
            theta ** (dimension_indices / head_size)
        )

        positions = torch.arange(
            maximum_length,
            dtype=torch.float32,
        )

        # Shape:
        # [maximum_length, head_size / 2]
        frequencies = torch.outer(
            positions,
            inverse_frequencies,
        )

        self.register_buffer(
            "cosine",
            frequencies.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sine",
            frequencies.sin(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x shape:
            [batch_size, num_heads, sequence_length, head_size]

        Returns cosine and sine tensors shaped for broadcasting:
            [1, 1, sequence_length, head_size / 2]
        """

        sequence_length = x.size(-2)

        cosine = self.cosine[:sequence_length]
        sine = self.sine[:sequence_length]

        cosine = cosine.to(
            device=x.device,
            dtype=x.dtype,
        )

        sine = sine.to(
            device=x.device,
            dtype=x.dtype,
        )

        return (
            cosine.unsqueeze(0).unsqueeze(0),
            sine.unsqueeze(0).unsqueeze(0),
        )


def apply_rotary_embedding(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate adjacent pairs of the final dimension.

    Input:
        [x0, x1, x2, x3, ...]

    Rotated pairs:
        x0' = x0*cos - x1*sin
        x1' = x0*sin + x1*cos
    """

    even_values = x[..., 0::2]
    odd_values = x[..., 1::2]

    rotated_even = (
        even_values * cosine
        - odd_values * sine
    )

    rotated_odd = (
        even_values * sine
        + odd_values * cosine
    )

    # Recombine the even and odd values in their original order.
    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    "Causal" means that a token can attend only to itself and tokens
    that appeared before it. It cannot look at future target tokens.
    """

    def __init__(self, config: VanusConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.dropout = config.dropout

        # Separate query, key and value projections make the internal
        # operation easier to inspect than a single combined projection.
        self.query_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.key_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.value_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.output_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        self.rotary_embedding = RotaryEmbedding(
            head_size=config.head_size,
            maximum_length=config.context_length,
            theta=config.rope_theta,
        )

        self.residual_dropout = nn.Dropout(
            config.dropout
        )

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert:

            [batch, sequence, hidden]

        into:

            [batch, heads, sequence, head_size]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        )

        return x.transpose(1, 2)

    def _combine_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert:

            [batch, heads, sequence, head_size]

        back into:

            [batch, sequence, hidden]
        """

        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        queries = self._split_heads(
            self.query_projection(x)
        )

        keys = self._split_heads(
            self.key_projection(x)
        )

        values = self._split_heads(
            self.value_projection(x)
        )

        cosine, sine = self.rotary_embedding(queries)

        queries = apply_rotary_embedding(
            queries,
            cosine,
            sine,
        )

        keys = apply_rotary_embedding(
            keys,
            cosine,
            sine,
        )

        attention_dropout = (
            self.dropout if self.training else 0.0
        )

        # PyTorch's optimized scaled-dot-product attention performs:
        #
        # softmax((Q @ K^T) / sqrt(head_size)) @ V
        #
        # is_causal=True automatically masks future positions.
        attention_output = F.scaled_dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            dropout_p=attention_dropout,
            is_causal=True,
        )

        attention_output = self._combine_heads(
            attention_output
        )

        output = self.output_projection(
            attention_output
        )

        return self.residual_dropout(output)


class SwiGLUFeedForward(nn.Module):
    """
    Llama-style SwiGLU feed-forward network.

    SwiGLU uses two parallel input projections:

        SiLU(gate(x)) * up(x)

    The result is then projected back to hidden_size.
    """

    def __init__(self, config: VanusConfig):
        super().__init__()

        self.gate_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_projection = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(
            self.gate_projection(x)
        )

        expanded = self.up_projection(x)

        combined = gated * expanded

        output = self.down_projection(combined)

        return self.dropout(output)


class TransformerBlock(nn.Module):
    """
    One pre-normalized Llama-style Transformer block.

    The block contains:

        RMSNorm
        Self-attention
        Residual connection
        RMSNorm
        SwiGLU feed-forward network
        Residual connection
    """

    def __init__(self, config: VanusConfig):
        super().__init__()

        self.attention_norm = RMSNorm(
            hidden_size=config.hidden_size,
            epsilon=config.rms_norm_epsilon,
        )

        self.attention = CausalSelfAttention(config)

        self.feed_forward_norm = RMSNorm(
            hidden_size=config.hidden_size,
            epsilon=config.rms_norm_epsilon,
        )

        self.feed_forward = SwiGLUFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization:
        # Normalize before attention, then add the residual.
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Normalize before the feed-forward network,
        # then add its residual.
        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        return x


class VanusModel(nn.Module):
    """
    Small decoder-only Llama-style language model.
    """

    def __init__(self, config: VanusConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            hidden_size=config.hidden_size,
            epsilon=config.rms_norm_epsilon,
        )

        self.language_model_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._initialize_weights)

        if config.tie_word_embeddings:
            # Input token embeddings and output vocabulary weights
            # become the same Parameter object.
            self.language_model_head.weight = (
                self.token_embedding.weight
            )

    def _initialize_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> VanusOutput:
        """
        input_ids:
            Integer token IDs with shape:
            [batch_size, sequence_length]

        targets:
            Expected next-token IDs with the same shape.

        Example:

            input_ids = [The, cat, is, happy]
            targets   = [cat, is, happy, .]
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]"
            )

        _, sequence_length = input_ids.shape

        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.config.context_length}"
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                "input_ids must use torch.long integer IDs"
            )

        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_norm(hidden_states)

        logits = self.language_model_head(
            hidden_states
        )

        loss = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have the same shape as input_ids"
                )

            # Convert:
            #
            # logits: [batch, sequence, vocabulary]
            # targets: [batch, sequence]
            #
            # into the shapes expected by cross_entropy:
            #
            # logits: [batch * sequence, vocabulary]
            # targets: [batch * sequence]
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.config.vocab_size,
                ),
                targets.reshape(-1),
            )

        return VanusOutput(
            logits=logits,
            loss=loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        maximum_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        This initial implementation recalculates attention for the entire
        active context after every new token. Later, a KV cache can make
        generation much faster.
        """

        if temperature < 0.0:
            raise ValueError(
                "temperature cannot be negative"
            )

        self.eval()

        generated = input_ids

        for _ in range(maximum_new_tokens):
            # The model can only process context_length tokens.
            context = generated[
                :,
                -self.config.context_length:
            ]

            output = self(context)

            # Only the final sequence position predicts the next token.
            next_token_logits = output.logits[:, -1, :]

            if temperature == 0.0:
                # Greedy generation: always select the most likely token.
                next_token = torch.argmax(
                    next_token_logits,
                    dim=-1,
                    keepdim=True,
                )

            else:
                next_token_logits = (
                    next_token_logits / temperature
                )

                if top_k is not None:
                    effective_top_k = min(
                        top_k,
                        next_token_logits.size(-1),
                    )

                    top_values, _ = torch.topk(
                        next_token_logits,
                        effective_top_k,
                    )

                    cutoff = top_values[:, -1].unsqueeze(-1)

                    next_token_logits = (
                        next_token_logits.masked_fill(
                            next_token_logits < cutoff,
                            float("-inf"),
                        )
                    )

                probabilities = F.softmax(
                    next_token_logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            generated = torch.cat(
                (generated, next_token),
                dim=1,
            )

            if (
                eos_token_id is not None
                and torch.all(next_token == eos_token_id)
            ):
                break

        return generated

    def count_parameters(self) -> int:
        """
        Count unique trainable parameters.

        Tracking parameter identity avoids counting tied embedding
        weights twice.
        """

        unique_parameters = {
            id(parameter): parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        }

        return sum(
            parameter.numel()
            for parameter in unique_parameters.values()
        )