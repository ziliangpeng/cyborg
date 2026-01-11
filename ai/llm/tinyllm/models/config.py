"""Model configuration for TinyLLM."""

from dataclasses import dataclass


@dataclass
class GPT2Config:
    """Configuration for GPT-2 model variants."""

    vocab_size: int = 50257
    n_positions: int = 1024  # max sequence length
    n_embd: int = 768  # embedding dimension
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_inner: int = 3072  # FFN intermediate size (4 * n_embd)
    layer_norm_epsilon: float = 1e-5

    @classmethod
    def gpt2_small(cls) -> "GPT2Config":
        """GPT-2 Small (124M parameters)."""
        return cls()

    @classmethod
    def gpt2_medium(cls) -> "GPT2Config":
        """GPT-2 Medium (355M parameters)."""
        return cls(n_embd=1024, n_layer=24, n_head=16, n_inner=4096)

    @classmethod
    def gpt2_large(cls) -> "GPT2Config":
        """GPT-2 Large (774M parameters)."""
        return cls(n_embd=1280, n_layer=36, n_head=20, n_inner=5120)

    @classmethod
    def gpt2_xl(cls) -> "GPT2Config":
        """GPT-2 XL (1.5B parameters)."""
        return cls(n_embd=1600, n_layer=48, n_head=25, n_inner=6400)
