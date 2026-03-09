"""Feed-forward network for TinyLLM."""

from collections.abc import Callable

from tinygrad import Tensor
from tinygrad.nn import Linear

from .activations import gelu, silu


class FeedForward:
    """Position-wise feed-forward network with configurable activation."""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: Callable[[Tensor], Tensor] = gelu):
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)
        self.activation = activation

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class GatedFeedForward:
    """SwiGLU gated feed-forward network used in LLaMA."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        self.gate_proj = Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = Linear(hidden_dim, embed_dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))
