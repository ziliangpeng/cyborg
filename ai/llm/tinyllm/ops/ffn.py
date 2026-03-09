"""Feed-forward network for TinyLLM."""

from collections.abc import Callable

from tinygrad import Tensor
from tinygrad.nn import Linear

from .activations import gelu


class FeedForward:
    """Position-wise feed-forward network with configurable activation."""

    def __init__(self, embed_dim: int, hidden_dim: int, activation: Callable[[Tensor], Tensor] = gelu):
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)
        self.activation = activation

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(x)))
