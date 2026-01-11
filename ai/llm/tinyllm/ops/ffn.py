"""Feed-forward network for TinyLLM."""

from tinygrad import Tensor
from tinygrad.nn import Linear

from .activations import gelu


class FeedForward:
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        """
        Initialize feed-forward network.

        Args:
            embed_dim: Embedding dimension (input/output)
            hidden_dim: Hidden dimension (typically 4 * embed_dim)
        """
        self.c_fc = Linear(embed_dim, hidden_dim)  # up projection
        self.c_proj = Linear(hidden_dim, embed_dim)  # down projection

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            (batch_size, seq_len, embed_dim)
        """
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        return x
