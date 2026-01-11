"""OPT-style multi-head self-attention for TinyLLM.

TODO: Find a way to combine this into the GPT-2 attention (attention.py) since
the math is identical - only the weight dict structure differs.
"""

import math

from tinygrad import Tensor
from tinygrad.nn import Linear


class OPTAttention:
    """Multi-head self-attention with separate Q, K, V projections (OPT style)."""

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initialize attention layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
        """
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        # Separate Q, K, V projections (OPT style)
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V separately
        q = self.q_proj(x)  # (batch, seq, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scaling  # (batch, num_heads, seq, seq)

        # Causal mask: prevent attending to future tokens
        # Use large negative value instead of -inf to avoid NaN in softmax
        causal_mask = Tensor.ones(seq_len, seq_len).triu(1) * -1e9
        attn_weights = attn_weights + causal_mask

        # Softmax and attend
        attn_weights = attn_weights.softmax(axis=-1)
        attn_output = attn_weights @ v  # (batch, num_heads, seq, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        return self.out_proj(attn_output)
