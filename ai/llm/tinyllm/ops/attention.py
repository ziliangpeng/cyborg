"""Multi-head self-attention for TinyLLM."""

import math

from tinygrad import Tensor
from tinygrad.nn import Linear


class CausalSelfAttention:
    """Multi-head self-attention with causal masking."""

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

        # Combined QKV projection
        self.c_attn = Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = Linear(embed_dim, embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to QKV
        qkv = self.c_attn(x)  # (batch, seq, 3 * embed_dim)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (batch, num_heads, seq, seq)

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
        return self.c_proj(attn_output)
