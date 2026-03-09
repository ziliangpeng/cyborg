"""Causal multi-head self-attention for TinyLLM."""

import math

from tinygrad import Tensor
from tinygrad.nn import Linear


class CausalAttention:
    """Multi-head causal self-attention with separate Q, K, V projections."""

    def __init__(self, embed_dim: int, num_heads: int, rope=None, bias: bool = True):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.rope = rope

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def __call__(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        causal_mask = Tensor.ones(seq_len, seq_len).triu(1) * -1e9
        attn_weights = (attn_weights + causal_mask).softmax(axis=-1)
        attn_output = attn_weights @ v

        # Reshape back and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
