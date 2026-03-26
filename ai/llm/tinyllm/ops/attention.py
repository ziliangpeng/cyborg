"""Causal multi-head self-attention for TinyLLM."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tinygrad import Tensor
from tinygrad.nn import Linear

if TYPE_CHECKING:
    from ..kv_cache import KVCache


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

    def __call__(
        self,
        x: Tensor,
        layer_idx: int | None = None,
        kv_cache: KVCache | None = None,
        start_pos=None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=start_pos)

        # Update KV cache and get full K, V (cached history + new)
        attn_bias: Tensor | None = None
        if kv_cache is not None and layer_idx is not None:
            k, v = kv_cache.update(layer_idx, k, v, start_pos=start_pos)
            attn_bias = kv_cache.attention_bias()

        total_seq_len = k.shape[2]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # (batch, heads, seq_len, total_seq_len)

        # Apply causal mask only during prefill (seq_len > 1).
        # During decode (seq_len == 1) the single query is the last token
        # and should attend to all cached positions — no causal masking needed.
        if seq_len > 1:
            causal_mask = Tensor.ones(seq_len, total_seq_len).triu(1) * -1e9
            attn_weights = attn_weights + causal_mask

        # Apply KV cache validity mask (PreallocKVCache: masks unwritten positions)
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.reshape(1, 1, 1, -1)

        attn_weights = attn_weights.softmax(axis=-1)
        attn_output = attn_weights @ v

        # Reshape back and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
