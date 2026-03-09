"""Rotary Position Embeddings (RoPE) for TinyLLM."""

from tinygrad import Tensor


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute cos/sin tables for RoPE.

    Args:
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length.
        theta: Base frequency (default 10000.0).

    Returns:
        Tuple of (cos, sin) tensors, each shape (max_seq_len, head_dim // 2).
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (Tensor.arange(0, half_dim).float() / half_dim))
    positions = Tensor.arange(0, max_seq_len).float()
    # Outer product: (max_seq_len, half_dim)
    angles = positions.reshape(max_seq_len, 1) * freqs.reshape(1, half_dim)
    return angles.cos(), angles.sin()


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor, shape (batch, heads, seq_len, head_dim).
        k: Key tensor, shape (batch, heads, seq_len, head_dim).
        cos: Cosine table, shape (max_seq_len, head_dim // 2).
        sin: Sine table, shape (max_seq_len, head_dim // 2).

    Returns:
        Tuple of rotated (q, k).
    """
    seq_len = q.shape[2]
    cos = cos[:seq_len]  # (seq_len, half_dim)
    sin = sin[:seq_len]

    # Reshape for broadcasting: (1, 1, seq_len, half_dim)
    cos = cos.reshape(1, 1, seq_len, cos.shape[-1])
    sin = sin.reshape(1, 1, seq_len, sin.shape[-1])

    def rotate(x: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        # Rotation: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        return Tensor.cat(x1 * cos - x2 * sin, x2 * cos + x1 * sin, dim=-1)

    return rotate(q), rotate(k)
