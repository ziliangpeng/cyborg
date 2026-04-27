"""KV Cache implementations for TinyLLM."""

import itertools
from abc import ABC, abstractmethod

from tinygrad import Tensor
from tinygrad.uop.ops import UOp

_instance_counter = itertools.count()


class KVCache(ABC):
    """Abstract base class for KV cache implementations."""

    @abstractmethod
    def update(self, layer_idx: int, k: Tensor, v: Tensor, start_pos=None) -> tuple[Tensor, Tensor]:
        """
        Update cache with new K, V tensors and return full K, V for attention.

        Args:
            layer_idx: Transformer layer index
            k: New key tensor (batch, heads, new_seq_len, head_dim)
            v: New value tensor (batch, heads, new_seq_len, head_dim)
            start_pos: Current write position (int or bound UOp). Used by
                       VariableKVCache; ignored by SimpleKVCache.

        Returns:
            Full (k, v) tensors including cached history
        """

    @abstractmethod
    def seq_len(self) -> int:
        """Return the number of tokens currently cached."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached KV tensors."""

    def attention_bias(self) -> Tensor | None:
        """
        Optional additive bias applied to attention logits before softmax.
        Shape: (max_seq_len,) — broadcast over batch, heads, and query positions.
        Returns None if no bias is needed (e.g. SimpleKVCache which returns cropped K/V).
        """
        return None

    def flush(self) -> None:  # noqa: B027  # default no-op; override in subclass
        """Realize all pending assigns. Default no-op; override in VariableKVCache."""
        pass


class SimpleKVCache(KVCache):
    """
    Straightforward KV cache that stores all past K, V tensors in memory.

    Stores one (K, V) pair per layer. On each update, appends new K, V to
    the cached history along the sequence dimension.

    Note: returns K/V with growing shapes (seq_len grows by 1 each decode step),
    which causes tinygrad to recompile a new attention kernel for each unique shape.
    For long sequences this limits performance — use VariableKVCache instead.
    """

    def __init__(self) -> None:
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def update(self, layer_idx: int, k: Tensor, v: Tensor, start_pos=None) -> tuple[Tensor, Tensor]:
        # start_pos is ignored for SimpleKVCache — we always cat
        if layer_idx in self._cache:
            cached_k, cached_v = self._cache[layer_idx]
            k = cached_k.cat(k, dim=2)  # concat along seq dim: (batch, heads, seq, head_dim)
            v = cached_v.cat(v, dim=2)
        self._cache[layer_idx] = (k, v)
        return k, v

    def realize(self) -> None:
        """Realize all cached tensors to flush lazy graph after each decode step."""
        self._cache = {idx: (k.realize(), v.realize()) for idx, (k, v) in self._cache.items()}

    def seq_len(self) -> int:
        if not self._cache:
            return 0
        first_k = next(iter(self._cache.values()))[0]
        return first_k.shape[2]

    def clear(self) -> None:
        self._cache.clear()


class VariableKVCache(KVCache):
    """
    KV cache using tinygrad's Variable API for dynamic position indexing.

    Pre-allocates a single (2, n_layers, batch, heads, max_seq_len, head_dim) buffer
    for all layers. Accepts start_pos (int or bound UOp) as an explicit argument to
    update(), so the caller (generate.py) controls Variable lifetime and JIT replay.

    During decode the caller passes a bound UOp (v.bind(concrete_pos)); the assign +
    slice kernels compile once in TinyJit and are replayed with different bound values
    each step — no per-step recompilation.

    This is the same pattern used in tinygrad's own LLM implementation
    (tinygrad/apps/llm.py). Requires TinyJit wrapping the forward pass when using
    Variable start_pos.

    Args:
        max_seq_len: Maximum sequence length
        n_layers: Number of transformer layers
        num_heads: Number of KV attention heads
        head_dim: Dimension per attention head
        batch: Batch size (default 1)
    """

    def __init__(self, max_seq_len: int, n_layers: int, num_heads: int, head_dim: int, batch: int = 1) -> None:
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self._pos = 0

        # Single pre-allocated buffer for all layers: (2, n_layers, batch, heads, max_seq_len, head_dim)
        # Index 0 = keys, index 1 = values
        self.cache = Tensor.zeros(2, n_layers, batch, num_heads, max_seq_len, head_dim).contiguous().realize()
        self._eager = True  # True during decode, False during prefill (lazy mode)

    def update(self, layer_idx: int, k: Tensor, v: Tensor, start_pos=None) -> tuple[Tensor, Tensor]:
        """
        Update cache at start_pos and return SYMBOLIC-sliced K/V [0:end_pos].

        Returns only valid positions so attention computes Q @ K^T over actual
        sequence length, not the full max_seq_len buffer. Compute scales with
        real sequence length — significant win for large models on short contexts.

        During decode with JIT, start_pos is a bound UOp so end_pos = sp + T is
        symbolic. The attention output shape (batch, heads, 1, head_dim) stays
        fixed regardless of K/V seq dim, so JIT compiles once and replays.

        During prefill, start_pos=0 (concrete int), so end_pos is concrete. No
        symbolic shapes involved — safe to call triu() in the causal mask.

        Args:
            layer_idx: Transformer layer index
            k, v: New key/value tensors (batch, heads, T, head_dim)
            start_pos: Write position — int during prefill, bound UOp during JIT decode.
                       If None, falls back to internal _pos counter.
        """
        T = k.shape[2]  # number of new tokens (1 during decode, >1 during prefill)
        is_decode = T == 1

        # During JIT decode, start_pos is a bound UOp; use it directly.
        # During prefill (or non-JIT eager), use internal _pos counter.
        if is_decode and isinstance(start_pos, UOp):
            sp = start_pos  # bound UOp — variable position for JIT replay
        elif start_pos is not None:
            sp = start_pos  # explicit int (e.g., 0 for prefill start)
        else:
            sp = self._pos  # fallback to internal counter

        # Write new K/V into the pre-allocated buffer at [sp : sp + T]
        # assign() is an in-place operation; realize() flushes the lazy graph immediately
        assigned = self.cache[:, layer_idx, :, :, sp : sp + T, :].assign(
            Tensor.stack(k[:, :, :T, :], v[:, :, :T, :])  # shape: (2, batch, heads, T, head_dim)
        )
        if self._eager:
            assigned.realize()

        # Advance internal write position only when sp is a concrete int.
        # When sp is a bound UOp (JIT decode), the caller (generate.py) manages position tracking.
        if not isinstance(sp, UOp):
            self._pos += T

        # Return SYMBOLIC slice [0:end_pos] -- only valid positions.
        # During decode: end_pos = sp+T is a symbolic UOp expression.
        #   K/V shape = (batch, heads, symbolic_end_pos, head_dim)
        #   Attention output = attn_weights @ V = (batch, heads, 1, head_dim) -- FIXED shape.
        #   JIT sees fixed output shape -> compiles once, replays with different symbolic K/V lengths.
        # During prefill: end_pos is a concrete int. No symbolic dims. Safe.
        end_pos = sp + T
        full_k = self.cache[0, layer_idx, :, :, 0:end_pos, :]
        full_v = self.cache[1, layer_idx, :, :, 0:end_pos, :]
        return full_k, full_v

    def flush(self) -> None:
        """Realize all pending lazy assigns. Call once after prefill to flush in one GPU sync."""
        self.cache.realize()
        self._eager = True  # restore eager mode for subsequent decode steps

    def attention_bias(self) -> Tensor | None:
        """No mask needed — update() only returns valid positions [0:end_pos]."""
        return None

    def seq_len(self) -> int:
        return self._pos

    def clear(self) -> None:
        self.cache.assign(Tensor.zeros_like(self.cache)).realize()
        self._pos = 0
