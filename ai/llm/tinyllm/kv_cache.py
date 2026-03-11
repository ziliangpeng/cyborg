"""KV Cache implementations for TinyLLM."""

from abc import ABC, abstractmethod

from tinygrad import Tensor


class KVCache(ABC):
    """Abstract base class for KV cache implementations."""

    @abstractmethod
    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Update cache with new K, V tensors and return full K, V for attention.

        Args:
            layer_idx: Transformer layer index
            k: New key tensor (batch, heads, new_seq_len, head_dim)
            v: New value tensor (batch, heads, new_seq_len, head_dim)

        Returns:
            Full (k, v) tensors including cached history
        """

    @abstractmethod
    def seq_len(self) -> int:
        """Return the number of tokens currently cached."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached KV tensors."""


class SimpleKVCache(KVCache):
    """
    Straightforward KV cache that stores all past K, V tensors in memory.

    Stores one (K, V) pair per layer. On each update, appends new K, V to
    the cached history along the sequence dimension.
    """

    def __init__(self) -> None:
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if layer_idx in self._cache:
            cached_k, cached_v = self._cache[layer_idx]
            k = cached_k.cat(k, dim=2)  # concat along seq dim: (batch, heads, seq, head_dim)
            v = cached_v.cat(v, dim=2)
        self._cache[layer_idx] = (k, v)
        return k, v

    def realize(self) -> None:
        """Realize all cached tensors to flush lazy graph after each decode step."""
        self._cache = {
            idx: (k.realize(), v.realize())
            for idx, (k, v) in self._cache.items()
        }

    def seq_len(self) -> int:
        if not self._cache:
            return 0
        first_k = next(iter(self._cache.values()))[0]
        return first_k.shape[2]

    def clear(self) -> None:
        self._cache.clear()
