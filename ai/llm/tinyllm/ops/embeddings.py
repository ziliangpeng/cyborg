"""Embedding layers for TinyLLM."""

from tinygrad import Tensor
from tinygrad.nn import Embedding


class TokenPositionEmbedding:
    """Combined token and absolute positional embeddings."""

    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int):
        """
        Initialize embeddings.

        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
        """
        self.wte = Embedding(vocab_size, embed_dim)  # token embeddings
        self.wpe = Embedding(max_seq_len, embed_dim)  # positional embeddings

    def __call__(self, input_ids: Tensor) -> Tensor:
        """
        Compute combined token and positional embeddings.

        Args:
            input_ids: (batch_size, seq_len) token indices

        Returns:
            (batch_size, seq_len, embed_dim) embeddings
        """
        seq_len = input_ids.shape[1]
        position_ids = Tensor.arange(seq_len).reshape(1, -1)

        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        return token_embeds + position_embeds
