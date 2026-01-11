"""Embedding layers for TinyLLM."""

from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear


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


class OPTEmbedding:
    """OPT embeddings with position offset and optional word embedding projection."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        word_embed_proj_dim: int | None = None,
        position_offset: int = 2,
    ):
        """
        Initialize OPT embeddings.

        Args:
            vocab_size: Size of vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Model hidden dimension
            word_embed_proj_dim: Optional down-projection dimension for token embeddings
                                 (used in OPT-350m where it's 512)
            position_offset: Position offset (OPT uses 2)
        """
        self.position_offset = position_offset
        self.word_embed_proj_dim = word_embed_proj_dim

        # Token embeddings - embed to word_embed_proj_dim if specified, else embed_dim
        actual_embed_dim = word_embed_proj_dim if word_embed_proj_dim else embed_dim
        self.embed_tokens = Embedding(vocab_size, actual_embed_dim)

        # Positional embeddings (OPT's learned positions start at offset 2)
        self.embed_positions = Embedding(max_seq_len + position_offset, embed_dim)

        # Optional projection layers (for OPT-350m)
        self.project_in = Linear(actual_embed_dim, embed_dim) if word_embed_proj_dim else None
        self.project_out = Linear(embed_dim, actual_embed_dim) if word_embed_proj_dim else None

    def __call__(self, input_ids: Tensor) -> Tensor:
        """
        Compute combined token and positional embeddings.

        Args:
            input_ids: (batch_size, seq_len) token indices

        Returns:
            (batch_size, seq_len, embed_dim) embeddings
        """
        seq_len = input_ids.shape[1]

        # Position IDs with offset
        position_ids = Tensor.arange(seq_len).reshape(1, -1) + self.position_offset

        # Token embeddings
        token_embeds = self.embed_tokens(input_ids)

        # Project up if using word_embed_proj_dim
        if self.project_in is not None:
            token_embeds = self.project_in(token_embeds)

        # Position embeddings
        position_embeds = self.embed_positions(position_ids)

        return token_embeds + position_embeds
