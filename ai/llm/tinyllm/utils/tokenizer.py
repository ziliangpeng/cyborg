"""Tokenizer abstraction for TinyLLM."""

import tiktoken


class Tokenizer:
    """Tokenizer for text encoding/decoding."""

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding to use
        """
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        return self.enc.decode(tokens)

    @classmethod
    def for_model(cls, model_name: str) -> "Tokenizer":
        """
        Factory method to get tokenizer for a model.

        Args:
            model_name: Name of the model

        Returns:
            Tokenizer configured for the model
        """
        # Map model names to encodings (extensible)
        encoding_map = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2",
            "gpt2-large": "gpt2",
            "gpt2-xl": "gpt2",
        }
        encoding = encoding_map.get(model_name, "gpt2")
        return cls(encoding)
