"""Tokenizer abstraction for TinyLLM."""

import tiktoken


class Tokenizer:
    """Tokenizer for text encoding/decoding."""

    def __init__(self, encoding_name: str = "gpt2", tokenizer_type: str = "tiktoken"):
        """
        Initialize tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding or HuggingFace model ID
            tokenizer_type: Either "tiktoken" or "huggingface"
        """
        self._is_tiktoken = tokenizer_type == "tiktoken"

        if self._is_tiktoken:
            self.enc = tiktoken.get_encoding(encoding_name)
        else:
            # HuggingFace transformers tokenizer for OPT and other models
            from transformers import AutoTokenizer

            self.enc = AutoTokenizer.from_pretrained(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self._is_tiktoken:
            return self.enc.encode(text)
        return self.enc.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        if self._is_tiktoken:
            return self.enc.decode(tokens)
        return self.enc.decode(tokens, skip_special_tokens=True)

    @classmethod
    def for_model(cls, model_name: str) -> "Tokenizer":
        """
        Factory method to get tokenizer for a model.

        Args:
            model_name: Name of the model

        Returns:
            Tokenizer configured for the model
        """
        # GPT-2 models use tiktoken
        if model_name in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
            return cls("gpt2", "tiktoken")

        # OPT models use HuggingFace tokenizer
        if model_name.startswith("facebook/opt-"):
            return cls(model_name, "huggingface")

        # Default to tiktoken gpt2
        return cls("gpt2", "tiktoken")
