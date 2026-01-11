"""GPT-2 model implementation for TinyLLM."""

from tinygrad import Tensor
from tinygrad.nn import LayerNorm

from ..ops.attention import CausalSelfAttention
from ..ops.embeddings import TokenPositionEmbedding
from ..ops.ffn import FeedForward
from .base import BaseModel
from .config import GPT2Config


class TransformerBlock:
    """Single transformer block with pre-LayerNorm."""

    def __init__(self, config: GPT2Config):
        """
        Initialize transformer block.

        Args:
            config: Model configuration
        """
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config.n_embd, config.n_inner)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply transformer block.

        Args:
            x: (batch_size, seq_len, n_embd)

        Returns:
            (batch_size, seq_len, n_embd)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(BaseModel):
    """GPT-2 Language Model."""

    def __init__(self, config: GPT2Config):
        """
        Initialize GPT-2 model.

        Args:
            config: Model configuration
        """
        self.config = config

        # Embeddings
        self.embeddings = TokenPositionEmbedding(config.vocab_size, config.n_positions, config.n_embd)

        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]

        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass returning logits.

        Args:
            input_ids: (batch_size, seq_len) token indices

        Returns:
            (batch_size, seq_len, vocab_size) logits
        """
        # Embeddings
        x = self.embeddings(input_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # LM head (tied weights with token embeddings)
        logits = x @ self.embeddings.wte.weight.T

        return logits

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2") -> "GPT2":
        """
        Load pre-trained GPT-2 model from HuggingFace.

        Args:
            model_name: HuggingFace model ID (e.g., "gpt2", "gpt2-medium")

        Returns:
            GPT2 model with loaded weights
        """
        from ..utils import load_weights

        # Determine config based on model name
        config_map = {
            "gpt2": GPT2Config.gpt2_small,
            "gpt2-medium": GPT2Config.gpt2_medium,
            "gpt2-large": GPT2Config.gpt2_large,
            "gpt2-xl": GPT2Config.gpt2_xl,
        }
        config_fn = config_map.get(model_name, GPT2Config.gpt2_small)
        config = config_fn()

        # Create model
        model = cls(config)

        # Load weights
        weights = load_weights(model_name)

        # Map weights to model (with transposition for Conv1D)
        _load_gpt2_weights(model, weights)

        return model


def _load_gpt2_weights(model: GPT2, weights: dict[str, Tensor]) -> None:
    """
    Load HuggingFace GPT-2 weights into model.

    Handles the Conv1D -> Linear weight transposition.
    GPT-2's Conv1D stores weights as (in_features, out_features),
    opposite of tinygrad Linear which expects (out_features, in_features).

    Args:
        model: GPT2 model instance
        weights: Dictionary of weight tensors from HuggingFace
    """
    # Embeddings (no transposition needed)
    model.embeddings.wte.weight = weights["wte.weight"]
    model.embeddings.wpe.weight = weights["wpe.weight"]

    # Transformer blocks
    for i, block in enumerate(model.blocks):
        prefix = f"h.{i}."

        # Layer norms (no transposition needed)
        block.ln_1.weight = weights[f"{prefix}ln_1.weight"]
        block.ln_1.bias = weights[f"{prefix}ln_1.bias"]
        block.ln_2.weight = weights[f"{prefix}ln_2.weight"]
        block.ln_2.bias = weights[f"{prefix}ln_2.bias"]

        # Attention (transpose weights!)
        block.attn.c_attn.weight = weights[f"{prefix}attn.c_attn.weight"].T
        block.attn.c_attn.bias = weights[f"{prefix}attn.c_attn.bias"]
        block.attn.c_proj.weight = weights[f"{prefix}attn.c_proj.weight"].T
        block.attn.c_proj.bias = weights[f"{prefix}attn.c_proj.bias"]

        # MLP (transpose weights!)
        block.mlp.c_fc.weight = weights[f"{prefix}mlp.c_fc.weight"].T
        block.mlp.c_fc.bias = weights[f"{prefix}mlp.c_fc.bias"]
        block.mlp.c_proj.weight = weights[f"{prefix}mlp.c_proj.weight"].T
        block.mlp.c_proj.bias = weights[f"{prefix}mlp.c_proj.bias"]

    # Final layer norm
    model.ln_f.weight = weights["ln_f.weight"]
    model.ln_f.bias = weights["ln_f.bias"]
