"""OPT model implementation for TinyLLM."""

from dataclasses import dataclass

from tinygrad import Tensor
from tinygrad.nn import LayerNorm, Linear

from ..ops.activations import relu
from ..ops.attention_opt import OPTAttention
from ..ops.embeddings import OPTEmbedding


@dataclass
class OPTConfig:
    """Configuration for OPT model variants."""

    vocab_size: int = 50272
    n_positions: int = 2048  # max sequence length
    n_embd: int = 768  # hidden_size
    n_layer: int = 12  # num_hidden_layers
    n_head: int = 12  # num_attention_heads
    n_inner: int = 3072  # ffn_dim (typically 4 * n_embd)
    layer_norm_epsilon: float = 1e-5
    word_embed_proj_dim: int | None = None  # For OPT-350m (512)
    position_offset: int = 2  # OPT positions start at 2

    @classmethod
    def opt_125m(cls) -> "OPTConfig":
        """OPT-125M (125M parameters)."""
        return cls()

    @classmethod
    def opt_350m(cls) -> "OPTConfig":
        """OPT-350M (355M parameters)."""
        return cls(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            word_embed_proj_dim=512,
        )

    @classmethod
    def opt_1_3b(cls) -> "OPTConfig":
        """OPT-1.3B (1.3B parameters)."""
        return cls(
            n_embd=2048,
            n_layer=24,
            n_head=32,
            n_inner=8192,
        )

    @classmethod
    def opt_2_7b(cls) -> "OPTConfig":
        """OPT-2.7B (2.7B parameters)."""
        return cls(
            n_embd=2560,
            n_layer=32,
            n_head=32,
            n_inner=10240,
        )

    @classmethod
    def opt_6_7b(cls) -> "OPTConfig":
        """OPT-6.7B (6.7B parameters)."""
        return cls(
            n_embd=4096,
            n_layer=32,
            n_head=32,
            n_inner=16384,
        )


class OPTFeedForward:
    """Position-wise feed-forward network with ReLU activation (OPT style)."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        """
        Initialize feed-forward network.

        Args:
            embed_dim: Embedding dimension (input/output)
            hidden_dim: Hidden dimension (typically 4 * embed_dim)
        """
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward transformation.

        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return x


class OPTTransformerBlock:
    """Single OPT transformer block with pre-LayerNorm."""

    def __init__(self, config: OPTConfig):
        """
        Initialize transformer block.

        Args:
            config: Model configuration
        """
        # Attention layer norm (before attention)
        self.self_attn_layer_norm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = OPTAttention(config.n_embd, config.n_head)

        # FFN layer norm (before FFN)
        self.final_layer_norm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = OPTFeedForward(config.n_embd, config.n_inner)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply transformer block.

        Args:
            x: (batch_size, seq_len, n_embd)

        Returns:
            (batch_size, seq_len, n_embd)
        """
        # Pre-LayerNorm style (like GPT-2)
        x = x + self.attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


class OPT:
    """OPT Language Model."""

    def __init__(self, config: OPTConfig):
        """
        Initialize OPT model.

        Args:
            config: Model configuration
        """
        self.config = config

        # Embeddings with OPT-specific handling
        self.embeddings = OPTEmbedding(
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            word_embed_proj_dim=config.word_embed_proj_dim,
            position_offset=config.position_offset,
        )

        # Transformer blocks
        self.blocks = [OPTTransformerBlock(config) for _ in range(config.n_layer)]

        # Final layer norm
        self.final_layer_norm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Separate LM head (OPT doesn't tie weights)
        # Output dimension depends on word_embed_proj_dim
        output_dim = config.word_embed_proj_dim if config.word_embed_proj_dim else config.n_embd
        self.lm_head = Linear(output_dim, config.vocab_size)

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
        x = self.final_layer_norm(x)

        # Project out if using word_embed_proj_dim
        if self.embeddings.project_out is not None:
            x = self.embeddings.project_out(x)

        # LM head
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/opt-125m") -> "OPT":
        """
        Load pre-trained OPT model from HuggingFace.

        Args:
            model_name: HuggingFace model ID (e.g., "facebook/opt-125m")

        Returns:
            OPT model with loaded weights
        """
        from ..utils import load_weights

        # Determine config based on model name
        config_map = {
            "facebook/opt-125m": OPTConfig.opt_125m,
            "facebook/opt-350m": OPTConfig.opt_350m,
            "facebook/opt-1.3b": OPTConfig.opt_1_3b,
            "facebook/opt-2.7b": OPTConfig.opt_2_7b,
            "facebook/opt-6.7b": OPTConfig.opt_6_7b,
        }
        config_fn = config_map.get(model_name, OPTConfig.opt_125m)
        config = config_fn()

        # Create model
        model = cls(config)

        # Load weights
        weights = load_weights(model_name)

        # Map weights to model
        _load_opt_weights(model, weights)

        return model


def _load_opt_weights(model: OPT, weights: dict[str, Tensor]) -> None:
    """
    Load HuggingFace OPT weights into model.

    OPT uses standard Linear format (out_features, in_features),
    same as tinygrad - NO transposition needed.

    Args:
        model: OPT model instance
        weights: Dictionary of weight tensors from HuggingFace
    """
    # Embeddings
    model.embeddings.embed_tokens.weight = weights["model.decoder.embed_tokens.weight"]
    model.embeddings.embed_positions.weight = weights["model.decoder.embed_positions.weight"]

    # Optional projection layers (for OPT-350m)
    if model.embeddings.project_in is not None:
        model.embeddings.project_in.weight = weights["model.decoder.project_in.weight"]
    if model.embeddings.project_out is not None:
        model.embeddings.project_out.weight = weights["model.decoder.project_out.weight"]

    # Transformer blocks
    for i, block in enumerate(model.blocks):
        prefix = f"model.decoder.layers.{i}."

        # Attention layer norm
        block.self_attn_layer_norm.weight = weights[f"{prefix}self_attn_layer_norm.weight"]
        block.self_attn_layer_norm.bias = weights[f"{prefix}self_attn_layer_norm.bias"]

        # Attention projections (no transposition needed for OPT)
        block.attn.q_proj.weight = weights[f"{prefix}self_attn.q_proj.weight"]
        block.attn.q_proj.bias = weights[f"{prefix}self_attn.q_proj.bias"]
        block.attn.k_proj.weight = weights[f"{prefix}self_attn.k_proj.weight"]
        block.attn.k_proj.bias = weights[f"{prefix}self_attn.k_proj.bias"]
        block.attn.v_proj.weight = weights[f"{prefix}self_attn.v_proj.weight"]
        block.attn.v_proj.bias = weights[f"{prefix}self_attn.v_proj.bias"]
        block.attn.out_proj.weight = weights[f"{prefix}self_attn.out_proj.weight"]
        block.attn.out_proj.bias = weights[f"{prefix}self_attn.out_proj.bias"]

        # FFN layer norm
        block.final_layer_norm.weight = weights[f"{prefix}final_layer_norm.weight"]
        block.final_layer_norm.bias = weights[f"{prefix}final_layer_norm.bias"]

        # FFN (no transposition needed for OPT)
        block.mlp.fc1.weight = weights[f"{prefix}fc1.weight"]
        block.mlp.fc1.bias = weights[f"{prefix}fc1.bias"]
        block.mlp.fc2.weight = weights[f"{prefix}fc2.weight"]
        block.mlp.fc2.bias = weights[f"{prefix}fc2.bias"]

    # Final layer norm
    model.final_layer_norm.weight = weights["model.decoder.final_layer_norm.weight"]
    model.final_layer_norm.bias = weights["model.decoder.final_layer_norm.bias"]

    # LM head
    model.lm_head.weight = weights["lm_head.weight"]
    if "lm_head.bias" in weights:
        model.lm_head.bias = weights["lm_head.bias"]
