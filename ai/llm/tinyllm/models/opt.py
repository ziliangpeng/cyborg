"""OPT model implementation for TinyLLM."""

from dataclasses import dataclass

from tinygrad import Tensor
from tinygrad.nn import LayerNorm, Linear

from ..ops.activations import relu
from ..ops.attention import CausalAttention
from ..ops.embeddings import OPTEmbedding
from ..ops.ffn import FeedForward
from .base import BaseModel


@dataclass
class OPTConfig:
    """Configuration for OPT model variants."""

    vocab_size: int = 50272
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 3072
    layer_norm_epsilon: float = 1e-5
    word_embed_proj_dim: int | None = None  # For OPT-350m (512)
    position_offset: int = 2  # OPT positions start at 2

    @classmethod
    def opt_125m(cls) -> "OPTConfig":
        return cls()

    @classmethod
    def opt_350m(cls) -> "OPTConfig":
        return cls(n_embd=1024, n_layer=24, n_head=16, n_inner=4096, word_embed_proj_dim=512)

    @classmethod
    def opt_1_3b(cls) -> "OPTConfig":
        return cls(n_embd=2048, n_layer=24, n_head=32, n_inner=8192)

    @classmethod
    def opt_2_7b(cls) -> "OPTConfig":
        return cls(n_embd=2560, n_layer=32, n_head=32, n_inner=10240)

    @classmethod
    def opt_6_7b(cls) -> "OPTConfig":
        return cls(n_embd=4096, n_layer=32, n_head=32, n_inner=16384)


class OPTTransformerBlock:
    """Single OPT transformer block with pre-LayerNorm."""

    def __init__(self, config: OPTConfig):
        self.self_attn_layer_norm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalAttention(config.n_embd, config.n_head)
        self.final_layer_norm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config.n_embd, config.n_inner, activation=relu)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


class OPT(BaseModel):
    """OPT Language Model."""

    def __init__(self, config: OPTConfig):
        self.config = config
        self.embeddings = OPTEmbedding(
            config.vocab_size,
            config.n_positions,
            config.n_embd,
            word_embed_proj_dim=config.word_embed_proj_dim,
            position_offset=config.position_offset,
        )
        self.blocks = [OPTTransformerBlock(config) for _ in range(config.n_layer)]
        self.final_layer_norm = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        output_dim = config.word_embed_proj_dim if config.word_embed_proj_dim else config.n_embd
        self.lm_head = Linear(output_dim, config.vocab_size)

    def __call__(self, input_ids: Tensor) -> Tensor:
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        if self.embeddings.project_out is not None:
            x = self.embeddings.project_out(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/opt-125m") -> "OPT":
        """Load pre-trained OPT model from HuggingFace."""
        from ..utils import load_weights

        config_map = {
            "facebook/opt-125m": OPTConfig.opt_125m,
            "facebook/opt-350m": OPTConfig.opt_350m,
            "facebook/opt-1.3b": OPTConfig.opt_1_3b,
            "facebook/opt-2.7b": OPTConfig.opt_2_7b,
            "facebook/opt-6.7b": OPTConfig.opt_6_7b,
        }
        config = config_map.get(model_name, OPTConfig.opt_125m)()
        model = cls(config)
        _load_opt_weights(model, load_weights(model_name))
        return model


def _load_opt_weights(model: OPT, weights: dict[str, Tensor]) -> None:
    """
    Load HuggingFace OPT weights into model.

    OPT uses standard Linear format (out_features, in_features) — no transposition needed.
    Different checkpoints use either "model.decoder.*" or bare "decoder.*" key prefixes.
    """
    # Detect key prefix: some checkpoints use "model.decoder.*", others bare "decoder.*"
    pfx = "model.decoder" if "model.decoder.embed_tokens.weight" in weights else "decoder"
    model.embeddings.embed_tokens.weight = weights[f"{pfx}.embed_tokens.weight"]
    model.embeddings.embed_positions.weight = weights[f"{pfx}.embed_positions.weight"]

    if model.embeddings.project_in is not None:
        model.embeddings.project_in.weight = weights[f"{pfx}.project_in.weight"]
    if model.embeddings.project_out is not None:
        model.embeddings.project_out.weight = weights[f"{pfx}.project_out.weight"]

    for i, block in enumerate(model.blocks):
        prefix = f"{pfx}.layers.{i}."

        block.self_attn_layer_norm.weight = weights[f"{prefix}self_attn_layer_norm.weight"]
        block.self_attn_layer_norm.bias = weights[f"{prefix}self_attn_layer_norm.bias"]

        block.attn.q_proj.weight = weights[f"{prefix}self_attn.q_proj.weight"]
        block.attn.q_proj.bias = weights[f"{prefix}self_attn.q_proj.bias"]
        block.attn.k_proj.weight = weights[f"{prefix}self_attn.k_proj.weight"]
        block.attn.k_proj.bias = weights[f"{prefix}self_attn.k_proj.bias"]
        block.attn.v_proj.weight = weights[f"{prefix}self_attn.v_proj.weight"]
        block.attn.v_proj.bias = weights[f"{prefix}self_attn.v_proj.bias"]
        block.attn.out_proj.weight = weights[f"{prefix}self_attn.out_proj.weight"]
        block.attn.out_proj.bias = weights[f"{prefix}self_attn.out_proj.bias"]

        block.final_layer_norm.weight = weights[f"{prefix}final_layer_norm.weight"]
        block.final_layer_norm.bias = weights[f"{prefix}final_layer_norm.bias"]

        block.mlp.fc1.weight = weights[f"{prefix}fc1.weight"]
        block.mlp.fc1.bias = weights[f"{prefix}fc1.bias"]
        block.mlp.fc2.weight = weights[f"{prefix}fc2.weight"]
        block.mlp.fc2.bias = weights[f"{prefix}fc2.bias"]

    model.final_layer_norm.weight = weights[f"{pfx}.final_layer_norm.weight"]
    model.final_layer_norm.bias = weights[f"{pfx}.final_layer_norm.bias"]

    model.lm_head.weight = weights["lm_head.weight"]
    if "lm_head.bias" in weights:
        model.lm_head.bias = weights["lm_head.bias"]
