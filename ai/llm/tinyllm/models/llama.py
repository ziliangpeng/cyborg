"""LLaMA 1 model implementation for TinyLLM."""

from dataclasses import dataclass

from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear

from ..ops.attention import CausalAttention
from ..ops.ffn import GatedFeedForward
from ..ops.rmsnorm import RMSNorm
from ..ops.rope import apply_rotary_emb, precompute_rope_freqs
from .base import BaseModel
from ..kv_cache import KVCache


@dataclass
class LlamaConfig:
    """Configuration for LLaMA model variants."""

    vocab_size: int = 32000
    n_positions: int = 2048
    n_embd: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_inner: int = 11008
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @classmethod
    def open_llama_3b(cls) -> "LlamaConfig":
        """Open LLaMA 3B (openlm-research/open_llama_3b)."""
        return cls(n_embd=3200, n_layer=26, n_head=32, n_inner=8640)

    @classmethod
    def open_llama_7b(cls) -> "LlamaConfig":
        """Open LLaMA 7B (openlm-research/open_llama_7b)."""
        return cls()

    @classmethod
    def open_llama_13b(cls) -> "LlamaConfig":
        """Open LLaMA 13B (openlm-research/open_llama_13b)."""
        return cls(n_embd=5120, n_layer=40, n_head=40, n_inner=13824)

    @classmethod
    def llama2_7b(cls) -> "LlamaConfig":
        """Meta LLaMA 2 7B (meta-llama/Llama-2-7b-hf)."""
        return cls(n_positions=4096)

    @classmethod
    def llama2_13b(cls) -> "LlamaConfig":
        """Meta LLaMA 2 13B (meta-llama/Llama-2-13b-hf)."""
        return cls(n_embd=5120, n_layer=40, n_head=40, n_inner=13824, n_positions=4096)


class _RopeCallable:
    """Wraps precomputed RoPE tables as a callable for CausalAttention."""

    def __init__(self, cos: Tensor, sin: Tensor):
        self.cos = cos
        self.sin = sin

    def __call__(self, q: Tensor, k: Tensor, start_pos: int = 0):
        seq_len = q.shape[2]
        cos = self.cos[start_pos:start_pos + seq_len]
        sin = self.sin[start_pos:start_pos + seq_len]
        return apply_rotary_emb(q, k, cos, sin)


class LlamaTransformerBlock:
    """Single LLaMA transformer block with RMSNorm and SwiGLU FFN."""

    def __init__(self, config: LlamaConfig, rope: _RopeCallable):
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalAttention(config.n_embd, config.n_head, rope=rope, bias=False)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = GatedFeedForward(config.n_embd, config.n_inner)

    def __call__(self, x: Tensor, layer_idx: int | None = None, kv_cache: KVCache | None = None, start_pos: int = 0) -> Tensor:
        x = x + self.attn(self.ln_1(x), layer_idx=layer_idx, kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class LLaMA(BaseModel):
    """LLaMA 1 Language Model."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.n_embd)

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(head_dim, config.n_positions, config.rope_theta)
        rope = _RopeCallable(cos, sin)

        self.blocks = [LlamaTransformerBlock(config, rope) for _ in range(config.n_layer)]
        self.ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, input_ids: Tensor, kv_cache: KVCache | None = None) -> Tensor:
        start_pos = kv_cache.seq_len() if kv_cache is not None else 0
        x = self.embed_tokens(input_ids)
        for i, block in enumerate(self.blocks):
            x = block(x, layer_idx=i, kv_cache=kv_cache, start_pos=start_pos)
        x = self.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_name: str = "openlm-research/open_llama_3b") -> "LLaMA":
        """Load pre-trained LLaMA model from HuggingFace."""
        from ..utils import load_weights

        config_map = {
            "openlm-research/open_llama_3b": LlamaConfig.open_llama_3b,
            "openlm-research/open_llama_7b": LlamaConfig.open_llama_7b,
            "openlm-research/open_llama_13b": LlamaConfig.open_llama_13b,
            "meta-llama/Llama-2-7b-hf": LlamaConfig.llama2_7b,
            "meta-llama/Llama-2-13b-hf": LlamaConfig.llama2_13b,
        }
        config_fn = config_map.get(model_name)
        if config_fn is None:
            raise ValueError(f"Unsupported LLaMA model: {model_name}. Supported: {list(config_map.keys())}")
        config = config_fn()
        model = cls(config)
        _load_llama_weights(model, load_weights(model_name))
        return model


def _load_llama_weights(model: LLaMA, weights: dict[str, Tensor]) -> None:
    """
    Load HuggingFace LLaMA weights into model.

    LLaMA uses standard Linear format — no transposition needed.
    No biases anywhere.
    """
    model.embed_tokens.weight = weights["model.embed_tokens.weight"]

    for i, block in enumerate(model.blocks):
        prefix = f"model.layers.{i}."

        block.ln_1.weight = weights[f"{prefix}input_layernorm.weight"]
        block.attn.q_proj.weight = weights[f"{prefix}self_attn.q_proj.weight"]
        block.attn.k_proj.weight = weights[f"{prefix}self_attn.k_proj.weight"]
        block.attn.v_proj.weight = weights[f"{prefix}self_attn.v_proj.weight"]
        block.attn.out_proj.weight = weights[f"{prefix}self_attn.o_proj.weight"]
        block.ln_2.weight = weights[f"{prefix}post_attention_layernorm.weight"]
        block.mlp.gate_proj.weight = weights[f"{prefix}mlp.gate_proj.weight"]
        block.mlp.up_proj.weight = weights[f"{prefix}mlp.up_proj.weight"]
        block.mlp.down_proj.weight = weights[f"{prefix}mlp.down_proj.weight"]

    model.ln_f.weight = weights["model.norm.weight"]
    model.lm_head.weight = weights["lm_head.weight"]
