"""GPT-2 model implementation for TinyLLM."""

from tinygrad import Tensor, TinyJit
from tinygrad.nn import LayerNorm
from tinygrad.uop.ops import UOp

from ..kv_cache import KVCache
from ..ops.activations import gelu
from ..ops.attention import CausalAttention
from ..ops.embeddings import TokenPositionEmbedding
from ..ops.ffn import FeedForward
from .base import BaseModel
from .config import GPT2Config


class TransformerBlock:
    """Single transformer block with pre-LayerNorm."""

    def __init__(self, config: GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalAttention(config.n_embd, config.n_head)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config.n_embd, config.n_inner, activation=gelu)

    def __call__(
        self, x: Tensor, layer_idx: int | None = None, kv_cache: KVCache | None = None, start_pos=None
    ) -> Tensor:
        x = x + self.attn(self.ln_1(x), layer_idx=layer_idx, kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(BaseModel):
    """GPT-2 Language Model."""

    def __init__(self, config: GPT2Config):
        self.config = config
        self.embeddings = TokenPositionEmbedding(config.vocab_size, config.n_positions, config.n_embd)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Active KV cache set by generate.py before JIT decode loop
        self._active_cache: KVCache | None = None
        # JIT-compiled decode step (token, start_pos) -> logits
        self._jit = TinyJit(self._jit_forward)

    def _jit_forward(self, input_ids: Tensor, start_pos: UOp) -> Tensor:
        """JIT-compiled single-token forward pass using self._active_cache."""
        return self._forward(input_ids, kv_cache=self._active_cache, start_pos=start_pos)

    def _forward(self, input_ids: Tensor, kv_cache: KVCache | None = None, start_pos=None) -> Tensor:
        """Core forward pass. start_pos can be int or bound UOp."""
        # Derive embedding start position:
        # - for VariableKVCache with bound UOp: use the UOp directly
        # - for other caches: use seq_len() if available, else 0
        if start_pos is None:
            start_pos = kv_cache.seq_len() if kv_cache is not None else 0
        x = self.embeddings(input_ids, start_pos=start_pos)
        for i, block in enumerate(self.blocks):
            x = block(x, layer_idx=i, kv_cache=kv_cache, start_pos=start_pos)
        x = self.ln_f(x)
        # LM head with tied weights
        return x @ self.embeddings.wte.weight.T

    def __call__(self, input_ids: Tensor, kv_cache: KVCache | None = None, start_pos=None) -> Tensor:
        # Use JIT only for single-token decode with a bound UOp start_pos
        if (
            kv_cache is not None
            and input_ids.shape[1] == 1
            and isinstance(start_pos, UOp)
            and self._active_cache is kv_cache
        ):
            return self._jit(input_ids, start_pos)
        return self._forward(input_ids, kv_cache=kv_cache, start_pos=start_pos)

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2") -> "GPT2":
        """Load pre-trained GPT-2 model from HuggingFace."""
        from ..utils import load_weights

        config_map = {
            "gpt2": GPT2Config.gpt2_small,
            "gpt2-medium": GPT2Config.gpt2_medium,
            "gpt2-large": GPT2Config.gpt2_large,
            "gpt2-xl": GPT2Config.gpt2_xl,
        }
        config = config_map.get(model_name, GPT2Config.gpt2_small)()
        model = cls(config)
        _load_gpt2_weights(model, load_weights(model_name))
        return model


def _load_gpt2_weights(model: GPT2, weights: dict[str, Tensor]) -> None:
    """
    Load HuggingFace GPT-2 weights into model.

    GPT-2's HF checkpoint uses Conv1D (stored as (in, out)) for QKV and FFN,
    so weights are transposed. The combined QKV weight is split into separate
    q_proj, k_proj, v_proj to match the unified CausalAttention interface.
    """
    model.embeddings.wte.weight = weights["wte.weight"]
    model.embeddings.wpe.weight = weights["wpe.weight"]

    for i, block in enumerate(model.blocks):
        prefix = f"h.{i}."

        block.ln_1.weight = weights[f"{prefix}ln_1.weight"]
        block.ln_1.bias = weights[f"{prefix}ln_1.bias"]
        block.ln_2.weight = weights[f"{prefix}ln_2.weight"]
        block.ln_2.bias = weights[f"{prefix}ln_2.bias"]

        # Split combined QKV weight (3*n_embd, n_embd) into separate projections
        qkv_weight = weights[f"{prefix}attn.c_attn.weight"].T  # (3*n_embd, n_embd)
        q_w, k_w, v_w = qkv_weight.chunk(3, dim=0)
        block.attn.q_proj.weight = q_w
        block.attn.k_proj.weight = k_w
        block.attn.v_proj.weight = v_w

        qkv_bias = weights[f"{prefix}attn.c_attn.bias"]  # (3*n_embd,)
        q_b, k_b, v_b = qkv_bias.chunk(3, dim=0)
        block.attn.q_proj.bias = q_b
        block.attn.k_proj.bias = k_b
        block.attn.v_proj.bias = v_b

        block.attn.out_proj.weight = weights[f"{prefix}attn.c_proj.weight"].T
        block.attn.out_proj.bias = weights[f"{prefix}attn.c_proj.bias"]

        block.mlp.fc1.weight = weights[f"{prefix}mlp.c_fc.weight"].T
        block.mlp.fc1.bias = weights[f"{prefix}mlp.c_fc.bias"]
        block.mlp.fc2.weight = weights[f"{prefix}mlp.c_proj.weight"].T
        block.mlp.fc2.bias = weights[f"{prefix}mlp.c_proj.bias"]

    model.ln_f.weight = weights["ln_f.weight"]
    model.ln_f.bias = weights["ln_f.bias"]
