"""TinyLLM: A minimal LLM library built on TinyGrad.

A minimal, educational LLM library using TinyGrad with zero PyTorch dependencies.
Provides clean implementations of LLM building blocks for multiple model families.
"""

# Apply tinygrad CPU compiler libm shim before any kernels compile.
# See _tinygrad_cpu_fix.py for the rationale; this fixes the
# RuntimeError: Attempting to relocate against an undefined symbol fmaxf
# that hits OPT in CI.
from . import _tinygrad_cpu_fix  # noqa: F401  # side-effect import
from .kv_cache import KVCache, SimpleKVCache
from .models import GPT2, OPT, BaseModel, GPT2Config, LLaMA, LlamaConfig, OPTConfig, generate
from .utils import load_weights

__all__ = [
    "BaseModel",
    "load_weights",
    "GPT2",
    "GPT2Config",
    "OPT",
    "OPTConfig",
    "LLaMA",
    "LlamaConfig",
    "generate",
    "KVCache",
    "SimpleKVCache",
]

__version__ = "0.4.0"
