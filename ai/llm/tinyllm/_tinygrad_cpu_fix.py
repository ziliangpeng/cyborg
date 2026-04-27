"""Monkey-patch tinygrad's CPU (clang JIT) compiler to provide fmaxf/fminf.

Tinygrad 0.11.0 compiles kernels with ``-O2 -nostdlib -ffreestanding``.
clang's vectorizer pass recognises the ternary ``(a > b ? a : b)`` pattern
emitted by tinygrad's renderer and lowers it to a ``fmaxf``/``fminf``/
``fmax``/``fmin`` libm intrinsic. With ``-nostdlib`` libm is not linked,
so the resulting ELF contains undefined ``fmaxf`` symbols and tinygrad's
ELF loader raises:

    RuntimeError: Attempting to relocate against an undefined symbol 'fmaxf'

This bites OPT (and any model whose softmax/topk emits the pattern),
even though GPT-2 happens not to vectorize through that path.

Neither preprocessor ``#define`` nor ``-fno-builtin`` prevent this — the
substitution is performed by clang's middle-end on the IR, after macro
expansion and after frontend builtin recognition.

Fix: append concrete ``fmaxf``/``fminf``/``fmax``/``fmin`` function
definitions to the source we feed to clang. The vectorizer is then
free to lower max/min to libm calls — those calls resolve to our own
inline-friendly implementations within the same translation unit, the
ELF is self-contained, and the loader is happy.

Importing this module patches ``ClangJITCompiler.compile`` once,
idempotently. It is imported as a side effect from
``ai/llm/tinyllm/__init__.py``.
"""

from __future__ import annotations

import importlib

_PATCHED = "_cyborg_libm_shim_applied"

# Concrete definitions for the libm symbols clang's vectorizer may emit.
# We mark them ``static`` to avoid clashes when more than one kernel TU
# is compiled, and use ``__attribute__((used))`` so the optimizer keeps
# them around even when not directly referenced in the source.
_LIBM_IMPL = """
#ifndef CYBORG_LIBM_IMPL
#define CYBORG_LIBM_IMPL
static float  __attribute__((used)) fmaxf(float a, float b)   { return a > b ? a : b; }
static float  __attribute__((used)) fminf(float a, float b)   { return a < b ? a : b; }
static double __attribute__((used)) fmax (double a, double b) { return a > b ? a : b; }
static double __attribute__((used)) fmin (double a, double b) { return a < b ? a : b; }
#endif
"""


def apply_libm_shim() -> None:
    """Wrap ClangJITCompiler.compile so the libm impls are appended once."""
    try:
        ops_cpu = importlib.import_module("tinygrad.runtime.ops_cpu")
    except ImportError:
        return

    cls = getattr(ops_cpu, "ClangJITCompiler", None)
    if cls is None or getattr(cls, _PATCHED, False):
        return

    original_compile = cls.compile

    def compile_with_shim(self, src: str):
        # Don't re-inject if something already added them.
        if "CYBORG_LIBM_IMPL" in src:
            return original_compile(self, src)
        # Append at the END so any earlier extern declarations still parse,
        # and so our static defs win when the linker resolves the call.
        return original_compile(self, src + _LIBM_IMPL)

    cls.compile = compile_with_shim
    setattr(cls, _PATCHED, True)


apply_libm_shim()
