"""Monkey-patch tinygrad's CPU (clang JIT) compiler to inline libm symbols.

Tinygrad 0.11.0 compiles kernels with ``-nostdlib -ffreestanding`` (see
``tinygrad.runtime.ops_cpu.ClangJITCompiler.compile``). When clang lowers
``(a > b) ? a : b`` to a call to ``fmaxf``/``fminf``/``fmaxd``/``fmind``
the resulting ELF references libm symbols that are not linked, and the
ELF loader raises:

    RuntimeError: Attempting to relocate against an undefined symbol 'fmaxf'

This bites OPT (and any model whose softmax reduction emits fmax/fmin)
even though GPT-2 happens to compile clean.

We can't change the compile flags without forking tinygrad, but we can
prepend a small C preamble to the source that defines fmaxf/fminf/etc.
as inline expressions. clang then has no reason to emit a libm call.

Importing this module is enough; it patches once, idempotently.
"""

from __future__ import annotations

import importlib

_PATCHED = "_cyborg_libm_shim_applied"

# Inline replacements for libm builtins clang may emit under -O2 freestanding.
# Cover both float (fmaxf/fminf) and double (fmax/fmin) variants.
_PREAMBLE = """
#ifndef CYBORG_LIBM_SHIM
#define CYBORG_LIBM_SHIM
static inline float  fmaxf(float a, float b)   { return a > b ? a : b; }
static inline float  fminf(float a, float b)   { return a < b ? a : b; }
static inline double fmax (double a, double b) { return a > b ? a : b; }
static inline double fmin (double a, double b) { return a < b ? a : b; }
#endif
"""


def apply_libm_shim() -> None:
    """Wrap ClangJITCompiler.compile so the libm preamble is prepended once."""
    try:
        ops_cpu = importlib.import_module("tinygrad.runtime.ops_cpu")
    except ImportError:
        return

    cls = getattr(ops_cpu, "ClangJITCompiler", None)
    if cls is None or getattr(cls, _PATCHED, False):
        return

    original_compile = cls.compile

    def compile_with_shim(self, src: str):
        # Don't re-inject the preamble if something already added it.
        if "CYBORG_LIBM_SHIM" in src:
            return original_compile(self, src)
        return original_compile(self, _PREAMBLE + src)

    cls.compile = compile_with_shim
    setattr(cls, _PATCHED, True)


apply_libm_shim()
