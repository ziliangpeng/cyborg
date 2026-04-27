"""Monkey-patch tinygrad's CPU JIT loader to support fmaxf/fminf calls.

Tinygrad 0.11.0 compiles kernels with ``-O2 -nostdlib -ffreestanding``.
clang's vectorizer lowers ternary ``(a > b ? a : b)`` patterns into
``fmaxf``/``fminf``/``fmax``/``fmin`` libm calls. Two problems follow:

1. With ``-nostdlib`` libm is not linked and ``fmaxf`` resolves to an
   undefined symbol — tinygrad's ELF loader raises:
       RuntimeError: Attempting to relocate against an undefined symbol 'fmaxf'

2. Even after we provide our own ``fmaxf``, clang emits the call with
   ``R_X86_64_PLT32`` (relocation type 4) because libm calls go through
   the PLT in the standard ABI. Tinygrad's ``elf.relocate()`` only
   supports ``R_X86_64_PC32`` (type 2) and the AARCH64 set, so it raises:
       NotImplementedError: Encountered unknown relocation type 4

This bites OPT (and any model whose softmax/topk emits the pattern),
even though GPT-2 happens not to vectorize through that path.

The fix has two parts:

* **Provide concrete fmaxf/fminf/fmax/fmin definitions** by appending
  them to the source we feed to clang. The vectorizer can emit calls,
  and they resolve to our own implementations within the same TU.

* **Teach the relocator about R_X86_64_PLT32**. PLT32 is just a 32-bit
  PC-relative reloc with extra ABI semantics for dynamic linking; in
  a self-contained JIT image the PC-relative arithmetic is identical
  to R_X86_64_PC32. So we route type 4 through the existing PC32 path.

Importing this module patches both ``ClangJITCompiler.compile`` and
``elf.relocate`` once, idempotently. It is imported as a side effect
from ``ai/llm/tinyllm/__init__.py``.
"""

from __future__ import annotations

import importlib

_PATCHED_COMPILER = "_cyborg_compiler_shim_applied"
_PATCHED_RELOC = "_cyborg_reloc_shim_applied"

# X86-64 ABI relocation type for PLT-resolved 32-bit PC-relative calls.
_R_X86_64_PLT32 = 4

_LIBM_IMPL = """
#ifndef CYBORG_LIBM_IMPL
#define CYBORG_LIBM_IMPL
/* Concrete fmaxf/fminf/fmax/fmin so clang's vectorizer-emitted libm
 * calls resolve within the same translation unit. */
static float  fmaxf(float a, float b)   { return a > b ? a : b; }
static float  fminf(float a, float b)   { return a < b ? a : b; }
static double fmax (double a, double b) { return a > b ? a : b; }
static double fmin (double a, double b) { return a < b ? a : b; }
#endif
"""


def _patch_compiler() -> None:
    try:
        ops_cpu = importlib.import_module("tinygrad.runtime.ops_cpu")
    except ImportError:
        return

    cls = getattr(ops_cpu, "ClangJITCompiler", None)
    if cls is None or getattr(cls, _PATCHED_COMPILER, False):
        return

    original_compile = cls.compile

    def compile_with_shim(self, src: str):
        if "CYBORG_LIBM_IMPL" in src:
            return original_compile(self, src)
        return original_compile(self, src + _LIBM_IMPL)

    cls.compile = compile_with_shim
    setattr(cls, _PATCHED_COMPILER, True)


def _patch_relocator() -> None:
    """Make elf.relocate accept R_X86_64_PLT32 (type 4)."""
    try:
        elf = importlib.import_module("tinygrad.runtime.support.elf")
    except ImportError:
        return
    if getattr(elf, _PATCHED_RELOC, False):
        return

    original_relocate = elf.relocate

    def relocate_with_plt32(instr: int, ploc: int, tgt: int, r_type: int):
        # PLT32 is mathematically identical to PC32 in a self-contained
        # JIT image (no dynamic linker, no PLT trampoline). Route it
        # through the existing PC32 case.
        if r_type == _R_X86_64_PLT32:
            try:
                libc = importlib.import_module("tinygrad.runtime.support.libc")
                return original_relocate(instr, ploc, tgt, libc.R_X86_64_PC32)
            except (ImportError, AttributeError):
                from tinygrad.helpers import i2u

                return i2u(32, tgt - ploc)
        return original_relocate(instr, ploc, tgt, r_type)

    elf.relocate = relocate_with_plt32
    setattr(elf, _PATCHED_RELOC, True)


def apply_libm_shim() -> None:
    """Apply both the compiler and the relocator patches, idempotent."""
    _patch_compiler()
    _patch_relocator()


apply_libm_shim()
