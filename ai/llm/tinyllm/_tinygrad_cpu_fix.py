"""Monkey-patch tinygrad's CPU (clang JIT) compiler to add -fno-builtin.

Tinygrad 0.11.0 compiles kernels with ``-O2 -nostdlib -ffreestanding`` (see
``tinygrad.runtime.ops_cpu.ClangJITCompiler.compile``). At -O2 clang
recognises the ternary ``(a > b ? a : b)`` pattern emitted by tinygrad's
C renderer and lowers it to a ``fmaxf`` (or ``fminf``) builtin call.
Because ``-nostdlib`` does not link libm, the resulting ELF references
an undefined ``fmaxf`` symbol and tinygrad's ELF loader raises:

    RuntimeError: Attempting to relocate against an undefined symbol 'fmaxf'

This bites OPT (and any model whose softmax reduction emits the pattern),
even though GPT-2 happens to compile clean.

Fix: add ``-fno-builtin`` so clang does not perform that pattern -> libm
substitution. Importing this module patches ``ClangJITCompiler.compile``
once, idempotently. Imported as a side effect from ``ai/llm/tinyllm/__init__.py``.
"""

from __future__ import annotations

import importlib
import subprocess

_PATCHED = "_cyborg_libm_shim_applied"


def apply_libm_shim() -> None:
    """Wrap ClangJITCompiler.compile to inject -fno-builtin into the clang args."""
    try:
        ops_cpu = importlib.import_module("tinygrad.runtime.ops_cpu")
    except ImportError:
        return

    cls = getattr(ops_cpu, "ClangJITCompiler", None)
    if cls is None or getattr(cls, _PATCHED, False):
        return

    original_check_output = subprocess.check_output

    def _patched_compile(self, src: str) -> bytes:
        # Re-implement the upstream compile() logic but with -fno-builtin appended.
        # Importing locally keeps the patch self-contained and avoids depending on
        # private state of the original method.
        import platform
        import sys

        from tinygrad.helpers import getenv
        from tinygrad.runtime.support.elf import jit_loader

        target = "x86_64" if sys.platform == "win32" else platform.machine()
        arch = "-march=native" if platform.machine() in ("x86_64", "AMD64") else "-mcpu=native"
        args = [
            arch,
            f"--target={target}-none-unknown-elf",
            "-O2",
            "-fPIC",
            "-ffreestanding",
            "-fno-math-errno",
            "-nostdlib",
            "-fno-ident",
            # Prevent clang from lowering ternary max/min into fmaxf/fminf
            # libm calls under -O2, which would leave undefined symbols when
            # combined with -nostdlib.
            "-fno-builtin",
        ]
        arch_args = ["-ffixed-x18"] if target == "arm64" else []
        cc = getenv("CC", "clang")
        obj = original_check_output(
            [cc, "-c", "-x", "c", *args, *arch_args, "-", "-o", "-"],
            input=src.encode("utf-8"),
        )
        return jit_loader(obj)

    cls.compile = _patched_compile
    setattr(cls, _PATCHED, True)


apply_libm_shim()
