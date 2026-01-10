"""TinyLLM: A minimal LLM library built on TinyGrad.

A minimal, educational LLM library using TinyGrad with zero PyTorch dependencies.
Provides clean implementations of LLM building blocks for multiple model families.
"""

from .utils import load_weights

__all__ = ['load_weights']

__version__ = '0.1.0'
