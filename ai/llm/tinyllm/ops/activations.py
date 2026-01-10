"""Activation functions for TinyLLM."""

import math

from tinygrad import Tensor


def gelu(x: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit activation.

    GPT-2 uses the approximate GELU:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return 0.5 * x * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))).tanh())
