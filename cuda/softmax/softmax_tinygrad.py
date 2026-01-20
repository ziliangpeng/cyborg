"""
TinyGrad softmax implementation for benchmarking.

This module provides softmax using TinyGrad's built-in operation.
Interface uses NumPy arrays for compatibility with other implementations.
"""

import numpy as np
from tinygrad import Device
from tinygrad.tensor import Tensor

# Ensure GPU usage
Device.DEFAULT = "CUDA"


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax using TinyGrad (built-in).

    Args:
        x: 1D float32 NumPy array

    Returns:
        1D float32 NumPy array with softmax output
    """
    t = Tensor(x, device="CUDA")
    result = t.softmax()
    result.realize()
    return result.numpy()


def benchmark(x: np.ndarray, iterations: int = 100, warmup: int = 10) -> list[float]:
    """
    Benchmark softmax kernel.

    Args:
        x: Input array (1D float32)
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        List of timing results in milliseconds
    """
    import time

    # Create tensor once (avoid repeated conversion overhead)
    t = Tensor(x, device="CUDA")

    # Warmup
    for _ in range(warmup):
        result = t.softmax()
        result.realize()
        # Force synchronization by reading a value
        _ = result.numpy().flatten()[0]

    # Benchmark with proper GPU synchronization
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = t.softmax()
        result.realize()
        # Force synchronization
        _ = result.numpy().flatten()[0]
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return times
