"""RMSNorm for TinyLLM."""

from tinygrad import Tensor


class RMSNorm:
    """Root Mean Square Layer Normalization (no bias, scale only)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = Tensor.ones(dim)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return (x / rms) * self.weight
