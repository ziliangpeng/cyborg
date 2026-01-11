"""Base model class for TinyLLM."""

from abc import ABC, abstractmethod

from tinygrad import Tensor


class BaseModel(ABC):
    """Base class for all TinyLLM models."""

    @abstractmethod
    def __call__(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass returning logits.

        Args:
            input_ids: (batch_size, seq_len) token indices

        Returns:
            (batch_size, seq_len, vocab_size) logits
        """
        pass

    def param_count(self) -> int:
        """
        Count total number of parameters in the model.

        Returns:
            Total parameter count
        """
        total = 0
        for tensor in self._get_all_tensors():
            total += tensor.numpy().size
        return total

    def _get_all_tensors(self) -> list[Tensor]:
        """
        Recursively collect all Tensor parameters from the model.

        Returns:
            List of all parameter tensors
        """
        tensors = []
        self._collect_tensors(self, tensors)
        return tensors

    def _collect_tensors(self, obj, tensors: list[Tensor]) -> None:
        """Recursively collect tensors from an object."""
        if isinstance(obj, Tensor):
            tensors.append(obj)
        elif hasattr(obj, "__dict__"):
            for value in obj.__dict__.values():
                self._collect_tensors(value, tensors)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._collect_tensors(item, tensors)
