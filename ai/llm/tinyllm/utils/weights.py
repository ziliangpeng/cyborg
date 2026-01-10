"""Weight loading utilities for TinyLLM."""

from pathlib import Path
from typing import Dict
from tinygrad import Tensor
from huggingface_hub import snapshot_download
from .safetensors_loader import SafetensorsLoader


def load_weights(model_name: str) -> Dict[str, Tensor]:
    """
    Load model weights from HuggingFace (auto-cached).

    Args:
        model_name: HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")

    Returns:
        Dict mapping weight names to TinyGrad Tensor objects

    Example:
        >>> weights = load_weights("gpt2")
        >>> print(f"Loaded {len(weights)} tensors")
        >>> print(f"Sample keys: {list(weights.keys())[:3]}")
    """
    # Download/get cached model (HuggingFace manages cache automatically at ~/.cache/huggingface/)
    cache_dir = _download_from_hf(model_name)

    # Load safetensors files from cache using our custom loader
    weights = _load_safetensors(cache_dir)

    # Convert to TinyGrad tensors
    return {name: Tensor(arr) for name, arr in weights.items()}


def _download_from_hf(model_name: str) -> Path:
    """
    Download model from HuggingFace (uses their cache at ~/.cache/huggingface/).

    Args:
        model_name: HuggingFace model ID

    Returns:
        Path to cached model directory
    """
    # snapshot_download handles caching automatically
    # It will download if not cached, or return cached path if already downloaded
    cache_dir = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.safetensors"],  # Only download safetensors files
    )
    return Path(cache_dir)


def _load_safetensors(cache_dir: Path) -> Dict[str, any]:
    """
    Load all safetensors files in directory and return as numpy arrays.

    Uses our custom safetensors loader implementation.

    Args:
        cache_dir: Directory containing .safetensors files

    Returns:
        Dict mapping weight names to numpy arrays

    Raises:
        FileNotFoundError: If no safetensors files found in cache_dir
    """
    weights = {}

    # Find all .safetensors files
    safetensors_files = sorted(cache_dir.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(
            f"No safetensors files found in {cache_dir}. "
            f"Model may not have safetensors format available."
        )

    # Load each file with our custom loader
    for file in safetensors_files:
        loader = SafetensorsLoader(file)
        file_weights = loader.load_all()
        weights.update(file_weights)

    return weights
