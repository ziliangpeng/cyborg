"""Weight loading utilities for TinyLLM."""

from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tinygrad import Tensor


def load_weights(model_name: str) -> dict[str, Tensor]:
    """
    Load model weights from HuggingFace (auto-cached).

    Supports both safetensors and PyTorch bin formats.

    Args:
        model_name: HuggingFace model ID (e.g., "gpt2", "facebook/opt-125m")

    Returns:
        Dict mapping weight names to TinyGrad Tensor objects

    Example:
        >>> weights = load_weights("gpt2")
        >>> print(f"Loaded {len(weights)} tensors")
        >>> print(f"Sample keys: {list(weights.keys())[:3]}")
    """
    # Download/get cached model (HuggingFace manages cache automatically at ~/.cache/huggingface/)
    cache_dir = _download_from_hf(model_name)

    # Try safetensors first, fall back to PyTorch bin
    weights = _load_weights_from_dir(cache_dir)

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
    # Try safetensors first, then fall back to pytorch_model.bin
    cache_dir = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.safetensors", "*.bin"],  # Download both formats
    )
    return Path(cache_dir)


def _load_weights_from_dir(cache_dir: Path) -> dict[str, Any]:
    """
    Load weights from directory, preferring safetensors over PyTorch bin.

    Args:
        cache_dir: Directory containing weight files

    Returns:
        Dict mapping weight names to numpy arrays

    Raises:
        FileNotFoundError: If no weight files found
    """
    # Try safetensors first
    safetensors_files = sorted(cache_dir.glob("*.safetensors"))
    if safetensors_files:
        return _load_safetensors(cache_dir)

    # Fall back to PyTorch bin
    bin_files = sorted(cache_dir.glob("*.bin"))
    if bin_files:
        return _load_pytorch_bin(cache_dir)

    raise FileNotFoundError(f"No weight files found in {cache_dir}. Expected .safetensors or .bin files.")


def _load_safetensors(cache_dir: Path) -> dict[str, Any]:
    """
    Load all safetensors files in directory and return as numpy arrays.

    Args:
        cache_dir: Directory containing .safetensors files

    Returns:
        Dict mapping weight names to numpy arrays
    """
    weights = {}

    # Find all .safetensors files
    safetensors_files = sorted(cache_dir.glob("*.safetensors"))

    # Load each file with safetensors library
    for file in safetensors_files:
        with safe_open(file, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    return weights


def _load_pytorch_bin(cache_dir: Path) -> dict[str, Any]:
    """
    Load all PyTorch bin files in directory and return as numpy arrays.

    Args:
        cache_dir: Directory containing .bin files

    Returns:
        Dict mapping weight names to numpy arrays
    """
    # TODO: Remove torch dependency by implementing custom pickle-based loader.
    # PyTorch .bin files are ZIP archives with pickle metadata + raw tensor data.
    # We can parse them without torch using zipfile + pickle + numpy.
    import torch

    weights = {}

    # Find all .bin files
    bin_files = sorted(cache_dir.glob("*.bin"))

    # Load each file with torch
    for file in bin_files:
        state_dict = torch.load(file, map_location="cpu", weights_only=True)
        for key, tensor in state_dict.items():
            weights[key] = tensor.numpy()

    return weights
