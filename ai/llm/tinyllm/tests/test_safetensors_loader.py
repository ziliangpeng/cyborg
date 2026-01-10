"""Test custom safetensors loader against official library."""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from ai.llm.tinyllm.utils.safetensors_loader import SafetensorsLoader, load_safetensors
from safetensors import safe_open
from huggingface_hub import hf_hub_download


def test_custom_vs_official_loader():
    """
    Test custom loader against official safetensors library.

    Uses a tiny test file from HuggingFace to verify:
    - Same tensor names
    - Same shapes
    - Same dtypes
    - Same values (within numerical precision)
    """
    print("Testing custom safetensors loader...\n")

    # Download tiny test file (4.22 KB)
    model_id = "hf-internal-testing/tiny-random-bert-sharded-safetensors"
    filename = "model-00001-of-00005.safetensors"

    print(f"Downloading test file: {model_id}/{filename}")
    filepath = hf_hub_download(repo_id=model_id, filename=filename)
    print(f"✓ Downloaded to: {filepath}")
    print()

    # Load with custom loader
    print("Loading with custom loader...")
    custom_loader = SafetensorsLoader(filepath)
    custom_tensors = custom_loader.load_all()
    print(f"✓ Custom loader: loaded {len(custom_tensors)} tensors")
    print(f"  Tensor names: {list(custom_tensors.keys())}")
    print()

    # Load with official loader
    print("Loading with official safetensors library...")
    official_tensors = {}
    with safe_open(filepath, framework="numpy") as f:
        for key in f.keys():
            official_tensors[key] = f.get_tensor(key)
    print(f"✓ Official loader: loaded {len(official_tensors)} tensors")
    print()

    # Compare results
    print("Comparing loaders...\n")

    # Check same number of tensors
    assert len(custom_tensors) == len(official_tensors), \
        f"Different number of tensors: {len(custom_tensors)} vs {len(official_tensors)}"
    print(f"✓ Same number of tensors: {len(custom_tensors)}")

    # Check same keys
    assert set(custom_tensors.keys()) == set(official_tensors.keys()), \
        "Different tensor names"
    print(f"✓ Same tensor names")

    # Compare each tensor
    for name in custom_tensors.keys():
        custom_arr = custom_tensors[name]
        official_arr = official_tensors[name]

        # Check shape
        assert custom_arr.shape == official_arr.shape, \
            f"Shape mismatch for '{name}': {custom_arr.shape} vs {official_arr.shape}"

        # Check dtype
        assert custom_arr.dtype == official_arr.dtype, \
            f"Dtype mismatch for '{name}': {custom_arr.dtype} vs {official_arr.dtype}"

        # Check values (exact match for integers, close match for floats)
        if np.issubdtype(custom_arr.dtype, np.integer):
            assert np.array_equal(custom_arr, official_arr), \
                f"Value mismatch for '{name}'"
        else:
            assert np.allclose(custom_arr, official_arr, rtol=1e-6, atol=1e-8), \
                f"Value mismatch for '{name}'"

        print(f"✓ {name}: shape={custom_arr.shape}, dtype={custom_arr.dtype}")

    print()
    print("=" * 60)
    print("✅ All tests passed! Custom loader matches official library.")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  - Loaded {len(custom_tensors)} tensors successfully")
    print(f"  - All shapes match")
    print(f"  - All dtypes match")
    print(f"  - All values match (within numerical precision)")
    print()
    print("✨ Custom safetensors loader is working correctly!")


if __name__ == "__main__":
    test_custom_vs_official_loader()
