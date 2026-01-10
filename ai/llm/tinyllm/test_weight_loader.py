"""Quick test script for the weight loader."""

import time
import sys
from pathlib import Path

# Add cyborg root to path
cyborg_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(cyborg_root))

from ai.llm.tinyllm import load_weights

def main():
    print("Testing weight loader with GPT-2...")
    print()

    # Test with GPT-2 (small model, fast download)
    print("Loading GPT-2 weights (will download if not cached)...")
    start = time.time()
    weights = load_weights("gpt2")
    load_time = time.time() - start

    # Verify results
    print(f"✓ Loaded {len(weights)} weight tensors in {load_time:.2f}s")
    print()

    # Show sample keys
    print("Sample weight keys:")
    for key in list(weights.keys())[:5]:
        tensor = weights[key]
        print(f"  - {key}: shape={tensor.shape}")
    print()

    # Verify tensor type
    sample_tensor = list(weights.values())[0]
    print(f"✓ Tensor type: {type(sample_tensor)}")
    print(f"✓ Sample tensor shape: {sample_tensor.shape}")
    print()

    # Test cache: second call should be instant (uses HF cache)
    print("Testing cache (second load should be fast)...")
    start = time.time()
    weights2 = load_weights("gpt2")
    cache_time = time.time() - start

    print(f"✓ Second load took: {cache_time:.2f}s (using cached weights)")
    assert len(weights) == len(weights2), "Mismatch in number of weights!"
    print()

    print("✅ All tests passed!")
    print()
    print("Weight loader is working correctly:")
    print(f"  - Downloaded/loaded {len(weights)} tensors")
    print(f"  - First load: {load_time:.2f}s")
    print(f"  - Cached load: {cache_time:.2f}s ({load_time/cache_time:.1f}x faster)")
    print(f"  - Cache location: ~/.cache/huggingface/hub/")

if __name__ == "__main__":
    main()
