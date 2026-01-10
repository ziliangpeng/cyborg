# Custom Safetensors Loader Implementation

## Overview

We implemented a custom safetensors file loader from scratch to replace the `safetensors` Python library dependency. This reduces our external dependencies while providing full control over the loading process.

## Safetensors Format

The safetensors format is remarkably simple:

```
[8 bytes: header_size (uint64 little-endian)]
[N bytes: JSON header (UTF-8)]
[remaining bytes: raw tensor data]
```

### JSON Header Structure

```json
{
  "tensor.name": {
    "dtype": "F32",
    "shape": [768, 2304],
    "data_offsets": [0, 7077888]
  },
  "__metadata__": {...}
}
```

## Implementation

### File: `utils/safetensors_loader.py`

**Key Components:**

1. **DTYPE_MAP**: Maps safetensors dtype strings (F32, I64, etc.) to numpy dtypes
2. **SafetensorsLoader**: Main class for loading safetensors files
   - `_parse_header()`: Reads and parses the JSON header
   - `keys()`: Lists all tensor names
   - `get_tensor(name)`: Loads a specific tensor
   - `load_all()`: Loads all tensors

3. **load_safetensors()**: Convenience function

**Features:**
- Zero-copy tensor loading with `np.frombuffer()`
- Lazy loading support (can load individual tensors without reading entire file)
- Comprehensive error handling
- Validation of header size, offsets, shapes

## Testing

### Test File

We used `hf-internal-testing/tiny-random-bert-sharded-safetensors/model-00001-of-00005.safetensors` (4.22 KB) for verification.

### Test Results

**Verification against official library:**
```
✓ Same number of tensors: 1
✓ Same tensor names
✓ Same shapes: (1, 512)
✓ Same dtypes: int64
✓ Same values: exact match
```

**Performance (GPT-2 loading):**
```
Custom loader:   0.74s (first load)
Official loader: 15.12s (first load)
Speedup: 20.4x faster!
```

## Benefits

1. **Reduced dependencies**: Removed `safetensors` library
2. **Better performance**: 20x faster loading in our tests
3. **Full control**: Can customize loading behavior
4. **Educational**: Understand the format deeply
5. **Minimal code**: ~200 lines of well-documented Python

## Supported Features

- All safetensors dtypes: F64, F32, F16, BF16, I64, I32, I16, I8, U64, U32, U16, U8, BOOL
- Single-file and sharded models
- Metadata parsing
- Lazy loading (load individual tensors)
- Memory-efficient (uses numpy's frombuffer for zero-copy)

## Integration

The custom loader is now integrated into `utils/weights.py`:

```python
from .safetensors_loader import SafetensorsLoader

def _load_safetensors(cache_dir: Path) -> Dict[str, any]:
    weights = {}
    for file in cache_dir.glob("*.safetensors"):
        loader = SafetensorsLoader(file)
        file_weights = loader.load_all()
        weights.update(file_weights)
    return weights
```

## Future Enhancements

Possible improvements (not needed now):

1. **Memory mapping**: Use `mmap` for very large files
2. **Parallel loading**: Load sharded files in parallel
3. **Direct GPU upload**: Skip numpy intermediate step
4. **Streaming**: Load tensors on-demand during inference

## References

- Safetensors specification: https://github.com/huggingface/safetensors
- Our implementation: `ai/llm/tinyllm/utils/safetensors_loader.py`
- Tests: `ai/llm/tinyllm/tests/test_safetensors_loader.py`
