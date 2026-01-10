"""Custom safetensors loader implementation.

This module implements a minimal safetensors file loader without external dependencies
(beyond Python stdlib and numpy). The safetensors format is:

    [8 bytes: header_size (uint64 little-endian)]
    [N bytes: JSON header (UTF-8)]
    [remaining bytes: raw tensor data]

The JSON header contains metadata for each tensor:
    {
        "tensor.name": {
            "dtype": "F32",
            "shape": [768, 2304],
            "data_offsets": [0, 7077888]  // byte offsets in data block
        },
        "__metadata__": {...}  // optional
    }
"""

import json
import struct
from pathlib import Path
from typing import Dict
import numpy as np


# Mapping from safetensors dtype strings to numpy dtypes
DTYPE_MAP = {
    # Floating point
    'F64': np.float64,
    'F32': np.float32,
    'F16': np.float16,
    'BF16': np.uint16,  # BFloat16 - stored as uint16, needs special handling

    # Signed integers
    'I64': np.int64,
    'I32': np.int32,
    'I16': np.int16,
    'I8': np.int8,

    # Unsigned integers
    'U64': np.uint64,
    'U32': np.uint32,
    'U16': np.uint16,
    'U8': np.uint8,

    # Boolean
    'BOOL': np.bool_,
}


class SafetensorsLoader:
    """Load tensors from safetensors file format."""

    def __init__(self, filepath: Path):
        """
        Initialize loader with a safetensors file.

        Args:
            filepath: Path to .safetensors file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.metadata = None
        self.data_start_offset = None
        self._parse_header()

    def _parse_header(self):
        """Parse the safetensors header to extract metadata."""
        with open(self.filepath, 'rb') as f:
            # Read header size (first 8 bytes, little-endian uint64)
            header_size_bytes = f.read(8)
            if len(header_size_bytes) != 8:
                raise ValueError("Invalid safetensors file: too short")

            header_size = struct.unpack('<Q', header_size_bytes)[0]

            # Validate header size (max 100MB as per spec)
            if header_size > 100 * 1024 * 1024:
                raise ValueError(f"Header size too large: {header_size} bytes")

            # Read JSON header
            header_bytes = f.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError("Invalid safetensors file: truncated header")

            # Parse JSON
            try:
                self.metadata = json.loads(header_bytes.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Invalid JSON header: {e}")

            # Data block starts after header
            self.data_start_offset = 8 + header_size

    def keys(self) -> list:
        """
        Get list of tensor names in the file.

        Returns:
            List of tensor names (excludes __metadata__)
        """
        return [k for k in self.metadata.keys() if k != "__metadata__"]

    def get_tensor(self, name: str) -> np.ndarray:
        """
        Load a specific tensor by name.

        Args:
            name: Name of the tensor to load

        Returns:
            Numpy array containing the tensor data

        Raises:
            KeyError: If tensor name not found
            ValueError: If tensor metadata is invalid
        """
        if name not in self.metadata:
            raise KeyError(f"Tensor '{name}' not found in file. Available: {self.keys()}")

        info = self.metadata[name]

        # Extract metadata
        dtype_str = info.get('dtype')
        shape = info.get('shape')
        data_offsets = info.get('data_offsets')

        if not all([dtype_str, shape is not None, data_offsets]):
            raise ValueError(f"Incomplete metadata for tensor '{name}'")

        # Map dtype
        if dtype_str not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        dtype = DTYPE_MAP[dtype_str]

        # Parse offsets
        if len(data_offsets) != 2:
            raise ValueError(f"Invalid data_offsets for tensor '{name}': {data_offsets}")
        begin, end = data_offsets

        # Validate offsets
        if begin < 0 or end < begin:
            raise ValueError(f"Invalid data offsets: [{begin}, {end}]")

        # Read tensor data
        byte_size = end - begin
        with open(self.filepath, 'rb') as f:
            f.seek(self.data_start_offset + begin)
            raw_bytes = f.read(byte_size)

            if len(raw_bytes) != byte_size:
                raise ValueError(f"Truncated tensor data for '{name}'")

        # Convert to numpy array
        try:
            arr = np.frombuffer(raw_bytes, dtype=dtype)
            arr = arr.reshape(tuple(shape))
        except ValueError as e:
            raise ValueError(f"Failed to reshape tensor '{name}': {e}")

        return arr

    def load_all(self) -> Dict[str, np.ndarray]:
        """
        Load all tensors from the file.

        Returns:
            Dict mapping tensor names to numpy arrays
        """
        tensors = {}
        for name in self.keys():
            tensors[name] = self.get_tensor(name)
        return tensors

    def get_metadata(self) -> dict:
        """
        Get the __metadata__ field if present.

        Returns:
            Metadata dict, or empty dict if not present
        """
        return self.metadata.get('__metadata__', {})


def load_safetensors(filepath: Path) -> Dict[str, np.ndarray]:
    """
    Convenience function to load all tensors from a safetensors file.

    Args:
        filepath: Path to .safetensors file

    Returns:
        Dict mapping tensor names to numpy arrays

    Example:
        >>> tensors = load_safetensors("model.safetensors")
        >>> print(f"Loaded {len(tensors)} tensors")
        >>> print(f"Keys: {list(tensors.keys())[:3]}")
    """
    loader = SafetensorsLoader(filepath)
    return loader.load_all()
