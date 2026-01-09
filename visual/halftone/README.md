# Halftone Library

Python library for halftone image processing with 8 different styles.

## Overview

Halftone is a technique for simulating continuous-tone images using dots of varying sizes or patterns. Originally developed for printing presses (which can only print solid ink), halftone creates the illusion of gray tones and colors through optical mixing.

## Styles

This library implements 8 halftone styles:

1. **CMYK** - 4-color separation with rotated screens (classic printing press)
2. **Grayscale Sqrt** - Accurate tone reproduction with square root scaling
3. **Grayscale Linear** - Stylistic with linear scaling (darker shadows)
4. **Floyd-Steinberg** - Error diffusion dithering (organic patterns)
5. **Ordered Dither** - Bayer matrix dithering (regular patterns)
6. **Stippling** - Stochastic dot placement (hand-drawn effect)
7. **Line Screen** - Parallel lines with varying thickness (engraving)
8. **Crosshatch** - Intersecting lines (technical drawing)

See `docs/HALFTONE_STYLES.md` for detailed descriptions.

## Installation

Add to your Bazel BUILD file:

```python
py_library(
    name = "my_app",
    srcs = ["my_app.py"],
    deps = [
        "//python/halftone",
    ],
)
```

## Usage

### Basic Example

```python
from PIL import Image
from halftone import (
    StyleType, CmykParams, ProcessParams, process
)

# Load image
img = Image.open("input.jpg")

# Configure style
style_params = CmykParams(sample=8, scale=1)
process_params = ProcessParams(antialias=True)

# Process
result = process(img, StyleType.CMYK, style_params, process_params)

# Save
result.save("output.png")
```

### Process Multiple Styles

```python
from halftone import (
    StyleType, CmykParams, GrayscaleParams,
    ProcessParams, process_multiple
)

img = Image.open("input.jpg")

styles = [
    (StyleType.CMYK, CmykParams(sample=8)),
    (StyleType.GRAYSCALE_SQRT, GrayscaleParams(sample=8)),
]

params = ProcessParams(antialias=True)
results = process_multiple(img, styles, params)

for style_name, result_img in results.items():
    result_img.save(f"output_{style_name}.png")
```

### All Styles

```python
from halftone import StyleType, all_style_names, parse_style_name

# Get all style names
for name in all_style_names():
    print(name)

# Parse style from name
style_type = parse_style_name("cmyk")
```

## API Reference

### Style Types

```python
class StyleType(Enum):
    CMYK
    GRAYSCALE_SQRT
    GRAYSCALE_LINEAR
    FLOYD_STEINBERG
    ORDERED_DITHER
    STIPPLING
    LINE_SCREEN
    CROSSHATCH
```

### Parameters

```python
@dataclass
class CmykParams:
    sample: int = 8
    scale: int = 1
    angles: Optional[tuple[float, float, float, float]] = None

@dataclass
class GrayscaleParams:
    sample: int = 8
    scale: int = 1
    angle: float = 0.0

@dataclass
class DitherParams:
    matrix_size: int = 4  # 2, 4, or 8

@dataclass
class StipplingParams:
    cell_size: int = 8
    density: float = 1.0
    randomness: float = 0.3

@dataclass
class LineScreenParams:
    angle: float = 45.0
    frequency: int = 8

@dataclass
class CrosshatchParams:
    angle1: float = 45.0
    angle2: float = -45.0
    frequency: int = 8

@dataclass
class ProcessParams:
    antialias: bool = True
```

### Main Functions

```python
def process(
    image: Image.Image,
    style_type: StyleType,
    style_params: Any,
    params: ProcessParams = ProcessParams()
) -> Image.Image:
    """Process image with single style."""

def process_multiple(
    image: Image.Image,
    styles: list[tuple[StyleType, Any]],
    params: ProcessParams = ProcessParams()
) -> dict[str, Image.Image]:
    """Process image with multiple styles."""
```

## Algorithm Details

### CMYK Halftone

1. Convert RGB to CMYK color space
2. Separate into 4 channels (Cyan, Magenta, Yellow, Black)
3. For each channel:
   - Rotate by screen angle (C:0°, M:15°, Y:30°, K:45°)
   - Sample grid and calculate dot sizes
   - Draw white dots on black background
   - Rotate back
4. Merge channels to CMYK image
5. Convert to RGB for display

### Grayscale Halftones

**Sqrt variant** (accurate):
- Uses `diameter = sqrt(intensity/255)` for proper tone curve
- White dots on black background
- Better midtone reproduction

**Linear variant** (stylistic):
- Uses `radius = (1 - intensity/255) * max_radius`
- Black dots on white background
- More dramatic shadows

### Dithering

**Floyd-Steinberg**:
- Error diffusion algorithm
- Distributes quantization error to neighbors
- Creates organic patterns

**Bayer**:
- Threshold matrix comparison
- Regular checkerboard patterns
- Fast processing

### Others

See individual module documentation for details on stippling, line screens, and crosshatch.

## Performance

- Most algorithms are O(n) where n = number of pixels
- CMYK is 4x slower (4 channels to process)
- Antialiasing adds 4x overhead
- Typical times on M4 Pro for 1MP image:
  - Bayer: ~50ms
  - Floyd-Steinberg: ~200ms
  - Stippling: ~500ms
  - CMYK: ~1-2s

### Future Optimizations

The current implementation prioritizes **readability and correctness** over performance. For production use cases requiring higher throughput, consider:

- **NumPy vectorization**: Replace loop-based sampling with array reshape + mean operations
  - Expected speedup: 2-3x for CMYK (1-2s → 0.4-0.7s)
  - Trade-off: More complex code, harder to maintain
  - Best for: Batch processing, web services, large images (12MP+)

- **Parallel processing**: Process CMYK channels in parallel using multiprocessing
  - Expected speedup: 2-3x on multi-core systems
  - Trade-off: Memory overhead, process spawning cost
  - Best for: Very large images or batch processing

- **scipy.ndimage**: Use scipy for image rotation in line_screen.py
  - Expected speedup: 150ms → 50ms for line screens
  - Trade-off: Adds scipy dependency
  - Current performance is acceptable

Current performance is sufficient for interactive CLI use. Optimize only when needed.

## See Also

- `docs/HALFTONE_STYLES.md` - Detailed style descriptions
- `docs/IMAGE_REDUCTION.md` - Overview of image reduction techniques
- `//apps/halftone` - Command-line tool
