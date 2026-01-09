# Halftone Styles Reference

## What is Halftone?

Halftone is a technique for simulating continuous-tone images using patterns of dots or lines. Originally developed for printing presses that can only print solid ink, halftone creates the illusion of different tones through:

- **Dot size variation** - Larger dots appear darker
- **Dot density** - More dots in an area appear darker
- **Optical mixing** - Your eye averages the pattern from a distance

**Key insight**: Halftone trades tonal range (shades) for spatial patterns (dots). From far away, patterns blend to create perceived tones.

## The 8 Styles

### 1. CMYK Halftone

**Algorithm**: Four-color separation with rotated screens

**How it works**:
- Separates image into Cyan, Magenta, Yellow, Black channels
- Each channel printed with dots at different angles (C:0°, M:15°, Y:30°, K:45°)
- Angles prevent moiré patterns, create rosette pattern
- Dots use sqrt(intensity) scaling for accurate tone reproduction

**Parameters**:
- `sample` (default: 8) - Dot grid size
- `scale` (default: 1) - Output resolution multiplier
- `angles` (default: [0, 15, 30, 45]) - Screen angles for each channel

**Visual characteristics**:
- Characteristic rosette pattern when zoomed in
- Accurate color reproduction
- Classic "printed magazine" look

**Best for**: Color images, printing press simulation, accurate reproduction

**Performance**: ~1-2s for 1MP (4x channels)

---

### 2. Grayscale Sqrt (Correct)

**Algorithm**: Square root tone curve, white dots on black

**How it works**:
- Convert to grayscale
- For each grid cell, calculate average brightness
- Dot diameter = sqrt(brightness / 255)
- White dots on black background represents ink coverage

**Why sqrt?** Human eyes perceive brightness logarithmically. Sqrt scaling ensures 50% gray has dots covering ~50% of area.

**Parameters**:
- `sample` (default: 8) - Dot grid size
- `scale` (default: 1) - Output scale
- `angle` (default: 0) - Screen rotation angle

**Visual characteristics**:
- Balanced midtones
- Similar overall brightness to original
- Classic newspaper look

**Best for**: Grayscale images, accurate newspaper reproduction, balanced tones

**Performance**: ~150-200ms for 1MP

---

### 3. Grayscale Linear (Stylistic)

**Algorithm**: Linear tone curve, black dots on white

**How it works**:
- Convert to grayscale
- For each grid cell: radius = (1 - brightness/255) * max_radius
- Black dots on white background
- Linear relationship between brightness and dot size

**Difference from sqrt**: Linear scaling makes midtones darker, creates more dramatic contrast.

**Parameters**:
- `sample` (default: 8) - Dot grid size
- `scale` (default: 1) - Output scale

**Visual characteristics**:
- Darker shadows, more contrast
- Artistic, dramatic look
- Less accurate to original brightness

**Best for**: Artistic effects, high contrast images, dramatic style

**Performance**: ~100-150ms for 1MP (simpler than sqrt)

---

### 4. Floyd-Steinberg Dithering

**Algorithm**: Error diffusion dithering

**How it works**:
- Convert to grayscale
- For each pixel, round to black (0) or white (255)
- Calculate quantization error
- Distribute error to neighboring pixels:
  ```
  Current pixel distributes to:
         [X]  7/16
  3/16  5/16  1/16
  ```
- Error "flows" through image, creating organic patterns

**Parameters**: None (automatic)

**Visual characteristics**:
- Organic, flowing dot patterns
- No regular structure
- Classic retro computer graphics look (e.g., early Mac)

**Best for**: Retro aesthetics, 1-bit graphics, organic patterns

**Performance**: ~200ms for 1MP (sequential algorithm)

---

### 5. Ordered Dithering (Bayer)

**Algorithm**: Threshold matrix dithering

**How it works**:
- Uses repeating Bayer matrix (2×2, 4×4, or 8×8)
- For each pixel: if intensity > matrix[y%size][x%size], set white, else black
- Creates regular checkerboard-like patterns

**Bayer Matrix (4×4)**:
```
 0  8  2 10
12  4 14  6
 3 11  1  9
15  7 13  5
(normalized to 0-1)
```

**Parameters**:
- `matrix_size` (default: 4) - Size of Bayer matrix (2, 4, or 8)
  - 2×2: Coarse, obvious checkerboard
  - 4×4: Medium detail, balanced
  - 8×8: Fine detail, smoother gradients

**Visual characteristics**:
- Very regular, geometric patterns
- Predictable, uniform appearance
- Fast processing

**Best for**: Uniform patterns, crosshatch effects, fast processing needs

**Performance**: ~50ms for 1MP (very fast, no error propagation)

---

### 6. Stippling

**Algorithm**: Stochastic dot placement

**How it works**:
- Divide image into cells
- For each cell, calculate darkness = 1 - (brightness / 255)
- Place random dots proportional to darkness
- Dot positions jittered for organic look

**Parameters**:
- `cell_size` (default: 8) - Grid cell size
- `density` (default: 1.0) - Dot density multiplier
  - < 1.0: Sparse, lighter
  - > 1.0: Dense, darker
- `randomness` (default: 0.3) - Position jitter amount

**Visual characteristics**:
- Random, organic dot placement
- Looks hand-drawn/artistic
- No regular structure
- Large file sizes (many small elements)

**Best for**: Artistic/illustration effects, pointillism style, hand-drawn look, scientific illustrations

**Performance**: ~500ms for 1MP (random generation overhead)

---

### 7. Line Screen

**Algorithm**: Parallel lines with varying thickness

**How it works**:
- Draw parallel lines at specified angle
- Sample image along each line position
- Line thickness proportional to darkness
- Rotate and crop to fit original dimensions

**Parameters**:
- `angle` (default: 45°) - Line direction
  - 0°: Vertical lines
  - 45°: Diagonal lines
  - 90°: Horizontal lines
- `frequency` (default: 8) - Line spacing in pixels

**Visual characteristics**:
- Very small file sizes (simple geometry)
- Strong directional effect
- Vintage engraving aesthetic
- Angle dramatically changes appearance

**Best for**: Vintage book illustrations, engraving/etching style, technical drawings, currency/banknote effects

**Performance**: ~150ms for 1MP

---

### 8. Crosshatch

**Algorithm**: Two intersecting line screens

**How it works**:
- Create two line screens at different angles (e.g., ±45°)
- Combine using minimum (darken where lines intersect)
- Creates woven/grid texture

**Parameters**:
- `angle1` (default: 45°) - First line angle
- `angle2` (default: -45°) - Second line angle
- `frequency` (default: 8) - Line spacing

**Visual characteristics**:
- Grid-like, woven appearance
- Stronger than single line screen
- Technical drawing aesthetic
- Medium file size

**Best for**: Architectural drawings, technical illustrations, fabric/textile textures, classical engraving style

**Performance**: ~300ms for 1MP (2× line screen)

---

## Comparison Matrix

| Style | Detail | Pattern | Speed | File Size | Best Use |
|-------|--------|---------|-------|-----------|----------|
| **CMYK** | High | Rosette | Slow | Large | Color printing |
| **Grayscale Sqrt** | High | Regular dots | Medium | Medium | Newspaper |
| **Grayscale Linear** | High | Regular dots | Fast | Medium | Artistic |
| **Floyd-Steinberg** | High | Organic | Medium | Large | Retro graphics |
| **Bayer** | Medium | Regular grid | Very Fast | Small | Uniform patterns |
| **Stippling** | High | Random | Slow | Large | Hand-drawn art |
| **Line Screen** | Medium | Lines | Fast | Tiny | Engraving |
| **Crosshatch** | Medium | Grid | Fast | Small | Technical |

## Technical Notes

### Dot Scaling: Linear vs Square Root

**Linear** (incorrect but stylistic):
- `radius = (1 - intensity/255) * max_radius`
- 50% gray → 50% radius → 25% area coverage
- Results in darker midtones

**Square root** (correct):
- `diameter = sqrt(intensity/255) * max_diameter`
- 50% gray → 71% diameter → ~50% area coverage
- Accurate tone reproduction

### Screen Angles (CMYK)

Standard angles prevent moiré interference:
- **Cyan**: 0° or 15°
- **Magenta**: 75° or 15°
- **Yellow**: 0° or 90° (least visible to eye)
- **Black**: 45° (most visible, diagonal less obvious than H/V)

Angle differences of 15-45° create rosette patterns instead of moiré.

### Antialiasing

When enabled:
- Renders at 4× resolution
- Scales down with LANCZOS resampling
- Smoother dot edges
- 4× processing time

## Examples

```bash
# Classic color printing
halftone image.jpg --style cmyk --sample 6

# Accurate grayscale newspaper
halftone image.jpg --style grayscale-sqrt --sample 8

# Dramatic artistic effect
halftone image.jpg --style grayscale-linear --sample 10

# Retro computer graphics
halftone image.jpg --style floyd-steinberg

# Uniform crosshatch
halftone image.jpg --style ordered-dither --matrix-size 4

# Hand-drawn stippling
halftone image.jpg --style stippling --density 1.2

# Vintage engraving
halftone image.jpg --style line-screen --angle 30

# Technical drawing
halftone image.jpg --style crosshatch --frequency 6
```
