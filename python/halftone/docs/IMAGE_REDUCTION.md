# Image Reduction Techniques

A comprehensive guide to image reduction and stylization techniques beyond halftone.

## What is Image Reduction?

**Image reduction** refers to techniques that simplify or reduce the information content of an image while maintaining recognizability. This includes:

- Reducing color depth (e.g., 24-bit → 1-bit)
- Reducing tonal range (continuous → discrete)
- Reducing detail (smooth → patterns)
- Abstracting to simpler representations

### Why Reduce Images?

1. **Technical constraints**: Limited display capabilities, printing limitations
2. **Bandwidth**: Smaller file sizes for transmission
3. **Artistic effects**: Stylization, retro aesthetics
4. **Emphasis**: Draw attention to structure over detail

## Halftone vs Other Techniques

**Halftone** is unique because it:
- Uses **spatial patterns** (dots/lines) to represent tone
- Creates **optical illusions** that blend at viewing distance
- **Simulates printing technology** historically
- **Preserves tonal gradients** better than simple thresholding

Other techniques either:
- Reduce colors directly (posterization)
- Extract features (edge detection)
- Apply different abstraction methods

---

## 1. Thresholding (Binary)

**Description**: Convert to pure black and white using a threshold value.

**Algorithm**:
```
for each pixel:
    if intensity > threshold:
        pixel = white
    else:
        pixel = black
```

**Characteristics**:
- Simplest reduction technique
- Loses all gray tones
- High contrast silhouettes
- Global threshold often fails for varying lighting

**Variations**:
- **Global threshold**: Single threshold for entire image
- **Adaptive threshold**: Local thresholds based on neighborhood
- **Otsu's method**: Automatically find optimal threshold

**Best for**: Silhouettes, high-contrast images, text/documents

**Comparison to halftone**: Halftone preserves tones; thresholding loses them entirely

---

## 2. Posterization (Color Quantization)

**Description**: Reduce number of colors to create flat, poster-like regions.

**Algorithm**:
```
colors_per_channel = N  # e.g., 4
step = 256 / N
for each pixel:
    r = round(r / step) * step
    g = round(g / step) * step
    b = round(b / step) * step
```

**Characteristics**:
- Flat color regions, no gradients
- Andy Warhol pop art style
- Cartoon/anime aesthetic
- Maintains color richness (just fewer colors)

**Typical palettes**:
- 8 colors (2 per channel): Very flat, graphic
- 27 colors (3 per channel): Balanced
- 64 colors (4 per channel): Subtle posterization

**Best for**: Pop art effects, retro posters, graphic design, color simplification

**Comparison to halftone**: Posterization keeps flat colors; halftone uses dots to simulate continuous tone

---

## 3. Edge Detection (Sketch Style)

**Description**: Detect edges to create line drawings/sketches.

**Common algorithms**:
- **Sobel**: Gradient-based edge detection
- **Canny**: Multi-stage edge detection (best quality)
- **Laplacian**: Second derivative edge detection
- **Scharr**: Improved Sobel kernel

**Characteristics**:
- Loses fill/color, keeps outlines
- Looks like pencil sketch or technical drawing
- Sensitive to noise
- Various line thicknesses possible

**Best for**: Technical drawings, sketches, artistic outlines, blueprint style

**Comparison to halftone**: Edge detection shows structure only; halftone preserves tones

---

## 4. Mosaic/Pixelation

**Description**: Divide into blocks and average each block.

**Algorithm**:
```
for each block:
    avg_color = average of all pixels in block
    set entire block to avg_color
```

**Characteristics**:
- Chunky pixel art effect
- Like looking through a grid
- 8-bit video game aesthetic
- Adjustable block size for more/less detail

**Best for**: Retro gaming aesthetic, censoring/privacy, artistic abstraction

**Comparison to halftone**: Pixelation averages regions; halftone uses patterns within regions

---

## 5. ASCII Art

**Description**: Represent image using ASCII characters based on brightness.

**Character mapping** (dark to light):
```
@%#*+=-:.
```

**Algorithm**:
```
for each block:
    avg_brightness = average brightness
    char = brightness_to_char(avg_brightness)
    output char
```

**Characteristics**:
- Text-based representation
- Terminal/retro computer aesthetic
- Variable density based on character choice
- Monospace font required for proper alignment

**Best for**: Terminal art, retro computer style, text-only displays

**Comparison to halftone**: ASCII uses characters; halftone uses geometric shapes

---

## 6. Oil Painting Effect

**Description**: Smooth into brush stroke-like regions.

**Algorithm**:
- For each pixel, find most common color in neighborhood
- Creates smooth, painterly regions
- Like artistic filter in Photoshop

**Characteristics**:
- Reduces detail but keeps color richness
- Artistic filter appearance
- Not technically "reduction" (may increase file size)

**Best for**: Artistic photos, painterly effects

**Comparison to halftone**: Oil painting smooths; halftone adds patterns

---

## 7. Palette Reduction (Indexed Color)

**Description**: Limit to specific historical or themed color palette.

**Popular palettes**:
- **Commodore 64**: 16 colors
- **NES**: 56 colors
- **Game Boy**: 4 shades of green
- **CGA**: 4 colors (magenta, cyan, white, black)
- **EGA**: 16 colors

**Algorithm**:
- For each pixel, find nearest color in palette
- Often uses dithering to smooth transitions

**Best for**: Retro gaming aesthetics, historical accuracy, nostalgia

**Comparison to halftone**: Palette reduction limits colors; often combined with dithering (which is a halftone technique)

---

## 8. Duotone/Tritone

**Description**: Map grayscale to 2-3 specific colors.

**Algorithm**:
```
grayscale = convert_to_grayscale(image)
for each pixel:
    map brightness to gradient between color1 and color2
```

**Characteristics**:
- Classic poster look
- Often black + one accent color
- Vintage aesthetic
- Like two-color screen printing

**Popular combinations**:
- Black + orange (vintage)
- Black + cyan (modern)
- Sepia tones (nostalgic)

**Best for**: Vintage posters, album covers, artistic photos

**Comparison to halftone**: Duotone maps to colors; halftone creates patterns

---

## 9. Contour/Isoline

**Description**: Like topographic map contour lines.

**Algorithm**:
- Group similar brightness values into bands
- Outline each band
- Creates layered appearance

**Characteristics**:
- Geographic/scientific visualization style
- Shows tonal regions clearly
- Adjustable number of contours
- Resembles elevation maps

**Best for**: Scientific visualization, geographic style, abstract art

**Comparison to halftone**: Contours show discrete bands; halftone shows continuous gradients via dots

---

## 10. Voronoi/Cell Shading

**Description**: Divide image into Voronoi cells with solid colors.

**Algorithm**:
- Sample points from image
- Create Voronoi diagram
- Fill each cell with averaged color

**Characteristics**:
- Stained glass effect
- Low-poly aesthetic
- Abstract, geometric
- Variable cell density

**Best for**: Stained glass effects, abstract art, low-poly style

**Comparison to halftone**: Voronoi uses irregular polygons; halftone uses regular dots

---

## Comparison Table

| Technique | Preserves Tones | Color | Pattern | Output Type | Best Use |
|-----------|----------------|-------|---------|-------------|----------|
| **Halftone** | ✓ (via patterns) | Yes (CMYK) | Dots/lines | Bitmap | Printing simulation |
| **Threshold** | ✗ | B&W only | None | Bitmap | Silhouettes |
| **Posterization** | ✗ | Limited | None | Bitmap | Pop art |
| **Edge Detection** | ✗ | B&W only | Lines | Bitmap | Sketches |
| **Mosaic** | ✓ (averaged) | Yes | Blocks | Bitmap | Retro gaming |
| **ASCII** | ✓ (via chars) | B&W only | Characters | Text | Terminal art |
| **Oil Painting** | ✓ | Yes | Brush strokes | Bitmap | Artistic |
| **Palette** | ✓ | Limited | Optional | Bitmap | Retro gaming |
| **Duotone** | ✓ | 2-3 colors | None | Bitmap | Vintage posters |
| **Contour** | ✗ (bands) | Yes | Lines | Bitmap | Scientific viz |
| **Voronoi** | ✓ (averaged) | Yes | Polygons | Bitmap | Stained glass |

## When to Use Each

- **Need accurate tone reproduction?** → Halftone (sqrt variant)
- **Want artistic/stylized?** → Halftone (linear), Posterization, Oil Painting
- **Need small file size?** → Threshold, Palette Reduction, Line Screen
- **Retro computer aesthetic?** → Floyd-Steinberg dithering, Palette Reduction
- **Printing simulation?** → CMYK Halftone
- **Technical drawings?** → Edge Detection, Crosshatch
- **Abstract art?** → Stippling, Voronoi, Contour

## Combining Techniques

Techniques can be combined:
- **Palette + Dithering**: Classic retro game look
- **Edge + Halftone**: Outlined halftone illustration
- **Posterization + Oil Painting**: Smooth poster effect
- **Duotone + Halftone**: Two-color halftone print

## Future Possibilities

Not yet implemented but related:
- **Weighted Voronoi Stippling**: Advanced stippling algorithm
- **Blue Noise Dithering**: Modern dithering with optimal distribution
- **Stochastic Screening**: Modern printing technique (FM screening)
- **Atkinson Dithering**: Apple ImageWriter algorithm
- **Neural Style Transfer**: AI-based stylization
- **Vector Quantization**: Machine learning-based palette reduction

## References

- **Halftone**: This library implements 8 halftone styles
- **Other techniques**: Would require separate implementations
- **Combinations**: Could be added as additional styles or post-processing
