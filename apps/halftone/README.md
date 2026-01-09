# Halftone CLI

Command-line tool for applying halftone effects to images with 8 different styles.

## Installation

### Using Bazel

```bash
# From repository root
bazel build //apps/halftone:halftone
```

### Using Python venv (Development)

```bash
# From repository root
cd apps/halftone
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Working Directory

The `apps/halftone/img/` directory is used for test images and outputs:
- Place your input images here
- Processed images will be saved here
- This directory is gitignored (not committed to repo)

## Usage

### List Available Styles

```bash
bazel run //apps/halftone:halftone -- --list-styles
```

Available styles:
- `cmyk` - CMYK 4-color halftone (printing press)
- `grayscale-sqrt` - Grayscale with sqrt tone curve (accurate)
- `grayscale-linear` - Grayscale with linear tone curve (stylistic)
- `floyd-steinberg` - Floyd-Steinberg error diffusion dithering
- `ordered-dither` - Bayer ordered dithering
- `stippling` - Random dot stippling (hand-drawn effect)
- `line-screen` - Parallel line screen (engraving)
- `crosshatch` - Intersecting lines (technical drawing)

### Single Style

```bash
# Process with auto-generated filename
bazel run //apps/halftone:halftone -- input.jpg --style cmyk
# Output: input_cmyk.png

# Specify custom output
bazel run //apps/halftone:halftone -- input.jpg --style floyd-steinberg -o output.png
```

### Multiple Styles

```bash
# Process with comma-separated styles
bazel run //apps/halftone:halftone -- input.jpg --styles cmyk,grayscale-sqrt,floyd-steinberg
# Output: input_cmyk.png, input_grayscale-sqrt.png, input_floyd-steinberg.png, input_comparison.png
```

### All Styles

```bash
# Process with all 8 styles
bazel run //apps/halftone:halftone -- input.jpg --styles all
# Output: 8 individual style images + input_comparison.png
```

### Custom Parameters

```bash
# CMYK with custom sample size and scale
bazel run //apps/halftone:halftone -- input.jpg --style cmyk --sample 6 --scale 2

# Stippling with custom density
bazel run //apps/halftone:halftone -- input.jpg --style stippling --density 1.5 --cell-size 6

# Line screen with custom angle
bazel run //apps/halftone:halftone -- input.jpg --style line-screen --angle 30 --frequency 10

# Disable antialiasing for faster processing
bazel run //apps/halftone:halftone -- input.jpg --style cmyk --no-antialias
```

## Parameters

### Global Parameters

- `--no-antialias` - Disable antialiasing (4x faster, jagged edges)

### Dot-Based Styles (cmyk, grayscale-*)

- `--sample N` - Sample/dot grid size in pixels (default: 8)
  - Smaller = finer detail, more dots
  - Larger = coarser, fewer dots
- `--scale N` - Output scale multiplier (default: 1)

### Dithering Styles (floyd-steinberg, ordered-dither)

- `--matrix-size N` - Bayer matrix size: 2, 4, or 8 (default: 4)

### Stippling

- `--cell-size N` - Cell size in pixels (default: 8)
- `--density F` - Dot density multiplier (default: 1.0)

### Line Screens (line-screen, crosshatch)

- `--angle F` - Line angle in degrees (default: 45.0)
- `--frequency N` - Line frequency/spacing (default: 8)

## Examples

```bash
# Place your images in apps/halftone/img/
# Outputs will be saved in the same directory

# Classic newspaper look
bazel run //apps/halftone:halftone -- apps/halftone/img/portrait.jpg --style grayscale-sqrt --sample 8

# Fine CMYK halftone
bazel run //apps/halftone:halftone -- apps/halftone/img/photo.jpg --style cmyk --sample 4 --scale 2

# Artistic stippling
bazel run //apps/halftone:halftone -- apps/halftone/img/landscape.jpg --style stippling --density 1.2

# Engraving effect
bazel run //apps/halftone:halftone -- apps/halftone/img/drawing.jpg --style line-screen --angle 30

# Compare all styles
bazel run //apps/halftone:halftone -- apps/halftone/img/test.jpg --styles all
```

## Output

- Single style mode: Creates one output file
- Multiple styles mode: Creates individual files + comparison grid
- Output format: Always PNG (lossless)
- Naming: `<input>_<style>.png` for individual, `<input>_comparison.png` for grid

## Performance

Typical processing times for 1MP image (M4 Pro):
- Floyd-Steinberg: ~200ms
- Bayer: ~50ms
- CMYK: ~1-2s
- Stippling: ~500ms
- Line screens: ~150ms

Antialiasing adds 4x overhead to dot-based styles.
