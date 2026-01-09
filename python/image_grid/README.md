# Image Grid Library

A reusable Python library for creating grids of labeled images with automatic layout optimization.

## Features

- **Automatic layout**: Intelligently determines optimal grid dimensions
- **Labeled images**: Add text labels below each image
- **Aspect-preserving resize**: Images are resized to fit cells while maintaining aspect ratio
- **Configurable**: Control padding, label height, and background color

## Usage

```python
from image_grid import GridLayout, LabeledImage, create_grid
from PIL import Image

# Load images
img1 = Image.open("image1.jpg")
img2 = Image.open("image2.jpg")
img3 = Image.open("image3.jpg")

# Create labeled images
labeled_images = [
    LabeledImage(image=img1, label="Style 1"),
    LabeledImage(image=img2, label="Style 2"),
    LabeledImage(image=img3, label="Style 3"),
]

# Create grid with automatic layout
layout = GridLayout.auto(len(labeled_images))
grid = create_grid(labeled_images, layout)

# Save result
grid.save("comparison.png")
```

## Layout Algorithm

The library automatically determines optimal column count based on the number of images:

- **1-3 images**: Single row (N columns)
- **4-9 images**: 3 columns (2-3 rows)
- **10-16 images**: 4 columns (3-4 rows)
- **17+ images**: 5 columns

This creates landscape-oriented grids that are wider than they are tall.

## Custom Layout

```python
# Manually specify grid configuration
layout = GridLayout(
    cols=4,                              # 4 columns
    padding=30,                          # 30px padding
    label_height=80,                     # 80px for labels
    background_color=(240, 240, 240)     # Light gray background
)

grid = create_grid(labeled_images, layout)
```

## Reusability

This library is designed to be reusable across the cyborg repository for any project that needs to display multiple images in a grid format.
