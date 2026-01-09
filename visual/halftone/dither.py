"""Dithering implementations (Floyd-Steinberg and Bayer)."""

import numpy as np
from PIL import Image

from .types import DitherParams, ProcessParams


def process_floyd_steinberg(image: Image.Image, params: DitherParams, process_params: ProcessParams) -> Image.Image:
    """
    Floyd-Steinberg error diffusion dithering.
    Creates organic, flowing dot patterns.

    Args:
        image: Input PIL Image
        params: Dither parameters
        process_params: Processing parameters

    Returns:
        Dithered PIL Image
    """
    gray = image.convert("L")
    pixels = np.array(gray, dtype=np.float32)
    height, width = pixels.shape

    for y in range(height):
        for x in range(width):
            old_pixel = pixels[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            pixels[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Distribute error to neighbors
            if x + 1 < width:
                pixels[y, x + 1] += error * 7 / 16
            if y + 1 < height:
                if x > 0:
                    pixels[y + 1, x - 1] += error * 3 / 16
                pixels[y + 1, x] += error * 5 / 16
                if x + 1 < width:
                    pixels[y + 1, x + 1] += error * 1 / 16

    result = Image.fromarray(pixels.astype(np.uint8), "L")
    return result


def process_ordered_dither(image: Image.Image, params: DitherParams, process_params: ProcessParams) -> Image.Image:
    """
    Ordered/Bayer dithering with threshold matrix.
    Creates regular checkerboard-like patterns.

    Args:
        image: Input PIL Image
        params: Dither parameters (matrix_size: 2, 4, or 8)
        process_params: Processing parameters

    Returns:
        Dithered PIL Image
    """
    # Bayer matrices
    bayer_2x2 = np.array([[0, 2], [3, 1]]) / 4.0

    bayer_4x4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0

    bayer_8x8 = (
        np.array(
            [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ]
        )
        / 64.0
    )

    # Select matrix
    matrix_map = {2: bayer_2x2, 4: bayer_4x4, 8: bayer_8x8}
    if params.matrix_size not in matrix_map:
        raise ValueError(f"Invalid matrix_size: {params.matrix_size}. Must be 2, 4, or 8")

    threshold_matrix = matrix_map[params.matrix_size]

    gray = image.convert("L")
    pixels = np.array(gray, dtype=np.float32) / 255.0
    height, width = pixels.shape

    result = np.zeros_like(pixels)
    m_height, m_width = threshold_matrix.shape

    for y in range(height):
        for x in range(width):
            threshold = threshold_matrix[y % m_height, x % m_width]
            result[y, x] = 255 if pixels[y, x] > threshold else 0

    output = Image.fromarray(result.astype(np.uint8), "L")
    return output
