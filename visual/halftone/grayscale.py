"""Grayscale halftone implementations."""

from PIL import Image, ImageDraw

from .common import crop_to_size, draw_circle, rotate_image, sample_region
from .types import GrayscaleParams, ProcessParams


def process_linear(image: Image.Image, params: GrayscaleParams, process_params: ProcessParams) -> Image.Image:
    """
    Linear grayscale halftone (stylistic version).
    Black dots on white background, linear tone curve.
    Creates more dramatic, darker shadows.

    Args:
        image: Input PIL Image
        params: Grayscale parameters
        process_params: Processing parameters

    Returns:
        Halftoned PIL Image
    """
    gray = image.convert("L")
    width, height = gray.size

    dot_size = params.sample
    scale = params.scale

    output_width = width * scale
    output_height = height * scale
    output = Image.new("L", (output_width, output_height), 255)
    draw = ImageDraw.Draw(output)

    for y in range(0, height, dot_size):
        for x in range(0, width, dot_size):
            avg_brightness = sample_region(gray, x, y, dot_size)

            # Linear scaling: darker = larger dots
            max_radius = (dot_size * scale) / 2
            radius = max_radius * (1 - avg_brightness / 255)

            if radius > 0:
                center_x = (x + dot_size / 2) * scale
                center_y = (y + dot_size / 2) * scale
                draw_circle(draw, center_x, center_y, radius, fill=0)

    return output


def process_sqrt(image: Image.Image, params: GrayscaleParams, process_params: ProcessParams) -> Image.Image:
    """
    Square root grayscale halftone (accurate reproduction).
    White dots on black background, sqrt tone curve.
    Better midtone reproduction, closer to original brightness.

    Args:
        image: Input PIL Image
        params: Grayscale parameters
        process_params: Processing parameters

    Returns:
        Halftoned PIL Image
    """
    gray = image.convert("L")
    original_size = gray.size

    sample = params.sample
    scale = params.scale
    angle = params.angle

    # Apply antialiasing if requested
    antialias_scale = 4 if process_params.antialias else 1
    if process_params.antialias:
        scale = scale * antialias_scale

    # Rotate image
    rotated = rotate_image(gray, angle)
    width, height = rotated.size

    # Create output
    output_width = width * scale
    output_height = height * scale
    output = Image.new("L", (output_width, output_height), 0)  # Black background
    draw = ImageDraw.Draw(output)

    # Draw dots
    for y in range(0, height, sample):
        for x in range(0, width, sample):
            avg_brightness = sample_region(rotated, x, y, sample)

            # Square root scaling for accurate tone
            diameter_ratio = (avg_brightness / 255) ** 0.5

            box_size = sample * scale
            draw_diameter = diameter_ratio * box_size

            if draw_diameter > 0:
                box_x = x * scale
                box_y = y * scale

                x1 = box_x + (box_size - draw_diameter) / 2
                y1 = box_y + (box_size - draw_diameter) / 2
                x2 = x1 + draw_diameter
                y2 = y1 + draw_diameter

                draw.ellipse([(x1, y1), (x2, y2)], fill=255)  # White dots

    # Rotate back
    output = rotate_image(output, -angle)

    # Crop to original size (scaled)
    target_width = original_size[0] * scale
    target_height = original_size[1] * scale
    output = crop_to_size(output, target_width, target_height)

    # Scale back down if antialiasing
    if process_params.antialias:
        final_width = original_size[0] * params.scale
        final_height = original_size[1] * params.scale
        output = output.resize((final_width, final_height), Image.Resampling.LANCZOS)

    return output
