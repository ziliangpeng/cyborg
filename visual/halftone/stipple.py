"""Stippling implementation."""

from PIL import Image, ImageDraw
import random
from .common import sample_region
from .types import StipplingParams, ProcessParams


def process_stippling(image: Image.Image, params: StipplingParams, process_params: ProcessParams) -> Image.Image:
    """
    Stochastic stippling - random dot placement.
    Creates hand-drawn/artistic effects.

    Args:
        image: Input PIL Image
        params: Stippling parameters
        process_params: Processing parameters

    Returns:
        Stippled PIL Image
    """
    gray = image.convert("L")
    width, height = gray.size

    cell_size = params.cell_size
    density = params.density
    randomness = params.randomness

    # Create output
    output = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(output)

    # Sample and place dots
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            # Get region brightness
            avg_brightness = sample_region(gray, x, y, cell_size)

            # Calculate number of dots based on darkness
            # Use cell area (cell_size²) to maintain consistent density
            darkness = 1 - (avg_brightness / 255)
            num_dots = int(darkness * (cell_size**2) * density)

            # Place random dots in cell
            for _ in range(num_dots):
                # Random position with some structure
                offset_x = random.uniform(-randomness * cell_size, (1 + randomness) * cell_size)
                offset_y = random.uniform(-randomness * cell_size, (1 + randomness) * cell_size)

                dot_x = x + cell_size / 2 + offset_x
                dot_y = y + cell_size / 2 + offset_y

                # Random dot size (small variation)
                dot_radius = random.uniform(0.8, 1.5)

                draw.ellipse([dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius], fill=0)

    return output
