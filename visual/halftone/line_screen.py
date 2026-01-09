"""Line screen and crosshatch implementations."""

import numpy as np
from PIL import Image, ImageDraw

from .types import CrosshatchParams, LineScreenParams, ProcessParams

# Sample every N pixels along line for performance (instead of every pixel)
LINE_SAMPLE_STEP = 10


def process_line_screen(image: Image.Image, params: LineScreenParams, process_params: ProcessParams) -> Image.Image:
    """
    Line screen halftone - parallel lines with varying thickness.
    Creates engraving-style effects.

    Args:
        image: Input PIL Image
        params: Line screen parameters
        process_params: Processing parameters

    Returns:
        Line screen PIL Image
    """
    gray = image.convert("L")
    width, height = gray.size

    angle = params.angle
    frequency = params.frequency

    # Simpler approach: rotate image, draw horizontal lines based on column averages, rotate back
    # Rotate image by angle
    rotated = gray.rotate(angle, expand=True, fillcolor=255)
    rot_width, rot_height = rotated.size

    # Create output
    output = Image.new("L", (rot_width, rot_height), 255)
    draw = ImageDraw.Draw(output)

    # Draw horizontal lines (will be at angle after rotation back)
    y_pos = 0
    while y_pos < rot_height:
        # Sample this horizontal line
        line_samples = []
        for x in range(0, rot_width, LINE_SAMPLE_STEP):
            if x < rot_width and y_pos < rot_height:
                line_samples.append(rotated.getpixel((x, y_pos)))

        if line_samples:
            avg_brightness = sum(line_samples) / len(line_samples)
            darkness = 1 - (avg_brightness / 255)
            line_thickness = max(1, int(darkness * frequency * 1.2))

            if line_thickness > 0:
                draw.line([(0, y_pos), (rot_width, y_pos)], fill=0, width=line_thickness)

        y_pos += frequency

    # Rotate back
    output = output.rotate(-angle, expand=True, fillcolor=255)

    # Crop to original size
    out_width, out_height = output.size
    left = (out_width - width) // 2
    top = (out_height - height) // 2
    output = output.crop((left, top, left + width, top + height))

    return output


def process_crosshatch(image: Image.Image, params: CrosshatchParams, process_params: ProcessParams) -> Image.Image:
    """
    Crosshatch - intersecting line screens.
    Creates technical drawing effects.

    Args:
        image: Input PIL Image
        params: Crosshatch parameters
        process_params: Processing parameters

    Returns:
        Crosshatch PIL Image
    """
    # Create two line screens with different angles
    line_params1 = LineScreenParams(angle=params.angle1, frequency=params.frequency)
    line_params2 = LineScreenParams(angle=params.angle2, frequency=params.frequency)

    line1 = process_line_screen(image, line_params1, process_params)
    line2 = process_line_screen(image, line_params2, process_params)

    # Combine them (darken where lines intersect)
    arr1 = np.array(line1)
    arr2 = np.array(line2)

    combined = np.minimum(arr1, arr2)

    result = Image.fromarray(combined.astype(np.uint8), "L")
    return result
