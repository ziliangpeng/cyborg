"""Line screen and crosshatch implementations."""

from PIL import Image, ImageDraw
import numpy as np
from .types import LineScreenParams, CrosshatchParams, ProcessParams

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

    # Create larger output for rotation
    diagonal = int(np.sqrt(width**2 + height**2))
    output = Image.new("L", (diagonal, diagonal), 255)
    draw = ImageDraw.Draw(output)

    # Draw lines
    angle_rad = np.radians(angle)

    y_pos = 0
    while y_pos < diagonal:
        # Sample line position to get average brightness
        line_samples = []
        for sample_x in range(0, width, LINE_SAMPLE_STEP):
            # Transform coordinates
            rot_x = int(sample_x * np.cos(angle_rad) + y_pos * np.sin(angle_rad))
            rot_y = int(-sample_x * np.sin(angle_rad) + y_pos * np.cos(angle_rad))

            if 0 <= rot_x < width and 0 <= rot_y < height:
                line_samples.append(gray.getpixel((rot_x, rot_y)))

        if line_samples:
            avg_brightness = sum(line_samples) / len(line_samples)
            darkness = 1 - (avg_brightness / 255)
            line_thickness = int(darkness * frequency * 0.8)

            if line_thickness > 0:
                draw.line([(0, y_pos), (diagonal, y_pos)], fill=0, width=line_thickness)

        y_pos += frequency

    # Rotate back
    output = output.rotate(-angle, expand=False, fillcolor=255)

    # Crop to original size
    left = (diagonal - width) // 2
    top = (diagonal - height) // 2
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
