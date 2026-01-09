"""CMYK halftone implementation."""

from PIL import Image, ImageDraw, ImageStat
from .common import sample_region, rotate_image, crop_to_size
from .types import CmykParams, ProcessParams


def rgb_to_cmyk(image: Image.Image) -> Image.Image:
    """
    Convert RGB image to CMYK.

    Args:
        image: RGB PIL Image

    Returns:
        CMYK PIL Image
    """
    return image.convert('CMYK')


def process_cmyk(
    image: Image.Image,
    params: CmykParams,
    process_params: ProcessParams
) -> Image.Image:
    """
    CMYK 4-color halftone (printing press simulation).
    Separates image into CMYK channels, processes each with rotated screen.

    Args:
        image: Input PIL Image
        params: CMYK parameters
        process_params: Processing parameters

    Returns:
        Halftoned PIL Image as RGB
    """
    # Convert to CMYK
    cmyk = rgb_to_cmyk(image)
    original_size = image.size

    sample = params.sample
    scale = params.scale
    angles = params.get_angles()

    # Apply antialiasing if requested
    antialias_scale = 4 if process_params.antialias else 1
    if process_params.antialias:
        scale = scale * antialias_scale

    # Split into channels
    channels = cmyk.split()

    # Process each channel
    processed_channels = []
    for channel, angle in zip(channels, angles):
        processed = _process_channel(channel, sample, scale, angle, original_size)
        processed_channels.append(processed)

    # Merge back to CMYK
    result = Image.merge('CMYK', processed_channels)

    # Scale back down if antialiasing
    if process_params.antialias:
        final_width = original_size[0] * params.scale
        final_height = original_size[1] * params.scale
        result = result.resize((final_width, final_height), Image.Resampling.LANCZOS)

    # Convert to RGB for display
    return result.convert('RGB')


def _process_channel(
    channel: Image.Image,
    sample: int,
    scale: int,
    angle: float,
    original_size: tuple[int, int]
) -> Image.Image:
    """
    Process a single CMYK channel with halftone.

    Args:
        channel: Single channel as grayscale image
        sample: Sample size
        scale: Output scale
        angle: Rotation angle for this channel
        original_size: Original image size

    Returns:
        Processed channel
    """
    # Rotate channel
    rotated = rotate_image(channel, angle)
    width, height = rotated.size

    # Create output
    output_width = width * scale
    output_height = height * scale
    output = Image.new('L', (output_width, output_height), 0)  # Black background
    draw = ImageDraw.Draw(output)

    # Draw dots
    for y in range(0, height, sample):
        for x in range(0, width, sample):
            avg_intensity = sample_region(rotated, x, y, sample)

            # Square root scaling
            diameter_ratio = (avg_intensity / 255) ** 0.5

            box_size = sample * scale
            draw_diameter = diameter_ratio * box_size

            if draw_diameter > 0.5:
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

    return output
