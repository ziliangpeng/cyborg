"""Common utilities for halftone processing."""

from PIL import Image, ImageDraw, ImageStat


def sample_region(image: Image.Image, x: int, y: int, size: int) -> float:
    """
    Sample a region of the image and return average brightness.

    Args:
        image: Grayscale PIL Image
        x, y: Top-left coordinates of region
        size: Size of region

    Returns:
        Average brightness (0-255)
    """
    width, height = image.size
    x_end = min(x + size, width)
    y_end = min(y + size, height)

    box = image.crop((x, y, x_end, y_end))
    return ImageStat.Stat(box).mean[0]


def draw_circle(draw: ImageDraw.Draw, center_x: float, center_y: float, radius: float, fill: int):
    """
    Draw a circle on the ImageDraw object.

    Args:
        draw: PIL ImageDraw object
        center_x, center_y: Center coordinates
        radius: Circle radius
        fill: Fill color (0-255)
    """
    if radius > 0:
        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill=fill
        )


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotate image by angle, expanding canvas to fit.

    Args:
        image: PIL Image
        angle: Rotation angle in degrees

    Returns:
        Rotated PIL Image
    """
    return image.rotate(angle, expand=True, fillcolor=0)


def crop_to_size(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Crop image to target size from center.

    Args:
        image: PIL Image
        target_width, target_height: Target dimensions

    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))
