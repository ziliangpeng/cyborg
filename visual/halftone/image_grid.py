"""Image grid layout library for arranging multiple images with labels."""

from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


@dataclass
class GridLayout:
    """Grid layout configuration."""
    cols: int
    padding: int = 20
    label_height: int = 60
    background_color: tuple[int, int, int] = (255, 255, 255)

    @classmethod
    def auto(cls, num_images: int) -> 'GridLayout':
        """Auto-determine optimal grid layout for N images."""
        cols = cls._optimal_columns(num_images)
        return cls(cols=cols)

    @staticmethod
    def _optimal_columns(n: int) -> int:
        """
        Calculate optimal number of columns for N images.
        Prefers landscape layouts (wider than tall).

        Examples:
          1-3 images -> n columns (single row)
          4-9 images -> 3 columns (2-3 rows)
          10-16 images -> 4 columns (3-4 rows)
          17+ images -> 5 columns
        """
        if n <= 3:
            return max(1, n)
        elif n <= 9:
            return 3
        elif n <= 16:
            return 4
        else:
            return 5


@dataclass
class LabeledImage:
    """Image with a text label."""
    image: Image.Image
    label: str


def resize_to_fit(
    image: Image.Image,
    max_width: int,
    max_height: int
) -> Image.Image:
    """
    Resize image to fit within dimensions while preserving aspect ratio.

    Args:
        image: PIL Image to resize
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized PIL Image
    """
    # Calculate aspect-preserving dimensions
    img_width, img_height = image.size
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    ratio = min(width_ratio, height_ratio)

    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def create_grid(
    images: list[LabeledImage],
    layout: GridLayout
) -> Image.Image:
    """
    Create a grid of labeled images.

    Args:
        images: List of LabeledImage objects
        layout: GridLayout configuration

    Returns:
        PIL Image containing the grid
    """
    if not images:
        raise ValueError("Must provide at least one image")

    cols = layout.cols
    rows = (len(images) + cols - 1) // cols

    # Find maximum dimensions among all images
    max_img_width = max(img.image.width for img in images)
    max_img_height = max(img.image.height for img in images)

    # Cell dimensions
    cell_width = max_img_width
    cell_height = max_img_height + layout.label_height

    # Grid dimensions
    grid_width = cols * cell_width + (cols + 1) * layout.padding
    grid_height = rows * cell_height + (rows + 1) * layout.padding

    # Create canvas
    grid = Image.new('RGB', (grid_width, grid_height), layout.background_color)
    draw = ImageDraw.Draw(grid)

    # Load font (portable - works on all platforms with Pillow 10+)
    font = ImageFont.load_default(size=24)

    # Place images and labels
    for idx, labeled_img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Calculate cell position
        x = layout.padding + col * (cell_width + layout.padding)
        y = layout.padding + row * (cell_height + layout.padding)

        # Resize image to fit cell (preserving aspect ratio)
        img = resize_to_fit(labeled_img.image, cell_width, max_img_height)

        # Center image in cell
        img_x = x + (cell_width - img.width) // 2
        img_y = y

        # Paste image
        grid.paste(img, (img_x, img_y))

        # Draw label below image
        label_y = y + max_img_height + 10

        # Center text
        bbox = draw.textbbox((0, 0), labeled_img.label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (cell_width - text_width) // 2

        draw.text((text_x, label_y), labeled_img.label, fill=(0, 0, 0), font=font)

    return grid
