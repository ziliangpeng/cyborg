"""Halftone image processing library."""

from typing import Any

from PIL import Image

from .cmyk import process_cmyk
from .dither import process_floyd_steinberg, process_ordered_dither
from .grayscale import process_linear, process_sqrt
from .line_screen import process_crosshatch, process_line_screen
from .stipple import process_stippling
from .types import (
    CmykParams,
    CrosshatchParams,
    DitherParams,
    GrayscaleParams,
    LineScreenParams,
    ProcessParams,
    StipplingParams,
    StyleType,
    all_style_names,
    get_style_name,
    parse_style_name,
)

# Re-export types for convenience
__all__ = [
    "StyleType",
    "CmykParams",
    "GrayscaleParams",
    "DitherParams",
    "StipplingParams",
    "LineScreenParams",
    "CrosshatchParams",
    "ProcessParams",
    "process",
    "process_multiple",
    "get_style_name",
    "parse_style_name",
    "all_style_names",
]


def process(
    image: Image.Image, style_type: StyleType, style_params: Any, params: ProcessParams = ProcessParams()
) -> Image.Image:
    """
    Process image with specified halftone style.

    Args:
        image: Input PIL Image
        style_type: Style to apply
        style_params: Style-specific parameters
        params: Global processing parameters

    Returns:
        Processed PIL Image

    Raises:
        ValueError: If style_type is invalid or params don't match style
    """
    if style_type == StyleType.CMYK:
        if not isinstance(style_params, CmykParams):
            raise ValueError("CMYK style requires CmykParams")
        return process_cmyk(image, style_params, params)

    elif style_type == StyleType.GRAYSCALE_SQRT:
        if not isinstance(style_params, GrayscaleParams):
            raise ValueError("Grayscale sqrt style requires GrayscaleParams")
        return process_sqrt(image, style_params, params)

    elif style_type == StyleType.GRAYSCALE_LINEAR:
        if not isinstance(style_params, GrayscaleParams):
            raise ValueError("Grayscale linear style requires GrayscaleParams")
        return process_linear(image, style_params, params)

    elif style_type == StyleType.FLOYD_STEINBERG:
        if not isinstance(style_params, DitherParams):
            raise ValueError("Floyd-Steinberg style requires DitherParams")
        return process_floyd_steinberg(image, style_params, params)

    elif style_type == StyleType.ORDERED_DITHER:
        if not isinstance(style_params, DitherParams):
            raise ValueError("Ordered dither style requires DitherParams")
        return process_ordered_dither(image, style_params, params)

    elif style_type == StyleType.STIPPLING:
        if not isinstance(style_params, StipplingParams):
            raise ValueError("Stippling style requires StipplingParams")
        return process_stippling(image, style_params, params)

    elif style_type == StyleType.LINE_SCREEN:
        if not isinstance(style_params, LineScreenParams):
            raise ValueError("Line screen style requires LineScreenParams")
        return process_line_screen(image, style_params, params)

    elif style_type == StyleType.CROSSHATCH:
        if not isinstance(style_params, CrosshatchParams):
            raise ValueError("Crosshatch style requires CrosshatchParams")
        return process_crosshatch(image, style_params, params)

    else:
        raise ValueError(f"Unknown style type: {style_type}")


def process_multiple(
    image: Image.Image, styles: list[tuple[StyleType, Any]], params: ProcessParams = ProcessParams()
) -> dict[str, Image.Image]:
    """
    Process image with multiple styles.

    Args:
        image: Input PIL Image
        styles: List of (StyleType, style_params) tuples
        params: Global processing parameters

    Returns:
        Dictionary mapping style name -> processed image
    """
    results = {}

    for style_type, style_params in styles:
        style_name = get_style_name(style_type)
        result_image = process(image, style_type, style_params, params)
        results[style_name] = result_image

    return results
