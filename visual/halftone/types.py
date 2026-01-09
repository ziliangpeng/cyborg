"""Type definitions for halftone processing."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class StyleType(Enum):
    """Halftone style types."""

    CMYK = auto()
    GRAYSCALE_SQRT = auto()
    GRAYSCALE_LINEAR = auto()
    FLOYD_STEINBERG = auto()
    ORDERED_DITHER = auto()
    STIPPLING = auto()
    LINE_SCREEN = auto()
    CROSSHATCH = auto()


@dataclass
class CmykParams:
    """Parameters for CMYK halftone."""

    sample: int = 8
    scale: int = 1
    angles: Optional[tuple[float, float, float, float]] = None  # C, M, Y, K angles

    def get_angles(self) -> tuple[float, float, float, float]:
        """Get angles with defaults if not specified."""
        return self.angles if self.angles else (0.0, 15.0, 30.0, 45.0)


@dataclass
class GrayscaleParams:
    """Parameters for grayscale halftone."""

    sample: int = 8
    scale: int = 1
    angle: float = 0.0


@dataclass
class DitherParams:
    """Parameters for dithering."""

    matrix_size: int = 4  # 2, 4, or 8 for Bayer


@dataclass
class StipplingParams:
    """Parameters for stippling."""

    cell_size: int = 8
    density: float = 0.015  # Adjusted for cell_size² formula
    randomness: float = 0.3


@dataclass
class LineScreenParams:
    """Parameters for line screen."""

    angle: float = 45.0
    frequency: int = 8


@dataclass
class CrosshatchParams:
    """Parameters for crosshatch."""

    angle1: float = 45.0
    angle2: float = -45.0
    frequency: int = 8


@dataclass
class ProcessParams:
    """Global processing parameters."""

    antialias: bool = True


# Style name mapping
STYLE_NAMES = {
    StyleType.CMYK: "cmyk",
    StyleType.GRAYSCALE_SQRT: "grayscale-sqrt",
    StyleType.GRAYSCALE_LINEAR: "grayscale-linear",
    StyleType.FLOYD_STEINBERG: "floyd-steinberg",
    StyleType.ORDERED_DITHER: "ordered-dither",
    StyleType.STIPPLING: "stippling",
    StyleType.LINE_SCREEN: "line-screen",
    StyleType.CROSSHATCH: "crosshatch",
}

# Reverse mapping
NAME_TO_STYLE = {v: k for k, v in STYLE_NAMES.items()}


def get_style_name(style_type: StyleType) -> str:
    """Get CLI-friendly style name."""
    return STYLE_NAMES[style_type]


def parse_style_name(name: str) -> StyleType:
    """Parse style name to StyleType."""
    if name not in NAME_TO_STYLE:
        valid_names = ", ".join(STYLE_NAMES.values())
        raise ValueError(f"Invalid style name: {name}. Valid names: {valid_names}")
    return NAME_TO_STYLE[name]


def all_style_names() -> list[str]:
    """List all available style names."""
    return list(STYLE_NAMES.values())
