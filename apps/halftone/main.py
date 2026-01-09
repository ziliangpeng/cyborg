#!/usr/bin/env python3
"""Halftone image processor CLI."""

import click
from pathlib import Path
from PIL import Image
import sys

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from halftone import (
    StyleType, ProcessParams,
    CmykParams, GrayscaleParams, DitherParams,
    StipplingParams, LineScreenParams, CrosshatchParams,
    process, process_multiple,
    parse_style_name, all_style_names
)
from image_grid import GridLayout, LabeledImage, create_grid


@click.command()
@click.argument('input', type=click.Path(exists=True), required=False)
@click.option('--style', type=str, help='Single style to apply')
@click.option('--styles', type=str, help='Comma-separated styles or "all"')
@click.option('-o', '--output', type=click.Path(), help='Output path (single style only)')
@click.option('--no-antialias', is_flag=True, help='Disable antialiasing')
@click.option('--sample', type=int, default=8, help='Sample size (dot-based styles)')
@click.option('--scale', type=int, default=1, help='Output scale multiplier')
@click.option('--angle', type=float, default=45.0, help='Angle (line/screen styles)')
@click.option('--density', type=float, default=1.0, help='Density (stippling)')
@click.option('--frequency', type=int, default=8, help='Frequency (line styles)')
@click.option('--matrix-size', type=int, default=4, help='Matrix size (Bayer: 2, 4, 8)')
@click.option('--cell-size', type=int, default=8, help='Cell size (stippling)')
@click.option('--list-styles', is_flag=True, help='List available styles')
def main(input, style, styles, output, no_antialias, sample, scale, angle,
         density, frequency, matrix_size, cell_size, list_styles):
    """Halftone image processor."""

    if list_styles:
        click.echo("Available styles:")
        for name in all_style_names():
            click.echo(f"  - {name}")
        return

    # Require input if not listing styles
    if not input:
        raise click.UsageError("INPUT argument is required (unless using --list-styles)")

    # Validate mutually exclusive options
    if style and styles:
        raise click.UsageError("Cannot use both --style and --styles")

    if not style and not styles:
        raise click.UsageError("Must specify either --style or --styles")

    # Load image
    try:
        img = Image.open(input)
    except (FileNotFoundError, OSError, Image.UnidentifiedImageError) as e:
        click.echo(f"Error loading image: {e}", err=True)
        sys.exit(1)

    input_path = Path(input)
    base_name = input_path.stem
    ext = ".png"  # Always output PNG

    params = ProcessParams(antialias=not no_antialias)

    # Single style mode
    if style:
        try:
            style_type = parse_style_name(style)
        except ValueError as e:
            click.echo(str(e), err=True)
            sys.exit(1)

        # Create appropriate params for style
        style_params = _create_style_params(
            style_type, sample, scale, angle, density,
            frequency, matrix_size, cell_size
        )

        # Process image
        try:
            result = process(img, style_type, style_params, params)
        except (ValueError, OSError) as e:
            click.echo(f"Error processing image: {e}", err=True)
            sys.exit(1)

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / f"{base_name}_{style}{ext}"

        # Save result
        try:
            result.save(output_path)
            click.echo(f"Saved: {output_path}")
        except (OSError, ValueError) as e:
            click.echo(f"Error saving image: {e}", err=True)
            sys.exit(1)

    # Multiple styles mode
    elif styles:
        # Parse style list
        if styles.lower() == "all":
            style_names = all_style_names()
        else:
            style_names = [s.strip() for s in styles.split(',')]

        # Parse style types and create params
        style_list = []
        for style_name in style_names:
            try:
                style_type = parse_style_name(style_name)
                style_params = _create_style_params(
                    style_type, sample, scale, angle, density,
                    frequency, matrix_size, cell_size
                )
                style_list.append((style_type, style_params))
            except ValueError as e:
                click.echo(f"Warning: Skipping invalid style '{style_name}': {e}", err=True)

        if not style_list:
            click.echo("No valid styles to process", err=True)
            sys.exit(1)

        # Process all styles
        try:
            results = process_multiple(img, style_list, params)
        except (ValueError, OSError) as e:
            click.echo(f"Error processing images: {e}", err=True)
            sys.exit(1)

        # Save individual outputs
        for style_name, result_img in results.items():
            output_path = input_path.parent / f"{base_name}_{style_name}{ext}"
            try:
                result_img.save(output_path)
                click.echo(f"Saved: {output_path}")
            except (OSError, ValueError) as e:
                click.echo(f"Error saving {style_name}: {e}", err=True)

        # Create comparison grid
        try:
            labeled_images = [
                LabeledImage(image=img, label=name)
                for name, img in results.items()
            ]
            layout = GridLayout.auto(len(labeled_images))
            grid_img = create_grid(labeled_images, layout)

            comparison_path = input_path.parent / f"{base_name}_comparison{ext}"
            grid_img.save(comparison_path)
            click.echo(f"Saved comparison: {comparison_path}")
        except (ValueError, OSError) as e:
            click.echo(f"Error creating comparison grid: {e}", err=True)


def _create_style_params(
    style_type: StyleType,
    sample: int,
    scale: int,
    angle: float,
    density: float,
    frequency: int,
    matrix_size: int,
    cell_size: int
):
    """Create appropriate parameter object for style type."""
    if style_type == StyleType.CMYK:
        return CmykParams(sample=sample, scale=scale)

    elif style_type in (StyleType.GRAYSCALE_SQRT, StyleType.GRAYSCALE_LINEAR):
        return GrayscaleParams(sample=sample, scale=scale, angle=angle)

    elif style_type in (StyleType.FLOYD_STEINBERG, StyleType.ORDERED_DITHER):
        return DitherParams(matrix_size=matrix_size)

    elif style_type == StyleType.STIPPLING:
        return StipplingParams(cell_size=cell_size, density=density)

    elif style_type == StyleType.LINE_SCREEN:
        return LineScreenParams(angle=angle, frequency=frequency)

    elif style_type == StyleType.CROSSHATCH:
        return CrosshatchParams(angle1=angle, angle2=-angle, frequency=frequency)

    else:
        raise ValueError(f"Unknown style type: {style_type}")


if __name__ == '__main__':
    main()
