"""CyberVision - Real-time GPU-accelerated computer vision web app."""

from __future__ import annotations

import click

from apps.cybervision.server import run_server


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8080, help="Port to bind to")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(host: str, port: int, debug: bool) -> None:
    """Start the CyberVision web server."""
    run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
