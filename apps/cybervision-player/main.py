"""CyberVision Video Player - Entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Add the app directory to the Python path so we can import server
sys.path.insert(0, str(Path(__file__).parent))

import server


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8081, help="Port to listen on")
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(host: str, port: int, debug: bool) -> None:
    """Run the CyberVision Video Player server."""
    server.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
