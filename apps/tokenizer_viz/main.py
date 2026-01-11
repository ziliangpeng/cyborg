#!/usr/bin/env python3
"""A web UI for tokenizer visualization."""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenizer visualization web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8081, type=int, help="Bind port (default: 8081)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose request logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from apps.tokenizer_viz.server import run_server
    except ModuleNotFoundError as e:
        raise SystemExit("Missing dependencies. Run via Bazel: `bazel run //apps/tokenizer_viz:tokenizer_viz`") from e

    run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
