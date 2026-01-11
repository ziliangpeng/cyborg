#!/usr/bin/env python3
"""A small web UI for visual.halftone."""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Halftone web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8080, type=int, help="Bind port (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose request logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from apps.halftone_web.server import run_server
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependencies. Run via Bazel: "
            "`bazel run //apps/halftone_web:halftone_web`"
        ) from e

    run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
