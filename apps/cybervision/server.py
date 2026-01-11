"""Simple HTTP server for CyberVision."""

from __future__ import annotations

import logging
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

WEBROOT = Path(__file__).parent / "webroot"
STATIC_ROOT = WEBROOT / "static"


def _safe_join(root: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel)
    if norm.startswith("..") or os.path.isabs(norm):
        return None
    return root / norm


class CyberVisionHandler(BaseHTTPRequestHandler):
    server_version = "CyberVision/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        base = WEBROOT.resolve()
        target = path.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        ctype, _ = mimetypes.guess_type(str(target))
        content_type = ctype or "application/octet-stream"

        try:
            data = target.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_bytes(HTTPStatus.OK, content_type, data)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_file(WEBROOT / "index.html")
            return
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            safe_path = _safe_join(STATIC_ROOT, rel)
            if safe_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(safe_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def run_server(*, host: str, port: int, debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    with ThreadingHTTPServer((host, port), CyberVisionHandler) as httpd:
        logging.info("Listening on http://%s:%d", host, port)
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logging.info("Shutting down")
