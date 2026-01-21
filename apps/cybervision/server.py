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

# Find the shared library location (Bazel runfiles)
_MODULE_DIR = Path(__file__).parent
SHARED_LIB_ROOT = _MODULE_DIR.parent.parent / "libs" / "cybervision-core"


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

    def _send_file(self, path: Path, allowed_base: Path | None = None) -> None:
        if not path.exists() or not path.is_file():
            logging.debug(f"Path not found: {path}")
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # Security check: verify path is within allowed base
        if allowed_base:
            # Use absolute() instead of resolve() to avoid following symlinks
            # Bazel uses symlinks in runfiles, so resolve() would break the path check
            base = allowed_base.absolute()
            target = path.absolute()
            logging.debug(f"Base: {base}, Target: {target}")
            try:
                target.relative_to(base)
            except ValueError as e:
                logging.debug(f"Path traversal check failed: {e}")
                self.send_error(HTTPStatus.FORBIDDEN)
                return

        ctype, _ = mimetypes.guess_type(str(path))
        content_type = ctype or "application/octet-stream"

        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_bytes(HTTPStatus.OK, content_type, data)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_file(WEBROOT / "index.html", allowed_base=WEBROOT)
            return

        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            safe_path = _safe_join(STATIC_ROOT, rel)
            if safe_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(safe_path, allowed_base=WEBROOT)
            return

        if path.startswith("/shaders/"):
            rel = path.removeprefix("/shaders/")
            safe_path = _safe_join(SHARED_LIB_ROOT / "shaders", rel)
            if safe_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            # For shared lib, check both original location and Bazel runfiles
            if not safe_path.exists():
                # Try Bazel runfiles location
                runfiles_path = (
                    _MODULE_DIR
                    / "cybervision.runfiles"
                    / "_main"
                    / "libs"
                    / "cybervision-core"
                    / "shaders"
                    / rel
                )
                if runfiles_path.exists():
                    safe_path = runfiles_path
            self._send_file(safe_path, allowed_base=None)  # Shared lib, less strict check
            return

        if path.startswith("/lib/"):
            rel = path.removeprefix("/lib/")
            safe_path = _safe_join(SHARED_LIB_ROOT, rel)
            if safe_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            # For shared lib, check both original location and Bazel runfiles
            if not safe_path.exists():
                runfiles_path = (
                    _MODULE_DIR / "cybervision.runfiles" / "_main" / "libs" / "cybervision-core" / rel
                )
                if runfiles_path.exists():
                    safe_path = runfiles_path
            self._send_file(safe_path, allowed_base=None)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def run_server(*, host: str, port: int, debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logging.info(f"WEBROOT: {WEBROOT.resolve()}")
    logging.info(f"WEBROOT exists: {WEBROOT.exists()}")
    logging.info(f"Shared lib root: {SHARED_LIB_ROOT.resolve()}")
    with ThreadingHTTPServer((host, port), CyberVisionHandler) as httpd:
        logging.info("Listening on http://%s:%d", host, port)
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logging.info("Shutting down")
