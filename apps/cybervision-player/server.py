"""HTTP server for CyberVision Video Player with video streaming support."""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

WEBROOT = Path(__file__).parent / "webroot"
STATIC_ROOT = WEBROOT / "static"
MODELS_ROOT = WEBROOT / "models"

# Find the shared library location (Bazel runfiles)
_MODULE_DIR = Path(__file__).parent
SHARED_LIB_ROOT = _MODULE_DIR.parent.parent / "libs" / "cybervision-core"


def _safe_join(root: Path, rel: str) -> Path | None:
    """Safely join a relative path to a root, preventing path traversal."""
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel)
    if norm.startswith("..") or os.path.isabs(norm):
        return None
    return root / norm


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """Parse HTTP Range header and return (start, end) byte positions."""
    match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not match:
        return None

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else file_size - 1

    if start >= file_size or end >= file_size or start > end:
        return None

    return start, end


class VideoPlayerHandler(BaseHTTPRequestHandler):
    server_version = "CyberVisionPlayer/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send_bytes(self, status: int, content_type: str, data: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, allowed_base: Path | None = None) -> None:
        if not path.exists() or not path.is_file():
            logging.debug(f"Path not found: {path}")
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # Security check: verify path is within allowed base
        if allowed_base:
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

    def _stream_video(self, file_path: Path) -> None:
        """Stream video file with Range request support for seeking."""
        if not file_path.exists() or not file_path.is_file():
            logging.warning(f"Video file not found: {file_path}")
            self.send_error(HTTPStatus.NOT_FOUND, "Video file not found")
            return

        file_size = file_path.stat().st_size
        range_header = self.headers.get("Range")

        ctype, _ = mimetypes.guess_type(str(file_path))
        content_type = ctype or "video/mp4"

        try:
            with open(file_path, "rb") as f:
                if range_header:
                    # Handle partial content request
                    range_result = _parse_range_header(range_header, file_size)
                    if range_result is None:
                        self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        return

                    start, end = range_result
                    content_length = end - start + 1

                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(content_length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()

                    f.seek(start)
                    remaining = content_length
                    chunk_size = 64 * 1024  # 64KB chunks
                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                else:
                    # Send full file
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()

                    chunk_size = 64 * 1024
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        self.wfile.write(chunk)

        except OSError as e:
            logging.error(f"Error streaming video: {e}")
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)

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
                runfiles_path = _MODULE_DIR / "cybervision-player.runfiles" / "_main" / "libs" / "cybervision-core" / "shaders" / rel
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
                runfiles_path = _MODULE_DIR / "cybervision-player.runfiles" / "_main" / "libs" / "cybervision-core" / rel
                if runfiles_path.exists():
                    safe_path = runfiles_path
            self._send_file(safe_path, allowed_base=None)
            return

        if path.startswith("/libs/"):
            rel = path.removeprefix("/libs/")
            safe_path = _safe_join(SHARED_LIB_ROOT.parent, rel)
            if safe_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            # For shared lib, check both original location and Bazel runfiles
            if not safe_path.exists():
                runfiles_path = _MODULE_DIR / "cybervision-player.runfiles" / "_main" / "libs" / rel
                if runfiles_path.exists():
                    safe_path = runfiles_path
            self._send_file(safe_path, allowed_base=None)
            return

        if path.startswith("/models/"):
            rel = path.removeprefix("/models/")
            safe_path = _safe_join(MODELS_ROOT, rel)
            if safe_path is None:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(safe_path, allowed_base=WEBROOT)
            return

        if path == "/api/video":
            query = parse_qs(parsed.query)
            video_paths = query.get("path", [])
            if not video_paths:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing 'path' parameter")
                return

            video_path = Path(video_paths[0])
            # Security: only allow absolute paths that user explicitly provides
            # Don't restrict to a base directory since user wants to play videos from anywhere
            if not video_path.is_absolute():
                self.send_error(HTTPStatus.BAD_REQUEST, "Path must be absolute")
                return

            self._stream_video(video_path)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def run_server(*, host: str, port: int, debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logging.info(f"WEBROOT: {WEBROOT.resolve()}")
    logging.info(f"WEBROOT exists: {WEBROOT.exists()}")
    logging.info(f"Shared lib root: {SHARED_LIB_ROOT.resolve()}")
    with ThreadingHTTPServer((host, port), VideoPlayerHandler) as httpd:
        logging.info("Listening on http://%s:%d", host, port)
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logging.info("Shutting down")
