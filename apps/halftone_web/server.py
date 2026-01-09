from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from visual.halftone import (
    CmykParams,
    CrosshatchParams,
    DitherParams,
    GrayscaleParams,
    LineScreenParams,
    ProcessParams,
    StipplingParams,
    StyleType,
    all_style_names,
    parse_style_name,
    process,
)

WEBROOT = Path(__file__).parent / "webroot"
STATIC_ROOT = WEBROOT / "static"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


@dataclass(frozen=True)
class Defaults:
    sample: int = 8
    scale: int = 1
    angle: float = 45.0
    density: float = 1.0
    frequency: int = 8
    matrix_size: int = 4
    cell_size: int = 8
    antialias: bool = True


def _create_style_params(style_type: StyleType, defaults: Defaults) -> Any:
    match style_type:
        case StyleType.CMYK:
            return CmykParams(sample=defaults.sample, scale=defaults.scale)
        case StyleType.GRAYSCALE_SQRT | StyleType.GRAYSCALE_LINEAR:
            return GrayscaleParams(sample=defaults.sample, scale=defaults.scale, angle=defaults.angle)
        case StyleType.FLOYD_STEINBERG | StyleType.ORDERED_DITHER:
            return DitherParams(matrix_size=defaults.matrix_size)
        case StyleType.STIPPLING:
            return StipplingParams(cell_size=defaults.cell_size, density=defaults.density)
        case StyleType.LINE_SCREEN:
            return LineScreenParams(angle=defaults.angle, frequency=defaults.frequency)
        case StyleType.CROSSHATCH:
            return CrosshatchParams(angle1=defaults.angle, angle2=-defaults.angle, frequency=defaults.frequency)
        case _:
            raise ValueError(f"Unknown style type: {style_type}")


def _read_exact(handler: BaseHTTPRequestHandler, length: int) -> bytes:
    remaining = length
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = handler.rfile.read(min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _extract_multipart_field(body: bytes, content_type: str, field_name: str) -> bytes | None:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=default).parsebytes(header + body)

    if not msg.is_multipart():
        return None

    for part in msg.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if name != field_name:
            continue
        payload = part.get_payload(decode=True)
        return payload

    return None


def _safe_join(root: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel)
    if norm.startswith("..") or os.path.isabs(norm):
        return None
    return root / norm


class HalftoneHandler(BaseHTTPRequestHandler):
    server_version = "HalftoneWeb/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        base = WEBROOT.absolute()
        target = path.absolute()
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

        if path in ("/", "/halftone"):
            self._send_file(WEBROOT / "halftone.html")
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

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        if parsed.path != "/api/halftone":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing Content-Length"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": "Upload too large"})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Expected multipart/form-data"})
            return

        body = _read_exact(self, length)
        if len(body) != length:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Incomplete request body"})
            return

        file_bytes = _extract_multipart_field(body, content_type, "image")
        if not file_bytes:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": 'Missing "image" field'})
            return

        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
            img = ImageOps.exif_transpose(img)
        except (UnidentifiedImageError, OSError) as e:
            logging.warning("Invalid image upload: %s", e)
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid or unsupported image format."})
            return

        defaults = Defaults()
        process_params = ProcessParams(antialias=defaults.antialias)

        results: list[dict[str, str]] = []
        for style_name in all_style_names():
            style_type = parse_style_name(style_name)
            style_params = _create_style_params(style_type, defaults)
            try:
                out_img = process(img, style_type, style_params, process_params)
            except Exception as e:
                logging.exception("Halftone processing failed for style=%s", style_name)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": f"Processing failed for {style_name}."}
                )
                return

            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            results.append({"style": style_name, "dataUrl": f"data:image/png;base64,{encoded}"})

        self._send_json(HTTPStatus.OK, {"ok": True, "results": results})


def run_server(*, host: str, port: int, debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    with ThreadingHTTPServer((host, port), HalftoneHandler) as httpd:
        logging.info("Listening on http://%s:%d", host, port)
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logging.info("Shutting down")
