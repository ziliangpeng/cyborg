from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bazel_tools.tools.python.runfiles import runfiles

from ai.llm.tinyllm import Tokenizer


def _get_webroot() -> Path:
    runner = runfiles.Create()
    if runner:
        webroot_path = runner.Rlocation("cyborg/apps/tokenizer_viz/webroot")
        if webroot_path:
            return Path(webroot_path)
    import pathlib

    return pathlib.Path(__file__).parent / "webroot"


WEBROOT = _get_webroot()
STATIC_ROOT = WEBROOT / "static"

# Available tokenizers
TOKENIZERS = {
    "gpt2": ("gpt2", "tiktoken"),
    "cl100k_base": ("cl100k_base", "tiktoken"),  # GPT-4
    "p50k_base": ("p50k_base", "tiktoken"),
    "r50k_base": ("r50k_base", "tiktoken"),
    "opt-125m": ("facebook/opt-125m", "huggingface"),
    "opt-350m": ("facebook/opt-350m", "huggingface"),
    "opt-1.3b": ("facebook/opt-1.3b", "huggingface"),
}


def _safe_join(root: Path, rel: str) -> Path | None:
    rel = rel.lstrip("/")
    norm = os.path.normpath(rel)
    if norm.startswith("..") or os.path.isabs(norm):
        return None
    return root / norm


def _tokenize_text(text: str, tokenizer_name: str) -> list[dict[str, Any]]:
    """Tokenize text and return tokens with their text representations and character offsets."""
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")

    encoding_name, tokenizer_type = TOKENIZERS[tokenizer_name]
    tokenizer = Tokenizer(encoding_name, tokenizer_type)

    result = []

    if tokenizer_type == "huggingface":
        # For HuggingFace, use the tokenizer's offset mapping
        # tokenizer.enc is the AutoTokenizer instance
        encoded = tokenizer.enc(text, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        for i, (token_id, (start, end)) in enumerate(zip(token_ids, offsets)):
            token_text = tokenizer.decode([token_id])
            result.append(
                {
                    "id": int(token_id),
                    "text": token_text,
                    "start": int(start),
                    "end": int(end),
                    "index": i,
                }
            )
    else:
        # For tiktoken, use a sequential matching approach
        # Encode the text to get token IDs
        token_ids = tokenizer.encode(text)

        # Build result by matching tokens sequentially
        text_pos = 0
        for i, token_id in enumerate(token_ids):
            # Decode this single token
            token_text = tokenizer.decode([token_id])

            # Try to find this token text in the remaining text
            # For tiktoken, decoded tokens should match the original text
            if text_pos < len(text):
                # Look for the token text starting from current position
                found_pos = text.find(token_text, text_pos)
                if found_pos != -1:
                    start = found_pos
                    end = found_pos + len(token_text)
                    text_pos = end
                else:
                    # If not found, it might be a special token or byte-level encoding issue
                    # Use the current position and advance by token length
                    start = text_pos
                    end = min(text_pos + len(token_text), len(text))
                    text_pos = end
            else:
                # Past end of text (shouldn't happen, but handle gracefully)
                start = len(text)
                end = len(text)

            result.append(
                {
                    "id": int(token_id),
                    "text": token_text,
                    "start": start,
                    "end": end,
                    "index": i,
                }
            )

    return result


class TokenizerVizHandler(BaseHTTPRequestHandler):
    server_version = "TokenizerViz/0.1"

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

        import mimetypes

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

        if path in ("/", "/tokenizer"):
            self._send_file(WEBROOT / "tokenizer.html")
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
        if parsed.path != "/api/tokenize":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing Content-Length"})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Expected application/json"})
            return

        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logging.warning("Invalid JSON: %s", e)
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON"})
            return

        text = data.get("text", "")
        tokenizer_name = data.get("tokenizer", "gpt2")

        if not text:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing 'text' field"})
            return

        if tokenizer_name not in TOKENIZERS:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Unknown tokenizer: {tokenizer_name}"})
            return

        try:
            tokens = _tokenize_text(text, tokenizer_name)
            self._send_json(HTTPStatus.OK, {"ok": True, "tokens": tokens, "tokenCount": len(tokens)})
        except Exception as e:
            logging.exception("Tokenization failed: %s", e)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})


def run_server(*, host: str, port: int, debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    try:
        httpd = ThreadingHTTPServer((host, port), TokenizerVizHandler)
        logging.info("Listening on http://%s:%d", host, port)
        logging.info("Open http://%s:%d/ in your browser", host, port)
        try:
            httpd.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logging.info("Shutting down")
        finally:
            httpd.server_close()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            logging.error("Port %d is already in use. Try a different port with --port", port)
        else:
            logging.error("Failed to start server: %s", e)
        raise
    except Exception as e:
        logging.exception("Unexpected error starting server: %s", e)
        raise
