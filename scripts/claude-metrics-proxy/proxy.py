#!/usr/bin/env python3
"""
Claude API Metrics Proxy

Captures streaming performance metrics:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- Total tokens (input/output)
- Request duration

Usage:
    python proxy.py [--port 8080] [--metrics-port 9090]

Then set:
    export ANTHROPIC_BASE_URL=http://localhost:8080
"""

import argparse
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# ============================================================================
# Metrics Configuration
# ============================================================================

# Histograms for latency metrics
TTFT_HISTOGRAM = Histogram(
    'claude_ttft_seconds',
    'Time to first token in seconds',
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
)

TPOT_HISTOGRAM = Histogram(
    'claude_tpot_milliseconds',
    'Time per output token in milliseconds',
    buckets=(5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500)
)

REQUEST_DURATION_HISTOGRAM = Histogram(
    'claude_request_duration_seconds',
    'Total request duration in seconds',
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300)
)

# Counters for token usage
INPUT_TOKENS_COUNTER = Counter(
    'claude_input_tokens_total',
    'Total input tokens consumed'
)

OUTPUT_TOKENS_COUNTER = Counter(
    'claude_output_tokens_total',
    'Total output tokens generated'
)

CACHE_READ_TOKENS_COUNTER = Counter(
    'claude_cache_read_tokens_total',
    'Total cache read tokens'
)

CACHE_CREATION_TOKENS_COUNTER = Counter(
    'claude_cache_creation_tokens_total',
    'Total cache creation tokens'
)

REQUEST_COUNTER = Counter(
    'claude_requests_total',
    'Total number of requests'
)

# Gauges for recent metrics (useful for real-time dashboards)
LAST_TTFT_GAUGE = Gauge('claude_last_ttft_seconds', 'Most recent TTFT')
LAST_TPOT_GAUGE = Gauge('claude_last_tpot_milliseconds', 'Most recent TPOT')

# ============================================================================
# Request Tracking
# ============================================================================

@dataclass
class RequestMetrics:
    """Tracks metrics for a single request."""
    start_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    output_tokens: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_times: list = field(default_factory=list)

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.first_token_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def tpot(self) -> Optional[float]:
        """Average time per output token in milliseconds."""
        if len(self.token_times) >= 2:
            # Calculate inter-token intervals
            intervals = []
            for i in range(1, len(self.token_times)):
                interval_ms = (self.token_times[i] - self.token_times[i-1]) * 1000
                intervals.append(interval_ms)
            if intervals:
                return sum(intervals) / len(intervals)
        return None

    @property
    def duration(self) -> float:
        """Total request duration in seconds."""
        end = self.last_token_time or time.time()
        return end - self.start_time


# Store recent requests for the /stats endpoint
recent_requests: deque = deque(maxlen=100)

# ============================================================================
# Proxy Application
# ============================================================================

# Global HTTP client - reuse connections
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage HTTP client lifecycle."""
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
    yield
    await http_client.aclose()

app = FastAPI(title="Claude Metrics Proxy", lifespan=lifespan)

ANTHROPIC_API_URL = "https://api.anthropic.com"


def record_metrics(metrics: RequestMetrics):
    """Record metrics to Prometheus."""

    # Token counters
    INPUT_TOKENS_COUNTER.inc(metrics.input_tokens)
    OUTPUT_TOKENS_COUNTER.inc(metrics.output_tokens)
    CACHE_READ_TOKENS_COUNTER.inc(metrics.cache_read_tokens)
    CACHE_CREATION_TOKENS_COUNTER.inc(metrics.cache_creation_tokens)
    REQUEST_COUNTER.inc()

    # Latency histograms
    if metrics.ttft is not None:
        TTFT_HISTOGRAM.observe(metrics.ttft)
        LAST_TTFT_GAUGE.set(metrics.ttft)

    if metrics.tpot is not None:
        TPOT_HISTOGRAM.observe(metrics.tpot)
        LAST_TPOT_GAUGE.set(metrics.tpot)

    REQUEST_DURATION_HISTOGRAM.observe(metrics.duration)

    # Store for /stats endpoint
    recent_requests.append({
        "timestamp": datetime.now().isoformat(),
        "ttft_ms": round(metrics.ttft * 1000, 2) if metrics.ttft else None,
        "tpot_ms": round(metrics.tpot, 2) if metrics.tpot else None,
        "duration_s": round(metrics.duration, 2),
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "cache_read_tokens": metrics.cache_read_tokens,
        "cache_creation_tokens": metrics.cache_creation_tokens,
    })

    # Log to console
    print(f"\n{'='*60}")
    print(f"Request completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"  TTFT:          {metrics.ttft*1000:.1f}ms" if metrics.ttft else "  TTFT:          N/A")
    print(f"  TPOT:          {metrics.tpot:.1f}ms" if metrics.tpot else "  TPOT:          N/A")
    print(f"  Duration:      {metrics.duration:.2f}s")
    print(f"  Input tokens:  {metrics.input_tokens}")
    print(f"  Output tokens: {metrics.output_tokens}")
    print(f"  Cache read:    {metrics.cache_read_tokens}")
    print(f"  Cache create:  {metrics.cache_creation_tokens}")
    print(f"{'='*60}\n", flush=True)


def filter_response_headers(headers: httpx.Headers) -> dict:
    """Filter out headers that shouldn't be forwarded to the client."""
    skip_headers = {
        "content-encoding",
        "content-length",
        "transfer-encoding",
    }
    return {k: v for k, v in headers.items() if k.lower() not in skip_headers}


async def stream_response_with_metrics(
    target_url: str,
    method: str,
    headers: dict,
    body: bytes,
    metrics: RequestMetrics,
):
    """Stream response from upstream while collecting metrics."""

    req = http_client.build_request(method, target_url, headers=headers, content=body)
    response = await http_client.send(req, stream=True)

    try:
        async for line in response.aiter_lines():
            if not line:
                yield line + "\n"
                continue

            # Parse SSE events
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix

                try:
                    event = json.loads(data)
                    event_type = event.get("type", "")

                    # Track first token
                    if event_type == "content_block_delta":
                        now = time.time()

                        if metrics.first_token_time is None:
                            metrics.first_token_time = now

                        metrics.token_times.append(now)
                        metrics.last_token_time = now

                        # Count output tokens from delta
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            metrics.output_tokens += 1

                    # Extract final usage stats
                    elif event_type == "message_delta":
                        usage = event.get("usage", {})
                        if "output_tokens" in usage:
                            metrics.output_tokens = usage["output_tokens"]

                    elif event_type == "message_start":
                        message = event.get("message", {})
                        usage = message.get("usage", {})
                        metrics.input_tokens = usage.get("input_tokens", 0)
                        metrics.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                        metrics.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)

                except json.JSONDecodeError:
                    pass

            yield line + "\n"
    finally:
        await response.aclose()
        record_metrics(metrics)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(request: Request, path: str):
    """Proxy requests to Anthropic API with metrics collection."""

    # Build target URL with query string
    target_url = f"{ANTHROPIC_API_URL}/v1/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    # Get request body
    body = await request.body()

    # Forward headers (except host and accept-encoding)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("accept-encoding", None)

    # Check if streaming
    is_streaming = False
    if body:
        try:
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)
        except json.JSONDecodeError:
            pass

    metrics = RequestMetrics()

    if is_streaming:
        # For streaming, we need to get headers first
        # Make a preliminary request to get status/headers, then stream
        return StreamingResponse(
            stream_response_with_metrics(target_url, request.method, headers, body, metrics),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming request
        response = await http_client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )

        # Extract metrics from response
        try:
            resp_json = response.json()
            usage = resp_json.get("usage", {})
            metrics.input_tokens = usage.get("input_tokens", 0)
            metrics.output_tokens = usage.get("output_tokens", 0)
            metrics.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            metrics.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
            metrics.first_token_time = time.time()
            metrics.last_token_time = time.time()
        except Exception:
            pass

        record_metrics(metrics)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=filter_response_headers(response.headers),
        )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/stats")
async def stats():
    """JSON stats for recent requests."""
    requests_list = list(recent_requests)

    if not requests_list:
        return {"message": "No requests recorded yet"}

    # Calculate aggregates
    ttfts = [r["ttft_ms"] for r in requests_list if r["ttft_ms"] is not None]
    tpots = [r["tpot_ms"] for r in requests_list if r["tpot_ms"] is not None]

    return {
        "total_requests": len(requests_list),
        "aggregates": {
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "avg_tpot_ms": round(sum(tpots) / len(tpots), 2) if tpots else None,
            "p50_ttft_ms": round(sorted(ttfts)[len(ttfts)//2], 2) if ttfts else None,
            "p50_tpot_ms": round(sorted(tpots)[len(tpots)//2], 2) if tpots else None,
            "p99_ttft_ms": round(sorted(ttfts)[int(len(ttfts)*0.99)], 2) if len(ttfts) > 1 else None,
            "p99_tpot_ms": round(sorted(tpots)[int(len(tpots)*0.99)], 2) if len(tpots) > 1 else None,
            "total_input_tokens": sum(r["input_tokens"] for r in requests_list),
            "total_output_tokens": sum(r["output_tokens"] for r in requests_list),
        },
        "recent_requests": requests_list[-10:]  # Last 10
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser(description="Claude API Metrics Proxy")
    parser.add_argument("--port", type=int, default=19418, help="Proxy port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Proxy host")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Claude API Metrics Proxy                           ║
╠══════════════════════════════════════════════════════════════╣
║  Proxy running on:     http://{args.host}:{args.port}                  ║
║  Prometheus metrics:   http://{args.host}:{args.port}/metrics          ║
║  JSON stats:           http://{args.host}:{args.port}/stats            ║
╠══════════════════════════════════════════════════════════════╣
║  To use with Claude Code:                                    ║
║                                                              ║
║    export ANTHROPIC_BASE_URL=http://localhost:{args.port}          ║
║    claude                                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
