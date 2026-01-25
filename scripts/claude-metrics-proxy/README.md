# Claude API Metrics Proxy

A proxy server that captures streaming performance metrics from Claude API requests.

## Metrics Captured

- **TTFT** (Time to First Token) - latency until first token arrives
- **TPOT** (Time Per Output Token) - average inter-token latency
- **Token counts** - input, output, cache read, cache creation
- **Request duration**

## Setup

```bash
cd scripts/claude-metrics-proxy
pip install -r requirements.txt
python proxy.py
```

## Usage

In another terminal, set the base URL and run Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:19418
claude
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/stats` | JSON stats with aggregates and recent requests |
| `/metrics` | Prometheus format metrics |

## Example Output

```bash
curl -s http://localhost:19418/stats | jq
```

```json
{
  "total_requests": 16,
  "aggregates": {
    "avg_ttft_ms": 769.18,
    "avg_tpot_ms": 40.11,
    "p50_ttft_ms": 520.66,
    "p50_tpot_ms": 31.05,
    "p99_ttft_ms": 2151.1,
    "p99_tpot_ms": 122.22,
    "total_input_tokens": 40678,
    "total_output_tokens": 263
  },
  "recent_requests": [...]
}
```

## Options

```
--port PORT    Proxy port (default: 19418)
--host HOST    Proxy host (default: 127.0.0.1)
```

## Run as macOS Daemon

Install the launchd service:

```bash
# Symlink plist to LaunchAgents
ln -sf ~/code/cyborg/scripts/claude-metrics-proxy/com.cyborg.claude-metrics-proxy.plist \
    ~/Library/LaunchAgents/

# Load and start the service
launchctl load ~/Library/LaunchAgents/com.cyborg.claude-metrics-proxy.plist

# Check status
launchctl list | grep claude-metrics-proxy

# View logs
tail -f /tmp/claude-metrics-proxy.log
```

To stop/unload:

```bash
launchctl unload ~/Library/LaunchAgents/com.cyborg.claude-metrics-proxy.plist
```
