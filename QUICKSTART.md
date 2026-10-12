# Quick Start Guide

Get Claude Code talking to any OpenAI-compatible provider in under 2 minutes.

## 1. Install

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## 2. Configure

```bash
cp .env.example .env
# Edit .env and set at minimum:
#   OPENAI_API_KEY="sk-your-key"
#   BIG_MODEL / MIDDLE_MODEL / SMALL_MODEL   (see .env.example for common providers)
```

## 3. Start and use

```bash
# Start the proxy
python start_proxy.py        # or: ./start.sh, uv run claude-code-proxy

# In another terminal, point Claude Code at it
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

The Claude Code desktop app also works against the proxy: its connectors/plugins
browse UI probes a small set of Anthropic control-plane endpoints, which the
proxy answers with minimal payloads (a fake org from `PROXY_ORG_ID`, a
non-expiring token echo, and empty connector/plugin listings), so the panels
open showing empty lists instead of erroring. See README's "Desktop App
Compatibility" section for details.

## How model mapping works

| Claude Code sends       | Proxy uses       |
|-------------------------|------------------|
| `claude-*opus*`         | `BIG_MODEL`      |
| `claude-*sonnet*`       | `MIDDLE_MODEL`   |
| `claude-*haiku*`        | `SMALL_MODEL`    |

Each tier accepts a comma-separated fallback list, and `/effort` levels are translated to your upstream's native reasoning parameters — see the README's "Model Mapping" and "Controlling reasoning effort" sections for full details and provider examples.

## Verify your setup

```bash
# Liveness probe (5-token upstream completion)
curl http://localhost:8082/test-connection

# Unit tests (no live server needed)
uv run pytest tests/test_effort_mapper.py tests/test_endpoints.py tests/test_context_manager.py

# Full suite (includes live-integration tests; requires the proxy running on :8082)
uv run pytest tests/
```

## What you need

- Python 3.9+
- An API key for an OpenAI-compatible provider (OpenAI, Azure OpenAI, Ollama, NVIDIA NIM, ...)
- Claude Code CLI installed
