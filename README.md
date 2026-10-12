# Claude Code Proxy

A proxy server that enables **Claude Code** to work with OpenAI-compatible API providers. Convert Claude API requests to OpenAI API calls, allowing you to use various LLM providers through the Claude Code CLI.

![Claude Code Proxy](demo.png)

## Features

- **Full Claude API Compatibility**: Complete `/v1/messages` endpoint support
- **Multiple Provider Support**: OpenAI, Azure OpenAI, local models (Ollama), and any OpenAI-compatible API
- **Smart Model Mapping**: Configure BIG / MIDDLE / SMALL models via environment variables, with per-tier fallback lists and automatic health-driven failover
- **Reasoning Effort Control**: Honors Claude Code's `/effort` levels (low/medium/high/xhigh/max) and maps them to each upstream's native API (top-level string or chat-template-kwargs depending on the model)
- **Function Calling**: Complete tool use support with proper conversion
- **Streaming Responses**: Real-time SSE streaming support
- **Image Support**: Base64 encoded image input
- **Custom Headers**: Automatic injection of custom HTTP headers for API requests
- **Desktop App Compatibility**: Minimal Anthropic control-plane endpoints (`/api/organizations`, `/v1/me`, a non-expiring `/v1/oauth/token` echo, empty connector/plugin listings) so the Claude Code desktop app's connectors/plugins browse UI opens instead of erroring; any other unknown `/v1` or `/api` path returns a well-formed Anthropic-style 404
- **Error Handling**: Comprehensive error handling and logging

## Quick Start

### 1. Install Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your API configuration
# Note: Environment variables are automatically loaded from .env file
```

### 3. Start Server

```bash
# Direct run
python start_proxy.py

# Or with UV
uv run claude-code-proxy

# Or with docker compose
docker compose up -d
```

### 4. Use with Claude Code

```bash
# If ANTHROPIC_API_KEY is not set in the proxy:
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="any-value" claude

# If ANTHROPIC_API_KEY is set in the proxy:
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="exact-matching-key" claude
```

## Configuration

The application automatically loads environment variables from a `.env` file in the project root using `python-dotenv`. You can also set environment variables directly in your shell.

### Environment Variables

**Required:**

- `OPENAI_API_KEY` - Your API key for the target provider

**Security:**

- `ANTHROPIC_API_KEY` - Expected Anthropic API key for client validation
  - If set, clients must provide this exact API key to access the proxy
  - If not set, any API key will be accepted

**Desktop App Compatibility:**

- `PROXY_ORG_ID` - Fake organization id returned by the desktop-compat endpoints (`GET /api/organizations`, `GET /v1/organizations`, `GET /v1/me`, and the `PROXY_ORG_ID`-echoed sub-resource listings) (default: `claude-code-proxy-org`)

**Model Configuration:**

Models are mapped by tier. Each tier accepts a comma-separated list — the first entry is the primary, the rest are automatic fallbacks with health-check-driven recovery:

- `BIG_MODEL` - Model(s) for Claude opus requests (default: `gpt-4o`)
- `MIDDLE_MODEL` - Model(s) for Claude sonnet requests (default: value of `BIG_MODEL`)
- `SMALL_MODEL` - Model(s) for Claude haiku requests (default: `gpt-4o-mini`)

**Reasoning Effort:**

Claude Code's `/effort` levels (`low`, `medium`, `high`, `xhigh`, `max`) flow through to the upstream. The proxy normalizes them (`xhigh` and `max` both map to the canonical `max`) and translates to each model's native vocabulary — e.g. GLM-5.x gets `low/high/max`, gpt-oss and deepseek-v4.x get `low/medium/high`, and Nemotron models get `chat_template_kwargs` thinking switches. Defaults, applied when the client sends no effort:

- `BIG_MODEL_EFFORT`, `MIDDLE_MODEL_EFFORT`, `SMALL_MODEL_EFFORT` - Per-tier default effort (`low|medium|high|max|xhigh`). An in-session `/effort` always overrides these.
- `REASONING_EFFORT` - Global default effort for all tiers when a per-tier value is not set (built-in fallback: `high`).

**Rate Limiting and Reliability:**

- `RPM_LIMIT` - Max requests per minute per API key (default: `40`)
- `MODEL_FAILURE_THRESHOLD` - Consecutive failures before switching to a fallback model (default: `3`)
- `MODEL_HEALTH_CHECK_INTERVAL` - Seconds between background probes of the primary model (default: `60`)

**API Configuration:**

- `OPENAI_BASE_URL` - API base URL (default: `https://api.openai.com/v1`)
- `AZURE_API_VERSION` - Azure OpenAI API version (e.g. `2024-03-01-preview`); see the Azure provider example below

**Server Settings:**

- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8082`)
- `LOG_LEVEL` - Logging level (default: `INFO`)

**Performance:**

- `MAX_TOKENS_LIMIT` - Max response tokens (default: `4096`)
- `MIN_TOKENS_LIMIT` - Min response tokens (default: `100`); raise this (e.g. `4096`) for reasoning models whose thinking consumes output budget
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: `90`)
- `MAX_RETRIES` - Retry attempts for failed requests (default: `2`)

**Multi-Key / Multi-Provider:**

- `OPENAI_API_KEY` accepts a comma-separated list of keys (`"key1,key2,..."`) — requests are round-robined across them with per-key RPM accounting and 60 s cooldown penalties on 429s. All keys share `OPENAI_BASE_URL`.

**Custom Headers:**

- `CUSTOM_HEADER_*` - Custom headers for API requests (e.g., `CUSTOM_HEADER_ACCEPT`, `CUSTOM_HEADER_AUTHORIZATION`)
  - Uncomment in `.env` file to enable custom headers

### Custom Headers Configuration

Add custom headers to your API requests by setting environment variables with the `CUSTOM_HEADER_` prefix:

```bash
# Uncomment to enable custom headers
# CUSTOM_HEADER_ACCEPT="application/jsonstream"
# CUSTOM_HEADER_CONTENT_TYPE="application/json"
# CUSTOM_HEADER_USER_AGENT="your-app/1.0.0"
# CUSTOM_HEADER_AUTHORIZATION="Bearer your-token"
# CUSTOM_HEADER_X_API_KEY="your-api-key"
# CUSTOM_HEADER_X_CLIENT_ID="your-client-id"
# CUSTOM_HEADER_X_CLIENT_VERSION="1.0.0"
# CUSTOM_HEADER_X_REQUEST_ID="unique-request-id"
# CUSTOM_HEADER_X_TRACE_ID="trace-123"
# CUSTOM_HEADER_X_SESSION_ID="session-456"
```

### Header Conversion Rules

Environment variables with the `CUSTOM_HEADER_` prefix are automatically converted to HTTP headers:

- Environment variable: `CUSTOM_HEADER_ACCEPT`
- HTTP Header: `ACCEPT`

- Environment variable: `CUSTOM_HEADER_X_API_KEY`
- HTTP Header: `X-API-KEY`

- Environment variable: `CUSTOM_HEADER_AUTHORIZATION`
- HTTP Header: `AUTHORIZATION`

### Supported Header Types

- **Content Type**: `ACCEPT`, `CONTENT-TYPE`
- **Authentication**: `AUTHORIZATION`, `X-API-KEY`
- **Client Identification**: `USER-AGENT`, `X-CLIENT-ID`, `X-CLIENT-VERSION`
- **Tracking**: `X-REQUEST-ID`, `X-TRACE-ID`, `X-SESSION-ID`

### Usage Example

```bash
# Basic configuration
OPENAI_API_KEY="sk-your-openai-api-key-here"
OPENAI_BASE_URL="https://api.openai.com/v1"

# Enable custom headers (uncomment as needed)
CUSTOM_HEADER_ACCEPT="application/jsonstream"
CUSTOM_HEADER_CONTENT_TYPE="application/json"
CUSTOM_HEADER_USER_AGENT="my-app/1.0.0"
CUSTOM_HEADER_AUTHORIZATION="Bearer my-token"
```

The proxy will automatically include these headers in all API requests to the target LLM provider.

### Model Mapping

The proxy maps Claude model requests to your configured models by tier:

| Claude Request    | Mapped To      | Default                      |
| ----------------- | -------------- | ---------------------------- |
| haiku models      | `SMALL_MODEL`  | `gpt-4o-mini`                |
| sonnet models     | `MIDDLE_MODEL` | value of `BIG_MODEL`         |
| opus models       | `BIG_MODEL`    | `gpt-4o`                     |
| `gpt-*` / `o1-*` / `ep-*` / `doubao-*` / `deepseek-*` | used as-is (tier `unknown`) | — |

Each tier variable accepts a comma-separated fallback list (e.g. `BIG_MODEL="a,a2,a3"`); on repeated failures the proxy moves down the list and probes the primary in the background to restore it when healthy. Reasoning effort defaults are independently tunable per tier via `BIG_MODEL_EFFORT` / `MIDDLE_MODEL_EFFORT` / `SMALL_MODEL_EFFORT`.

### Provider Examples

#### OpenAI

```bash
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
BIG_MODEL="gpt-4o"
MIDDLE_MODEL="gpt-4o"
SMALL_MODEL="gpt-4o-mini"
```

#### Azure OpenAI

```bash
OPENAI_API_KEY="your-azure-key"
OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
BIG_MODEL="gpt-4"
MIDDLE_MODEL="gpt-4"
SMALL_MODEL="gpt-35-turbo"
```

#### Local Models (Ollama)

```bash
OPENAI_API_KEY="dummy-key"  # Required but can be dummy
OPENAI_BASE_URL="http://localhost:11434/v1"
BIG_MODEL="llama3.1:70b"
MIDDLE_MODEL="llama3.1:70b"
SMALL_MODEL="llama3.1:8b"
```

#### Other Providers

Any OpenAI-compatible API can be used by setting the appropriate `OPENAI_BASE_URL`.

## Usage Examples

### Basic Chat

```python
import httpx

response = httpx.post(
    "http://localhost:8082/v1/messages",
    json={
        "model": "claude-3-5-sonnet-20241022",  # Maps to MIDDLE_MODEL
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
```

## Integration with Claude Code

This proxy is designed to work seamlessly with Claude Code CLI:

```bash
# Start the proxy
./start.sh

# Use Claude Code with the proxy
ANTHROPIC_BASE_URL=http://localhost:8082 claude

# Or set permanently
export ANTHROPIC_BASE_URL=http://localhost:8082
claude
```

### Controlling reasoning effort in Claude Code

In Claude Code 2.1.x, use `/effort low|medium|high|xhigh` to change it on the fly, or set a starting level with `CLAUDE_CODE_EFFORT_LEVEL`. The proxy routes each level to the correct parameter for your configured upstream model:

| `/effort` | GLM-5.x (`z-ai/glm-5.3`) | gpt-oss (`openai/gpt-oss-*`) | deepseek-v4.1-flash | Nemotron (`nvidia/nemotron-*`) | kimi-k2-thinking / qwen3-*-thinking |
|---|---|---|---|---|---|
| `low` | `low` | `low` | `low` | thinking **off** | *(omitted)* |
| `medium` | `high` | `medium` | `high` | thinking on, `medium_effort` | *(omitted)* |
| `high` | `high` | `high` | `high` | thinking on, `medium_effort` | *(omitted)* |
| `xhigh` | **`max`** | `high` (cap) | `high` (cap) | thinking on (full) | *(omitted)* |

NVIDIA NIM's deepseek-v4.1-flash changed its effort contract on 2026-10-11: it previously took a numeric 1–100 value, then started rejecting integers ("expected string or map") and also rejects `"medium"` in-stream. The table reflects the live-verified string vocabulary.

Notes:

- In-session `/effort` always wins over the proxy env vars.
- A `max` level maps to the same canonical value as `xhigh`, so its upstream translation is identical to the `xhigh` row.
- The env defaults are independent per tier — a common pattern is `SMALL_MODEL_EFFORT=low` (cheap haiku with thinking off) while leaving `BIG_MODEL` at the built-in `high`.
- `kimi-k2-thinking` / `qwen3-*-thinking` are always-on reasoning models: the proxy sends no reasoning parameter to them (their chat template forces a thinking trace).

### Endpoints reference

| Path                                | Purpose                                  |
| ----------------------------------- | ---------------------------------------- |
| `POST /v1/messages`                 | Main messages endpoint (Claude schema)   |
| `POST /v1/messages/count_tokens`    | Token-count endpoint (Anthropic format)  |
| `GET  /v1/models`                   | Models list (one Claude alias per tier)    |
| `GET  /health`                      | Health + per-key RPM usage/penalties     |
| `GET  /test-connection`             | Liveness: probes upstream with a 5-token completion |
| `GET/HEAD /api/hello` and `/hello`  | Tiny probe for Claude extensions         |
| `GET  /`                            | Version + effective config summary       |

## Testing

```bash
# Unit tests (no live server needed)
uv run pytest tests/test_effort_mapper.py tests/test_endpoints.py tests/test_context_manager.py

# Full suite (includes live-integration tests; requires the proxy running on :8082)
uv run pytest tests/

# Validate the reasoning-effort envelope against a live NVIDIA NIM endpoint
NVIDIA_API_KEY=nvapi-... uv run scripts/test_nim_effort.py [model]
```

## Development

### Using UV

```bash
# Install dependencies
uv sync

# Run server
uv run claude-code-proxy

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

### Project Structure

```
claude-code-proxy/
├── src/
│   ├── main.py                  # Server entrypoint
│   ├── api/endpoints.py         # FastAPI routes (/v1/messages, /v1/models, /health, ...)
│   ├── conversion/
│   │   ├── request_converter.py   # Claude request -> OpenAI request
│   │   ├── response_converter.py  # OpenAI response -> Claude response
│   │   └── effort_mapper.py       # Reasoning-effort translation per model family
│   ├── core/
│   │   ├── client.py              # Async upstream client with multi-key RPM + retries
│   │   ├── config.py              # Env-var configuration
│   │   ├── model_manager.py       # Tier mapping + fallback/health failover
│   │   ├── context_manager.py     # Context-window truncation
│   │   └── provider_discovery.py  # Upstream provider detection
│   └── models/                    # Pydantic schemas (Claude + OpenAI)
├── tests/                       # Unit tests (pytest)
├── scripts/                     # Live/verification scripts (need running proxy or real keys)
├── start_proxy.py               # Startup script
├── .env.example                 # Config template
└── README.md                    # This file
```

## Performance

- **Async/await** for high concurrency
- **Connection pooling** for efficiency
- **Streaming support** for real-time responses
- **Configurable timeouts** and retries
- **Smart error handling** with detailed logging

## License

MIT License
