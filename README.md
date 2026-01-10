# Claude Code Azure Proxy

A proxy server that enables **Claude Code** to work with Azure OpenAI and other OpenAI-compatible API providers. Convert Claude API requests to OpenAI/Azure OpenAI API calls, allowing you to use various LLM providers through the Claude Code CLI.

![Claude Code Proxy](demo.png)

## Why Use This?

- **Use Claude Code with Azure OpenAI**: Perfect for enterprise environments or regions where Claude API is not available
- **Cost Optimization**: Use Azure OpenAI credits or enterprise agreements with Claude Code
- **Privacy**: Keep your data within your Azure tenant
- **Flexibility**: Support for any OpenAI-compatible API (OpenAI, Azure OpenAI, local models via Ollama, etc.)

## Features

- **Full Claude API Compatibility**: Complete `/v1/messages` endpoint support
- **Azure OpenAI Support**: First-class support for Azure OpenAI with proper API version handling
- **Multiple Provider Support**: OpenAI, Azure OpenAI, local models (Ollama), and any OpenAI-compatible API
- **Smart Model Mapping**: Configure BIG/MIDDLE/SMALL models via environment variables
- **Function Calling**: Complete tool use support with proper conversion
- **Streaming Responses**: Real-time SSE streaming support
- **Image Support**: Base64 encoded image input
- **Custom Headers**: Automatic injection of custom HTTP headers for API requests
- **Wrapper Scripts**: Easy-to-use bash wrapper scripts for seamless integration
- **Error Handling**: Comprehensive error handling and logging

## Prerequisites

Before using this proxy, you need:

1. **Python 3.8+** - For running the proxy server
2. **Claude Code CLI** - The CLI tool this proxy works with
   - Install from: https://claude.ai/code
   - Or via npm: `npm install -g @anthropics/claude-code`
   - Verify installation: `claude --version`
3. **Azure OpenAI Access** (or another OpenAI-compatible API)
   - Azure OpenAI resource with deployments
   - API key and endpoint URL
4. **Git** - For cloning the repository

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/schwarztim/claude-code-azure-proxy.git
cd claude-code-azure-proxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure for Azure OpenAI

```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your Azure OpenAI settings
# Required fields for Azure OpenAI:
# - OPENAI_API_KEY: Your Azure OpenAI API key
# - OPENAI_BASE_URL: Your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
# - AZURE_API_VERSION: API version (e.g., 2024-08-01-preview)
# - BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL: Your Azure deployment names
```

**Example `.env` for Azure OpenAI:**

```bash
# Azure OpenAI Configuration
OPENAI_API_KEY=your-azure-api-key-here
OPENAI_BASE_URL=https://your-resource-name.openai.azure.com/

# Azure-specific setting (triggers AsyncAzureOpenAI client)
AZURE_API_VERSION=2024-08-01-preview

# Model Mapping - Use your Azure deployment names
BIG_MODEL=gpt-4
MIDDLE_MODEL=gpt-4
SMALL_MODEL=gpt-35-turbo

# Optional: Server Configuration
REQUEST_TIMEOUT=300
MAX_TOKENS_LIMIT=131072
```

### 3. Start the Proxy

**Option A: Manual Start**

```bash
# Activate virtual environment
source venv/bin/activate

# Start the proxy
python start_proxy.py

# Or use the start script
./start.sh
```

**Option B: Use the Wrapper Script (Recommended)**

The wrapper script automatically manages the proxy and launches Claude Code:

```bash
# Make wrapper executable (first time only)
chmod +x GPT-wrapper.sh

# Run Claude Code with Azure OpenAI
./GPT-wrapper.sh

# Or create an alias for convenience
alias GPT="/path/to/claude-code-azure-proxy/GPT-wrapper.sh"

# Then just run:
GPT
```

**Option C: Docker**

```bash
docker compose up -d
```

### 4. Use with Claude Code

**Method 1: Using Environment Variables**

```bash
# If ANTHROPIC_API_KEY is not set in the proxy:
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="any-value" claude

# If ANTHROPIC_API_KEY is set in the proxy for validation:
ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY="exact-matching-key" claude
```

**Method 2: Using the Wrapper Script (Easiest)**

```bash
./GPT-wrapper.sh
# This automatically:
# - Starts the proxy if not running
# - Configures Claude Code to use the proxy
# - Launches Claude Code with proper settings
```

## Configuration

The application automatically loads environment variables from a `.env` file in the project root using `python-dotenv`.

### Environment Variables

**Required:**

- `OPENAI_API_KEY` - Your API key (Azure API key or OpenAI API key)

**Azure OpenAI Specific:**

- `OPENAI_BASE_URL` - Your Azure OpenAI endpoint (e.g., `https://your-resource.openai.azure.com/`)
- `AZURE_API_VERSION` - Azure API version (e.g., `2024-08-01-preview`)
  - When this is set, the proxy uses Azure OpenAI client
  - Required for Azure OpenAI compatibility

**Model Configuration:**

- `BIG_MODEL` - Model for Claude Opus requests (default: `gpt-4o`)
  - For Azure: Use your deployment name (e.g., `gpt-4`)
- `MIDDLE_MODEL` - Model for Claude Sonnet requests (default: `gpt-4o`)
  - For Azure: Use your deployment name (e.g., `gpt-4`)
- `SMALL_MODEL` - Model for Claude Haiku requests (default: `gpt-4o-mini`)
  - For Azure: Use your deployment name (e.g., `gpt-35-turbo`)

**Security:**

- `ANTHROPIC_API_KEY` - Expected API key for client validation (optional)
  - If set, clients must provide this exact API key
  - If not set, any API key will be accepted

**Server Settings:**

- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8082`)
- `LOG_LEVEL` - Logging level (default: `WARNING`)

**Performance:**

- `MAX_TOKENS_LIMIT` - Maximum tokens limit (default: `4096`)
- `MIN_TOKENS_LIMIT` - Minimum tokens limit (default: `4096`)
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: `90`)
- `MAX_RETRIES` - Maximum retry attempts (default: `2`)

**Custom Headers:**

- `CUSTOM_HEADER_*` - Custom headers for API requests
  - Format: `CUSTOM_HEADER_<NAME>=value`
  - Example: `CUSTOM_HEADER_X_API_KEY=your-key`

## Wrapper Scripts

Two wrapper scripts are provided for easy integration:

### GPT-wrapper.sh (Recommended)
- Uses `--dangerously-skip-permissions` flag for direct execution
- Larger context window configuration (128K output tokens)
- Isolated Claude Code config in `.claude-gpt-config/`

### gpt-wrapper.sh
- Standard Claude Code behavior
- Prompts for permission as normal

Both scripts:
- Automatically start/manage the proxy process
- Detect if proxy is already running
- Use the script's directory (no hardcoded paths)
- Log output to `proxy.log` for debugging

## Monitoring

Check the proxy logs to verify requests are being sent to Azure OpenAI:

```bash
# Watch the logs in real-time
tail -f proxy.log

# The logs will show:
# - Incoming requests from Claude Code
# - Conversion from Claude to OpenAI format
# - Requests to your Azure OpenAI endpoint
# - Response handling
```

## Azure OpenAI Examples

### Example 1: Standard Azure OpenAI Setup

```bash
# .env configuration
OPENAI_API_KEY=abc123def456...
OPENAI_BASE_URL=https://my-company.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview
BIG_MODEL=gpt-4-deployment
MIDDLE_MODEL=gpt-4-deployment
SMALL_MODEL=gpt-35-turbo-deployment
```

### Example 2: Multiple Azure Deployments

```bash
# .env configuration
OPENAI_API_KEY=abc123def456...
OPENAI_BASE_URL=https://my-company.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview
BIG_MODEL=gpt-4-turbo-deployment
MIDDLE_MODEL=gpt-4-deployment
SMALL_MODEL=gpt-35-turbo-16k-deployment
```

## Other Provider Examples

### OpenAI

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
BIG_MODEL=gpt-4o
MIDDLE_MODEL=gpt-4o
SMALL_MODEL=gpt-4o-mini
```

### Local Models (Ollama)

```bash
OPENAI_API_KEY=dummy-key
OPENAI_BASE_URL=http://localhost:11434/v1
BIG_MODEL=llama3.1:70b
MIDDLE_MODEL=llama3.1:70b
SMALL_MODEL=llama3.1:8b
```

## Troubleshooting

### "sudo: no tty present" or "sudo: a terminal is required" Errors

When Claude Code tries to run sudo commands, it may fail because there's no interactive terminal for password prompts.

**Solutions:**
1. **Manual approach**: Copy the command and run it yourself in a terminal
2. **Passwordless sudo**: Configure specific commands to run without passwords (recommended)
3. **Full guide**: See [SUDO_SETUP.md](SUDO_SETUP.md) for detailed instructions

**Quick passwordless setup for common commands:**
```bash
# Create sudoers file for Claude Code
sudo visudo -f /etc/sudoers.d/claude-code

# Add these lines (replace 'youruser' with your username):
youruser ALL=(ALL) NOPASSWD: /usr/bin/mkdir
youruser ALL=(ALL) NOPASSWD: /usr/bin/systemctl
youruser ALL=(ALL) NOPASSWD: /usr/bin/pacman -S*
youruser ALL=(ALL) NOPASSWD: /usr/bin/apt install *
```

See [SUDO_SETUP.md](SUDO_SETUP.md) for security considerations and more options.

### "Unsupported parameter: 'max_tokens'" Error

If you see this error, your Azure deployment requires `max_completion_tokens` instead of `max_tokens`. This proxy automatically handles this for newer Azure API versions.

**Solution:** Ensure your `AZURE_API_VERSION` is set to `2024-08-01-preview` or later.

### Proxy Not Starting

```bash
# Check if dependencies are installed
source venv/bin/activate
pip install -r requirements.txt

# Check if .env file exists and is configured
cat .env

# Check for errors in the log
tail -f proxy.log
```

### Connection Refused

```bash
# Verify proxy is running
ps aux | grep "python.*start_proxy"

# Check if port 8082 is in use
lsof -i :8082

# Restart the proxy
kill $(cat .proxy.pid)
./start.sh
```

### Stale Python Cache

If you make code changes and they're not reflected:

```bash
# Clear Python bytecode cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Restart the proxy
kill $(cat .proxy.pid)
./start.sh
```

## Development

### Project Structure

```
claude-code-azure-proxy/
├── src/
│   ├── api/           # API endpoints
│   ├── conversion/    # Request/response converters
│   ├── core/          # Core functionality (client, config)
│   └── models/        # Data models
├── tests/             # Test files
├── .env.example       # Example configuration
├── start.sh           # Manual start script
├── GPT-wrapper.sh     # Wrapper script (skip permissions)
├── gpt-wrapper.sh     # Wrapper script (standard)
└── start_proxy.py     # Entry point
```

### Running Tests

```bash
pytest
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Security Considerations

**Important Security Notes:**

1. **Never commit your `.env` file** - It contains sensitive API keys
2. **Azure Endpoints**: Your Azure endpoint URL is specific to your organization
3. **API Keys**: Rotate your API keys regularly
4. **Network Security**: Consider restricting proxy access to localhost only
5. **Logging**: Be aware that debug logging may contain sensitive information

The `.gitignore` file is configured to exclude:
- `.env` (your actual configuration with secrets)
- `.proxy.pid` (process ID file)
- `.claude-gpt-config/` (Claude Code config directory)
- `proxy.log` (may contain sensitive request data)

## License

MIT License - See LICENSE file for details

## Credits

Based on [claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy) by fuergaosi233

## Support

For issues, questions, or contributions, please open an issue on GitHub.
