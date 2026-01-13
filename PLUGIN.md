# Azure OpenAI Plugin for Claude Code

A Claude Code plugin that enables seamless integration with Azure OpenAI and other OpenAI-compatible providers. No more manual proxy management - just install the plugin and use Claude Code with your Azure OpenAI deployment!

## What This Plugin Does

This plugin makes it easy to use Azure OpenAI (or any OpenAI-compatible API) with Claude Code by:

1. **Bundling the proxy server** - No separate setup required
2. **Providing convenient commands** - Use `/azure-openai` to manage everything
3. **Auto-starting the proxy** - Automatically starts when you begin a Claude Code session
4. **Handling configuration** - Simple setup wizard for Azure OpenAI credentials

## Installation

### Option 1: Install from Plugin Directory

```bash
# Clone or copy this repository
git clone https://github.com/schwarztim/claude-code-azure-proxy.git

# Install as a Claude Code plugin
claude plugin install ./claude-code-azure-proxy

# Or install from a specific path
claude plugin install /path/to/claude-code-azure-proxy
```

### Option 2: Install from GitHub (when published)

```bash
# Install directly from GitHub
claude plugin install schwarztim/claude-code-azure-proxy
```

### Option 3: Install from Plugin Marketplace (future)

```bash
# Once published to the plugin marketplace
claude plugin install azure-openai
```

## Quick Start

After installation, configure and start the proxy:

### 1. Run the Setup Wizard

```bash
claude
> /azure-openai setup
```

You'll be prompted for:
- Azure OpenAI API Key
- Azure OpenAI Endpoint URL (e.g., `https://your-resource.openai.azure.com/`)
- API Version (default: `2024-08-01-preview`)
- Model Deployments:
  - **Big Model** (for Claude Opus requests) - e.g., `gpt-4`
  - **Middle Model** (for Claude Sonnet requests) - e.g., `gpt-4`
  - **Small Model** (for Claude Haiku requests) - e.g., `gpt-35-turbo`

### 2. Start Using Claude Code

That's it! The proxy will start automatically in future sessions. You can now use Claude Code normally, and all requests will be routed through your Azure OpenAI deployment.

```bash
claude
> What is the capital of France?
# Response will come from your Azure OpenAI deployment
```

## Plugin Commands

The plugin provides a `/azure-openai` skill with several commands:

### `/azure-openai` or `/azure-openai help`
Show current status and available commands

### `/azure-openai setup`
Interactive setup wizard for Azure OpenAI configuration

### `/azure-openai start`
Manually start the proxy server

### `/azure-openai stop`
Stop the proxy server

### `/azure-openai restart`
Restart the proxy server

### `/azure-openai status`
Check proxy status and configuration

### `/azure-openai test`
Test the connection to Azure OpenAI

### `/azure-openai config`
Display current configuration (with sensitive values masked)

## Configuration

### Plugin Settings

The plugin supports configuration through Claude Code settings (`.claude/settings.json`):

```json
{
  "plugins": {
    "azure-openai": {
      "enabled": true,
      "autoStart": true,
      "apiKey": "your-azure-api-key",
      "baseUrl": "https://your-resource.openai.azure.com/",
      "apiVersion": "2024-08-01-preview",
      "bigModel": "gpt-4",
      "middleModel": "gpt-4",
      "smallModel": "gpt-35-turbo",
      "proxyPort": 8082,
      "maxTokens": 131072,
      "requestTimeout": 300
    }
  }
}
```

### Environment Variables

Alternatively, configure via `.env` file in the plugin directory:

```bash
# Azure OpenAI Configuration
OPENAI_API_KEY=your-azure-api-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview

# Model Mapping
BIG_MODEL=gpt-4
MIDDLE_MODEL=gpt-4
SMALL_MODEL=gpt-35-turbo

# Optional Settings
PORT=8082
MAX_TOKENS_LIMIT=131072
REQUEST_TIMEOUT=300
AZURE_OPENAI_AUTO_START=true
```

### Disabling Auto-Start

If you prefer to start the proxy manually:

**.claude/settings.json:**
```json
{
  "plugins": {
    "azure-openai": {
      "autoStart": false
    }
  }
}
```

**Or in .env:**
```bash
AZURE_OPENAI_AUTO_START=false
```

Then use `/azure-openai start` when needed.

## How It Works

The plugin architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ Claude Code CLI                                             │
│                                                             │
│  1. Session starts → session-start hook activates          │
│  2. Hook starts embedded proxy server (if auto-start enabled)│
│  3. Hook sets ANTHROPIC_BASE_URL=http://localhost:8082    │
│  4. All API calls go to local proxy                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Azure OpenAI Proxy (embedded in plugin)                    │
│                                                             │
│  • Receives Claude API format requests                     │
│  • Converts to OpenAI/Azure OpenAI format                  │
│  • Forwards to Azure OpenAI endpoint                       │
│  • Converts responses back to Claude format                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Azure OpenAI Service                                        │
│                                                             │
│  Your actual Azure OpenAI deployment                        │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

1. **You use Claude Code normally** - No special syntax or commands needed
2. **Claude Code makes API request** - But to `http://localhost:8082` instead of Anthropic
3. **Plugin proxy receives request** - In Claude API format (`/v1/messages`)
4. **Proxy converts format** - Claude API → OpenAI API
5. **Proxy forwards to Azure** - Using your Azure OpenAI deployment
6. **Response comes back** - Converted from OpenAI format → Claude format
7. **Claude Code displays response** - As if it came from Anthropic

## Supported Features

The proxy supports all major Claude Code features:

- ✅ **Streaming responses** - Real-time output
- ✅ **Function calling / Tool use** - Full tool support
- ✅ **Multimodal inputs** - Images and text
- ✅ **All models** - Maps Claude model names to your Azure deployments
- ✅ **Context windows** - Configurable token limits
- ✅ **Stop sequences** - Full parameter support
- ✅ **Token counting** - Estimation via `/v1/messages/count_tokens`

## Other OpenAI-Compatible Providers

While designed for Azure OpenAI, this plugin works with any OpenAI-compatible API:

### OpenAI (official)

```bash
# .env configuration
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
BIG_MODEL=gpt-4o
MIDDLE_MODEL=gpt-4o
SMALL_MODEL=gpt-4o-mini
```

### Local Models (Ollama)

```bash
# .env configuration
OPENAI_API_KEY=dummy-key
OPENAI_BASE_URL=http://localhost:11434/v1
BIG_MODEL=llama3.1:70b
MIDDLE_MODEL=llama3.1:70b
SMALL_MODEL=llama3.1:8b
```

### LiteLLM Gateway

```bash
# .env configuration
OPENAI_API_KEY=sk-litellm-...
OPENAI_BASE_URL=https://your-litellm-server:4000
BIG_MODEL=gpt-4
MIDDLE_MODEL=gpt-4
SMALL_MODEL=gpt-3.5-turbo
```

## Troubleshooting

### Check Plugin Status

```bash
claude
> /azure-openai status
```

This will show:
- Proxy running status
- Configuration status
- Recent log entries
- Port information

### View Logs

```bash
# In the plugin directory
tail -f proxy.log
```

### Test Connection

```bash
claude
> /azure-openai test
```

This sends a test request to verify Azure OpenAI connectivity.

### Common Issues

#### "Proxy not starting"

1. Check Python version: `python3 --version` (needs 3.8+)
2. Check if virtual environment exists: `ls -la venv/`
3. Reinstall dependencies:
   ```bash
   cd ~/.claude/plugins/azure-openai  # or wherever installed
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

#### "Connection refused"

1. Verify proxy is running: `/azure-openai status`
2. Check port isn't in use: `lsof -i :8082`
3. Restart proxy: `/azure-openai restart`

#### "Unsupported parameter: 'max_tokens'"

Your Azure API version is too old. Update in `.env`:
```bash
AZURE_API_VERSION=2024-08-01-preview
```

### Manual Proxy Management

If auto-start isn't working, you can manage the proxy manually:

```bash
# Navigate to plugin directory
cd ~/.claude/plugins/azure-openai

# Start proxy
source venv/bin/activate
python start_proxy.py &

# In another terminal, use Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=any-value
claude
```

## Uninstallation

```bash
# Stop the proxy first
claude
> /azure-openai stop

# Uninstall the plugin
claude plugin uninstall azure-openai

# Clean up (optional)
rm -rf ~/.claude/plugins/azure-openai
```

## Development

### Plugin Structure

```
claude-code-azure-proxy/
├── plugin.json              # Plugin manifest
├── skills/
│   └── azure-openai.md      # Skill definition
├── hooks/
│   └── session-start.sh     # Auto-start hook
├── src/                     # Proxy source code
│   ├── api/                 # API endpoints
│   ├── conversion/          # Format converters
│   ├── core/                # Core functionality
│   └── models/              # Data models
├── start_proxy.py           # Proxy entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Example configuration
└── README.md                # Main documentation
```

### Testing Changes

```bash
# Reload plugin
claude plugin reload azure-openai

# Test changes
claude
> /azure-openai status
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Claude Code
5. Submit a pull request

## Security Notes

- **Never commit `.env` files** - Contains sensitive API keys
- **API keys are masked** in `/azure-openai config` output
- **Proxy binds to localhost only** - Not exposed to network
- **PID files are excluded** from version control
- **Logs may contain sensitive data** - Check before sharing

## Comparison: Before vs. After

### Before (Manual Setup)

```bash
# Terminal 1: Start proxy manually
cd claude-code-azure-proxy
source venv/bin/activate
python start_proxy.py

# Terminal 2: Use Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=any-value
claude
```

### After (Plugin)

```bash
# One-time setup
claude plugin install azure-openai
claude
> /azure-openai setup

# Then just use Claude Code normally
claude
> Hello, world!
# Automatically uses Azure OpenAI
```

## License

MIT License - See LICENSE file for details

## Credits

Based on [claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy) by fuergaosi233

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/schwarztim/claude-code-azure-proxy/issues
- Documentation: https://github.com/schwarztim/claude-code-azure-proxy

## Links

- Main Repository: https://github.com/schwarztim/claude-code-azure-proxy
- Claude Code: https://claude.ai/code
- Azure OpenAI: https://azure.microsoft.com/en-us/products/ai-services/openai-service
