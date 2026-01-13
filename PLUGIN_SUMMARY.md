# Azure OpenAI Plugin - Implementation Summary

This document summarizes the Claude Code plugin implementation for Azure OpenAI integration.

## What Was Created

### Core Plugin Files

1. **plugin.json** - Plugin manifest
   - Defines plugin metadata (name, version, author)
   - Declares the `/azure-openai` skill
   - Defines `sessionStart` hook for auto-starting proxy
   - Specifies plugin settings schema

2. **skills/azure-openai.md** - Skill definition
   - Provides instructions for Claude to handle `/azure-openai` commands
   - Supports subcommands: setup, start, stop, restart, status, test, config
   - Includes error handling and user interaction patterns

3. **hooks/session-start.sh** - Session start hook
   - Automatically starts proxy when Claude Code session begins
   - Checks if auto-start is enabled
   - Manages virtual environment and dependencies
   - Configures ANTHROPIC_BASE_URL environment variable

### Documentation Files

1. **PLUGIN.md** - Complete plugin documentation
   - Installation instructions
   - Usage guide
   - Configuration options
   - Troubleshooting section
   - Comparison with manual setup

2. **INSTALL.md** - Installation guide
   - Step-by-step instructions for plugin installation
   - Manual installation instructions
   - Comprehensive troubleshooting
   - Uninstallation guide

3. **PLUGIN_SUMMARY.md** - This file
   - Overview of plugin implementation
   - Architecture documentation

### Modified Files

1. **README.md** - Updated to include plugin option
   - Added "Two Ways to Use This" section
   - Links to PLUGIN.md for plugin users

2. **.env.example** - Added plugin configuration
   - Added AZURE_OPENAI_AUTO_START setting

## Plugin Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Experience                          │
│                                                                 │
│  1. Install: claude plugin install azure-openai                │
│  2. Setup: /azure-openai setup                                 │
│  3. Use Claude Code normally                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Plugin Components                           │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ plugin.json     │  │ skills/          │  │ hooks/        │ │
│  │                 │  │  azure-openai.md │  │  session-     │ │
│  │ - Manifest      │  │                  │  │  start.sh     │ │
│  │ - Settings      │  │ - Commands       │  │               │ │
│  │ - Declarations  │  │ - Setup wizard   │  │ - Auto-start  │ │
│  └─────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Embedded Proxy Server                       │
│                                                                 │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ src/       │  │ start_proxy  │  │ Configuration          │ │
│  │  - api/    │  │  .py         │  │  - .env                │ │
│  │  - core/   │  │              │  │  - requirements.txt    │ │
│  │  - models/ │  │ FastAPI      │  │  - venv/               │ │
│  └────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                 │
│  Listens on: http://localhost:8082                             │
│  Implements: /v1/messages (Claude API)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Azure OpenAI Service                        │
│                                                                 │
│  User's Azure OpenAI deployment                                 │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### Installation Flow

1. User runs `claude plugin install azure-openai`
2. Claude Code:
   - Downloads/copies plugin files
   - Installs to `~/.claude/plugins/azure-openai/`
   - Creates Python virtual environment
   - Installs dependencies from requirements.txt
   - Makes hooks executable

### First Use Flow

1. User starts Claude Code
2. Plugin's `session-start.sh` hook runs
3. Hook checks if configured (`.env` exists with `OPENAI_API_KEY`)
4. If not configured, exits silently
5. If configured:
   - Checks if proxy already running
   - Starts proxy in background
   - Sets `ANTHROPIC_BASE_URL` environment variable
   - Claude Code now routes requests to local proxy

### Setup Flow

1. User runs `/azure-openai setup`
2. Claude (via skill) prompts for:
   - Azure OpenAI API key
   - Endpoint URL
   - API version
   - Model deployments
3. Skill creates/updates `.env` file
4. Offers to start proxy immediately
5. Future sessions auto-start via hook

### Request Flow

1. User types command in Claude Code
2. Claude Code makes API request to `http://localhost:8082/v1/messages`
3. Proxy receives request in Claude API format
4. Proxy converts to OpenAI/Azure OpenAI format
5. Proxy forwards to Azure OpenAI
6. Azure OpenAI responds
7. Proxy converts response back to Claude format
8. Claude Code displays response to user

## Key Design Decisions

### Why a Plugin?

**Problem**: Users had to manually:
- Start proxy server
- Set environment variables
- Keep terminal open
- Restart proxy if it crashed

**Solution**: Plugin automates all of this:
- Auto-starts on session begin
- Manages process lifecycle
- Provides convenient commands
- Handles errors gracefully

### Why Not Pure Plugin?

Plugins **cannot** intercept Claude Code's API calls directly. The proxy is still necessary because:
- Claude Code expects Claude API format (`/v1/messages`)
- Azure OpenAI uses OpenAI API format (different structure)
- Format conversion requires running code
- Plugins can't modify core Claude Code behavior

Therefore: **Plugin manages the proxy, proxy handles conversion**

### Why Session Hook?

The `sessionStart` hook:
- Runs automatically when Claude Code starts
- Can start background processes
- Can set environment variables
- Provides seamless experience

Alternative approaches (and why they don't work):
- ❌ Pure skill - Can't run automatically, user must invoke
- ❌ MCP server - Wrong abstraction, designed for tools/data
- ❌ Build-time hook - Runs for every command, too frequent

### Configuration Strategy

Two-tier configuration:

1. **Plugin settings** (`.claude/settings.json`)
   - User-friendly
   - Integrated with Claude Code
   - Can use settings UI

2. **Environment variables** (`.env`)
   - Familiar to developers
   - Works without plugin
   - Supports existing workflows

The hook and skill read from **both**, with settings taking precedence.

## File Structure

```
claude-code-azure-proxy/
├── plugin.json                    # Plugin manifest [NEW]
├── skills/                        # [NEW]
│   └── azure-openai.md           # Skill definition
├── hooks/                         # [NEW]
│   └── session-start.sh          # Auto-start hook
├── src/                           # Proxy implementation [EXISTING]
│   ├── api/
│   │   └── endpoints.py
│   ├── conversion/
│   │   ├── request_converter.py
│   │   └── response_converter.py
│   ├── core/
│   │   ├── client.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── logging.py
│   │   └── model_manager.py
│   ├── models/
│   │   ├── claude.py
│   │   └── openai.py
│   └── main.py
├── start_proxy.py                 # Entry point [EXISTING]
├── requirements.txt               # Dependencies [EXISTING]
├── .env.example                   # Config template [MODIFIED]
├── README.md                      # Main docs [MODIFIED]
├── PLUGIN.md                      # Plugin docs [NEW]
├── INSTALL.md                     # Install guide [NEW]
├── PLUGIN_SUMMARY.md             # This file [NEW]
└── ...
```

## Usage Examples

### First-Time Setup

```bash
# Install plugin
$ claude plugin install schwarztim/claude-code-azure-proxy

# Start Claude Code
$ claude

# Run setup
> /azure-openai setup

Azure OpenAI Setup Wizard
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Let me help you configure Azure OpenAI.

Please enter your Azure OpenAI API Key:
> ********************************

Please enter your Azure OpenAI Endpoint URL:
> https://my-company.openai.azure.com/

Please enter API version (default: 2024-08-01-preview):
> [Enter]

Please enter Big Model deployment name (for Claude Opus):
> gpt-4

Please enter Middle Model deployment name (for Claude Sonnet):
> gpt-4

Please enter Small Model deployment name (for Claude Haiku):
> gpt-35-turbo

✓ Configuration saved
✓ Proxy started
✓ Claude Code configured

You're all set! Use Claude Code normally.
```

### Daily Usage

```bash
# Just start Claude Code
$ claude

# Proxy starts automatically (silent)
# Use normally
> What's the weather like today?

# [Response from Azure OpenAI]
```

### Managing Proxy

```bash
$ claude

# Check status
> /azure-openai status

Azure OpenAI Proxy Status
━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: ✓ Running (PID: 12345)
Port: 8082
Configuration: ✓ Valid
Azure OpenAI: ✓ Connected

Endpoint: https://my-company.openai.azure.com/
Models:
  • Big: gpt-4
  • Middle: gpt-4
  • Small: gpt-35-turbo

# Stop proxy
> /azure-openai stop

✓ Proxy stopped

# Start proxy
> /azure-openai start

✓ Proxy started (PID: 12346)
```

## Testing the Plugin

### Manual Testing Steps

1. **Install the plugin**:
   ```bash
   claude plugin install ./
   ```

2. **Verify installation**:
   ```bash
   claude plugin list
   # Should show "azure-openai"
   ```

3. **Test setup command**:
   ```bash
   claude
   > /azure-openai setup
   # Follow prompts
   ```

4. **Test proxy start**:
   ```bash
   > /azure-openai start
   # Should start successfully
   ```

5. **Test status command**:
   ```bash
   > /azure-openai status
   # Should show running
   ```

6. **Test actual usage**:
   ```bash
   > Write a Python function to calculate factorial
   # Should get response from Azure OpenAI
   ```

7. **Test auto-start**:
   ```bash
   # Exit Claude Code
   > exit

   # Restart Claude Code
   $ claude

   # Proxy should auto-start
   > /azure-openai status
   # Should show running
   ```

### Automated Testing

```bash
# Test plugin manifest
python3 -m json.tool plugin.json

# Test hook script
bash -n hooks/session-start.sh

# Test proxy startup
source venv/bin/activate
python start_proxy.py &
sleep 2
curl http://localhost:8082/health
```

## Future Enhancements

Possible improvements:

1. **Plugin Settings UI**
   - Visual configuration editor
   - Integrated with Claude Code settings

2. **Health Monitoring**
   - Auto-restart on crash
   - Alert on errors
   - Performance metrics

3. **Multiple Profiles**
   - Switch between Azure/OpenAI/local
   - Per-project configurations

4. **Enhanced Setup**
   - Auto-detect Azure credentials
   - Validate deployments
   - Test connection during setup

5. **Plugin Marketplace**
   - Publish to official marketplace
   - Version updates
   - User ratings/feedback

## Compatibility

- **Claude Code**: Requires plugin support (check version)
- **Python**: 3.8 or higher
- **Operating Systems**:
  - ✅ Linux
  - ✅ macOS
  - ⚠️ Windows (requires adaptation of session-start.sh)
- **Azure OpenAI API**: 2024-08-01-preview or newer recommended

## Security Considerations

- API keys stored in `.env` (not in settings JSON)
- `.env` excluded from version control (.gitignore)
- Keys masked in `/azure-openai config` output
- Proxy binds to localhost only (not exposed to network)
- Session hook runs in user context (no privilege escalation)

## Support

- **Documentation**: See PLUGIN.md and README.md
- **Issues**: https://github.com/schwarztim/claude-code-azure-proxy/issues
- **Discussions**: https://github.com/schwarztim/claude-code-azure-proxy/discussions

## Credits

- Original proxy: Based on [claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy)
- Plugin implementation: schwarztim
- Claude Code: Anthropic

## License

MIT License - See LICENSE file for details
