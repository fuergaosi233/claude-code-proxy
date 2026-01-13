# Installation Guide

This guide covers both plugin installation and manual setup.

## Plugin Installation (Recommended)

### Prerequisites

- Claude Code CLI installed ([Installation Guide](https://claude.ai/code))
- Python 3.8 or higher
- Azure OpenAI resource with deployments (or another OpenAI-compatible API)

### Step 1: Install the Plugin

Choose one of these methods:

#### From GitHub

```bash
claude plugin install schwarztim/claude-code-azure-proxy
```

#### From Local Directory

```bash
# Clone the repository first
git clone https://github.com/schwarztim/claude-code-azure-proxy.git

# Install from local directory
claude plugin install ./claude-code-azure-proxy
```

#### From Marketplace (when available)

```bash
claude plugin install azure-openai
```

### Step 2: Initial Setup

Start Claude Code and run the setup wizard:

```bash
claude
```

Then in the Claude Code session:

```
> /azure-openai setup
```

Follow the prompts to configure:

1. **Azure OpenAI API Key** - Your Azure OpenAI API key
2. **Azure OpenAI Endpoint** - e.g., `https://your-resource.openai.azure.com/`
3. **API Version** - e.g., `2024-08-01-preview`
4. **Model Deployments**:
   - Big Model (for Claude Opus) - e.g., `gpt-4`
   - Middle Model (for Claude Sonnet) - e.g., `gpt-4`
   - Small Model (for Claude Haiku) - e.g., `gpt-35-turbo`

### Step 3: Start Using

That's it! The proxy will start automatically. Just use Claude Code normally:

```bash
claude
> Write a Python function to calculate fibonacci numbers
```

All requests will now go through your Azure OpenAI deployment.

### Verifying Installation

Check that everything is working:

```bash
claude
> /azure-openai status
```

You should see:
- ✅ Proxy status: Running
- ✅ Configuration: Valid
- ✅ Azure OpenAI: Connected

### Managing the Proxy

The plugin provides several commands:

```bash
# Show status
/azure-openai status

# Start proxy manually
/azure-openai start

# Stop proxy
/azure-openai stop

# Restart proxy
/azure-openai restart

# Test connection
/azure-openai test

# View configuration
/azure-openai config
```

## Manual Installation (Advanced)

For users who prefer manual control or need custom configurations.

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/schwarztim/claude-code-azure-proxy.git
cd claude-code-azure-proxy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
nano .env  # or vim, code, etc.
```

Required settings for Azure OpenAI:

```bash
OPENAI_API_KEY=your-azure-api-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview
BIG_MODEL=gpt-4
MIDDLE_MODEL=gpt-4
SMALL_MODEL=gpt-35-turbo
```

### Step 3: Start the Proxy

#### Option A: Using the wrapper script (recommended)

```bash
chmod +x GPT-wrapper.sh
./GPT-wrapper.sh
```

This will:
- Start the proxy automatically
- Launch Claude Code
- Configure environment variables
- Stop the proxy when you exit

#### Option B: Manual start

```bash
# Terminal 1: Start proxy
source venv/bin/activate
python start_proxy.py

# Terminal 2: Use Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=any-value
claude
```

#### Option C: Using Docker

```bash
docker compose up -d
```

### Step 4: Verify

Check that the proxy is running:

```bash
# Check proxy logs
tail -f proxy.log

# Test the endpoint
curl http://localhost:8082/health
```

## Troubleshooting

### Plugin Installation Issues

#### "Plugin not found"

Make sure the repository is public and accessible:

```bash
# Try installing from local directory instead
git clone https://github.com/schwarztim/claude-code-azure-proxy.git
claude plugin install ./claude-code-azure-proxy
```

#### "Python dependencies failed to install"

The plugin will try to install Python dependencies automatically. If this fails:

```bash
# Manually install dependencies
cd ~/.claude/plugins/azure-openai  # or wherever installed
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### "Permission denied" errors

Make sure the hook script is executable:

```bash
cd ~/.claude/plugins/azure-openai
chmod +x hooks/session-start.sh
```

### Runtime Issues

#### "Proxy not starting"

1. Check Python version:
   ```bash
   python3 --version  # Should be 3.8+
   ```

2. Check if virtual environment exists:
   ```bash
   ls -la ~/.claude/plugins/azure-openai/venv
   ```

3. Manually reinstall dependencies:
   ```bash
   cd ~/.claude/plugins/azure-openai
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

#### "Connection refused"

1. Check if proxy is running:
   ```bash
   claude
   > /azure-openai status
   ```

2. Check if port is in use:
   ```bash
   lsof -i :8082
   ```

3. Try a different port in `.env`:
   ```bash
   PORT=8083
   ```

#### "Invalid API key" errors

1. Verify your Azure OpenAI API key in `.env`
2. Check that the endpoint URL is correct
3. Verify API version is supported
4. Test connection:
   ```bash
   claude
   > /azure-openai test
   ```

### Configuration Issues

#### "Unsupported parameter: 'max_tokens'"

Your Azure API version is too old. Update in `.env`:

```bash
AZURE_API_VERSION=2024-08-01-preview
```

#### "Model not found" errors

Verify your model deployment names match what's in `.env`:

```bash
# In Azure Portal:
# Go to your Azure OpenAI resource → Deployments
# Note the deployment names (not model names)

# In .env, use deployment names:
BIG_MODEL=your-gpt-4-deployment-name
SMALL_MODEL=your-gpt-35-turbo-deployment-name
```

### Getting Help

If you're still having issues:

1. **Check logs**:
   ```bash
   # Plugin installation
   tail -f ~/.claude/plugins/azure-openai/proxy.log

   # Manual installation
   tail -f ./proxy.log
   ```

2. **Test connection**:
   ```bash
   # Check proxy health
   curl http://localhost:8082/health

   # Test Azure OpenAI connection
   curl http://localhost:8082/test-connection
   ```

3. **Enable debug logging**:

   In `.env`:
   ```bash
   LOG_LEVEL=DEBUG
   ```

   Then restart the proxy.

4. **Open an issue**:

   Visit: https://github.com/schwarztim/claude-code-azure-proxy/issues

   Include:
   - Your operating system
   - Python version
   - Claude Code version
   - Relevant log output (with API keys redacted)

## Uninstallation

### Plugin Uninstallation

```bash
# Stop the proxy
claude
> /azure-openai stop

# Uninstall plugin
claude plugin uninstall azure-openai

# Optional: Remove plugin directory
rm -rf ~/.claude/plugins/azure-openai
```

### Manual Installation Cleanup

```bash
# Stop the proxy
kill $(cat .proxy.pid)

# Remove files
cd ..
rm -rf claude-code-azure-proxy
```

## Next Steps

After successful installation:

1. Read [PLUGIN.md](PLUGIN.md) for detailed plugin usage
2. Check out [README.md](README.md) for proxy features and configuration
3. See [examples/](examples/) for usage examples
4. Join discussions at https://github.com/schwarztim/claude-code-azure-proxy/discussions

## Support

- 📖 Documentation: [README.md](README.md), [PLUGIN.md](PLUGIN.md)
- 🐛 Issues: https://github.com/schwarztim/claude-code-azure-proxy/issues
- 💬 Discussions: https://github.com/schwarztim/claude-code-azure-proxy/discussions
- 🌐 Claude Code: https://claude.ai/code
