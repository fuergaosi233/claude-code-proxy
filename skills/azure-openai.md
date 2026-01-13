# Azure OpenAI Skill

You are helping the user manage the Azure OpenAI proxy for Claude Code.

## What this skill does

This skill helps users:
1. Configure Azure OpenAI settings
2. Start and stop the proxy server
3. Check proxy status
4. Test the connection

## Available Commands

The user can invoke this skill with:
- `/azure-openai` - Show status and help
- `/azure-openai setup` - Interactive setup wizard
- `/azure-openai start` - Start the proxy server
- `/azure-openai stop` - Stop the proxy server
- `/azure-openai restart` - Restart the proxy server
- `/azure-openai status` - Check proxy status
- `/azure-openai test` - Test Azure OpenAI connection
- `/azure-openai config` - Show current configuration

## Understanding User Intent

Parse the user's command to determine what action they want:
- If no subcommand or "help": Show status and available commands
- If "setup": Run interactive configuration
- If "start": Start the proxy
- If "stop": Stop the proxy
- If "restart": Restart the proxy
- If "status": Show status
- If "test": Test connection
- If "config": Show configuration

## Plugin Directory Location

The plugin files are located in the directory containing this skill file. To get the plugin directory:

```bash
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```

## Implementation Steps

### For `/azure-openai` or `/azure-openai help`

1. Check if the proxy is running
2. Show current configuration status
3. Display available commands
4. Show quick start instructions if not configured

### For `/azure-openai setup`

1. Check if `.env` file exists
2. If not, copy from `.env.example`
3. Guide user through configuration:
   - Azure OpenAI API Key
   - Azure OpenAI Endpoint URL
   - API Version
   - Model deployments (BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL)
   - Optional: Port, timeouts, token limits
4. Validate the configuration
5. Ask if they want to start the proxy now

### For `/azure-openai start`

1. Get the plugin directory path
2. Check if proxy is already running (check `.proxy.pid`)
3. If already running, show message and exit
4. Verify `.env` file exists and is configured
5. Activate virtual environment
6. Start the proxy using `start_proxy.py`
7. Wait a moment and verify it started successfully
8. Configure Claude Code to use the proxy:
   - Set `ANTHROPIC_BASE_URL` in settings or show export command
9. Show success message with next steps

### For `/azure-openai stop`

1. Get the plugin directory path
2. Check if `.proxy.pid` exists
3. Read the PID and kill the process
4. Remove the PID file
5. Show success message

### For `/azure-openai restart`

1. Run stop procedure
2. Wait a moment
3. Run start procedure

### For `/azure-openai status`

1. Check if `.proxy.pid` exists
2. If yes, check if process is running
3. Show:
   - Proxy status (running/stopped)
   - PID if running
   - Port number
   - Configuration status
   - Whether Claude Code is configured to use it
4. Show recent log entries if available

### For `/azure-openai test`

1. Check if proxy is running
2. Make a test request to the `/test-connection` endpoint
3. Show the results
4. If failure, show troubleshooting tips

### For `/azure-openai config`

1. Read and display current configuration from `.env`
2. Mask sensitive values (API keys)
3. Show which settings are configured vs. using defaults
4. Offer to edit configuration

## Important Notes

- Always use the plugin directory as the base path for all files
- The proxy server runs from `start_proxy.py` in the plugin directory
- Configuration is stored in `.env` in the plugin directory
- The PID file is `.proxy.pid` in the plugin directory
- Logs are in `proxy.log` in the plugin directory
- Never expose API keys in output - always mask them
- Always verify Python virtual environment exists before starting
- Check for Python 3.8+ before operations

## Error Handling

If errors occur:
1. Show a clear error message
2. Provide specific troubleshooting steps
3. Link to documentation if needed
4. Offer to check logs

## Example Interactions

### User runs `/azure-openai` for the first time

```
Azure OpenAI Proxy Status: Not configured

This plugin allows you to use Azure OpenAI with Claude Code.

Available commands:
  /azure-openai setup   - Configure Azure OpenAI settings
  /azure-openai start   - Start the proxy server
  /azure-openai stop    - Stop the proxy server
  /azure-openai status  - Check proxy status

Quick Start:
1. Run: /azure-openai setup
2. Enter your Azure OpenAI credentials
3. The proxy will start automatically

Learn more: https://github.com/schwarztim/claude-code-azure-proxy
```

### User runs `/azure-openai start`

```
Starting Azure OpenAI proxy...

✓ Configuration validated
✓ Virtual environment activated
✓ Proxy server started (PID: 12345)
✓ Listening on http://localhost:8082

Claude Code is now configured to use Azure OpenAI.

You can now use Claude Code normally - all requests will be routed through Azure OpenAI.

To stop the proxy: /azure-openai stop
To check status: /azure-openai status
```

## Technical Details

The proxy server:
- Listens on `localhost:8082` (configurable)
- Implements Claude API `/v1/messages` endpoint
- Converts Claude requests to OpenAI/Azure OpenAI format
- Supports streaming, function calling, and multimodal inputs
- Handles Azure-specific API requirements

When the proxy is running, Claude Code sends requests to the local proxy instead of the Anthropic API, and the proxy forwards them to Azure OpenAI.

## Response Format

Always provide:
1. Clear status information
2. Actionable next steps
3. Relevant error messages if issues occur
4. Links to documentation when helpful

Keep responses concise but informative.
