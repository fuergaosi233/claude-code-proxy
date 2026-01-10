#!/bin/bash

# Claude Code Proxy Wrapper for GPT (with --dangerously-skip-permissions)
# This script automatically manages the proxy and connects Claude Code to Azure OpenAI
# Uses separate config directory to not interfere with normal 'claude' command

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROXY_DIR="$SCRIPT_DIR"
PROXY_PORT=8082
PID_FILE="$PROXY_DIR/.proxy.pid"
GPT_CONFIG_DIR="$PROXY_DIR/.claude-gpt-config"

# Save the current working directory where user launched the command
ORIGINAL_PWD="$(pwd)"

# Create separate config directory for GPT mode
mkdir -p "$GPT_CONFIG_DIR"

# Function to check if proxy is running
is_proxy_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to start the proxy in the background
start_proxy() {
    echo "Starting Claude Code Proxy (Azure OpenAI)..."

    # Change to proxy dir only for starting the server
    (
        cd "$PROXY_DIR"
        source venv/bin/activate
        nohup python start_proxy.py > "$PROXY_DIR/proxy.log" 2>&1 &
        echo $! > "$PID_FILE"
    )

    # Wait a moment for the server to start
    sleep 2

    # Verify it started
    if is_proxy_running; then
        echo "Proxy started successfully on http://localhost:$PROXY_PORT"
    else
        echo "Warning: Proxy may not have started correctly"
    fi
}

# Check if proxy is running, start if not
if ! is_proxy_running; then
    start_proxy
fi

# Change back to original directory before launching claude
cd "$ORIGINAL_PWD"

# Run claude with proxy settings and bypass permissions
# Use separate HOME to keep config completely isolated from normal claude
# Use ANTHROPIC_AUTH_TOKEN (not ANTHROPIC_API_KEY) and set API_KEY to empty
export HOME="$GPT_CONFIG_DIR"
export ANTHROPIC_BASE_URL="http://localhost:$PROXY_PORT"
export ANTHROPIC_AUTH_TOKEN="proxy-mode"
export ANTHROPIC_API_KEY=""

# Configure for larger context window (GPT models support much larger contexts)
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000"
export MAX_THINKING_TOKENS="64000"

claude --dangerously-skip-permissions "$@"
