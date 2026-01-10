#!/bin/bash

# sudo-GPT: Claude Code with sudo access (--dangerously-skip-permissions)
# This variant allows sudo commands to execute with NOPASSWD if configured
# Requires: /etc/sudoers.d/claude-code to be set up (run ./setup-sudo.sh)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROXY_DIR="$SCRIPT_DIR"
PROXY_PORT=8082
PID_FILE="$PROXY_DIR/.proxy.pid"
GPT_CONFIG_DIR="$PROXY_DIR/.claude-gpt-config"
ORIGINAL_PWD="$(pwd)"

# Check if NOPASSWD is configured
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  WARNING: sudo-GPT requires passwordless sudo to be configured"
    echo ""
    echo "Claude Code will fail when trying to run sudo commands."
    echo ""
    echo "To enable sudo support, run:"
    echo "  cd $PROXY_DIR"
    echo "  ./setup-sudo.sh"
    echo ""
    echo "Or use regular 'GPT' command if you don't need sudo."
    echo ""
    read -p "Continue anyway? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        exit 0
    fi
fi

# Create separate config directory
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
    (
        cd "$PROXY_DIR"
        source venv/bin/activate
        nohup python start_proxy.py > "$PROXY_DIR/proxy.log" 2>&1 &
        echo $! > "$PID_FILE"
    )
    sleep 2
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

cd "$ORIGINAL_PWD"

# Run claude with proxy settings (SKIP permission prompts)
export HOME="$GPT_CONFIG_DIR"
export ANTHROPIC_BASE_URL="http://localhost:$PROXY_PORT"
export ANTHROPIC_AUTH_TOKEN="proxy-mode"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000"
export MAX_THINKING_TOKENS="64000"

# Check if running as root and drop privileges if needed
if [ "$EUID" -eq 0 ]; then
    if [ -n "$SUDO_USER" ]; then
        exec sudo -u "$SUDO_USER" \
            HOME="$GPT_CONFIG_DIR" \
            ANTHROPIC_BASE_URL="http://localhost:$PROXY_PORT" \
            ANTHROPIC_AUTH_TOKEN="proxy-mode" \
            ANTHROPIC_API_KEY="" \
            CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000" \
            MAX_THINKING_TOKENS="64000" \
            claude --dangerously-skip-permissions "$@"
    else
        echo "⚠️  ERROR: Running as root without sudo is not supported."
        exit 1
    fi
else
    claude --dangerously-skip-permissions "$@"
fi
