#!/bin/bash

# gpt: Claude Code with permission prompts (no sudo)
# Safe daily use with oversight

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROXY_DIR="$SCRIPT_DIR"
PROXY_PORT=8082
PID_FILE="$PROXY_DIR/.proxy.pid"
GPT_CONFIG_DIR="$PROXY_DIR/.claude-gpt-config"
ORIGINAL_PWD="$(pwd)"

mkdir -p "$GPT_CONFIG_DIR"

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

start_proxy() {
    echo "Starting Claude Code Proxy (Azure OpenAI)..."
    (
        cd "$PROXY_DIR"
        source .venv/bin/activate
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

if ! is_proxy_running; then
    start_proxy
fi

cd "$ORIGINAL_PWD"

export HOME="$GPT_CONFIG_DIR"
export ANTHROPIC_BASE_URL="http://localhost:$PROXY_PORT"
export ANTHROPIC_AUTH_TOKEN="proxy-mode"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000"
export MAX_THINKING_TOKENS="64000"

if [ "$EUID" -eq 0 ]; then
    if [ -n "$SUDO_USER" ]; then
        exec sudo -u "$SUDO_USER" \
            HOME="$GPT_CONFIG_DIR" \
            ANTHROPIC_BASE_URL="http://localhost:$PROXY_PORT" \
            ANTHROPIC_AUTH_TOKEN="proxy-mode" \
            ANTHROPIC_API_KEY="" \
            CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000" \
            MAX_THINKING_TOKENS="64000" \
            claude "$@"
    else
        echo "ERROR: Running as root without sudo is not supported."
        exit 1
    fi
else
    claude "$@"
fi
