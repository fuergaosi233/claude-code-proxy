#!/bin/bash

# Azure OpenAI Plugin - Session Start Hook
# This hook automatically starts the Azure OpenAI proxy if enabled in settings

set -e

# Get the plugin directory (parent of hooks/)
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source environment if .env exists
if [ -f "$PLUGIN_DIR/.env" ]; then
    set -a
    source "$PLUGIN_DIR/.env"
    set +a
fi

# Check if auto-start is enabled (default: true if .env exists and has API key)
AUTO_START="${AZURE_OPENAI_AUTO_START:-true}"

# Only proceed if auto-start is enabled
if [ "$AUTO_START" != "true" ]; then
    exit 0
fi

# Check if .env is configured
if [ ! -f "$PLUGIN_DIR/.env" ] || [ -z "$OPENAI_API_KEY" ]; then
    # Not configured yet, skip silently
    exit 0
fi

# Check if proxy is already running
PID_FILE="$PLUGIN_DIR/.proxy.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        # Proxy already running, configure Claude Code to use it
        PROXY_PORT="${PORT:-8082}"
        export ANTHROPIC_BASE_URL="http://localhost:${PROXY_PORT}"
        export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-azure-openai-proxy}"
        exit 0
    else
        # Stale PID file, remove it
        rm -f "$PID_FILE"
    fi
fi

# Start the proxy in the background
cd "$PLUGIN_DIR"

# Check if virtual environment exists
if [ ! -d "$PLUGIN_DIR/venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
else
    source venv/bin/activate
fi

# Start the proxy
nohup python start_proxy.py > proxy.log 2>&1 &
PROXY_PID=$!
echo $PROXY_PID > "$PID_FILE"

# Wait a moment for the proxy to start
sleep 2

# Verify it started
if ! ps -p "$PROXY_PID" > /dev/null 2>&1; then
    echo "Failed to start Azure OpenAI proxy. Check proxy.log for details."
    rm -f "$PID_FILE"
    exit 1
fi

# Configure Claude Code to use the proxy
PROXY_PORT="${PORT:-8082}"
export ANTHROPIC_BASE_URL="http://localhost:${PROXY_PORT}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-azure-openai-proxy}"

echo "Azure OpenAI proxy started successfully (PID: $PROXY_PID)"
