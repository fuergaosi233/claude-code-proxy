#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Activate the virtual environment
source .venv/bin/activate

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading configuration from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the proxy server
echo "Starting Claude Code Proxy on http://localhost:${PORT:-8082}"
echo "Using Azure OpenAI endpoint: ${OPENAI_BASE_URL}"
echo ""
echo "To use with Claude Code, run:"
echo "  ANTHROPIC_BASE_URL=http://localhost:${PORT:-8082} ANTHROPIC_API_KEY=\"any-value\" claude"
echo ""

python start_proxy.py
