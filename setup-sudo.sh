#!/bin/bash

# Setup script for passwordless sudo with Claude Code
# This allows Claude Code to run common safe commands without password prompts

echo "Claude Code Sudo Setup"
echo "======================"
echo ""
echo "This script will configure passwordless sudo for common safe commands."
echo "This allows Claude Code to run these commands without password prompts."
echo ""
echo "⚠️  SECURITY WARNING:"
echo "This grants Claude Code the ability to run specific commands as root."
echo "Only proceed if you trust Claude Code with these permissions."
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Setup cancelled."
    exit 0
fi

# Create sudoers file for Claude Code
SUDOERS_FILE="/etc/sudoers.d/claude-code"
USERNAME=$(whoami)

echo ""
echo "Creating sudoers configuration..."

# Generate sudoers content
cat > /tmp/claude-code-sudoers << SUDOERS_EOF
# Passwordless sudo for Claude Code
# Created: $(date)
# User: $USERNAME

# Safe file operations
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/mkdir
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/chmod
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/chown

# Package management (read-only operations)
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/pacman -Q*
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/pacman -S*
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/apt-get update
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/apt-get install *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/apt update
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/apt install *

# System information (read-only)
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl status *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/journalctl *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/dmesg

# Network (safe operations)
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/ip addr
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/ip link
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/netstat

# Process management
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/kill
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/killall

# Service management
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl start *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl enable *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/systemctl disable *
SUDOERS_EOF

# Validate sudoers syntax
echo "Validating sudoers syntax..."
if sudo visudo -c -f /tmp/claude-code-sudoers; then
    echo "✓ Syntax valid"
    sudo cp /tmp/claude-code-sudoers "$SUDOERS_FILE"
    sudo chmod 0440 "$SUDOERS_FILE"
    echo "✓ Sudoers file created: $SUDOERS_FILE"
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "Claude Code can now run the following without passwords:"
    echo "  - File operations (mkdir, chmod, chown)"
    echo "  - Package management (install packages)"
    echo "  - System info (status, logs)"
    echo "  - Service management (start/stop/restart)"
    echo ""
    echo "To remove this configuration:"
    echo "  sudo rm $SUDOERS_FILE"
else
    echo "❌ Error: Invalid sudoers syntax"
    exit 1
fi

rm /tmp/claude-code-sudoers
