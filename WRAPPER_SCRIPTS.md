# Wrapper Scripts Guide

This repository provides 4 wrapper scripts for different Claude Code usage patterns.

## The Four Wrappers

| Command | Sudo Access | Permission Prompts | Use Case |
|---------|-------------|-------------------|----------|
| `gpt` | ❌ No | ✅ Yes (asks) | Safe daily use with oversight |
| `GPT` | ❌ No | ❌ No (`--dangerously-skip-permissions`) | Fast iteration, no sudo needed |
| `sudo-gpt` | ✅ Yes* | ✅ Yes (asks) | System tasks with oversight |
| `sudo-GPT` | ✅ Yes* | ❌ No (`--dangerously-skip-permissions`) | Maximum automation + sudo |

*Requires passwordless sudo setup (run `./setup-sudo.sh` first)

## How It Works

### Without Sudo (gpt & GPT)

These wrappers launch Claude Code normally. If Claude Code tries to run a `sudo` command:
- **Result**: Command fails with "sudo: a terminal is required to read the password"
- **Why**: No interactive TTY for password input
- **Effect**: Claude Code effectively cannot use sudo

**This is by design** - it prevents Claude Code from running elevated commands.

### With Sudo (sudo-gpt & sudo-GPT)

These wrappers check if passwordless sudo is configured. If configured:
- **Result**: Sudo commands execute without password prompts
- **Why**: NOPASSWD in /etc/sudoers.d/claude-code
- **Effect**: Claude Code can run specific commands as root

**Important**: These require setup first:
```bash
./setup-sudo.sh
```

## Setup Instructions

### 1. For Regular Use (No Sudo)

```bash
# Reload your shell config
source ~/.bashrc

# Use gpt or GPT immediately
gpt    # or GPT
```

No additional setup needed. Sudo commands will fail (which is safe).

### 2. For Sudo-Enabled Use

```bash
# Run the setup script
./setup-sudo.sh

# This creates /etc/sudoers.d/claude-code
# Enables NOPASSWD for safe commands only

# Reload your shell
source ~/.bashrc

# Now use sudo-gpt or sudo-GPT
sudo-gpt    # or sudo-GPT
```

## Security Model

### What gpt/GPT Can Do

✅ Read any file you can read
✅ Write to directories you own
✅ Install packages to your home directory (pip install --user, npm install)
✅ Modify your own files
✅ Run user-level services

❌ Cannot install system packages
❌ Cannot modify system files
❌ Cannot change system services
❌ Cannot access other users' files

### What sudo-gpt/sudo-GPT Can Do (After Setup)

Everything above, PLUS:

✅ Install system packages (pacman -S, apt install)
✅ Create directories in system locations (mkdir /opt/myapp)
✅ Change file permissions (chmod, chown)
✅ Start/stop/restart system services (systemctl)
✅ View system logs (journalctl)

❌ Cannot destroy data (rm -rf, dd, mkfs not in NOPASSWD list)
❌ Cannot modify firewall (iptables, ufw not in NOPASSWD list)
❌ Cannot add users (useradd not in NOPASSWD list)
❌ Cannot shutdown system (shutdown, reboot not in NOPASSWD list)

See [SUDO_SETUP.md](SUDO_SETUP.md) for the complete list of allowed commands.

## When to Use Which

### Use `gpt` when:
- Learning and exploring
- Working with your own code
- You want to review every action
- You don't need system-level changes

### Use `GPT` when:
- You trust the task completely
- Rapid iteration/development
- You don't need system-level changes
- You're comfortable with automatic execution

### Use `sudo-gpt` when:
- Installing packages
- Setting up development environments
- Configuring system services
- You want oversight of each step

### Use `sudo-GPT` when:
- Automating infrastructure setup
- Running deployment scripts
- You trust the automation fully
- Maximum speed is desired

⚠️ **Warning**: `sudo-GPT` is the most dangerous option - no permission prompts AND sudo access!

## Testing

### Test Regular Wrappers (No Sudo)

```bash
# Start gpt
gpt

# Ask it to run: sudo mkdir /tmp/test
# Expected: Command should FAIL with TTY error
# This proves sudo is blocked
```

### Test Sudo Wrappers (After Setup)

```bash
# Start sudo-gpt
sudo-gpt

# Ask it to run: sudo mkdir /tmp/test
# Expected: Command should SUCCEED without password
# This proves sudo access is working
```

## Switching Between Modes

You can run different commands in different terminals:

```bash
# Terminal 1: Safe mode, no sudo
gpt

# Terminal 2: Sudo mode for system setup
sudo-GPT

# Terminal 3: Regular terminal for manual commands
# (your normal shell)
```

Each is independent.

## Removing Sudo Access

If you enabled sudo but want to disable it:

```bash
sudo rm /etc/sudoers.d/claude-code
```

After removal:
- `gpt` and `GPT` - still work (they never needed sudo)
- `sudo-gpt` and `sudo-GPT` - will warn and fail when trying sudo

To re-enable, just run `./setup-sudo.sh` again.

## Aliases Setup

The aliases are defined in `~/.bashrc_custom`:

```bash
alias gpt='/home/tim/claude-code-proxy/gpt-wrapper.sh'
alias GPT='/home/tim/claude-code-proxy/GPT-wrapper.sh'
alias sudo-gpt='/home/tim/claude-code-proxy/sudo-gpt-wrapper.sh'
alias sudo-GPT='/home/tim/claude-code-proxy/sudo-GPT-wrapper.sh'
```

After any changes, reload with:
```bash
source ~/.bashrc
```

## Files

- `gpt-wrapper.sh` - Regular, asks for permissions
- `GPT-wrapper.sh` - Regular, skip permissions
- `sudo-gpt-wrapper.sh` - Sudo-enabled, asks for permissions
- `sudo-GPT-wrapper.sh` - Sudo-enabled, skip permissions
- `setup-sudo.sh` - Configure passwordless sudo
- `SUDO_SETUP.md` - Detailed sudo security guide

## FAQ

**Q: Can gpt/GPT use sudo if I type my password manually?**
A: No. There's no interactive terminal for password input. The command will always fail.

**Q: Does sudo-gpt require me to type my password?**
A: No. If setup correctly, passwordless sudo is enabled for specific commands only.

**Q: Can I customize which commands get NOPASSWD?**
A: Yes. Edit `/etc/sudoers.d/claude-code` or modify `setup-sudo.sh` before running it.

**Q: Is NOPASSWD global or per-wrapper?**
A: Global. Once enabled, it affects ALL terminal sessions, not just Claude Code.

**Q: So I can use sudo without password in my regular terminal too?**
A: Yes, for the commands listed in `/etc/sudoers.d/claude-code` only.

**Q: What if I run `sudo gpt` (sudo space gpt)?**
A: This runs the wrapper itself with sudo (unnecessary). Use `sudo-gpt` (the dedicated alias).

**Q: Can I run sudo-GPT without setting up NOPASSWD?**
A: It will warn you and ask if you want to continue. Sudo commands will fail if you proceed.

## Best Practices

1. **Start with `gpt`** - Get familiar with Claude Code without risk
2. **Move to `GPT`** when comfortable - Faster iteration
3. **Only use sudo-* when needed** - Don't leave sudo access enabled all the time
4. **Review setup-sudo.sh** before running - Understand what it enables
5. **Remove sudo access** when done - `sudo rm /etc/sudoers.d/claude-code`
6. **Audit regularly** - Check `/etc/sudoers.d/claude-code` periodically

## Troubleshooting

**"Command not found: gpt"**
- Run: `source ~/.bashrc`
- Check: `cat ~/.bashrc_custom` for aliases

**sudo-gpt warns about missing NOPASSWD**
- Run: `./setup-sudo.sh`
- Verify: `sudo -n true` (should not ask for password)

**Sudo commands still ask for password**
- Check: `ls -la /etc/sudoers.d/claude-code` (should exist)
- Verify: `sudo cat /etc/sudoers.d/claude-code` (check contents)
- Test: `sudo -n mkdir /tmp/test` (should work without password)

## Summary

You now have full control:
- **gpt/GPT** = Safe mode (no sudo)
- **sudo-gpt/sudo-GPT** = Power mode (sudo enabled)

Choose the right tool for the job!
