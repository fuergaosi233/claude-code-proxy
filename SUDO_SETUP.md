# Sudo Configuration for Claude Code

When Claude Code runs commands that require elevated privileges, it may fail with:
```
sudo: a terminal is required to read the password
```

This happens because Claude Code has no interactive TTY to prompt for your password.

## Solutions

You have three options to handle sudo commands:

### Option 1: Interactive Sudo (Safest)

**Pros:** Maximum security, you approve each sudo command
**Cons:** Requires manual password entry

When Claude Code suggests a sudo command:
1. Copy the command
2. Run it yourself in a terminal
3. Enter your password when prompted

This is the safest approach as you maintain full control.

### Option 2: Passwordless Sudo for Safe Commands (Recommended)

**Pros:** Convenient, still secure for common operations
**Cons:** Requires one-time setup

Configure specific commands to run without passwords.

**Quick Setup:**

```bash
# Run the setup script
sudo /tmp/setup-sudo-for-claude.sh
```

This grants passwordless sudo for:
- File operations (`mkdir`, `chmod`, `chown`)
- Package management (`pacman -S`, `apt install`)
- System info (`systemctl status`, `journalctl`)
- Service management (`systemctl start/stop/restart`)

**Manual Setup:**

Create `/etc/sudoers.d/claude-code`:

```bash
# Replace 'youruser' with your username
youruser ALL=(ALL) NOPASSWD: /usr/bin/mkdir
youruser ALL=(ALL) NOPASSWD: /usr/bin/chmod
youruser ALL=(ALL) NOPASSWD: /usr/bin/chown
youruser ALL=(ALL) NOPASSWD: /usr/bin/systemctl
youruser ALL=(ALL) NOPASSWD: /usr/bin/pacman -S*
youruser ALL=(ALL) NOPASSWD: /usr/bin/apt-get install *
```

**Validate before saving:**
```bash
sudo visudo -c -f /etc/sudoers.d/claude-code
```

### Option 3: Full Passwordless Sudo (Dangerous)

**⚠️ WARNING: This is extremely dangerous and NOT recommended**

**Pros:** No password prompts ever
**Cons:** Complete root access without oversight

```bash
# Create /etc/sudoers.d/claude-code-dangerous
youruser ALL=(ALL) NOPASSWD: ALL
```

This gives Claude Code unrestricted root access. Only use in isolated/test environments.

## Security Considerations

### What Can Go Wrong?

With passwordless sudo, Claude Code could:
- Install/remove any software
- Modify system files
- Change network settings
- Delete important data
- Expose sensitive information

### Best Practices

1. **Limit Scope**: Only grant NOPASSWD for specific commands
2. **Review Commands**: Check what Claude Code wants to run before granting access
3. **Test Environment**: Try passwordless sudo in a VM first
4. **Regular Audit**: Review `/etc/sudoers.d/claude-code` periodically
5. **Remove When Done**: Delete the sudoers file when testing is complete

### Recommended Approach

For most users:
1. Use **Option 2** with limited commands
2. Review the sudoers file contents before applying
3. Only add commands you're comfortable automating
4. Remove access when you don't need it

### Safe Commands for NOPASSWD

Generally safe (read-only or low-risk):
- `systemctl status`
- `journalctl`
- `ip addr`, `ip link`
- `netstat`
- `pacman -Q` (query installed packages)

Moderate risk (can modify system but predictable):
- `mkdir`, `chmod`, `chown` (file operations)
- `systemctl start/stop/restart` (services)
- `pacman -S`, `apt install` (package installation)

High risk (avoid NOPASSWD):
- `rm`, `dd`, `mkfs` (data destruction)
- `iptables`, `ufw` (firewall changes)
- `useradd`, `usermod` (user management)
- `shutdown`, `reboot` (system control)

## Testing Your Setup

After configuring passwordless sudo:

```bash
# Test that sudo works without password
sudo mkdir /tmp/test-sudo-nopasswd

# If it doesn't ask for password, it works!
ls -la /tmp/test-sudo-nopasswd
sudo rm -rf /tmp/test-sudo-nopasswd
```

## Removing Passwordless Sudo

To remove the configuration:

```bash
sudo rm /etc/sudoers.d/claude-code
```

## Alternative: Sudo Askpass

For GUI environments, you can use an askpass helper:

```bash
# Install askpass (varies by distro)
sudo pacman -S openssh-askpass  # Arch
sudo apt install ssh-askpass     # Debian/Ubuntu

# Set environment variable
export SUDO_ASKPASS=/usr/bin/ssh-askpass

# Claude Code can now trigger password prompts
```

Add to your wrapper script:
```bash
export SUDO_ASKPASS=/usr/bin/ssh-askpass
export SUDO_ASKPASS_PROMPT="Claude Code needs sudo access"
```

## Troubleshooting

### "sudo: no tty present and no askpass program specified"

This means sudo needs a password but can't prompt for it.

**Solution**: Choose Option 2 (passwordless for specific commands) or Option 4 (askpass)

### Commands still ask for password

Check your sudoers file:
```bash
sudo cat /etc/sudoers.d/claude-code
```

Ensure:
1. Your username is correct
2. The command path is exact (use `which command`)
3. File permissions are 0440: `sudo chmod 0440 /etc/sudoers.d/claude-code`

### "syntax error" when editing sudoers

Always use `visudo` to edit:
```bash
sudo visudo -f /etc/sudoers.d/claude-code
```

Never edit directly with a text editor - syntax errors can lock you out!

## Summary

**For casual use**: Option 1 (manual)
**For regular use**: Option 2 (limited NOPASSWD)
**For development**: Option 2 with more commands
**Never**: Option 3 (full NOPASSWD) in production

The goal is to balance convenience with security. Start restrictive and only expand permissions as needed.
