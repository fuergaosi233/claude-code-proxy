# Claude Code Proxy - Binary Packaging Guide

This document provides detailed instructions for packaging the Claude Code Proxy Python project into binary executables.

## Packaging Overview

We use PyInstaller to package the Python FastAPI application into a standalone binary executable that can run without requiring Python to be installed on the target system.

### Generated Binary Files

- **Directory Version**: `dist/claude-code-proxy/` (6.4MB executable + 49MB dependencies)
- **Single-file Version**: `dist/claude-code-proxy-single` (24MB standalone executable)

## Environment Requirements

### Development Environment

- Python 3.9+
- uv package manager
- Operating Systems: Linux/macOS/Windows

### Installing Packaging Tools

```bash
# Install PyInstaller using uv
uv add --dev pyinstaller
```

## Packaging Steps

### 1. Directory Version Packaging (Recommended for development/testing)

```bash
# Use custom spec file
uv run pyinstaller claude-proxy.spec
```

### 2. Single-file Version Packaging (Recommended for deployment)

```bash
# Quick single-file packaging
uv run pyinstaller --onefile --name claude-code-proxy-single src/main.py
```

### 3. Custom Configuration Packaging

```bash
# Include specific modules
uv run pyinstaller --onefile --hidden-import=src.api.endpoints src/main.py

# Include data files
uv run pyinstaller --onefile --add-data="src:src" src/main.py
```

## PyInstaller Configuration Details

### claude-proxy.spec Configuration

```python
# Main configuration items explanation
a = Analysis(
    ['src/main.py'],           # Entry file
    pathex=['.'],              # Search path
    binaries=[],               # Binary files
    datas=[('src', 'src')],    # Data files
    hiddenimports=[...],       # Hidden import modules
)

# Key hidden imports included
hiddenimports=[
    'src.api.endpoints',
    'src.core.config',
    'src.core.client',
    'uvicorn.logging',
    'uvicorn.loops.auto',
    'fastapi.openapi.utils',
    'pydantic.v1',
]
```

## Cross-platform Packaging

### Linux (Current Environment)

```bash
# Build on Linux system
uv run pyinstaller --onefile src/main.py
# Output: claude-code-proxy-single (Linux ELF)
```

### Windows

```bash
# Build on Windows system
pyinstaller --onefile src/main.py
# Output: claude-code-proxy-single.exe
```

### macOS

```bash
# Build on macOS system
pyinstaller --onefile src/main.py
# Output: claude-code-proxy-single (macOS executable)
```

## Usage Instructions

### Basic Execution

```bash
# Directory version
./dist/claude-code-proxy/claude-code-proxy

# Single-file version
./dist/claude-code-proxy-single
```

### Environment Variable Configuration

```bash
# Set required environment variables
export OPENAI_API_KEY="your-api-key-here"

# Optional environment variables
export ANTHROPIC_API_KEY="anthropic-key"
export HOST="0.0.0.0"
export PORT="8082"
export LOG_LEVEL="INFO"
```

### Viewing Help

```bash
./claude-code-proxy --help
```

### Starting the Server

```bash
# Run in foreground
OPENAI_API_KEY="your-key" ./claude-code-proxy

# Run in background
OPENAI_API_KEY="your-key" nohup ./claude-code-proxy &
```

## Deployment Instructions

### System Requirements

- **Memory**: Minimum 512MB RAM
- **Storage**: Minimum 100MB available space
- **Network**: Access to OpenAI API
- **Permissions**: No root privileges required

### Production Environment Deployment

```bash
# 1. Upload binary file to server
scp claude-code-proxy-single user@server:/opt/claude-proxy/

# 2. Set environment variables
cat > /opt/claude-proxy/.env << EOF
OPENAI_API_KEY=sk-your-actual-key
ANTHROPIC_API_KEY=your-anthropic-key
HOST=0.0.0.0
PORT=8082
LOG_LEVEL=WARNING
EOF

# 3. Create system service (systemd)
cat > /etc/systemd/system/claude-proxy.service << EOF
[Unit]
Description=Claude Code Proxy
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/claude-proxy
EnvironmentFile=/opt/claude-proxy/.env
ExecStart=/opt/claude-proxy/claude-code-proxy-single
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 4. Start service
systemctl daemon-reload
systemctl start claude-proxy
systemctl enable claude-proxy
```

### Docker Container Deployment

```dockerfile
FROM alpine:latest
RUN apk add --no-cache libstdc++
COPY claude-code-proxy-single /usr/local/bin/claude-code-proxy
EXPOSE 8082
CMD ["claude-code-proxy"]
```

## Troubleshooting

### Common Issues

1. **Permission Errors**

   ```bash
   # Add execution permissions
   chmod +x claude-code-proxy
   ```

2. **Missing Dependencies Error**

   - Regenerate spec file to include more hidden imports
   - Check PyInstaller version compatibility

3. **Environment Variables Not Set**

   ```bash
   # Check environment variables
   env | grep -E "(OPENAI|ANTHROPIC)"
   ```

4. **Port Already in Use**

   ```bash
   # Find process using port
   lsof -i :8082
   # Or use different port
   PORT=8083 ./claude-code-proxy
   ```

5. **Insufficient Memory**
   - Single-file version requires more memory for decompression
   - Consider using directory version

### Log Debugging

```bash
# Enable detailed logging
LOG_LEVEL=DEBUG ./claude-code-proxy

# View system logs
journalctl -u claude-proxy -f
```

## Performance Optimization

### Reducing Binary Size

1. **Using UPX Compression** (Windows only)

   ```bash
   pyinstaller --onefile --upx-dir=/path/to/upx src/main.py
   ```

2. **Removing Unnecessary Dependencies**

   - Clean up requirements.txt
   - Use --exclude to exclude large modules

3. **Selective Imports**
   - Import only required modules
   - Avoid wildcard imports

### Startup Optimization

```bash
# Preload optimization
export PYTHONOPTIMIZE=2

# Memory optimization
export PYTHONDONTWRITEBYTECODE=1
```

## File Structure Explanation

```
dist/
├── claude-code-proxy/           # Directory version
│   ├── claude-code-proxy        # Main executable (6.4MB)
│   └── _internal/               # Dependencies (49MB)
│       ├── base_library.zip
│       ├── libpython3.10.so.1.0
│       └── ...
└── claude-code-proxy-single     # Single-file version (24MB)
```

## Version Management

### Updating Binaries

1. Rebuild the project
2. Backup old version
3. Deploy new version
4. Restart service

### Rollback Strategy

```bash
# Keep previous version
cp claude-code-proxy claude-code-proxy.bak

# Rollback command
mv claude-code-proxy.bak claude-code-proxy
systemctl restart claude-proxy
```

## Security Considerations

1. **API Key Security**

   - Use environment variables instead of hardcoding
   - Rotate keys regularly
   - Restrict file permissions

2. **Network Security**

   - Use HTTPS
   - Configure firewall rules
   - Monitor access logs

3. **System Security**
   - Run under non-root user
   - Update system regularly
   - Monitor resource usage

## Next Steps

1. **Testing**: Validate functionality in target environment
2. **Monitoring**: Set up log rotation and monitoring
3. **Backup**: Regularly backup configurations and binaries
4. **Documentation**: Update deployment and maintenance documentation

## Support Contact

If you encounter issues, please check:

1. System logs: `journalctl -u claude-proxy`
2. Application logs: Check stdout/stderr output
3. Network connectivity: Test API endpoint connectivity
4. Dependency integrity: Verify binary file integrity

---

_Last updated: August 22, 2024_
_PyInstaller version: 6.15.0_
