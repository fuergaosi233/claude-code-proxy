[CmdletBinding()]
param(
  [string[]]$ClaudeArgs
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pidFile = Join-Path $repoRoot ".proxy.pid"
$logFile = Join-Path $repoRoot "proxy.log"
$claudeConfigDir = Join-Path $repoRoot ".claude-gpt-config"

function Test-ProxyRunning {
  if (!(Test-Path $pidFile)) { return $false }
  $pid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if (!$pid) { return $false }
  return [bool](Get-Process -Id $pid -ErrorAction SilentlyContinue)
}

function Start-Proxy {
  if (Test-ProxyRunning) { return }

  New-Item -ItemType Directory -Force -Path $claudeConfigDir | Out-Null

  $venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
  $py = if (Test-Path $venvPython) { $venvPython } else { "python" }

  $p = Start-Process -FilePath $py -ArgumentList @("start_proxy.py") -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $logFile -RedirectStandardError $logFile -PassThru
  Set-Content -Path $pidFile -Value $p.Id

  Start-Sleep -Seconds 1
}

Start-Proxy

$env:ANTHROPIC_BASE_URL = "http://localhost:8082"
$env:ANTHROPIC_AUTH_TOKEN = "proxy-mode"
$env:ANTHROPIC_API_KEY = ""

# Best-effort config isolation. Claude Code on Windows may still use %APPDATA%.
$env:HOME = $claudeConfigDir

# Configure for larger context window (matches bash wrapper intent)
$env:CLAUDE_CODE_MAX_OUTPUT_TOKENS = "128000"
$env:MAX_THINKING_TOKENS = "64000"

$claude = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claude) {
  throw "'claude' not found on PATH. Install @anthropics/claude-code and ensure it's available in your terminal."
}

if ($null -eq $ClaudeArgs) { $ClaudeArgs = @() }

& $claude.Source @ClaudeArgs
