[CmdletBinding()]
param(
  [string]$ClaudeArgs = ""
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
if (-not $env:ANTHROPIC_API_KEY) { $env:ANTHROPIC_API_KEY = "any-value" }

# Best-effort config isolation. Claude Code on Windows may still use %APPDATA%.
$env:HOME = $claudeConfigDir

$claude = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claude) {
  throw "'claude' not found on PATH. Install @anthropics/claude-code and ensure it's available in your terminal."
}

$argList = @("--dangerously-skip-permissions", "--output-format", "stream-json", "--max-output-tokens", "131072")
if ($ClaudeArgs) {
  # Allow passing through raw args as a single string.
  $argList += ($ClaudeArgs -split ' ')
}

& $claude.Source @argList
