[CmdletBinding()]
param(
  [string]$Python = "python",
  [string]$VenvPath = "venv",
  [string]$HostAddress,
  [int]$Port
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot (Join-Path $VenvPath "Scripts\python.exe")

if (Test-Path $venvPython) {
  $py = $venvPython
} else {
  $py = $Python
}

if ($HostAddress) { $env:HOST = $HostAddress }
if ($Port) { $env:PORT = $Port }

& $py "start_proxy.py"
