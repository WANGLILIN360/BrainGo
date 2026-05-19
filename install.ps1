# BrainDB installation script for Windows (PowerShell)
#
# Usage:
#   .\install.ps1              # Build and install everything
#   .\install.ps1 cli          # Build CLI only
#   .\install.ps1 server       # Build Server only
#   .\install.ps1 python       # Build Python wheel only
#   .\install.ps1 release      # Build optimized release binaries

param([string]$Target = "all")

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "target\debug"
$ReleaseDir = Join-Path $ScriptDir "target\release"
$BinDir = Join-Path $env:USERPROFILE ".local\bin"

# Ensure cargo is in PATH
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

function Build-Cli {
    Write-Host "Building braindb-cli..." -ForegroundColor Cyan
    cargo build --features cli --no-default-features @args
    if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }
    Copy-Item (Join-Path $BuildDir "braindb-cli.exe") $BinDir -Force
    Write-Host "braindb-cli installed to $BinDir" -ForegroundColor Green
}

function Build-Server {
    Write-Host "Building braindb-server..." -ForegroundColor Cyan
    cargo build --features server --no-default-features @args
    if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }
    Copy-Item (Join-Path $BuildDir "braindb-server.exe") $BinDir -Force
    Write-Host "braindb-server installed to $BinDir" -ForegroundColor Green
}

function Build-Python {
    Write-Host "Building braindb Python package..." -ForegroundColor Cyan
    pip install maturin
    maturin develop --features pyo3-extension-module --no-default-features
    Write-Host "braindb Python package installed" -ForegroundColor Green
}

function Build-Release {
    Write-Host "Building release binaries..." -ForegroundColor Cyan
    cargo build --release --features cli,server --no-default-features
    if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir -Force | Out-Null }
    Copy-Item (Join-Path $ReleaseDir "braindb-cli.exe") $BinDir -Force
    Copy-Item (Join-Path $ReleaseDir "braindb-server.exe") $BinDir -Force
    Write-Host "Release binaries installed to $BinDir" -ForegroundColor Green
}

switch ($Target) {
    "cli"     { Build-Cli }
    "server"  { Build-Server }
    "python"  { Build-Python }
    "release" { Build-Release }
    "all" {
        Build-Cli
        Build-Server
        Write-Host ""
        Write-Host "BrainDB installed! Commands:" -ForegroundColor Yellow
        Write-Host "  braindb-cli build -o net.braindb -n 100"
        Write-Host "  braindb-cli info net.braindb"
        Write-Host "  braindb-cli run net.braindb -d 100 --stimulus `"0:30`""
        Write-Host "  braindb-cli query net.braindb downstream 0"
        Write-Host "  braindb-server  # -> http://localhost:3000"
    }
    default {
        Write-Host "Usage: .\install.ps1 [cli|server|python|release|all]" -ForegroundColor Red
        exit 1
    }
}
