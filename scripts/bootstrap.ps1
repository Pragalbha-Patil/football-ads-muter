$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/Pragalbha-Patil/football-ads-muter.git"
$InstallRoot = Join-Path $HOME "football-ads-muter"

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "git")) {
    if (Test-Command "winget") {
        Write-Host "Installing Git..."
        winget install --id Git.Git -e --source winget
        $env:Path = "$env:ProgramFiles\Git\cmd;$env:Path"
    } else {
        throw "Git is required and winget was not found. Install Git from https://git-scm.com/downloads, then run this script again."
    }
}

if (-not (Test-Command "git")) {
    throw "Git installation did not appear on PATH. Open a new PowerShell window and run this script again."
}

if (-not (Test-Command "uv")) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$HOME\.local\bin;$env:Path"
}

if (-not (Test-Command "uv")) {
    throw "uv installation did not appear on PATH. Open a new PowerShell window and run this script again."
}

if (Test-Path $InstallRoot) {
    Write-Host "Updating $InstallRoot..."
    git -C $InstallRoot pull --ff-only
} else {
    Write-Host "Cloning into $InstallRoot..."
    git clone $RepoUrl $InstallRoot
}

Set-Location $InstallRoot
uv sync

Write-Host ""
Write-Host "Ready."
Write-Host "Audio muting is most precise on Windows. Linux uses pactl with PulseAudio or PipeWire Pulse. macOS uses built-in osascript for system output mute/restore."
Write-Host ""
Write-Host "Run the starter model with:"
Write-Host "uv run python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --mute-after 3 --record-data data/live-test --no-debug"
