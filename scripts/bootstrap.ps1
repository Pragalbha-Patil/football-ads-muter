$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/Pragalbha-Patil/football-ads-muter.git"
$InstallRoot = Join-Path $HOME "football-ads-muter"

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command "git")) {
    throw "Git is required. Install it from https://git-scm.com/downloads, then run this script again."
}

if (-not (Test-Command "uv")) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$HOME\.local\bin;$env:Path"
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
Write-Host "Run the starter model with:"
Write-Host "uv run python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --mute-after 3 --record-data data/live-test --no-debug"
