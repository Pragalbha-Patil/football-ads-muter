#!/usr/bin/env sh
set -eu

repo_url="https://github.com/Pragalbha-Patil/football-ads-muter.git"
install_root="${HOME}/football-ads-muter"

has_command() {
  command -v "$1" >/dev/null 2>&1
}

run_with_privilege() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif has_command sudo; then
    sudo "$@"
  elif has_command doas; then
    doas "$@"
  else
    echo "Need root privileges to install system packages, but sudo/doas was not found." >&2
    return 1
  fi
}

install_system_packages() {
  packages="$*"

  if [ -z "$packages" ]; then
    return 0
  fi

  if has_command apt-get; then
    run_with_privilege apt-get update
    run_with_privilege apt-get install -y $packages
  elif has_command dnf; then
    run_with_privilege dnf install -y $packages
  elif has_command pacman; then
    run_with_privilege pacman -Sy --needed --noconfirm $packages
  elif has_command zypper; then
    run_with_privilege zypper install -y $packages
  elif has_command apk; then
    run_with_privilege apk add $packages
  elif has_command brew; then
    brew install $packages
  else
    echo "No supported package manager found for: $packages" >&2
    return 1
  fi
}

download_to_stdout() {
  url="$1"

  if has_command curl; then
    curl -LsSf "$url"
  elif has_command wget; then
    wget -qO- "$url"
  elif has_command fetch; then
    fetch -qo- "$url"
  else
    echo "Need curl, wget, or fetch to download $url" >&2
    return 1
  fi
}

system_name="$(uname -s 2>/dev/null || echo unknown)"

if ! has_command curl && ! has_command wget && ! has_command fetch; then
  echo "Installing curl..."
  case "$system_name" in
    Darwin) install_system_packages curl || {
      echo "Install curl, wget, fetch, or Homebrew, then rerun this script." >&2
      exit 1
    } ;;
    *) install_system_packages curl || {
      echo "Install curl, wget, or fetch with your OS package manager, then rerun this script." >&2
      exit 1
    } ;;
  esac
fi

if ! has_command git; then
  echo "Installing git..."
  install_system_packages git || {
    echo "Install git with your OS package manager, then rerun this script." >&2
    exit 1
  }
fi

if ! has_command uv; then
  echo "Installing uv..."
  download_to_stdout https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ "$system_name" = "Linux" ] && ! has_command pactl; then
  echo "Installing pactl for Linux audio muting..."
  if has_command apt-get || has_command dnf || has_command zypper; then
    install_system_packages pulseaudio-utils || echo "Could not install pulseaudio-utils. Install pactl manually for Linux audio muting."
  elif has_command pacman; then
    install_system_packages libpulse || echo "Could not install libpulse. Install pactl manually for Linux audio muting."
  elif has_command apk; then
    install_system_packages pulseaudio-utils || echo "Could not install pulseaudio-utils. Install pactl manually for Linux audio muting."
  elif has_command brew; then
    install_system_packages pulseaudio || echo "Could not install pulseaudio. Install pactl manually for Linux audio muting."
  else
    echo "pactl not found and no supported package manager was detected. Linux muting will be unavailable until pactl is installed."
  fi
fi

if [ "$system_name" = "Darwin" ] && ! has_command osascript; then
  echo "osascript was not found. It is normally built into macOS; audio muting will be unavailable."
fi

if [ -d "$install_root/.git" ]; then
  echo "Updating ${install_root}..."
  git -C "$install_root" pull --ff-only
else
  echo "Cloning into ${install_root}..."
  git clone "$repo_url" "$install_root"
fi

cd "$install_root"
uv sync

cat <<'EOF'

Ready.

Audio muting is most precise on Windows. Linux uses pactl with PulseAudio or PipeWire Pulse. macOS uses built-in osascript for system output mute/restore.

Run the starter model with:
uv run python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --mute-after 3 --record-data data/live-test --no-debug
EOF
