#!/usr/bin/env sh
set -eu

repo_url="https://github.com/Pragalbha-Patil/football-ads-muter.git"
install_root="${HOME}/football-ads-muter"

if ! command -v git >/dev/null 2>&1; then
  echo "Git is required. Install it with your OS package manager, then run this script again." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
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

Windows supports browser audio muting through pycaw. macOS/Linux can still collect data and train models, but audio-session muting is not implemented.

Run the starter model with:
uv run python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --mute-after 3 --record-data data/live-test --no-debug
EOF
