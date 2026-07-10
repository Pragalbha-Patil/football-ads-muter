# Football Ads Muter

Detects whether a browser stream is showing football or ads by sampling the screen, then fades browser audio down during ads and restores it instantly when football returns.

## Table of contents

- [Requirements](#requirements)
- [Install from GitHub](#install-from-github)
- [Run](#run)
- [Quickstart with starter model](#quickstart-with-starter-model)
- [Local model workflow](#local-model-workflow)
- [Self-training bootstrap](#self-training-bootstrap)
- [Football-only anomaly model](#football-only-anomaly-model)
- [Logs](#logs)
- [Tuning](#tuning)
- [Limitations](#limitations)

## Requirements

- `uv`
- A Chromium/Firefox browser audio session
- Windows, macOS, or Linux

The script uses:

- `opencv-python`
- `mss`
- `numpy`
- `pycaw` for Windows per-browser audio sessions
- `scikit-learn` and `joblib` for local model training/inference

Audio muting support:

- Windows: per-browser session mute/restore through `pycaw`.
- Linux: best-effort browser sink-input mute/restore through `pactl` with PulseAudio or PipeWire Pulse.
- macOS: best-effort system output mute/restore through `osascript`.

Linux package hints:

- Debian/Ubuntu: `sudo apt install pulseaudio-utils`
- Fedora: `sudo dnf install pulseaudio-utils`
- Arch: `sudo pacman -S libpulse`

On PipeWire systems, `pactl` usually talks to `pipewire-pulse`. macOS does not need an extra package for `osascript`.

You do not need to install them manually when using the `uv run --with ...` command below.

Recorded screenshots, logs, and trained model files are intentionally ignored by Git. They can contain private screen contents and should stay local unless you explicitly choose to share them.

If you prefer a project environment instead of per-command dependencies:

```powershell
uv sync
```

## Install from GitHub

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/Pragalbha-Patil/football-ads-muter/master/scripts/bootstrap.ps1 | iex
```

macOS/Linux:

```sh
curl -LsSf https://raw.githubusercontent.com/Pragalbha-Patil/football-ads-muter/master/scripts/bootstrap.sh | sh
```

If `curl` is not installed but `wget` is:

```sh
wget -qO- https://raw.githubusercontent.com/Pragalbha-Patil/football-ads-muter/master/scripts/bootstrap.sh | sh
```

The bootstrap scripts install or update the repo and run `uv sync`. They also try to install missing prerequisites when the platform has a supported package manager:

- Windows: installs Git through `winget` if needed, and installs `uv`.
- Linux: installs missing `curl`, `git`, `uv`, and `pactl` where possible.
- macOS: installs missing `curl`/`git` through Homebrew when available, and installs `uv`. `osascript` is built in.

System package installation may ask for `sudo`, `doas`, administrator approval, or package-manager confirmation.

Audio muting is most precise on Windows. Linux support requires `pactl` from packages such as `pulseaudio-utils` or `libpulse`. macOS support uses built-in `osascript` and currently changes the system output volume/mute state, not just the browser.

## Run

From this folder:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw python football_ad_muter.py --monitor 2 --duration 5400 --mute-after 2 --context-grace 4 --mute-fade 2 --no-debug
```

## Quickstart with starter model

This repo includes `models/football_one_class.joblib`, a starter one-class model trained on football/highlights screen captures. It learns "normal football" and treats sufficiently different screens as non-football.

Run it against the monitor where the browser stream is visible:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib --with scikit-learn python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --mute-after 3 --record-data data/live-test --no-debug
```

Tune from there:

- If it mutes during football, try `--model-threshold 0.45`.
- If it misses non-football breaks, try `--model-threshold 0.55` or `0.6`.

Only load `.joblib` model files from sources you trust. Joblib uses Python pickle under the hood.

The included model is a starting point, not a universal detector. For best accuracy, collect and train on your own browser, monitor, broadcaster, resolution, and display-scaling setup.

Useful options:

- `--monitor 2`: capture monitor 2.
- `--duration 5400`: run for 90 minutes.
- `--mute-after 2`: begin muting after 2 consecutive non-football seconds.
- `--context-grace 4`: keep closeups, crowd shots, and camera switches protected briefly after real play.
- `--mute-fade 2`: fade browser volume to zero over 2 seconds.
- `--record-data data/session-1`: save sampled browser screenshots and feature rows for training.
- `--auto-label`: fill the training label column with conservative automatic labels.
- `--model models/football_ad_classifier.joblib`: use a trained local model instead of the heuristic decision.
- `--model-threshold 0.65`: require this much model confidence before treating a frame as football.
- `--no-debug`: disable the OpenCV preview window and use terminal logs only.

Stop early with `Ctrl+C` in the terminal.

## Local model workflow

The football stream can keep running in the browser. The script samples the screen, records frames, and controls the browser audio session.

### 1. Record training data

Run a normal match session with recording enabled:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib python football_ad_muter.py --monitor 2 --duration 1800 --record-data data/session-1 --no-debug
```

For bootstrap training without manual labels, add `--auto-label`:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib python football_ad_muter.py --monitor 2 --duration 1800 --record-data data/session-1 --auto-label --no-debug
```

This creates:

```text
data/session-1/
  frames/
  labels.csv
  recording_config.json
```

`labels.csv` contains one row per sampled frame. The `label` column is intentionally blank.

### 2. Label examples

Use the labeling helper:

```powershell
uv run --with opencv-python python label_frames.py data/session-1/labels.csv
```

Keys:

- `f`: football
- `r`: replay
- `c`: closeup
- `d`: crowd
- `a`: ad
- `u`: unknown
- `s`: skip
- `n`: next
- `b`: back
- `q`: quit

You can also open `data/session-1/labels.csv` directly and fill the `label` column for useful rows.

Accepted football-like labels:

- `football`
- `replay`
- `closeup`
- `crowd`
- `play`

Accepted ad-like labels:

- `ad`
- `ads`
- `commercial`
- `break`

Rows labeled `unknown`, `skip`, `unsure`, `bad_frame`, or left blank are ignored during training.

You do not need to label every frame. Start with a balanced set, for example 100 football-like frames and 100 ad frames. Add more examples whenever the muter gets something wrong.

### 3. Train the model

```powershell
uv run --with scikit-learn --with joblib python train_model.py data/session-1/labels.csv --output models/football_ad_classifier.joblib
```

If you used `--auto-label`, the same command works because the `label` column is already filled. For older recordings with blank labels, you can bootstrap from the predicted column:

```powershell
uv run --with scikit-learn --with joblib python train_model.py data/session-1/labels.csv --use-predicted --output models/football_ad_classifier.joblib
```

You can train from multiple sessions:

```powershell
uv run --with scikit-learn --with joblib python train_model.py data/session-1/labels.csv data/session-2/labels.csv --output models/football_ad_classifier.joblib
```

The trainer prints label counts and, when there are enough examples, a validation report.

### 4. Run with the trained model

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib --with scikit-learn python football_ad_muter.py --monitor 2 --duration 5400 --model models/football_ad_classifier.joblib --model-threshold 0.65 --record-data data/session-2 --no-debug
```

Keeping `--record-data` on while using the model lets you collect the next batch of examples. Label mistakes and uncertain cases, retrain, then run again.

The improvement loop is:

```text
record browser frames -> label useful examples -> train -> run model -> collect mistakes -> retrain
```

## Self-training bootstrap

To let the project collect conservative automatic labels and retrain after each recording chunk:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with scikit-learn --with joblib python self_train.py --monitor 1 --chunk-duration 600
```

Use the monitor number where the browser stream is visible. The model is saved to `models/football_ad_classifier.joblib` and reused on the next collection cycle.

This is a bootstrap loop, not perfect supervision. It gets the model started, but the biggest accuracy gains still come from reviewing `data/self-train/*/labels.csv`, correcting wrong labels with `label_frames.py --all`, and retraining.

## Football-only anomaly model

If the other monitor is currently showing only football/highlights and no ads, train a one-class model instead. This learns "normal football" and treats future non-football screens as anomalies.

Record football-only examples:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib python football_ad_muter.py --monitor 2 --duration 600 --record-data data/football-only/session-1 --assume-label football --no-debug
```

Train the one-class model:

```powershell
uv run --with scikit-learn --with joblib python train_one_class.py data/football-only/session-1/labels.csv --output models/football_one_class.joblib
```

Run with it:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib --with scikit-learn python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --record-data data/football-only/session-2 --no-debug
```

## Logs

The script prints one line per sampled frame, including:

- `Pitch`: green-pitch score.
- `Scoreboard`: whether a scoreboard-like top overlay was detected.
- `BoardDensity` / `BoardTile`: scoreboard detector debug values.
- `Ball`: experimental white-ball candidate signal, logged for debugging.
- `ReplayRule`: whether green replay/play footage was detected without relying on the scoreboard.
- `Lines`: white pitch-line signal near green areas.
- `SceneCut`: frame-to-frame color-layout change score.
- `Context`: whether recent football context is protecting closeups/cutaways.
- `ScoreboardContext`: whether the scoreboard is helping classify the frame as football.
- `Muted`: current audio state.

Recorded training rows also include 3x3 regional layout features:

- `grid_green_*`: where green pitch appears in the frame.
- `grid_motion_*`: where motion appears in the frame.
- `grid_edge_*`: where visual structure/text/lines appear in the frame.

The trainers automatically use these newer columns when present. Older recordings still work; they train with the feature columns they contain.

## Tuning

If ads are classified as football, lower `--context-grace` or increase the green threshold constants in `football_ad_muter.py`.

If football closeups or replay shots get muted, increase `--context-grace` slightly, for example:

```powershell
--context-grace 6
```

If muting feels too abrupt or too slow, adjust:

```powershell
--mute-fade 3
```

## Limitations

- The detector looks at screen pixels, not video-stream metadata. Browser controls, end screens, buffering screens, overlays, menus, and DRM-protected/black captures can affect results.
- The included `models/football_one_class.joblib` is a starter model trained on one local browser/monitor/broadcast setup. It should be retrained for different broadcasters, resolutions, scorebugs, display scaling, or viewing habits.
- One-class mode learns "normal football" from football-only examples. It does not learn every possible ad or non-football state, so unusual football shots can look anomalous and sports-like ads can look normal.
- Training with `--assume-label football` is only safe when the capture really contains football/highlights only. If menus, desktop windows, ads, or paused screens are recorded, the anomaly model can learn those as normal too.
- The feature set is still visual and approximate. Green-pitch, pitch-line, scoreboard, motion, ball-ish, scene-change, and grid-layout signals can all be confused by replays, closeups, crowd shots, dark scenes, studio segments, sponsor boards, or fast highlight edits.
- The ball detector is experimental and noisy. It is useful as one feature among many, not as a standalone football signal.
- The scoreboard detector can miss small, transparent, animated, or unusually placed scoreboards. It can also mistake dense graphics for a scoreboard.
- Audio control precision depends on the OS. Windows targets browser audio sessions, Linux targets browser sink inputs when `pactl` can see them, and macOS currently controls system output volume/mute.
- If multiple tabs are playing audio in the same browser, they may be faded or restored together.
- If the process is force-stopped while audio is muted, the browser or system audio may remain muted or at low volume. Restore it from the OS volume controls if that happens.
- `.joblib` model files use pickle-style loading. Only load models from sources you trust.
