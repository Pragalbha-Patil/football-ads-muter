# Football Ads Muter

Detects whether a browser stream is showing football or ads by sampling the screen, then fades browser audio down during ads and restores it instantly when football returns.

## Requirements

- Windows
- `uv`
- A Chromium/Firefox browser audio session

The script uses:

- `opencv-python`
- `mss`
- `numpy`
- `pycaw`
- `scikit-learn` and `joblib` for local model training/inference

You do not need to install them manually when using the `uv run --with ...` command below.

Recorded screenshots, logs, and trained model files are intentionally ignored by Git. They can contain private screen contents and should stay local unless you explicitly choose to share them.

## Run

From this folder:

```powershell
uv run --with opencv-python --with mss --with numpy --with pycaw python football_ad_muter.py --monitor 2 --duration 5400 --mute-after 2 --context-grace 4 --mute-fade 2 --no-debug
```

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
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib python football_ad_muter.py --monitor 2 --duration 5400 --model models/football_ad_classifier.joblib --model-threshold 0.65 --record-data data/session-2 --no-debug
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
uv run --with opencv-python --with mss --with numpy --with pycaw --with joblib python football_ad_muter.py --monitor 2 --model models/football_one_class.joblib --model-threshold 0.5 --record-data data/football-only/session-2 --no-debug
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

- Detection is heuristic. It looks at screen pixels, not the video stream metadata, so unusual camera angles, overlays, graphics, and sponsor boards can confuse it.
- Ads can be misclassified as football if they contain green backgrounds, high-motion sports clips, or scoreboard-like graphics near the top of the screen.
- Football can be misclassified as ads during player closeups, crowd shots, replay transitions, dark scenes, or shots where the pitch and scoreboard are both missing.
- The ball detector is experimental and noisy. It is logged for debugging, but it is not trusted as a standalone football signal.
- The scoreboard detector can miss small, transparent, animated, or unusually placed scoreboards. It can also mistake dense ad graphics for a scoreboard.
- Browser or DRM-protected video may capture as black, blank, or partially incorrect frames depending on hardware acceleration, browser settings, and the streaming site.
- Audio control works at the browser session level. If multiple tabs are playing audio in the same browser, they may be faded or restored together.
- If the process is force-stopped while audio is muted, Windows may leave the browser muted or at low volume. Restore it from the volume mixer if that happens.
- The script was tuned around one match stream and monitor setup. Different broadcasters, resolutions, scoreboards, or display scaling may need threshold changes.
