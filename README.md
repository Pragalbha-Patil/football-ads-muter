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

You do not need to install them manually when using the `uv run --with ...` command below.

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
- `--no-debug`: disable the OpenCV preview window and use terminal logs only.

Stop early with `Ctrl+C` in the terminal.

## Logs

The script prints one line per sampled frame, including:

- `Pitch`: green-pitch score.
- `Scoreboard`: whether a scoreboard-like top overlay was detected.
- `BoardDensity` / `BoardTile`: scoreboard detector debug values.
- `Ball`: experimental white-ball candidate signal, logged for debugging.
- `ReplayRule`: whether green replay/play footage was detected without relying on the scoreboard.
- `Context`: whether recent football context is protecting closeups/cutaways.
- `ScoreboardContext`: whether the scoreboard is helping classify the frame as football.
- `Muted`: current audio state.

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
