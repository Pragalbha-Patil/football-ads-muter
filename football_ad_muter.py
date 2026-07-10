import argparse
import csv
import json
import platform
import re
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np


# -----------------------------
# CONFIG
# -----------------------------

FPS = 1

GREEN_THRESHOLD = 0.22
REPLAY_GREEN_THRESHOLD = 0.35
LOW_GREEN_CONTEXT_THRESHOLD = 0.08
BALL_CONTEXT_SECONDS = 6
SCOREBOARD_HOLD_SECONDS = 4
SCOREBOARD_MIN_GREEN_THRESHOLD = 0.06
AD_BREAK_GREEN_THRESHOLD = 0.08
AD_BREAK_LINE_THRESHOLD = 0.03
AD_BREAK_SCENE_CHANGE_THRESHOLD = 0.55

MUTE_AFTER_SECONDS = 2
UNMUTE_AFTER_SECONDS = 2
MUTE_FADE_SECONDS = 2.0
MUTE_FADE_STEPS = 8

SHOW_DEBUG = True

MODEL_THRESHOLD = 0.65
UNKNOWN_SECONDS = 999.0
GRID_ROWS = 3
GRID_COLS = 3

GRID_FEATURE_NAMES = [
    f"grid_{metric}_r{row}c{col}"
    for metric in ("green", "motion", "edge")
    for row in range(GRID_ROWS)
    for col in range(GRID_COLS)
]

FEATURE_NAMES = [
    "green",
    "scoreboard",
    "board_density",
    "board_tile",
    "motion_score",
    "motion",
    "ball",
    "replay_rule",
    "recent_context",
    "scoreboard_context",
    "scoreboard_support",
    "ad_break_reset",
    "seconds_since_strong",
    "seconds_since_scoreboard",
    "pitch_line_score",
    "scene_change_score",
    *GRID_FEATURE_NAMES,
]

CSV_FIELDS = [
    "timestamp",
    "frame",
    *FEATURE_NAMES,
    "heuristic_probability",
    "model_probability",
    "predicted",
    "muted",
    "label",
]

FOOTBALL_AUTO_LABEL_MIN_GREEN = 0.18
AD_AUTO_LABEL_MAX_GREEN = 0.08


def log(message):
    print(f"{time.strftime('%H:%M:%S')} | {message}", flush=True)


# -----------------------------
# Cross-platform browser mute
# -----------------------------

BROWSER_PROCESS_NAMES = ("chrome", "chromium", "msedge", "edge", "firefox", "brave", "opera", "vivaldi")
windows_browser_volumes = {}
macos_volume_state = None
linux_sink_states = {}


def browser_name_match(name):
    clean = name.lower()
    return any(browser in clean for browser in BROWSER_PROCESS_NAMES)


def windows_browser_audio_sessions():
    try:
        from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    except Exception as exc:
        log(f"pycaw unavailable for Windows browser audio control: {exc}")
        return []

    sessions = AudioUtilities.GetAllSessions()
    browser_sessions = []

    for session in sessions:
        if session.Process:
            name = session.Process.name().lower()
            if browser_name_match(name):
                volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                browser_sessions.append((session.Process.pid, name, volume))

    return browser_sessions


def mute_windows_browsers(mute: bool, fade_seconds=MUTE_FADE_SECONDS):
    sessions = windows_browser_audio_sessions()

    if not sessions:
        log("no browser audio session found")
        return

    if not mute:
        for pid, _, volume in sessions:
            volume.SetMute(False, None)
            volume.SetMasterVolume(windows_browser_volumes.get(pid, 1.0), None)

        log("browser volume restored instantly")
        return

    starts = []
    for pid, _, volume in sessions:
        current = volume.GetMasterVolume()
        windows_browser_volumes.setdefault(pid, current)
        volume.SetMute(False, None)
        starts.append((volume, current))

    steps = max(1, MUTE_FADE_STEPS)
    delay = fade_seconds / steps if fade_seconds > 0 else 0

    for step in range(1, steps + 1):
        factor = max(0.0, 1.0 - step / steps)
        for volume, start in starts:
            volume.SetMasterVolume(start * factor, None)

        log(f"browser volume fading down: {int(factor * 100)}%")
        if delay:
            time.sleep(delay)

    for volume, _ in starts:
        volume.SetMasterVolume(0.0, None)
        volume.SetMute(True, None)

    log("browser volume reached 0 and is muted")


def run_quiet(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def macos_output_state():
    result = run_quiet(["osascript", "-e", "output volume of (get volume settings)", "-e", "output muted of (get volume settings)"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "osascript failed")

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return int(lines[0]), lines[1].lower() == "true"


def mute_macos_output(mute: bool, fade_seconds=MUTE_FADE_SECONDS):
    global macos_volume_state

    if shutil.which("osascript") is None:
        log("osascript unavailable for macOS audio control")
        return

    if not mute:
        if macos_volume_state is None:
            run_quiet(["osascript", "-e", "set volume without output muted"])
            log("macOS output unmuted")
            return

        volume, was_muted = macos_volume_state
        run_quiet(["osascript", "-e", f"set volume output volume {volume}", "-e", f"set volume output muted {str(was_muted).lower()}"])
        log("macOS output volume restored")
        return

    volume, was_muted = macos_output_state()
    if macos_volume_state is None:
        macos_volume_state = (volume, was_muted)

    steps = max(1, MUTE_FADE_STEPS)
    delay = fade_seconds / steps if fade_seconds > 0 else 0
    for step in range(1, steps + 1):
        factor = max(0.0, 1.0 - step / steps)
        run_quiet(["osascript", "-e", f"set volume output volume {round(volume * factor)}"])
        log(f"macOS output fading down: {int(factor * 100)}%")
        if delay:
            time.sleep(delay)

    run_quiet(["osascript", "-e", "set volume output muted true"])
    log("macOS output muted")


def linux_browser_sink_inputs():
    if shutil.which("pactl") is None:
        log("pactl unavailable for Linux audio control")
        return []

    result = run_quiet(["pactl", "list", "sink-inputs"])
    if result.returncode != 0:
        log(f"pactl failed: {result.stderr.strip()}")
        return []

    inputs = []
    current_index = None
    current_lines = []

    for line in result.stdout.splitlines():
        match = re.match(r"Sink Input #(\d+)", line)
        if match:
            if current_index is not None and browser_name_match("\n".join(current_lines)):
                inputs.append(current_index)
            current_index = match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_index is not None and browser_name_match("\n".join(current_lines)):
        inputs.append(current_index)

    return inputs


def linux_sink_block(index):
    result = run_quiet(["pactl", "list", "sink-inputs"])
    if result.returncode != 0:
        return ""

    block = ""
    capture = False
    for line in result.stdout.splitlines():
        if line.startswith(f"Sink Input #{index}"):
            capture = True
            block = line + "\n"
            continue
        if capture and line.startswith("Sink Input #"):
            break
        if capture:
            block += line + "\n"

    return block


def linux_sink_state(index):
    block = linux_sink_block(index)
    match = re.search(r"Mute:\s+(yes|no)", block)
    volume_match = re.search(r"Volume:.*?(\d+)%", block)
    return {
        "muted": bool(match and match.group(1) == "yes"),
        "volume": f"{volume_match.group(1)}%" if volume_match else "100%",
    }


def mute_linux_browsers(mute: bool, fade_seconds=MUTE_FADE_SECONDS):
    inputs = linux_browser_sink_inputs()

    if not inputs:
        log("no Linux browser sink input found")
        return

    if not mute:
        for index in inputs:
            previous = linux_sink_states.get(index, {"muted": False, "volume": "100%"})
            run_quiet(["pactl", "set-sink-input-volume", index, previous["volume"]])
            run_quiet(["pactl", "set-sink-input-mute", index, "1" if previous["muted"] else "0"])
        log("Linux browser sink inputs restored")
        return

    for index in inputs:
        linux_sink_states.setdefault(index, linux_sink_state(index))

    steps = max(1, MUTE_FADE_STEPS)
    delay = fade_seconds / steps if fade_seconds > 0 else 0
    for step in range(1, steps + 1):
        factor = max(0.0, 1.0 - step / steps)
        for index in inputs:
            run_quiet(["pactl", "set-sink-input-volume", index, f"{int(factor * 100)}%"])
        log(f"Linux browser sink inputs fading down: {int(factor * 100)}%")
        if delay:
            time.sleep(delay)

    for index in inputs:
        run_quiet(["pactl", "set-sink-input-mute", index, "1"])
    log("Linux browser sink inputs muted")


def mute_chrome(mute: bool, fade_seconds=MUTE_FADE_SECONDS):
    system = platform.system()

    if system == "Windows":
        mute_windows_browsers(mute, fade_seconds)
    elif system == "Darwin":
        mute_macos_output(mute, fade_seconds)
    elif system == "Linux":
        mute_linux_browsers(mute, fade_seconds)
    elif mute:
        log(f"MUTE fade requested over {fade_seconds}s, but {system} audio control is unsupported")
    else:
        log(f"UNMUTE requested, but {system} audio control is unsupported")


# -----------------------------
# Green pitch detection
# -----------------------------

def pitch_score(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 40, 40])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    return np.count_nonzero(mask) / mask.size


# -----------------------------
# Scoreboard detection
# -----------------------------

def scoreboard_present(frame):
    h, _ = frame.shape[:2]

    top = frame[:int(h * 0.18), :]

    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    global_density = np.count_nonzero(edges) / edges.size
    max_tile_density = 0.0
    max_tile_contrast = 0.0

    rows = 3
    cols = 12
    tile_h = max(1, edges.shape[0] // rows)
    tile_w = max(1, edges.shape[1] // cols)

    for row in range(rows):
        for col in range(cols):
            y1 = row * tile_h
            x1 = col * tile_w
            y2 = edges.shape[0] if row == rows - 1 else y1 + tile_h
            x2 = edges.shape[1] if col == cols - 1 else x1 + tile_w

            edge_tile = edges[y1:y2, x1:x2]
            gray_tile = gray[y1:y2, x1:x2]
            tile_density = np.count_nonzero(edge_tile) / edge_tile.size
            tile_contrast = float(gray_tile.std())

            max_tile_density = max(max_tile_density, tile_density)
            max_tile_contrast = max(max_tile_contrast, tile_contrast)

    present = global_density > 0.035 or (max_tile_density > 0.055 and max_tile_contrast > 25)

    return present, global_density, max_tile_density


# -----------------------------
# Ball-ish white object detection
# -----------------------------

def ball_present(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 70, 255]))
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
    white[:int(white.shape[0] * 0.12), :] = 0

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8 or area > 220:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / bh if bh else 0

        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(green.shape[1], x + bw + pad)
        y2 = min(green.shape[0], y + bh + pad)
        nearby_green = np.count_nonzero(green[y1:y2, x1:x2]) / max(1, (y2 - y1) * (x2 - x1))

        if 0.45 <= circularity <= 1.35 and 0.5 <= aspect <= 2.0 and nearby_green > 0.15:
            return True

    return False


# -----------------------------
# Higher-quality visual features
# -----------------------------

def green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))


def pitch_line_score(frame):
    h, w = frame.shape[:2]
    small_w = 640
    small_h = max(1, int(h * (small_w / w)))
    small = cv2.resize(frame, (small_w, small_h))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))
    white[:int(small_h * 0.12), :] = 0

    nearby_green = cv2.dilate(green, np.ones((13, 13), np.uint8), iterations=1)
    candidates = cv2.bitwise_and(white, nearby_green)
    edges = cv2.Canny(candidates, 50, 150)
    min_line_length = max(25, small_w // 20)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=min_line_length, maxLineGap=8)

    if lines is None:
        return 0.0

    total_length = 0.0
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)[:40]:
        total_length += float(np.hypot(x2 - x1, y2 - y1))

    return min(1.0, total_length / (small_w * 1.5))


previous_hist = None


def scene_change_score(frame):
    global previous_hist

    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    if previous_hist is None:
        previous_hist = hist
        return 0.0

    correlation = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
    previous_hist = hist
    return round(float(max(0.0, min(1.0, 1.0 - correlation))), 6)


def ad_break_reset_signal(green, board, ball, line_score, scene_score):
    return (
        scene_score >= AD_BREAK_SCENE_CHANGE_THRESHOLD
        and green <= AD_BREAK_GREEN_THRESHOLD
        and line_score <= AD_BREAK_LINE_THRESHOLD
        and not board
        and not ball
    )


def visual_football_support(green, board, ball):
    return (
        green > LOW_GREEN_CONTEXT_THRESHOLD
        or ((board or ball) and green > SCOREBOARD_MIN_GREEN_THRESHOLD)
    )


def regional_features(frame, previous_gray_frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    green = green_mask(frame)
    edges = cv2.Canny(gray, 80, 160)
    diff = cv2.absdiff(previous_gray_frame, gray) if previous_gray_frame is not None else None
    features = {}

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y1 = row * h // GRID_ROWS
            y2 = (row + 1) * h // GRID_ROWS
            x1 = col * w // GRID_COLS
            x2 = (col + 1) * w // GRID_COLS
            cell_area = max(1, (y2 - y1) * (x2 - x1))
            suffix = f"r{row}c{col}"

            features[f"grid_green_{suffix}"] = round(float(np.count_nonzero(green[y1:y2, x1:x2]) / cell_area), 6)
            features[f"grid_edge_{suffix}"] = round(float(np.count_nonzero(edges[y1:y2, x1:x2]) / cell_area), 6)
            if diff is None:
                features[f"grid_motion_{suffix}"] = 999.0
            else:
                features[f"grid_motion_{suffix}"] = round(float(np.mean(diff[y1:y2, x1:x2])), 6)

    return features


# -----------------------------
# Crowd detection
# -----------------------------

previous = None


def crowd_motion_score(frame):
    global previous

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous is None:
        previous = gray
        return 999.0

    diff = cv2.absdiff(previous, gray)

    previous = gray

    return float(np.mean(diff))


def crowd_motion(frame):
    return crowd_motion_score(frame) > 4


def load_classifier(path):
    if not path:
        return None

    try:
        import joblib
    except Exception as exc:
        log(f"could not import joblib for model loading: {exc}")
        return None

    model_path = Path(path)
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        log(f"could not load model from {model_path}: {exc}")
        return None

    if not isinstance(bundle, dict) or "model" not in bundle:
        log(f"model file {model_path} is not a football classifier bundle")
        return None

    log(f"loaded local classifier from {model_path}")
    return bundle


def model_probability(bundle, features):
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    kind = bundle.get("kind", "binary")
    labels = list(bundle.get("labels", []))
    values = [[features[name] for name in feature_names]]

    if kind == "one_class_football":
        score = float(model.score_samples(values)[0])
        threshold = float(bundle["threshold"])
        scale = max(0.000001, float(bundle.get("score_scale", 1.0)))
        return max(0.0, min(1.0, 0.5 + ((score - threshold) / scale)))

    if hasattr(model, "predict_proba") and "football" in labels:
        probabilities = model.predict_proba(values)[0]
        return float(probabilities[labels.index("football")])

    prediction = model.predict(values)[0]
    return 1.0 if prediction == "football" else 0.0


def prepare_recorder(record_dir, args):
    if not record_dir:
        return None

    root = Path(record_dir)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "labels.csv"
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)

    if needs_header:
        writer.writeheader()

    metadata_path = root / "recording_config.json"
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fps": FPS,
        "feature_names": FEATURE_NAMES,
        "args": vars(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"recording training data to {root}")
    return {
        "root": root,
        "frames_dir": frames_dir,
        "handle": handle,
        "writer": writer,
    }


def automatic_label(features, heuristic_football):
    if heuristic_football and (
        features["green"] >= FOOTBALL_AUTO_LABEL_MIN_GREEN
        or features["replay_rule"]
        or (features["scoreboard"] and features["motion"])
    ):
        return "football"

    if (
        not heuristic_football
        and features["green"] <= AD_AUTO_LABEL_MAX_GREEN
        and not features["scoreboard"]
        and not features["replay_rule"]
    ):
        return "ad"

    return ""


def record_sample(recorder, frame, features, heuristic_probability, model_prob, predicted, muted, label=""):
    if not recorder:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = f"{timestamp}.jpg"
    frame_path = recorder["frames_dir"] / filename
    cv2.imwrite(str(frame_path), frame)

    row = {
        "timestamp": timestamp,
        "frame": f"frames/{filename}",
        "heuristic_probability": round(heuristic_probability, 5),
        "model_probability": "" if model_prob is None else round(model_prob, 5),
        "predicted": predicted,
        "muted": int(muted),
        "label": label,
    }
    row.update({name: features[name] for name in FEATURE_NAMES})
    recorder["writer"].writerow(row)
    recorder["handle"].flush()


def close_recorder(recorder):
    if recorder:
        recorder["handle"].close()


def parse_args():
    parser = argparse.ArgumentParser(description="Mute browser audio when football appears to cut to ads.")
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="MSS monitor number to capture. Defaults to the primary monitor.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for this many seconds, then exit. Omit to run until Esc/Ctrl+C.",
    )
    parser.add_argument(
        "--mute-after",
        type=int,
        default=MUTE_AFTER_SECONDS,
        help="Mute after this many consecutive non-football seconds.",
    )
    parser.add_argument(
        "--context-grace",
        type=float,
        default=BALL_CONTEXT_SECONDS,
        help="Keep treating closeups/crowd/camera switches as match coverage for this many seconds after play.",
    )
    parser.add_argument(
        "--mute-fade",
        type=float,
        default=MUTE_FADE_SECONDS,
        help="Fade browser volume to zero over this many seconds when muting.",
    )
    parser.add_argument(
        "--record-data",
        type=str,
        default=None,
        help="Directory where frames and labels.csv should be saved for local training.",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Fill labels.csv with conservative automatic labels for bootstrap training.",
    )
    parser.add_argument(
        "--assume-label",
        choices=["football", "ad"],
        default=None,
        help="Fill every recorded row with this label. Useful for one-class football training during highlights.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained local classifier .joblib file. Uses heuristics if omitted or unavailable.",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=MODEL_THRESHOLD,
        help="Minimum model football probability required to treat a frame as football.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable the OpenCV debug preview window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_debug = SHOW_DEBUG and not args.no_debug
    classifier = load_classifier(args.model)
    recorder = prepare_recorder(args.record_data, args)

    sct = mss.MSS()
    if args.monitor is None:
        monitor = next((mon for mon in sct.monitors[1:] if mon.get("is_primary")), sct.monitors[1])
    else:
        monitor = sct.monitors[args.monitor]

    history = deque(maxlen=args.mute_after)
    muted = False
    seen_football = False
    last_strong_football_at = None
    last_scoreboard_at = None
    started_at = time.monotonic()

    monitor_label = args.monitor if args.monitor is not None else "primary"
    log(
                "starting detector | "
                f"monitor={monitor_label} | "
                f"bounds=left:{monitor['left']} top:{monitor['top']} "
                f"width:{monitor['width']} height:{monitor['height']} | "
        f"duration={args.duration or 'until stopped'}s | "
        f"mute_after={args.mute_after}s | context_grace={args.context_grace}s | "
        f"mute_fade={args.mute_fade}s | "
        f"model={'on' if classifier else 'off'} | "
        f"recording={'on' if recorder else 'off'} | "
        f"debug_window={show_debug}"
    )

    try:
        while True:
            if args.duration is not None and time.monotonic() - started_at >= args.duration:
                log("duration reached, exiting")
                break

            img = np.array(sct.grab(monitor))
            frame = img[:, :, :3]

            green = pitch_score(frame)
            board, board_density, board_tile = scoreboard_present(frame)
            line_score = pitch_line_score(frame)
            scene_score = scene_change_score(frame)
            grid_features = regional_features(frame, previous)
            motion_score = crowd_motion_score(frame)
            motion = motion_score > 4
            ball = ball_present(frame)
            now = time.monotonic()

            if board:
                last_scoreboard_at = now

            live_broadcast = green > GREEN_THRESHOLD and board and motion
            replay_or_close_play = green > REPLAY_GREEN_THRESHOLD and motion
            strong_football = live_broadcast or replay_or_close_play

            if strong_football:
                last_strong_football_at = now

            ad_break_reset = ad_break_reset_signal(green, board, ball, line_score, scene_score)
            if ad_break_reset:
                last_strong_football_at = None
                last_scoreboard_at = None

            recent_match_context = (
                last_strong_football_at is not None
                and now - last_strong_football_at <= args.context_grace
                and motion
                and (board or green > LOW_GREEN_CONTEXT_THRESHOLD)
            )
            scoreboard_recently_seen = (
                last_scoreboard_at is not None
                and now - last_scoreboard_at <= SCOREBOARD_HOLD_SECONDS
            )
            scoreboard_supported_by_match = (
                green > SCOREBOARD_MIN_GREEN_THRESHOLD
                or (
                    last_strong_football_at is not None
                    and now - last_strong_football_at <= args.context_grace
                )
            )
            scoreboard_match_context = (
                scoreboard_recently_seen
                and motion
                and scoreboard_supported_by_match
            )
            heuristic_football = (strong_football or recent_match_context or scoreboard_match_context) and not ad_break_reset
            seconds_since_strong = (
                UNKNOWN_SECONDS if last_strong_football_at is None else min(UNKNOWN_SECONDS, now - last_strong_football_at)
            )
            seconds_since_scoreboard = (
                UNKNOWN_SECONDS if last_scoreboard_at is None else min(UNKNOWN_SECONDS, now - last_scoreboard_at)
            )
            features = {
                "green": round(float(green), 6),
                "scoreboard": int(board),
                "board_density": round(float(board_density), 6),
                "board_tile": round(float(board_tile), 6),
                "motion_score": round(float(motion_score), 6),
                "motion": int(motion),
                "ball": int(ball),
                "replay_rule": int(replay_or_close_play),
                "recent_context": int(recent_match_context),
                "scoreboard_context": int(scoreboard_match_context),
                "scoreboard_support": int(scoreboard_supported_by_match),
                "ad_break_reset": int(ad_break_reset),
                "seconds_since_strong": round(float(seconds_since_strong), 3),
                "seconds_since_scoreboard": round(float(seconds_since_scoreboard), 3),
                "pitch_line_score": round(float(line_score), 6),
                "scene_change_score": round(float(scene_score), 6),
            }
            features.update(grid_features)

            model_prob = None
            if classifier:
                try:
                    model_prob = model_probability(classifier, features)
                except Exception as exc:
                    log(f"model prediction failed, falling back to heuristics: {exc}")
                    classifier = None

            if model_prob is not None:
                football = model_prob >= args.model_threshold and (heuristic_football or visual_football_support(green, board, ball))
            else:
                football = heuristic_football
            football = football and not ad_break_reset
            heuristic_probability = 1.0 if heuristic_football else 0.0

            history.append(football)
            seen_football = seen_football or football

            if seen_football and len(history) == args.mute_after and not any(history) and not muted:
                log(f"no football detected for {args.mute_after}s, muting browser")
                mute_chrome(True, args.mute_fade)
                muted = True

            if muted:
                if football:
                    log("football detected, unmuting browser instantly")
                    mute_chrome(False)
                    muted = False

            status_line = (
                f"{'FOOTBALL' if football else 'ADS'} | "
                f"Pitch: {green:.2f} | Scoreboard: {board} | Motion: {motion} | "
                f"BoardDensity: {board_density:.3f} | BoardTile: {board_tile:.3f} | "
                f"Ball: {ball} | ReplayRule: {replay_or_close_play} | "
                f"Lines: {line_score:.2f} | SceneCut: {scene_score:.2f} | "
                f"Context: {recent_match_context} | ScoreboardContext: {scoreboard_match_context} | "
                f"ScoreboardSupport: {scoreboard_supported_by_match} | "
                f"VisualSupport: {visual_football_support(green, board, ball)} | "
                f"AdBreakReset: {ad_break_reset} | "
            )
            if model_prob is not None:
                status_line += f"ModelProb: {model_prob:.2f} | "
            status_line += f"Muted: {muted}"
            log(status_line)

            record_sample(
                recorder,
                frame,
                features,
                heuristic_probability,
                model_prob,
                "football" if football else "ad",
                muted,
                args.assume_label or (automatic_label(features, heuristic_football) if args.auto_label else ""),
            )

            if show_debug:
                status = "FOOTBALL" if football else "ADS"

                cv2.putText(
                    frame,
                    status,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if football else (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Pitch: {green:.2f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Scoreboard: {board}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Motion: {motion}",
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Ball: {ball}",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Detector", frame)

                if cv2.waitKey(1) == 27:
                    break

            time.sleep(1 / FPS)

    except KeyboardInterrupt:
        log("interrupted, exiting")

    finally:
        if muted:
            log("restoring browser audio before exit")
            mute_chrome(False)
        close_recorder(recorder)
        cv2.destroyAllWindows()
        log("detector stopped")


if __name__ == "__main__":
    main()
