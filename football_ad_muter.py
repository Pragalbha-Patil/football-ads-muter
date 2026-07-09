import argparse
import time
from collections import deque

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

MUTE_AFTER_SECONDS = 2
UNMUTE_AFTER_SECONDS = 2
MUTE_FADE_SECONDS = 2.0
MUTE_FADE_STEPS = 8

SHOW_DEBUG = True


def log(message):
    print(f"{time.strftime('%H:%M:%S')} | {message}", flush=True)


# -----------------------------
# Windows mute
# -----------------------------

try:
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

    browser_volumes = {}

    def browser_audio_sessions():
        sessions = AudioUtilities.GetAllSessions()

        for session in sessions:
            if session.Process:
                name = session.Process.name().lower()

                if "chrome" in name or "edge" in name or "firefox" in name:
                    volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                    yield session.Process.pid, name, volume

    def mute_chrome(mute: bool, fade_seconds=MUTE_FADE_SECONDS):
        sessions = list(browser_audio_sessions())

        if not sessions:
            log("no browser audio session found")
            return

        if not mute:
            for pid, _, volume in sessions:
                volume.SetMute(False, None)
                volume.SetMasterVolume(browser_volumes.get(pid, 1.0), None)

            log("browser volume restored instantly")
            return

        starts = []
        for pid, _, volume in sessions:
            current = volume.GetMasterVolume()
            browser_volumes.setdefault(pid, current)
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

except Exception:

    def mute_chrome(mute: bool, fade_seconds=MUTE_FADE_SECONDS):
        if mute:
            log(f"MUTE fade requested over {fade_seconds}s")
        else:
            log("UNMUTE instantly requested")


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
# Crowd detection
# -----------------------------

previous = None


def crowd_motion(frame):
    global previous

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous is None:
        previous = gray
        return True

    diff = cv2.absdiff(previous, gray)

    previous = gray

    motion = np.mean(diff)

    return motion > 4


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
        "--no-debug",
        action="store_true",
        help="Disable the OpenCV debug preview window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_debug = SHOW_DEBUG and not args.no_debug

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
            motion = crowd_motion(frame)
            ball = ball_present(frame)
            now = time.monotonic()

            if board:
                last_scoreboard_at = now

            live_broadcast = green > GREEN_THRESHOLD and board and motion
            replay_or_close_play = green > REPLAY_GREEN_THRESHOLD and motion
            strong_football = live_broadcast or replay_or_close_play

            if strong_football:
                last_strong_football_at = now

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
            football = strong_football or recent_match_context or scoreboard_match_context

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

            log(
                f"{'FOOTBALL' if football else 'ADS'} | "
                f"Pitch: {green:.2f} | Scoreboard: {board} | Motion: {motion} | "
                f"BoardDensity: {board_density:.3f} | BoardTile: {board_tile:.3f} | "
                f"Ball: {ball} | ReplayRule: {replay_or_close_play} | "
                f"Context: {recent_match_context} | ScoreboardContext: {scoreboard_match_context} | "
                f"ScoreboardSupport: {scoreboard_supported_by_match} | "
                f"Muted: {muted}"
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
        cv2.destroyAllWindows()
        log("detector stopped")


if __name__ == "__main__":
    main()
