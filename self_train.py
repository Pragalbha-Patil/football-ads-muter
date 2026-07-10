import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


FOOTBALL_LABELS = {"football", "match", "play", "replay", "closeup", "crowd", "studio_football"}
AD_LABELS = {"ad", "ads", "advert", "advertisement", "commercial", "break", "non_football"}


def label_counts(csv_paths):
    counts = {"football": 0, "ad": 0}

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row.get("label", "").strip().lower().replace(" ", "_").replace("-", "_")
                if label in FOOTBALL_LABELS:
                    counts["football"] += 1
                elif label in AD_LABELS:
                    counts["ad"] += 1

    return counts


def run_command(command):
    print(" ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, check=False).returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Continuously collect auto-labeled samples and retrain locally.")
    parser.add_argument("--monitor", type=int, required=True, help="MSS monitor number where the browser stream is running.")
    parser.add_argument("--data-dir", default="data/self-train", help="Directory for recording sessions.")
    parser.add_argument("--model", default="models/football_ad_classifier.joblib", help="Model output path.")
    parser.add_argument("--chunk-duration", type=int, default=600, help="Seconds to record before each training attempt.")
    parser.add_argument("--cycles", type=int, default=0, help="Number of collect/train cycles. 0 means keep running.")
    parser.add_argument("--mute-after", type=int, default=2)
    parser.add_argument("--context-grace", type=float, default=4)
    parser.add_argument("--mute-fade", type=float, default=2.0)
    parser.add_argument("--model-threshold", type=float, default=0.65)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_path = Path(args.model)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        session_name = datetime.now().strftime("session-%Y%m%d-%H%M%S")
        session_dir = data_dir / session_name
        command = [
            sys.executable,
            "football_ad_muter.py",
            "--monitor",
            str(args.monitor),
            "--duration",
            str(args.chunk_duration),
            "--record-data",
            str(session_dir),
            "--auto-label",
            "--mute-after",
            str(args.mute_after),
            "--context-grace",
            str(args.context_grace),
            "--mute-fade",
            str(args.mute_fade),
            "--no-debug",
        ]

        if model_path.exists():
            command.extend(["--model", str(model_path), "--model-threshold", str(args.model_threshold)])

        print(f"\ncollecting cycle {cycle} into {session_dir}", flush=True)
        code = run_command(command)
        if code != 0:
            print(f"collection exited with code {code}; retrying after a short pause", flush=True)
            time.sleep(5)
            continue

        csv_paths = sorted(data_dir.glob("session-*/labels.csv"))
        counts = label_counts(csv_paths)
        print(f"auto-label counts so far: {counts}", flush=True)

        if counts["football"] < 5 or counts["ad"] < 5:
            print("not enough of both classes to train yet; continuing collection", flush=True)
            continue

        train_command = [
            sys.executable,
            "train_model.py",
            *[str(path) for path in csv_paths],
            "--output",
            str(model_path),
        ]
        print(f"training model after cycle {cycle}", flush=True)
        run_command(train_command)


if __name__ == "__main__":
    main()
