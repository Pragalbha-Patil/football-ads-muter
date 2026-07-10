import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


GRID_ROWS = 3
GRID_COLS = 3

GRID_FEATURE_NAMES = [
    f"grid_{metric}_r{row}c{col}"
    for metric in ("green", "motion", "edge")
    for row in range(GRID_ROWS)
    for col in range(GRID_COLS)
]

BASE_FEATURE_NAMES = [
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
    "seconds_since_strong",
    "seconds_since_scoreboard",
]

EXTRA_FEATURE_NAMES = [
    "ad_break_reset",
    "pitch_line_score",
    "scene_change_score",
    *GRID_FEATURE_NAMES,
]

FEATURE_NAMES = [*BASE_FEATURE_NAMES, *EXTRA_FEATURE_NAMES]

FEATURE_DEFAULTS = {
    "ad_break_reset": 0.0,
    "pitch_line_score": 0.0,
    "scene_change_score": 0.0,
    **{name: 0.0 for name in GRID_FEATURE_NAMES},
}

FOOTBALL_LABELS = {"football", "match", "play", "replay", "closeup", "crowd", "studio_football"}


def available_feature_names(csv_paths):
    fieldnames = set()

    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames.update(reader.fieldnames or [])

    missing_base = [name for name in BASE_FEATURE_NAMES if name not in fieldnames]
    if missing_base:
        raise ValueError(f"training CSVs are missing base feature columns: {', '.join(missing_base)}")

    return [*BASE_FEATURE_NAMES, *[name for name in EXTRA_FEATURE_NAMES if name in fieldnames]]


def load_football_rows(csv_paths, use_all=False):
    rows = []
    feature_names = available_feature_names(csv_paths)

    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in ["label"] if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

            for row_number, row in enumerate(reader, start=2):
                label = row.get("label", "").strip().lower().replace(" ", "_").replace("-", "_")
                if not use_all and label not in FOOTBALL_LABELS:
                    continue

                try:
                    rows.append(
                        [
                            float(row.get(name, FEATURE_DEFAULTS.get(name, 0.0)) or FEATURE_DEFAULTS.get(name, 0.0))
                            for name in feature_names
                        ]
                    )
                except ValueError as exc:
                    raise ValueError(f"{csv_path}:{row_number} has invalid feature values") from exc

    return rows, feature_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train a one-class football/highlights anomaly model.")
    parser.add_argument("csv", nargs="+", help="One or more labels.csv files recorded from football/highlights.")
    parser.add_argument("--output", default="models/football_one_class.joblib", help="Where to save the model bundle.")
    parser.add_argument(
        "--use-all",
        action="store_true",
        help="Use every row as football, ignoring labels. Only use this on recordings that contain football/highlights only.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.03,
        help="Expected fraction of unusual football frames in the training data.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=2.0,
        help="Training score percentile used as the football/anomaly boundary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, feature_names = load_football_rows(args.csv, args.use_all)

    if len(rows) < 30:
        raise SystemExit("Need at least 30 football/highlights rows for one-class training. 5-10 minutes is better.")

    x = np.array(rows, dtype=float)
    model = IsolationForest(
        n_estimators=300,
        contamination=args.contamination,
        random_state=42,
    )
    model.fit(x)

    scores = model.score_samples(x)
    threshold = float(np.percentile(scores, args.threshold_percentile))
    score_scale = float(max(np.percentile(scores, 95) - threshold, 0.000001))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "kind": "one_class_football",
        "model": model,
        "feature_names": feature_names,
        "threshold": threshold,
        "score_scale": score_scale,
        "training_rows": len(rows),
        "contamination": args.contamination,
    }
    joblib.dump(bundle, output_path)

    print(f"trained one-class football model on {len(rows)} rows")
    print(f"features used: {len(feature_names)}")
    print(f"threshold={threshold:.6f} score_scale={score_scale:.6f}")
    print(f"saved model to {output_path}")


if __name__ == "__main__":
    main()
