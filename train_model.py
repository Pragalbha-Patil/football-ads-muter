import argparse
import csv
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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
    "pitch_line_score",
    "scene_change_score",
    *GRID_FEATURE_NAMES,
]

FEATURE_NAMES = [*BASE_FEATURE_NAMES, *EXTRA_FEATURE_NAMES]

FEATURE_DEFAULTS = {
    "pitch_line_score": 0.0,
    "scene_change_score": 0.0,
    **{name: 0.0 for name in GRID_FEATURE_NAMES},
}

FOOTBALL_LABELS = {"football", "match", "play", "replay", "closeup", "crowd", "studio_football"}
AD_LABELS = {"ad", "ads", "advert", "advertisement", "commercial", "break", "non_football"}
SKIP_LABELS = {"", "unknown", "skip", "unsure", "bad_frame"}


def normalize_label(label):
    clean = label.strip().lower().replace(" ", "_").replace("-", "_")

    if clean in FOOTBALL_LABELS:
        return "football"
    if clean in AD_LABELS:
        return "ad"
    if clean in SKIP_LABELS:
        return None

    raise ValueError(f"unsupported label: {label!r}")


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


def load_rows(csv_paths, use_predicted=False):
    rows = []
    feature_names = available_feature_names(csv_paths)

    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in ["label"] if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

            for row_number, row in enumerate(reader, start=2):
                raw_label = row.get("label", "")
                if use_predicted and not raw_label.strip():
                    raw_label = row.get("predicted", "")

                label = normalize_label(raw_label)
                if label is None:
                    continue

                try:
                    features = [
                        float(row.get(name, FEATURE_DEFAULTS.get(name, 0.0)) or FEATURE_DEFAULTS.get(name, 0.0))
                        for name in feature_names
                    ]
                except ValueError as exc:
                    raise ValueError(f"{csv_path}:{row_number} has invalid feature values") from exc

                rows.append((features, label))

    return rows, feature_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train a local football/ad classifier from labeled CSV data.")
    parser.add_argument(
        "csv",
        nargs="+",
        help="One or more labels.csv files created by football_ad_muter.py --record-data.",
    )
    parser.add_argument(
        "--output",
        default="models/football_ad_classifier.joblib",
        help="Where to save the trained model bundle.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of labeled rows to reserve for validation when enough data exists.",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=300,
        help="Number of random forest trees.",
    )
    parser.add_argument(
        "--use-predicted",
        action="store_true",
        help="Use the predicted column when label is blank. Useful only for heuristic bootstrap training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows, feature_names = load_rows(args.csv, args.use_predicted)

    if len(rows) < 10:
        raise SystemExit("Need at least 10 labeled rows before training. More is better; a few hundred is a good start.")

    x = [features for features, _ in rows]
    y = [label for _, label in rows]
    label_counts = {label: y.count(label) for label in sorted(set(y))}

    if set(label_counts) != {"ad", "football"}:
        raise SystemExit(f"Need both football and ad labels. Current counts: {label_counts}")

    can_validate = min(label_counts.values()) >= 2 and len(rows) >= 20
    if can_validate:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=42,
            stratify=y,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test = [], []

    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)

    print(f"trained on {len(x_train)} rows")
    print(f"label counts: {label_counts}")
    print(f"features used: {len(feature_names)}")

    if x_test:
        predictions = model.predict(x_test)
        print("\nconfusion matrix [ad, football]:")
        print(confusion_matrix(y_test, predictions, labels=["ad", "football"]))
        print("\nclassification report:")
        print(classification_report(y_test, predictions, labels=["ad", "football"]))
    else:
        print("validation skipped because there are not enough labeled examples yet")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "labels": list(model.classes_),
        "label_counts": label_counts,
    }
    joblib.dump(bundle, output_path)
    print(f"\nsaved model to {output_path}")


if __name__ == "__main__":
    main()
