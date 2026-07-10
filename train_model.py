import argparse
import csv
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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
    "seconds_since_strong",
    "seconds_since_scoreboard",
]

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


def load_rows(csv_paths):
    rows = []

    for csv_path in csv_paths:
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in FEATURE_NAMES + ["label"] if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

            for row_number, row in enumerate(reader, start=2):
                label = normalize_label(row.get("label", ""))
                if label is None:
                    continue

                try:
                    features = [float(row[name]) for name in FEATURE_NAMES]
                except ValueError as exc:
                    raise ValueError(f"{csv_path}:{row_number} has invalid feature values") from exc

                rows.append((features, label))

    return rows


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
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.csv)

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
        "feature_names": FEATURE_NAMES,
        "labels": list(model.classes_),
        "label_counts": label_counts,
    }
    joblib.dump(bundle, output_path)
    print(f"\nsaved model to {output_path}")


if __name__ == "__main__":
    main()
