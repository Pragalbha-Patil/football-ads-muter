import argparse
import csv
from pathlib import Path

import cv2


KEY_LABELS = {
    ord("f"): "football",
    ord("r"): "replay",
    ord("c"): "closeup",
    ord("d"): "crowd",
    ord("a"): "ad",
    ord("u"): "unknown",
    ord("s"): "skip",
}


def load_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def save_rows(csv_path, fieldnames, rows):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_overlay(image, row, index, total):
    label = row.get("label") or "blank"
    predicted = row.get("predicted") or "unknown"
    model_probability = row.get("model_probability") or "n/a"
    text_lines = [
        f"{index + 1}/{total} | label={label} | predicted={predicted} | model={model_probability}",
        "f football | r replay | c closeup | d crowd | a ad | u unknown | s skip | n next | b back | q quit",
    ]

    y = 34
    for text in text_lines:
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        y += 30


def parse_args():
    parser = argparse.ArgumentParser(description="Quickly label recorded football/ad frames.")
    parser.add_argument("csv", help="labels.csv created by football_ad_muter.py --record-data.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visit all rows instead of only blank labels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    root = csv_path.parent
    fieldnames, rows = load_rows(csv_path)

    if "label" not in fieldnames or "frame" not in fieldnames:
        raise SystemExit("CSV must contain frame and label columns")

    indexes = [i for i, row in enumerate(rows) if args.all or not row.get("label", "").strip()]
    if not indexes:
        print("No unlabeled rows found. Use --all to review existing labels.")
        return

    cursor = 0
    while 0 <= cursor < len(indexes):
        row_index = indexes[cursor]
        row = rows[row_index]
        frame_path = root / row["frame"]
        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read {frame_path}; marking as bad_frame")
            row["label"] = "bad_frame"
            cursor += 1
            continue

        preview = image.copy()
        draw_overlay(preview, row, cursor, len(indexes))
        cv2.imshow("Label football frames", preview)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q") or key == 27:
            break
        if key == ord("n") or key == 13 or key == 32:
            cursor += 1
            continue
        if key == ord("b"):
            cursor = max(0, cursor - 1)
            continue
        if key in KEY_LABELS:
            row["label"] = KEY_LABELS[key]
            save_rows(csv_path, fieldnames, rows)
            cursor += 1
            continue

    save_rows(csv_path, fieldnames, rows)
    cv2.destroyAllWindows()
    print(f"Saved labels to {csv_path}")


if __name__ == "__main__":
    main()
