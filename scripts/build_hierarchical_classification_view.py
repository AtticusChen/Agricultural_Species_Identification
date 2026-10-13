#!/usr/bin/env python3
"""Build a three-class crop view that separates Bidens hard negatives."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-root", type=Path, required=True)
    parser.add_argument("--detection-root", type=Path, required=True)
    parser.add_argument("--manifest-name", choices=("scene", "patch"), default="patch")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_class(sample_role: str, original_source: str) -> str:
    if sample_role == "positive":
        return "mikania"
    if "/三叶鬼针草/" in f"/{original_source}":
        return "bidens"
    return "other"


def main() -> None:
    args = parse_args()
    classification_root = args.classification_root.resolve()
    detection_root = args.detection_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    with (detection_root / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        detection_rows = list(csv.DictReader(handle))
    detection_by_image = {row["image"]: row for row in detection_rows}

    manifest_path = (
        classification_root / "manifests" / f"{args.manifest_name}.csv"
    )
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        classification_rows = list(csv.DictReader(handle))

    counts: Counter[tuple[str, str]] = Counter()
    output_rows: list[dict[str, str]] = []
    for row in classification_rows:
        detection_row = detection_by_image[row["source_image"]]
        class_name = target_class(
            detection_row["sample_role"], detection_row["source_image"]
        )
        source = classification_root / row["image"]
        destination = output / row["split"] / class_name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.path.relpath(source.resolve(), destination.parent))
        counts[(row["split"], class_name)] += 1
        output_rows.append(
            {
                **row,
                "hierarchical_class": class_name,
                "original_source": detection_row["source_image"],
            }
        )

    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True)
    with (manifest_dir / f"{args.manifest_name}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "classification_root": str(classification_root),
        "detection_root": str(detection_root),
        "manifest_name": args.manifest_name,
        "counts": [
            {"split": split, "class": class_name, "count": count}
            for (split, class_name), count in sorted(counts.items())
        ],
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
