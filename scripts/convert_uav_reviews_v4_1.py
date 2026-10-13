#!/usr/bin/env python3
"""Record the user's 2026-09-14 UAV photograph-label correction without editing v4."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

CORRECTION = (
    "2026-09-14 user correction: all UAV photographs are reported to contain Mikania; "
    "photograph-level presence only, not botanist-confirmed per-image identification. "
    "Existing positive and negative regions remain assistant_visual_provisional. "
)


def convert_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Preserve split, site, ROI strings and original review text exactly."""
    converted = []
    for source in rows:
        row = dict(source)
        row.update(
            label="positive",
            provenance="user_reported_positive",
            original_review_label=source["label"],
            region_provenance="assistant_visual_provisional",
            review=CORRECTION + "Original visual review: " + source.get("review", ""),
        )
        converted.append(row)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise FileExistsError("An existing review file must never be overwritten")
    before = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 375 or len({row["filename"] for row in rows}) != 375:
        raise ValueError("Expected exactly 375 unique reviewed UAV photographs")
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 278, "val": 54, "test_uav": 43}:
        raise ValueError(f"Unexpected frozen split counts: {counts}")
    converted = convert_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != before:
        raise RuntimeError("Source review file changed")
    print(f"Wrote {len(converted)} user-reported positive photograph labels; split counts {counts}")


if __name__ == "__main__":
    main()
