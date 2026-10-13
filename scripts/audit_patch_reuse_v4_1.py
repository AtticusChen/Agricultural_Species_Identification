#!/usr/bin/env python3
"""Verify unchanged local-supervision inputs before reusing an existing patch model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(root: Path) -> list[dict[str, str]]:
    with (root / "manifests/regions.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(old: Path, new: Path) -> dict:
    old_rows, new_rows = read_rows(old), read_rows(new)
    fields = ("source_image", "box_xyxy", "label", "split", "domain", "group_id")

    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[field] for field in fields)

    if Counter(map(key, old_rows)) != Counter(map(key, new_rows)):
        raise ValueError("Patch source/box/label/split/domain/group multisets differ")
    available = defaultdict(list)
    for row in new_rows:
        available[key(row)].append(row)
    matches = []
    for old_row in old_rows:
        new_row = available[key(old_row)].pop()
        old_path, new_path = old / old_row["image"], new / new_row["image"]
        old_hash, new_hash = sha256(old_path), sha256(new_path)
        same_bytes = old_hash == new_hash
        if not same_bytes:
            with Image.open(old_path) as old_image, Image.open(new_path) as new_image:
                old_rgb, new_rgb = old_image.convert("RGB"), new_image.convert("RGB")
                if old_rgb.size != new_rgb.size or old_rgb.tobytes() != new_rgb.tobytes():
                    raise ValueError(f"Patch pixels differ: {old_path} versus {new_path}")
        matches.append(
            {
                **{field: old_row[field] for field in fields},
                "old_image": old_row["image"],
                "new_image": new_row["image"],
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "same_bytes": same_bytes,
            }
        )
    return {
        "old_dataset": str(old),
        "new_dataset": str(new),
        "compared_fields": fields,
        "old_region_manifest_sha256": sha256(old / "manifests/regions.csv"),
        "new_region_manifest_sha256": sha256(new / "manifests/regions.csv"),
        "rows": len(matches),
        "source_box_label_split_multiset_equal": True,
        "byte_identical_count": sum(row["same_bytes"] for row in matches),
        "decoded_pixel_identical_count": len(matches),
        "by_split": dict(Counter(row["split"] for row in matches)),
        "reuse_conclusion": (
            "Patch inputs and supervision are unchanged; reusing the prior patch weights is valid. "
            "This does not make new whole-image labels or model quality expert-confirmed."
        ),
        "matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.old.resolve(), args.new.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "matches"}))


if __name__ == "__main__":
    main()
