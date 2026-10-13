#!/usr/bin/env python3
"""Join recorded visual reviews with spatial split assignments; no inferred labels."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    reviews = {}
    for name in ("uav_v4_review_part1.json", "uav_v4_review_part2.json"):
        partial = json.loads((ROOT / "configs" / name).read_text())
        if reviews.keys() & partial.keys():
            raise ValueError("Overlapping review ownership")
        reviews.update(partial)
    with (ROOT / "artifacts/data-audit-v4/uav_images.csv").open(newline="") as handle:
        images = list(csv.DictReader(handle))
    if set(reviews) != {row["filename"] for row in images}:
        raise ValueError("Every UAV file must have one explicit review")
    rows = []
    for image in images:
        review = reviews[image["filename"]]
        site = int(image["uav_group"].split("-")[-1])
        split = "val" if site in {2, 3, 10} else ("test_uav" if site in {8, 9, 11, 12} else "train")
        rows.append(
            dict(
                filename=image["filename"],
                label=review["label"],
                split=split,
                group_id=("uav-site-002-003" if site in {2, 3} else image["uav_group"]),
                review=review["reason"],
                regions=";".join(" ".join(map(str, box)) for box in review.get("regions", [])),
                negative_regions=";".join(
                    " ".join(map(str, box)) for box in review.get("negative_regions", [])
                ),
            )
        )
    target = ROOT / "configs/uav_v4_reviews.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(target)


if __name__ == "__main__":
    main()
