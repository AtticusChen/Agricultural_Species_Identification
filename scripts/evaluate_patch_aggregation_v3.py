#!/usr/bin/env python3
"""Evaluate robust aggregation rules for sliding-patch classification.

All thresholds and ensemble choices are selected on the validation split. The
internal and external test splits are only scored after those choices are fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluate_classifiers_v3 import (
    balanced_summary,
    mikania_probability,
    select_threshold,
    tile_starts,
)
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--patch-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="2")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--scene-scores", type=Path)
    return parser.parse_args()


def load_scene_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["image"]: float(row["scene"])
            for row in csv.DictReader(handle)
        }


def quantile(sorted_scores: list[float], fraction: float) -> float:
    if not sorted_scores:
        return 0.0
    position = (len(sorted_scores) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_scores[lower]
    weight = position - lower
    return sorted_scores[lower] * (1 - weight) + sorted_scores[upper] * weight


def aggregate(scores: list[float], scene_score: float | None) -> dict[str, float]:
    descending = sorted(scores, reverse=True)
    ascending = list(reversed(descending))

    def top_mean(count: int) -> float:
        selected = descending[: min(count, len(descending))]
        return fmean(selected) if selected else 0.0

    output = {
        "patch_max": descending[0],
        "patch_second": descending[min(1, len(descending) - 1)],
        "patch_third": descending[min(2, len(descending) - 1)],
        "patch_top2_mean": top_mean(2),
        "patch_top3_mean": top_mean(3),
        "patch_top5_mean": top_mean(5),
        "patch_mean": fmean(descending),
        "patch_q75": quantile(ascending, 0.75),
        "patch_q90": quantile(ascending, 0.90),
        "patch_fraction_50": sum(score >= 0.50 for score in scores) / len(scores),
        "patch_fraction_70": sum(score >= 0.70 for score in scores) / len(scores),
        "patch_fraction_90": sum(score >= 0.90 for score in scores) / len(scores),
    }
    if scene_score is not None:
        output["scene"] = scene_score
        for weight in (0.25, 0.50, 0.75):
            output[f"scene_patch_top2_w{weight:.2f}"] = (
                weight * scene_score + (1 - weight) * output["patch_top2_mean"]
            )
            output[f"scene_patch_second_w{weight:.2f}"] = (
                weight * scene_score + (1 - weight) * output["patch_second"]
            )
    return output


def main() -> None:
    args = parse_args()
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene_scores = load_scene_scores(
        args.scene_scores.resolve() if args.scene_scores else None
    )

    with (dataset / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["split"] != "train"
        ]

    from ultralytics import YOLO

    model = YOLO(str(args.patch_model.resolve()))
    records: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        with Image.open(dataset / row["image"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        tiles = [
            (
                left,
                top,
                image.crop(
                    (
                        left,
                        top,
                        min(image.width, left + args.tile_size),
                        min(image.height, top + args.tile_size),
                    )
                ),
            )
            for top in tile_starts(image.height, args.tile_size, args.overlap)
            for left in tile_starts(image.width, args.tile_size, args.overlap)
        ]
        predictions = model.predict(
            source=[tile[2] for tile in tiles],
            imgsz=args.image_size,
            device=args.device,
            batch=64,
            verbose=False,
        )
        scores = [mikania_probability(prediction) for prediction in predictions]
        features = aggregate(scores, scene_scores.get(row["image"]))
        records.append(
            {
                "image": row["image"],
                "split": row["split"],
                "sample_role": row["sample_role"],
                "split_unit_id": row["split_unit_id"],
                "tile_count": len(tiles),
                **features,
            }
        )
        for (left, top, tile), score in zip(tiles, scores, strict=True):
            raw_rows.append(
                {
                    "image": row["image"],
                    "split": row["split"],
                    "sample_role": row["sample_role"],
                    "left": left,
                    "top": top,
                    "right": left + tile.width,
                    "bottom": top + tile.height,
                    "score": score,
                }
            )
        if index % 25 == 0:
            print(f"scored {index}/{len(rows)}", flush=True)

    excluded = {"image", "split", "sample_role", "split_unit_id", "tile_count"}
    methods = [key for key in records[0] if key not in excluded]
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "patch_model": str(args.patch_model.resolve()),
        "threshold_source": "val",
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "methods": {},
    }
    for method in methods:
        validation_positive = [
            float(record[method])
            for record in records
            if record["split"] == "val" and record["sample_role"] == "positive"
        ]
        validation_negative = [
            float(record[method])
            for record in records
            if record["split"] == "val" and record["sample_role"] != "positive"
        ]
        selection = select_threshold(validation_positive, validation_negative)
        threshold = float(selection["selected"]["threshold"])

        def split_scores(
            split: str, positive: bool, active_method: str = method
        ) -> list[float]:
            return [
                float(record[active_method])
                for record in records
                if record["split"] == split
                and (record["sample_role"] == "positive") == positive
            ]

        report["methods"][method] = {
            "validation": selection,
            "test_internal": balanced_summary(
                split_scores("test_internal", True),
                split_scores("test_internal", False),
                threshold,
            ),
            "test_external": balanced_summary(
                split_scores("test_external", True),
                split_scores("test_external", False),
                threshold,
            ),
        }

    selected_method = max(
        methods,
        key=lambda method: (
            float(
                report["methods"][method]["validation"]["selected"][
                    "balanced_accuracy"
                ]
            ),
            -float(
                report["methods"][method]["validation"]["selected"][
                    "false_positive_rate"
                ]
            ),
            float(
                report["methods"][method]["validation"]["selected"]["sensitivity"]
            ),
        ),
    )
    report["selected_method"] = selected_method
    report["selected_result"] = report["methods"][selected_method]

    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "tile_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"selected_method": selected_method, **report["selected_result"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
