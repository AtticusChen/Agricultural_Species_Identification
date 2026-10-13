#!/usr/bin/env python3
"""Evaluate scene and sliding-patch Mikania classifiers without test-set tuning."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REGRESSION_STEMS = (
    "IMG_20250228_140606",
    "IMG_20250227_155423",
    "IMG_20250228_140632",
    "IMG_20251113_163941",
    "IMG_20251118_121305",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene-model", type=Path, required=True)
    parser.add_argument("--patch-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-scene", default="0")
    parser.add_argument("--device-patch", default="1")
    parser.add_argument("--scene-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.20)
    return parser.parse_args()


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def mikania_probability(result: Any) -> float:
    class_id = next(
        (class_id for class_id, name in result.names.items() if name == "mikania"),
        None,
    )
    if class_id is None:
        raise ValueError(f"Classifier has no mikania class: {result.names}")
    return float(result.probs.data[class_id].item())


def classify_scene(model: Any, image: Image.Image, image_size: int, device: str) -> float:
    result = model.predict(
        source=image, imgsz=image_size, device=device, verbose=False
    )[0]
    return mikania_probability(result)


def classify_tiles(
    model: Any,
    image: Image.Image,
    image_size: int,
    device: str,
    tile_size: int,
    overlap: float,
) -> tuple[float, float, int]:
    crops = [
        image.crop(
            (
                left,
                top,
                min(image.width, left + tile_size),
                min(image.height, top + tile_size),
            )
        )
        for top in tile_starts(image.height, tile_size, overlap)
        for left in tile_starts(image.width, tile_size, overlap)
    ]
    results = model.predict(
        source=crops, imgsz=image_size, device=device, batch=64, verbose=False
    )
    scores = sorted((mikania_probability(result) for result in results), reverse=True)
    top_count = min(3, len(scores))
    return scores[0], sum(scores[:top_count]) / top_count, len(scores)


def balanced_summary(
    positive_scores: Iterable[float],
    negative_scores: Iterable[float],
    threshold: float,
) -> dict[str, float | int | None]:
    positives = list(positive_scores)
    negatives = list(negative_scores)
    true_positive = sum(score >= threshold for score in positives)
    false_positive = sum(score >= threshold for score in negatives)
    sensitivity = true_positive / len(positives) if positives else 0.0
    false_positive_rate = false_positive / len(negatives) if negatives else None
    specificity = 1 - false_positive_rate if false_positive_rate is not None else None
    return {
        "threshold": threshold,
        "positive_images": len(positives),
        "negative_images": len(negatives),
        "true_positive": true_positive,
        "false_negative": len(positives) - true_positive,
        "false_positive": false_positive,
        "true_negative": len(negatives) - false_positive,
        "sensitivity": sensitivity,
        "false_positive_rate": false_positive_rate,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2 if specificity is not None else None
        ),
    }


def threshold_candidates(scores: list[float]) -> list[float]:
    fixed = [index / 100 for index in range(1, 100)]
    unique = sorted(set(scores))
    midpoints = [
        (left + right) / 2 for left, right in zip(unique, unique[1:], strict=False)
    ]
    return sorted(set([0.0, 1.0, *fixed, *midpoints]))


def select_threshold(positive: list[float], negative: list[float]) -> dict[str, Any]:
    candidates = threshold_candidates(positive + negative)
    summaries = [balanced_summary(positive, negative, threshold) for threshold in candidates]
    if any(item["balanced_accuracy"] is None for item in summaries):
        raise ValueError("Threshold selection requires both positive and negative samples")
    selected = max(
        summaries,
        key=lambda item: (
            float(item["balanced_accuracy"]),
            -float(item["false_positive_rate"]),
            float(item["sensitivity"]),
            float(item["threshold"]),
        ),
    )
    return {
        "selected": selected,
        "curve": [
            item
            for item in summaries
            if round(float(item["threshold"]), 2)
            in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}
        ],
    }


def split_scores(
    rows: list[dict[str, str]], scores: dict[str, float], split: str
) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if row["split"] == split]
    positive = [scores[row["image"]] for row in selected if row["sample_role"] == "positive"]
    negative = [scores[row["image"]] for row in selected if row["sample_role"] != "positive"]
    return positive, negative


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    with (dataset / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["split"] != "train"
        ]

    from ultralytics import YOLO

    scene_model = YOLO(str(args.scene_model.resolve()))
    patch_model = YOLO(str(args.patch_model.resolve()))
    score_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_path = dataset / row["image"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        scene_score = classify_scene(
            scene_model, image, args.scene_size, args.device_scene
        )
        patch_max, patch_top3, tile_count = classify_tiles(
            patch_model,
            image,
            args.patch_size,
            args.device_patch,
            args.tile_size,
            args.overlap,
        )
        score_rows.append(
            {
                "image": row["image"],
                "split": row["split"],
                "sample_role": row["sample_role"],
                "split_unit_id": row["split_unit_id"],
                "scene": scene_score,
                "patch_max": patch_max,
                "patch_top3": patch_top3,
                "ensemble_max": max(scene_score, patch_max),
                "ensemble_mean": (scene_score + patch_max) / 2,
                "tile_count": tile_count,
            }
        )
        if index % 25 == 0:
            print(f"scored {index}/{len(rows)}", flush=True)

    methods = ("scene", "patch_max", "patch_top3", "ensemble_max", "ensemble_mean")
    score_maps = {
        method: {row["image"]: float(row[method]) for row in score_rows}
        for method in methods
    }
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "scene_model": str(args.scene_model.resolve()),
        "patch_model": str(args.patch_model.resolve()),
        "threshold_source": "val",
        "methods": {},
    }
    for method in methods:
        validation_positive, validation_negative = split_scores(
            rows, score_maps[method], "val"
        )
        selection = select_threshold(validation_positive, validation_negative)
        threshold = float(selection["selected"]["threshold"])
        test_positive, test_negative = split_scores(
            rows, score_maps[method], "test_internal"
        )
        external_positive, external_negative = split_scores(
            rows, score_maps[method], "test_external"
        )
        report["methods"][method] = {
            "validation": selection,
            "test_internal": balanced_summary(test_positive, test_negative, threshold),
            "test_external": balanced_summary(
                external_positive, external_negative, threshold
            ),
        }

    fields = list(score_rows[0])
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
