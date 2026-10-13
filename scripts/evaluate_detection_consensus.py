#!/usr/bin/env python3
"""Evaluate a two-detector spatial-consensus filter on field images."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from evaluate_detection_candidates import (
    REGRESSION_STEMS,
    full_predict,
    iou,
    locate_regression_image,
    nms,
    render,
    select_threshold,
    smaller_box_overlap,
    summarize_scores,
    tile_starts,
)
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--consensus-overlap", type=float, default=0.30)
    return parser.parse_args()


def consensus_boxes(
    primary: list[dict[str, Any]],
    verifier: list[dict[str, Any]],
    minimum_overlap: float,
) -> list[dict[str, Any]]:
    agreed: list[dict[str, Any]] = []
    for primary_box in primary:
        matches = [
            verifier_box
            for verifier_box in verifier
            if max(
                iou(primary_box["box_xyxy"], verifier_box["box_xyxy"]),
                smaller_box_overlap(
                    primary_box["box_xyxy"], verifier_box["box_xyxy"]
                ),
            )
            >= minimum_overlap
        ]
        if not matches:
            continue
        verifier_box = max(
            matches,
            key=lambda item: (
                max(
                    iou(primary_box["box_xyxy"], item["box_xyxy"]),
                    smaller_box_overlap(
                        primary_box["box_xyxy"], item["box_xyxy"]
                    ),
                ),
                item["confidence"],
            ),
        )
        agreed.append(
            {
                "confidence": math.sqrt(
                    primary_box["confidence"] * verifier_box["confidence"]
                ),
                "box_xyxy": primary_box["box_xyxy"],
                "primary_confidence": primary_box["confidence"],
                "verifier_confidence": verifier_box["confidence"],
            }
        )
    return nms(agreed)


def consensus_full_predict(
    primary_model: Any,
    verifier_model: Any,
    image: Image.Image,
    image_size: int,
    device: str,
    minimum_overlap: float,
) -> list[dict[str, Any]]:
    primary = full_predict(primary_model, image, image_size, device)
    verifier = full_predict(verifier_model, image, image_size, device)
    return consensus_boxes(primary, verifier, minimum_overlap)


def consensus_tiled_predict(
    primary_model: Any,
    verifier_model: Any,
    image: Image.Image,
    image_size: int,
    device: str,
    tile_size: int,
    tile_overlap: float,
    minimum_overlap: float,
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for top in tile_starts(image.height, tile_size, tile_overlap):
        for left in tile_starts(image.width, tile_size, tile_overlap):
            crop = image.crop(
                (
                    left,
                    top,
                    min(image.width, left + tile_size),
                    min(image.height, top + tile_size),
                )
            )
            tile_boxes = consensus_full_predict(
                primary_model,
                verifier_model,
                crop,
                image_size,
                device,
                minimum_overlap,
            )
            for box in tile_boxes:
                x1, y1, x2, y2 = box["box_xyxy"]
                boxes.append(
                    {
                        **box,
                        "box_xyxy": [x1 + left, y1 + top, x2 + left, y2 + top],
                    }
                )
    return nms(boxes)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    validation_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test_internal"]

    from ultralytics import YOLO

    primary_model = YOLO(str(args.primary.resolve()))
    verifier_model = YOLO(str(args.verifier.resolve()))
    full_scores: dict[str, float] = {}
    tiled_scores: dict[str, float] = {}
    for row in validation_rows + test_rows:
        with Image.open(dataset / row["image"]) as opened:
            image = opened.convert("RGB")
        full = consensus_full_predict(
            primary_model,
            verifier_model,
            image,
            args.image_size,
            args.device,
            args.consensus_overlap,
        )
        tiled = consensus_tiled_predict(
            primary_model,
            verifier_model,
            image,
            args.image_size,
            args.device,
            args.tile_size,
            args.tile_overlap,
            args.consensus_overlap,
        )
        full_scores[row["image"]] = max(
            (box["confidence"] for box in full), default=0.0
        )
        tiled_scores[row["image"]] = max(
            (box["confidence"] for box in tiled), default=0.0
        )

    def split_scores(
        selected_rows: list[dict[str, str]], scores: dict[str, float]
    ) -> tuple[list[float], list[float]]:
        positives = [
            scores[row["image"]]
            for row in selected_rows
            if row["sample_role"] == "positive"
        ]
        negatives = [
            scores[row["image"]]
            for row in selected_rows
            if row["sample_role"] != "positive"
        ]
        return positives, negatives

    val_positive_full, val_negative_full = split_scores(validation_rows, full_scores)
    val_positive_tiled, val_negative_tiled = split_scores(
        validation_rows, tiled_scores
    )
    full_threshold = select_threshold(val_positive_full, val_negative_full)
    tiled_threshold = select_threshold(val_positive_tiled, val_negative_tiled)
    test_positive_full, test_negative_full = split_scores(test_rows, full_scores)
    test_positive_tiled, test_negative_tiled = split_scores(test_rows, tiled_scores)

    regression: dict[str, Any] = {}
    for stem in REGRESSION_STEMS:
        image_path = locate_regression_image(args.raw_data, dataset, stem)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        full = consensus_full_predict(
            primary_model,
            verifier_model,
            image,
            args.image_size,
            args.device,
            args.consensus_overlap,
        )
        tiled = consensus_tiled_predict(
            primary_model,
            verifier_model,
            image,
            args.image_size,
            args.device,
            args.tile_size,
            args.tile_overlap,
            args.consensus_overlap,
        )
        render(image, full, full_threshold, output / f"{stem}__full.jpg")
        render(image, tiled, tiled_threshold, output / f"{stem}__tiled.jpg")
        regression[stem] = {
            "expected": "negative" if stem == "IMG_20251118_121305" else "positive",
            "full_count": sum(
                box["confidence"] >= full_threshold for box in full
            ),
            "tiled_count": sum(
                box["confidence"] >= tiled_threshold for box in tiled
            ),
            "full_top_confidence": max(
                (box["confidence"] for box in full), default=0.0
            ),
            "tiled_top_confidence": max(
                (box["confidence"] for box in tiled), default=0.0
            ),
        }

    report = {
        "primary": str(args.primary.resolve()),
        "verifier": str(args.verifier.resolve()),
        "dataset": str(dataset),
        "consensus_overlap": args.consensus_overlap,
        "full_threshold": full_threshold,
        "tiled_threshold": tiled_threshold,
        "validation": {
            "full": summarize_scores(val_positive_full, val_negative_full),
            "tiled": summarize_scores(val_positive_tiled, val_negative_tiled),
        },
        "test_internal": {
            "full": summarize_scores(test_positive_full, test_negative_full),
            "tiled": summarize_scores(test_positive_tiled, test_negative_tiled),
        },
        "regression": regression,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
