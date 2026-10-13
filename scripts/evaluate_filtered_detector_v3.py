#!/usr/bin/env python3
"""Evaluate low-threshold detector proposals filtered by a patch classifier."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from evaluate_classifiers_v3 import mikania_probability
from evaluate_detection_candidates import full_predict, iou
from PIL import Image, ImageOps

DETECTOR_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25)
CLASSIFIER_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-device", default="4")
    parser.add_argument("--classifier-device", default="5")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--classifier-size", type=int, default=320)
    parser.add_argument("--context", type=float, default=0.20)
    return parser.parse_args()


def load_ground_truth(path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    boxes: list[tuple[float, ...]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) != 5 or int(fields[0]) != 0:
            continue
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        boxes.append(
            (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            )
        )
    return boxes


def classifier_crop(
    image: Image.Image, box: list[float], context: float
) -> Image.Image:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    side = max(box_width, box_height) * (1 + 2 * context)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_box = (
        max(0, round(center_x - side / 2)),
        max(0, round(center_y - side / 2)),
        min(image.width, round(center_x + side / 2)),
        min(image.height, round(center_y + side / 2)),
    )
    return image.crop(crop_box)


def add_classifier_scores(
    classifier: Any,
    image: Image.Image,
    proposals: list[dict[str, Any]],
    image_size: int,
    device: str,
    context: float,
) -> list[dict[str, Any]]:
    if not proposals:
        return []
    crops = [classifier_crop(image, proposal["box_xyxy"], context) for proposal in proposals]
    results = classifier.predict(
        source=crops, imgsz=image_size, device=device, batch=64, verbose=False
    )
    enriched: list[dict[str, Any]] = []
    for proposal, result in zip(proposals, results, strict=True):
        classifier_score = mikania_probability(result)
        enriched.append(
            {
                **proposal,
                "classifier_score": classifier_score,
                "combined_score": math.sqrt(
                    float(proposal["confidence"]) * classifier_score
                ),
            }
        )
    return enriched


def filtered(
    proposals: list[dict[str, Any]], detector_threshold: float, classifier_threshold: float
) -> list[dict[str, Any]]:
    return [
        proposal
        for proposal in proposals
        if float(proposal["confidence"]) >= detector_threshold
        and float(proposal["classifier_score"]) >= classifier_threshold
    ]


def box_counts(
    selected: list[dict[str, Any]], ground_truth: list[tuple[float, ...]]
) -> tuple[int, int, int]:
    unmatched = set(range(len(ground_truth)))
    true_positive = 0
    for proposal in sorted(
        selected, key=lambda item: float(item["combined_score"]), reverse=True
    ):
        matches = [
            (iou(tuple(proposal["box_xyxy"]), ground_truth[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(matches, default=(0.0, -1))
        if best_iou >= 0.5:
            true_positive += 1
            unmatched.remove(best_index)
    false_positive = len(selected) - true_positive
    false_negative = len(ground_truth) - true_positive
    return true_positive, false_positive, false_negative


def summarize(
    records: list[dict[str, Any]],
    detector_threshold: float,
    classifier_threshold: float,
) -> dict[str, float | int | None]:
    positive = [record for record in records if record["sample_role"] == "positive"]
    negative = [record for record in records if record["sample_role"] != "positive"]
    positive_hits = sum(
        bool(filtered(record["proposals"], detector_threshold, classifier_threshold))
        for record in positive
    )
    false_positive_images = sum(
        bool(filtered(record["proposals"], detector_threshold, classifier_threshold))
        for record in negative
    )
    sensitivity = positive_hits / len(positive) if positive else None
    false_positive_rate = (
        false_positive_images / len(negative) if negative else None
    )
    balanced_accuracy = (
        (sensitivity + 1 - false_positive_rate) / 2
        if sensitivity is not None and false_positive_rate is not None
        else None
    )
    true_positive_boxes = false_positive_boxes = false_negative_boxes = 0
    for record in records:
        counts = box_counts(
            filtered(record["proposals"], detector_threshold, classifier_threshold),
            record["ground_truth"],
        )
        true_positive_boxes += counts[0]
        false_positive_boxes += counts[1]
        false_negative_boxes += counts[2]
    precision = (
        true_positive_boxes / (true_positive_boxes + false_positive_boxes)
        if true_positive_boxes + false_positive_boxes
        else 0.0
    )
    recall = (
        true_positive_boxes / (true_positive_boxes + false_negative_boxes)
        if true_positive_boxes + false_negative_boxes
        else 0.0
    )
    box_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "detector_threshold": detector_threshold,
        "classifier_threshold": classifier_threshold,
        "positive_images": len(positive),
        "negative_images": len(negative),
        "true_positive_images": positive_hits,
        "false_positive_images": false_positive_images,
        "sensitivity": sensitivity,
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": balanced_accuracy,
        "box_true_positive": true_positive_boxes,
        "box_false_positive": false_positive_boxes,
        "box_false_negative": false_negative_boxes,
        "box_precision": precision,
        "box_recall": recall,
        "box_f1": box_f1,
    }


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (dataset / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["split"] != "train"
        ]

    from ultralytics import YOLO

    detector = YOLO(str(args.detector.resolve()))
    classifier = YOLO(str(args.classifier.resolve()))
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_path = dataset / row["image"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        proposals = full_predict(
            detector, image, args.detector_size, args.detector_device
        )
        proposals = add_classifier_scores(
            classifier,
            image,
            proposals,
            args.classifier_size,
            args.classifier_device,
            args.context,
        )
        records.append(
            {
                "image": row["image"],
                "split": row["split"],
                "sample_role": row["sample_role"],
                "split_unit_id": row["split_unit_id"],
                "ground_truth": load_ground_truth(
                    dataset / row["label"], image.width, image.height
                ),
                "proposals": proposals,
            }
        )
        if index % 25 == 0:
            print(f"scored {index}/{len(rows)}", flush=True)

    validation = [record for record in records if record["split"] == "val"]
    grid = [
        summarize(validation, detector_threshold, classifier_threshold)
        for detector_threshold in DETECTOR_THRESHOLDS
        for classifier_threshold in CLASSIFIER_THRESHOLDS
    ]
    selected = max(
        grid,
        key=lambda item: (
            float(item["balanced_accuracy"] or 0.0),
            float(item["box_f1"] or 0.0),
            float(item["sensitivity"] or 0.0),
        ),
    )
    detector_threshold = float(selected["detector_threshold"])
    classifier_threshold = float(selected["classifier_threshold"])
    report = {
        "dataset": str(dataset),
        "detector": str(args.detector.resolve()),
        "classifier": str(args.classifier.resolve()),
        "selection_split": "val",
        "selected": selected,
        "test_internal": summarize(
            [record for record in records if record["split"] == "test_internal"],
            detector_threshold,
            classifier_threshold,
        ),
        "test_external": summarize(
            [record for record in records if record["split"] == "test_external"],
            detector_threshold,
            classifier_threshold,
        ),
        "validation_grid": grid,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
