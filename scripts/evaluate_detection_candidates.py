#!/usr/bin/env python3
"""Compare trained detector candidates on box metrics and field failure cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
REGRESSION_STEMS = (
    "IMG_20250228_140606",
    "IMG_20250227_155423",
    "IMG_20250228_140632",
    "IMG_20251113_163941",
    "IMG_20251118_121305",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-spec",
        action="append",
        required=True,
        help="NAME|PATH|IMGSZ|DEVICE",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    return parser.parse_args()


def parse_spec(raw: str) -> tuple[str, Path, int, str]:
    fields = raw.split("|")
    if len(fields) != 4:
        raise ValueError("model spec must be NAME|PATH|IMGSZ|DEVICE")
    name, path, image_size, device = fields
    return name, Path(path), int(image_size), device


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def smaller_box_overlap(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    smaller_area = min(left_area, right_area)
    return intersection / smaller_area if smaller_area else 0.0


def nms(boxes: list[dict[str, Any]], threshold: float = 0.45) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(boxes, key=lambda item: item["confidence"], reverse=True):
        if all(
            iou(candidate["box_xyxy"], item["box_xyxy"]) < threshold
            and smaller_box_overlap(candidate["box_xyxy"], item["box_xyxy"]) < 0.75
            for item in kept
        ):
            kept.append(candidate)
    return kept


def result_boxes(result: Any, left: int = 0, top: int = 0) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for box in result.boxes:
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        boxes.append(
            {
                "confidence": float(box.conf[0].item()),
                "box_xyxy": [xyxy[0] + left, xyxy[1] + top, xyxy[2] + left, xyxy[3] + top],
            }
        )
    return boxes


def full_predict(
    model: Any, image: Image.Image, image_size: int, device: str
) -> list[dict[str, Any]]:
    result = model.predict(
        source=image,
        imgsz=image_size,
        conf=0.01,
        iou=0.45,
        max_det=300,
        device=device,
        verbose=False,
    )[0]
    return result_boxes(result)


def tiled_predict(
    model: Any,
    image: Image.Image,
    image_size: int,
    device: str,
    tile_size: int,
    overlap: float,
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for top in tile_starts(image.height, tile_size, overlap):
        for left in tile_starts(image.width, tile_size, overlap):
            crop = image.crop(
                (
                    left,
                    top,
                    min(image.width, left + tile_size),
                    min(image.height, top + tile_size),
                )
            )
            result = model.predict(
                source=crop,
                imgsz=image_size,
                conf=0.01,
                iou=0.45,
                max_det=300,
                device=device,
                verbose=False,
            )[0]
            boxes.extend(result_boxes(result, left, top))
    return nms(boxes)


def render(image: Image.Image, boxes: list[dict[str, Any]], threshold: float, output: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(5, round(min(canvas.size) / 200))
    for box in boxes:
        if box["confidence"] < threshold:
            continue
        xyxy = box["box_xyxy"]
        draw.rectangle(xyxy, outline="#ffe000", width=width)
        draw.text((xyxy[0], xyxy[1]), f'{box["confidence"]:.2f}', fill="#ffe000")
    canvas.save(output, quality=92, optimize=True)


def rates(scores: list[float], thresholds: tuple[float, ...]) -> dict[str, float]:
    return {
        f"{threshold:.2f}": sum(score >= threshold for score in scores) / len(scores)
        if scores
        else 0.0
        for threshold in thresholds
    }


def select_threshold(positive_scores: list[float], negative_scores: list[float]) -> float:
    sensitivity = rates(positive_scores, THRESHOLDS)
    false_positive_rate = rates(negative_scores, THRESHOLDS)
    balanced_accuracy = {
        key: (sensitivity[key] + (1 - false_positive_rate[key])) / 2
        for key in sensitivity
    }
    return max(
        THRESHOLDS,
        key=lambda threshold: (
            balanced_accuracy[f"{threshold:.2f}"],
            sensitivity[f"{threshold:.2f}"],
        ),
    )


def summarize_scores(
    positive_scores: list[float], negative_scores: list[float]
) -> dict[str, dict[str, float]]:
    sensitivity = rates(positive_scores, THRESHOLDS)
    false_positive_rate = rates(negative_scores, THRESHOLDS)
    return {
        "sensitivity": sensitivity,
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": {
            key: (sensitivity[key] + (1 - false_positive_rate[key])) / 2
            for key in sensitivity
        },
    }


def locate_regression_image(raw_data: Path, dataset: Path, stem: str) -> Path:
    """Locate a regression image even when the raw-data archive is not on the GPU host."""
    candidates = [
        raw_data / f"{stem}.jpg",
        raw_data / f"{stem}.JPG",
    ]
    for split in ("train", "val", "test_internal", "test_external"):
        candidates.extend(
            (
                dataset / "images" / split / f"{stem}.jpg",
                dataset / "images" / split / f"{stem}.JPG",
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate regression image: {stem}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.resolve()
    project_root = args.project_root.resolve()
    with (dataset / "manifests" / "samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation_rows = [row for row in rows if row["split"] == "val"]
    validation_positive_rows = [
        row for row in validation_rows if row["sample_role"] == "positive"
    ]
    validation_negative_rows = [
        row for row in validation_rows if row["sample_role"] != "positive"
    ]
    test_rows = [row for row in rows if row["split"] == "test_internal"]
    positive_rows = [row for row in test_rows if row["sample_role"] == "positive"]
    negative_rows = [row for row in test_rows if row["sample_role"] != "positive"]

    from ultralytics import YOLO

    report: dict[str, Any] = {
        "dataset": str(dataset),
        "validation": {
            "positive_images": len(validation_positive_rows),
            "negative_images": len(validation_negative_rows),
        },
        "test_internal": {
            "positive_images": len(positive_rows),
            "negative_images": len(negative_rows),
        },
        "models": {},
    }
    for name, model_path, image_size, device in map(parse_spec, args.model_spec):
        resolved_model = model_path if model_path.is_absolute() else project_root / model_path
        model = YOLO(str(resolved_model))
        metrics = model.val(
            data=str(dataset / "data.yaml"),
            split="test",
            imgsz=image_size,
            batch=8,
            conf=0.001,
            iou=0.70,
            device=device,
            plots=False,
            verbose=False,
            project=str(args.output / "ultralytics"),
            name=name,
        )

        full_scores: dict[str, float] = {}
        tiled_scores: dict[str, float] = {}
        for row in validation_rows + test_rows:
            image_path = dataset / row["image"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            boxes = full_predict(model, image, image_size, device)
            tiled = tiled_predict(
                model,
                image,
                image_size,
                device,
                args.tile_size,
                args.tile_overlap,
            )
            full_scores[row["image"]] = max(
                (item["confidence"] for item in boxes), default=0.0
            )
            tiled_scores[row["image"]] = max(
                (item["confidence"] for item in tiled), default=0.0
            )

        validation_positive_full = [
            full_scores[row["image"]] for row in validation_positive_rows
        ]
        validation_negative_full = [
            full_scores[row["image"]] for row in validation_negative_rows
        ]
        validation_positive_tiled = [
            tiled_scores[row["image"]] for row in validation_positive_rows
        ]
        validation_negative_tiled = [
            tiled_scores[row["image"]] for row in validation_negative_rows
        ]
        selected_threshold = select_threshold(
            validation_positive_full, validation_negative_full
        )
        selected_tiled_threshold = select_threshold(
            validation_positive_tiled, validation_negative_tiled
        )
        positive_scores = [full_scores[row["image"]] for row in positive_rows]
        negative_scores = [full_scores[row["image"]] for row in negative_rows]
        positive_tiled_scores = [tiled_scores[row["image"]] for row in positive_rows]
        negative_tiled_scores = [tiled_scores[row["image"]] for row in negative_rows]
        full_summary = summarize_scores(positive_scores, negative_scores)
        tiled_summary = summarize_scores(positive_tiled_scores, negative_tiled_scores)

        regression: dict[str, Any] = {}
        model_output = args.output / name
        model_output.mkdir(parents=True, exist_ok=True)
        for stem in REGRESSION_STEMS:
            image_path = locate_regression_image(args.raw_data, dataset, stem)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            full = full_predict(model, image, image_size, device)
            tiled = tiled_predict(
                model,
                image,
                image_size,
                device,
                args.tile_size,
                args.tile_overlap,
            )
            render(image, full, selected_threshold, model_output / f"{stem}__full.jpg")
            render(
                image,
                tiled,
                selected_tiled_threshold,
                model_output / f"{stem}__tiled.jpg",
            )
            regression[stem] = {
                "expected": "negative" if stem == "IMG_20251118_121305" else "positive",
                "full": {
                    "count": sum(item["confidence"] >= selected_threshold for item in full),
                    "top_confidence": max((item["confidence"] for item in full), default=0.0),
                    "detections": full,
                },
                "tiled": {
                    "count": sum(
                        item["confidence"] >= selected_tiled_threshold for item in tiled
                    ),
                    "top_confidence": max((item["confidence"] for item in tiled), default=0.0),
                    "detections": tiled,
                },
            }

        report["models"][name] = {
            "model": str(resolved_model),
            "image_size": image_size,
            "box_metrics": json_safe(metrics.results_dict),
            "threshold_selection": {
                "source_split": "val",
                "full": {
                    "selected_threshold": selected_threshold,
                    **summarize_scores(
                        validation_positive_full, validation_negative_full
                    ),
                },
                "tiled": {
                    "selected_threshold": selected_tiled_threshold,
                    **summarize_scores(
                        validation_positive_tiled, validation_negative_tiled
                    ),
                },
            },
            "image_level_sensitivity": full_summary["sensitivity"],
            "image_level_false_positive_rate": full_summary["false_positive_rate"],
            "image_level_balanced_accuracy": full_summary["balanced_accuracy"],
            "tiled_image_level_sensitivity": tiled_summary["sensitivity"],
            "tiled_image_level_false_positive_rate": tiled_summary[
                "false_positive_rate"
            ],
            "tiled_image_level_balanced_accuracy": tiled_summary[
                "balanced_accuracy"
            ],
            "selected_threshold": selected_threshold,
            "selected_tiled_threshold": selected_tiled_threshold,
            "positive_top_confidences": positive_scores,
            "negative_top_confidences": negative_scores,
            "positive_tiled_top_confidences": positive_tiled_scores,
            "negative_tiled_top_confidences": negative_tiled_scores,
            "regression": regression,
        }
        print(
            name,
            report["models"][name]["box_metrics"],
            f"full_threshold={selected_threshold:.2f}",
            f"tiled_threshold={selected_tiled_threshold:.2f}",
        )

    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
