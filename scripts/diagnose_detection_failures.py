#!/usr/bin/env python3
"""Generate reproducible overlays for reported Mikania detection failures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mikania_app.predictor import Detection, OnnxDetector  # noqa: E402

DEFAULT_STEMS = (
    "IMG_20250228_140606",
    "IMG_20250227_155423",
    "IMG_20250228_140632",
    "IMG_20251113_163941",
    "IMG_20251118_121305",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "model_store/mikania-yolo26s-v2/model.onnx",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "raw-data")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/failure_diagnosis_v1"
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--diagnostic-threshold", type=float, default=0.05)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("stems", nargs="*", default=list(DEFAULT_STEMS))
    return parser.parse_args()


def find_label(data: Path, stem: str) -> Path:
    candidates = (data / f"{stem}.txt", data / f"{stem}.xml.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No label found for {stem}")


def load_yolo_labels(path: Path, image_size: tuple[int, int]) -> list[Detection]:
    width, height = image_size
    labels: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, box_width, box_height = line.split()
        if int(class_id) != 0:
            continue
        cx_value = float(cx) * width
        cy_value = float(cy) * height
        width_value = float(box_width) * width
        height_value = float(box_height) * height
        labels.append(
            Detection(
                label="mikania",
                confidence=1.0,
                box_xyxy=(
                    cx_value - width_value / 2,
                    cy_value - height_value / 2,
                    cx_value + width_value / 2,
                    cy_value + height_value / 2,
                ),
            )
        )
    return labels


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def box_iou(left: Detection, right: Detection) -> float:
    lx1, ly1, lx2, ly2 = left.box_xyxy
    rx1, ry1, rx2, ry2 = right.box_xyxy
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(box_iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def tiled_predict(
    detector: OnnxDetector,
    image: Image.Image,
    threshold: float,
    tile_size: int,
    overlap: float,
) -> list[Detection]:
    detections: list[Detection] = []
    x_starts = tile_starts(image.width, tile_size, overlap)
    y_starts = tile_starts(image.height, tile_size, overlap)
    for top in y_starts:
        for left in x_starts:
            crop = image.crop((left, top, left + tile_size, top + tile_size))
            for detection in detector.predict(crop, threshold):
                x1, y1, x2, y2 = detection.box_xyxy
                detections.append(
                    Detection(
                        label=detection.label,
                        confidence=detection.confidence,
                        box_xyxy=(x1 + left, y1 + top, x2 + left, y2 + top),
                    )
                )
    return nms(detections)


def best_ious(ground_truth: list[Detection], predictions: list[Detection]) -> list[float]:
    return [
        max((box_iou(target, prediction) for prediction in predictions), default=0.0)
        for target in ground_truth
    ]


def render_overlay(
    image: Image.Image,
    ground_truth: list[Detection],
    predictions: list[Detection],
    output_path: Path,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(5, round(min(canvas.size) / 200))
    for index, target in enumerate(ground_truth, start=1):
        draw.rectangle(target.box_xyxy, outline="#00d9ff", width=line_width)
        draw.text((target.box_xyxy[0], target.box_xyxy[1]), f"GT {index}", fill="#00d9ff")
    for index, prediction in enumerate(predictions, start=1):
        draw.rectangle(prediction.box_xyxy, outline="#ffe000", width=line_width)
        draw.text(
            (prediction.box_xyxy[0], prediction.box_xyxy[1] + line_width * 2),
            f"P{index} {prediction.confidence:.2f}",
            fill="#ffe000",
        )
    canvas.save(output_path, quality=92, optimize=True)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detector = OnnxDetector(args.model)
    report: dict[str, object] = {
        "model": str(args.model.resolve()),
        "threshold": args.threshold,
        "diagnostic_threshold": args.diagnostic_threshold,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "images": {},
    }
    image_reports: dict[str, object] = report["images"]  # type: ignore[assignment]

    for stem in args.stems:
        image_path = args.data / f"{stem}.jpg"
        label_path = find_label(args.data, stem)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        ground_truth = load_yolo_labels(label_path, image.size)
        full_low = detector.predict(image, args.diagnostic_threshold)
        full_default = [item for item in full_low if item.confidence >= args.threshold]
        tiled_low = tiled_predict(
            detector,
            image,
            args.diagnostic_threshold,
            args.tile_size,
            args.tile_overlap,
        )
        tiled_default = [item for item in tiled_low if item.confidence >= args.threshold]

        render_overlay(
            image,
            ground_truth,
            full_default,
            args.output / f"{stem}__full.jpg",
        )
        render_overlay(
            image,
            ground_truth,
            tiled_default,
            args.output / f"{stem}__tiled.jpg",
        )
        image_reports[stem] = {
            "image": str(image_path.resolve()),
            "label": str(label_path.resolve()),
            "image_size": list(image.size),
            "ground_truth_count": len(ground_truth),
            "full": {
                "diagnostic_count": len(full_low),
                "default_count": len(full_default),
                "best_iou_per_ground_truth": best_ious(ground_truth, full_default),
                "detections": [asdict(item) for item in full_low],
            },
            "tiled": {
                "diagnostic_count": len(tiled_low),
                "default_count": len(tiled_default),
                "best_iou_per_ground_truth": best_ious(ground_truth, tiled_default),
                "detections": [asdict(item) for item in tiled_low],
            },
        }
        print(
            f"{stem}: GT={len(ground_truth)} "
            f"full@{args.threshold:.2f}={len(full_default)} "
            f"tiled@{args.threshold:.2f}={len(tiled_default)}"
        )

    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
