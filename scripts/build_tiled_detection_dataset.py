#!/usr/bin/env python3
"""Build a tile-matched detector dataset with mined hard-negative tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

SPLITS = ("train", "val", "test_internal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--negative-tiles-per-image", type=int, default=3)
    parser.add_argument("--inference-size", type=int, default=640)
    parser.add_argument("--version", default="mikania_detection_tiles_v2.1.0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def load_boxes(path: Path, width: int, height: int) -> list[tuple[float, ...]]:
    boxes: list[tuple[float, ...]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) != 5 or int(fields[0]) != 0:
            continue
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        x1 = (center_x - box_width / 2) * width
        y1 = (center_y - box_height / 2) * height
        x2 = (center_x + box_width / 2) * width
        y2 = (center_y + box_height / 2) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_for_tile(
    boxes: list[tuple[float, ...]],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> list[str]:
    tile_width = right - left
    tile_height = bottom - top
    labels: list[str] = []
    for x1, y1, x2, y2 in boxes:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if not (left <= center_x < right and top <= center_y < bottom):
            continue
        clipped_x1 = max(float(left), x1) - left
        clipped_y1 = max(float(top), y1) - top
        clipped_x2 = min(float(right), x2) - left
        clipped_y2 = min(float(bottom), y2) - top
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue
        normalized_center_x = (clipped_x1 + clipped_x2) / 2 / tile_width
        normalized_center_y = (clipped_y1 + clipped_y2) / 2 / tile_height
        normalized_width = (clipped_x2 - clipped_x1) / tile_width
        normalized_height = (clipped_y2 - clipped_y1) / tile_height
        labels.append(
            "0 "
            f"{normalized_center_x:.8f} {normalized_center_y:.8f} "
            f"{normalized_width:.8f} {normalized_height:.8f}"
        )
    return labels


def max_confidence(result: Any) -> float:
    return max((float(box.conf[0].item()) for box in result.boxes), default=0.0)


def save_tile(
    image: Image.Image,
    output: Path,
    split: str,
    name: str,
    labels: list[str],
) -> tuple[Path, Path]:
    relative_image = Path("images") / split / f"{name}.jpg"
    relative_label = Path("labels") / split / f"{name}.txt"
    image.save(output / relative_image, format="JPEG", quality=95, optimize=True)
    (output / relative_label).write_text(
        "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
    )
    return relative_image, relative_label


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    model_path = args.model.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    (output / "manifests").mkdir(parents=True)

    with (source / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    source_rows = [row for row in source_rows if row["split"] in SPLITS]

    from ultralytics import YOLO

    mining_model = YOLO(str(model_path))
    manifest_rows: list[dict[str, str]] = []
    for row in source_rows:
        image_path = source / row["image"]
        label_path = source / row["label"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        boxes = load_boxes(label_path, image.width, image.height)
        candidates: list[tuple[int, int, int, int, Image.Image, list[str]]] = []
        for top in tile_starts(image.height, args.tile_size, args.overlap):
            for left in tile_starts(image.width, args.tile_size, args.overlap):
                right = min(image.width, left + args.tile_size)
                bottom = min(image.height, top + args.tile_size)
                labels = boxes_for_tile(boxes, left, top, right, bottom)
                crop = image.crop((left, top, right, bottom))
                candidates.append((left, top, right, bottom, crop, labels))

        if row["sample_role"] == "positive":
            selected = [(candidate, 0.0) for candidate in candidates if candidate[-1]]
        else:
            results = mining_model.predict(
                source=[candidate[4] for candidate in candidates],
                imgsz=args.inference_size,
                conf=0.001,
                iou=0.45,
                device=args.device,
                verbose=False,
            )
            ranked = sorted(
                zip(candidates, map(max_confidence, results), strict=True),
                key=lambda item: (-item[1], item[0][1], item[0][0]),
            )
            selected = ranked[: args.negative_tiles_per_image]

        for index, (candidate, score) in enumerate(selected, start=1):
            left, top, right, bottom, crop, labels = candidate
            name = f"{Path(row['image']).stem}__x{left:04d}_y{top:04d}_{index:02d}"
            relative_image, relative_label = save_tile(
                crop, output, row["split"], name, labels
            )
            manifest_rows.append(
                {
                    "image": relative_image.as_posix(),
                    "label": relative_label.as_posix(),
                    "split": row["split"],
                    "sample_role": (
                        "positive_tile" if labels else "mined_hard_negative_tile"
                    ),
                    "box_count": str(len(labels)),
                    "parent_image": row["image"],
                    "parent_sample_role": row["sample_role"],
                    "left": str(left),
                    "top": str(top),
                    "right": str(right),
                    "bottom": str(bottom),
                    "mining_confidence": f"{score:.8f}",
                    "sha256": sha256(output / relative_image),
                }
            )

    fields = list(manifest_rows[0])
    manifest_path = output / "manifests" / "samples.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output / "data.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test_internal\n\n"
        "names:\n"
        "  0: mikania\n",
        encoding="utf-8",
    )
    split_counts = Counter(row["split"] for row in manifest_rows)
    role_counts = Counter(row["sample_role"] for row in manifest_rows)
    summary = {
        "version": args.version,
        "parent_dataset": source.name,
        "parent_manifest_sha256": sha256(source / "manifests" / "samples.csv"),
        "mining_model": str(model_path),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "negative_tiles_per_image": args.negative_tiles_per_image,
        "image_count": len(manifest_rows),
        "box_count": sum(int(row["box_count"]) for row in manifest_rows),
        "split_image_counts": dict(sorted(split_counts.items())),
        "sample_role_counts": dict(sorted(role_counts.items())),
        "manifest_sha256": sha256(manifest_path),
    }
    (output / "manifests" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
