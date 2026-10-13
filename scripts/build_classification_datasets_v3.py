#!/usr/bin/env python3
"""Build scene- and patch-classification datasets from the frozen detector split."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

SPLITS = ("train", "val", "test_internal", "test_external")
CLASSES = ("mikania", "non_mikania")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-patches", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--box-context", type=float, default=0.20)
    parser.add_argument("--version", default="mikania_classification_v3.0.0")
    return parser.parse_args()


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative)


def load_boxes(path: Path, width: int, height: int) -> list[tuple[float, ...]]:
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


def expanded_square(
    box: tuple[float, ...], width: int, height: int, context: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    side = max(box_width, box_height) * (1 + 2 * context)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = max(0, round(center_x - side / 2))
    top = max(0, round(center_y - side / 2))
    right = min(width, round(center_x + side / 2))
    bottom = min(height, round(center_y + side / 2))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def tile_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    center = max(0, (length - tile_size) // 2)
    return sorted({0, center, length - tile_size})


def select_evenly(items: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if len(items) <= count:
        return items
    if count <= 1:
        return [items[len(items) // 2]]
    indices = {round(index * (len(items) - 1) / (count - 1)) for index in range(count)}
    return [items[index] for index in sorted(indices)]


def save_crop(image: Image.Image, box: tuple[int, int, int, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(path, format="JPEG", quality=94, optimize=True)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if args.negative_patches < 1:
        raise ValueError("negative-patches must be positive")
    if args.tile_size < 128:
        raise ValueError("tile-size must be at least 128")
    if not 0 <= args.box_context <= 1:
        raise ValueError("box-context must be in [0, 1]")

    with (source / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in SPLITS]

    scene_manifest: list[dict[str, str]] = []
    patch_manifest: list[dict[str, str]] = []
    for split in SPLITS:
        for class_name in CLASSES:
            (output / "scene" / split / class_name).mkdir(parents=True, exist_ok=True)
            (output / "patch" / split / class_name).mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = source / row["image"]
        label_path = source / row["label"]
        class_name = "mikania" if row["sample_role"] == "positive" else "non_mikania"
        scene_path = output / "scene" / row["split"] / class_name / image_path.name
        link_image(image_path, scene_path)
        scene_manifest.append(
            {
                "image": scene_path.relative_to(output).as_posix(),
                "class": class_name,
                "split": row["split"],
                "source_image": row["image"],
                "split_unit_id": row["split_unit_id"],
                "sample_role": row["sample_role"],
            }
        )

        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        boxes = load_boxes(label_path, image.width, image.height)
        if class_name == "mikania":
            for index, box in enumerate(boxes, start=1):
                crop_box = expanded_square(
                    box, image.width, image.height, args.box_context
                )
                name = f"{image_path.stem}__box{index:02d}.jpg"
                patch_path = output / "patch" / row["split"] / class_name / name
                save_crop(image, crop_box, patch_path)
                patch_manifest.append(
                    {
                        "image": patch_path.relative_to(output).as_posix(),
                        "class": class_name,
                        "split": row["split"],
                        "source_image": row["image"],
                        "split_unit_id": row["split_unit_id"],
                        "crop_xyxy": ",".join(map(str, crop_box)),
                    }
                )
        else:
            positions = [
                (left, top)
                for top in tile_starts(image.height, args.tile_size)
                for left in tile_starts(image.width, args.tile_size)
            ]
            for index, (left, top) in enumerate(
                select_evenly(positions, args.negative_patches), start=1
            ):
                crop_box = (
                    left,
                    top,
                    min(image.width, left + args.tile_size),
                    min(image.height, top + args.tile_size),
                )
                name = f"{image_path.stem}__negative{index:02d}.jpg"
                patch_path = output / "patch" / row["split"] / class_name / name
                save_crop(image, crop_box, patch_path)
                patch_manifest.append(
                    {
                        "image": patch_path.relative_to(output).as_posix(),
                        "class": class_name,
                        "split": row["split"],
                        "source_image": row["image"],
                        "split_unit_id": row["split_unit_id"],
                        "crop_xyxy": ",".join(map(str, crop_box)),
                    }
                )

    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True)
    for name, manifest in (("scene", scene_manifest), ("patch", patch_manifest)):
        path = manifest_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)

    summary = {
        "version": args.version,
        "parent_dataset": source.name,
        "negative_patches_per_image": args.negative_patches,
        "tile_size": args.tile_size,
        "box_context": args.box_context,
    }
    # JSON object keys cannot be tuples, so expose the counters as readable rows.
    summary["scene_counts"] = [
        {"split": split, "class": class_name, "count": count}
        for (split, class_name), count in sorted(
            Counter((row["split"], row["class"]) for row in scene_manifest).items()
        )
    ]
    summary["patch_counts"] = [
        {"split": split, "class": class_name, "count": count}
        for (split, class_name), count in sorted(
            Counter((row["split"], row["class"]) for row in patch_manifest).items()
        )
    ]
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
