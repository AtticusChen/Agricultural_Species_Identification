#!/usr/bin/env python3
"""Create a symlink-only classification view with repeated negative training images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SPLITS = ("train", "val", "test_internal", "test_external")
CLASSES = ("mikania", "non_mikania")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-copies", type=int, default=2)
    return parser.parse_args()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent))


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if args.negative_copies < 1:
        raise ValueError("negative-copies must be positive")

    counts: list[dict[str, str | int]] = []
    for split in SPLITS:
        for class_name in CLASSES:
            source_dir = source / split / class_name
            destination_dir = output / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(
                path
                for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            copies = (
                args.negative_copies
                if split == "train" and class_name == "non_mikania"
                else 1
            )
            for image in images:
                for copy_index in range(copies):
                    suffix = f"__repeat{copy_index + 1:02d}" if copies > 1 else ""
                    destination = destination_dir / f"{image.stem}{suffix}{image.suffix.lower()}"
                    link(image, destination)
            counts.append(
                {
                    "split": split,
                    "class": class_name,
                    "source_images": len(images),
                    "copies": copies,
                    "output_images": len(images) * copies,
                }
            )

    (output / "summary.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "negative_copies": args.negative_copies,
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
