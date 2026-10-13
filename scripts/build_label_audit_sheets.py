#!/usr/bin/env python3
"""Build filename-labelled contact sheets for auditing YOLO annotations.

The source images are read from the provenance paths stored in samples.csv so
the audit never changes either the raw data or the prepared dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-unit-id", action="append", default=[])
    parser.add_argument("--candidate-cohort", action="append", default=[])
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def read_yolo_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not path.exists():
        return boxes
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.strip().split()
        if len(fields) < 5:
            continue
        _, cx, cy, width, height = fields[:5]
        boxes.append(tuple(map(float, (cx, cy, width, height))))
    return boxes


def draw_annotation(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> None:
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    line_width = max(4, round(max(image_width, image_height) / 500))
    for cx, cy, width, height in boxes:
        x1 = (cx - width / 2) * image_width
        y1 = (cy - height / 2) * image_height
        x2 = (cx + width / 2) * image_width
        y2 = (cy + height / 2) * image_height
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=line_width)


def build_tile(
    image_path: Path,
    label_path: Path,
    title: str,
    tile_width: int,
    tile_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    caption_height = 52
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    draw_annotation(image, read_yolo_boxes(label_path))
    image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height + caption_height), "white")
    tile.paste(image, ((tile_width - image.width) // 2, (tile_height - image.height) // 2))
    draw = ImageDraw.Draw(tile)
    draw.text((12, tile_height + 8), title, font=font, fill="black")
    return tile


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected_ids = set(args.split_unit_id)
    if selected_ids:
        rows = [row for row in rows if row.get("split_unit_id") in selected_ids]
    selected_cohorts = set(args.candidate_cohort)
    if selected_cohorts:
        rows = [row for row in rows if row.get("candidate_cohort") in selected_cohorts]
    rows.sort(
        key=lambda row: (
            row.get("split_unit_id", row.get("candidate_cohort", "")),
            row["source_image"],
        )
    )

    font = load_font(26)
    page_size = args.columns * args.rows
    sheet_width = args.columns * args.tile_width
    sheet_height = args.rows * (args.tile_height + 52)
    page_count = math.ceil(len(rows) / page_size)

    for page_index in range(page_count):
        page_rows = rows[page_index * page_size : (page_index + 1) * page_size]
        sheet = Image.new("RGB", (sheet_width, sheet_height), (232, 236, 240))
        for index, row in enumerate(page_rows):
            image_path = args.project_root / row["source_image"]
            source_label = row.get("source_label", "")
            label_path = args.project_root / source_label if source_label else Path("__no_label__")
            group_name = row.get("split_unit_id", row.get("candidate_cohort", "unknown"))
            box_count = row.get("box_count", "0")
            title = f"{group_name}  {image_path.stem}  boxes={box_count}"
            tile = build_tile(
                image_path,
                label_path,
                title,
                args.tile_width,
                args.tile_height,
                font,
            )
            x = index % args.columns * args.tile_width
            y = index // args.columns * (args.tile_height + 52)
            sheet.paste(tile, (x, y))
        output_path = args.output_dir / f"audit_{page_index + 1:02d}_of_{page_count:02d}.jpg"
        sheet.save(output_path, quality=92)
        print(output_path)


if __name__ == "__main__":
    main()
