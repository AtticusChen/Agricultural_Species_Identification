from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _label_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(candidate, size=15)
    return ImageFont.load_default()


def _new_output(output_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new versioned path: {output}")
    output.mkdir(parents=True)
    return output


def create_contact_sheets(
    manifest_path: Path,
    pilot_root: Path,
    output_dir: Path,
    *,
    per_page: int = 20,
    columns: int = 4,
) -> list[Path]:
    if per_page <= 0 or columns <= 0:
        raise ValueError("per_page and columns must be positive")

    root = pilot_root.expanduser().resolve()
    manifest = manifest_path.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    by_modality: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_modality[row["modality"]].append(row)

    output = _new_output(output_dir)
    tile_width, tile_height, label_height = 320, 240, 26
    page_width = columns * tile_width
    font = _label_font()
    written: list[Path] = []

    for modality, records in sorted(by_modality.items()):
        for page_index, start in enumerate(range(0, len(records), per_page), start=1):
            page_records = records[start : start + per_page]
            rows_on_page = math.ceil(len(page_records) / columns)
            page_height = rows_on_page * (tile_height + label_height)
            page = Image.new("RGB", (page_width, page_height), "white")
            draw = ImageDraw.Draw(page)
            for slot, row in enumerate(page_records):
                relative_image = Path(row["pilot_image"])
                if relative_image.is_absolute() or ".." in relative_image.parts:
                    raise ValueError(f"Invalid pilot image: {row['pilot_image']}")
                image_path = root / relative_image
                if not image_path.is_file():
                    raise FileNotFoundError(f"Pilot image does not exist: {image_path}")
                with Image.open(image_path) as source:
                    source.load()
                    thumbnail = ImageOps.contain(
                        ImageOps.exif_transpose(source).convert("RGB"),
                        (tile_width, tile_height),
                    )
                x = (slot % columns) * tile_width
                y = (slot // columns) * (tile_height + label_height)
                paste_x = x + (tile_width - thumbnail.width) // 2
                paste_y = y + (tile_height - thumbnail.height) // 2
                page.paste(thumbnail, (paste_x, paste_y))
                draw.rectangle(
                    (x, y + tile_height, x + tile_width, y + tile_height + label_height),
                    fill="white",
                )
                draw.text(
                    (x + 6, y + tile_height + 6),
                    f"{row['pilot_id']}  {row['cohort']}",
                    fill="black",
                    font=font,
                )

            destination = output / f"{modality}_page_{page_index:02d}.jpg"
            page.save(destination, quality=88, optimize=True)
            written.append(destination)

    return written
