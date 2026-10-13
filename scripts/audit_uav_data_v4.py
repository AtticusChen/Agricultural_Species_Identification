#!/usr/bin/env python3
"""Read-only image provenance, UAV spatial grouping, and contact-sheet audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
ANNOTATION_SUFFIXES = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coordinates(gps: dict) -> tuple[float | None, float | None]:
    def convert(value: object, ref: object) -> float | None:
        try:
            a, b, c = value
            number = float(a) + float(b) / 60 + float(c) / 3600
            return -number if str(ref) in {"S", "W", "b'S'", "b'W'"} else number
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    lat = convert(gps.get(2), gps.get(1))
    lon = convert(gps.get(4), gps.get(3))
    if lat == 0 and lon == 0:
        return None, None
    return lat, lon


def capture_time(exif: dict, path: Path) -> str:
    for tag in (36867, 36868, 306):
        value = str(exif.get(tag, ""))
        if re.match(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", value):
            return value[:10].replace(":", "-") + value[10:19]
    match = re.search(r"(20\d{6})[_-]?(\d{6})", path.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").isoformat(" ")
    return ""


def inspect(path: Path, root: Path) -> dict:
    row = {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "filename": path.name,
        "source_folder": str(path.relative_to(root).parent),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
        "width": "",
        "height": "",
        "camera": "",
        "captured_at": "",
        "latitude": None,
        "longitude": None,
        "dhash": "",
        "pixel_sha256": "",
        "uav_group": "",
        "uav_session": "",
        "error": "",
    }
    try:
        with Image.open(path) as source:
            row.update(width=source.width, height=source.height)
            exif = source.getexif()
            gps = exif.get_ifd(34853) if 34853 in exif else {}
            lat, lon = coordinates(gps)
            row.update(
                latitude=lat,
                longitude=lon,
                camera=str(exif.get(272, "")),
                captured_at=capture_time(exif, path),
            )
            tiny = (
                ImageOps.exif_transpose(source)
                .convert("L")
                .resize((9, 8), Image.Resampling.LANCZOS)
            )
            values = list(tiny.tobytes())
            bits = [values[y * 9 + x] > values[y * 9 + x + 1] for y in range(8) for x in range(8)]
            row["dhash"] = f"{sum(int(bit) << (63 - i) for i, bit in enumerate(bits)):016x}"
    except (OSError, ValueError, SyntaxError) as exc:
        row["error"] = str(exc)
    return row


def pixel_digest(path: Path) -> str:
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        value = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
        value.update(rgb.tobytes())
        return value.hexdigest()


def distance(a: dict, b: dict) -> float:
    lat1, lon1 = math.radians(a["latitude"]), math.radians(a["longitude"])
    lat2, lon2 = math.radians(b["latitude"]), math.radians(b["longitude"])
    value = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371000 * 2 * math.asin(min(1, math.sqrt(value)))


def has_gps(row: dict) -> bool:
    return row["latitude"] is not None and row["longitude"] is not None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_uav(rows: list[dict], radius: float) -> list[dict]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, row in enumerate(rows):
        if has_gps(row):
            for j, prior in enumerate(rows[:i]):
                if has_gps(prior) and distance(row, prior) <= radius:
                    parents[find(i)] = find(j)
        else:
            timestamp = row["captured_at"]
            # Missing GPS uses a conservative entire date/time block, never frame-random.
            block = timestamp[:13] if timestamp else "unknown"
            for j, prior in enumerate(rows[:i]):
                prior_time = prior["captured_at"]
                prior_block = prior_time[:13] if prior_time else "unknown"
                if not has_gps(prior) and block == prior_block:
                    parents[find(i)] = find(j)
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    groups = []
    for number, members in enumerate(
        sorted(components.values(), key=lambda items: min(r["filename"] for r in items)), 1
    ):
        group_id = f"uav-site-{number:03d}"
        gps_members = [row for row in members if has_gps(row)]
        for row in members:
            row["uav_group"] = group_id
            row["uav_session"] = group_id + "|" + (row["captured_at"][:10] or "unknown")
        groups.append(
            {
                "uav_group": group_id,
                "count": len(members),
                "gps_count": len(gps_members),
                "dates": "|".join(
                    sorted({row["captured_at"][:10] or "unknown" for row in members})
                ),
                "latitude_mean": sum(r["latitude"] for r in gps_members) / len(gps_members)
                if gps_members
                else "",
                "longitude_mean": sum(r["longitude"] for r in gps_members) / len(gps_members)
                if gps_members
                else "",
                "latitude_min": min((r["latitude"] for r in gps_members), default=""),
                "latitude_max": max((r["latitude"] for r in gps_members), default=""),
                "longitude_min": min((r["longitude"] for r in gps_members), default=""),
                "longitude_max": max((r["longitude"] for r in gps_members), default=""),
                "group_method": "GPS connected components"
                if gps_members
                else "capture-hour block fallback",
            }
        )
    return groups


def contact_sheets(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fonts = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    font_path = next((path for path in fonts if path.exists()), None)
    font = ImageFont.truetype(str(font_path), 15) if font_path else ImageFont.load_default()
    ordered = sorted(rows, key=lambda row: (row["uav_group"], row["captured_at"], row["filename"]))
    index_rows = []
    for start in range(0, len(ordered), 20):
        page = start // 20 + 1
        sheet = Image.new("RGB", (1600, 5 * 267), "#f5f5f5")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(ordered[start : start + 20]):
            x, y = (offset % 4) * 400, (offset // 4) * 267
            try:
                with Image.open(row["path"]) as image:
                    thumbnail = ImageOps.contain(
                        ImageOps.exif_transpose(image).convert("RGB"), (398, 225)
                    )
                    sheet.paste(thumbnail, (x + (400 - thumbnail.width) // 2, y))
            except (OSError, ValueError):
                pass
            display_number = start + offset + 1
            text = f"#{display_number:03d} {row['filename']}"
            while draw.textlength(text, font=font) > 396 and len(text) > 10:
                text = text[:-5] + "..."
            draw.text((x + 2, y + 226), text, font=font, fill="#111111")
            draw.text(
                (x + 2, y + 245),
                row["uav_group"] + "  " + row["captured_at"][:10],
                font=font,
                fill="#333333",
            )
            row["contact_index"] = display_number
            index_rows.append(
                {
                    "contact_index": display_number,
                    "page": page,
                    "position": offset + 1,
                    "uav_group": row["uav_group"],
                    "relative_path": row["relative_path"],
                }
            )
        sheet.save(output / f"uav-contact-{page:02d}.jpg", quality=92)
    write_csv(output.parent / "contact_index.csv", index_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-radius-m", type=float, default=150)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")
    )
    image_paths = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    uav_paths = [path for path in image_paths if "airport" in path.relative_to(root).parts]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        uav_rows = list(executor.map(lambda path: inspect(path, root), uav_paths))
        groups = group_uav(uav_rows, args.spatial_radius_m)
        contact_sheets(uav_rows, output / "contact-sheets")
        write_csv(output / "uav_images.csv", uav_rows)
        write_csv(output / "uav_groups.csv", groups)
        print(
            json.dumps(
                {
                    "stage": "uav_ready",
                    "images": len(uav_rows),
                    "groups": len(groups),
                    "output": str(output),
                }
            ),
            flush=True,
        )
        uav_set = set(uav_paths)
        other_rows = list(
            executor.map(
                lambda path: inspect(path, root), [p for p in image_paths if p not in uav_set]
            )
        )
    rows = uav_rows + other_rows
    manifest = args.old_dataset / "manifests" / "samples.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    old_sha, old_names = defaultdict(list), defaultdict(list)
    for row in old_rows:
        for sha in {row.get("sha256", ""), row.get("parent_sha256", "")} - {""}:
            old_sha[sha].append(row)
        for name in {Path(row["image"]).name, Path(row.get("source_image", "")).name} - {""}:
            old_names[name].append(row)
    pixel_cache = {}
    for row in rows:
        matches = old_sha.get(row["sha256"], [])
        match_type = "sha256_or_parent_sha256" if matches else ""
        if not matches and old_names.get(row["filename"]):
            row["pixel_sha256"] = pixel_digest(Path(row["path"]))
            for candidate in old_names[row["filename"]]:
                old_path = args.old_dataset / candidate["image"]
                if str(old_path) not in pixel_cache:
                    pixel_cache[str(old_path)] = pixel_digest(old_path)
                if pixel_cache[str(old_path)] == row["pixel_sha256"]:
                    matches.append(candidate)
            if matches:
                match_type = "same_decoded_pixels"
        row["old_match_type"] = match_type
        row["old_splits"] = "|".join(sorted({r["split"] for r in matches}))
        row["old_images"] = "|".join(sorted({r["image"] for r in matches}))
        row["old_roles"] = "|".join(sorted({r.get("sample_role", "") for r in matches}))
        row["old_group_ids"] = "|".join(sorted({r.get("split_unit_id", "") for r in matches}))
    old_paths = [args.old_dataset / row["image"] for row in old_rows]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        old_metadata = list(executor.map(lambda path: inspect(path, args.old_dataset), old_paths))
    overlap_rows = []
    for old, metadata in zip(old_rows, old_metadata, strict=True):
        if not has_gps(metadata):
            continue
        distances = [(distance(metadata, uav), uav) for uav in uav_rows if has_gps(uav)]
        if not distances:
            continue
        nearest, uav = min(distances, key=lambda item: item[0])
        close_groups = sorted({r["uav_group"] for d, r in distances if d <= args.spatial_radius_m})
        overlap_rows.append(
            {
                "old_image": old["image"],
                "old_split": old["split"],
                "old_group_id": old.get("group_id", ""),
                "old_split_unit_id": old.get("split_unit_id", ""),
                "latitude": metadata["latitude"],
                "longitude": metadata["longitude"],
                "nearest_uav_group": uav["uav_group"],
                "nearest_uav_distance_m": round(nearest, 2),
                "uav_groups_within_radius": "|".join(close_groups),
            }
        )
    for group in groups:
        overlaps = [
            row
            for row in overlap_rows
            if group["uav_group"] in row["uav_groups_within_radius"].split("|")
        ]
        group["old_ground_overlap_splits"] = "|".join(
            sorted({row["old_split"] for row in overlaps})
        )
        group["old_ground_overlap_count"] = len(overlaps)
        group["old_ground_holdout_overlap_count"] = sum(
            row["old_split"] != "train" for row in overlaps
        )
    annotations = [path for path in files if path.suffix.lower() in ANNOTATION_SUFFIXES]
    duplicate_groups = defaultdict(list)
    for row in rows:
        duplicate_groups[row["sha256"]].append(row["relative_path"])
    summary = {
        "root": str(root),
        "old_dataset": str(args.old_dataset),
        "images": len(rows),
        "image_source_counts": dict(Counter(row["source_folder"] for row in rows)),
        "uav_images": len(uav_rows),
        "uav_groups": len(groups),
        "uav_gps_images": sum(has_gps(row) for row in uav_rows),
        "old_ground_gps_images": sum(has_gps(row) for row in old_metadata),
        "old_ground_missing_gps_images": sum(not has_gps(row) for row in old_metadata),
        "annotation_file_counts": dict(Counter(path.suffix.lower() for path in annotations)),
        "annotation_paths": [str(path.relative_to(root)) for path in annotations],
        "old_match_counts": dict(Counter(row["old_match_type"] or "unmatched" for row in rows)),
        "old_split_match_counts": dict(Counter(row["old_splits"] or "unmatched" for row in rows)),
        "uav_old_match_count": sum(bool(row["old_match_type"]) for row in uav_rows),
        "byte_duplicate_groups": [names for names in duplicate_groups.values() if len(names) > 1],
        "errors": [row for row in rows if row["error"]],
        "spatial_radius_m": args.spatial_radius_m,
        "limitations": [
            "GPS is camera position, not ground footprint; views can overlap beyond 150 m.",
            "Connected components merge chains and repeated sites; do not split frames randomly.",
            "UAV groups near historical holdout locations must not enter that holdout's training.",
            "No GPS proximity match cannot establish independence when historical images lack GPS.",
            "Missing GPS uses capture-hour blocks; distinct blocks may depict the same ground.",
            "No labels assigned; mixed UAV positives and negatives require expert review.",
            "Perceptual dHash is recorded for triage, not proof of duplicate identity.",
        ],
    }
    write_csv(output / "images.csv", rows)
    write_csv(output / "uav_images.csv", uav_rows)
    write_csv(output / "uav_groups.csv", groups)
    write_csv(output / "old_ground_uav_overlap.csv", overlap_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "images": len(rows),
                "uav_images": len(uav_rows),
                "groups": len(groups),
                "old_matches": summary["old_match_counts"],
                "errors": len(summary["errors"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
