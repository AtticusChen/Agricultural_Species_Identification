from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
UAV_COHORT = "无人机"
ROOT_COHORT = "__root__"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    relative_path: str
    cohort: str
    modality: Literal["ground", "uav"]
    candidate_label: str
    width: int
    height: int
    file_bytes: int
    captured_at: str
    camera: str
    has_gps: bool
    group_id: str
    sha256: str
    dhash: str
    feature: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BadImage:
    relative_path: str
    error: str


def _clean_text(value: object) -> str:
    return str(value or "").replace("\x00", "").strip()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value.strip())
    return token.strip("-") or "unknown"


def infer_cohort(relative_path: Path) -> str:
    return ROOT_COHORT if len(relative_path.parts) == 1 else relative_path.parts[0]


def infer_modality(cohort: str) -> Literal["ground", "uav"]:
    return "uav" if cohort == UAV_COHORT else "ground"


def infer_candidate_label(cohort: str) -> str:
    if cohort == ROOT_COHORT:
        return "mikania_candidate"
    if cohort == UAV_COHORT:
        return "mikania_patch_candidate"
    if cohort == "其他":
        return "background_candidate"
    return f"hard_negative_candidate:{cohort}"


def _parse_exif_datetime(value: object) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def build_group_id(
    modality: str,
    cohort: str,
    camera: str,
    captured_at: datetime | None,
    filename: str,
    window_minutes: int,
) -> str:
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    if captured_at is not None:
        floored_minute = (captured_at.minute // window_minutes) * window_minutes
        time_bucket = captured_at.replace(minute=floored_minute, second=0).strftime(
            "%Y%m%d-%H%M"
        )
    else:
        stem_prefix = re.split(r"[_-]", Path(filename).stem, maxsplit=2)
        time_bucket = "-".join(stem_prefix[:2]) or "unknown"
    return "|".join(
        (
            _safe_token(modality),
            _safe_token(cohort),
            _safe_token(camera or "unknown-camera"),
            _safe_token(time_bucket),
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    if hasattr(gray, "get_flattened_data"):
        pixels = list(gray.get_flattened_data())
    else:
        pixels = list(gray.getdata())
    bits = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{bits:0{size * size // 4}x}"


def _visual_feature(image: Image.Image) -> tuple[float, ...]:
    sample = image.copy()
    sample.thumbnail((256, 256), Image.Resampling.LANCZOS)
    rgb = sample.convert("RGB")
    histogram = rgb.histogram()
    channel_features: list[float] = []
    bins_per_channel = 16
    source_bins_per_target = 256 // bins_per_channel
    pixel_count = max(1, rgb.width * rgb.height)
    for channel in range(3):
        channel_hist = histogram[channel * 256 : (channel + 1) * 256]
        for start in range(0, 256, source_bins_per_target):
            channel_features.append(
                sum(channel_hist[start : start + source_bins_per_target]) / pixel_count
            )
    luminance = rgb.convert("L")
    brightness = ImageStat.Stat(luminance).mean[0] / 255.0
    edge_strength = ImageStat.Stat(luminance.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0
    aspect_ratio = rgb.width / max(1, rgb.height)
    return tuple((*channel_features, brightness, edge_strength, aspect_ratio))


def _has_gps(exif: Any) -> bool:
    try:
        return bool(exif.get_ifd(34853))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def inspect_image(path: Path, data_root: Path, window_minutes: int) -> ImageRecord:
    relative = path.relative_to(data_root)
    cohort = infer_cohort(relative)
    with Image.open(path) as raw:
        raw.load()
        exif = raw.getexif()
        captured = _parse_exif_datetime(exif.get(36867) or exif.get(306))
        camera = _clean_text(exif.get(272))
        oriented = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = oriented.size
        record = ImageRecord(
            relative_path=relative.as_posix(),
            cohort=cohort,
            modality=infer_modality(cohort),
            candidate_label=infer_candidate_label(cohort),
            width=width,
            height=height,
            file_bytes=path.stat().st_size,
            captured_at=captured.isoformat(sep=" ") if captured else "",
            camera=camera,
            has_gps=_has_gps(exif),
            group_id=build_group_id(
                modality=infer_modality(cohort),
                cohort=cohort,
                camera=camera,
                captured_at=captured,
                filename=path.name,
                window_minutes=window_minutes,
            ),
            sha256=_sha256(path),
            dhash=_dhash(oriented),
            feature=_visual_feature(oriented),
        )
    return record


def scan_dataset(
    data_root: Path, group_window_minutes: int = 5
) -> tuple[list[ImageRecord], list[BadImage]]:
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    records: list[ImageRecord] = []
    bad_images: list[BadImage] = []
    for path in image_paths:
        try:
            records.append(inspect_image(path, root, group_window_minutes))
        except (OSError, UnidentifiedImageError, ValueError) as error:
            bad_images.append(
                BadImage(relative_path=path.relative_to(root).as_posix(), error=str(error))
            )
    return records, bad_images


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def summarize_records(records: list[ImageRecord], bad_images: list[BadImage]) -> dict[str, Any]:
    cohorts = Counter(record.cohort for record in records)
    modalities = Counter(record.modality for record in records)
    dimensions = Counter(f"{record.width}x{record.height}" for record in records)
    dates = Counter(record.captured_at[:10] or "unknown" for record in records)
    cameras = Counter(record.camera or "unknown" for record in records)
    groups = Counter(record.group_id for record in records)

    exact_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        exact_groups[record.sha256].append(record.relative_path)
    exact_duplicates = [paths for paths in exact_groups.values() if len(paths) > 1]

    near_duplicate_count = 0
    near_duplicate_examples: list[list[str | int]] = []
    by_cohort: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_cohort[record.cohort].append(record)
    for cohort_records in by_cohort.values():
        for index, left in enumerate(cohort_records):
            for right in cohort_records[index + 1 :]:
                distance = _hamming_distance(left.dhash, right.dhash)
                if distance <= 3:
                    near_duplicate_count += 1
                    if len(near_duplicate_examples) < 100:
                        near_duplicate_examples.append(
                            [left.relative_path, right.relative_path, distance]
                        )

    return {
        "total_images": len(records),
        "total_bytes": sum(record.file_bytes for record in records),
        "bad_image_count": len(bad_images),
        "cohort_counts": dict(cohorts.most_common()),
        "modality_counts": dict(modalities.most_common()),
        "dimension_counts": dict(dimensions.most_common()),
        "date_counts": dict(dates.most_common()),
        "camera_counts": dict(cameras.most_common()),
        "gps_image_count": sum(record.has_gps for record in records),
        "group_count": len(groups),
        "largest_groups": groups.most_common(20),
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_pair_count_dhash_le_3": near_duplicate_count,
        "near_duplicate_examples": near_duplicate_examples,
        "bad_images": [asdict(item) for item in bad_images],
    }


def _require_new_output(output_dir: Path) -> Path:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new versioned path: {output}")
    output.mkdir(parents=True)
    return output


def write_audit(
    records: list[ImageRecord], bad_images: list[BadImage], output_dir: Path
) -> Path:
    output = _require_new_output(output_dir)
    inventory_path = output / "inventory.csv"
    fieldnames = [
        "relative_path",
        "cohort",
        "modality",
        "candidate_label",
        "width",
        "height",
        "file_bytes",
        "captured_at",
        "camera",
        "has_gps",
        "group_id",
        "sha256",
        "dhash",
    ]
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row.pop("feature")
            writer.writerow(row)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summarize_records(records, bad_images), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
