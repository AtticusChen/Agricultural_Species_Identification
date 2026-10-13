from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

_JPEG_EOI = b"\xff\xd9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_jpeg_eoi(path: Path) -> bool:
    if path.stat().st_size < 2:
        return False
    with path.open("rb") as handle:
        handle.seek(-2, 2)
        return handle.read() == _JPEG_EOI


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_detection_jpegs(
    source_root: Path,
    output_root: Path,
    version: str,
) -> Path:
    source = source_root.resolve()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not version.strip():
        raise ValueError("version cannot be empty")

    samples_path = source / "manifests" / "samples.csv"
    summary_path = source / "manifests" / "summary.json"
    if not samples_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Source dataset must contain samples.csv and summary.json")

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with samples_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError("Source samples.csv is empty")

    (output / "manifests").mkdir(parents=True)
    for split in ("train", "val", "test_internal", "test_external"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    output_rows: list[dict[str, str]] = []
    repair_rows: list[dict[str, str]] = []
    for row in source_rows:
        relative_image = Path(row["image"])
        relative_label = Path(row["label"])
        source_image = source / relative_image
        source_label = source / relative_label
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"Missing sample pair: {relative_image}, {relative_label}")

        target_image = output / relative_image
        target_label = output / relative_label
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, target_label)

        before_sha256 = _sha256(source_image)
        repaired = not _has_jpeg_eoi(target_image)
        if repaired:
            with target_image.open("ab") as handle:
                handle.write(_JPEG_EOI)
        with Image.open(target_image) as image:
            image.verify()
        if not _has_jpeg_eoi(target_image):
            raise ValueError(f"JPEG EOI repair failed: {relative_image}")

        after_sha256 = _sha256(target_image)
        output_row = dict(row)
        output_row["parent_sha256"] = before_sha256
        output_row["sha256"] = after_sha256
        output_row["jpeg_eoi_appended"] = "1" if repaired else "0"
        output_rows.append(output_row)
        if repaired:
            repair_rows.append(
                {
                    "image": relative_image.as_posix(),
                    "before_sha256": before_sha256,
                    "after_sha256": after_sha256,
                    "action": "append_ffd9",
                }
            )

    for name in ("data.yaml", "data_external.yaml", "README.md"):
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, output / name)
    for manifest in ("pending_missing_labels.csv", "pending_negative_review.csv"):
        source_path = source / "manifests" / manifest
        if source_path.is_file():
            shutil.copy2(source_path, output / "manifests" / manifest)

    fieldnames = list(source_rows[0])
    if "parent_sha256" not in fieldnames:
        sha_index = fieldnames.index("sha256")
        fieldnames.insert(sha_index, "parent_sha256")
    if "jpeg_eoi_appended" not in fieldnames:
        fieldnames.append("jpeg_eoi_appended")
    output_samples = output / "manifests" / "samples.csv"
    _write_csv(output_samples, fieldnames, output_rows)
    _write_csv(
        output / "manifests" / "jpeg_eoi_repairs.csv",
        ["image", "before_sha256", "after_sha256", "action"],
        repair_rows,
    )

    summary = dict(source_summary)
    summary.update(
        {
            "version": version,
            "parent_version": source_summary.get("version"),
            "parent_summary_sha256": _sha256(summary_path),
            "jpeg_eoi_repair_method": "append_ffd9_without_reencoding",
            "jpeg_eoi_repaired_count": len(repair_rows),
            "manifest_sha256": _sha256(output_samples),
        }
    )
    (output / "manifests" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output

