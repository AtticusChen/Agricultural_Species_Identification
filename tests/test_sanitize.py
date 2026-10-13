from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from mikania_data.sanitize import sanitize_detection_jpegs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sanitize_detection_jpegs_appends_eoi_without_changing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for split in ("train", "val", "test_internal", "test_external"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    image_path = source / "images" / "train" / "sample.jpg"
    Image.new("RGB", (32, 32), "green").save(image_path, "JPEG")
    original = image_path.read_bytes()[:-2]
    image_path.write_bytes(original)
    (source / "labels" / "train" / "sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    (source / "data.yaml").write_text("train: images/train\n", encoding="utf-8")
    (source / "manifests").mkdir()
    fieldnames = ["image", "label", "split", "sha256"]
    with (source / "manifests" / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "image": "images/train/sample.jpg",
                "label": "labels/train/sample.txt",
                "split": "train",
                "sha256": _sha256(image_path),
            }
        )
    (source / "manifests" / "summary.json").write_text(
        json.dumps({"version": "parent", "training_ready_image_count": 1}),
        encoding="utf-8",
    )

    output = sanitize_detection_jpegs(source, tmp_path / "output", "child")

    assert image_path.read_bytes() == original
    sanitized = output / "images" / "train" / "sample.jpg"
    assert sanitized.read_bytes() == original + b"\xff\xd9"
    with Image.open(sanitized) as image:
        image.verify()
    summary = json.loads(
        (output / "manifests" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["version"] == "child"
    assert summary["parent_version"] == "parent"
    assert summary["jpeg_eoi_repaired_count"] == 1
    with (output / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["parent_sha256"] == _sha256(image_path)
    assert row["sha256"] == _sha256(sanitized)
    assert row["jpeg_eoi_appended"] == "1"
