from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from PIL import Image

from mikania_data.dataset import DetectionDatasetConfig, build_detection_dataset


def _random_image(path: Path, seed: int) -> None:
    random_bytes = random.Random(seed).randbytes(32 * 32 * 3)
    Image.frombytes("RGB", (32, 32), random_bytes).save(path)


def test_build_detection_dataset_normalizes_and_isolates(tmp_path: Path) -> None:
    raw = tmp_path / "raw-data"
    negatives = tmp_path / "data"
    raw.mkdir()
    (negatives / "其他").mkdir(parents=True)
    (raw / "classes.txt").write_text("Mikania micrantha\n", encoding="utf-8")

    labeled_names = [
        "IMG_20250101_100001",
        "IMG_20250101_101001",
        "IMG_20250101_102001",
        "IMG_20250102_100001",
        "IMG_20250102_101001",
        "IMG_20250102_102001",
        "IMG_20260101_100001",
        "IMG_20260101_101001",
    ]
    for index, stem in enumerate(labeled_names):
        _random_image(raw / f"{stem}.jpg", index)
        suffix = ".xml.txt" if index == 0 else ".txt"
        (raw / f"{stem}{suffix}").write_text(
            "0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8"
        )
    _random_image(raw / "IMG_20250103_100001.jpg", 99)
    _random_image(negatives / "其他" / "negative.jpg", 100)

    config = DetectionDatasetConfig(
        version="test_v1",
        seed=17,
        group_window_minutes=5,
        near_duplicate_dhash_threshold=0,
        split_fractions={"train": 0.5, "val": 0.25, "test_internal": 0.25},
        external_years=(2026,),
        class_names=("mikania",),
        link_mode="symlink",
        negative_review_cohorts=("其他",),
        missing_label_triage={},
    )

    output = build_detection_dataset(raw, negatives, tmp_path / "dataset", config)

    summary = json.loads(
        (output / "manifests" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["training_ready_image_count"] == 8
    assert summary["split_image_counts"]["test_external"] == 2
    assert summary["missing_label_count"] == 1
    assert summary["negative_review_candidate_count"] == 1
    assert summary["canonicalized_xml_txt_label_count"] == 1
    assert not list((output / "labels").rglob("*.xml.txt"))
    assert (raw / "IMG_20250101_100001.xml.txt").is_file()
    assert (output / "data.yaml").is_file()
    assert (output / "data_external.yaml").is_file()

    with (output / "manifests" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    units_by_split = {
        split: {row["split_unit_id"] for row in rows if row["split"] == split}
        for split in {row["split"] for row in rows}
    }
    split_names = sorted(units_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            assert units_by_split[left].isdisjoint(units_by_split[right])
