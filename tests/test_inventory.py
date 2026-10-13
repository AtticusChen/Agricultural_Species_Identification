from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from mikania_data.inventory import (
    ROOT_COHORT,
    build_group_id,
    infer_candidate_label,
    infer_cohort,
    scan_dataset,
)


def test_infer_cohort_and_candidate_labels() -> None:
    assert infer_cohort(Path("IMG_0001.jpg")) == ROOT_COHORT
    assert infer_cohort(Path("假臭草/IMG_0002.jpg")) == "假臭草"
    assert infer_candidate_label(ROOT_COHORT) == "mikania_candidate"
    assert infer_candidate_label("其他") == "background_candidate"
    assert infer_candidate_label("水蓼") == "hard_negative_candidate:水蓼"


def test_group_id_uses_capture_window() -> None:
    first = build_group_id(
        "ground",
        ROOT_COHORT,
        "camera",
        datetime(2026, 8, 2, 10, 1, 59),
        "a.jpg",
        5,
    )
    second = build_group_id(
        "ground",
        ROOT_COHORT,
        "camera",
        datetime(2026, 8, 2, 10, 4, 1),
        "b.jpg",
        5,
    )
    third = build_group_id(
        "ground",
        ROOT_COHORT,
        "camera",
        datetime(2026, 8, 2, 10, 5, 0),
        "c.jpg",
        5,
    )
    assert first == second
    assert first != third


def test_scan_dataset_reads_nested_images(tmp_path: Path) -> None:
    (tmp_path / "假臭草").mkdir()
    Image.new("RGB", (64, 48), "green").save(tmp_path / "root.jpg")
    Image.new("RGB", (48, 64), "purple").save(tmp_path / "假臭草" / "negative.jpg")

    records, bad = scan_dataset(tmp_path)

    assert not bad
    assert len(records) == 2
    assert {record.cohort for record in records} == {ROOT_COHORT, "假臭草"}
    assert all(len(record.sha256) == 64 for record in records)
    assert all(len(record.dhash) == 16 for record in records)

