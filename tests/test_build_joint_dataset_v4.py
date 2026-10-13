from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

SPEC = importlib.util.spec_from_file_location(
    "build_joint_dataset_v4",
    Path(__file__).parents[1] / "scripts" / "build_joint_dataset_v4.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "convert_uav_reviews_v4_1",
    Path(__file__).parents[1] / "scripts" / "convert_uav_reviews_v4_1.py",
)
assert CONVERTER_SPEC is not None and CONVERTER_SPEC.loader is not None
converter = importlib.util.module_from_spec(CONVERTER_SPEC)
CONVERTER_SPEC.loader.exec_module(converter)


def source(tmp_path: Path, name: str, color: str = "green") -> Path:
    path = tmp_path / "airport" / name
    path.parent.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_uncertain_crop_source_cannot_match_held_out_image(tmp_path: Path) -> None:
    held_out = source(tmp_path, "held_out.png")
    source(tmp_path, "uncertain_crop_source.png")
    samples = [
        {"image": str(held_out), "domain": "uav", "group_id": "test-flight", "split": "test_uav"}
    ]
    reviews = [
        {
            "filename": "uncertain_crop_source.png",
            "label": "uncertain",
            "group_id": "train-flight",
            "split": "train",
            "negative_regions": "0 0 0.2 0.2",
        }
    ]
    with pytest.raises(ValueError, match="Exact image leakage"):
        builder.audit_source_splits(samples, reviews, tmp_path)


def test_uncertain_crop_source_cannot_split_its_capture_group(tmp_path: Path) -> None:
    held_out = source(tmp_path, "held_out.png")
    source(tmp_path, "uncertain.png", "blue")
    samples = [
        {"image": str(held_out), "domain": "uav", "group_id": "same-flight", "split": "test_uav"}
    ]
    reviews = [
        {
            "filename": "uncertain.png",
            "label": "uncertain",
            "group_id": "same-flight",
            "split": "train",
            "negative_regions": "0 0 1 1",
        }
    ]
    with pytest.raises(ValueError, match="Group split leakage"):
        builder.audit_source_splits(samples, reviews, tmp_path)


def test_same_source_crop_and_image_share_split_and_hash(tmp_path: Path) -> None:
    path = source(tmp_path, "positive.png")
    samples = [{"image": str(path), "domain": "uav", "group_id": "flight", "split": "train"}]
    reviews = [
        {
            "filename": "positive.png",
            "label": "positive",
            "group_id": "flight",
            "split": "train",
            "negative_regions": "0 0 0.2 0.2",
        }
    ]
    builder.audit_source_splits(samples, reviews, tmp_path)
    assert len(samples[0]["sha256"]) == 64
    assert reviews[0]["source_sha256"] == samples[0]["sha256"]


def test_user_correction_preserves_roi_and_original_review() -> None:
    original = {
        "filename": "a.jpg",
        "label": "uncertain",
        "split": "train",
        "group_id": "site-1",
        "review": "partial vegetation unclear",
        "regions": "0 0 .4 .4",
        "negative_regions": ".6 .6 1 1",
    }
    row = converter.convert_rows([original])[0]
    assert row["label"] == "positive"
    assert row["provenance"] == "user_reported_positive"
    assert row["original_review_label"] == "uncertain"
    assert row["region_provenance"] == "assistant_visual_provisional"
    for field in ("filename", "split", "group_id", "regions", "negative_regions"):
        assert row[field] == original[field]
    assert "2026-09-14" in row["review"] and original["review"] in row["review"]
    assert original["label"] == "uncertain"


def test_jpeg_eoi_repair_keeps_original_bytes_and_hash(tmp_path: Path) -> None:
    path = source(tmp_path, "truncated.jpg")
    original = path.read_bytes()[:-2]
    path.write_bytes(original)
    samples = [{"image": str(path), "domain": "uav", "group_id": "site", "split": "train"}]
    builder.audit_source_splits(samples, [], tmp_path)
    assert path.read_bytes() == original
    assert samples[0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert samples[0]["decode_warning"] == "jpeg_eoi_appended_in_memory"
    assert len(samples[0]["pixel_sha256"]) == 64


def test_differently_encoded_identical_pixels_cannot_cross_splits(tmp_path: Path) -> None:
    first = source(tmp_path, "plain.png")
    second = tmp_path / "airport" / "metadata.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("note", "different container; identical decoded image")
    with Image.open(first) as image:
        image.save(second, pnginfo=metadata)
    assert first.read_bytes() != second.read_bytes()
    samples = [
        {"image": str(first), "domain": "uav", "group_id": "a", "split": "train"},
        {"image": str(second), "domain": "uav", "group_id": "b", "split": "test_uav"},
    ]
    with pytest.raises(ValueError, match="Decoded pixel image leakage"):
        builder.audit_source_splits(samples, [], tmp_path)


def test_held_out_sources_are_fully_decoded(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    samples = [{"image": str(path), "domain": "uav", "group_id": "a", "split": "test_uav"}]
    with pytest.raises(OSError):
        builder.audit_source_splits(samples, [], tmp_path)


def test_builder_separates_user_photo_and_assistant_roi_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ground = tmp_path / "ground"
    ground.mkdir()
    Image.new("RGB", (12, 12), "orange").save(ground / "g.png")
    (ground / "g.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    builder.write_csv(
        ground / "manifests/samples.csv",
        [
            {
                "image": "g.png",
                "label": "g.txt",
                "split": "train",
                "sample_role": "positive",
                "split_unit_id": "ground-site",
                "source_image": "raw-data/g.png",
            }
        ],
    )
    uav = source(tmp_path, "uav.png")
    source(tmp_path, "no_roi.png", "purple")
    reviews = tmp_path / "reviews.csv"
    template = {
        "filename": uav.name,
        "label": "positive",
        "split": "train",
        "group_id": "uav-site",
        "provenance": "user_reported_positive",
        "original_review_label": "uncertain",
        "regions": "0 0 0.5 0.5",
        "negative_regions": "0.5 0.5 1 1",
    }
    builder.write_csv(
        reviews,
        [
            template,
            {
                **template,
                "filename": "no_roi.png",
                "regions": "",
                "negative_regions": "",
            },
        ],
    )
    output = tmp_path / "dataset"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_joint_dataset_v4.py",
            "--ground",
            str(ground),
            "--upload",
            str(tmp_path),
            "--reviews",
            str(reviews),
            "--output",
            str(output),
            "--version",
            "mikania_joint_v4_1_0",
        ],
    )
    builder.main()
    samples = builder.read_csv(output / "manifests/samples.csv")
    regions = builder.read_csv(output / "manifests/regions.csv")
    uav_samples = [row for row in samples if row["domain"] == "uav"]
    assert len(uav_samples) == 2
    assert all(row["provenance"] == "user_reported_positive" for row in uav_samples)
    roi = [row for row in regions if row["domain"] == "uav"]
    assert len(roi) == 2  # One reviewed positive and negative; no automatic positive tiles.
    assert {row["label"] for row in roi} == {"positive", "negative"}
    assert all(row["provenance"] == "assistant_visual_provisional" for row in roi)
    assert all(row["source_label_provenance"] == "user_reported_positive" for row in roi)
    assert all(row["source_sha256"] == uav_samples[0]["sha256"] for row in roi)
