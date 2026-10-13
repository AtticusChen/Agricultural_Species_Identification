from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "evaluate_models_v4", Path(__file__).parents[1] / "scripts" / "evaluate_models_v4.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def row(
    name: str,
    label: str,
    score: float,
    *,
    domain: str = "ground",
    split: str = "val",
    group: str | None = None,
    patch: float | None = None,
) -> dict:
    return {
        "image": f"/{name}.jpg",
        "label": label,
        "scene": score,
        "patch_second": patch,
        "domain": domain,
        "split": split,
        "group_id": group or name,
    }


def test_single_class_metrics_are_unavailable_and_cannot_select() -> None:
    positive = [row("p", "positive", 0.9)]
    negative = [row("n", "negative", 0.1)]
    pm = evaluation.classification_metrics(positive, "scene_only", 0.5)
    nm = evaluation.classification_metrics(negative, "scene_only", 0.5)
    assert pm["sensitivity"] == 1
    assert pm["false_positive_rate"] is None
    assert pm["balanced_accuracy"] is None
    assert nm["sensitivity"] is None
    assert nm["specificity"] == 1
    assert nm["balanced_accuracy"] is None
    for values in (positive, negative):
        with pytest.raises(ValueError, match="positives and negatives"):
            evaluation.select_configuration(values)


def test_validation_selection_does_not_use_test_scores() -> None:
    validation = [row("vp", "positive", 0.9), row("vn", "negative", 0.1)]
    held_out = [
        row("tp", "positive", 0.0, split="test_internal"),
        row("tn", "negative", 1.0, split="test_internal"),
    ]
    before = evaluation.select_configuration(validation)
    after = evaluation.select_configuration(validation + held_out)
    assert before == after
    report = evaluation.summarize_scores(held_out, before)
    assert report["threshold"] == before["selected"]["threshold"]
    assert report["splits"]["test_internal"]["overall"]["balanced_accuracy"] == 0


def test_domain_fpr_constraint_cannot_be_hidden_by_ground_imbalance() -> None:
    values = [row(f"gn{i}", "negative", 0.1) for i in range(20)]
    values += [
        row("un", "negative", 0.8, domain="uav"),
        row("gp", "positive", 0.7),
        row("up", "positive", 0.95, domain="uav"),
    ]
    selected = evaluation.select_configuration(values)["selected"]
    assert selected["threshold"] > 0.8
    assert selected["validation"]["by_domain"]["uav"]["false_positive_rate"] == 0
    assert selected["validation"]["by_domain"]["ground"]["sensitivity"] == 0


def test_group_macro_does_not_count_adjacent_frames_as_independent_groups() -> None:
    values = [row(f"p{i}", "positive", 0.9, group="long-flight") for i in range(9)]
    values += [row("p-miss", "positive", 0.1, group="short-flight")]
    metrics = evaluation.classification_metrics(values, "scene_only", 0.5)
    assert metrics["sensitivity"] == 0.9
    assert metrics["group_macro"]["sensitivity"] == 0.5
    assert metrics["group_macro"]["positive_groups"] == 2


def test_unknown_labels_and_cross_split_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown domain or label"):
        evaluation.validate_rows([row("x", "unreviewed", 0.5)])
    with pytest.raises(ValueError, match="Group crosses splits"):
        evaluation.validate_rows(
            [
                row("a", "positive", 0.8, group="flight"),
                row("b", "positive", 0.8, split="test_uav", group="flight"),
            ]
        )


def test_patch_only_and_saturated_negative_have_valid_decisions() -> None:
    values = [row("p", "positive", 0.9, patch=0.9), row("n", "negative", 0.1, patch=1.0)]
    for item in values:
        item["scene"] = ""
    result = evaluation.select_configuration(values)["selected"]
    assert result["method"] == "patch_second_only"
    assert result["threshold"] > 1.0
    assert result["validation"]["overall"]["false_positive_rate"] == 0


def test_fixed_candidates_and_missing_uav_negative_are_explicit() -> None:
    values = [
        row("p", "positive", 0.8, patch=0.9),
        row("n", "negative", 0.1, patch=0.2),
        row("u", "positive", 0.7, domain="uav", patch=0.8),
    ]
    result = evaluation.select_configuration(values)
    assert [item["method"] for item in result["candidates"]] == list(evaluation.METHODS)
    assert result["validation_missing_classes"] == {"uav": ["negative"]}
    assert result["selected"]["validation"]["by_domain"]["uav"]["balanced_accuracy"] is None


def test_tile_coverage_includes_edges_and_small_images() -> None:
    assert evaluation.tile_boxes(640, 480) == [(0, 0, 640, 480)]
    boxes = evaluation.tile_boxes(4000, 2250)
    assert boxes[0] == (0, 0, 1280, 1280)
    assert boxes[-1] == (2720, 970, 4000, 2250)
    assert len(boxes) == 8


def test_nonfinite_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        evaluation.classification_metrics([row("nan", "positive", math.nan)], "scene_only", 0.5)


def test_deployed_baseline_remains_fixed_on_new_data() -> None:
    values = [
        row("p", "positive", 0.9, patch=0.9, split="test_uav", domain="uav"),
        row("n", "negative", 0.99, patch=0.99, split="test_uav", domain="uav"),
    ]
    report = evaluation.summarize_scores(values, evaluation.baseline_configuration())
    assert report["threshold"] == 0.88
    assert report["method"] == "ensemble_scene025_patch075"
    assert report["splits"]["test_uav"]["overall"]["false_positive_rate"] == 1


def test_square_preprocessing_preserves_whole_landscape_and_padding() -> None:
    image = Image.new("RGB", (4000, 2250), (0, 255, 0))
    padded = evaluation.pad_square(image)
    assert padded.size == (768, 768)
    assert padded.getpixel((384, 0)) == (114, 114, 114)
    assert padded.getpixel((384, 384)) == (0, 255, 0)


def test_missing_jpeg_end_marker_repaired_only_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), "green").save(path)
    damaged = path.read_bytes()[:-2]
    path.write_bytes(damaged)
    image, action = evaluation.load_image(path)
    assert image.size == (32, 32)
    assert action == "jpeg_eoi_appended_in_memory"
    assert path.read_bytes() == damaged


def test_invalid_non_jpeg_is_not_silently_decoded(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"not a valid image")
    with pytest.raises(OSError):
        evaluation.load_image(path)


def test_resume_rejects_changed_weights_and_incomplete_patches(tmp_path: Path) -> None:
    metadata = {"scene_model_sha256": "same-weight"}
    (tmp_path / "scoring_config.json").write_text(json.dumps(metadata))
    record = row("p", "positive", 0.9)
    record.update(patch_max="", tile_count=0)
    for name, fields, records in (
        ("scores.csv", evaluation.SCORE_FIELDS, [record]),
        ("patch_scores.csv", evaluation.PATCH_FIELDS, []),
    ):
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    assert evaluation.resume_images(tmp_path, metadata, [record]) == {"/p.jpg"}
    with pytest.raises(ValueError, match="changed scoring configuration"):
        evaluation.resume_images(tmp_path, {"scene_model_sha256": "new-weight"}, [record])
    with (tmp_path / "patch_scores.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluation.PATCH_FIELDS)
        orphan = row("orphan", "negative", 0.1)
        writer.writerow({key: orphan[key] for key in evaluation.IDENTITY_FIELDS})
    with pytest.raises(ValueError, match="Incomplete patch"):
        evaluation.resume_images(tmp_path, metadata, [record])
