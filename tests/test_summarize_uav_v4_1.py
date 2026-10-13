from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
summary = importlib.import_module("summarize_uav_v4_1")


@pytest.fixture
def report_inputs() -> tuple[list[dict], list[dict], list[dict], dict]:
    old_rows: list[dict] = []
    new_rows: list[dict] = []
    reviews: list[dict] = []
    # Original labels are historical visual-review labels, not expert findings.
    cohorts = (
        ("positive", "uav-site-008", 8, 4, 8),
        ("positive", "uav-site-011", 10, 10, 5),
        ("positive", "uav-site-012", 6, 0, 6),
        ("uncertain", "uav-site-008", 18, 9, 18),
        ("uncertain", "uav-site-009", 1, 1, 0),
    )
    for original_label, group, count, old_hits, new_hits in cohorts:
        for index in range(count):
            filename = f"DJI_{len(reviews):04d}.JPG"
            reviews.append(
                dict(filename=filename, label=original_label, split="test_uav", group_id=group)
            )
            identity = dict(
                image=f"/dataset/uav/{filename}",
                split="test_uav",
                domain="uav",
                label="positive",
                group_id=group,
            )
            old_score = 0.9 if index < old_hits else 0.2
            old_rows.append(dict(identity, scene=old_score, patch_second=old_score))
            # Equality must predict positive; high scene alone must not suffice.
            new_rows.append(
                dict(identity, scene=0.99, patch_second=0.5 if index < new_hits else 0.49)
            )
    for index in range(332):
        reviews.append(
            dict(
                filename=f"REVIEW_{index:04d}.JPG",
                label="positive" if index % 2 else "uncertain",
                split="train",
                group_id="uav-site-002",
            )
        )
    for index in range(307):
        identity = dict(
            image=f"/dataset/ground/GROUND_{index:04d}.jpg",
            split="val" if index % 2 else "test_internal",
            domain="ground",
            label="positive" if index % 3 else "negative",
            group_id=f"ground-{index:04d}",
        )
        old_rows.append(dict(identity, scene=0.4, patch_second=0.4))
        new_rows.append(dict(identity, scene=0.6, patch_second=0.6))
    selection = {
        "threshold_source": "val",
        "frozen_at": "2026-09-14T00:00:00+00:00",
        "model_variant": "H1",
        "selected": {"method": "scene_patch_min", "threshold": 0.5},
    }
    return old_rows, new_rows, reviews, selection


def test_cohort_counts_recall_and_single_class_limits(report_inputs: tuple) -> None:
    report = summary.build_report(*report_inputs)
    cohorts = report["cohorts"]
    expected = {
        "original_positive24": (24, 3, 14, 19, 0.5, 5 / 6),
        "original_uncertain19": (19, 2, 10, 18, 0.75, 0.5),
        "all43": (43, 4, 24, 37, 0.625, 0.625),
    }
    assert set(cohorts) == set(expected)
    for name, (images, groups, old_hits, new_hits, old_group, new_group) in expected.items():
        cohort = cohorts[name]
        assert cohort["images"] == images
        assert cohort["groups"] == groups
        for model, hits, group_recall in (
            ("old", old_hits, old_group),
            ("new", new_hits, new_group),
        ):
            metrics = cohort[model]
            assert metrics["images"] == metrics["positive_images"] == images
            assert metrics["negative_images"] == 0
            assert metrics["true_positive"] == hits
            assert metrics["false_negative"] == images - hits
            assert metrics["sensitivity"] == pytest.approx(hits / images)
            assert metrics["group_macro"]["sensitivity"] == pytest.approx(group_recall)
            for key in ("false_positive_rate", "specificity", "balanced_accuracy"):
                assert metrics[key] is None
                assert metrics["group_macro"][key] is None


def test_extra_group_breakdown_uses_only_original_uncertain_images(report_inputs: tuple) -> None:
    extras = summary.build_report(*report_inputs)["extra_by_group"]
    assert set(extras) == {"uav-site-008", "uav-site-009"}
    for group, count, old_hits, new_hits in (
        ("uav-site-008", 18, 9, 18),
        ("uav-site-009", 1, 1, 0),
    ):
        assert extras[group]["images"] == count
        assert extras[group]["groups"] == 1
        assert extras[group]["old"]["true_positive"] == old_hits
        assert extras[group]["new"]["true_positive"] == new_hits
        assert extras[group]["new"]["false_positive_rate"] is None


def test_old_rule_remains_the_deployed_weighted_baseline(report_inputs: tuple) -> None:
    old_rows = report_inputs[0]
    old_rows[0].update(scene=0.99, patch_second=0.86)  # Weighted score 0.8925: positive.
    old_rows[1].update(scene=0.1, patch_second=1.0)  # Weighted score 0.775: negative.
    old_rows[2].update(scene=0.88, patch_second=0.88)  # Equality is positive.
    report = summary.build_report(*report_inputs)
    assert report["old_rule"] == {
        "method": "ensemble_scene025_patch075", "threshold": 0.88
    }
    assert report["cohorts"]["original_positive24"]["old"]["true_positive"] == 13


def test_historical_labels_are_preserved_without_expert_certification(report_inputs: tuple) -> None:
    report = summary.build_report(*report_inputs)
    assert report["model_selection_performed"] is False
    assert report["validated_counts"] == {
        "old_score_rows": 350,
        "new_score_rows": 350,
        "original_review_rows": 375,
        "test_uav_positive_rows": 43,
    }
    original = {row["filename"]: row for row in report_inputs[2]}
    assert len(report["per_image"]) == 43
    for row in report["per_image"]:
        assert row["label"] == "positive"
        assert row["original_label"] == original[row["filename"]]["label"]
        assert row["group_id"] == original[row["filename"]]["group_id"]
        assert row["split"] == "test_uav"
    notes = " ".join(report["notes"]).lower()
    assert "user report" in notes
    assert "not expert certification" in notes


def test_matching_is_by_identity_not_input_order_and_inputs_are_unchanged(
    report_inputs: tuple,
) -> None:
    before = copy.deepcopy(report_inputs)
    report = summary.build_report(*report_inputs)
    old_rows, new_rows, reviews, selection = report_inputs
    reordered = summary.build_report(
        old_rows, list(reversed(new_rows)), list(reversed(reviews)), selection
    )
    assert reordered["cohorts"] == report["cohorts"]
    assert reordered["extra_by_group"] == report["extra_by_group"]
    assert report_inputs == before


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("operation", ["duplicate", "missing"])
def test_duplicate_or_missing_score_rows_are_rejected(
    report_inputs: tuple, side: int, operation: str
) -> None:
    rows = report_inputs[side]
    if operation == "duplicate":
        rows[-1] = dict(rows[-2])
    else:
        rows.pop()
    with pytest.raises(ValueError):
        summary.build_report(*report_inputs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("image", "/other-root/ground/GROUND_0306.jpg"),
        ("split", "train"),
        ("domain", "uav"),
        ("label", "positive"),
        ("group_id", "different-ground-group"),
    ],
)
def test_global_score_identity_mismatch_is_rejected(
    report_inputs: tuple, field: str, replacement: str
) -> None:
    # A discrepancy outside the 43-image UAV subset must still be rejected.
    assert report_inputs[1][-1][field] != replacement
    report_inputs[1][-1][field] = replacement
    with pytest.raises(ValueError):
        summary.build_report(*report_inputs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("image", "/dataset/uav/UNKNOWN.JPG"), ("group_id", "uav-site-099"),
     ("split", "val"), ("domain", "ground"), ("label", "negative")],
)
def test_shared_score_mutations_cannot_change_original_uav_identity(
    report_inputs: tuple, field: str, replacement: str
) -> None:
    for rows in report_inputs[:2]:
        rows[0][field] = replacement
    with pytest.raises(ValueError):
        summary.build_report(*report_inputs)


@pytest.mark.parametrize(
    "operation", ["duplicate", "missing", "wrong_cohort", "wrong_group", "wrong_extra_group"]
)
def test_original_review_completeness_and_cohort_membership_are_required(
    report_inputs: tuple, operation: str
) -> None:
    reviews = report_inputs[2]
    if operation == "duplicate":
        reviews[-1] = dict(reviews[-2])
    elif operation == "missing":
        reviews.pop()
    elif operation == "wrong_cohort":
        reviews[0]["label"] = "uncertain"
    elif operation == "wrong_group":
        # Keep score/review agreement and 24/19 totals, but change the known groups.
        reviews[0]["group_id"] = "uav-site-011"
        for rows in report_inputs[:2]:
            rows[0]["group_id"] = "uav-site-011"
    else:
        reviews[24]["group_id"] = "uav-site-009"
        for rows in report_inputs[:2]:
            rows[24]["group_id"] = "uav-site-009"
    with pytest.raises(ValueError):
        summary.build_report(*report_inputs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("threshold_source", "test_uav"), ("threshold_source", None), ("frozen_at", "")],
)
def test_unfrozen_or_non_validation_rule_is_rejected(
    report_inputs: tuple, field: str, replacement: object
) -> None:
    report_inputs[3][field] = replacement
    with pytest.raises(ValueError):
        summary.build_report(*report_inputs)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1, None])
def test_invalid_frozen_threshold_is_rejected(report_inputs: tuple, threshold: object) -> None:
    report_inputs[3]["selected"]["threshold"] = threshold
    with pytest.raises(ValueError):
        summary.build_report(*report_inputs)


def test_frozen_method_and_threshold_are_used_without_reselection(
    report_inputs: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid_selection(*args: object, **kwargs: object) -> None:
        pytest.fail("A reporting function must not select a model or threshold")

    evaluation = importlib.import_module("evaluate_models_v4")
    selector = importlib.import_module("select_candidate_v4_1")
    monkeypatch.setattr(evaluation, "select_configuration", forbid_selection)
    monkeypatch.setattr(evaluation, "threshold_candidates", forbid_selection)
    monkeypatch.setattr(selector, "select_rules", forbid_selection)
    for name in ("select_configuration", "threshold_candidates", "select_rules"):
        if hasattr(summary, name):
            monkeypatch.setattr(summary, name, forbid_selection)

    baseline = summary.build_report(*report_inputs)
    report_inputs[3]["selected"]["threshold"] = 0.51
    raised_threshold = summary.build_report(*report_inputs)
    assert raised_threshold["cohorts"]["all43"]["new"]["true_positive"] == 0
    assert raised_threshold["cohorts"]["all43"]["old"] == baseline["cohorts"]["all43"]["old"]
    report_inputs[3]["selected"] = {"method": "scene_only", "threshold": 0.5}
    scene_only = summary.build_report(*report_inputs)
    assert scene_only["cohorts"]["all43"]["new"]["true_positive"] == 43


def test_markdown_renders_cohorts_and_added_groups(report_inputs: tuple) -> None:
    markdown = summary.render_markdown(summary.build_report(*report_inputs))
    assert isinstance(markdown, str)
    for count in (24, 19, 43, 18):
        assert str(count) in markdown
    assert "uav-site-008" in markdown
    assert "uav-site-009" in markdown
    assert "null" in markdown
    assert "非专家" in markdown


def test_cli_writes_validated_reports_and_source_hashes(
    report_inputs: tuple, tmp_path: Path
) -> None:
    inputs: dict[str, Path] = {}
    for key, rows in zip(
        ("old_scores", "new_scores", "original_reviews"), report_inputs[:3], strict=True
    ):
        path = tmp_path / f"{key}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        inputs[key] = path
    inputs["selection"] = tmp_path / "selection.json"
    inputs["selection"].write_text(json.dumps(report_inputs[3]), encoding="utf-8")
    output = tmp_path / "report"
    command = [sys.executable, str(Path(__file__).parents[1] / "scripts" / "summarize_uav_v4_1.py")]
    for key, path in inputs.items():
        command.extend([f"--{key.replace('_', '-')}", str(path)])
    command.extend(["--output", str(output)])
    completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(completed.stdout)["test_uav_positive_rows"] == 43
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["cohorts"]["all43"]["new"]["true_positive"] == 37
    assert report["generated_at"]
    for key, path in inputs.items():
        assert report["inputs"][key] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert (output / "report.md").read_text(encoding="utf-8") == summary.render_markdown(report)
