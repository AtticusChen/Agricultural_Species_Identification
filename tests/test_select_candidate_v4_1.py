from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
selection = importlib.import_module("select_candidate_v4_1")
evaluation = importlib.import_module("evaluate_models_v4")


def row(name, label, scene, patch, split="val", domain="ground"):
    return dict(
        image=f"/{name}.jpg",
        label=label,
        scene=scene,
        patch_second=patch,
        split=split,
        domain=domain,
        group_id=name,
    )


def test_dual_rule_cannot_pass_on_scene_evidence_alone():
    item = row("negative", "negative", 0.99, 0.01)
    assert evaluation.method_score(item, "scene_patch_min") == 0.01
    assert evaluation.method_score(item, "ensemble_scene025_patch075") > 0.2


def test_selector_is_strictly_validation_only_and_requires_both_labels():
    with pytest.raises(ValueError, match="validation-only"):
        selection.select_rules([row("x", "positive", 0.9, 0.9, "test_uav", "uav")])
    with pytest.raises(ValueError, match="both positive and negative"):
        selection.select_rules([row("x", "positive", 0.9, 0.9)])


def test_zero_fpr_and_dual_tie_preference():
    result = selection.select_rules(
        [
            row("gp", "positive", 0.9, 0.9),
            row("up", "positive", 0.8, 0.8, domain="uav"),
            row("gn", "negative", 0.2, 0.1),
        ]
    )["selected"]
    assert result["method"] == "scene_patch_min"
    assert result["validation"]["by_domain"]["ground"]["false_positive_rate"] == 0
    assert result["validation"]["by_domain"]["uav"]["false_positive_rate"] is None
    assert result["domain_macro_group_sensitivity"] == 1


def test_historical_v4_candidates_are_unchanged():
    assert "scene_patch_min" not in evaluation.METHODS
