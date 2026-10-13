from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mikania_app import predictor
from mikania_app.predictor import (
    UltralyticsEnsembleClassifier,
    ensemble_patch_predict,
    ensemble_patch_prediction,
    patch_classification_predict,
)


def test_patch_classifier_requires_distributed_image_evidence() -> None:
    image = Image.new("RGB", (2304, 2304), "green")

    detections = patch_classification_predict(
        lambda _crops: [0.99, 0.20, 0.10, 0.05],
        image,
        threshold=0.70,
    )

    assert detections == []


def test_patch_classifier_returns_multiple_supported_regions() -> None:
    image = Image.new("RGB", (2304, 2304), "green")

    detections = patch_classification_predict(
        lambda _crops: [0.95, 0.90, 0.80, 0.10],
        image,
        threshold=0.70,
    )

    assert len(detections) == 3
    assert {item.box_xyxy for item in detections} == {
        (0.0, 0.0, 1280.0, 1280.0),
        (1024.0, 0.0, 2304.0, 1280.0),
        (0.0, 1024.0, 1280.0, 2304.0),
    }
    assert all(item.label == "mikania_region" for item in detections)


def test_patch_classifier_supports_small_images() -> None:
    image = Image.new("RGB", (640, 480), "green")

    detections = patch_classification_predict(
        lambda _crops: [0.91],
        image,
        threshold=0.70,
    )

    assert [item.box_xyxy for item in detections] == [(0.0, 0.0, 640.0, 480.0)]


def test_ensemble_rejects_local_evidence_without_scene_support() -> None:
    image = Image.new("RGB", (2304, 2304), "green")

    detections = ensemble_patch_predict(
        lambda _image: 0.10,
        lambda _crops: [0.99, 0.95, 0.10, 0.05],
        image,
        threshold=0.88,
    )

    assert detections == []


def test_ensemble_returns_regions_when_both_signals_support_presence() -> None:
    image = Image.new("RGB", (2304, 2304), "green")

    detections = ensemble_patch_predict(
        lambda _image: 0.99,
        lambda _crops: [0.99, 0.95, 0.10, 0.05],
        image,
        threshold=0.88,
    )

    assert [item.confidence for item in detections] == [0.99, 0.95]


def test_low_threshold_presence_survives_without_displayable_regions() -> None:
    result = ensemble_patch_prediction(
        lambda _image: 0.60,
        lambda _crops: [0.08, 0.06, 0.03, 0.02],
        Image.new("RGB", (2304, 2304)),
        threshold=0.12478668255425873,
    )
    assert result.present is True
    assert result.score == pytest.approx(0.25 * 0.60 + 0.75 * 0.06)
    assert result.detections == []


def test_scene_only_presence_does_not_require_local_evidence() -> None:
    result = ensemble_patch_prediction(
        lambda _image: 0.80,
        lambda _crops: [0.01],
        Image.new("RGB", (640, 480)),
        threshold=0.70,
        scene_weight=1.0,
    )
    assert result.present is True
    assert result.score == 0.80
    assert result.detections == []


def test_negative_image_does_not_show_an_isolated_high_patch() -> None:
    result = ensemble_patch_prediction(
        lambda _image: 0.01,
        lambda _crops: [0.99, 0.02, 0.01, 0.01],
        Image.new("RGB", (2304, 2304)),
        threshold=0.12478668255425873,
    )
    assert result.present is False
    assert result.score < 0.12478668255425873
    assert result.detections == []


@pytest.mark.parametrize("padded", [False, True])
def test_classifier_padding_matches_evaluation_and_keeps_v3_default(
    monkeypatch: pytest.MonkeyPatch, padded: bool
) -> None:
    calls = []

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            images = kwargs["source"] if isinstance(kwargs["source"], list) else [kwargs["source"]]
            return [0.8 for _image in images]

    model = UltralyticsEnsembleClassifier(
        Path("scene.pt"),
        Path("patch.pt"),
        scene_image_size=512 if padded else 384,
        patch_image_size=320,
        scene_pad_square=padded,
        patch_pad_square=padded,
    )
    monkeypatch.setattr(model, "_load_scene_model", lambda: Model())
    monkeypatch.setattr(model.patch_model, "_load_model", lambda: Model())
    monkeypatch.setattr(predictor, "_ultralytics_mikania_probability", lambda value: value)
    result = model.predict_result(Image.new("RGB", (640, 480), "green"), 0.70)
    assert result.present is True
    patch_call, scene_call = calls
    expected_size = (768, 768) if padded else (640, 480)
    assert patch_call["source"][0].size == expected_size
    assert scene_call["source"].size == expected_size
    assert patch_call["imgsz"] == 320
    assert scene_call["imgsz"] == (512 if padded else 384)
    if padded:
        assert scene_call["source"].getpixel((384, 0)) == (114, 114, 114)


def test_min_rule_requires_local_support_even_when_scene_is_high() -> None:
    image = Image.new("RGB", (640, 480))
    weighted = ensemble_patch_prediction(
        lambda _image: 0.99,
        lambda _crops: [0.10],
        image,
        0.30,
    )
    gated = ensemble_patch_prediction(
        lambda _image: 0.99,
        lambda _crops: [0.10],
        image,
        0.30,
        decision_method="scene_patch_min",
    )
    assert weighted.present is True
    assert weighted.score == pytest.approx(0.25 * 0.99 + 0.75 * 0.10)
    assert gated.present is False
    assert gated.score == 0.10
    assert gated.detections == []


def test_default_weighted_rule_preserves_v3_score_exactly() -> None:
    image = Image.new("RGB", (2304, 2304))
    default = ensemble_patch_prediction(
        lambda _image: 0.99,
        lambda _crops: [0.99, 0.95, 0.10, 0.05],
        image,
        0.88,
    )
    explicit = ensemble_patch_prediction(
        lambda _image: 0.99,
        lambda _crops: [0.99, 0.95, 0.10, 0.05],
        image,
        0.88,
        decision_method="weighted_sum",
    )
    assert default == explicit
    assert default.score == 0.25 * 0.99 + 0.75 * 0.95
    assert default.present is True
    assert len(default.detections) == 2


def test_unknown_decision_rule_is_rejected_before_inference() -> None:
    with pytest.raises(ValueError, match="判定规则"):
        UltralyticsEnsembleClassifier(Path("scene.pt"), Path("patch.pt"), decision_method="other")
    with pytest.raises(ValueError, match="判定规则"):
        ensemble_patch_prediction(
            lambda _image: pytest.fail("Should reject configuration before inference"),
            lambda _crops: pytest.fail("Should reject configuration before inference"),
            Image.new("RGB", (32, 32)),
            0.5,
            decision_method="other",
        )
