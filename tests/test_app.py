from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mikania_app.main import _default_detector, create_app
from mikania_app.predictor import Detection, PresencePrediction, UltralyticsEnsembleClassifier


class FakeDetector:
    ready = True
    model_name = "fake-mikania-model"

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        assert image.mode == "RGB"
        assert threshold == 0.3
        return [Detection("mikania", 0.91, (3.0, 4.0, 24.0, 28.0))]


class PreciseFakeDetector(FakeDetector):
    def predict_precise(self, image: Image.Image, threshold: float) -> list[Detection]:
        assert image.mode == "RGB"
        assert threshold == 0.15
        return [
            Detection("mikania", 0.88, (2.0, 3.0, 14.0, 16.0)),
            Detection("mikania", 0.76, (18.0, 18.0, 30.0, 30.0)),
        ]


def _image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 32), "#678452").save(output, format="JPEG")
    return output.getvalue()


def test_health_reports_model_ready() -> None:
    client = TestClient(create_app(FakeDetector()))

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": "fake-mikania-model",
        "default_threshold": 0.15,
        "default_thresholds": {"precise": 0.15, "fast": 0.15},
        "default_scan_mode": "precise",
    }


def test_predict_returns_presence_and_annotated_image() -> None:
    client = TestClient(create_app(FakeDetector()))

    response = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
        data={"threshold": "0.3", "scan_mode": "fast"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["present"] is True
    assert payload["message"] == "识别到薇甘菊"
    assert payload["count"] == 1
    assert payload["scan_mode"] == "fast"
    assert payload["detections"][0]["confidence"] == 0.91
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")


def test_predict_uses_precise_scan_by_default() -> None:
    client = TestClient(create_app(PreciseFakeDetector()))

    response = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["scan_mode"] == "precise"
    assert payload["threshold"] == 0.15
    assert payload["count"] == 2


def test_predict_rejects_non_image_content_type() -> None:
    client = TestClient(create_app(FakeDetector()))

    response = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        data={"threshold": "0.3", "scan_mode": "fast"},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "仅支持 JPG、PNG 或 WebP 图片"


def test_api_preserves_positive_presence_without_regions_and_exact_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_threshold = 0.12478668255425873
    monkeypatch.setenv("MIKANIA_DEFAULT_THRESHOLD", str(frozen_threshold))

    class PresenceModel(FakeDetector):
        def predict_result(self, image: Image.Image, threshold: float) -> PresencePrediction:
            assert threshold == frozen_threshold
            return PresencePrediction(True, 0.195, [])

    client = TestClient(create_app(PresenceModel()))
    response = client.post(
        "/api/v1/predict", files={"file": ("leaf.jpg", _image_bytes(), "image/jpeg")}
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["present"] is True
    assert payload["score"] == 0.195
    assert payload["count"] == 0
    assert payload["detections"] == []
    assert "暂无可靠区域" in payload["message"]
    assert payload["threshold"] == frozen_threshold


def test_v4_classifier_configuration_is_explicit_and_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIKANIA_MODEL_TASK", "ensemble")
    monkeypatch.setenv("MIKANIA_MODEL_PATH", "patch.pt")
    monkeypatch.setenv("MIKANIA_SCENE_MODEL_PATH", "scene.pt")
    baseline = _default_detector()
    assert isinstance(baseline, UltralyticsEnsembleClassifier)
    assert baseline.model_name == "mikania-ensemble-v3"
    assert baseline.scene_pad_square is False
    assert baseline.patch_model.pad_square is False
    monkeypatch.setenv("MIKANIA_MODEL_ID", "mikania-ensemble-v4")
    monkeypatch.setenv("MIKANIA_SCENE_PAD_SQUARE", "1")
    monkeypatch.setenv("MIKANIA_PATCH_PAD_SQUARE", "true")
    monkeypatch.setenv("MIKANIA_SCENE_IMAGE_SIZE", "512")
    monkeypatch.setenv("MIKANIA_CLASSIFIER_IMAGE_SIZE", "320")
    model = _default_detector()
    assert model.model_name == "mikania-ensemble-v4"
    assert model.scene_image_size == 512
    assert model.patch_model.image_size == 320
    assert model.scene_pad_square is True
    assert model.patch_model.pad_square is True
    assert model.pad_size == model.patch_model.pad_size == 768


def test_upload_missing_jpeg_end_marker_is_decoded_in_memory() -> None:
    client = TestClient(create_app(FakeDetector()))
    damaged_image = _image_bytes()[:-2]
    response = client.post(
        "/api/v1/predict",
        files={"file": ("leaf.jpg", damaged_image, "image/jpeg")},
        data={"threshold": "0.3", "scan_mode": "fast"},
    )
    assert response.status_code == 200
    assert response.json()["present"] is True


def test_ensemble_decision_method_environment_accepts_only_supported_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIKANIA_MODEL_TASK", "ensemble")
    monkeypatch.setenv("MIKANIA_MODEL_PATH", "patch.pt")
    monkeypatch.setenv("MIKANIA_SCENE_MODEL_PATH", "scene.pt")
    assert _default_detector().decision_method == "weighted_sum"
    monkeypatch.setenv("MIKANIA_DECISION_METHOD", "scene_patch_min")
    assert _default_detector().decision_method == "scene_patch_min"
    monkeypatch.setenv("MIKANIA_DECISION_METHOD", "unsupported")
    with pytest.raises(ValueError, match="判定规则"):
        _default_detector()
