from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageOps

from mikania_app.predictor import Detection, OnnxDetector, precise_predict

ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "model_store" / "mikania-yolo26s-v2" / "model.onnx"
IMAGE_PATH = ROOT / "raw-data" / "IMG_20250227_155423.jpg"
NEGATIVE_IMAGE_PATH = ROOT / "raw-data" / "IMG_20251118_121305.jpg"


class _FakeSession:
    def __init__(self, input_shape: list[int]) -> None:
        self.input_shape = input_shape

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="images", shape=self.input_shape)]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="output0", shape=[1, 300, 6])]


def test_onnx_detector_uses_exported_static_image_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *_args, **_kwargs: _FakeSession([1, 3, 960, 960]),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)

    detector = OnnxDetector(model_path)
    detector._load_session()

    assert detector.image_size == 960


def test_onnx_detector_rejects_configured_size_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    fake_runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *_args, **_kwargs: _FakeSession([1, 3, 960, 960]),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)

    detector = OnnxDetector(model_path, image_size=640)

    with pytest.raises(ValueError, match="与配置的 640 不一致"):
        detector._load_session()


def test_precise_predict_offsets_tiles_and_removes_duplicates() -> None:
    image = Image.new("RGB", (2304, 1280), "white")

    def predict_crop(_image: Image.Image, _threshold: float) -> list[Detection]:
        return [Detection("mikania", 0.8, (900.0, 200.0, 1200.0, 600.0))]

    detections = precise_predict(predict_crop, image, 0.15)

    assert [item.box_xyxy for item in detections] == [
        (900.0, 200.0, 1200.0, 600.0),
        (1924.0, 200.0, 2224.0, 600.0),
    ]


def test_precise_predict_removes_box_contained_in_stronger_detection() -> None:
    image = Image.new("RGB", (1280, 1280), "white")

    def predict_crop(_image: Image.Image, _threshold: float) -> list[Detection]:
        return [
            Detection("mikania", 0.9, (100.0, 100.0, 600.0, 600.0)),
            Detection("mikania", 0.7, (180.0, 180.0, 400.0, 400.0)),
        ]

    detections = precise_predict(predict_crop, image, 0.15)

    assert [item.confidence for item in detections] == [0.9]


@pytest.mark.skipif(
    not MODEL_PATH.is_file() or not IMAGE_PATH.exists(), reason="model fixture missing"
)
def test_onnx_detector_matches_exported_model_result() -> None:
    detector = OnnxDetector(MODEL_PATH)
    with Image.open(IMAGE_PATH) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    detections = detector.predict(image, threshold=0.15)

    assert detector.model_name == "mikania-yolo26s-v2"
    assert len(detections) == 2
    assert all(item.label == "mikania" for item in detections)
    assert all(item.confidence >= 0.15 for item in detections)


@pytest.mark.skipif(
    not MODEL_PATH.is_file() or not NEGATIVE_IMAGE_PATH.exists(),
    reason="model fixture missing",
)
def test_onnx_precise_scan_rejects_reported_false_positive() -> None:
    detector = OnnxDetector(MODEL_PATH)
    with Image.open(NEGATIVE_IMAGE_PATH) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    detections = detector.predict_precise(image, threshold=0.15)

    assert detections == []
