from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from mikania_app.main import _default_detector
from mikania_app.onnx_classifier import (
    OnnxClassifierBranch,
    OnnxEnsembleClassifier,
    classifier_tensor,
)

ROOT = Path(__file__).parents[1]


def test_classifier_tensor_matches_frozen_pil_pipeline():
    image = Image.new("RGB", (131, 57), (12, 45, 240))
    expected = ImageOps.pad(image, (768, 768), color=(114, 114, 114))
    expected = expected.resize((320, 320), Image.Resampling.BILINEAR)
    expected = np.asarray(expected, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    actual = classifier_tensor(image, 320)
    assert np.array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert actual.dtype == np.float32


@pytest.mark.parametrize(
    "names,index",
    [("{0: 'mikania', 1: 'non_mikania'}", 0), ("{0: 'non_mikania', 1: 'mikania'}", 1)],
)
def test_onnx_uses_declared_class_index(monkeypatch, tmp_path, names, index):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture")
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images", shape=[1, 3, 320, 320])],
        get_outputs=lambda: [SimpleNamespace(name="probs", shape=[1, 2])],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={"names": names}),
        run=lambda *_: [np.array([[0.2, 0.8]], dtype=np.float32)],
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace, InferenceSession=lambda *_args, **_kwargs: session
        ),
    )
    result = OnnxClassifierBranch(model, 320).classify([Image.new("RGB", (24, 24))])
    assert result == pytest.approx([0.2 if index == 0 else 0.8])
    session.run = lambda *_: [np.array([[float("nan"), 0.2]], dtype=np.float32)]
    with pytest.raises(ValueError, match="无效概率"):
        OnnxClassifierBranch(model, 320).classify([Image.new("RGB", (24, 24))])


def test_main_rejects_onnx_with_wrong_decision_config(monkeypatch):
    monkeypatch.setenv("MIKANIA_MODEL_TASK", "ensemble")
    monkeypatch.setenv("MIKANIA_MODEL_PATH", "patch.onnx")
    monkeypatch.setenv("MIKANIA_DECISION_METHOD", "weighted_sum")
    with pytest.raises(ValueError, match="冻结"):
        _default_detector()


def load_launcher():
    spec = importlib.util.spec_from_file_location("v41_launcher", ROOT / "deploy/v4_1/launcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_rejects_missing_models(monkeypatch, tmp_path):
    launcher = load_launcher()
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.json").write_text("{}")
    (models / "export.json").write_text(json.dumps({"models": {"scene": {"sha256": "wrong"}}}))
    with pytest.raises(RuntimeError, match="模型缺失"):
        launcher.configure()


@pytest.mark.skipif(
    not (ROOT / "model_store/mikania-ensemble-v4.1/scene.onnx").exists(),
    reason="Frozen model is optional in source checkout",
)
def test_real_v4_1_onnx_returns_finite_prediction():
    models = ROOT / "model_store/mikania-ensemble-v4.1"
    classifier = OnnxEnsembleClassifier(models / "scene.onnx", models / "patch.onnx")
    prediction = classifier.predict_result(
        Image.new("RGB", (96, 64), "#68834d"), 0.03896419005468488
    )
    assert np.isfinite(prediction.score)
    assert prediction.present == (prediction.score >= 0.03896419005468488)


def test_launcher_only_reuses_matching_offline_instance(monkeypatch):
    launcher = load_launcher()
    payload = {"status": "ready", "model": "mikania-ensemble-v4.1", "default_threshold": 0.04}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(
        launcher.urllib.request,
        "build_opener",
        lambda *_: SimpleNamespace(open=lambda *_args, **_kwargs: Response()),
    )
    config = {"model_id": payload["model"], "threshold": 0.04}
    assert not launcher.existing_app(18041, config)
    payload["deployment_id"] = launcher.DEPLOYMENT_ID
    assert launcher.existing_app(18041, config)
