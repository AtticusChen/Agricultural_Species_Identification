from __future__ import annotations

import json
from pathlib import Path

import pytest

from mikania_training.config import YoloExperimentConfig


def _write_config(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "experiment_id": "S0_yolo26n_640_smoke",
        "description": "test",
        "model": "yolo26n.pt",
        "data": "datasets/example/data.yaml",
        "dataset_manifest": "datasets/example/manifests/summary.json",
        "output_root": "experiments/runs",
        "device": 0,
        "seed": 20260804,
        "train": {"epochs": 10, "imgsz": 640},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_training_config_loads_and_resolves_project_paths(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "config.json")
    config = YoloExperimentConfig.from_json(path)

    data, manifest, output = config.resolved_paths(tmp_path)

    assert config.seed == 20260804
    assert data == tmp_path / "datasets/example/data.yaml"
    assert manifest == tmp_path / "datasets/example/manifests/summary.json"
    assert output == tmp_path / "experiments/runs"


def test_training_config_rejects_managed_argument_override(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "config.json", train={"epochs": 10, "device": 5}
    )

    with pytest.raises(ValueError, match="managed arguments"):
        YoloExperimentConfig.from_json(path)


def test_training_config_rejects_path_escape(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "config.json", data="../outside.yaml")
    config = YoloExperimentConfig.from_json(path)

    with pytest.raises(ValueError, match="escapes project root"):
        config.resolved_paths(tmp_path)

