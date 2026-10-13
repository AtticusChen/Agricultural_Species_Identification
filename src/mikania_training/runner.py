from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import YoloExperimentConfig


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _environment_snapshot(torch_module: Any) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if torch_module.cuda.is_available():
        index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch_module.cuda.get_device_capability(index)),
        }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
        "torchvision": _package_version("torchvision"),
        "ultralytics": _package_version("ultralytics")
        or _package_version("ultralytics-opencv-headless"),
        "gpu": gpu,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_yolo_experiment(config_path: Path, project_root: Path) -> Path:
    config_path = config_path.resolve()
    root = project_root.resolve()
    os.chdir(root)
    config = YoloExperimentConfig.from_json(config_path)
    data_path, manifest_path, output_root = config.resolved_paths(root)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    log_dir = root / "experiments" / "logs" / config.experiment_id
    if log_dir.exists():
        raise FileExistsError(
            f"Experiment log already exists; choose a new experiment_id: {log_dir}"
        )
    log_dir.mkdir(parents=True)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, log_dir / "config.json")

    yolo_config_root = root / ".ultralytics"
    (yolo_config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_root))
    os.environ.setdefault("WANDB_DISABLED", "true")

    try:
        import torch
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install the project with the 'train' extra."
        ) from error

    if isinstance(config.device, int) and torch.cuda.is_available():
        torch.cuda.set_device(config.device)
    settings.update({"sync": False, "runs_dir": str(output_root)})
    record: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "description": config.description,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "config_sha256": _sha256(config_path),
        "dataset_manifest": str(manifest_path.relative_to(root)),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "environment": _environment_snapshot(torch),
        "managed_arguments": {
            "model": config.model,
            "data": str(data_path),
            "project": str(output_root),
            "name": config.experiment_id,
            "device": config.device,
            "seed": config.seed,
        },
        "train_arguments": config.train,
    }
    record_path = log_dir / "run.json"
    _write_json(record_path, record)

    try:
        model = YOLO(config.model)
        model_source = Path(config.model)
        if not model_source.is_absolute():
            model_source = root / model_source
        if model_source.is_file():
            record["model_source"] = str(model_source.relative_to(root))
            record["model_source_sha256"] = _sha256(model_source)
        results = model.train(
            data=str(data_path),
            project=str(output_root),
            name=config.experiment_id,
            device=config.device,
            seed=config.seed,
            **config.train,
        )
        trainer = model.trainer
        record.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "save_dir": str(trainer.save_dir),
                "best_checkpoint": str(trainer.best),
                "last_checkpoint": str(trainer.last),
                "best_fitness": _json_safe(trainer.best_fitness),
                "metrics": _json_safe(getattr(results, "results_dict", {})),
            }
        )
    except BaseException as error:
        record.update(
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "finished_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _write_json(record_path, record)
        raise

    _write_json(record_path, record)
    return record_path
