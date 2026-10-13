from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$")
_RESERVED_TRAIN_ARGUMENTS = {"data", "device", "model", "name", "project", "seed"}


@dataclass(frozen=True, slots=True)
class YoloExperimentConfig:
    experiment_id: str
    description: str
    model: str
    data: Path
    dataset_manifest: Path
    output_root: Path
    device: int | str
    seed: int
    train: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> YoloExperimentConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            experiment_id=str(payload["experiment_id"]),
            description=str(payload.get("description", "")),
            model=str(payload["model"]),
            data=Path(payload["data"]),
            dataset_manifest=Path(payload["dataset_manifest"]),
            output_root=Path(payload.get("output_root", "experiments/runs")),
            device=payload.get("device", 0),
            seed=int(payload["seed"]),
            train=dict(payload.get("train", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not _EXPERIMENT_ID_PATTERN.fullmatch(self.experiment_id):
            raise ValueError(
                "experiment_id must be 3-80 safe characters: letters, digits, '.', '_' or '-'"
            )
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.data.is_absolute() or self.dataset_manifest.is_absolute():
            raise ValueError("data and dataset_manifest must be project-relative paths")
        if self.output_root.is_absolute():
            raise ValueError("output_root must be a project-relative path")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        collisions = sorted(_RESERVED_TRAIN_ARGUMENTS.intersection(self.train))
        if collisions:
            raise ValueError(
                "train cannot override managed arguments: " + ", ".join(collisions)
            )
        epochs = self.train.get("epochs")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("train.epochs must be a positive integer")

    def resolved_paths(self, project_root: Path) -> tuple[Path, Path, Path]:
        root = project_root.resolve()
        data = _resolve_within(root, self.data)
        manifest = _resolve_within(root, self.dataset_manifest)
        output = _resolve_within(root, self.output_root)
        return data, manifest, output


def _resolve_within(root: Path, relative_path: Path) -> Path:
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {relative_path}") from error
    return resolved

