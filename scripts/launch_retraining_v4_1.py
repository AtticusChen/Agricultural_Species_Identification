#!/usr/bin/env python3
"""Start the predeclared V4.1 jobs on the existing private training server."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/retraining-v4_1"
    launch_record = output / "launch.json"
    manifest = root / "datasets/mikania_joint_v4_1_0/manifests/samples.csv"
    if launch_record.exists():
        raise FileExistsError("Launch already recorded; inspect it instead of duplicating jobs")
    if not manifest.is_file():
        raise FileNotFoundError("Dataset audit/build must finish before launching")
    snapshot = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in snapshot.splitlines()}
    for device in (1, 2, 3):
        if used.get(device, 100000) >= 500:
            raise RuntimeError(f"GPU {device} is not free: {used}")
    (output / "console").mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable,
        "scripts/train_classifier_v4.py",
        "--data",
        "datasets/mikania_joint_v4_1_0/scene",
        "--manifest",
        str(manifest),
        "--output",
        "experiments/retraining_v4_1",
        "--size",
        "512",
        "--epochs",
        "80",
        "--batch",
        "32",
    ]
    commands = {
        "H1_scene512_alluav": common
        + [
            "--name",
            "H1_scene512_alluav",
            "--device",
            "1",
            "--model",
            "yolo26s-cls.pt",
            "--lr",
            "0.001",
        ],
        "H2_scene512_continue_G1": common
        + [
            "--name",
            "H2_scene512_continue_G1",
            "--device",
            "2",
            "--model",
            "experiments/retraining_v4/G1_scene512_generic/weights/best.pt",
            "--lr",
            "0.0002",
        ],
        "baseline_scores": [
            sys.executable,
            "scripts/evaluate_models_v4.py",
            "--manifest",
            str(manifest),
            "--scene-model",
            "model_store/mikania-yolo26s-scene-v3/model.pt",
            "--patch-model",
            "model_store/mikania-yolo26s-cls-v3/model.pt",
            "--scene-size",
            "384",
            "--patch-size",
            "320",
            "--device",
            "3",
            "--splits",
            "val",
            "test_internal",
            "test_external",
            "test_uav",
            "--output",
            "artifacts/retraining-v4_1/baseline_scores",
        ],
    }
    record = {
        "started_at": datetime.now(UTC).isoformat(),
        "python": sys.executable,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "gpu_preflight": used,
        "jobs": {},
    }
    for name, command in commands.items():
        log = output / "console" / f"{name}.log"
        with log.open("x") as handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        record["jobs"][name] = {"pid": process.pid, "command": command, "log": str(log)}
        launch_record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
