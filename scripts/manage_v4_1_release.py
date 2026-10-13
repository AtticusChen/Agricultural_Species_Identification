"""Stage/promote this explicitly approved release; preserve V3 for rollback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

PROJECT = Path("/home/cx/Agricultural_Species_Identification")
RELEASE = PROJECT / "artifacts/release-v4_1-20261013"
UNIT = Path("/home/cx/.config/systemd/user/mikania-app.service")
DROPIN = UNIT.parent / "mikania-app.service.d/41-model-v4_1.conf"
PREVIEW_UNIT = "mikania-v41-preview-20261013"
PYTHON = "/home/cx/.conda/envs/mikania-detect/bin/python"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(args, text=True, capture_output=True, timeout=90)
    if result.returncode:
        print(
            json.dumps(
                {
                    "command": args,
                    "code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        )
    return result


def health(port: int, model: str, timeout: int = 45) -> dict:
    deadline = time.monotonic() + timeout
    last = "no response"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/api/v1/health", timeout=3) as response:
                data = json.load(response)
            if data.get("status") == "ready" and data.get("model") == model:
                return data
            last = str(data)
        except Exception as error:
            last = str(error)
        time.sleep(0.5)
    raise RuntimeError(f"Health failed: {last}")


def stop(unit: str) -> None:
    run("systemctl", "--user", "stop", unit)
    state = run("systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID")
    if (
        state.returncode
        or "ActiveState=inactive" not in state.stdout
        or "MainPID=0" not in state.stdout
    ):
        raise RuntimeError(f"Could not verify stopped service: {state.stdout}")


def environment() -> dict[str, str]:
    config = json.loads((RELEASE / "model.json").read_text())
    return {
        "PYTHONPATH": str(RELEASE / "src"),
        "PYTHONUNBUFFERED": "1",
        "YOLO_CONFIG_DIR": str(PROJECT / ".ultralytics"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MIKANIA_MODEL_PATH": str(RELEASE / "models/patch.pt"),
        "MIKANIA_SCENE_MODEL_PATH": str(RELEASE / "models/scene.pt"),
        "MIKANIA_MODEL_TASK": "ensemble",
        "MIKANIA_MODEL_ID": config["model_id"],
        "MIKANIA_DEVICE": "0",
        "MIKANIA_DEFAULT_SCAN_MODE": "precise",
        "MIKANIA_DECISION_METHOD": config["decision_method"],
        "MIKANIA_PRECISE_THRESHOLD": str(config["threshold"]),
        "MIKANIA_FAST_THRESHOLD": str(config["threshold"]),
        "MIKANIA_CLASSIFIER_IMAGE_SIZE": str(config["patch_image_size"]),
        "MIKANIA_SCENE_IMAGE_SIZE": str(config["scene_image_size"]),
        "MIKANIA_CLASSIFIER_TILE_SIZE": str(config["tile_size"]),
        "MIKANIA_CLASSIFIER_TILE_OVERLAP": str(config["overlap"]),
        "MIKANIA_SCENE_PAD_SQUARE": "1",
        "MIKANIA_PATCH_PAD_SQUARE": "1",
        "MIKANIA_PAD_SIZE": str(config["pad_size"]),
        "MIKANIA_REGION_THRESHOLD": str(config["region_threshold"]),
    }


def prepare() -> None:
    if DROPIN.exists():
        raise RuntimeError("Release drop-in already exists; inspect instead of overwriting")
    health(8000, "mikania-ensemble-v3", timeout=5)
    config = json.loads((RELEASE / "model.json").read_text())
    backup = RELEASE / "backup-v3"
    backup.mkdir(exist_ok=False)
    shutil.copy2(UNIT, backup / UNIT.name)
    shutil.copytree(
        PROJECT / "src/mikania_app",
        backup / "mikania_app",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    run("systemctl", "--user", "cat", "mikania-app")
    (RELEASE / "models").mkdir(exist_ok=True)
    for branch in ("scene", "patch"):
        source = PROJECT / config[f"{branch}_source"]
        if digest(source) != config[f"{branch}_pt_sha256"]:
            raise RuntimeError(f"Frozen weight mismatch: {source}")
        shutil.copy2(source, RELEASE / "models" / f"{branch}.pt")
    env = environment()
    (RELEASE / "server.env").write_text("".join(f"{key}={value}\n" for key, value in env.items()))
    env["MIKANIA_DEVICE"] = "cpu"
    args = ["systemd-run", "--user", f"--unit={PREVIEW_UNIT}", f"--working-directory={PROJECT}"]
    args.extend(f"--setenv={key}={value}" for key, value in env.items())
    run(
        *args,
        PYTHON,
        "-m",
        "uvicorn",
        "mikania_app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    )
    result = health(8001, config["model_id"])
    (RELEASE / "preview-health.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def rollback() -> None:
    if DROPIN.exists():
        saved = RELEASE / f"disabled-v41-{time.time_ns()}.conf"
        shutil.move(str(DROPIN), saved)
    run("systemctl", "--user", "daemon-reload")
    run("systemctl", "--user", "restart", "mikania-app")
    result = health(8000, "mikania-ensemble-v3")
    (RELEASE / "rollback-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def promote() -> None:
    if DROPIN.exists():
        raise RuntimeError("Already promoted or unmanaged drop-in; inspect first")
    validation = json.loads((RELEASE / "preview-validation.json").read_text())
    if validation.get("status") != "passed":
        raise RuntimeError("Real frozen-score regression must pass before promotion")
    backup = RELEASE / "backup-v3"
    if digest(UNIT) != digest(backup / UNIT.name):
        raise RuntimeError("Base service changed since backup")
    for name in ("main.py", "predictor.py"):
        if digest(PROJECT / "src/mikania_app" / name) != digest(backup / "mikania_app" / name):
            raise RuntimeError("Production source changed since backup")
    try:
        stop("mikania-app")
        DROPIN.parent.mkdir(exist_ok=True)
        DROPIN.write_text(f"[Service]\nEnvironmentFile={RELEASE}/server.env\n")
        run("systemctl", "--user", "daemon-reload")
        run("systemctl", "--user", "start", "mikania-app")
        result = health(8000, "mikania-ensemble-v4.1")
        threshold = json.loads((RELEASE / "model.json").read_text())["threshold"]
        if result["default_threshold"] != threshold:
            raise RuntimeError("Frozen threshold mismatch after promotion")
    except Exception:
        rollback()
        raise
    (RELEASE / "promotion-result.json").write_text(json.dumps(result, indent=2) + "\n")
    stop(PREVIEW_UNIT)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "promote", "rollback"))
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT or os.getuid() == 0:
        raise SystemExit("Run only as cx in the project's remote directory")
    {"prepare": prepare, "promote": promote, "rollback": rollback}[args.action]()
