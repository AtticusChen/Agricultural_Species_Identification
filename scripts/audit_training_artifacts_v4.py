#!/usr/bin/env python3
"""Read-only remote/local audit of the three already completed v4 training runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/cx/Agricultural_Species_Identification"
RUNS = ("G1_scene512_generic", "G2_scene512_finetune", "G3_patch320_finetune")
ARTIFACTS = ("run.json", "args.yaml", "results.csv", "weights/best.pt", "weights/last.pt")

REMOTE_PROBE = r'''
import hashlib, inspect, json
from pathlib import Path
import torch
import ultralytics
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.metrics import ClassifyMetrics

root = Path("/home/cx/Agricultural_Species_Identification")
names = ("G1_scene512_generic", "G2_scene512_finetune", "G3_patch320_finetune")
artifacts = ("run.json", "args.yaml", "results.csv", "weights/best.pt", "weights/last.pt")
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
out = {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "runs": {}}
out["fitness_source"] = inspect.getsource(ClassifyMetrics.fitness.fget)
out["checkpoint_save_source"] = inspect.getsource(BaseTrainer.save_model)
out["checkpoint_finalization_source"] = inspect.getsource(BaseTrainer.final_eval)
for name in names:
    directory = root / "experiments/retraining_v4" / name
    record = json.loads((directory / "run.json").read_text())
    initial = root / record["arguments"]["model"]
    manifest = root / record["arguments"]["manifest"]
    result = {
        "artifacts": {rel: {"sha256": digest(directory / rel), "bytes": (directory / rel).stat().st_size} for rel in artifacts},
        "initial_path": str(initial), "initial_sha256": digest(initial),
        "manifest_path": str(manifest), "manifest_sha256": digest(manifest),
        "checkpoints": {},
    }
    for kind in ("best", "last"):
        ckpt = torch.load(directory / "weights" / (kind + ".pt"), map_location="cpu", weights_only=False)
        history = ckpt.get("train_results") or {}
        result["checkpoints"][kind] = {
            "stored_epoch": ckpt.get("epoch"),
            "optimizer_state_present": ckpt.get("optimizer") is not None,
            "scaler_state_present": ckpt.get("scaler") is not None,
            "stored_best_fitness": ckpt.get("best_fitness"),
            "train_metrics": ckpt.get("train_metrics"),
            "train_history_epoch_count": len(history.get("epoch", [])),
            "train_history_epochs": history.get("epoch", []),
            "date": ckpt.get("date"), "version": ckpt.get("version"),
            "train_args": {k: ckpt.get("train_args", {}).get(k) for k in ("model", "data", "imgsz", "epochs", "seed", "batch", "patience", "optimizer", "lr0")},
        }
        del ckpt
    out["runs"][name] = result
print("V4_AUDIT_JSON " + json.dumps(out, ensure_ascii=False))
'''


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    remote_command = "/home/cx/.conda/envs/mikania-detect/bin/python -c " + shlex.quote(REMOTE_PROBE)
    process = subprocess.run(
        ["ssh", "-S", "/tmp/mikania-v4-ssh-%r-%h-%p", "cx@10.33.231.250", remote_command],
        text=True, capture_output=True, check=True, timeout=120,
    )
    lines = [line for line in process.stdout.splitlines() if line.startswith("V4_AUDIT_JSON ")]
    if len(lines) != 1:
        raise ValueError("Remote audit did not return exactly one result sentinel")
    remote = json.loads(lines[0].split(" ", 1)[1])
    formula = remote["fitness_source"]
    if "top1" not in formula or "top5" not in formula or "/ 2" not in formula:
        raise ValueError("Classification fitness implementation changed; do not infer best epoch")
    if "if self.best_fitness == self.fitness:" not in remote["checkpoint_save_source"]:
        raise ValueError("Best-checkpoint tie rule changed; do not infer best epoch")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Artifact-only independent training audit; no scoring, application changes, or evaluation integrity verdict.",
        "audit_skill": "experiment-audit (deterministic artifact checks only)",
        "review_independence": "same-family; independent artifact reviewer, not training-script author",
        "acceptance_status": "deterministic_checks_only",
        "remote_root": REMOTE_ROOT,
        "local_root": str(ROOT),
        "runtime": {k: remote[k] for k in ("torch", "ultralytics")},
        "epoch_numbering": "one-based CSV epoch",
        "best_epoch_method": "Last epoch attaining maximal (validation top1 + top5) / 2, matching the installed trainer equality-save rule. Inferred from the CSV because final checkpoint epochs may be stripped.",
        "implementation_evidence": {k: remote[k] for k in ("fitness_source", "checkpoint_save_source", "checkpoint_finalization_source")},
        "runs": [],
    }
    for name in RUNS:
        directory = ROOT / "experiments/retraining_v4" / name
        record = json.loads((directory / "run.json").read_text())
        with (directory / "results.csv").open(newline="") as stream:
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
        epochs = [int(float(row["epoch"])) for row in rows]
        fitness = [(float(row["metrics/accuracy_top1"]) + float(row["metrics/accuracy_top5"])) / 2 for row in rows]
        maximum = max(fitness)
        ties = [epoch for epoch, value in zip(epochs, fitness) if math.isclose(value, maximum, rel_tol=0, abs_tol=1e-12)]
        best_epoch = ties[-1]
        remote_run = remote["runs"][name]
        artifacts = {}
        for relative in ARTIFACTS:
            local_path = directory / relative
            local_hash = digest(local_path)
            remote_artifact = remote_run["artifacts"][relative]
            artifacts[relative] = {
                "local_path": str(local_path),
                "bytes": local_path.stat().st_size,
                "local_sha256": local_hash,
                "remote_sha256": remote_artifact["sha256"],
                "sha256_matches": local_hash == remote_artifact["sha256"],
            }
        best_checkpoint = remote_run["checkpoints"]["best"]
        stored_fitness = (best_checkpoint.get("train_metrics") or {}).get("fitness")
        best_row = rows[epochs.index(best_epoch)]
        metric_keys = ("metrics/accuracy_top1", "metrics/accuracy_top5", "val/loss")
        def metrics_match(checkpoint: dict, row: dict) -> bool:
            metrics = checkpoint.get("train_metrics") or {}
            return all(key in metrics and math.isclose(float(metrics[key]), float(row[key]), abs_tol=1e-5) for key in metric_keys)
        checks = {
            "run_status_completed": record.get("status") == "completed",
            "nonempty_contiguous_epoch_history": bool(epochs) and epochs == list(range(1, epochs[-1] + 1)),
            "actual_epochs_not_above_requested": epochs[-1] <= record["arguments"]["epochs"],
            "all_artifact_local_remote_sha256_match": all(a["sha256_matches"] for a in artifacts.values()),
            "initial_sha256_matches_launch_record": remote_run["initial_sha256"] == record["initial_weights_sha256"],
            "manifest_sha256_matches_launch_record": remote_run["manifest_sha256"] == record["manifest_sha256"],
            "best_sha256_differs_from_initial": artifacts["weights/best.pt"]["local_sha256"] != record["initial_weights_sha256"],
            "last_sha256_differs_from_initial": artifacts["weights/last.pt"]["local_sha256"] != record["initial_weights_sha256"],
            "best_checkpoint_fitness_matches_csv_maximum": stored_fitness is not None and math.isclose(float(stored_fitness), maximum, abs_tol=1e-5),
            "best_checkpoint_metrics_match_selected_epoch": metrics_match(best_checkpoint, best_row),
            "last_checkpoint_metrics_match_final_epoch": metrics_match(remote_run["checkpoints"]["last"], rows[-1]),
            "best_checkpoint_full_history_count_matches_csv": best_checkpoint["train_history_epoch_count"] == len(rows),
            "last_checkpoint_full_history_count_matches_csv": remote_run["checkpoints"]["last"]["train_history_epoch_count"] == len(rows),
        }
        report["runs"].append({
            "name": name,
            "status": record.get("status"),
            "requested_epochs": record["arguments"]["epochs"],
            "actual_epochs": len(rows),
            "final_epoch": epochs[-1],
            "best_epoch": best_epoch,
            "first_maximal_fitness_epoch": ties[0],
            "best_epoch_is_inferred": True,
            "all_maximal_fitness_epochs": ties,
            "stopped_before_requested_epochs": epochs[-1] < record["arguments"]["epochs"],
            "initial_path": remote_run["initial_path"],
            "initial_weights_sha256": record["initial_weights_sha256"],
            "initial_weights_current_remote_sha256": remote_run["initial_sha256"],
            "manifest_sha256": record["manifest_sha256"],
            "arguments": record["arguments"],
            "started_at": record.get("started_at"), "finished_at": record.get("finished_at"),
            "artifacts": artifacts,
            "checkpoints": remote_run["checkpoints"],
            "checks": checks,
            "artifact_verdict": "pass" if all(checks.values()) else "fail",
        })
    report["artifact_verdict"] = "pass" if all(run["artifact_verdict"] == "pass" for run in report["runs"]) else "fail"
    stripped = any(not run["checkpoints"]["last"]["optimizer_state_present"] for run in report["runs"])
    report["reproducibility_note"] = (
        "Both best.pt and last.pt are synced, together with run.json, args.yaml, and results.csv. "
        + ("Final checkpoints have stripped optimizer state and epoch metadata; they support inference/retraining but are not exact optimizer-state resume snapshots. " if stripped else "")
        + "Keeping these files does not alone guarantee bitwise reproduction of a fresh training run. Initial-weight provenance and manifest hashes were verified on the remote host."
    )
    destination = ROOT / "artifacts/retraining-v4/training-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"artifact_verdict": report["artifact_verdict"], "report": str(destination), "runs": [{k: run[k] for k in ("name", "status", "actual_epochs", "best_epoch", "artifact_verdict")} for run in report["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
