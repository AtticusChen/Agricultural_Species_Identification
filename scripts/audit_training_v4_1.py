#!/usr/bin/env python3
"""Check downloaded V4.1 artifacts against separately collected remote hashes."""

import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/retraining-v4_1"
    remote = json.loads((output / "remote-training-hashes.json").read_text())
    records = []
    for name, reference in remote.items():
        folder = root / "experiments/retraining_v4_1" / name
        record = json.loads((folder / "run.json").read_text())
        with (folder / "results.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        maximum = max(float(row["metrics/accuracy_top1"]) for row in rows)
        best_epochs = [
            int(row["epoch"]) for row in rows if float(row["metrics/accuracy_top1"]) == maximum
        ]
        local = {file: digest(folder / file) for file in reference["files"]}
        checks = {
            "completed": record["status"] == "completed",
            "all_local_remote_artifact_hashes_match": local == reference["files"],
            "initial_hash_matches_launch": (
                record["initial_weights_sha256"] == reference["initial_sha256"]
            ),
            "manifest_hashes_match": (
                record["manifest_sha256"]
                == reference["manifest_sha256"]
                == digest(output / "manifests/samples.csv")
            ),
            "best_weights_changed_from_initial": (
                local["weights/best.pt"] != reference["initial_sha256"]
            ),
        }
        records.append(
            {
                "name": name,
                "actual_epochs": len(rows),
                "best_epoch_inferred": best_epochs[-1],
                "all_best_top1_epochs": best_epochs,
                "best_val_top1": maximum,
                "checks": checks,
                "files_sha256": local,
            }
        )
    result = {
        "scope": "Deterministic artifact consistency checks, not independent model validation",
        "verdict": "pass" if all(all(r["checks"].values()) for r in records) else "fail",
        "runs": records,
        "note": "Best epoch inferred from final maximal validation top1; no exact resume claim.",
    }
    (output / "training-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["verdict"] != "pass":
        raise RuntimeError("Artifact checks failed")


if __name__ == "__main__":
    main()
