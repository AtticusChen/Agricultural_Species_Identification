#!/usr/bin/env python3
"""Freeze a model variant and its decision rule from validation scores only."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from evaluate_models_v4 import read_csv, select_configuration, sha256


def choose(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("At least one validation candidate is required")
    # Threshold magnitudes are not comparable across separately trained models.
    # Equal recall/FPR across variants retains the declared input order.
    return max(
        candidates,
        key=lambda candidate: (
            candidate["selection"]["selected"]["domain_macro_group_sensitivity"],
            -candidate["selection"]["selected"]["domain_macro_group_false_positive_rate"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    manifest_hash = None
    for directory in args.validation_directories:
        rows = read_csv(directory / "scores.csv")
        if any(row["split"] != "val" for row in rows):
            raise ValueError("Candidate selection accepts validation rows only")
        metadata = json.loads((directory / "scoring_config.json").read_text())
        if metadata["splits"] != ["val"]:
            raise ValueError("Scoring provenance is not validation-only")
        if manifest_hash is not None and manifest_hash != metadata["manifest_sha256"]:
            raise ValueError("Candidates were evaluated on different manifests")
        manifest_hash = metadata["manifest_sha256"]
        candidates.append(
            {
                "name": directory.name.removesuffix("_val"),
                "score_file": str((directory / "scores.csv").resolve()),
                "score_sha256": sha256(directory / "scores.csv"),
                "scoring": metadata,
                "selection": select_configuration(rows),
            }
        )
    reference = {row["image"] for row in read_csv(args.validation_directories[0] / "scores.csv")}
    for directory in args.validation_directories[1:]:
        if {row["image"] for row in read_csv(directory / "scores.csv")} != reference:
            raise ValueError("Candidates must contain the identical validation image set")
    winner = choose(candidates)
    frozen = {
        **winner["selection"],
        "model_variant": winner["name"],
        "scoring": winner["scoring"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "variant_selection_policy": (
            "Validation-only; maximize equal-domain/group sensitivity under each domain's "
            "image and group FPR cap of 0.10, then minimize FPR; exact variant ties retain "
            "declared candidate order. No holdout scores are read."
        ),
        "variant_candidates": candidates,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"variant": winner["name"], **frozen["selected"]}, indent=2))


if __name__ == "__main__":
    main()
