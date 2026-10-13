#!/usr/bin/env python3
"""Freeze V4.1 model and rule using validation only and zero observed val FPR."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from evaluate_models_v4 import (
    IDENTITY_FIELDS,
    _selection_stats,
    breakdown,
    method_score,
    read_csv,
    sha256,
    threshold_candidates,
    validate_rows,
)

# Ties prefer dual evidence; this list is frozen before candidate scoring.
RULES = ("scene_patch_min", "ensemble_scene025_patch075")


def select_rules(rows: list[dict]) -> dict:
    validate_rows(rows)
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("V4.1 selection requires validation-only scores")
    if {row["label"] for row in rows} != {"positive", "negative"}:
        raise ValueError("Validation must contain both positive and negative labels")
    candidates = []
    for rule in RULES:
        feasible = []
        for threshold in threshold_candidates(method_score(row, rule) for row in rows):
            metrics = breakdown(rows, rule, threshold)
            accepted, recall, fpr = _selection_stats(metrics, max_fpr=0.0)
            if accepted:
                feasible.append(
                    {
                        "method": rule,
                        "threshold": threshold,
                        "validation": metrics,
                        "domain_macro_group_sensitivity": recall,
                        "domain_macro_group_false_positive_rate": fpr,
                    }
                )
        candidates.append(
            max(
                feasible,
                key=lambda item: (item["domain_macro_group_sensitivity"], item["threshold"]),
            )
        )
    return {
        "candidates": candidates,
        "selected": max(candidates, key=lambda item: item["domain_macro_group_sensitivity"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = []
    identities = None
    reference_meta = None
    for directory in args.validation_directories:
        rows = read_csv(directory / "scores.csv")
        current = sorted(tuple(row[key] for key in IDENTITY_FIELDS) for row in rows)
        meta = json.loads((directory / "scoring_config.json").read_text())
        if meta["splits"] != ["val"]:
            raise ValueError("Expected validation-only scoring provenance")
        if identities is not None and current != identities:
            raise ValueError("Different validation identities/labels/splits/groups")
        if reference_meta is not None:
            for key in (
                "manifest_sha256",
                "patch_model_sha256",
                "scene_size",
                "patch_size",
                "scene_pad_square",
                "patch_pad_square",
                "tile_size",
                "overlap",
            ):
                if meta[key] != reference_meta[key]:
                    raise ValueError(f"Incompatible scoring configuration: {key}")
        identities, reference_meta = current, meta
        variants.append(
            {
                "name": directory.name.removesuffix("_val"),
                "scoring": meta,
                "score_file": str((directory / "scores.csv").resolve()),
                "score_sha256": sha256(directory / "scores.csv"),
                "selection": select_rules(rows),
            }
        )
    winner = max(
        variants, key=lambda item: item["selection"]["selected"]["domain_macro_group_sensitivity"]
    )
    result = {
        "schema_version": 1,
        "threshold_source": "val",
        "max_validation_false_positive_rate": 0.0,
        "model_variant": winner["name"],
        "scoring": winner["scoring"],
        "selected": winner["selection"]["selected"],
        "variant_candidates": variants,
        "frozen_at": datetime.now(UTC).isoformat(),
        "selection_policy": (
            "Zero observed image/group FPR per domain with negatives; maximize equal-domain "
            "positive-group recall, highest threshold within rule, dual-evidence on rule tie, "
            "declared H1 then H2 model order on model tie. Validation only."
        ),
        "limitations": [
            "Zero validation FPR does not establish zero population FPR.",
            "UAV user-reported all-positive labels provide no UAV false-positive evaluation.",
            "Historical ground and partially reused UAV sets are regression development sets.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"variant": winner["name"], **result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
