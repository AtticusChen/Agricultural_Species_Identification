#!/usr/bin/env python3
"""Compare the frozen v4 candidate to deployed v3 without threshold retuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_models_v4 import SCORE_FIELDS, method_score, read_csv, sha256, validate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--candidate-scores", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    old_rows = read_csv(args.baseline_scores)
    new_rows = [row for path in args.candidate_scores for row in read_csv(path)]
    validate_rows(old_rows)
    validate_rows(new_rows)
    old_by_image = {row["image"]: row for row in old_rows}
    if set(old_by_image) != {row["image"] for row in new_rows}:
        raise ValueError("Baseline and candidate must cover identical images")
    rows = []
    for new in new_rows:
        old = old_by_image[new["image"]]
        for field in ("split", "domain", "label", "group_id"):
            if old[field] != new[field]:
                raise ValueError(f"Mismatched {field}: {new['image']}")
        old_score = method_score(old, baseline["method"])
        new_score = method_score(new, candidate["method"])
        expected = new["label"] == "positive"
        rows.append(
            {
                **{
                    key: new[key]
                    for key in ("image", "source", "split", "domain", "label", "group_id")
                },
                "old_score": old_score,
                "new_score": new_score,
                "old_present": old_score >= baseline["threshold"],
                "new_present": new_score >= candidate["threshold"],
                "old_correct": (old_score >= baseline["threshold"]) == expected,
                "new_correct": (new_score >= candidate["threshold"]) == expected,
            }
        )
    comparisons = {}
    for split in ("val", "test_internal", "test_external", "test_uav"):
        old = baseline["splits"][split]["overall"]
        new = candidate["splits"][split]["overall"]
        members = [row for row in rows if row["split"] == split]
        comparisons[split] = {
            "old": old,
            "new": new,
            "corrected_images": sum(
                not row["old_correct"] and row["new_correct"] for row in members
            ),
            "regressed_images": sum(
                row["old_correct"] and not row["new_correct"] for row in members
            ),
        }
    ground = comparisons["test_internal"]
    gates = {
        "ground_false_positive_count_not_increased": (
            ground["new"]["false_positive"] <= ground["old"]["false_positive"]
        ),
        "ground_true_positive_count_not_decreased": (
            ground["new"]["true_positive"] >= ground["old"]["true_positive"]
        ),
    }
    for split in ("test_external", "test_uav"):
        gates[f"{split}_true_positive_count_not_decreased"] = (
            comparisons[split]["new"]["true_positive"] >= comparisons[split]["old"]["true_positive"]
        )
    output = {
        "baseline_report_sha256": sha256(args.baseline),
        "candidate_report_sha256": sha256(args.candidate),
        "baseline_threshold": baseline["threshold"],
        "candidate_threshold": candidate["threshold"],
        "candidate_method": candidate["method"],
        "splits": comparisons,
        "deployment_gates": gates,
        "deployment_recommended_by_regression_gate": all(gates.values()),
        "limitations": [
            "Historical ground sets are regression development sets, not fresh blind tests.",
            "UAV reference labels are provisional assistant visual review, not expert truth.",
            "No confirmed whole-image UAV negatives: UAV FPR and balanced accuracy unknown.",
            "Regression gate does not establish field readiness or exact localization accuracy.",
            "No tuning of methods or thresholds on these heldout results is performed.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    with (args.output / "candidate-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(new_rows)
    with (args.output / "per-image-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "deployment_gates": gates,
                "splits": {
                    split: {
                        "old_tp_fp": [item["old"]["true_positive"], item["old"]["false_positive"]],
                        "new_tp_fp": [item["new"]["true_positive"], item["new"]["false_positive"]],
                    }
                    for split, item in comparisons.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
