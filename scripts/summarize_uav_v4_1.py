#!/usr/bin/env python3
"""Summarize fixed V4.1 UAV cohorts without selecting a model or threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from evaluate_models_v4 import (
    IDENTITY_FIELDS,
    Row,
    baseline_configuration,
    classification_metrics,
    method_score,
    read_csv,
    sha256,
    validate_rows,
)

EXPECTED_SCORE_ROWS = 350
EXPECTED_REVIEW_ROWS = 375
EXPECTED_GROUP_COUNTS = {
    "positive": {"uav-site-008": 8, "uav-site-011": 10, "uav-site-012": 6},
    "uncertain": {"uav-site-008": 18, "uav-site-009": 1},
}
COHORT_LABELS = {
    "original_positive24": "原暂定阳性24张",
    "original_uncertain19": "原不确定、现按用户报告阳性的19张",
    "all43": "全部43张",
}


def read_original_reviews(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "label", "split", "group_id"}
        if missing := required - set(reader.fieldnames or ()):
            raise ValueError(f"Missing original review columns: {sorted(missing)}")
        return list(reader)


def _review_subset(reviews: list[Row]) -> dict[str, Row]:
    if len(reviews) != EXPECTED_REVIEW_ROWS:
        raise ValueError(f"Expected 375 original reviews, got {len(reviews)}")
    indexed: dict[str, Row] = {}
    for row in reviews:
        for field in ("filename", "label", "split", "group_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Original review needs nonempty {field}")
        filename = row["filename"]
        if Path(filename).name != filename or "\\" in filename:
            raise ValueError(f"Original filename must be a basename: {filename}")
        if filename in indexed:
            raise ValueError(f"Duplicate original review filename: {filename}")
        if row["label"] not in {"positive", "negative", "uncertain"}:
            raise ValueError(f"Unknown original review label: {row['label']}")
        if row["split"] not in {"train", "val", "test_uav"}:
            raise ValueError(f"Unknown original UAV split: {row['split']}")
        indexed[filename] = row
    subset = {name: row for name, row in indexed.items() if row["split"] == "test_uav"}
    if len(subset) != 43:
        raise ValueError(f"Expected 43 original test_uav reviews, got {len(subset)}")
    for label, expected in EXPECTED_GROUP_COUNTS.items():
        observed = dict(
            Counter(row["group_id"] for row in subset.values() if row["label"] == label)
        )
        if observed != expected:
            raise ValueError(
                f"Original {label} cohort groups/counts changed: {observed}; expected {expected}"
            )
    return subset


def _score_subset(rows: list[Row], label: str) -> dict[str, Row]:
    validate_rows(rows)
    if len(rows) != EXPECTED_SCORE_ROWS:
        raise ValueError(f"Expected 350 {label} score rows, got {len(rows)}")
    selected = [row for row in rows if row["split"] == "test_uav"]
    if len(selected) != 43:
        raise ValueError(f"Expected 43 test_uav rows in {label} scores, got {len(selected)}")
    indexed: dict[str, Row] = {}
    for row in selected:
        if row["label"] != "positive" or row["domain"] != "uav":
            raise ValueError("All 43 test_uav score rows must be positive and domain=uav")
        filename = Path(row["image"]).name
        if filename in indexed:
            raise ValueError(f"Duplicate test_uav filename in {label} scores: {filename}")
        indexed[filename] = row
    return indexed


def _frozen_rule(selection: Row) -> Row:
    if selection.get("threshold_source") != "val":
        raise ValueError("A validation-only frozen selection is required")
    if not isinstance(selection.get("frozen_at"), str) or not selection["frozen_at"].strip():
        raise ValueError("V4.1 selection must contain a nonempty frozen_at marker")
    chosen = selection.get("selected")
    if not isinstance(chosen, dict) or not isinstance(chosen.get("method"), str):
        raise ValueError("Frozen selection needs selected.method and selected.threshold")
    try:
        threshold = float(chosen["threshold"])
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Invalid frozen threshold") from error
    if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError("Invalid frozen threshold")
    # Validate only the supplied rule; never enumerate or compare alternatives.
    method_score(
        {"image": "frozen-rule-validation", "scene": 0.0, "patch_second": 0.0}, chosen["method"]
    )
    return {"method": chosen["method"], "threshold": threshold}


def _metrics(rows: list[Row], rule: Row) -> Row:
    metrics = classification_metrics(rows, rule["method"], rule["threshold"])
    for level in (metrics, metrics["group_macro"]):
        if any(level[key] is not None for key in ("false_positive_rate", "balanced_accuracy")):
            raise ValueError("Positive-only UAV cohorts must have null FPR and balanced accuracy")
    return metrics


def build_report(
    old_rows: list[Row],
    new_rows: list[Row],
    original_reviews: list[Row],
    selection: Row,
) -> Row:
    """Validate identities and apply frozen rules to the specified historical cohorts."""
    original = _review_subset(original_reviews)
    old = _score_subset(old_rows, "old")
    new = _score_subset(new_rows, "new")
    old_identities = {row["image"]: tuple(row[key] for key in IDENTITY_FIELDS) for row in old_rows}
    new_identities = {row["image"]: tuple(row[key] for key in IDENTITY_FIELDS) for row in new_rows}
    if old_identities != new_identities:
        raise ValueError("Old/new score identities differ: image/split/domain/label/group_id")
    if set(old) != set(original) or set(new) != set(original):
        raise ValueError("UAV filenames differ from the original 43-row test_uav subset")
    for name, reviewed in original.items():
        for score in (old[name], new[name]):
            if any(score[key] != reviewed[key] for key in ("split", "group_id")):
                raise ValueError(f"Original subset split/group changed for {name}")
    old_rule = baseline_configuration()["selected"]
    if old_rule != {"method": "ensemble_scene025_patch075", "threshold": 0.88}:
        raise ValueError("The deployed old baseline rule unexpectedly changed")
    new_rule = _frozen_rule(selection)

    def summarize(names: list[str]) -> Row:
        return {
            "images": len(names),
            "groups": len({original[name]["group_id"] for name in names}),
            "filenames": names,
            "old": _metrics([old[name] for name in names], old_rule),
            "new": _metrics([new[name] for name in names], new_rule),
        }

    positive = sorted(name for name, row in original.items() if row["label"] == "positive")
    uncertain = sorted(name for name, row in original.items() if row["label"] == "uncertain")
    cohorts = {
        "original_positive24": summarize(positive),
        "original_uncertain19": summarize(uncertain),
        "all43": summarize(sorted(original)),
    }
    extra = {
        group: summarize([name for name in uncertain if original[name]["group_id"] == group])
        for group in EXPECTED_GROUP_COUNTS["uncertain"]
    }
    per_image = []
    for name in sorted(original):
        old_score = method_score(old[name], old_rule["method"])
        new_score = method_score(new[name], new_rule["method"])
        per_image.append(
            {
                "filename": name,
                "image": old[name]["image"],
                "split": "test_uav",
                "domain": "uav",
                "label": "positive",
                "original_label": original[name]["label"],
                "group_id": original[name]["group_id"],
                "old_score": old_score,
                "new_score": new_score,
                "old_detected": old_score >= old_rule["threshold"],
                "new_detected": new_score >= new_rule["threshold"],
            }
        )
    return {
        "schema_version": 1,
        "task": "uav_positive_cohort_presence_summary",
        "split": "test_uav",
        "model_selection_performed": False,
        "threshold_source": "val",
        "selection_frozen_at": selection["frozen_at"],
        "model_variant": selection.get("model_variant"),
        "old_rule": old_rule,
        "new_rule": new_rule,
        "validated_counts": {
            "old_score_rows": len(old_rows),
            "new_score_rows": len(new_rows),
            "original_review_rows": len(original_reviews),
            "test_uav_positive_rows": 43,
        },
        "cohorts": cohorts,
        "extra_by_group": extra,
        "per_image": per_image,
        "notes": [
            "Both score files have identical image/split/domain/label/group identities.",
            "Cohort membership uses original review filenames and preserves their split/group.",
            "Original positive24 labels were provisional visual reviews, not expert certification.",
            "Original uncertain19 are now evaluated as positive per user report, "
            "not expert certification.",
            "All cohorts are positive-only: image/group FPR and balanced accuracy are null.",
            "The two historical-label cohorts share site008; their group counts cannot be added.",
            "This is a descriptive regression breakdown, not an independent blind benchmark "
            "or model selection.",
        ],
    }


def render_markdown(report: Row) -> str:
    old_rule, new_rule = report["old_rule"], report["new_rule"]
    lines = [
        "# V4.1 UAV阳性子集汇总",
        "",
        f"旧规则：`{old_rule['method']}`，阈值 `{old_rule['threshold']:.12g}`。",
        f"新规则：`{new_rule['method']}`，冻结阈值 `{new_rule['threshold']:.12g}`。",
        "只应用冻结决策；没有重新选择模型或阈值。",
        "",
        "| 子集 | 图像数 | 拍摄组数 | 旧检出 | 新检出 | 旧召回率 | 新召回率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def table_row(label: str, item: Row) -> str:
        old, new = item["old"], item["new"]
        return (
            f"| {label} | {item['images']} | {item['groups']} | "
            f"{old['true_positive']} | {new['true_positive']} | "
            f"{old['sensitivity']:.1%} | {new['sensitivity']:.1%} |"
        )

    lines.extend(table_row(COHORT_LABELS[key], item) for key, item in report["cohorts"].items())
    lines.extend(
        [
            "",
            "## 原不确定19张的拍摄组明细",
            "",
            "| 拍摄组 | 图像数 | 拍摄组数 | 旧检出 | 新检出 | 旧召回率 | 新召回率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(table_row(group, item) for group, item in report["extra_by_group"].items())
    lines.extend(
        [
            "",
            "全部子集均无阴性样本，图像级及组级误报率（FPR）、平衡准确率（BA）为 `null`，"
            "不能据此判断误报风险。",
            "24/19描述旧版的初筛记录；本轮43张均按用户报告作为阳性，均非专家认证。",
            "两子集共享 site008，组数不可相加；全部43张是两子集的并集，不是额外独立试验。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-scores", type=Path, required=True)
    parser.add_argument("--new-scores", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--original-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        read_csv(args.old_scores),
        read_csv(args.new_scores),
        read_original_reviews(args.original_reviews),
        json.loads(args.selection.read_text(encoding="utf-8")),
    )
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["inputs"] = {
        key: {"path": str(path.resolve()), "sha256": sha256(path)}
        for key, path in (
            ("old_scores", args.old_scores),
            ("new_scores", args.new_scores),
            ("selection", args.selection),
            ("original_reviews", args.original_reviews),
        )
    }
    markdown = render_markdown(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **report["validated_counts"]}))


if __name__ == "__main__":
    main()
