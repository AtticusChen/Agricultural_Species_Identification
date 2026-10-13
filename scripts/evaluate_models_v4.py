#!/usr/bin/env python3
"""Score frozen image splits, then select and apply image-presence decisions.

Scoring and selection are separate operations. First score validation images with
--manifest ... --splits val. Then use --scores ... to save selection.json. Score
held-out images only after that file is frozen, and summarize them with
--scores ... --selection selection.json. These are image-classification metrics;
tile scores are coarse evidence and never establish detection mAP or mask quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from PIL import Image, ImageOps

SPLITS = ("train", "val", "test_internal", "test_external", "test_uav")
DOMAINS = ("ground", "uav")
LABELS = ("positive", "negative")
IDENTITY_FIELDS = ("image", "split", "domain", "label", "group_id")
PROVENANCE_FIELDS = ("provenance", "source")
SCORE_FIELDS = (
    *IDENTITY_FIELDS,
    *PROVENANCE_FIELDS,
    "scene",
    "patch_second",
    "patch_max",
    "tile_count",
)
PATCH_FIELDS = (*IDENTITY_FIELDS, *PROVENANCE_FIELDS, "left", "top", "right", "bottom", "score")
METHODS = {
    "scene_only": {"scene": 1.0},
    "ensemble_scene025_patch075": {"scene": 0.25, "patch_second": 0.75},
    "ensemble_scene050_patch050": {"scene": 0.50, "patch_second": 0.50},
}
PATCH_ONLY_METHOD = {"patch_second_only": {"patch_second": 1.0}}
Row = dict[str, Any]


def read_csv(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(IDENTITY_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return list(reader)


def validate_rows(rows: Sequence[Row], *, require_paths: bool = False) -> None:
    seen: set[str] = set()
    groups: dict[str, str] = {}
    for row in rows:
        if any(not str(row.get(field, "")).strip() for field in IDENTITY_FIELDS):
            raise ValueError("Every row needs nonempty image, split, domain, label and group_id")
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split: {row['split']}")
        if row["domain"] not in DOMAINS or row["label"] not in LABELS:
            raise ValueError(f"Unknown domain or label for {row['image']}")
        if row["image"] in seen:
            raise ValueError(f"Duplicate image row: {row['image']}")
        seen.add(row["image"])
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Group crosses splits: {row['group_id']}")
        if require_paths:
            path = Path(row["image"])
            if not path.is_absolute() or not path.is_file():
                raise ValueError(f"Image must be an existing absolute path: {path}")


def score_value(row: Row, field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid {field} score for {row['image']}") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Score outside [0, 1] for {row['image']}: {field}={value}")
    return value


def method_score(row: Row, method: str) -> float:
    if method == "scene_patch_min":
        # V4.1's separately registered dual-evidence rule. Do not add it to
        # METHODS: rerunning the historical V4 selector must remain unchanged.
        return min(score_value(row, "scene"), score_value(row, "patch_second"))
    weights = {**METHODS, **PATCH_ONLY_METHOD}.get(method)
    if weights is None:
        raise ValueError(f"Unknown fixed candidate method: {method}")
    return sum(weight * score_value(row, field) for field, weight in weights.items())


def available_methods(rows: Sequence[Row]) -> list[str]:
    if not rows:
        raise ValueError("No validation rows to select a model configuration")
    scene = all(row.get("scene") not in (None, "") for row in rows)
    patch = all(row.get("patch_second") not in (None, "") for row in rows)
    if scene and patch:
        return list(METHODS)
    if scene:
        return ["scene_only"]
    if patch:
        return list(PATCH_ONLY_METHOD)
    raise ValueError("Validation rows need consistent scene or patch scores")


def classification_metrics(rows: Sequence[Row], method: str, threshold: float) -> Row:
    """Report raw image counts and class-specific equal-weight group rates."""
    counts = {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0}
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        positive = row["label"] == "positive"
        predicted = method_score(row, method) >= threshold
        key = (
            ("true_positive" if predicted else "false_negative")
            if positive
            else ("false_positive" if predicted else "true_negative")
        )
        counts[key] += 1
        groups[row["label"]][row["group_id"]].append(predicted)
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["false_positive"] + counts["true_negative"]
    sensitivity = counts["true_positive"] / positives if positives else None
    fpr = counts["false_positive"] / negatives if negatives else None
    specificity = 1 - fpr if fpr is not None else None
    group_sensitivity = (
        fmean(fmean(values) for values in groups["positive"].values())
        if groups["positive"]
        else None
    )
    group_fpr = (
        fmean(fmean(values) for values in groups["negative"].values())
        if groups["negative"]
        else None
    )
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    return {
        "images": len(rows),
        "positive_images": positives,
        "negative_images": negatives,
        "groups": len({row["group_id"] for row in rows}),
        "label_provenance_counts": dict(
            Counter(row.get("provenance", "unspecified") for row in rows)
        ),
        **counts,
        "sensitivity": sensitivity,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
        "precision": counts["true_positive"] / predicted_positive if predicted_positive else None,
        "group_macro": {
            "positive_groups": len(groups["positive"]),
            "negative_groups": len(groups["negative"]),
            "sensitivity": group_sensitivity,
            "false_positive_rate": group_fpr,
            "specificity": 1 - group_fpr if group_fpr is not None else None,
            "balanced_accuracy": (
                (group_sensitivity + 1 - group_fpr) / 2
                if group_sensitivity is not None and group_fpr is not None
                else None
            ),
        },
    }


def breakdown(rows: Sequence[Row], method: str, threshold: float) -> Row:
    return {
        "overall": classification_metrics(rows, method, threshold),
        "by_domain": {
            domain: classification_metrics(
                [row for row in rows if row["domain"] == domain], method, threshold
            )
            for domain in DOMAINS
        },
    }


def threshold_candidates(scores: Iterable[float]) -> list[float]:
    unique = sorted(set(scores))
    # The nextafter endpoint permits all-negative decisions even when a model
    # emits exactly 1.0; equality is positive throughout training and deployment.
    return sorted(
        {
            0.0,
            1.0,
            math.nextafter(1.0, math.inf),
            *[(left + right) / 2 for left, right in zip(unique, unique[1:], strict=False)],
        }
    )


def _selection_stats(metrics: Row, max_fpr: float) -> tuple[bool, float, float]:
    recalls: list[float] = []
    group_fprs: list[float] = []
    for item in metrics["by_domain"].values():
        group = item["group_macro"]
        if group["sensitivity"] is not None:
            recalls.append(group["sensitivity"])
        if item["false_positive_rate"] is not None:
            if item["false_positive_rate"] > max_fpr + 1e-12:
                return False, 0.0, 1.0
            if group["false_positive_rate"] > max_fpr + 1e-12:
                return False, 0.0, 1.0
            group_fprs.append(group["false_positive_rate"])
    return True, fmean(recalls), fmean(group_fprs)


def select_configuration(rows: Sequence[Row], max_fpr: float = 0.10) -> Row:
    """Choose a fixed rule and threshold using validation rows exclusively."""
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    validate_rows(rows)
    validation = [row for row in rows if row["split"] == "val"]
    if {row["label"] for row in validation} != set(LABELS):
        raise ValueError("Threshold selection requires validation positives and negatives")
    candidates: list[Row] = []
    for method in available_methods(validation):
        best: Row | None = None
        best_key: tuple[float, float, float] | None = None
        for threshold in threshold_candidates(method_score(row, method) for row in validation):
            metrics = breakdown(validation, method, threshold)
            feasible, recall, fpr = _selection_stats(metrics, max_fpr)
            key = (recall, -fpr, threshold)
            if feasible and (best_key is None or key > best_key):
                best_key = key
                best = {
                    "method": method,
                    "threshold": threshold,
                    "validation": metrics,
                    "domain_macro_group_sensitivity": recall,
                    "domain_macro_group_false_positive_rate": fpr,
                }
        if best is None:
            raise ValueError(f"No feasible threshold for {method}")
        candidates.append(best)
    # Deterministic ties retain the simpler rule declared first in METHODS.
    selected = max(
        candidates,
        key=lambda item: (
            item["domain_macro_group_sensitivity"],
            -item["domain_macro_group_false_positive_rate"],
        ),
    )
    missing_class_domains = {
        domain: [
            label
            for label in LABELS
            if not any(row["domain"] == domain and row["label"] == label for row in validation)
        ]
        for domain in sorted({row["domain"] for row in validation})
    }
    return {
        "schema_version": 1,
        "threshold_source": "val",
        "max_validation_false_positive_rate": max_fpr,
        "selection_objective": (
            "Maximize equal-domain, equal-positive-group sensitivity subject to both image "
            "and negative-group FPR <= cap within every domain with validation negatives; "
            "ties prefer lower domain/group FPR, then a higher threshold, then simpler rule."
        ),
        "validation_missing_classes": {
            domain: labels for domain, labels in missing_class_domains.items() if labels
        },
        "candidates": candidates,
        "selected": selected,
        "limitations": [
            "Validation FPR is an observed constraint, not a population error-rate guarantee.",
            "A domain without negatives provides recall only; its FPR is unvalidated.",
            "Image presence decisions do not measure bounding-box or segmentation accuracy.",
        ],
    }


def summarize_scores(rows: Sequence[Row], selection: Row) -> Row:
    """Apply a previously fixed configuration without selecting any threshold."""
    validate_rows(rows)
    if selection.get("threshold_source") != "val":
        raise ValueError("A validation-only selection file is required")
    chosen = selection["selected"]
    method = chosen["method"]
    threshold = float(chosen["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError("Invalid frozen threshold")
    return {
        "task": "image_presence_classification",
        "method": method,
        "threshold": threshold,
        "threshold_source": "val",
        "splits": {
            split: breakdown([row for row in rows if row["split"] == split], method, threshold)
            for split in SPLITS
            if split != "train" and any(row["split"] == split for row in rows)
        },
        "notes": [
            "Raw per-image decision scores determine presence, independently of returned regions.",
            "Patch probabilities describe coarse support, not exact object boxes or masks.",
            "Existing v3 sets are development regression checks, not blind tests.",
            "UAV assistant/user-reported labels are not expert certification.",
        ],
    }


def baseline_configuration() -> Row:
    """The deployed v3 rule is a frozen control, never recalibrated on v4 scores."""
    return {
        "schema_version": 1,
        "threshold_source": "val",
        "selection_policy": "Frozen deployed v3 configuration selected on the original v3 val set",
        "selected": {"method": "ensemble_scene025_patch075", "threshold": 0.88},
    }


def pad_square(image: Image.Image) -> Image.Image:
    # Match build_joint_dataset_v4.py exactly (Pillow's default BICUBIC resize).
    return ImageOps.pad(image, (768, 768), color=(114, 114, 114))


def tile_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    def starts(length: int) -> list[int]:
        if length <= 1280:
            return [0]
        values = list(range(0, length - 1280 + 1, 1024))
        if values[-1] != length - 1280:
            values.append(length - 1280)
        return values

    return [
        (left, top, min(width, left + 1280), min(height, top + 1280))
        for top in starts(height)
        for left in starts(width)
    ]


def mikania_probability(result: Any) -> float:
    class_ids = [key for key, name in result.names.items() if name == "mikania"]
    if len(class_ids) != 1 or result.probs is None:
        raise ValueError("Expected a classification model with exactly one mikania class")
    probability = float(result.probs.data[class_ids[0]].item())
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability")
    return probability


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path: Path) -> tuple[Image.Image, str | None]:
    """Only repair a missing JPEG end marker in memory; never rewrite a source."""
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB"), None
    except OSError:
        data = path.read_bytes()
        if not data.startswith(b"\xff\xd8") or data.endswith(b"\xff\xd9"):
            raise
        with Image.open(io.BytesIO(data + b"\xff\xd9")) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        return image, "jpeg_eoi_appended_in_memory"


def resume_images(output: Path, metadata: Row, selected_rows: Sequence[Row]) -> set[str]:
    previous = json.loads((output / "scoring_config.json").read_text())
    for key, value in metadata.items():
        if key != "decode_policy" and previous.get(key) != value:
            raise ValueError(f"Cannot resume with changed scoring configuration: {key}")
    completed = read_csv(output / "scores.csv")
    validate_rows(completed)
    expected = {row["image"]: row for row in selected_rows}
    for row in completed:
        if row["image"] not in expected or any(
            row[field] != expected[row["image"]][field] for field in IDENTITY_FIELDS
        ):
            raise ValueError("Completed scores do not match the requested manifest rows")
        for field in ("scene", "patch_second", "patch_max"):
            if row.get(field) not in ("", None):
                score_value(row, field)
    completed_images = {row["image"] for row in completed}
    patch_rows = read_csv(output / "patch_scores.csv")
    if any(row["image"] not in completed_images for row in patch_rows):
        raise ValueError("Incomplete patch records require inspection before resuming")
    patch_counts = Counter(row["image"] for row in patch_rows)
    if any(patch_counts[row["image"]] != int(row["tile_count"]) for row in completed):
        raise ValueError("Completed patch counts differ from per-image tile counts")
    return completed_images


def score_manifest(args: argparse.Namespace, output: Path) -> None:
    rows = read_csv(args.manifest)
    validate_rows(rows)
    selected_rows = [row for row in rows if row["split"] in args.splits]
    if not selected_rows:
        raise ValueError("No images match the requested splits")
    validate_rows(selected_rows, require_paths=True)
    from ultralytics import YOLO

    scene_model = YOLO(str(args.scene_model)) if args.scene_model else None
    patch_model = YOLO(str(args.patch_model)) if args.patch_model else None
    metadata = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "splits": args.splits,
        "scene_model": str(args.scene_model.resolve()) if args.scene_model else None,
        "scene_model_sha256": sha256(args.scene_model) if args.scene_model else None,
        "patch_model": str(args.patch_model.resolve()) if args.patch_model else None,
        "patch_model_sha256": sha256(args.patch_model) if args.patch_model else None,
        "scene_size": args.scene_size,
        "patch_size": args.patch_size,
        "scene_pad_square": args.scene_pad_square,
        "patch_pad_square": args.patch_pad_square,
        "scene_pad_size": 768 if args.scene_pad_square else None,
        "tile_size": 1280,
        "overlap": 0.20,
        "device": args.device,
        "presence_score_independent_of_region_display": True,
        "decode_policy": "strict Pillow; missing JPEG EOI retry in memory, separately recorded",
    }
    completed_images = resume_images(output, metadata, selected_rows) if args.resume else set()
    (output / "scoring_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    file_mode = "a" if args.resume else "w"
    with (
        (output / "scores.csv").open(file_mode, encoding="utf-8", newline="") as image_handle,
        (output / "patch_scores.csv").open(file_mode, encoding="utf-8", newline="") as patch_handle,
    ):
        image_writer = csv.DictWriter(image_handle, fieldnames=SCORE_FIELDS)
        patch_writer = csv.DictWriter(patch_handle, fieldnames=PATCH_FIELDS)
        if not args.resume:
            image_writer.writeheader()
            patch_writer.writeheader()
        for index, row in enumerate(selected_rows, 1):
            if row["image"] in completed_images:
                continue
            identity = {key: row[key] for key in IDENTITY_FIELDS}
            identity.update({key: row.get(key, "") for key in PROVENANCE_FIELDS})
            scores: Row = {
                **identity,
                "scene": None,
                "patch_second": None,
                "patch_max": None,
                "tile_count": 0,
            }
            image, decode_action = load_image(Path(row["image"]))
            if decode_action:
                repair_path = output / "image_decode_repairs.csv"
                new_file = not repair_path.exists()
                with repair_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("image", "source_sha256", "action"))
                    if new_file:
                        writer.writeheader()
                    writer.writerow(
                        {
                            "image": row["image"],
                            "source_sha256": sha256(Path(row["image"])),
                            "action": decode_action,
                        }
                    )
            if scene_model is not None:
                scene_image = pad_square(image) if args.scene_pad_square else image
                result = scene_model.predict(
                    source=scene_image, imgsz=args.scene_size, device=args.device, verbose=False
                )[0]
                scores["scene"] = mikania_probability(result)
            if patch_model is not None:
                boxes = tile_boxes(image.width, image.height)
                crops = [image.crop(box) for box in boxes]
                if args.patch_pad_square:
                    crops = [pad_square(crop) for crop in crops]
                results = patch_model.predict(
                    source=crops,
                    imgsz=args.patch_size,
                    device=args.device,
                    batch=32,
                    verbose=False,
                )
                probabilities = [mikania_probability(result) for result in results]
                if len(probabilities) != len(boxes):
                    raise ValueError("Patch model returned a different number of predictions")
                ordered = sorted(probabilities, reverse=True)
                scores.update(
                    patch_second=ordered[min(1, len(ordered) - 1)],
                    patch_max=ordered[0],
                    tile_count=len(boxes),
                )
                for box, probability in zip(boxes, probabilities, strict=True):
                    patch_writer.writerow(
                        {
                            **identity,
                            "left": box[0],
                            "top": box[1],
                            "right": box[2],
                            "bottom": box[3],
                            "score": probability,
                        }
                    )
            image_writer.writerow(scores)
            image_handle.flush()
            patch_handle.flush()
            if index % 25 == 0 or index == len(selected_rows):
                print(f"Scored {index}/{len(selected_rows)} images", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--scores", nargs="+", type=Path)
    parser.add_argument("--scene-model", type=Path)
    parser.add_argument("--patch-model", type=Path)
    parser.add_argument("--scene-pad-square", action="store_true")
    parser.add_argument("--patch-pad-square", action="store_true")
    parser.add_argument("--scene-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS[1:], default=["val"])
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--selection", type=Path, help="Apply this frozen validation selection")
    decision.add_argument("--baseline-v3", action="store_true", help="Apply deployed v3 at 0.88")
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument(
        "--resume", action="store_true", help="Resume matching incomplete score files"
    )
    args = parser.parse_args()
    if args.manifest and not (args.scene_model or args.patch_model):
        parser.error("Scoring requires --scene-model and/or --patch-model")
    if args.manifest and (args.selection or args.baseline_v3):
        parser.error("Frozen decisions apply to --scores, not model scoring")
    if args.scene_size < 1 or args.patch_size < 1:
        parser.error("Image sizes must be positive")
    if args.resume and not args.manifest:
        parser.error("--resume applies only to model scoring")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.manifest:
        score_manifest(args, output)
        return
    rows = [row for path in args.scores for row in read_csv(path)]
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
    elif args.baseline_v3:
        selection = baseline_configuration()
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    else:
        if any(row["split"] != "val" for row in rows):
            raise ValueError(
                "Freeze selection using validation-only scores before reading held-out scores"
            )
        selection = select_configuration(rows, args.max_fpr)
        (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    report = summarize_scores(rows, selection)
    report["score_files"] = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in args.scores
    ]
    if args.selection:
        report["selection_file"] = str(args.selection.resolve())
        report["selection_sha256"] = sha256(args.selection)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "method": report["method"],
                "threshold": report["threshold"],
                "splits": report["splits"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
