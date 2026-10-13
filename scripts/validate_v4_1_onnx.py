"""Record CPU ONNX parity and replay frozen evaluation rows without retuning."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from mikania_app.onnx_classifier import OnnxEnsembleClassifier
from mikania_app.predictor import UltralyticsEnsembleClassifier, decode_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--scores", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.scores:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    threshold = 0.03896419005468488
    onnx = OnnxEnsembleClassifier(args.models / "scene.onnx", args.models / "patch.onnx")
    pt = UltralyticsEnsembleClassifier(
        args.models / "scene.pt",
        args.models / "patch.pt",
        device="cpu",
        scene_image_size=512,
        patch_image_size=320,
        scene_pad_square=True,
        patch_pad_square=True,
        decision_method="scene_patch_min",
    )
    import torch

    torch.set_num_threads(2)
    checks = []
    counts = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "count": 0})
    parity_groups = set()
    for index, row in enumerate(rows):
        image = decode_image(Path(row["image"]).read_bytes())
        started = perf_counter()
        result = onnx.predict_result(image, threshold)
        record = {
            "source": row["source"],
            "split": row["split"],
            "label": row["label"],
            "historical_gpu_score": min(float(row["scene"]), float(row["patch_second"])),
            "onnx_cpu_score": result.score,
            "present": result.present,
            "regions": len(result.detections),
            "seconds": perf_counter() - started,
        }
        record["same_historical_decision"] = result.present == (
            record["historical_gpu_score"] >= threshold
        )
        record["historical_score_difference"] = abs(record["historical_gpu_score"] - result.score)
        # Cover each split/label and every historical disagreement with a CPU PT comparison.
        group = (row["split"], row["label"])
        if group not in parity_groups or not record["same_historical_decision"]:
            expected = pt.predict_result(image, threshold)
            error = abs(result.score - expected.score)
            same_boxes = len(result.detections) == len(expected.detections)
            region_error = 0.0
            if same_boxes:
                for a, b in zip(result.detections, expected.detections, strict=True):
                    same_boxes = same_boxes and a.box_xyxy == b.box_xyxy
                    region_error = max(region_error, abs(a.confidence - b.confidence))
            record["cpu_pt_comparison"] = {
                "score": expected.score,
                "score_error": error,
                "region_score_error": region_error,
                "passed": error < 1e-4
                and region_error < 1e-4
                and same_boxes
                and expected.present == result.present,
            }
            parity_groups.add(group)
        positive = row["label"] == "positive"
        counts[row["split"]][
            ("tp" if result.present else "fn") if positive else ("fp" if result.present else "tn")
        ] += 1
        counts[row["split"]]["count"] += 1
        checks.append(record)
        if (index + 1) % 25 == 0:
            print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    parity_ok = all(r.get("cpu_pt_comparison", {"passed": True})["passed"] for r in checks)
    report = {
        "status": "passed" if parity_ok else "failed",
        "threshold": threshold,
        "scope": "Frozen-data numerical replay; not new blind accuracy evaluation",
        "cpu_pt_onnx_parity": parity_ok,
        "counts": dict(counts),
        "historical_decision_disagreements": sum(not r["same_historical_decision"] for r in checks),
        "max_gpu_cpu_score_difference": max(r["historical_score_difference"] for r in checks),
        "samples": checks,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}))
    if not parity_ok:
        raise SystemExit("ONNX does not match CPU PT")


if __name__ == "__main__":
    main()
