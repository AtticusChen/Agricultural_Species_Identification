"""Compare the deployed API to frozen historical V4.1 scores, not a new benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    threshold = 0.03896419005468488
    with args.scores.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Every negative, all model misses, plus positive examples in every split.
    selected = (
        rows
        if args.all
        else [
            row
            for row in rows
            if row["label"] == "negative"
            or min(float(row["scene"]), float(row["patch_second"])) < threshold
        ]
    )
    if not args.all:
        for split in ("test_internal", "test_external", "test_uav"):
            selected.extend([r for r in rows if r["split"] == split and r not in selected][:2])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url + "/api/v1/health", timeout=10) as response:
        health = json.load(response)
    if health["model"] != "mikania-ensemble-v4.1" or health["default_threshold"] != threshold:
        raise RuntimeError(f"Incorrect active model: {health}")
    results = []
    for row in selected:
        path = Path(row["image"])
        image = path.read_bytes()
        # Match the previously recorded in-memory repair without editing source data.
        if image.startswith(b"\xff\xd8") and not image.endswith(b"\xff\xd9"):
            image += b"\xff\xd9"
        boundary = "mikania-v41-regression"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="sample.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode()
        body += image + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            args.url + "/api/v1/predict",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with opener.open(request, timeout=180) as response:
            payload = json.load(response)
        expected = min(float(row["scene"]), float(row["patch_second"]))
        difference = abs(payload["score"] - expected)
        passed = (
            difference < 0.0001
            and payload["present"] == (expected >= threshold)
            and payload["threshold"] == threshold
            and payload["model"] == health["model"]
        )
        record = {
            "source": row["source"],
            "split": row["split"],
            "label": row["label"],
            "expected_score": expected,
            "actual_score": payload["score"],
            "difference": difference,
            "present": payload["present"],
            "regions": payload["count"],
            "passed": passed,
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report = {
        "status": "passed" if all(r["passed"] for r in results) else "failed",
        "scope": "Frozen-score deployment regression, not independent accuracy evaluation",
        "health": health,
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["status"] != "passed":
        raise SystemExit("API does not match the frozen candidate")


if __name__ == "__main__":
    main()
