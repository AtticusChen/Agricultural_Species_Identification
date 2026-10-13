"""Export the two frozen classifiers on the research server, without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(args.project / ".ultralytics")
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import DEFAULT_MEAN, DEFAULT_STD, classify_transforms

    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "config": config,
        "torch": torch.__version__,
        "mean": DEFAULT_MEAN,
        "std": DEFAULT_STD,
        "models": {},
    }
    if tuple(DEFAULT_MEAN) != (0, 0, 0) or tuple(DEFAULT_STD) != (1, 1, 1):
        raise RuntimeError("Unexpected normalization; portable preprocessing must be reviewed")
    for branch in ("scene", "patch"):
        source = args.project / config[f"{branch}_source"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != config[f"{branch}_pt_sha256"]:
            raise RuntimeError(f"Unrecognized frozen model {source}")
        copied = args.output / f"{branch}.pt"
        shutil.copy2(source, copied)
        model = YOLO(str(copied))
        size = config[f"{branch}_image_size"]
        if set(model.names.values()) != {"mikania", "other"}:
            # Save the real names rather than assuming an index or accepting an unknown taxonomy.
            print(f"Class names: {model.names}")
        class_id = next((i for i, name in model.names.items() if name == "mikania"), None)
        if class_id is None:
            raise RuntimeError("Model has no mikania class")
        exported = Path(
            model.export(
                format="onnx",
                imgsz=size,
                batch=1,
                dynamic=False,
                simplify=False,
                opset=17,
                device="cpu",
            )
        )
        report["models"][branch] = {
            "file": exported.name,
            "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
            "class_names": model.names,
            "mikania_index": class_id,
            "image_size": size,
            "transforms": repr(classify_transforms(size=size)),
        }
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
