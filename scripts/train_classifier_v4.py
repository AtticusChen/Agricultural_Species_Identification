#!/usr/bin/env python3
"""Run one recorded classification experiment in the existing remote environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    # Bind before torch initializes CUDA; otherwise parallel runs can all land on GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False, "wandb": False, "mlflow": False})
    run = args.output.resolve() / args.name
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    record = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "initial_weights_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "physical_gpu": args.device,
        "checkpoint_selection": "validation fitness only; test unused during training",
    }
    record_path = run / "run.json"

    def save():
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    save()
    try:
        model = YOLO(str(args.model.resolve()))
        result = model.train(
            data=str(args.data.resolve()),
            project=str(args.output.resolve()),
            name=args.name,
            exist_ok=True,
            device=args.device,
            seed=args.seed,
            imgsz=args.size,
            epochs=args.epochs,
            patience=20,
            batch=args.batch,
            workers=8,
            optimizer="AdamW",
            lr0=args.lr,
            lrf=0.05,
            weight_decay=0.001,
            warmup_epochs=3,
            cos_lr=True,
            dropout=0.15,
            scale=0.05,
            fliplr=0.5,
            flipud=0.25,
            auto_augment=None,
            hsv_h=0.01,
            hsv_s=0.25,
            hsv_v=0.25,
            erasing=0.0,
            cache=False,
            plots=True,
            save=True,
            deterministic=True,
        )
        record.update(
            status="completed",
            best=str(model.trainer.best),
            metrics={k: float(v) for k, v in result.results_dict.items()},
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        save()


if __name__ == "__main__":
    main()
