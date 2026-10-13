#!/usr/bin/env python3
"""Create a versioned PT/ONNX model release with verified metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--evaluation-model-key", required=True)
    parser.add_argument("--fast-threshold", type=float)
    parser.add_argument("--precise-threshold", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_metrics(
    model: dict[str, Any], mode: str, threshold_override: float | None = None
) -> dict[str, float]:
    if mode == "precise":
        selected = float(model["selected_tiled_threshold"])
        prefix = "tiled_image_level"
    else:
        selected = float(model["selected_threshold"])
        prefix = "image_level"
    threshold = selected if threshold_override is None else threshold_override
    key = f"{threshold:.2f}"
    if key not in model[f"{prefix}_sensitivity"]:
        raise ValueError(f"Threshold {key} was not evaluated for {mode} mode")
    return {
        "threshold": threshold,
        "sensitivity": float(model[f"{prefix}_sensitivity"][key]),
        "false_positive_rate": float(
            model[f"{prefix}_false_positive_rate"][key]
        ),
        "balanced_accuracy": float(model[f"{prefix}_balanced_accuracy"][key]),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    for required in (args.pt, args.onnx, args.evaluation_report):
        if not required.is_file():
            raise FileNotFoundError(required)

    report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    try:
        candidate = report["models"][args.evaluation_model_key]
    except KeyError as error:
        raise KeyError(
            f"Evaluation model key not found: {args.evaluation_model_key}"
        ) from error

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(args.onnx.resolve()), providers=["CPUExecutionProvider"]
    )
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    if (
        len(input_shape) != 4
        or input_shape[0] != 1
        or input_shape[1] != 3
        or input_shape[2] != input_shape[3]
        or not isinstance(input_shape[2], int)
    ):
        raise ValueError(f"Unsupported ONNX input shape: {input_shape}")
    if output_shape != [1, 300, 6]:
        raise ValueError(f"Unsupported ONNX output shape: {output_shape}")

    output.mkdir(parents=True)
    packaged_pt = output / "model.pt"
    packaged_onnx = output / "model.onnx"
    shutil.copy2(args.pt, packaged_pt)
    shutil.copy2(args.onnx, packaged_onnx)
    fast_metrics = selected_metrics(candidate, "fast", args.fast_threshold)
    precise_metrics = selected_metrics(
        candidate, "precise", args.precise_threshold
    )
    metadata = {
        "model_id": args.model_id,
        "model_file": "model.onnx",
        "model_sha256": sha256(packaged_onnx),
        "pt_file": "model.pt",
        "pt_sha256": sha256(packaged_pt),
        "source_experiment": args.source_experiment,
        "training_dataset": args.training_dataset,
        "evaluation_dataset": Path(report["dataset"]).name,
        "evaluation_model_key": args.evaluation_model_key,
        "threshold_selection_split": candidate["threshold_selection"]["source_split"],
        "threshold_selection_method": (
            "validation metrics plus five-image regression visual review"
        ),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "class_names": ["mikania"],
        "default_scan_mode": "precise",
        "default_thresholds": {
            "fast": fast_metrics["threshold"],
            "precise": precise_metrics["threshold"],
        },
        "test_internal": report["test_internal"],
        "image_level_metrics": {
            "fast": fast_metrics,
            "precise": precise_metrics,
        },
        "box_metrics": candidate["box_metrics"],
        "regression": {
            stem: {
                "expected": result["expected"],
                "fast_count": sum(
                    detection["confidence"] >= fast_metrics["threshold"]
                    for detection in result["full"]["detections"]
                ),
                "precise_count": sum(
                    detection["confidence"] >= precise_metrics["threshold"]
                    for detection in result["tiled"]["detections"]
                ),
            }
            for stem, result in candidate["regression"].items()
        },
        "runtime": "ONNX Runtime CPU",
        "license": (
            "Ultralytics YOLO / AGPL-3.0; review licensing before proprietary "
            "or commercial distribution"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    positive_count = metadata["test_internal"]["positive_images"]
    negative_count = metadata["test_internal"]["negative_images"]
    card = (
        f"# 薇甘菊检测模型 {args.model_id}\n\n"
        "该模型用于地面 RGB 照片中的薇甘菊单类目标检测。网页默认采用 1280 像素"
        "重叠局部块进行精细扫描，再合并重复标框；快速模式只扫描整图一次。\n\n"
        "## 来源\n\n"
        f"- 训练运行：`{args.source_experiment}`\n"
        f"- 训练数据：`{args.training_dataset}`\n"
        f"- 独立评估数据：`{metadata['evaluation_dataset']}`\n"
        f"- 输入：`{input_shape[2]} × {input_shape[3]}` letterbox\n\n"
        "## 照片级独立测试结果\n\n"
        f"阈值在验证集确定，以下结果来自 {positive_count} 张正样本和 "
        f"{negative_count} 张非薇甘菊内部测试图片。\n\n"
        f"- 精细扫描：阈值 `{precise_metrics['threshold']:.2f}`，灵敏度 "
        f"`{precise_metrics['sensitivity']:.3f}`，误报率 "
        f"`{precise_metrics['false_positive_rate']:.3f}`，平衡准确率 "
        f"`{precise_metrics['balanced_accuracy']:.3f}`。\n"
        f"- 快速扫描：阈值 `{fast_metrics['threshold']:.2f}`，灵敏度 "
        f"`{fast_metrics['sensitivity']:.3f}`，误报率 "
        f"`{fast_metrics['false_positive_rate']:.3f}`，平衡准确率 "
        f"`{fast_metrics['balanced_accuracy']:.3f}`。\n\n"
        "## 使用限制\n\n"
        "- 结果用于辅助筛查，不代替植物分类专家现场鉴定；\n"
        "- “未识别到”只表示没有标框超过当前阈值；\n"
        "- 检测框代表疑似覆盖区域，不代表独立植株数量；\n"
        "- 当前非薇甘菊测试集规模仍有限，新增地区、季节和相似物种后应继续"
        "回归验证。\n\n"
        "## 许可证\n\n"
        "模型由 Ultralytics YOLO 训练和导出。闭源、商业服务、硬件嵌入或其他"
        "专有使用前，应确认并满足适用的 AGPL-3.0 或商业许可要求。\n"
    )
    (output / "model_card.md").write_text(card, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
