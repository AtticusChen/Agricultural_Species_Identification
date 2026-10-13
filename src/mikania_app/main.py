from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .predictor import (
    ENSEMBLE_DECISION_METHODS,
    Detector,
    InvalidImageError,
    OnnxDetector,
    UltralyticsDetector,
    UltralyticsEnsembleClassifier,
    UltralyticsPatchClassifier,
    decode_image,
    render_annotated_image,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_THRESHOLD = 0.15
DEFAULT_SCAN_MODE = "precise"
WEB_EXAMPLES = (
    ("ground", "地面实景", "地面实景示例.jpg"),
    ("aerial", "无人机实景", "无人机实景示例.jpg"),
    ("other", "其他植物", "其他植物示例.jpg"),
)


def _default_threshold(environment_name: str) -> float:
    raw_value = os.getenv(
        environment_name,
        os.getenv("MIKANIA_DEFAULT_THRESHOLD", str(DEFAULT_THRESHOLD)),
    )
    try:
        threshold = float(raw_value)
    except ValueError:
        return DEFAULT_THRESHOLD
    return threshold if 0 <= threshold <= 1 else DEFAULT_THRESHOLD


def _default_scan_mode() -> str:
    value = os.getenv("MIKANIA_DEFAULT_SCAN_MODE", DEFAULT_SCAN_MODE)
    return value if value in {"precise", "fast"} else DEFAULT_SCAN_MODE


def _default_detector() -> Detector:
    model_path = Path(os.getenv("MIKANIA_MODEL_PATH", "models/best.pt"))
    device_value = os.getenv("MIKANIA_DEVICE")
    device: str | int | None = None
    if device_value:
        device = int(device_value) if device_value.isdigit() else device_value
    model_task = os.getenv("MIKANIA_MODEL_TASK", "detect").strip().lower()
    if model_task == "ensemble" and model_path.suffix.lower() == ".onnx":
        from .onnx_classifier import OnnxEnsembleClassifier

        if (
            os.getenv("MIKANIA_DECISION_METHOD") != "scene_patch_min"
            or os.getenv("MIKANIA_SCENE_PAD_SQUARE") != "1"
            or os.getenv("MIKANIA_PATCH_PAD_SQUARE") != "1"
        ):
            raise ValueError("ONNX组合分类仅支持已冻结的V4.1双证据补方形配置")
        return OnnxEnsembleClassifier(
            Path(os.environ["MIKANIA_SCENE_MODEL_PATH"]),
            model_path,
            scene_image_size=int(os.getenv("MIKANIA_SCENE_IMAGE_SIZE", "512")),
            patch_image_size=int(os.getenv("MIKANIA_CLASSIFIER_IMAGE_SIZE", "320")),
            tile_size=int(os.getenv("MIKANIA_CLASSIFIER_TILE_SIZE", "1280")),
            overlap=float(os.getenv("MIKANIA_CLASSIFIER_TILE_OVERLAP", "0.2")),
            pad_size=int(os.getenv("MIKANIA_PAD_SIZE", "768")),
            region_threshold=float(os.getenv("MIKANIA_REGION_THRESHOLD", "0.37")),
            model_id=os.getenv("MIKANIA_MODEL_ID", "mikania-ensemble-v4.1"),
        )
    if model_task in {"classify", "ensemble"}:
        if model_path.suffix.lower() == ".onnx":
            raise ValueError("分类模型当前需要使用 PT 格式")
        image_size = int(os.getenv("MIKANIA_CLASSIFIER_IMAGE_SIZE", "384"))
        tile_size = int(os.getenv("MIKANIA_CLASSIFIER_TILE_SIZE", "1280"))
        overlap = float(os.getenv("MIKANIA_CLASSIFIER_TILE_OVERLAP", "0.20"))
        pad_size = int(os.getenv("MIKANIA_PAD_SIZE", "768"))
        patch_pad = os.getenv("MIKANIA_PATCH_PAD_SQUARE", "0").lower() in {"1", "true", "yes"}
        if model_task == "ensemble":
            decision_method = os.getenv("MIKANIA_DECISION_METHOD", "weighted_sum").strip().lower()
            if decision_method not in ENSEMBLE_DECISION_METHODS:
                raise ValueError(f"不支持的图像判定规则：{decision_method}")
            scene_path = Path(os.environ["MIKANIA_SCENE_MODEL_PATH"])
            scene_image_size = int(os.getenv("MIKANIA_SCENE_IMAGE_SIZE", "384"))
            scene_weight = float(os.getenv("MIKANIA_SCENE_WEIGHT", "0.25"))
            region_threshold = float(os.getenv("MIKANIA_REGION_THRESHOLD", "0.37"))
            return UltralyticsEnsembleClassifier(
                scene_path,
                model_path,
                device=device,
                scene_image_size=scene_image_size,
                patch_image_size=image_size,
                tile_size=tile_size,
                overlap=overlap,
                scene_weight=scene_weight,
                region_threshold=region_threshold,
                model_id=os.getenv("MIKANIA_MODEL_ID", "mikania-ensemble-v3"),
                scene_pad_square=os.getenv("MIKANIA_SCENE_PAD_SQUARE", "0").lower()
                in {"1", "true", "yes"},
                patch_pad_square=patch_pad,
                pad_size=pad_size,
                decision_method=decision_method,
            )
        return UltralyticsPatchClassifier(
            model_path,
            device=device,
            image_size=image_size,
            tile_size=tile_size,
            overlap=overlap,
            pad_square=patch_pad,
            pad_size=pad_size,
            model_id=os.getenv("MIKANIA_MODEL_ID"),
        )
    if model_task != "detect":
        raise ValueError(f"不支持的模型任务：{model_task}")
    if model_path.suffix.lower() == ".onnx":
        return OnnxDetector(model_path, device=device or "cpu")
    return UltralyticsDetector(model_path, device=device)


def create_app(detector: Detector | None = None) -> FastAPI:
    app = FastAPI(
        title="薇甘菊目标识别系统",
        version="4.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.detector = detector or _default_detector()
    app.state.default_scan_mode = _default_scan_mode()
    app.state.default_thresholds = {
        "precise": _default_threshold("MIKANIA_PRECISE_THRESHOLD"),
        "fast": _default_threshold("MIKANIA_FAST_THRESHOLD"),
    }
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/v1/examples")
    async def examples() -> dict[str, list[dict[str, str]]]:
        available = []
        for example_id, title, filename in WEB_EXAMPLES:
            image_name = f"example-{example_id}.jpg"
            thumbnail_name = f"thumb-{example_id}.jpg"
            files = [STATIC_DIR / "media" / name for name in (image_name, thumbnail_name)]
            if all(path.is_file() and not path.is_symlink() for path in files):
                available.append(
                    {
                        "id": example_id,
                        "title": title,
                        "url": f"/assets/media/{image_name}",
                        "thumbnail": f"/assets/media/{thumbnail_name}",
                        "filename": filename,
                    }
                )
        return {"examples": available}

    @app.get("/api/v1/health")
    async def health() -> dict[str, object]:
        active_detector: Detector = app.state.detector
        result = {
            "status": "ready" if active_detector.ready else "model_missing",
            "model": active_detector.model_name,
            "default_threshold": app.state.default_thresholds[app.state.default_scan_mode],
            "default_thresholds": app.state.default_thresholds,
            "default_scan_mode": app.state.default_scan_mode,
        }
        deployment_id = os.getenv("MIKANIA_DEPLOYMENT_ID")
        if deployment_id:
            result["deployment_id"] = deployment_id
        return result

    @app.post("/api/v1/predict")
    async def predict(
        file: Annotated[UploadFile, File()],
        threshold: Annotated[float | None, Form(ge=0, le=1)] = None,
        scan_mode: Annotated[Literal["precise", "fast"] | None, Form()] = None,
    ) -> dict[str, object]:
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="仅支持 JPG、PNG 或 WebP 图片")

        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not contents:
            raise HTTPException(status_code=400, detail="上传的图片为空")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="单张图片不能超过 20 MB")

        try:
            image = decode_image(contents)
        except InvalidImageError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        active_detector: Detector = app.state.detector
        if not active_detector.ready:
            raise HTTPException(status_code=503, detail="检测模型尚未准备完成")

        actual_scan_mode = scan_mode or app.state.default_scan_mode
        actual_threshold = (
            app.state.default_thresholds[actual_scan_mode] if threshold is None else threshold
        )

        started = perf_counter()
        score = None
        try:
            presence_method = getattr(active_detector, "predict_result", None)
            precise_method = getattr(active_detector, "predict_precise", None)
            if callable(presence_method):
                prediction = presence_method(image, actual_threshold)
                detections = prediction.detections
                present = prediction.present
                score = prediction.score
            elif actual_scan_mode == "precise" and callable(precise_method):
                detections = precise_method(image, actual_threshold)
                present = bool(detections)
            else:
                detections = active_detector.predict(image, actual_threshold)
                present = bool(detections)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail="检测模型尚未准备完成") from error
        elapsed_ms = (perf_counter() - started) * 1000

        message = "识别到薇甘菊" if present else "未识别到薇甘菊"
        if present and not detections:
            message = "识别到疑似薇甘菊，暂无可靠区域"
        return {
            "present": present,
            "score": score,
            "message": message,
            "count": len(detections),
            "threshold": actual_threshold,
            "scan_mode": actual_scan_mode,
            "model": active_detector.model_name,
            "latency_ms": round(elapsed_ms, 1),
            "image_width": image.width,
            "image_height": image.height,
            "detections": [item.to_dict() for item in detections],
            "annotated_image": render_annotated_image(image, detections),
        }

    return app


app = create_app()
