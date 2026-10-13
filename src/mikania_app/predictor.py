from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 50_000_000
PRECISE_TILE_SIZE = 1280
PRECISE_TILE_OVERLAP = 0.20
ENSEMBLE_DECISION_METHODS = {"weighted_sum", "scene_patch_min"}


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a safe, decodable image."""


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PresencePrediction:
    present: bool
    score: float
    detections: list[Detection]


def _pad_classifier_image(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.pad(image, (size, size), color=(114, 114, 114))


class Detector(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]: ...


class UltralyticsDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str | int | None = None,
        image_size: int = 640,
    ) -> None:
        self.model_path = model_path.resolve()
        self.device = device
        self.image_size = image_size
        self._model: object | None = None

    @property
    def ready(self) -> bool:
        return self.model_path.is_file()

    @property
    def model_name(self) -> str:
        if self.model_path.stem in {"best", "last"} and self.model_path.parent.name == "weights":
            return self.model_path.parent.parent.name
        return self.model_path.stem

    def _load_model(self) -> object:
        if not self.ready:
            raise FileNotFoundError(f"检测模型不存在：{self.model_path}")
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        model = self._load_model()
        results = model.predict(  # type: ignore[attr-defined]
            source=image,
            conf=threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            raw_name = result.names.get(class_id, "mikania")
            label = "mikania" if raw_name in {"mikania", "Mikania"} else str(raw_name)
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf[0].item()),
                    box_xyxy=coordinates,  # type: ignore[arg-type]
                )
            )
        return detections

    def predict_precise(self, image: Image.Image, threshold: float) -> list[Detection]:
        return precise_predict(self.predict, image, threshold)


class UltralyticsPatchClassifier:
    """Classify overlapping regions and use their robust aggregate for presence."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | int | None = None,
        image_size: int = 384,
        tile_size: int = PRECISE_TILE_SIZE,
        overlap: float = PRECISE_TILE_OVERLAP,
        pad_square: bool = False,
        pad_size: int = 768,
        model_id: str | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.device = device
        self.image_size = image_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.pad_square = pad_square
        self.pad_size = pad_size
        self.model_id = model_id
        self._model: object | None = None

    @property
    def ready(self) -> bool:
        return self.model_path.is_file()

    @property
    def model_name(self) -> str:
        if self.model_id:
            return self.model_id
        if self.model_path.stem in {"best", "last", "model"}:
            return self.model_path.parent.name
        return self.model_path.stem

    def _load_model(self) -> object:
        if not self.ready:
            raise FileNotFoundError(f"分类模型不存在：{self.model_path}")
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def _classify(self, crops: list[Image.Image]) -> list[float]:
        model = self._load_model()
        if self.pad_square:
            crops = [_pad_classifier_image(crop, self.pad_size) for crop in crops]
        results = model.predict(  # type: ignore[attr-defined]
            source=crops,
            imgsz=self.image_size,
            device=self.device,
            batch=64,
            verbose=False,
        )
        scores: list[float] = []
        for result in results:
            scores.append(_ultralytics_mikania_probability(result))
        return scores

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        return patch_classification_predict(
            self._classify,
            image,
            threshold,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

    def predict_precise(self, image: Image.Image, threshold: float) -> list[Detection]:
        return self.predict(image, threshold)


class UltralyticsEnsembleClassifier:
    """Combine whole-image context with repeated local evidence."""

    def __init__(
        self,
        scene_model_path: Path,
        patch_model_path: Path,
        *,
        device: str | int | None = None,
        scene_image_size: int = 384,
        patch_image_size: int = 320,
        tile_size: int = PRECISE_TILE_SIZE,
        overlap: float = PRECISE_TILE_OVERLAP,
        scene_weight: float = 0.25,
        region_threshold: float = 0.37,
        model_id: str = "mikania-ensemble-v3",
        scene_pad_square: bool = False,
        patch_pad_square: bool = False,
        pad_size: int = 768,
        decision_method: str = "weighted_sum",
    ) -> None:
        if decision_method not in ENSEMBLE_DECISION_METHODS:
            raise ValueError(f"不支持的图像判定规则：{decision_method}")
        self.scene_model_path = scene_model_path.resolve()
        self.patch_model = UltralyticsPatchClassifier(
            patch_model_path,
            device=device,
            image_size=patch_image_size,
            tile_size=tile_size,
            overlap=overlap,
            pad_square=patch_pad_square,
            pad_size=pad_size,
        )
        self.device = device
        self.scene_image_size = scene_image_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.scene_weight = scene_weight
        self.decision_method = decision_method
        self.region_threshold = region_threshold
        self.model_id = model_id
        self.scene_pad_square = scene_pad_square
        self.pad_size = pad_size
        self._scene_model: object | None = None

    @property
    def ready(self) -> bool:
        return self.scene_model_path.is_file() and self.patch_model.ready

    @property
    def model_name(self) -> str:
        return self.model_id

    def _load_scene_model(self) -> object:
        if not self.scene_model_path.is_file():
            raise FileNotFoundError(f"整图分类模型不存在：{self.scene_model_path}")
        if self._scene_model is None:
            from ultralytics import YOLO

            self._scene_model = YOLO(str(self.scene_model_path))
        return self._scene_model

    def _classify_scene(self, image: Image.Image) -> float:
        model = self._load_scene_model()
        if self.scene_pad_square:
            image = _pad_classifier_image(image, self.pad_size)
        result = model.predict(  # type: ignore[attr-defined]
            source=image,
            imgsz=self.scene_image_size,
            device=self.device,
            verbose=False,
        )[0]
        return _ultralytics_mikania_probability(result)

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        return self.predict_result(image, threshold).detections

    def predict_result(self, image: Image.Image, threshold: float) -> PresencePrediction:
        return ensemble_patch_prediction(
            self._classify_scene,
            self.patch_model._classify,
            image,
            threshold,
            tile_size=self.tile_size,
            overlap=self.overlap,
            scene_weight=self.scene_weight,
            region_threshold=self.region_threshold,
            decision_method=self.decision_method,
        )

    def predict_precise(self, image: Image.Image, threshold: float) -> list[Detection]:
        return self.predict(image, threshold)


class OnnxDetector:
    """Portable CPU detector for the exported end-to-end YOLO26 ONNX model."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | int | None = "cpu",
        image_size: int | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.device = device
        self.image_size = image_size
        self._session: object | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None

    @property
    def ready(self) -> bool:
        return self.model_path.is_file()

    @property
    def model_name(self) -> str:
        if self.model_path.stem in {"model", "best", "last"}:
            return self.model_path.parent.name
        return self.model_path.stem

    def _load_session(self) -> object:
        if not self.ready:
            raise FileNotFoundError(f"检测模型不存在：{self.model_path}")
        if self._session is None:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if str(self.device).lower() != "cpu" and "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            session = ort.InferenceSession(str(self.model_path), providers=providers)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            input_shape = input_info.shape
            if (
                len(input_shape) != 4
                or input_shape[0] != 1
                or input_shape[1] != 3
                or not isinstance(input_shape[2], int)
                or input_shape[2] != input_shape[3]
            ):
                raise ValueError(f"不支持的 ONNX 输入尺寸：{input_info.shape}")
            exported_image_size = input_shape[2]
            if self.image_size is not None and self.image_size != exported_image_size:
                raise ValueError(
                    f"ONNX 输入尺寸为 {exported_image_size}，与配置的 {self.image_size} 不一致"
                )
            if output_info.shape != [1, 300, 6]:
                raise ValueError(f"不支持的 ONNX 输出尺寸：{output_info.shape}")
            self.image_size = exported_image_size
            self._session = session
            self._input_name = input_info.name
            self._output_name = output_info.name
        return self._session

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        import cv2
        import numpy as np

        session = self._load_session()
        input_name = self._input_name
        output_name = self._output_name
        if input_name is None or output_name is None:
            raise RuntimeError("ONNX 会话没有完成初始化")
        if self.image_size is None:
            raise RuntimeError("ONNX 输入尺寸没有完成初始化")

        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        scale = min(self.image_size / height, self.image_size / width)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        pad_width = (self.image_size - resized_width) / 2
        pad_height = (self.image_size - resized_height) / 2
        left = round(pad_width - 0.1)
        right = round(pad_width + 0.1)
        top = round(pad_height - 0.1)
        bottom = round(pad_height + 0.1)

        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(letterboxed.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        outputs = session.run([output_name], {input_name: tensor})  # type: ignore[attr-defined]
        predictions = outputs[0][0]

        detections: list[Detection] = []
        for row in predictions:
            confidence = float(row[4])
            if confidence < threshold:
                continue
            class_id = int(row[5])
            if class_id != 0:
                continue
            x1 = (float(row[0]) - left) / scale
            y1 = (float(row[1]) - top) / scale
            x2 = (float(row[2]) - left) / scale
            y2 = (float(row[3]) - top) / scale
            detections.append(
                Detection(
                    label="mikania",
                    confidence=confidence,
                    box_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def predict_precise(self, image: Image.Image, threshold: float) -> list[Detection]:
        return precise_predict(self.predict, image, threshold)


def precise_predict(
    predict_crop: Callable[[Image.Image, float], list[Detection]],
    image: Image.Image,
    threshold: float,
    *,
    tile_size: int = PRECISE_TILE_SIZE,
    overlap: float = PRECISE_TILE_OVERLAP,
) -> list[Detection]:
    """Scan a high-resolution image in overlapping tiles and merge duplicate boxes."""
    if image.width <= tile_size and image.height <= tile_size:
        return _nms_detections(predict_crop(image, threshold))

    detections: list[Detection] = []
    for top in _tile_starts(image.height, tile_size, overlap):
        for left in _tile_starts(image.width, tile_size, overlap):
            crop = image.crop(
                (
                    left,
                    top,
                    min(image.width, left + tile_size),
                    min(image.height, top + tile_size),
                )
            )
            for detection in predict_crop(crop, threshold):
                x1, y1, x2, y2 = detection.box_xyxy
                detections.append(
                    Detection(
                        label=detection.label,
                        confidence=detection.confidence,
                        box_xyxy=(x1 + left, y1 + top, x2 + left, y2 + top),
                    )
                )
    return _nms_detections(detections)


def patch_classification_predict(
    classify_batch: Callable[[list[Image.Image]], list[float]],
    image: Image.Image,
    threshold: float,
    *,
    tile_size: int = PRECISE_TILE_SIZE,
    overlap: float = PRECISE_TILE_OVERLAP,
    quantile: float = 0.75,
) -> list[Detection]:
    """Return coarse positive regions only when the image aggregate is positive."""
    boxes = [
        (
            left,
            top,
            min(image.width, left + tile_size),
            min(image.height, top + tile_size),
        )
        for top in _tile_starts(image.height, tile_size, overlap)
        for left in _tile_starts(image.width, tile_size, overlap)
    ]
    scores = classify_batch([image.crop(box) for box in boxes])
    if len(scores) != len(boxes):
        raise ValueError("分类结果数量与图像区域数量不一致")
    if _quantile(scores, quantile) < threshold:
        return []
    return [
        Detection(
            label="mikania_region",
            confidence=score,
            box_xyxy=tuple(float(value) for value in box),
        )
        for box, score in zip(boxes, scores, strict=True)
        if score >= threshold
    ]


def ensemble_patch_predict(
    classify_scene: Callable[[Image.Image], float],
    classify_batch: Callable[[list[Image.Image]], list[float]],
    image: Image.Image,
    threshold: float,
    *,
    tile_size: int = PRECISE_TILE_SIZE,
    overlap: float = PRECISE_TILE_OVERLAP,
    scene_weight: float = 0.25,
    region_threshold: float = 0.37,
    decision_method: str = "weighted_sum",
) -> list[Detection]:
    """Backward-compatible region list; use ensemble_patch_prediction for presence."""
    return ensemble_patch_prediction(
        classify_scene,
        classify_batch,
        image,
        threshold,
        tile_size=tile_size,
        overlap=overlap,
        scene_weight=scene_weight,
        region_threshold=region_threshold,
        decision_method=decision_method,
    ).detections


def ensemble_patch_prediction(
    classify_scene: Callable[[Image.Image], float],
    classify_batch: Callable[[list[Image.Image]], list[float]],
    image: Image.Image,
    threshold: float,
    *,
    tile_size: int = PRECISE_TILE_SIZE,
    overlap: float = PRECISE_TILE_OVERLAP,
    scene_weight: float = 0.25,
    region_threshold: float = 0.37,
    decision_method: str = "weighted_sum",
) -> PresencePrediction:
    """Keep image presence independent of optional coarse region display."""
    if decision_method not in ENSEMBLE_DECISION_METHODS:
        raise ValueError(f"不支持的图像判定规则：{decision_method}")
    boxes = [
        (
            left,
            top,
            min(image.width, left + tile_size),
            min(image.height, top + tile_size),
        )
        for top in _tile_starts(image.height, tile_size, overlap)
        for left in _tile_starts(image.width, tile_size, overlap)
    ]
    scores = classify_batch([image.crop(box) for box in boxes])
    if len(scores) != len(boxes):
        raise ValueError("分类结果数量与图像区域数量不一致")
    ordered = sorted(scores, reverse=True)
    repeated_patch_score = ordered[min(1, len(ordered) - 1)]
    scene_score = classify_scene(image)
    decision_score = (
        min(scene_score, repeated_patch_score)
        if decision_method == "scene_patch_min"
        else scene_weight * scene_score + (1 - scene_weight) * repeated_patch_score
    )
    if decision_score < threshold:
        return PresencePrediction(False, decision_score, [])
    detections = [
        Detection(
            label="mikania_region",
            confidence=score,
            box_xyxy=tuple(float(value) for value in box),
        )
        for box, score in zip(boxes, scores, strict=True)
        if score >= region_threshold
    ]
    return PresencePrediction(True, decision_score, detections)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0 <= fraction <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ultralytics_mikania_probability(result: object) -> float:
    names = result.names  # type: ignore[attr-defined]
    class_id = next(
        (class_index for class_index, name in names.items() if name == "mikania"),
        None,
    )
    if class_id is None:
        raise ValueError(f"分类模型缺少 mikania 类：{names}")
    return float(result.probs.data[class_id].item())  # type: ignore[attr-defined]


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _nms_detections(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(
            _box_iou(candidate, existing) < iou_threshold
            and _smaller_box_overlap(candidate, existing) < 0.75
            for existing in kept
        ):
            kept.append(candidate)
    return kept


def _box_iou(left: Detection, right: Detection) -> float:
    left_x1, left_y1, left_x2, left_y2 = left.box_xyxy
    right_x1, right_y1, right_x2, right_y2 = right.box_xyxy
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _smaller_box_overlap(left: Detection, right: Detection) -> float:
    left_x1, left_y1, left_x2, left_y2 = left.box_xyxy
    right_x1, right_y1, right_x2, right_y2 = right.box_xyxy
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    smaller_area = min(left_area, right_area)
    return intersection / smaller_area if smaller_area else 0.0


def decode_image(contents: bytes) -> Image.Image:
    def decode(payload: bytes) -> Image.Image:
        with Image.open(BytesIO(payload)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("图片尺寸无效或像素过大")
            source.load()
            corrected = ImageOps.exif_transpose(source)
            return corrected.convert("RGB")

    try:
        try:
            return decode(contents)
        except OSError:
            if not contents.startswith(b"\xff\xd8") or contents.endswith(b"\xff\xd9"):
                raise
            return decode(contents + b"\xff\xd9")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImageError("文件不是可识别的 JPG、PNG 或 WebP 图片") from error


def render_annotated_image(image: Image.Image, detections: list[Detection]) -> str:
    has_regions = any(item.label == "mikania_region" for item in detections)
    canvas = image.convert("RGBA") if has_regions else image.copy()
    if has_regions:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        region_width = max(3, round(min(canvas.size) / 220))
        for detection in detections:
            if detection.label != "mikania_region":
                continue
            box = _clamp_box(detection.box_xyxy, canvas.size)
            alpha = min(92, 28 + round(55 * detection.confidence))
            overlay_draw.rounded_rectangle(
                box,
                radius=max(4, region_width * 2),
                fill=(79, 205, 130, alpha),
                outline=(190, 255, 105, 225),
                width=region_width,
            )
        canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(min(canvas.size) / 180))
    font = None

    for detection in detections:
        if detection.label == "mikania_region":
            continue
        left, top, right, bottom = _clamp_box(detection.box_xyxy, canvas.size)
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=max(4, line_width * 2),
            outline="#d7ff5f",
            width=line_width,
        )
        label = f"Mikania {detection.confidence:.0%}"
        text_box = draw.textbbox((left, top), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, top - text_height - line_width * 3)
        draw.rounded_rectangle(
            (
                left,
                label_top,
                min(canvas.width, left + text_width + line_width * 4),
                label_top + text_height + line_width * 2,
            ),
            radius=line_width,
            fill="#183a2b",
        )
        draw.text(
            (left + line_width * 2, label_top + line_width),
            label,
            fill="#f4f0dd",
            font=font,
        )

    output = BytesIO()
    canvas.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _clamp_box(
    box: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    return (
        max(0, min(width - 1, round(left))),
        max(0, min(height - 1, round(top))),
        max(0, min(width - 1, round(right))),
        max(0, min(height - 1, round(bottom))),
    )
