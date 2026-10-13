"""CPU-only V4.1 inference, using the same padding and decision as the PT model."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from PIL import Image

from .predictor import (
    Detection,
    PresencePrediction,
    _pad_classifier_image,
    ensemble_patch_prediction,
)


def classifier_tensor(image: Image.Image, image_size: int, pad_size: int = 768) -> np.ndarray:
    # Training/evaluation uses PIL ImageOps.pad (bicubic), then torchvision's
    # PIL bilinear resize, ToTensor, and identity normalization. The intermediate
    # image is square, so its center crop is a no-op.
    square = _pad_classifier_image(image.convert("RGB"), pad_size)
    resized = square.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.0)
    return np.ascontiguousarray(tensor[None])


class OnnxClassifierBranch:
    def __init__(self, path: Path, image_size: int, pad_size: int = 768) -> None:
        self.path = path.resolve()
        self.image_size = image_size
        self.pad_size = pad_size
        self._session = None
        self._input_name = ""
        self._output_name = ""
        self._positive_index = 0

    @property
    def ready(self) -> bool:
        return self.path.is_file()

    def _load(self):
        if self._session is None:
            if not self.ready:
                raise FileNotFoundError(f"分类模型不存在：{self.path}")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or inputs[0].shape != [1, 3, self.image_size, self.image_size]:
                raise ValueError("ONNX分类模型输入尺寸与冻结配置不一致")
            if len(outputs) != 1 or outputs[0].shape != [1, 2]:
                raise ValueError("ONNX分类模型必须输出两类概率")
            names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get("names", "{}"))
            if not isinstance(names, dict) or set(names.values()) != {"mikania", "non_mikania"}:
                raise ValueError("ONNX分类模型类别信息无效")
            self._positive_index = int(
                next(index for index, name in names.items() if name == "mikania")
            )
            if self._positive_index not in (0, 1):
                raise ValueError("薇甘菊类别索引无效")
            self._input_name, self._output_name = inputs[0].name, outputs[0].name
            self._session = session
        return self._session

    def classify(self, images: list[Image.Image]) -> list[float]:
        session = self._load()
        values = []
        for image in images:
            tensor = classifier_tensor(image, self.image_size, self.pad_size)
            output = session.run([self._output_name], {self._input_name: tensor})[0]
            if (
                output.shape != (1, 2)
                or not np.isfinite(output).all()
                or np.min(output) < 0
                or np.max(output) > 1
                or abs(float(output.sum()) - 1.0) > 1e-4
            ):
                raise ValueError("ONNX分类模型返回了无效概率")
            values.append(float(output[0, self._positive_index]))
        return values


class OnnxEnsembleClassifier:
    def __init__(
        self,
        scene_model_path: Path,
        patch_model_path: Path,
        *,
        scene_image_size: int = 512,
        patch_image_size: int = 320,
        tile_size: int = 1280,
        overlap: float = 0.2,
        pad_size: int = 768,
        region_threshold: float = 0.37,
        model_id: str = "mikania-ensemble-v4.1",
    ) -> None:
        self.scene = OnnxClassifierBranch(scene_model_path, scene_image_size, pad_size)
        self.patch = OnnxClassifierBranch(patch_model_path, patch_image_size, pad_size)
        self.tile_size, self.overlap = tile_size, overlap
        self.region_threshold, self.model_id = region_threshold, model_id

    @property
    def ready(self) -> bool:
        return self.scene.ready and self.patch.ready

    @property
    def model_name(self) -> str:
        return self.model_id

    def predict_result(self, image: Image.Image, threshold: float) -> PresencePrediction:
        return ensemble_patch_prediction(
            lambda picture: self.scene.classify([picture])[0],
            self.patch.classify,
            image,
            threshold,
            tile_size=self.tile_size,
            overlap=self.overlap,
            region_threshold=self.region_threshold,
            decision_method="scene_patch_min",
        )

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        return self.predict_result(image, threshold).detections

    def predict_precise(self, image: Image.Image, threshold: float) -> list[Detection]:
        return self.predict(image, threshold)
