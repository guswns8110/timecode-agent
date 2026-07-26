from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import OBJECT_MODEL
from .search_query import ObjectConstraint


@dataclass(frozen=True)
class DetectionObservation:
    count: int
    confidence: float


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if not len(others):
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], others[:, :2])
    bottom_right = np.minimum(box[2:], others[:, 2:])
    size = np.maximum(0.0, bottom_right - top_left)
    intersection = size[:, 0] * size[:, 1]
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0,
        others[:, 3] - others[:, 1],
    )
    return intersection / np.maximum(area + other_area - intersection, 1e-6)


def _nms_indices(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> list[int]:
    if not len(boxes):
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[_box_iou(boxes[current], boxes[remaining]) < threshold]
    return keep


class ObjectSearchEngine:
    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = model_path
        self.device_request = device
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        use_cuda = torch.cuda.is_available() and self.device_request != "cpu"
        self._device = "cuda" if use_cuda else "cpu"
        source = (
            str(self.model_path)
            if (self.model_path / ".complete").is_file()
            else OBJECT_MODEL
        )
        self._processor = AutoProcessor.from_pretrained(source)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(source)
            .to(self._device)
            .eval()
        )

    def inspect(
        self,
        frame_path: str,
        constraint: ObjectConstraint,
    ) -> DetectionObservation:
        import torch
        from PIL import Image

        self._load()
        assert self._processor is not None
        assert self._model is not None

        with Image.open(frame_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            inputs = self._processor(
                images=image,
                text=f"{constraint.detector_label}.",
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                outputs = self._model(**inputs)

        arguments = {
            "target_sizes": [(height, width)],
            "text_threshold": 0.25,
        }
        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.28,
                **arguments,
            )
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.28,
                **arguments,
            )

        boxes = results[0]["boxes"].detach().float().cpu().numpy()
        scores = results[0]["scores"].detach().float().cpu().numpy()
        if not len(boxes):
            return DetectionObservation(count=0, confidence=0.0)

        frame_area = max(1.0, float(width * height))
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0,
            boxes[:, 3] - boxes[:, 1],
        )
        visible = areas >= frame_area * 0.001
        boxes = boxes[visible]
        scores = scores[visible]
        keep = _nms_indices(boxes, scores)
        if not keep:
            return DetectionObservation(count=0, confidence=0.0)
        kept_scores = scores[keep]
        return DetectionObservation(
            count=len(keep),
            confidence=float(np.mean(kept_scores)),
        )

    def release(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
