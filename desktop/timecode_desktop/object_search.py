from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import OBJECT_MODEL, VISION_MODEL
from .search_query import ObjectConstraint

_GENDER_KINDS = {"man", "woman", "child"}
_PERSON_PROMPTS = (
    ("man", "a photo of an adult man"),
    ("man", "a male adult person"),
    ("woman", "a photo of an adult woman"),
    ("woman", "a female adult person"),
    ("child", "a photo of a child"),
    ("child", "a young boy or girl"),
    ("non_person", "a graphic icon or illustration, not a real person"),
    ("non_person", "an object, screen graphic, or empty background"),
)


@dataclass(frozen=True)
class DetectionObservation:
    count: int
    confidence: float
    total_people: int | None = None
    semantic_confidence: float = 0.0


@dataclass(frozen=True)
class PersonPrediction:
    kind: str
    confidence: float


def _pooled_features(output):
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return output


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


def _person_observations(
    predictions: list[PersonPrediction],
    constraints: tuple[ObjectConstraint, ...],
) -> list[DetectionObservation]:
    real_people = [
        prediction
        for prediction in predictions
        if prediction.kind != "non_person"
    ]
    observations: list[DetectionObservation] = []
    for constraint in constraints:
        if constraint.person_kind == "person":
            matched = real_people
        else:
            matched = [
                prediction
                for prediction in real_people
                if prediction.kind == constraint.person_kind
            ]
        observations.append(
            DetectionObservation(
                count=len(matched),
                confidence=(
                    float(np.mean([item.confidence for item in matched]))
                    if matched
                    else 0.0
                ),
                total_people=len(real_people),
            )
        )
    return observations


def exact_person_total_matches(
    constraints: tuple[ObjectConstraint, ...],
    observations: list[DetectionObservation],
) -> bool:
    paired = list(zip(constraints, observations, strict=True))
    gendered = [
        (constraint, observation)
        for constraint, observation in paired
        if constraint.person_kind in _GENDER_KINDS
        and not constraint.excluded
    ]
    if not gendered or not all(
        constraint.comparison == "exact"
        and constraint.count is not None
        for constraint, _ in gendered
    ):
        return True

    generic_exact = next(
        (
            constraint.count
            for constraint, _ in paired
            if constraint.person_kind == "person"
            and constraint.comparison == "exact"
            and constraint.count is not None
        ),
        None,
    )
    expected_total = (
        generic_exact
        if generic_exact is not None
        else sum(constraint.count or 0 for constraint, _ in gendered)
    )
    observed_total = next(
        (
            observation.total_people
            for _, observation in gendered
            if observation.total_people is not None
        ),
        0,
    )
    return observed_total == expected_total


class ObjectSearchEngine:
    def __init__(
        self,
        model_path: Path,
        vision_model_path: Path | None = None,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.vision_model_path = vision_model_path
        self.device_request = device
        self._model = None
        self._processor = None
        self._vision_model = None
        self._vision_processor = None
        self._person_text_features = None
        self._object_text_features: dict[str, object] = {}
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

    def _load_vision(self) -> None:
        if self._vision_model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        source = (
            str(self.vision_model_path)
            if self.vision_model_path is not None
            and (self.vision_model_path / ".complete").is_file()
            else VISION_MODEL
        )
        self._vision_processor = AutoProcessor.from_pretrained(source)
        self._vision_model = (
            AutoModel.from_pretrained(source)
            .to(self._device)
            .eval()
        )

    def _detect(
        self,
        image,
        label: str,
        box_threshold: float,
        minimum_area: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None
        width, height = image.size
        inputs = self._processor(
            images=image,
            text=f"{label}.",
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
                threshold=box_threshold,
                **arguments,
            )
        except TypeError:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                **arguments,
            )

        boxes = results[0]["boxes"].detach().float().cpu().numpy()
        scores = results[0]["scores"].detach().float().cpu().numpy()
        if not len(boxes):
            return boxes, scores
        frame_area = max(1.0, float(width * height))
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
            0.0,
            boxes[:, 3] - boxes[:, 1],
        )
        dimensions = np.column_stack(
            (
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            )
        )
        visible = (
            (areas >= frame_area * minimum_area)
            & (dimensions[:, 0] >= 24)
            & (dimensions[:, 1] >= 24)
        )
        boxes = boxes[visible]
        scores = scores[visible]
        keep = _nms_indices(boxes, scores)
        return boxes[keep], scores[keep]

    @staticmethod
    def _crop_person(image, box: np.ndarray):
        width, height = image.size
        x1, y1, x2, y2 = (float(value) for value in box)
        padding_x = (x2 - x1) * 0.08
        padding_y = (y2 - y1) * 0.08
        crop = (
            max(0, int(x1 - padding_x)),
            max(0, int(y1 - padding_y)),
            min(width, int(x2 + padding_x)),
            min(height, int(y2 + padding_y)),
        )
        return image.crop(crop).convert("RGB")

    def _classify_people(
        self,
        image,
        boxes: np.ndarray,
        detector_scores: np.ndarray,
    ) -> list[PersonPrediction]:
        if not len(boxes):
            return []
        import torch

        self._load_vision()
        assert self._vision_processor is not None
        assert self._vision_model is not None
        crops = [self._crop_person(image, box) for box in boxes]
        image_inputs = self._vision_processor(
            images=crops,
            return_tensors="pt",
        )
        image_inputs = {
            key: value.to(self._device)
            for key, value in image_inputs.items()
        }
        with torch.inference_mode():
            image_features = _pooled_features(
                self._vision_model.get_image_features(**image_inputs)
            )
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            if self._person_text_features is None:
                text_inputs = self._vision_processor(
                    text=[prompt for _, prompt in _PERSON_PROMPTS],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text_inputs = {
                    key: value.to(self._device)
                    for key, value in text_inputs.items()
                }
                text_features = _pooled_features(
                    self._vision_model.get_text_features(**text_inputs)
                )
                self._person_text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )

            similarities = (
                image_features @ self._person_text_features.T
            ).float().cpu().numpy()

        prompt_kinds = [kind for kind, _ in _PERSON_PROMPTS]
        predictions: list[PersonPrediction] = []
        for row, detector_score in zip(
            similarities,
            detector_scores,
            strict=True,
        ):
            kind_scores = {
                kind: max(
                    float(row[index])
                    for index, prompt_kind in enumerate(prompt_kinds)
                    if prompt_kind == kind
                )
                for kind in set(prompt_kinds)
            }
            human_rank = sorted(
                (
                    (score, kind)
                    for kind, score in kind_scores.items()
                    if kind in _GENDER_KINDS
                ),
                reverse=True,
            )
            best_human_score, best_human_kind = human_rank[0]
            second_human_score = human_rank[1][0]
            non_person_score = kind_scores["non_person"]
            if non_person_score >= best_human_score - 0.005:
                predictions.append(
                    PersonPrediction(
                        kind="non_person",
                        confidence=float(detector_score),
                    )
                )
                continue
            margin = best_human_score - second_human_score
            predictions.append(
                PersonPrediction(
                    kind=(
                        best_human_kind
                        if margin >= 0.012
                        else "person"
                    ),
                    confidence=float(
                        np.clip(
                            float(detector_score) * 0.65
                            + (0.5 + margin * 4.0) * 0.35,
                            0.0,
                            1.0,
                        )
                    ),
                )
            )
        return predictions

    def _verify_object_boxes(
        self,
        image,
        boxes: np.ndarray,
        detector_scores: np.ndarray,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(boxes):
            return boxes, detector_scores
        import torch

        self._load_vision()
        assert self._vision_processor is not None
        assert self._vision_model is not None
        crops = [self._crop_person(image, box) for box in boxes]
        image_inputs = self._vision_processor(
            images=crops,
            return_tensors="pt",
        )
        image_inputs = {
            key: value.to(self._device)
            for key, value in image_inputs.items()
        }
        with torch.inference_mode():
            image_features = _pooled_features(
                self._vision_model.get_image_features(**image_inputs)
            )
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )
            text_features = self._object_text_embedding(label)
            similarities = (
                image_features @ text_features.T
            ).float().cpu().numpy()

        margins = similarities[:, 0] - similarities[:, 1]
        keep = margins >= -0.01
        verified_scores = np.clip(
            detector_scores * 0.72
            + np.clip((margins + 0.04) / 0.12, 0.0, 1.0) * 0.28,
            0.0,
            1.0,
        )
        return boxes[keep], verified_scores[keep]

    def _object_text_embedding(self, label: str):
        import torch

        self._load_vision()
        assert self._vision_processor is not None
        assert self._vision_model is not None
        text_features = self._object_text_features.get(label)
        if text_features is not None:
            return text_features
        text_inputs = self._vision_processor(
            text=[
                f"a clear photo of {label}",
                f"an unrelated object or background without {label}",
            ],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {
            key: value.to(self._device)
            for key, value in text_inputs.items()
        }
        with torch.inference_mode():
            text_features = _pooled_features(
                self._vision_model.get_text_features(**text_inputs)
            )
            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )
        self._object_text_features[label] = text_features
        return text_features

    def _semantic_object_confidence(self, image, label: str) -> float:
        import torch

        self._load_vision()
        assert self._vision_processor is not None
        assert self._vision_model is not None
        inputs = self._vision_processor(
            images=[image],
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            image_features = _pooled_features(
                self._vision_model.get_image_features(**inputs)
            )
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )
            similarities = (
                image_features @ self._object_text_embedding(label).T
            ).float().cpu().numpy()[0]
        margin = float(similarities[0] - similarities[1])
        return float(np.clip((margin + 0.035) / 0.09, 0.0, 1.0))

    def inspect_constraints(
        self,
        frame_path: str,
        constraints: tuple[ObjectConstraint, ...],
    ) -> list[DetectionObservation]:
        from PIL import Image

        with Image.open(frame_path) as opened:
            image = opened.convert("RGB")
            observations: dict[int, DetectionObservation] = {}
            person_pairs = [
                (index, constraint)
                for index, constraint in enumerate(constraints)
                if constraint.person_kind is not None
            ]
            if person_pairs:
                boxes, scores = self._detect(
                    image,
                    "a real human person",
                    box_threshold=0.34,
                    minimum_area=0.002,
                )
                person_constraints = tuple(
                    constraint
                    for _, constraint in person_pairs
                )
                person_results = _person_observations(
                    self._classify_people(image, boxes, scores),
                    person_constraints,
                )
                for (index, _), observation in zip(
                    person_pairs,
                    person_results,
                    strict=True,
                ):
                    observations[index] = observation

            for index, constraint in enumerate(constraints):
                if constraint.person_kind is not None:
                    continue
                boxes, scores = self._detect(
                    image,
                    constraint.detector_label,
                    box_threshold=0.23,
                    minimum_area=0.0008,
                )
                boxes, scores = self._verify_object_boxes(
                    image,
                    boxes,
                    scores,
                    constraint.detector_label,
                )
                observations[index] = DetectionObservation(
                    count=len(boxes),
                    confidence=(
                        float(np.mean(scores))
                        if len(scores)
                        else 0.0
                    ),
                    semantic_confidence=self._semantic_object_confidence(
                        image,
                        constraint.detector_label,
                    ),
                )
        return [
            observations.get(index, DetectionObservation(0, 0.0))
            for index in range(len(constraints))
        ]

    def inspect(
        self,
        frame_path: str,
        constraint: ObjectConstraint,
    ) -> DetectionObservation:
        return self.inspect_constraints(frame_path, (constraint,))[0]

    def release(self) -> None:
        self._model = None
        self._processor = None
        self._vision_model = None
        self._vision_processor = None
        self._person_text_features = None
        self._object_text_features.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
