from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .audio_search import AudioSearchEngine, audio_relevance
from .config import VISION_MODEL
from .object_search import (
    ObjectSearchEngine,
    exact_person_total_matches,
)
from .ocr_search import ScreenTextIndexer
from .search_query import ObjectConstraint
from .temporal_search import TemporalSearchEngine, temporal_relevance
from .translation_search import SceneQueryTranslator, contains_hangul

_INDEX_VERSION = 3


@dataclass
class VisualHit:
    workspace: str
    video: str
    start: float
    end: float
    score: float
    frame: str
    source: str = "화면 의미"
    evidence: str = ""
    vector: np.ndarray | None = field(default=None, repr=False)


def _batched(values: list, size: int) -> Iterable[list]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _pooled_features(output):
    """Return the embedding tensor across Transformers 4.x and 5.x.

    SigLIP2's get_*_features returned a tensor in older Transformers releases.
    Transformers 5.x returns BaseModelOutputWithPooling instead, whose actual
    embedding tensor is stored in pooler_output.
    """
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return output


def _sigmoid_relevance(
    cosine_scores: np.ndarray,
    logit_scale: float,
    logit_bias: float,
) -> np.ndarray:
    logits = np.clip(
        cosine_scores * np.exp(logit_scale) + logit_bias,
        -60.0,
        60.0,
    )
    return 1.0 / (1.0 + np.exp(-logits))


def _image_attributes(image, previous_gray: np.ndarray | None):
    from PIL import Image

    small = image.convert("RGB").resize(
        (96, 54),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(small, dtype=np.float32) / 255.0
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    gray = red * 0.2126 + green * 0.7152 + blue * 0.0722
    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        np.maximum(maximum, 1e-4),
    )
    motion = (
        0.0
        if previous_gray is None
        else float(
            np.clip(
                np.mean(np.abs(gray - previous_gray)) / 0.22,
                0.0,
                1.0,
            )
        )
    )
    values = np.asarray(
        [
            np.clip(float(np.mean(gray)), 0.0, 1.0),
            np.clip(float(np.std(gray)) / 0.24, 0.0, 1.0),
            np.clip(float(np.mean(saturation)), 0.0, 1.0),
            np.clip(
                0.5 + float(np.mean(red) - np.mean(blue)),
                0.0,
                1.0,
            ),
            motion,
        ],
        dtype=np.float32,
    )
    return values, gray


def _attribute_relevance(
    attributes: np.ndarray,
    preferences: tuple[tuple[str, float], ...],
) -> np.ndarray:
    columns = {
        "brightness": 0,
        "contrast": 1,
        "saturation": 2,
        "warmth": 3,
        "motion": 4,
    }
    scores: list[np.ndarray] = []
    weights: list[float] = []
    for name, direction in preferences:
        column = columns.get(name)
        if column is None or column >= attributes.shape[1]:
            continue
        value = np.clip(attributes[:, column], 0.0, 1.0)
        scores.append(value if direction >= 0 else 1.0 - value)
        weights.append(max(0.1, abs(direction)))
    if not scores:
        return np.zeros(len(attributes), dtype=np.float32)
    return np.average(np.stack(scores), axis=0, weights=weights)


def _scene_boundaries(workspace: Path, duration: float) -> list[float]:
    boundaries = [0.0, max(0.0, duration)]
    scene_path = workspace / "scenes.json"
    if scene_path.is_file():
        try:
            scenes = json.loads(scene_path.read_text(encoding="utf-8"))
            boundaries.extend(
                float(item["t"])
                for item in scenes
                if 0.0 < float(item.get("t", 0.0)) < duration
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return sorted(set(boundaries))


def _shot_span(
    timestamp: float,
    boundaries: list[float],
    fallback_seconds: float,
) -> tuple[float, float]:
    if len(boundaries) < 2:
        return timestamp, timestamp + fallback_seconds
    position = max(
        0,
        min(
            bisect.bisect_right(boundaries, timestamp) - 1,
            len(boundaries) - 2,
        ),
    )
    return boundaries[position], boundaries[position + 1]


class VisualSearchEngine:
    def __init__(
        self,
        model_path: Path,
        object_model_path: Path | None = None,
        temporal_model_path: Path | None = None,
        translation_model_path: Path | None = None,
        ocr_model_path: Path | None = None,
        audio_model_path: Path | None = None,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.object_model_path = object_model_path
        self.temporal_model_path = temporal_model_path
        self.translation_model_path = translation_model_path
        self.ocr_model_path = ocr_model_path
        self.audio_model_path = audio_model_path
        self.device_request = device
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._logit_scale: float | None = None
        self._logit_bias: float | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        use_cuda = torch.cuda.is_available() and self.device_request != "cpu"
        self._device = "cuda" if use_cuda else "cpu"
        source = (
            str(self.model_path)
            if (self.model_path / ".complete").is_file()
            else VISION_MODEL
        )
        self._processor = AutoProcessor.from_pretrained(source)
        self._model = AutoModel.from_pretrained(source).to(self._device).eval()
        scale = getattr(self._model, "logit_scale", None)
        bias = getattr(self._model, "logit_bias", None)
        self._logit_scale = (
            float(scale.detach().float().cpu().item())
            if scale is not None
            else None
        )
        self._logit_bias = (
            float(bias.detach().float().cpu().item())
            if bias is not None
            else None
        )

    def _release_model(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _image_embeddings(self, images: list) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = _pooled_features(
                self._model.get_image_features(**inputs)
            )
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def _text_embeddings(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = _pooled_features(
                self._model.get_text_features(**inputs)
            )
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def _relevance_scores(self, cosine_scores: np.ndarray) -> np.ndarray:
        if self._logit_scale is None or self._logit_bias is None:
            return np.clip((cosine_scores - 0.15) / 0.2, 0.0, 1.0)
        return _sigmoid_relevance(
            cosine_scores,
            self._logit_scale,
            self._logit_bias,
        )

    def build_index(
        self,
        workspace: Path,
        sample_seconds: float = 1.0,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        import av
        from PIL import Image

        manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
        video_path = Path(manifest["video"])
        frames_dir = workspace / "visual-index" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        duration = float(manifest.get("duration") or 0)
        wanted = np.arange(0, max(duration, sample_seconds), sample_seconds)
        timestamps: list[float] = []
        frame_paths: list[str] = []
        attribute_rows: list[np.ndarray] = []
        previous_gray = None
        next_index = 0

        for frame in container.decode(stream):
            if next_index >= len(wanted):
                break
            timestamp = float(frame.time or 0)
            if timestamp + 0.001 < wanted[next_index]:
                continue
            image = frame.to_image().convert("RGB")
            path = frames_dir / f"t{int(timestamp * 1000):012d}.jpg"
            image.save(path, quality=84)
            timestamps.append(timestamp)
            frame_paths.append(str(path))
            attributes, previous_gray = _image_attributes(
                image,
                previous_gray,
            )
            attribute_rows.append(attributes)
            next_index += 1
            if progress:
                progress(next_index, len(wanted) * 4)
        container.close()

        embeddings: list[np.ndarray] = []
        self._load()
        batch_size = 16 if self._device == "cuda" else 4
        for path_batch in _batched(frame_paths, batch_size):
            images: list[Image.Image] = []
            try:
                for frame_path in path_batch:
                    with Image.open(frame_path) as opened:
                        images.append(opened.convert("RGB"))
                embeddings.append(self._image_embeddings(images))
            finally:
                for image in images:
                    image.close()
        matrix = (
            np.concatenate(embeddings).astype(np.float16)
            if embeddings
            else np.empty((0, 0), dtype=np.float16)
        )
        self._release_model()
        temporal_matrix = np.empty((0, 0), dtype=np.float16)
        if self.temporal_model_path is not None and len(timestamps):
            temporal = TemporalSearchEngine(
                self.temporal_model_path,
                device=self.device_request,
            )
            temporal_matrix = temporal.build_embeddings(
                video_path,
                np.asarray(timestamps, dtype=np.float32),
                sample_seconds,
                progress=(
                    None
                    if progress is None
                    else lambda done, total: progress(
                        len(wanted) + done,
                        len(wanted) * 4,
                    )
                ),
            )
            temporal.release()
        if self.ocr_model_path is not None and frame_paths:
            ScreenTextIndexer(
                self.ocr_model_path,
                device=self.device_request,
            ).build_index(
                workspace,
                frame_paths,
                timestamps,
                sample_seconds,
                progress=(
                    None
                    if progress is None
                    else lambda done, total: progress(
                        len(wanted) * 2 + done,
                        len(wanted) * 4,
                    )
                ),
            )
        audio_matrix = np.empty((0, 0), dtype=np.float16)
        if self.audio_model_path is not None and timestamps:
            audio_matrix = AudioSearchEngine(
                self.audio_model_path,
                device=self.device_request,
            ).build_embeddings(
                video_path,
                np.asarray(timestamps, dtype=np.float32),
                workspace / "visual-index",
                progress=(
                    None
                    if progress is None
                    else lambda done, total: progress(
                        len(wanted) * 3 + done,
                        len(wanted) * 4,
                    )
                ),
            )
        output = workspace / "visual-index" / "index.npz"
        np.savez_compressed(
            output,
            index_version=np.asarray([_INDEX_VERSION], dtype=np.int16),
            embeddings=matrix,
            temporal_embeddings=temporal_matrix,
            audio_embeddings=audio_matrix,
            visual_attributes=(
                np.stack(attribute_rows).astype(np.float16)
                if attribute_rows
                else np.empty((0, 5), dtype=np.float16)
            ),
            timestamps=np.asarray(timestamps, dtype=np.float32),
            frames=np.asarray(frame_paths),
            video=np.asarray([str(video_path)]),
            duration=np.asarray([duration], dtype=np.float32),
            sample_seconds=np.asarray([sample_seconds], dtype=np.float32),
        )
        return output

    def search(
        self,
        query: str,
        workspaces: list[Path],
        top: int = 12,
        min_score: float = 0.3,
        constraint: ObjectConstraint | None = None,
        constraints: tuple[ObjectConstraint, ...] = (),
        variants: tuple[str, ...] = (),
        temporal_variants: tuple[str, ...] = (),
        facets: tuple[str, ...] = (),
        negative_terms: tuple[str, ...] = (),
        open_object_terms: tuple[str, ...] = (),
        attribute_preferences: tuple[tuple[str, float], ...] = (),
        use_audio: bool = False,
        use_temporal: bool = False,
    ) -> list[VisualHit]:
        base_constraints = constraints or (
            (constraint,) if constraint is not None else ()
        )
        translation_inputs = list(
            dict.fromkeys(
                value
                for value in (
                    query,
                    *facets,
                    *open_object_terms,
                    *negative_terms,
                )
                if value
            )
        )
        translation_map = {value: value for value in translation_inputs}
        if self.translation_model_path is not None and any(
            contains_hangul(value) for value in translation_inputs
        ):
            translator = SceneQueryTranslator(
                self.translation_model_path,
                device=self.device_request,
            )
            translated_values = translator.translate_many(translation_inputs)
            translation_map.update(
                zip(
                    translation_inputs,
                    translated_values,
                    strict=True,
                )
            )
            translator.release()
        translated = translation_map.get(query, query)
        dynamic_constraints = tuple(
            ObjectConstraint(
                display_name=term,
                detector_label=translation_map.get(term, term),
                plural_label=translation_map.get(term, term),
                unit_name="개",
            )
            for term in open_object_terms
            if term
        )
        effective_constraints = (*base_constraints, *dynamic_constraints)

        recall_texts = list(
            dict.fromkeys(
                value
                for value in (
                    query,
                    translated,
                    *variants,
                    *(translation_map.get(term, term) for term in open_object_terms),
                )
                if value
            )
        )
        specific_texts = list(
            dict.fromkeys(
                value
                for value in (
                    query,
                    translated,
                )
                if value
            )
        )
        facet_texts = list(
            dict.fromkeys(
                value
                for facet in facets
                if facet and facet != query
                for value in (facet, translation_map.get(facet, facet))
                if value and value not in specific_texts
            )
        )
        negative_texts = list(
            dict.fromkeys(
                value
                for term in negative_terms
                for value in (term, translation_map.get(term, term))
                if value
            )
        )
        recall_vectors = self._text_embeddings(recall_texts)
        specific_vectors = self._text_embeddings(specific_texts)
        facet_vectors = (
            self._text_embeddings(facet_texts)
            if facet_texts
            else None
        )
        negative_vectors = (
            self._text_embeddings(negative_texts)
            if negative_texts
            else None
        )
        self._release_model()
        temporal_texts = list(
            dict.fromkeys(
                value
                for value in (translated, *temporal_variants)
                if use_temporal
                and value
                and not contains_hangul(value)
            )
        )
        temporal_vectors = None
        audio_texts = list(
            dict.fromkeys(
                value
                for value in (
                    translated,
                    *(
                        translation_map.get(facet, facet)
                        for facet in facets
                    ),
                )
                if value and not contains_hangul(value)
            )
        )
        audio_vectors = None
        audio_engine = (
            AudioSearchEngine(
                self.audio_model_path,
                device=self.device_request,
            )
            if self.audio_model_path is not None
            and audio_texts
            and use_audio
            else None
        )
        hits: list[VisualHit] = []
        valid_index_seen = False
        outdated_index_seen = False
        for workspace in workspaces:
            index_path = workspace / "visual-index" / "index.npz"
            if not index_path.is_file():
                continue
            data = np.load(index_path, allow_pickle=False)
            if (
                "index_version" not in data.files
                or int(data["index_version"][0]) != _INDEX_VERSION
            ):
                outdated_index_seen = True
                continue
            valid_index_seen = True
            embeddings = data["embeddings"].astype(np.float32)
            if (
                embeddings.size == 0
                or embeddings.shape[1] != recall_vectors.shape[1]
            ):
                continue
            recall_cosine = embeddings @ recall_vectors.T
            recall_scores = np.max(
                np.column_stack(
                    [
                        self._relevance_scores(recall_cosine[:, column])
                        for column in range(recall_cosine.shape[1])
                    ]
                ),
                axis=1,
            )
            specific_cosine = embeddings @ specific_vectors.T
            semantic_scores = np.max(
                np.column_stack(
                    [
                        self._relevance_scores(specific_cosine[:, column])
                        for column in range(specific_cosine.shape[1])
                    ]
                ),
                axis=1,
            )
            if facet_vectors is not None:
                facet_cosine = embeddings @ facet_vectors.T
                facet_matrix = np.column_stack(
                    [
                        self._relevance_scores(facet_cosine[:, column])
                        for column in range(facet_cosine.shape[1])
                    ]
                )
                facet_scores = (
                    np.mean(facet_matrix, axis=1) * 0.7
                    + np.min(facet_matrix, axis=1) * 0.3
                )
                semantic_scores = (
                    semantic_scores * 0.65
                    + facet_scores * 0.35
            )
            ranking_scores = semantic_scores.copy()
            temporal_used = False
            motion_scores = None
            if (
                self.temporal_model_path is not None
                and temporal_texts
                and "temporal_embeddings" in data.files
            ):
                temporal_embeddings = data[
                    "temporal_embeddings"
                ].astype(np.float32)
                if (
                    temporal_embeddings.ndim == 2
                    and len(temporal_embeddings) == len(ranking_scores)
                    and temporal_embeddings.size
                ):
                    if temporal_vectors is None:
                        temporal_engine = TemporalSearchEngine(
                            self.temporal_model_path,
                            device=self.device_request,
                        )
                        temporal_vectors = temporal_engine.text_embeddings(
                            temporal_texts
                        )
                        temporal_engine.release()
                    temporal_cosine = temporal_embeddings @ temporal_vectors.T
                    motion_scores = temporal_relevance(
                        np.max(temporal_cosine, axis=1)
                    )
                    ranking_scores = (
                        semantic_scores * 0.52
                        + motion_scores * 0.48
                    )
                    temporal_used = True
            audio_used = False
            audio_scores = None
            if (
                audio_engine is not None
                and "audio_embeddings" in data.files
            ):
                audio_embeddings = data["audio_embeddings"].astype(np.float32)
                if (
                    audio_embeddings.ndim == 2
                    and len(audio_embeddings) == len(ranking_scores)
                    and audio_embeddings.size
                ):
                    if audio_vectors is None:
                        audio_vectors = audio_engine.text_embeddings(audio_texts)
                    audio_cosine = audio_embeddings @ audio_vectors.T
                    audio_scores = audio_relevance(
                        np.max(audio_cosine, axis=1)
                    )
                    ranking_scores = (
                        ranking_scores * 0.82
                        + audio_scores * 0.18
                    )
                    audio_used = True
            attribute_used = False
            attribute_scores = None
            if (
                attribute_preferences
                and "visual_attributes" in data.files
            ):
                attributes = data["visual_attributes"].astype(np.float32)
                if (
                    attributes.ndim == 2
                    and len(attributes) == len(ranking_scores)
                    and attributes.size
                ):
                    attribute_scores = _attribute_relevance(
                        attributes,
                        attribute_preferences,
                    )
                    ranking_scores = (
                        ranking_scores * 0.86
                        + attribute_scores * 0.14
                    )
                    attribute_used = True
            if negative_vectors is not None:
                negative_cosine = embeddings @ negative_vectors.T
                negative_scores = np.max(
                    np.column_stack(
                        [
                            self._relevance_scores(
                                negative_cosine[:, column]
                            )
                            for column in range(
                                negative_cosine.shape[1]
                            )
                        ]
                    ),
                    axis=1,
                )
                ranking_scores = np.clip(
                    ranking_scores - negative_scores * 0.28,
                    0.0,
                    1.0,
                )
            candidate_scores = np.maximum(
                ranking_scores,
                recall_scores * 0.82,
            )
            if not effective_constraints:
                eligible = np.flatnonzero(ranking_scores >= min_score)
                count = min(max(top * 3, top), len(eligible))
                sort_scores = ranking_scores
            else:
                eligible = np.arange(len(ranking_scores))
                count = min(max(top * 8, 48), len(eligible))
                sort_scores = candidate_scores
            if not count:
                continue
            indices = eligible[
                np.argsort(sort_scores[eligible])[-count:]
            ]
            duration = (
                float(data["duration"][0])
                if "duration" in data.files
                else float(
                    json.loads(
                        (workspace / "manifest.json").read_text(
                            encoding="utf-8"
                        )
                    ).get("duration")
                    or 0.0
                )
            )
            sample_seconds = (
                float(data["sample_seconds"][0])
                if "sample_seconds" in data.files
                else 1.0
            )
            boundaries = _scene_boundaries(workspace, duration)
            for index in indices:
                timestamp = float(data["timestamps"][index])
                start, end = _shot_span(
                    timestamp,
                    boundaries,
                    sample_seconds,
                )
                score_evidence = [
                    f"화면 {semantic_scores[index] * 100:.0f}%"
                ]
                if temporal_used and motion_scores is not None:
                    score_evidence.append(
                        f"동작 {motion_scores[index] * 100:.0f}%"
                    )
                if audio_used and audio_scores is not None:
                    score_evidence.append(
                        f"오디오 {audio_scores[index] * 100:.0f}%"
                    )
                if attribute_used and attribute_scores is not None:
                    score_evidence.append(
                        f"톤 {attribute_scores[index] * 100:.0f}%"
                    )
                hits.append(
                    VisualHit(
                        workspace=workspace.name,
                        video=str(data["video"][0]),
                        start=start,
                        end=end,
                        score=float(ranking_scores[index]),
                        frame=str(data["frames"][index]),
                        source=(
                            "복합 장면 분석"
                            if temporal_used or audio_used or attribute_used
                            else "화면 의미"
                        ),
                        evidence=" · ".join(score_evidence),
                        vector=embeddings[index],
                    )
                )
        if outdated_index_seen and not valid_index_seen:
            self._release_model()
            if audio_engine is not None:
                audio_engine.release()
            raise RuntimeError(
                "검색 정확도 엔진이 업그레이드되었습니다. "
                "영상 추가 및 분석에서 기존 영상을 다시 선택해 "
                "한 번 재분석해 주세요."
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        self._release_model()
        if audio_engine is not None:
            audio_engine.release()
        if effective_constraints:
            if self.object_model_path is None:
                raise RuntimeError("객체·인원 검색 모델 경로가 없습니다.")
            detector = ObjectSearchEngine(
                self.object_model_path,
                vision_model_path=self.model_path,
                device=self.device_request,
            )
            verified: list[VisualHit] = []
            maximum_checks = max(top * 8, 48)
            for hit in hits[:maximum_checks]:
                detected = detector.inspect_constraints(
                    hit.frame,
                    effective_constraints,
                )
                observations = list(
                    zip(effective_constraints, detected, strict=True)
                )
                soft_matches: list[str] = []
                failed = False
                for item, observation in observations:
                    if (
                        item.excluded
                        and observation.semantic_confidence >= 0.68
                    ):
                        failed = True
                        break
                    if item.matches(observation.count):
                        continue
                    semantic_fallback = (
                        item.person_kind is None
                        and not item.excluded
                        and item.count is None
                        and hit.score >= 0.26
                        and observation.semantic_confidence >= 0.62
                    )
                    if semantic_fallback:
                        soft_matches.append(item.display_name)
                    else:
                        failed = True
                        break
                if failed:
                    continue
                if not exact_person_total_matches(
                    effective_constraints,
                    detected,
                ):
                    continue
                confirmed_confidences = [
                    observation.confidence
                    for item, observation in observations
                    if item.display_name not in soft_matches
                ]
                if confirmed_confidences:
                    detection_confidence = float(
                        np.mean(confirmed_confidences)
                    )
                    hit.score = min(
                        0.99,
                        max(0.0, hit.score) * 0.52
                        + min(
                            1.0,
                            detection_confidence / 0.65,
                        )
                        * 0.48,
                    )
                else:
                    hit.score *= 0.88
                hit.source = "객체·인원 확인"
                object_evidence = [
                    (
                        f"{item.display_name} 화면 의미 "
                        f"{observation.semantic_confidence * 100:.0f}%"
                        if item.display_name in soft_matches
                        else item.evidence(observation.count)
                    )
                    for item, observation in observations
                ]
                hit.evidence = " · ".join(
                    [hit.evidence, *object_evidence]
                ).strip(" ·")
                person_total = next(
                    (
                        observation.total_people
                        for item, observation in observations
                        if item.person_kind is not None
                        and observation.total_people is not None
                    ),
                    None,
                )
                if person_total is not None:
                    hit.evidence += f" · 전체 인물 {person_total}명 확인"
                verified.append(hit)
                if len(verified) >= top * 2:
                    break
            hits = sorted(verified, key=lambda item: item.score, reverse=True)
            detector.release()

        deduped: list[VisualHit] = []
        for hit in hits:
            duplicate = False
            for old in deduped:
                if old.workspace != hit.workspace:
                    continue
                if (
                    abs(old.start - hit.start) < 0.01
                    and abs(old.end - hit.end) < 0.01
                ):
                    duplicate = True
                    break
                if abs(old.start - hit.start) > 8.0:
                    continue
                if old.vector is None or hit.vector is None:
                    continue
                similarity = float(np.dot(old.vector, hit.vector))
                if similarity >= 0.92:
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(hit)
            if len(deduped) >= top:
                break
        return deduped
