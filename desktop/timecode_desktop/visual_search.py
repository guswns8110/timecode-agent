from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import VISION_MODEL
from .object_search import ObjectSearchEngine
from .ocr_search import ScreenTextIndexer
from .search_query import ObjectConstraint
from .temporal_search import TemporalSearchEngine, temporal_relevance
from .translation_search import SceneQueryTranslator, contains_hangul


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


class VisualSearchEngine:
    def __init__(
        self,
        model_path: Path,
        object_model_path: Path | None = None,
        temporal_model_path: Path | None = None,
        translation_model_path: Path | None = None,
        ocr_model_path: Path | None = None,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.object_model_path = object_model_path
        self.temporal_model_path = temporal_model_path
        self.translation_model_path = translation_model_path
        self.ocr_model_path = ocr_model_path
        self.device_request = device
        self._model = None
        self._processor = None
        self._device = "cpu"

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
        assert self._model is not None
        scale = getattr(self._model, "logit_scale", None)
        bias = getattr(self._model, "logit_bias", None)
        if scale is None or bias is None:
            return np.clip((cosine_scores - 0.15) / 0.2, 0.0, 1.0)
        logit_scale = float(scale.detach().float().cpu().item())
        logit_bias = float(bias.detach().float().cpu().item())
        return _sigmoid_relevance(
            cosine_scores,
            logit_scale,
            logit_bias,
        )

    def build_index(
        self,
        workspace: Path,
        sample_seconds: float = 2.0,
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
        images: list[Image.Image] = []
        timestamps: list[float] = []
        frame_paths: list[str] = []
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
            images.append(image)
            timestamps.append(timestamp)
            frame_paths.append(str(path))
            next_index += 1
            if progress:
                progress(next_index, len(wanted) * 3)
        container.close()

        embeddings: list[np.ndarray] = []
        self._load()
        batch_size = 16 if self._device == "cuda" else 4
        for batch in _batched(images, batch_size):
            embeddings.append(self._image_embeddings(batch))
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
                        len(wanted) * 3,
                    )
                ),
            )
            temporal.release()
        if self.ocr_model_path is not None and images:
            ScreenTextIndexer(
                self.ocr_model_path,
                device=self.device_request,
            ).build_index(
                workspace,
                images,
                timestamps,
                sample_seconds,
                progress=(
                    None
                    if progress is None
                    else lambda done, total: progress(
                        len(wanted) * 2 + done,
                        len(wanted) * 3,
                    )
                ),
            )
        output = workspace / "visual-index" / "index.npz"
        np.savez_compressed(
            output,
            index_version=np.asarray([2], dtype=np.int16),
            embeddings=matrix,
            temporal_embeddings=temporal_matrix,
            timestamps=np.asarray(timestamps, dtype=np.float32),
            frames=np.asarray(frame_paths),
            video=np.asarray([str(video_path)]),
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
    ) -> list[VisualHit]:
        effective_constraints = constraints or (
            (constraint,) if constraint is not None else ()
        )
        translated = ""
        if (
            self.translation_model_path is not None
            and contains_hangul(query)
        ):
            translator = SceneQueryTranslator(
                self.translation_model_path,
                device=self.device_request,
            )
            translated = translator.translate(query)
            translator.release()

        recall_texts = list(
            dict.fromkeys(
                value
                for value in (query, translated, *variants)
                if value
            )
        )
        specific_texts = list(
            dict.fromkeys(
                value
                for value in (
                    query,
                    translated,
                    *temporal_variants,
                )
                if value
            )
        )
        recall_vectors = self._text_embeddings(recall_texts)
        specific_vectors = self._text_embeddings(specific_texts)
        temporal_texts = list(
            dict.fromkeys(
                value
                for value in (translated, *temporal_variants)
                if value and not contains_hangul(value)
            )
        )
        temporal_vectors = None
        hits: list[VisualHit] = []
        for workspace in workspaces:
            index_path = workspace / "visual-index" / "index.npz"
            if not index_path.is_file():
                continue
            data = np.load(index_path, allow_pickle=False)
            embeddings = data["embeddings"].astype(np.float32)
            if embeddings.size == 0:
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
            ranking_scores = semantic_scores.copy()
            temporal_used = False
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
            for index in indices:
                timestamp = float(data["timestamps"][index])
                hits.append(
                    VisualHit(
                        workspace=workspace.name,
                        video=str(data["video"][0]),
                        start=timestamp,
                        end=timestamp + 2.0,
                        score=float(ranking_scores[index]),
                        frame=str(data["frames"][index]),
                        source=(
                            "장면·동작 의미"
                            if temporal_used
                            else "화면 의미"
                        ),
                        vector=embeddings[index],
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        self._release_model()
        if effective_constraints:
            if self.object_model_path is None:
                raise RuntimeError("객체·인원 검색 모델 경로가 없습니다.")
            detector = ObjectSearchEngine(
                self.object_model_path,
                device=self.device_request,
            )
            verified: list[VisualHit] = []
            maximum_checks = max(top * 8, 48)
            for hit in hits[:maximum_checks]:
                observations = [
                    (
                        item,
                        detector.inspect(hit.frame, item),
                    )
                    for item in effective_constraints
                ]
                if not all(
                    item.matches(observation.count)
                    for item, observation in observations
                ):
                    continue
                detection_confidence = float(
                    np.mean(
                        [
                            observation.confidence
                            for _, observation in observations
                        ]
                    )
                )
                hit.score = min(
                    0.99,
                    max(0.0, hit.score) * 0.48
                    + min(1.0, detection_confidence / 0.65) * 0.52,
                )
                hit.source = "객체·인원 확인"
                hit.evidence = " · ".join(
                    item.evidence(observation.count)
                    for item, observation in observations
                )
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
