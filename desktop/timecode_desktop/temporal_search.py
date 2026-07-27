from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .config import TEMPORAL_MODEL


def _pooled_features(output):
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return output


def _small_frame(image):
    converted = image.convert("RGB")
    converted.thumbnail((384, 384))
    return converted.copy()


def _iter_clips(
    video_path: Path,
    timestamps: np.ndarray,
    clip_seconds: float,
    frames_per_clip: int = 8,
) -> Iterable[list]:
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    starts = [float(value) for value in timestamps]
    current_clip = 0
    current_target = 0
    images: list = []
    last_image = None

    try:
        for frame in container.decode(stream):
            if current_clip >= len(starts):
                break
            timestamp = float(frame.time or 0.0)
            span = max(0.25, clip_seconds * 0.875)
            wanted = starts[current_clip] + (
                span * current_target / max(1, frames_per_clip - 1)
            )
            if timestamp + 0.001 < wanted:
                continue
            last_image = _small_frame(frame.to_image())
            images.append(last_image)
            current_target += 1
            if current_target < frames_per_clip:
                continue
            yield images
            images = []
            current_clip += 1
            current_target = 0
    finally:
        container.close()

    while current_clip < len(starts):
        filler = last_image
        if filler is None:
            from PIL import Image

            filler = Image.new("RGB", (224, 224), "black")
        while len(images) < frames_per_clip:
            images.append(filler.copy())
        yield images
        images = []
        current_clip += 1


class TemporalSearchEngine:
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
        from transformers import AutoModel, AutoProcessor

        use_cuda = torch.cuda.is_available() and self.device_request != "cpu"
        self._device = "cuda" if use_cuda else "cpu"
        source = (
            str(self.model_path)
            if (self.model_path / ".complete").is_file()
            else TEMPORAL_MODEL
        )
        self._processor = AutoProcessor.from_pretrained(source)
        self._model = AutoModel.from_pretrained(source).to(self._device).eval()

    def build_embeddings(
        self,
        video_path: Path,
        timestamps: np.ndarray,
        clip_seconds: float,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None

        features: list[np.ndarray] = []
        total = len(timestamps)
        for index, clip in enumerate(
            _iter_clips(video_path, timestamps, clip_seconds)
        ):
            inputs = self._processor(
                videos=clip,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                output = _pooled_features(
                    self._model.get_video_features(**inputs)
                )
                output = output / output.norm(dim=-1, keepdim=True)
            features.append(output.float().cpu().numpy()[0])
            if progress:
                progress(index + 1, total)
        if not features:
            return np.empty((0, 0), dtype=np.float16)
        return np.stack(features).astype(np.float16)

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output = _pooled_features(
                self._model.get_text_features(**inputs)
            )
            output = output / output.norm(dim=-1, keepdim=True)
        return output.float().cpu().numpy()

    def release(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def temporal_relevance(cosine_scores: np.ndarray) -> np.ndarray:
    return np.clip((cosine_scores - 0.08) / 0.28, 0.0, 1.0)
