from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import VISION_MODEL


@dataclass
class VisualHit:
    workspace: str
    video: str
    start: float
    end: float
    score: float
    frame: str


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
        source = str(self.model_path) if self.model_path.is_dir() else VISION_MODEL
        self._processor = AutoProcessor.from_pretrained(source)
        self._model = AutoModel.from_pretrained(source).to(self._device).eval()

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

    def _text_embedding(self, text: str) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = _pooled_features(
                self._model.get_text_features(**inputs)
            )
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()[0]

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
                progress(next_index, len(wanted))
        container.close()

        embeddings: list[np.ndarray] = []
        batch_size = 16 if self._device == "cuda" else 4
        for batch in _batched(images, batch_size):
            embeddings.append(self._image_embeddings(batch))
        matrix = (
            np.concatenate(embeddings).astype(np.float16)
            if embeddings
            else np.empty((0, 0), dtype=np.float16)
        )
        output = workspace / "visual-index" / "index.npz"
        np.savez_compressed(
            output,
            embeddings=matrix,
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
        min_score: float = 0.5,
    ) -> list[VisualHit]:
        query_vector = self._text_embedding(query)
        hits: list[VisualHit] = []
        for workspace in workspaces:
            index_path = workspace / "visual-index" / "index.npz"
            if not index_path.is_file():
                continue
            data = np.load(index_path, allow_pickle=False)
            embeddings = data["embeddings"].astype(np.float32)
            if embeddings.size == 0:
                continue
            cosine_scores = embeddings @ query_vector
            scores = self._relevance_scores(cosine_scores)
            eligible = np.flatnonzero(scores >= min_score)
            if not len(eligible):
                continue
            count = min(top, len(eligible))
            indices = eligible[np.argsort(scores[eligible])[-count:]]
            for index in indices:
                timestamp = float(data["timestamps"][index])
                hits.append(
                    VisualHit(
                        workspace=workspace.name,
                        video=str(data["video"][0]),
                        start=timestamp,
                        end=timestamp + 2.0,
                        score=float(scores[index]),
                        frame=str(data["frames"][index]),
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top]
