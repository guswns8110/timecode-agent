from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .config import AUDIO_MODEL

_SAMPLE_RATE = 48_000
_WINDOW_SECONDS = 8.0
_INDEX_STRIDE_SECONDS = 2.0


def _pooled_features(output):
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    return output


def _batched(values: list, size: int) -> Iterable[list]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def audio_relevance(cosine_scores: np.ndarray) -> np.ndarray:
    return np.clip((cosine_scores - 0.05) / 0.3, 0.0, 1.0)


class AudioSearchEngine:
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
            else AUDIO_MODEL
        )
        self._processor = AutoProcessor.from_pretrained(source)
        self._model = AutoModel.from_pretrained(source).to(self._device).eval()

    @staticmethod
    def _extract_wav(video_path: Path, destination: Path) -> bool:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            check=False,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return completed.returncode == 0 and destination.stat().st_size > 44

    @staticmethod
    def _clips_from_wav(
        source: wave.Wave_read,
        timestamps: list[float],
    ) -> list[np.ndarray]:
        clips: list[np.ndarray] = []
        wanted_frames = int(_WINDOW_SECONDS * _SAMPLE_RATE)
        total_frames = source.getnframes()
        for timestamp in timestamps:
            start_seconds = max(
                0.0,
                float(timestamp) - _WINDOW_SECONDS / 2,
            )
            start_frame = min(
                total_frames,
                int(start_seconds * _SAMPLE_RATE),
            )
            source.setpos(start_frame)
            raw = source.readframes(wanted_frames)
            samples = (
                np.frombuffer(raw, dtype="<i2").astype(np.float32)
                / 32768.0
            )
            clips.append(samples)
        return clips

    def build_embeddings(
        self,
        video_path: Path,
        timestamps: np.ndarray,
        temporary_dir: Path,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        total = len(timestamps)
        if not total:
            return np.empty((0, 0), dtype=np.float16)
        selected_indices = [0]
        last_timestamp = float(timestamps[0])
        for index, timestamp in enumerate(timestamps[1:], start=1):
            if float(timestamp) - last_timestamp < _INDEX_STRIDE_SECONDS:
                continue
            selected_indices.append(index)
            last_timestamp = float(timestamp)
        selected_timestamps = timestamps[selected_indices]
        descriptor, temporary_name = tempfile.mkstemp(
            suffix=".wav",
            prefix="tca-audio-",
            dir=temporary_dir,
        )
        os.close(descriptor)
        wav_path = Path(temporary_name)
        try:
            if not self._extract_wav(video_path, wav_path):
                if progress:
                    progress(total, total)
                return np.empty((0, 0), dtype=np.float16)
            self._load()
            assert self._processor is not None
            assert self._model is not None
            import torch

            features: list[np.ndarray] = []
            batch_size = 8 if self._device == "cuda" else 2
            completed = 0
            timestamp_values = [
                float(value)
                for value in selected_timestamps
            ]
            with wave.open(str(wav_path), "rb") as source:
                for timestamp_batch in _batched(
                    timestamp_values,
                    batch_size,
                ):
                    batch = self._clips_from_wav(
                        source,
                        timestamp_batch,
                    )
                    inputs = self._processor(
                        audios=batch,
                        sampling_rate=_SAMPLE_RATE,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {
                        key: value.to(self._device)
                        for key, value in inputs.items()
                    }
                    with torch.inference_mode():
                        output = _pooled_features(
                            self._model.get_audio_features(**inputs)
                        )
                        output = output / output.norm(
                            dim=-1,
                            keepdim=True,
                        )
                    features.append(output.float().cpu().numpy())
                    completed += len(batch)
                    if progress:
                        progress(
                            min(
                                total,
                                round(
                                    completed
                                    / max(1, len(selected_timestamps))
                                    * total
                                ),
                            ),
                            total,
                        )
            selected_matrix = np.concatenate(features).astype(np.float16)
            nearest = np.searchsorted(
                selected_timestamps,
                timestamps,
                side="left",
            )
            nearest = np.clip(nearest, 0, len(selected_timestamps) - 1)
            previous = np.maximum(nearest - 1, 0)
            use_previous = (
                np.abs(timestamps - selected_timestamps[previous])
                <= np.abs(timestamps - selected_timestamps[nearest])
            )
            nearest = np.where(use_previous, previous, nearest)
            return selected_matrix[nearest]
        finally:
            wav_path.unlink(missing_ok=True)
            self.release()

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
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
