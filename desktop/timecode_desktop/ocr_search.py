from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np

from video_agent.fsio import write_text_atomic

_WHITESPACE = re.compile(r"\s+")
LOGGER = logging.getLogger(__name__)


class ScreenTextIndexer:
    """Create a searchable Korean/English screen-text timeline."""

    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = model_path
        self.device_request = device
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr
        import torch

        use_cuda = torch.cuda.is_available() and self.device_request != "cpu"
        self._reader = easyocr.Reader(
            ["ko", "en"],
            gpu=use_cuda,
            model_storage_directory=str(self.model_path),
            user_network_directory=str(self.model_path),
            download_enabled=False,
            verbose=False,
        )

    @staticmethod
    def _line_key(item: tuple) -> tuple[float, float]:
        box = item[0]
        x = min(float(point[0]) for point in box)
        y = min(float(point[1]) for point in box)
        return (round(y / 24.0), x)

    def scan(
        self,
        images: list,
        timestamps: list[float],
        sample_seconds: float,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        self._load()
        assert self._reader is not None
        entries: list[dict] = []
        failures = 0
        total = len(images)
        for index, (image_source, timestamp) in enumerate(
            zip(images, timestamps, strict=True),
            start=1,
        ):
            try:
                if isinstance(image_source, (str, Path)):
                    from PIL import Image

                    with Image.open(image_source) as opened:
                        pixels = np.asarray(opened.convert("RGB"))
                else:
                    pixels = np.asarray(image_source)
                results = self._reader.readtext(
                    pixels,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                    batch_size=1,
                )
            except Exception:
                failures += 1
                LOGGER.warning(
                    "OCR 프레임 분석 실패: %.3fs",
                    timestamp,
                    exc_info=True,
                )
                if progress:
                    progress(index, total)
                continue
            lines = []
            for result in sorted(results, key=self._line_key):
                if len(result) < 3 or float(result[2]) < 0.25:
                    continue
                text = _WHITESPACE.sub(" ", str(result[1])).strip()
                if text:
                    lines.append(text)
            if lines:
                text = " ".join(lines)
                compact = text.replace(" ", "")
                searchable = (
                    f"{text} {compact}"
                    if compact and compact != text
                    else text
                )
                entries.append(
                    {
                        "start": float(timestamp),
                        "end": float(timestamp + sample_seconds),
                        "text": searchable,
                    }
                )
            if progress:
                progress(index, total)
        if total and failures == total:
            raise RuntimeError("모든 프레임의 화면 글자 OCR 분석에 실패했습니다.")
        return entries

    def build_index(
        self,
        workspace: Path,
        images: list,
        timestamps: list[float],
        sample_seconds: float,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        try:
            entries = self.scan(
                images,
                timestamps,
                sample_seconds,
                progress,
            )
            output = workspace / "ocr_transcript.json"
            write_text_atomic(
                output,
                json.dumps(entries, ensure_ascii=False, indent=2),
            )
            return output
        finally:
            self.release()

    def release(self) -> None:
        self._reader = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
