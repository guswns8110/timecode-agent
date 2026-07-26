from __future__ import annotations

import re
from pathlib import Path

from .config import TRANSLATION_MODEL

_HANGUL = re.compile(r"[가-힣]")


def contains_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text))


class SceneQueryTranslator:
    """Translate arbitrary Korean scene descriptions for video-text models."""

    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = model_path
        self.device_request = device
        self._model = None
        self._tokenizer = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        use_cuda = torch.cuda.is_available() and self.device_request != "cpu"
        self._device = "cuda" if use_cuda else "cpu"
        source = (
            str(self.model_path)
            if (self.model_path / ".complete").is_file()
            else TRANSLATION_MODEL
        )
        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(source)
            .to(self._device)
            .eval()
        )

    def translate(self, text: str) -> str:
        if not contains_hangul(text):
            return text.strip()

        import torch

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        inputs = self._tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=160,
        )
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=3,
                early_stopping=True,
            )
        translated = self._tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        return translated or text.strip()

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
