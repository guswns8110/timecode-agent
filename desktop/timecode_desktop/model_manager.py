from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import (
    ASR_MODELS,
    OBJECT_MODEL,
    OCR_MODEL,
    TEMPORAL_MODEL,
    TRANSLATION_MODEL,
    VISION_MODEL,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class ModelState:
    name: str
    repo_id: str
    path: Path
    installed: bool


class ModelManager:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def states(self) -> list[ModelState]:
        items = [
            *ASR_MODELS.items(),
            ("vision", VISION_MODEL),
            ("objects", OBJECT_MODEL),
            ("temporal", TEMPORAL_MODEL),
            ("translation", TRANSLATION_MODEL),
            ("ocr", OCR_MODEL),
        ]
        return [
            ModelState(
                name=name,
                repo_id=repo,
                path=self.path_for(name),
                installed=self.is_installed(name),
            )
            for name, repo in items
        ]

    def path_for(self, name: str) -> Path:
        return self.model_dir / name

    def is_installed(self, name: str) -> bool:
        path = self.path_for(name)
        return path.is_dir() and (path / ".complete").is_file()

    def install_all(self, progress: ProgressCallback | None = None) -> None:
        for name, repo in [
            *ASR_MODELS.items(),
            ("vision", VISION_MODEL),
            ("objects", OBJECT_MODEL),
            ("temporal", TEMPORAL_MODEL),
            ("translation", TRANSLATION_MODEL),
            ("ocr", OCR_MODEL),
        ]:
            self.install(name, repo, progress)

    def install(
        self,
        name: str,
        repo_id: str,
        progress: ProgressCallback | None = None,
    ) -> Path:
        if name == "ocr":
            return self._install_ocr(progress)

        from huggingface_hub import snapshot_download

        destination = self.path_for(name)
        destination.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(name, 0, 1)
        allow_patterns = [
            "*.json",
            "*.model",
            "*.spm",
            "*.txt",
            "*.safetensors",
        ]
        if name in ASR_MODELS or name == "translation":
            allow_patterns.append("*.bin")
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        (destination / ".complete").write_text(repo_id, encoding="utf-8")
        if progress:
            progress(name, 1, 1)
        return destination

    def _install_ocr(
        self,
        progress: ProgressCallback | None = None,
    ) -> Path:
        import easyocr

        destination = self.path_for("ocr")
        destination.mkdir(parents=True, exist_ok=True)
        if progress:
            progress("ocr", 0, 1)
        easyocr.Reader(
            ["ko", "en"],
            gpu=False,
            model_storage_directory=str(destination),
            user_network_directory=str(destination),
            verbose=False,
        )
        (destination / ".complete").write_text(
            OCR_MODEL,
            encoding="utf-8",
        )
        if progress:
            progress("ocr", 1, 1)
        return destination
