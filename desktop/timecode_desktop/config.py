from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

APP_NAME = "Timecode Agent"
ASR_MODELS = {
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}
VISION_MODEL = "google/siglip2-base-patch16-224"
OBJECT_MODEL = "IDEA-Research/grounding-dino-tiny"
TEMPORAL_MODEL = "microsoft/xclip-base-patch32"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ko-en"
OCR_MODEL = "EasyOCR Korean/English"


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "TimecodeAgent"
    return Path.home() / ".timecode-agent"


@dataclass
class AppConfig:
    library_dir: str = str(app_data_dir() / "library")
    model_dir: str = str(app_data_dir() / "models")
    language: str = "ko"
    whisper_model: str = "medium"
    visual_sample_seconds: float = 2.0

    @property
    def path(self) -> Path:
        return app_data_dir() / "settings.json"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "AppConfig":
        path = app_data_dir() / "settings.json"
        if not path.is_file():
            config = cls()
            config.save()
            return config
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            known = {
                key: value
                for key, value in values.items()
                if key in cls.__dataclass_fields__
            }
            return cls(**known)
        except (OSError, TypeError, ValueError):
            return cls()

    def ensure_dirs(self) -> None:
        Path(self.library_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
