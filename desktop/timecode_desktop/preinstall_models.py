from __future__ import annotations

import sys
from pathlib import Path

from .config import AppConfig
from .model_manager import ModelManager

MODEL_LABELS = {
    "medium": "Medium 음성",
    "large-v3": "Large-v3 음성",
    "vision": "화면 의미",
    "objects": "객체·인원",
    "temporal": "장면·동작",
    "translation": "한국어 자연어",
    "ocr": "화면 글자 OCR",
    "audio": "오디오 분위기",
}


def main() -> int:
    config = AppConfig.load()
    config.ensure_dirs()
    manager = ModelManager(Path(config.model_dir))

    def report(name: str, done: int, total: int) -> None:
        state = "완료" if done == total else "다운로드 중"
        print(f"[{MODEL_LABELS.get(name, name)}] {state}", flush=True)

    try:
        manager.install_all(report)
    except Exception as exc:
        print(f"모델 설치 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
