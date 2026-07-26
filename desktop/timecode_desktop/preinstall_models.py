from __future__ import annotations

import sys
from pathlib import Path

from .config import AppConfig
from .model_manager import ModelManager


def main() -> int:
    config = AppConfig.load()
    config.ensure_dirs()
    manager = ModelManager(Path(config.model_dir))

    def report(name: str, done: int, total: int) -> None:
        state = "완료" if done == total else "다운로드 중"
        print(f"[{name}] {state}", flush=True)

    try:
        manager.install_all(report)
    except Exception as exc:
        print(f"모델 설치 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
