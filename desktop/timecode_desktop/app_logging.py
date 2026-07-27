from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import app_data_dir

_CONFIGURED = False


def app_log_path() -> Path:
    return app_data_dir() / "logs" / "app.log"


def configure_logging() -> Path:
    global _CONFIGURED

    path = app_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if _CONFIGURED:
        return path

    handler = RotatingFileHandler(
        path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.captureWarnings(True)

    previous_hook = sys.excepthook

    def log_unhandled(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback,
    ) -> None:
        logging.getLogger("timecode_desktop").critical(
            "처리되지 않은 오류",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        previous_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_unhandled
    _CONFIGURED = True
    logging.getLogger(__name__).info("앱 로그 시작: %s", path)
    return path
