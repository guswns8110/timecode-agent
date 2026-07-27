from __future__ import annotations

import io
import logging
import os
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .app_logging import app_log_path
from .config import AppConfig
from .diagnostics import inspect_environment
from .library_status import in_progress_marker
from .model_manager import ModelManager
from .search_query import SearchMode
from .search_service import UnifiedSearch
from .visual_search import VisualSearchEngine

LOGGER = logging.getLogger(__name__)
MODEL_LABELS = {
    "medium": "Medium 음성 모델",
    "large-v3": "Large-v3 음성 모델",
    "vision": "화면 의미 모델",
    "objects": "객체·인원 확인 모델",
    "temporal": "장면·동작 분석 모델",
    "translation": "한국어 자연어 모델",
    "ocr": "화면 글자 OCR 모델",
    "audio": "오디오 분위기 모델",
}


class ModelInstallWorker(QObject):
    progress = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, manager: ModelManager):
        super().__init__()
        self.manager = manager

    @Slot()
    def run(self) -> None:
        try:
            self.manager.install_all(
                lambda name, done, total: self.progress.emit(
                    f"{MODEL_LABELS.get(name, name)} 설치 중 ({done}/{total})"
                )
            )
            self.finished.emit()
        except Exception as exc:
            LOGGER.exception("모델 설치 실패")
            self.failed.emit(str(exc))


class AnalyzeWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, video: Path, config: AppConfig):
        super().__init__()
        self.video = video
        self.config = config

    @Slot()
    def run(self) -> None:
        buffer = io.StringIO()
        try:
            self.progress.emit(f"{self.video.name} · 분석 준비 중", 2)
            from video_agent.ingest import ingest

            os.environ["VIDEO_AGENT_ASR_DEVICE"] = "auto"
            self.progress.emit("영상 정보 및 대사 분석 중", 10)
            output = Path(self.config.library_dir) / self.video.stem
            output.mkdir(parents=True, exist_ok=True)
            marker = in_progress_marker(output)
            marker.touch(exist_ok=True)
            started = time.monotonic()
            with redirect_stdout(buffer), redirect_stderr(buffer):
                workspace = ingest(
                    self.video,
                    out=output,
                    model=str(Path(self.config.model_dir) / self.config.whisper_model),
                    lang=self.config.language,
                    force_whisper=True,
                    signals=True,
                )
            self.progress.emit(
                "화면·동작·오디오 분위기·화면 글자 인덱스 생성 중",
                75,
            )
            engine = VisualSearchEngine(
                model_path=Path(self.config.model_dir) / "vision",
                object_model_path=Path(self.config.model_dir) / "objects",
                temporal_model_path=Path(self.config.model_dir) / "temporal",
                translation_model_path=(
                    Path(self.config.model_dir) / "translation"
                ),
                ocr_model_path=Path(self.config.model_dir) / "ocr",
                audio_model_path=Path(self.config.model_dir) / "audio",
            )
            engine.build_index(
                workspace.root,
                sample_seconds=self.config.visual_sample_seconds,
                progress=lambda done, total: self.progress.emit(
                    "화면·동작·오디오 분위기·화면 글자 인덱스 생성 중",
                    75 + int(24 * done / max(total, 1)),
                ),
            )
            marker.unlink(missing_ok=True)
            elapsed = time.monotonic() - started
            self.progress.emit("분석 완료", 100)
            self.finished.emit(f"{workspace.root}\n완료 시간: {elapsed / 60:.1f}분")
        except Exception as exc:
            LOGGER.exception("영상 분석 실패: %s", self.video)
            captured = buffer.getvalue().strip()
            detail = str(exc) or exc.__class__.__name__
            if captured:
                detail = f"{detail}\n\n분석 출력:\n{captured[-4000:]}"
            else:
                detail = f"{detail}\n\n{traceback.format_exc()[-4000:]}"
            detail = f"{detail}\n\n상세 로그: {app_log_path()}"
            self.failed.emit(detail)


class SearchWorker(QObject):
    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        query: str,
        config: AppConfig,
        mode: SearchMode = "auto",
    ):
        super().__init__()
        self.query = query
        self.config = config
        self.mode = mode

    @Slot()
    def run(self) -> None:
        try:
            service = UnifiedSearch(
                library=Path(self.config.library_dir),
                vision_model=Path(self.config.model_dir) / "vision",
                object_model=Path(self.config.model_dir) / "objects",
                temporal_model=Path(self.config.model_dir) / "temporal",
                translation_model=(
                    Path(self.config.model_dir) / "translation"
                ),
                audio_model=Path(self.config.model_dir) / "audio",
            )
            self.finished.emit(
                service.as_dicts(
                    service.run(self.query, mode=self.mode)
                )
            )
        except Exception as exc:
            LOGGER.exception("검색 실패: %s", self.query)
            self.failed.emit(str(exc))


class DiagnosticsWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, storage_path: Path):
        super().__init__()
        self.storage_path = storage_path

    @Slot()
    def run(self) -> None:
        try:
            report = inspect_environment(self.storage_path)
            self.finished.emit(report.to_dict())
        except Exception as exc:
            LOGGER.exception("환경 진단 실패")
            self.failed.emit(str(exc))
