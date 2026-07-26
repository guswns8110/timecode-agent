from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .app_logging import app_log_path
from .config import APP_NAME, AppConfig
from .diagnostics import live_usage
from .library_status import completed_workspaces
from .model_manager import ModelManager
from .search_query import SearchMode, parse_search_intent
from .workers import (
    AnalyzeWorker,
    DiagnosticsWorker,
    ModelInstallWorker,
    SearchWorker,
)

LOGGER = logging.getLogger(__name__)


def _timecode(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SeekSlider(QSlider):
    seekRequested = Signal(int)

    def _set_from_x(self, x: float) -> int:
        value = QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            round(x),
            max(1, self.width() - 1),
        )
        self.setValue(value)
        return value

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self._set_from_x(event.position().x())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self.isSliderDown()
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            value = self._set_from_x(event.position().x())
            self.seekRequested.emit(value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            self.isSliderDown()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            value = self._set_from_x(event.position().x())
            self.setSliderDown(False)
            self.seekRequested.emit(value)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SearchResultCard(QFrame):
    def __init__(self, hit: dict, on_open):
        super().__init__()
        self.hit = hit
        self.setObjectName("resultCard")
        layout = QHBoxLayout(self)

        thumbnail = QLabel()
        thumbnail.setFixedSize(112, 64)
        thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if hit.get("thumbnail"):
            pixmap = QPixmap(hit["thumbnail"])
            thumbnail.setPixmap(
                pixmap.scaled(
                    thumbnail.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            thumbnail.setText("대사/OCR")
        layout.addWidget(thumbnail)

        text_layout = QVBoxLayout()
        title = QLabel(f"{Path(hit['video']).name}  ·  {_timecode(hit['start'])}")
        title.setObjectName("resultTitle")
        detail = QLabel(hit.get("text") or hit.get("source", ""))
        detail.setWordWrap(True)
        meta = QLabel(f"{hit.get('source', '')}  ·  관련도 {hit['score'] * 100:.0f}%")
        meta.setObjectName("muted")
        text_layout.addWidget(title)
        text_layout.addWidget(detail)
        text_layout.addWidget(meta)
        layout.addLayout(text_layout, 1)

        open_button = QPushButton("재생")
        open_button.clicked.connect(lambda: on_open(hit))
        layout.addWidget(open_button)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()
        self.config.ensure_dirs()
        self.model_manager = ModelManager(Path(self.config.model_dir))
        self.threads: list[QThread] = []
        self.workers: list[QObject] = []
        self.current_hits: list[dict] = []
        self.pending_video: Path | None = None
        self.pending_search: tuple[str, SearchMode] | None = None
        self.pending_seek_ms: int | None = None
        self.analysis_started_at: float | None = None
        self.analysis_estimated_seconds: float | None = None
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)
        self._setup_player()
        self._setup_ui()
        self._setup_menu()
        self.refresh_library()
        self.run_diagnostics()

        self.usage_timer = QTimer(self)
        self.usage_timer.timeout.connect(self.update_usage)
        self.usage_timer.start(2500)

    def _setup_player(self) -> None:
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.mediaStatusChanged.connect(self._media_status_changed)

    def _setup_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 10)

        top = QHBoxLayout()
        self.search_mode = QComboBox()
        self.search_mode.addItem("자동 검색", "auto")
        self.search_mode.addItem("장면 검색", "scene")
        self.search_mode.addItem("대사 검색", "dialogue")
        self.search_mode.setFixedWidth(110)
        self.search_mode.currentIndexChanged.connect(
            self._search_mode_changed
        )
        self.search_input = QComboBox()
        self.search_input.setEditable(True)
        self.search_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.search_input.lineEdit().setPlaceholderText(
            "예: 비행기가 활주로에서 이륙하는 장면 찾아줘"
        )
        self.search_input.lineEdit().returnPressed.connect(self.start_search)
        self.search_button = QPushButton("검색")
        self.search_button.setObjectName("primary")
        self.search_button.clicked.connect(self.start_search)
        self.add_button = QPushButton("영상 추가 및 분석")
        self.add_button.setObjectName("addButton")
        self.add_button.clicked.connect(self.choose_video)
        top.addWidget(self.search_mode)
        top.addWidget(self.search_input, 1)
        top.addWidget(self.search_button)
        top.addWidget(self.add_button)
        root_layout.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._library_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._results_panel())
        splitter.setSizes([240, 820, 420])
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.usage_label = QLabel("환경 확인 중…")
        self.statusBar().addPermanentWidget(self.usage_label)
        self._search_mode_changed()

    def _search_mode_changed(self, *_args) -> None:
        mode = self.search_mode.currentData()
        placeholders = {
            "auto": "장면·분위기·화면 글자 또는 찾을 대사를 입력하세요",
            "scene": "장르·인물·사물·톤·감성·동작·화면 문구를 자연어로 입력하세요",
            "dialogue": "예: ‘완벽하네’라고 말하는 대사",
        }
        self.search_input.lineEdit().setPlaceholderText(
            placeholders.get(mode, placeholders["auto"])
        )

    def _library_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("영상 라이브러리")
        heading.setObjectName("heading")
        self.library_list = QListWidget()
        self.library_list.itemDoubleClicked.connect(self.open_library_item)
        self.refresh_button = QPushButton("새로고침")
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.clicked.connect(self.refresh_library)
        layout.addWidget(heading)
        layout.addWidget(self.library_list, 1)
        layout.addWidget(self.refresh_button)
        return panel

    def _center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.addWidget(self.video_widget, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("재생")
        self.play_button.setObjectName("playButton")
        self.play_button.clicked.connect(self.toggle_playback)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.timeline = SeekSlider(Qt.Orientation.Horizontal)
        self.timeline.seekRequested.connect(self._seek_to)
        controls.addWidget(self.play_button)
        controls.addWidget(self.timeline, 1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        self.task_label = QLabel("분석할 영상을 추가하세요.")
        self.task_progress = QProgressBar()
        self.task_progress.setValue(0)
        layout.addWidget(self.task_label)
        layout.addWidget(self.task_progress)
        return panel

    def _results_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("검색 결과")
        heading.setObjectName("heading")
        self.results_body = QWidget()
        self.results_layout = QVBoxLayout(self.results_body)
        self.results_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.results_body)
        layout.addWidget(heading)
        layout.addWidget(scroll, 1)
        return panel

    def _setup_menu(self) -> None:
        file_menu = self.menuBar().addMenu("파일")
        add_action = QAction("영상 추가", self)
        add_action.triggered.connect(self.choose_video)
        file_menu.addAction(add_action)
        folder_action = QAction("라이브러리 위치 변경", self)
        folder_action.triggered.connect(self.choose_library)
        file_menu.addAction(folder_action)

        tools_menu = self.menuBar().addMenu("도구")
        model_action = QAction("필수 모델 설치/복구", self)
        model_action.triggered.connect(self.install_models)
        tools_menu.addAction(model_action)
        diagnostic_action = QAction("환경 다시 확인", self)
        diagnostic_action.triggered.connect(self.run_diagnostics)
        tools_menu.addAction(diagnostic_action)

        model_menu = self.menuBar().addMenu("음성 모델")
        medium = QAction("Medium", self, checkable=True)
        large = QAction("Large-v3", self, checkable=True)
        medium.setChecked(self.config.whisper_model == "medium")
        large.setChecked(self.config.whisper_model == "large-v3")
        medium.triggered.connect(lambda: self.set_model("medium", medium, large))
        large.triggered.connect(lambda: self.set_model("large-v3", large, medium))
        model_menu.addAction(medium)
        model_menu.addAction(large)

    def set_model(self, name: str, selected: QAction, other: QAction) -> None:
        selected.setChecked(True)
        other.setChecked(False)
        self.config.whisper_model = name
        self.config.save()
        self.statusBar().showMessage(f"음성 모델: {name}", 3000)

    def choose_library(self) -> None:
        self.statusBar().showMessage("라이브러리로 사용할 폴더를 선택하세요.")
        chosen = QFileDialog.getExistingDirectory(
            self, "영상 라이브러리 폴더 선택", self.config.library_dir
        )
        if chosen:
            self.config.library_dir = chosen
            self.config.save()
            self.config.ensure_dirs()
            self.refresh_library()
            self.run_diagnostics()
        else:
            self.statusBar().showMessage("폴더 선택을 취소했습니다.", 3000)

    def choose_video(self) -> None:
        self.statusBar().showMessage("분석할 영상 파일을 선택하세요.")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "분석할 영상 선택",
            "",
            "Video (*.mp4 *.mov *.mkv *.avi *.mxf *.m4v);;All files (*)",
        )
        if not path:
            self.statusBar().showMessage("영상 선택을 취소했습니다.", 3000)
            return
        video = Path(path)
        LOGGER.info("영상 선택: %s", video)
        if not all(state.installed for state in self.model_manager.states()):
            answer = QMessageBox.question(
                self,
                "모델 설치 필요",
                "필수 모델 설치가 완료되지 않았습니다. 지금 설치할까요?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.pending_video = video
                self.install_models()
            return
        self.start_analysis(video)

    def _run_thread(self, worker, run_slot, finished_slot=None) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(run_slot)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        if hasattr(worker, "failed"):
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            worker.failed.connect(self.show_error)
        if finished_slot:
            worker.finished.connect(finished_slot)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda: self._forget_thread(thread, worker)
        )
        self.threads.append(thread)
        self.workers.append(worker)
        thread.start()

    def _forget_thread(self, thread: QThread, worker: QObject) -> None:
        if thread in self.threads:
            self.threads.remove(thread)
        if worker in self.workers:
            self.workers.remove(worker)

    def install_models(self) -> None:
        if all(state.installed for state in self.model_manager.states()):
            QMessageBox.information(
                self,
                "모델 확인",
                "음성·화면·객체·장면 동작·OCR·자연어 모델이 모두 설치되어 있습니다.",
            )
            return
        self.task_label.setText(
            "음성·화면·객체·장면 동작·OCR·자연어 모델을 설치합니다…"
        )
        self.task_progress.setRange(0, 0)
        worker = ModelInstallWorker(self.model_manager)
        worker.progress.connect(self.task_label.setText)
        worker.finished.connect(self.models_installed)
        self._run_thread(worker, worker.run)

    def models_installed(self) -> None:
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(100)
        self.task_label.setText("필수 모델 설치 완료")
        QMessageBox.information(self, "설치 완료", "필수 모델 설치가 완료됐습니다.")
        if self.pending_video is not None:
            video = self.pending_video
            self.pending_video = None
            self.start_analysis(video)
        elif self.pending_search is not None:
            query, mode = self.pending_search
            self.pending_search = None
            self._launch_search(query, mode)

    def start_analysis(self, video: Path) -> None:
        if self.analysis_started_at is not None:
            QMessageBox.information(
                self,
                "분석 진행 중",
                "현재 영상 분석이 끝난 후 다음 영상을 추가해주세요.",
            )
            return
        LOGGER.info("영상 분석 시작: %s", video)
        self.analysis_started_at = time.monotonic()
        self.analysis_estimated_seconds = self._estimate_analysis_time(video)
        self.add_button.setEnabled(False)
        self.search_button.setEnabled(False)
        self.search_mode.setEnabled(False)
        self.open_video(str(video), 0)
        self.task_label.setText(f"{video.name} · 분석 준비 중…")
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(1)
        self.statusBar().showMessage(f"{video.name} 분석을 시작합니다.")
        worker = AnalyzeWorker(video, self.config)
        worker.progress.connect(self.analysis_progress)
        worker.finished.connect(self.analysis_finished)
        self._run_thread(worker, worker.run)

    def analysis_progress(self, message: str, value: int) -> None:
        self.task_label.setText(message)
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(value)
        if (
            value >= 76
            and self.analysis_started_at is not None
            and self.analysis_estimated_seconds is not None
        ):
            elapsed = time.monotonic() - self.analysis_started_at
            observed_total = elapsed / max(0.01, value / 100.0)
            self.analysis_estimated_seconds = max(
                elapsed + 5.0,
                self.analysis_estimated_seconds * 0.6
                + observed_total * 0.4,
            )

    def analysis_finished(self, message: str) -> None:
        LOGGER.info("영상 분석 완료: %s", message)
        self.analysis_started_at = None
        self.analysis_estimated_seconds = None
        self.add_button.setEnabled(True)
        self.search_button.setEnabled(True)
        self.search_mode.setEnabled(True)
        self.task_label.setText(message)
        self.task_progress.setValue(100)
        self.refresh_library()

    def refresh_library(self) -> None:
        self.library_list.clear()
        count = 0
        for workspace in completed_workspaces(self.config.library_dir):
            try:
                manifest = json.loads(
                    (workspace / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            item = QListWidgetItem(
                str(manifest.get("title") or Path(manifest["video"]).name)
            )
            item.setData(Qt.ItemDataRole.UserRole, manifest["video"])
            self.library_list.addItem(item)
            count += 1
        if hasattr(self, "task_label") and self.analysis_started_at is None:
            if count:
                self.task_label.setText(
                    f"분석된 영상 {count}개 · 검색하거나 목록을 더블클릭하세요."
                )
            else:
                self.task_label.setText(
                    "분석된 영상이 없습니다. ‘영상 추가 및 분석’을 눌러주세요."
                )
        if self.statusBar() is not None:
            self.statusBar().showMessage(
                f"영상 라이브러리 새로고침 완료 · {count}개",
                3000,
            )

    def open_library_item(self, item: QListWidgetItem) -> None:
        video = item.data(Qt.ItemDataRole.UserRole)
        self.open_video(video, 0)

    def start_search(self) -> None:
        query = self.search_input.currentText().strip()
        mode = self.search_mode.currentData() or "auto"
        if not query:
            QMessageBox.information(
                self,
                "검색어 필요",
                "찾고 싶은 장면이나 대사를 입력해주세요.",
            )
            self.search_input.setFocus()
            return
        if self.analysis_started_at is not None:
            QMessageBox.information(
                self,
                "영상 분석 중",
                "영상 분석이 100% 완료된 후 검색해주세요.",
            )
            return
        if not completed_workspaces(self.config.library_dir):
            QMessageBox.information(
                self,
                "분석된 영상 없음",
                "완전히 분석된 영상이 없습니다.\n"
                "‘영상 추가 및 분석’ 후 진행률이 100%가 될 때까지 기다려주세요.",
            )
            return
        intent = parse_search_intent(query, mode=mode)
        required_models: set[str] = set()
        if intent.visual_query is not None:
            required_models.update({"vision", "temporal", "translation"})
        if intent.screen_text_query is not None:
            required_models.add("ocr")
        if intent.object_constraints:
            required_models.add("objects")
        if intent.dialogue_only:
            required_models.clear()
        missing = [
            name
            for name in required_models
            if not self.model_manager.is_installed(name)
        ]
        if missing:
            answer = QMessageBox.question(
                self,
                "장면 검색 모델 필요",
                "선택한 장면 검색에 필요한 추가 모델이 없습니다.\n"
                "지금 설치할까요?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.pending_search = (query, mode)
                self.install_models()
            return
        self._launch_search(query, mode)

    def _launch_search(self, query: str, mode: SearchMode) -> None:
        self.search_button.setEnabled(False)
        self.search_mode.setEnabled(False)
        self.task_label.setText(f"‘{query}’ 검색 준비 중…")
        labels = {
            "auto": "대사·화면을 자동으로 판단해 검색 중…",
            "scene": "화면·객체·장면 동작·화면 글자를 검색 중…",
            "dialogue": "음성 대사만 검색 중…",
        }
        self.statusBar().showMessage(labels[mode])
        worker = SearchWorker(query, self.config, mode=mode)
        worker.finished.connect(self.show_search_results)
        self._run_thread(worker, worker.run)

    def show_search_results(self, hits: list) -> None:
        self.search_button.setEnabled(True)
        self.search_mode.setEnabled(True)
        self.current_hits = hits
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for hit in hits:
            self.results_layout.insertWidget(
                self.results_layout.count() - 1,
                SearchResultCard(hit, self.open_hit),
            )
        self.statusBar().showMessage(f"검색 결과 {len(hits)}개", 5000)
        self.task_label.setText(f"검색 완료 · 결과 {len(hits)}개")
        if not hits:
            QMessageBox.information(
                self,
                "검색 결과 없음",
                "조건을 충분히 만족하는 결과를 찾지 못했습니다.\n"
                "장면 검색은 새 분석 방식으로 분석된 영상에서 가장 정확합니다.",
            )

    def open_hit(self, hit: dict) -> None:
        self.open_video(hit["video"], hit["start"])

    def open_video(self, path: str, seconds: float) -> None:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            self.show_error(f"영상 파일을 찾을 수 없습니다:\n{resolved}")
            return
        source = QUrl.fromLocalFile(str(resolved))
        target = max(0, int(seconds * 1000))
        if self.player.source() == source and self.player.duration() > 0:
            self.pending_seek_ms = None
            self.player.setPosition(target)
        else:
            self.pending_seek_ms = target
            self.player.setSource(source)
        self.player.play()
        QTimer.singleShot(100, self._apply_pending_seek)
        self.play_button.setText("일시정지")
        self.statusBar().showMessage(
            f"재생: {resolved.name} · {_timecode(seconds)}",
            3000,
        )

    def toggle_playback(self) -> None:
        if self.player.source().isEmpty():
            QMessageBox.information(
                self,
                "재생할 영상 없음",
                "먼저 영상을 추가하거나 라이브러리 항목을 더블클릭하세요.",
            )
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_button.setText("재생")
        else:
            self.player.play()
            self.play_button.setText("일시정지")

    def _seek_to(self, position: int) -> None:
        if self.player.source().isEmpty():
            return
        self.pending_seek_ms = None
        target = max(0, min(int(position), self.player.duration()))
        self.player.setPosition(target)
        if self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
            self.player.play()
            self.play_button.setText("일시정지")
        self.time_label.setText(
            f"{_timecode(target / 1000)} / "
            f"{_timecode(self.player.duration() / 1000)}"
        )

    def _apply_pending_seek(self) -> None:
        if self.pending_seek_ms is None or self.player.source().isEmpty():
            return
        if self.player.duration() <= 0:
            return
        target = max(0, min(self.pending_seek_ms, self.player.duration()))
        self.pending_seek_ms = None
        self.player.setPosition(target)
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.player.play()
        self.play_button.setText("일시정지")

    def _media_status_changed(self, status) -> None:
        if status in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            QTimer.singleShot(0, self._apply_pending_seek)

    def _position_changed(self, position: int) -> None:
        if not self.timeline.isSliderDown():
            self.timeline.setValue(position)
        self.time_label.setText(
            f"{_timecode(position / 1000)} / {_timecode(self.player.duration() / 1000)}"
        )

    def _duration_changed(self, duration: int) -> None:
        self.timeline.setRange(0, duration)
        if duration > 0:
            QTimer.singleShot(0, self._apply_pending_seek)

    def run_diagnostics(self) -> None:
        worker = DiagnosticsWorker(Path(self.config.library_dir))
        worker.finished.connect(self.show_diagnostics)
        self._run_thread(worker, worker.run)

    def show_diagnostics(self, report: dict) -> None:
        gpu = report["gpus"][0]["name"] if report.get("gpus") else "GPU 없음 · CPU 모드"
        status = (
            f"{gpu} · RAM {report['ram_total_gb']}GB · "
            f"여유공간 {report['disk_free_gb']}GB · "
            f"{report['recommendation']}"
        )
        self.statusBar().showMessage(status, 10000)
        if report.get("warnings"):
            self.task_label.setText(" / ".join(report["warnings"]))

    def update_usage(self) -> None:
        try:
            usage = live_usage()
        except Exception:
            return
        gpu_text = ""
        if usage["gpus"]:
            gpu = usage["gpus"][0]
            gpu_text = (
                f" · GPU {gpu['utilization'] or 0}% "
                f"({gpu['vram_total_gb'] - gpu['vram_free_gb']:.1f}/"
                f"{gpu['vram_total_gb']:.1f}GB)"
            )
        self.usage_label.setText(
            f"CPU {usage['cpu_percent']:.0f}% · "
            f"RAM {usage['ram_used_gb']}/{usage['ram_total_gb']}GB"
            f"{gpu_text}"
        )
        if (
            self.analysis_started_at is not None
            and self.analysis_estimated_seconds is not None
        ):
            elapsed = time.monotonic() - self.analysis_started_at
            remaining = max(0, self.analysis_estimated_seconds - elapsed)
            base = self.task_label.text().split(" · 예상 남은 시간")[0]
            self.task_label.setText(
                f"{base} · 예상 남은 시간 약 {max(1, int(remaining / 60))}분"
            )

    def _estimate_analysis_time(self, video: Path) -> float | None:
        try:
            import av

            with av.open(str(video)) as container:
                duration = float(container.duration or 0) / 1_000_000
            if duration <= 0:
                return None
            gpu_available = bool(live_usage().get("gpus"))
            if gpu_available:
                factor = 0.35 if self.config.whisper_model == "large-v3" else 0.23
            else:
                factor = 2.2 if self.config.whisper_model == "large-v3" else 1.35
            visual_factor = 0.75 if gpu_available else 2.4
            return max(60.0, duration * (factor + visual_factor))
        except Exception:
            return None

    def show_error(self, message: str) -> None:
        LOGGER.error("백그라운드 작업 실패: %s", message)
        self.analysis_started_at = None
        self.analysis_estimated_seconds = None
        self.add_button.setEnabled(True)
        self.search_button.setEnabled(True)
        self.search_mode.setEnabled(True)
        self.task_progress.setRange(0, 100)
        self.task_label.setText("오류가 발생했습니다.")
        QMessageBox.critical(
            self,
            "오류",
            f"{message}\n\n앱 로그:\n{app_log_path()}",
        )
