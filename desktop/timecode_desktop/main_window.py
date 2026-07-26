from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, QUrl
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
    QVBoxLayout,
    QWidget,
)

from video_agent.workspace_discovery import find_workspaces

from .config import APP_NAME, AppConfig
from .diagnostics import live_usage
from .model_manager import ModelManager
from .workers import (
    AnalyzeWorker,
    DiagnosticsWorker,
    ModelInstallWorker,
    SearchWorker,
)


def _timecode(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        self.current_hits: list[dict] = []
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

    def _setup_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 10)

        top = QHBoxLayout()
        self.search_input = QComboBox()
        self.search_input.setEditable(True)
        self.search_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.search_input.lineEdit().setPlaceholderText(
            "예: 비행기가 활주로에서 이륙하는 장면 찾아줘"
        )
        self.search_input.lineEdit().returnPressed.connect(self.start_search)
        search_button = QPushButton("검색")
        search_button.setObjectName("primary")
        search_button.clicked.connect(self.start_search)
        add_button = QPushButton("영상 추가 및 분석")
        add_button.clicked.connect(self.choose_video)
        top.addWidget(self.search_input, 1)
        top.addWidget(search_button)
        top.addWidget(add_button)
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

    def _library_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel("영상 라이브러리")
        heading.setObjectName("heading")
        self.library_list = QListWidget()
        self.library_list.itemDoubleClicked.connect(self.open_library_item)
        refresh = QPushButton("새로고침")
        refresh.clicked.connect(self.refresh_library)
        layout.addWidget(heading)
        layout.addWidget(self.library_list, 1)
        layout.addWidget(refresh)
        return panel

    def _center_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.addWidget(self.video_widget, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("재생")
        self.play_button.clicked.connect(self.toggle_playback)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.sliderMoved.connect(self.player.setPosition)
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
        chosen = QFileDialog.getExistingDirectory(
            self, "영상 라이브러리 폴더 선택", self.config.library_dir
        )
        if chosen:
            self.config.library_dir = chosen
            self.config.save()
            self.config.ensure_dirs()
            self.refresh_library()
            self.run_diagnostics()

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "분석할 영상 선택",
            "",
            "Video (*.mp4 *.mov *.mkv *.avi *.mxf *.m4v);;All files (*)",
        )
        if not path:
            return
        if not all(state.installed for state in self.model_manager.states()):
            answer = QMessageBox.question(
                self,
                "모델 설치 필요",
                "필수 모델 설치가 완료되지 않았습니다. 지금 설치할까요?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.install_models()
            return
        self.start_analysis(Path(path))

    def _run_thread(self, worker, run_slot, finished_slot=None) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(run_slot)
        worker.finished.connect(thread.quit)
        if hasattr(worker, "failed"):
            worker.failed.connect(thread.quit)
            worker.failed.connect(self.show_error)
        if finished_slot:
            worker.finished.connect(finished_slot)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._forget_thread(thread))
        self.threads.append(thread)
        thread.start()

    def _forget_thread(self, thread: QThread) -> None:
        if thread in self.threads:
            self.threads.remove(thread)

    def install_models(self) -> None:
        self.task_label.setText("Medium, Large-v3, 화면 검색 모델을 설치합니다…")
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

    def start_analysis(self, video: Path) -> None:
        self.analysis_started_at = time.monotonic()
        self.analysis_estimated_seconds = self._estimate_analysis_time(video)
        worker = AnalyzeWorker(video, self.config)
        worker.progress.connect(self.analysis_progress)
        worker.finished.connect(self.analysis_finished)
        self._run_thread(worker, worker.run)

    def analysis_progress(self, message: str, value: int) -> None:
        self.task_label.setText(message)
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(value)

    def analysis_finished(self, message: str) -> None:
        self.analysis_started_at = None
        self.analysis_estimated_seconds = None
        self.task_label.setText(message)
        self.task_progress.setValue(100)
        self.refresh_library()

    def refresh_library(self) -> None:
        self.library_list.clear()
        for workspace in find_workspaces([self.config.library_dir]):
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

    def open_library_item(self, item: QListWidgetItem) -> None:
        video = item.data(Qt.ItemDataRole.UserRole)
        self.open_video(video, 0)

    def start_search(self) -> None:
        query = self.search_input.currentText().strip()
        if not query:
            return
        self.statusBar().showMessage("대사·OCR·화면을 통합 검색 중…")
        worker = SearchWorker(query, self.config)
        worker.finished.connect(self.show_search_results)
        self._run_thread(worker, worker.run)

    def show_search_results(self, hits: list) -> None:
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

    def open_hit(self, hit: dict) -> None:
        self.open_video(hit["video"], hit["start"])

    def open_video(self, path: str, seconds: float) -> None:
        self.player.setSource(QUrl.fromLocalFile(str(Path(path).resolve())))
        self.player.setPosition(int(seconds * 1000))
        self.player.play()
        self.play_button.setText("일시정지")

    def toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_button.setText("재생")
        else:
            self.player.play()
            self.play_button.setText("일시정지")

    def _position_changed(self, position: int) -> None:
        if not self.timeline.isSliderDown():
            self.timeline.setValue(position)
        self.time_label.setText(
            f"{_timecode(position / 1000)} / {_timecode(self.player.duration() / 1000)}"
        )

    def _duration_changed(self, duration: int) -> None:
        self.timeline.setRange(0, duration)

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
            visual_factor = 0.18 if gpu_available else 0.65
            return max(60.0, duration * (factor + visual_factor))
        except Exception:
            return None

    def show_error(self, message: str) -> None:
        self.task_progress.setRange(0, 100)
        self.task_label.setText("오류가 발생했습니다.")
        QMessageBox.critical(self, "오류", message)
