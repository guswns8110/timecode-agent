from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QPoint, Qt, Signal, Slot
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


class ProbeWorker(QObject):
    finished = Signal()

    def __init__(self, calls: dict[str, int]):
        super().__init__()
        self.calls = calls

    @Slot()
    def run(self) -> None:
        self.calls["worker"] += 1
        self.finished.emit()


def main() -> int:
    calls = {
        "search": 0,
        "add": 0,
        "play": 0,
        "refresh": 0,
        "seek": 0,
        "worker": 0,
    }

    def record(name: str):
        def handler(self, *args) -> None:
            calls[name] += 1

        return handler

    MainWindow.start_search = record("search")
    MainWindow.choose_video = record("add")
    MainWindow.toggle_playback = record("play")
    MainWindow.refresh_library = record("refresh")
    MainWindow._seek_to = record("seek")
    MainWindow.run_diagnostics = lambda self: None

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.processEvents()

    for button in (
        window.search_button,
        window.add_button,
        window.play_button,
        window.refresh_button,
    ):
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        app.processEvents()

    expected = {
        "search": 1,
        "add": 1,
        "play": 1,
        "refresh": 2,
        "seek": 0,
        "worker": 0,
    }
    if calls != expected:
        raise RuntimeError(f"버튼 연결 테스트 실패: {calls!r} != {expected!r}")

    window.timeline.setRange(0, 1000)
    QTest.mouseClick(
        window.timeline,
        Qt.MouseButton.LeftButton,
        pos=QPoint(window.timeline.width() * 3 // 4, window.timeline.height() // 2),
    )
    app.processEvents()
    if calls["seek"] != 1 or window.timeline.value() < 700:
        raise RuntimeError("재생바 탐색 연결 테스트 실패")

    worker = ProbeWorker(calls)
    window._run_thread(worker, worker.run)
    deadline = time.monotonic() + 5
    while window.threads and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    if window.threads or calls["worker"] != 1:
        raise RuntimeError("백그라운드 버튼 작업 실행 테스트 실패")

    window.close()
    print("ui-buttons-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
