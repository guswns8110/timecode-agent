from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .app_logging import configure_logging

STYLESHEET = """
QWidget {
    background: #111419;
    color: #e8edf3;
    font-family: "Pretendard", "Malgun Gothic", sans-serif;
    font-size: 13px;
}
QMainWindow, QMenuBar, QMenu, QStatusBar { background: #0d1014; }
QFrame#panel {
    background: #171b22;
    border: 1px solid #272e39;
    border-radius: 9px;
}
QLabel#heading { font-size: 16px; font-weight: 700; padding: 4px; }
QLabel#muted { color: #8b97a6; font-size: 11px; }
QLabel#resultTitle { font-weight: 650; }
QFrame#resultCard {
    background: #1d222b;
    border: 1px solid #2b3340;
    border-radius: 7px;
}
QPushButton {
    background: #262d38;
    border: 1px solid #343e4d;
    border-radius: 6px;
    padding: 8px 13px;
}
QPushButton:hover { background: #313a47; }
QPushButton#primary { background: #5a67f2; border-color: #7280ff; }
QComboBox, QLineEdit, QListWidget {
    background: #0f1217;
    border: 1px solid #303846;
    border-radius: 7px;
    padding: 9px;
}
QProgressBar {
    background: #0f1217;
    border: 1px solid #303846;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk { background: #5a67f2; border-radius: 4px; }
QScrollArea { border: 0; background: transparent; }
QVideoWidget { background: #07090c; border-radius: 7px; }
"""


def main() -> int:
    log_path = configure_logging()
    logger = logging.getLogger(__name__)
    app = QApplication(sys.argv)
    app.setApplicationName("Timecode Agent")
    app.setOrganizationName("Timecode Agent")
    app.setStyleSheet(STYLESHEET)
    try:
        from .main_window import MainWindow

        window = MainWindow()
    except Exception:
        logger.exception("메인 창 생성 실패")
        QMessageBox.critical(
            None,
            "Timecode Agent 실행 오류",
            f"프로그램을 시작하지 못했습니다.\n\n로그 파일:\n{log_path}",
        )
        return 1
    window.show()
    logger.info("메인 창 표시 완료")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
