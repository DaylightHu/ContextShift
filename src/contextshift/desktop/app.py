"""Launch a desktop session after the selected input path is ready."""

import sys
from pathlib import Path


def main() -> None:
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from contextshift.desktop.window import MainWindow
    except ImportError as exc:
        raise SystemExit(
            "缺少桌面依赖，请运行：uv sync --extra desktop；语音模式另加 --extra speech"
        ) from exc
    application = QApplication(sys.argv)
    application.setApplicationName("ContextShift")
    application.setOrganizationName("ContextShift")
    application.setQuitOnLastWindowClosed(False)
    window = MainWindow(manage_application=True)
    QTimer.singleShot(0, window.launch)
    # Keep the window alive while Qt's event loop runs.
    application._contextshift_window = window
    raise SystemExit(application.exec())


def project_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path.home()
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
