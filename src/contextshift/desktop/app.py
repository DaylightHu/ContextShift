"""A small native control panel for live audio and cited repository questions."""

import sys
from pathlib import Path


def main() -> None:
    try:
        from PySide6.QtWidgets import QApplication

        from contextshift.desktop.window import MainWindow
    except ImportError as exc:
        raise SystemExit(
            "缺少桌面依赖，请运行：uv sync --extra desktop --extra speech"
        ) from exc
    application = QApplication(sys.argv)
    application.setApplicationName("ContextShift")
    application.setOrganizationName("ContextShift")
    window = MainWindow()
    window.show()
    # Keep the window alive while Qt's event loop runs.
    application._contextshift_window = window
    raise SystemExit(application.exec())


def project_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path.home()
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main()
