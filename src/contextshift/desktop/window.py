"""Dark chat window; only the GUI thread touches message widgets."""

from pathlib import Path

from pydantic import ValidationError
from PySide6.QtCore import QSettings
from PySide6.QtGui import QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from contextshift.config import Settings
from contextshift.desktop.app import project_directory
from contextshift.desktop.chat import ChatFeed, ChatMessage, Composer
from contextshift.desktop.worker import AssistantWorker, DesktopOptions
from contextshift.speech.audio import list_input_devices

_STYLE = """
QMainWindow, QDialog { background: #151722; }
QWidget { color: #cad3f5; font-size: 13px; }
QLabel#logo { color: #8bd5ca; background: #213233; border: 1px solid #334c4c; border-radius: 11px; font-size: 25px; padding: 4px 10px; }
QLabel#brand { font-size: 21px; font-weight: 600; color: #edf0fc; }
QLabel#eyebrow { color: #747d99; font-size: 10px; }
QLabel#state { color: #a6da95; background: #202f2a; border: 1px solid #2e443b; padding: 7px 11px; border-radius: 13px; font-size: 12px; }
QLabel#muted { color: #8a92ad; font-size: 12px; }
QLabel#timestamp { color: #717a94; font-size: 11px; }
QLabel#error { color: #edb3b3; background: #35252d; border: 1px solid #5f3643; border-radius: 9px; padding: 10px; }
QFrame#audioStrip { background: #1a1e2a; border: 1px solid #2a3043; border-radius: 10px; }
QScrollArea#feed { background: #10131c; border: 1px solid #262c3d; border-radius: 13px; }
QWidget#feedContent { background: #10131c; }
QFrame#bubble { background: #1b2030; border: 1px solid #2b3348; border-radius: 12px; }
QFrame#bubbleAI { background: #211e30; border: 1px solid #3a3051; border-radius: 12px; }
QTextBrowser#message { font-size: 15px; color: #d7def5; background: transparent; border: none; padding: 0; }
QLabel#welcomeMark { color: #8bd5ca; font-size: 48px; font-weight: 500; }
QLabel#welcomeTitle { color: #e5e9f7; font-size: 25px; font-weight: 600; }
QLineEdit, QComboBox { background: #202535; border: 1px solid #343c52; border-radius: 7px; padding: 7px 9px; color: #c9d2ec; min-height: 18px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #8bd5ca; }
QComboBox:disabled { color: #747d96; border-color: #2b3042; }
QComboBox::drop-down { border: none; width: 23px; }
QComboBox QAbstractItemView { background: #24293a; color: #d7def5; selection-background-color: #39435d; }
QPushButton { background: #232939; color: #c9d2ec; border: 1px solid #353d52; border-radius: 8px; padding: 8px 13px; }
QPushButton:hover { background: #30384b; border-color: #536078; }
QPushButton:disabled { color: #657088; background: #1d2230; border-color: #2c3243; }
QPushButton#primary { color: #152724; background: #8bd5ca; border-color: #8bd5ca; font-weight: 600; padding: 10px 20px; }
QPushButton#primary:hover { background: #a1e1d8; }
QPushButton#primary:disabled { background: #304d49; color: #819e98; border-color: #304d49; }
QPushButton#send { color: #191728; background: #c6a0f6; border-color: #c6a0f6; font-weight: 600; }
QPushButton#send:hover { background: #d6b7fc; }
QPushButton#send:disabled { color: #746882; background: #322b40; border-color: #453850; }
QPushButton#copy { padding: 0 3px; background: transparent; border: none; color: #7e87a0; font-size: 11px; }
QPushButton#copy:hover { color: #ccd5f0; }
QPushButton#quiet { background: transparent; border: none; color: #929db9; }
QPlainTextEdit#composer { background: #1c2130; border: 1px solid #354158; border-radius: 12px; padding: 10px; color: #d7def5; font-size: 14px; selection-background-color: #40566a; }
QPlainTextEdit#composer:focus { border-color: #8bd5ca; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #56627b; border-radius: 4px; background: #202535; }
QCheckBox::indicator:checked { background: #8bd5ca; border-color: #8bd5ca; }
QScrollBar:vertical { background: #10131c; width: 9px; margin: 7px 2px; }
QScrollBar::handle:vertical { background: #394159; min-height: 35px; border-radius: 3px; }
QScrollBar::handle:vertical:hover { background: #53617c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


class MainWindow(QMainWindow):
    def __init__(
        self, *, preferences: QSettings | None = None, discover_devices: bool = True
    ) -> None:
        super().__init__()
        self.preferences = preferences if preferences is not None else QSettings()
        self.worker: AssistantWorker | None = None
        self._active_epoch = -1
        self._active_message: ChatMessage | None = None
        self._stopping = False
        self._closing = False
        self.setWindowTitle("ContextShift")
        self.resize(970, 850)
        self.setMinimumSize(740, 630)
        self.setStyleSheet(_STYLE)
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, "#151722"),
            (QPalette.ColorRole.Base, "#1c2130"),
            (QPalette.ColorRole.Text, "#d7def5"),
            (QPalette.ColorRole.ButtonText, "#d7def5"),
            (QPalette.ColorRole.PlaceholderText, "#737f9c"),
        ):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(15)

        header = QHBoxLayout()
        logo = QLabel("›_")
        logo.setObjectName("logo")
        header.addWidget(logo)
        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        brand = QLabel("ContextShift")
        brand.setObjectName("brand")
        title_column.addWidget(brand)
        eyebrow = QLabel("LIVE CONTEXT  /  对话中的下一句")
        eyebrow.setObjectName("eyebrow")
        title_column.addWidget(eyebrow)
        header.addLayout(title_column)
        header.addStretch()
        self.state = QLabel("● 未启动")
        self.state.setObjectName("state")
        header.addWidget(self.state)
        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self._show_settings)
        header.addWidget(self.settings_button)
        layout.addLayout(header)

        strip = QFrame()
        strip.setObjectName("audioStrip")
        sources = QHBoxLayout(strip)
        sources.setContentsMargins(12, 10, 12, 10)
        sources.setSpacing(10)
        microphone_label = QLabel("麦克风")
        microphone_label.setStyleSheet("color: #8bd5ca;")
        sources.addWidget(microphone_label)
        self.microphone = QComboBox()
        self.microphone.setMinimumWidth(150)
        sources.addWidget(self.microphone, 1)
        computer_label = QLabel("会议音频")
        computer_label.setStyleSheet("color: #8aadf4;")
        sources.addWidget(computer_label)
        self.system = QComboBox()
        self.system.setMinimumWidth(150)
        sources.addWidget(self.system, 1)
        self.refresh_devices = QPushButton("刷新")
        self.refresh_devices.setObjectName("quiet")
        self.refresh_devices.clicked.connect(self._discover_devices)
        sources.addWidget(self.refresh_devices)
        layout.addWidget(strip)

        self.error = QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        self.feed = ChatFeed()
        layout.addWidget(self.feed, 1)

        compose_row = QHBoxLayout()
        compose_row.setSpacing(10)
        self.question = Composer()
        self.question.submitted.connect(self._ask)
        compose_row.addWidget(self.question, 1)
        self.ask_button = QPushButton("发送 ↵")
        self.ask_button.setObjectName("send")
        self.ask_button.setFixedSize(84, 80)
        self.ask_button.clicked.connect(self._ask)
        compose_row.addWidget(self.ask_button)
        layout.addLayout(compose_row)

        footer = QHBoxLayout()
        self.start_button = QPushButton("开始监听")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._toggle_running)
        self.start_button.setToolTip("启动新会话；停止后可保留当前消息查看")
        self.cancel_button = QPushButton("取消分析")
        self.cancel_button.clicked.connect(self._cancel_answer)
        self.clear_button = QPushButton("清空界面")
        self.clear_button.setObjectName("quiet")
        self.clear_button.setToolTip("清空消息显示并取消当前分析；保留本次会话上下文")
        self.clear_button.clicked.connect(self._clear_chat)
        footer.addWidget(self.start_button)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.clear_button)
        footer.addStretch()
        hint = QLabel("Enter 发送 · Shift+Enter 换行")
        hint.setObjectName("muted")
        footer.addWidget(hint)
        layout.addLayout(footer)

        self._build_settings()
        if discover_devices:
            self._discover_devices()
        else:
            self.microphone.addItem("选择麦克风", None)
            self.system.addItem("选择回环输入", None)
        self._update_controls()

    def _build_settings(self) -> None:
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("ContextShift · 设置")
        self.settings_dialog.setMinimumWidth(640)
        column = QVBoxLayout(self.settings_dialog)
        column.setContentsMargins(24, 22, 24, 22)
        column.setSpacing(18)
        title = QLabel("连接与本地模型")
        title.setObjectName("brand")
        column.addWidget(title)
        form = QFormLayout()
        form.setSpacing(12)
        base = project_directory()
        self.env_file = self._path_row(
            form,
            "配置文件",
            str(self.preferences.value("env", str(base / ".env"))),
            False,
        )
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("留空则读取 .env；这里输入的 Key 不会保存")
        form.addRow("API Key", self.api_key)
        self.asr_model = self._path_row(
            form,
            "ASR 模型",
            str(
                self.preferences.value("asr", str(base / "models/asr/SenseVoiceSmall"))
            ),
            True,
        )
        self.vad_model = self._path_row(
            form,
            "VAD 模型",
            str(self.preferences.value("vad", str(base / "models/vad/fsmn-vad"))),
            True,
        )
        self.audio_enabled = QCheckBox("启用麦克风与会议音频监听")
        self.audio_enabled.setChecked(self.preferences.value("audio", True, type=bool))
        self.audio_enabled.toggled.connect(self._update_controls)
        form.addRow("语音", self.audio_enabled)
        column.addLayout(form)
        note = QLabel(
            "麦克风发言仅补充上下文；会议音频触发分析。\n关闭语音监听后，仍可手动输入问题。"
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        column.addWidget(note)
        done = QPushButton("完成")
        done.setObjectName("primary")
        done.clicked.connect(self.settings_dialog.hide)
        column.addWidget(done)

    def _path_row(self, form, title: str, value: str, directory: bool) -> QLineEdit:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit(value)
        browse = QPushButton("选择")

        def choose():
            if directory:
                path = QFileDialog.getExistingDirectory(self, title, field.text())
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, title, str(Path.home()), "所有文件 (*)"
                )
            if path:
                field.setText(path)

        browse.clicked.connect(choose)
        row.addWidget(field, 1)
        row.addWidget(browse)
        form.addRow(title, container)
        return field

    def _show_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _discover_devices(self) -> None:
        try:
            devices = list_input_devices()
        except Exception:  # noqa: BLE001 - allow text mode without speech extras
            devices = []
        mic = (
            self.microphone.currentData()
            if self.microphone.count()
            else self.preferences.value("microphone", None)
        )
        system = (
            self.system.currentData()
            if self.system.count()
            else self.preferences.value("system", None)
        )
        self.microphone.clear()
        self.system.clear()
        self.microphone.addItem("选择麦克风", None)
        self.system.addItem("选择回环输入", None)
        for device in devices:
            self.microphone.addItem(device.name, device.index)
            self.system.addItem(device.name, device.index)
            if mic is None and "blackhole" not in device.name.lower():
                mic = device.index
            if system is None and "blackhole" in device.name.lower():
                system = device.index
        for combo, value in ((self.microphone, mic), (self.system, system)):
            if value is not None:
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    value = None
            combo.setCurrentIndex(max(0, combo.findData(value)))
        if not devices:
            self.audio_enabled.setChecked(False)
            self.state.setText("● 文字模式可用")

    def _load_options(self) -> DesktopOptions:
        values = {"_env_file": self.env_file.text().strip() or None}
        if self.api_key.text().strip():
            values["DEEPSEEK_API_KEY"] = self.api_key.text().strip()
        settings = Settings(**values)
        microphone, system = self.microphone.currentData(), self.system.currentData()
        enabled = self.audio_enabled.isChecked()
        if enabled:
            if microphone is None or system is None or microphone == system:
                raise ValueError("请选择不同的麦克风和会议回环输入设备。")
            for title, value in (
                ("ASR", self.asr_model.text()),
                ("VAD", self.vad_model.text()),
            ):
                if not Path(value).expanduser().is_dir():
                    raise ValueError(f"{title} 模型目录不存在，请在设置中选择。")
        return DesktopOptions(
            settings,
            enabled,
            microphone,
            system,
            Path(self.asr_model.text()).expanduser(),
            Path(self.vad_model.text()).expanduser(),
        )

    def _toggle_running(self) -> None:
        if self.worker is not None:
            self._stop_assistant()
            return
        try:
            options = self._load_options()
        except ValidationError:
            self._show_error(
                "请在设置中选择有效的 .env，或填写 API Key，并检查配置数值。"
            )
            self._show_settings()
            return
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.feed.clear()
        self.error.hide()
        self._active_message = None
        self._active_epoch = -1
        self._stopping = False
        self.settings_dialog.hide()
        worker = AssistantWorker(options, self)
        self.worker = worker

        def current(function):
            return lambda *args: function(*args) if self.worker is worker else None

        worker.ready.connect(current(self._ready))
        worker.response_started.connect(current(self._response_started))
        worker.response.connect(current(self._response))
        worker.response_finished.connect(current(self._response_finished))
        worker.transcript.connect(current(self._transcript))
        worker.status.connect(current(self._set_status))
        worker.failed.connect(current(self._show_error))
        worker.finished.connect(current(self._assistant_finished))
        self.state.setText("● 正在启动")
        self._save_preferences()
        self._update_controls()
        worker.start()

    def _ready(self) -> None:
        if not self._stopping:
            self.question.setEnabled(True)
            self.ask_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            self.question.setFocus()

    def _transcript(self, role: str, text: str, current_question: bool = False) -> None:
        if not text.strip() or self._stopping:
            return
        source = "computer" if role == "面试官" else "microphone"
        title = "我的输入" if role == "手动输入" else None
        self.feed.add_message(source, text, title=title)
        if current_question and self._active_epoch >= 0:
            self._ensure_reply()

    def _ensure_reply(self) -> ChatMessage:
        if self._active_message is None:
            self._active_message = self.feed.add_message("ai", state="分析中")
        return self._active_message

    def _end_reply(self, state: str) -> None:
        if self._active_message is not None and self._active_message.pinned:
            self._active_message.set_state(state)
            if not self._active_message.text:
                self._active_message.set_text(f"本次分析{state.removeprefix('已')}。")

    def _response_started(self, epoch: int) -> None:
        if self._stopping:
            return
        self._end_reply("已中断")
        self._active_epoch = epoch
        self._active_message = None
        self.error.hide()

    def _response(self, epoch: int, text: str) -> None:
        if epoch == self._active_epoch and not self._stopping and text:
            self._ensure_reply().append_text(text)

    def _response_finished(self, epoch: int, state: str) -> None:
        if epoch == self._active_epoch and not self._stopping:
            self._end_reply(state)

    def _ask(self) -> None:
        text = self.question.toPlainText().strip()
        if (
            text
            and self.worker is not None
            and not self._stopping
            and self.ask_button.isEnabled()
        ):
            self.worker.ask(text)
            self.question.clear()

    def _cancel_answer(self) -> None:
        self._end_reply("已取消")
        self._active_epoch = -1
        if self.worker is not None:
            self.worker.cancel_answer()

    def _clear_chat(self) -> None:
        self._cancel_answer()
        self._active_message = None
        self.feed.clear()

    def _stop_assistant(self) -> None:
        if self.worker is None:
            return
        self._end_reply("已停止")
        self._active_epoch = -1
        self._stopping = True
        self.state.setText("● 正在停止")
        self.worker.request_stop()
        self._update_controls()

    def _assistant_finished(self) -> None:
        self._end_reply("已停止")
        worker = self.worker
        self.worker = None
        self._active_epoch = -1
        self._stopping = False
        self.state.setText("● 已停止")
        if worker is not None:
            worker.deleteLater()
        self._update_controls()
        if self._closing:
            self.close()

    def _set_status(self, text: str) -> None:
        if not self._stopping:
            self.state.setText("● " + text)

    def _show_error(self, text: str) -> None:
        self.error.setText(text)
        self.error.show()
        self._end_reply("失败")

    def _update_controls(self, *_args) -> None:
        running = self.worker is not None
        editable = not running and not self._closing
        self.settings_button.setEnabled(editable)
        self.settings_dialog.setEnabled(editable)
        self.audio_enabled.setEnabled(editable)
        for widget in (self.microphone, self.system, self.refresh_devices):
            widget.setEnabled(editable and self.audio_enabled.isChecked())
        self.start_button.setText(
            "正在停止…"
            if self._stopping
            else "停止助手"
            if running
            else "开始监听"
            if self.audio_enabled.isChecked()
            else "启动助手"
        )
        self.start_button.setEnabled(not self._stopping and not self._closing)
        if not running or self._stopping:
            self.question.setEnabled(False)
            self.ask_button.setEnabled(False)
            self.cancel_button.setEnabled(False)

    def _save_preferences(self) -> None:
        for key, value in {
            "env": self.env_file.text(),
            "asr": self.asr_model.text(),
            "vad": self.vad_model.text(),
            "audio": self.audio_enabled.isChecked(),
            "microphone": self.microphone.currentData(),
            "system": self.system.currentData(),
        }.items():
            if value is not None:
                self.preferences.setValue(key, value)
        self.preferences.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_preferences()
        self.settings_dialog.hide()
        if self.worker is not None:
            self._closing = True
            self._stop_assistant()
            event.ignore()
            return
        event.accept()
