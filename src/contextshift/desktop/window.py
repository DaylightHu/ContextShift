"""A listening window appears only after the full startup barrier is met."""

from pathlib import Path
from time import perf_counter

from pydantic import ValidationError
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
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
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from contextshift.config import Settings
from contextshift.desktop.app import project_directory
from contextshift.desktop.chat import Avatar, ChatFeed, ChatMessage, TestComposer
from contextshift.desktop.worker import AssistantWorker, DesktopOptions, ReplyMetrics
from contextshift.desktop.workspace import WorkspacePanel
from contextshift.semantic_index import SemanticProgress
from contextshift.speech.audio import list_input_devices

_STYLE = """
QMainWindow, QDialog { background: #151722; }
QWidget { color: #cad3f5; font-size: 13px; }
QLabel#statusDot { color: #8bd5ca; font-size: 12px; }
QLabel#testDetails { color: #8793ae; font-size: 11px; }
QToolButton { color: #8793ae; background: transparent; border: none; font-size: 20px; padding: 1px 7px; }
QToolButton:hover { color: #d7def5; background: #232939; border-radius: 7px; }
QLabel#error { color: #edb3b3; background: #35252d; border: 1px solid #5f3643; border-radius: 8px; padding: 9px; }
QScrollArea#feed { background: #10131c; border: none; }
QWidget#feedContent { background: #10131c; }
QFrame#bubble { background: #1b2030; border: 1px solid #2b3348; border-radius: 12px; }
QFrame#bubbleAI { background: #211e30; border: 1px solid #3a3051; border-radius: 12px; }
QTextBrowser#message { font-size: 15px; color: #d7def5; background: transparent; border: none; padding: 0; }
QLineEdit, QComboBox, QPlainTextEdit { background: #202535; border: 1px solid #343c52; border-radius: 7px; padding: 7px 9px; color: #c9d2ec; min-height: 18px; }
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border-color: #8bd5ca; }
QComboBox::drop-down { border: none; width: 23px; }
QComboBox QAbstractItemView { background: #24293a; color: #d7def5; selection-background-color: #39435d; }
QPushButton { background: #232939; color: #c9d2ec; border: 1px solid #353d52; border-radius: 8px; padding: 8px 13px; }
QPushButton:hover { background: #30384b; }
QPushButton#primary { color: #152724; background: #8bd5ca; border: none; font-weight: 600; padding: 10px 20px; }
QPushButton:disabled { color: #657088; background: #1d2230; }
QScrollBar:vertical { background: #10131c; width: 9px; margin: 7px 2px; }
QScrollBar::handle:vertical { background: #394159; min-height: 35px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QTreeWidget#projectTree { background: #10131c; border: none; color: #b7c2d9; font-size: 12px; }
QTreeWidget::item { padding: 5px 2px; }
QTreeWidget::item:selected { background: #263447; }
QLabel#workspaceStatus { color: #8793ae; font-size: 11px; }
QLabel#workspaceTitle { color: #cad3f5; font-weight: 600; }
QFrame#activityPanel { background: #1b2030; border: 1px solid #343c52; border-radius: 8px; }
QLabel#activityText { color: #b7c2d9; font-size: 11px; }
QProgressBar#activityProgress { border: none; background: #30384b; border-radius: 3px; max-height: 5px; min-height: 5px; }
QProgressBar#activityProgress::chunk { background: #8bd5ca; border-radius: 3px; }
QSplitter::handle { background: #252d3c; width: 1px; }
QPushButton#sourceLink { color: #8bd5ca; background: transparent; border: none; padding: 2px; font-size: 11px; text-align: left; }
"""


class LoadingDialog(QDialog):
    cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ContextShift")
        self.setFixedSize(390, 112)
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 20, 24, 20)
        row.setSpacing(16)
        row.addWidget(Avatar("ai"))
        self.message = QLabel("正在准备…")
        self.message.setWordWrap(True)
        row.addWidget(self.message, 1)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.cancelled.emit()
        event.accept()

    def reject(self) -> None:
        self.close()


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        preferences: QSettings | None = None,
        discover_devices: bool = True,
        manage_application: bool = False,
    ) -> None:
        super().__init__()
        self.preferences = preferences if preferences is not None else QSettings()
        self.worker: AssistantWorker | None = None
        self._active_epoch = -1
        self._active_message: ChatMessage | None = None
        self._closing = False
        self._restart_options: DesktopOptions | None = None
        self._failure = ""
        self._ready = False
        self.composer: TestComposer | None = None
        self._manage_application = manage_application
        self.setWindowTitle("ContextShift")
        self.resize(1180, 760)
        self.setMinimumSize(540, 400)
        self.setStyleSheet(_STYLE)
        palette = self.palette()
        for role, color in (
            (QPalette.ColorRole.Window, "#151722"),
            (QPalette.ColorRole.Base, "#1c2130"),
            (QPalette.ColorRole.Text, "#d7def5"),
            (QPalette.ColorRole.ButtonText, "#d7def5"),
        ):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self._layout = layout
        layout.setContentsMargins(12, 5, 12, 12)
        layout.setSpacing(3)
        toolbar = QHBoxLayout()
        toolbar.addStretch()
        self.state = QLabel("●")
        self.state.setObjectName("statusDot")
        self.state.setToolTip("正在监听")
        toolbar.addWidget(self.state)
        self.settings_button = QToolButton()
        self.settings_button.setText("⚙")
        self.settings_button.setToolTip("设置")
        self.settings_button.setAccessibleName("设置")
        self.settings_button.clicked.connect(self._open_settings)
        toolbar.addWidget(self.settings_button)
        layout.addLayout(toolbar)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace = WorkspacePanel(self.preferences)
        self.workspace.index_changed.connect(self._repository_changed)
        self.workspace.focus_changed.connect(self._file_focus_changed)
        self.workspace.idle.connect(self._workspace_idle)
        self.splitter.addWidget(self.workspace)
        chat = QWidget()
        self._chat_layout = QVBoxLayout(chat)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self.feed = ChatFeed()
        self._chat_layout.addWidget(self.feed, 1)
        self.activity = QFrame()
        self.activity.setObjectName("activityPanel")
        activity_layout = QVBoxLayout(self.activity)
        activity_layout.setContentsMargins(12, 7, 12, 7)
        activity_layout.setSpacing(3)
        self.index_activity = QLabel("本地向量索引：等待工作区")
        self.answer_activity = QLabel("回答：等待提问")
        for label in (self.index_activity, self.answer_activity):
            label.setObjectName("activityText")
            activity_layout.addWidget(label)
        self.index_progress = QProgressBar()
        self.index_progress.setObjectName("activityProgress")
        self.index_progress.setTextVisible(False)
        self.index_progress.hide()
        activity_layout.addWidget(self.index_progress)
        self._chat_layout.addWidget(self.activity)
        self._index_text = "本地向量索引：等待工作区"
        self._answer_text = "回答：等待提问"
        self._index_started: float | None = None
        self._answer_started: float | None = None
        self._activity_timer = QTimer(self)
        self._activity_timer.timeout.connect(self._refresh_activity)
        self._activity_timer.start(1000)
        self.splitter.addWidget(chat)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([260, 900])
        layout.addWidget(self.splitter, 1)
        self.loading = LoadingDialog(self)
        self.loading.cancelled.connect(self.close)
        self._build_settings()
        self._discover_on_mode_change = discover_devices
        if discover_devices and not self.test_mode.isChecked():
            self._discover_devices()
        else:
            self.microphone.addItem("选择麦克风", None)
            self.system.addItem("选择回环输入", None)
        QTimer.singleShot(0, self.workspace.restore)

    def _repository_changed(self, index):
        self._end_reply("上下文已更新")
        self._active_epoch = -1
        self._active_message = None
        if self.worker is not None:
            self.worker.set_repository(
                index,
                pending=self.workspace.root is not None and index is None,
                root=self.workspace.root,
            )

    def _file_focus_changed(self, relative):
        if self.worker is not None:
            self.worker.set_active_file(relative)

    def _workspace_idle(self):
        if self._closing:
            self.close()

    def _sources_ready(self, epoch, references, elapsed):
        if self._closing or not self._ready or epoch != self._active_epoch:
            return
        if references:
            self._ensure_reply().set_sources(
                references, self.workspace.open_source_in_vscode
            )
        if self.composer is not None:
            self.composer.details.setText(
                f"{self.worker.options.settings.model} · 检索 {elapsed:.3f}s · {len(references)} 处来源"
            )

    def _build_settings(self) -> None:
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("ContextShift · 设置")
        self.settings_dialog.setMinimumWidth(620)
        column = QVBoxLayout(self.settings_dialog)
        column.setContentsMargins(22, 22, 22, 22)
        form = QFormLayout()
        form.setSpacing(12)
        base = project_directory()
        self.test_mode = QCheckBox("切换为测试模式")
        self.test_mode.setChecked(self.preferences.value("test_mode", False, type=bool))
        self.test_mode.setToolTip(
            "只用文字模拟双方发言，不采集音频；切换模式会清空当前会话。"
        )
        self.test_mode.toggled.connect(self._update_audio_fields)
        form.addRow(self.test_mode)
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
        self.microphone, self.system = QComboBox(), QComboBox()
        form.addRow("麦克风", self.microphone)
        form.addRow("会议回环", self.system)
        column.addLayout(form)
        self.error = QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        self.error.hide()
        column.addWidget(self.error)
        row = QHBoxLayout()
        refresh = QPushButton("刷新设备")
        refresh.clicked.connect(self._discover_devices)
        self.apply_button = QPushButton("保存并继续")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self.launch)
        row.addWidget(refresh)
        row.addStretch()
        row.addWidget(self.apply_button)
        column.addLayout(row)
        self.settings_dialog.rejected.connect(self._configuration_closed)
        self._update_audio_fields()

    def _update_audio_fields(self) -> None:
        if not hasattr(self, "microphone"):
            return
        enabled = not self.test_mode.isChecked()
        for widget in (
            self.asr_model.parentWidget(),
            self.vad_model.parentWidget(),
            self.microphone,
            self.system,
        ):
            widget.setEnabled(enabled)
        if (
            enabled
            and getattr(self, "_discover_on_mode_change", False)
            and self.microphone.count() <= 1
        ):
            self._discover_devices()

    def _path_row(self, form, title: str, value: str, directory: bool) -> QLineEdit:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit(value)
        button = QPushButton("选择")

        def choose():
            if directory:
                path = QFileDialog.getExistingDirectory(self, title, field.text())
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, title, str(Path.home()), "所有文件 (*)"
                )
            if path:
                field.setText(path)

        button.clicked.connect(choose)
        row.addWidget(field, 1)
        row.addWidget(button)
        form.addRow(title, container)
        return field

    def _discover_devices(self) -> None:
        try:
            devices = list_input_devices()
        except Exception:  # noqa: BLE001 - show configuration rather than a broken listening window
            devices = []
        for combo, key, system in (
            (self.microphone, "microphone", False),
            (self.system, "system", True),
        ):
            saved_name = self.preferences.value(key + "_name", "")
            current_name = (
                combo.currentText() if combo.currentData() is not None else saved_name
            )
            combo.clear()
            combo.addItem("选择回环输入" if system else "选择麦克风", None)
            selected = 0
            for device in devices:
                combo.addItem(device.name, device.index)
                if (
                    current_name
                    and device.name == current_name
                    or not current_name
                    and not selected
                    and ("blackhole" in device.name.lower()) == system
                ):
                    selected = combo.count() - 1
            combo.setCurrentIndex(selected)

    def _load_options(self) -> DesktopOptions:
        values = {"_env_file": self.env_file.text().strip() or None}
        if self.api_key.text().strip():
            values["DEEPSEEK_API_KEY"] = self.api_key.text().strip()
        settings = Settings(**values)
        microphone, system = self.microphone.currentData(), self.system.currentData()
        test_mode = self.test_mode.isChecked()
        if not test_mode and (
            microphone is None or system is None or microphone == system
        ):
            raise ValueError("请选择不同的麦克风和会议回环输入设备。")
        for name, value in (
            ("ASR", self.asr_model.text()),
            ("VAD", self.vad_model.text()),
        ):
            if not test_mode and not Path(value).expanduser().is_dir():
                raise ValueError(f"{name} 模型目录不存在，请重新选择。")
        return DesktopOptions(
            settings,
            microphone,
            system,
            Path(self.asr_model.text()).expanduser(),
            Path(self.vad_model.text()).expanduser(),
            test_mode=test_mode,
        )

    def launch(self) -> None:
        if self._closing:
            return
        try:
            options = self._load_options()
        except (ValidationError, ValueError) as exc:
            self.error.setText(
                "请提供有效的 .env 或 API Key，并检查配置数值。"
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            self.error.show()
            self._show_settings()
            return
        self.error.hide()
        self.settings_dialog.hide()
        self.hide()
        self._ready = False
        self._active_epoch = -1
        self.loading.message.setText("正在准备…")
        self.loading.show()
        if self.worker is not None:
            self._restart_options = options
            self.worker.request_stop()
        else:
            self._start_worker(options)

    def _start_worker(self, options: DesktopOptions) -> None:
        self.feed.clear()
        self._active_message = None
        self._failure = ""
        if self.composer is not None:
            self._chat_layout.removeWidget(self.composer)
            self.composer.hide()
            self.composer.deleteLater()
            self.composer = None
        if options.test_mode:
            self.composer = TestComposer()
            self.composer.submitted.connect(self._send_test_message)
            self.composer.setEnabled(False)
            self.composer.details.setText(options.settings.model)
            self.composer.details.setToolTip(
                f"思考：{options.settings.reasoning_effort}\n"
                f"云端输出上限：{options.settings.max_output_tokens} tokens\n"
                "复用正式模式的上下文、提示词与本地输出限制。"
            )
            self._chat_layout.addWidget(self.composer)
        worker = AssistantWorker(options, self)
        self.worker = worker
        worker.set_repository(
            self.workspace.index,
            pending=self.workspace.root is not None and self.workspace.index is None,
            root=self.workspace.root,
        )

        def current(function):
            return lambda *args: function(*args) if self.worker is worker else None

        worker.ready.connect(current(self._became_ready))
        worker.status.connect(current(self._status))
        worker.response_started.connect(current(self._response_started))
        worker.response.connect(current(self._response))
        worker.response_finished.connect(current(self._response_finished))
        worker.transcript.connect(current(self._transcript))
        worker.failed.connect(current(self._failed))
        worker.request_failed.connect(current(self._request_failed))
        worker.metrics.connect(current(self._metrics))
        worker.sources.connect(current(self._sources_ready))
        worker.semantic_progress.connect(current(self._semantic_progress))
        worker.turn_progress.connect(current(self._turn_progress))
        worker.finished.connect(current(self._worker_finished))
        self._save_preferences()
        worker.start()

    def _became_ready(self) -> None:
        if self._closing or self._restart_options is not None:
            return
        self._ready = True
        self.loading.hide()
        self.show()
        test_mode = self.worker is not None and self.worker.options.test_mode
        self.state.setToolTip("测试模式 · 音频已关闭" if test_mode else "正在监听")
        self.state.setStyleSheet("color: #c6a0f6;" if test_mode else "")
        if self.composer is not None:
            self.composer.setEnabled(True)
            self.composer.editor.setFocus()

    def _send_test_message(self) -> None:
        if (
            not self._ready
            or self._closing
            or self.composer is None
            or self.worker is None
        ):
            return
        text = self.composer.editor.toPlainText().strip()
        if self.worker.submit_text(text, self.composer.role.currentData()):
            self.composer.editor.clear()
            self.composer.editor.setFocus()

    def _request_failed(self, epoch: int, text: str) -> None:
        if self._ready and not self._closing and epoch == self._active_epoch:
            self.feed.add_message("ai", f"请求失败：{text}")
            if self.composer is not None:
                self.composer.details.setText(
                    f"{self.worker.options.settings.model} · 请求失败，可重新发送"
                )

    def _metrics(self, epoch: int, metrics: ReplyMetrics) -> None:
        if (
            epoch != self._active_epoch
            or not self._ready
            or self._closing
            or self.composer is None
        ):
            return
        first = (
            f"首字 {metrics.first_token_seconds:.2f}s"
            if metrics.first_token_seconds is not None
            else "无文本输出"
        )
        finish = {
            "local_limit": "本地截断",
            "provider_limit": "云端截断",
        }.get(metrics.finish_reason, "完成")
        self.composer.details.setText(
            f"{self.worker.options.settings.model} · {first} · "
            f"{metrics.elapsed_seconds:.2f}s · {finish}"
            + (
                f" · 检索 {metrics.retrieval_seconds:.3f}s · {metrics.source_count} 处来源"
                if self.workspace.root is not None
                else ""
            )
        )
        if self._active_message is not None:
            self._active_message.setToolTip(
                f"{self.worker.options.settings.model}\n{self.composer.details.text()}\n"
                f"显示字符：{metrics.output_characters}"
            )

    def _status(self, text: str) -> None:
        if not self._closing:
            self.loading.message.setText(text)
            self.state.setToolTip(text)

    def _refresh_activity(self) -> None:
        def display(text: str, started: float | None) -> str:
            return f"{text} · {perf_counter() - started:.0f}s" if started else text

        self.index_activity.setText(display(self._index_text, self._index_started))
        self.answer_activity.setText(display(self._answer_text, self._answer_started))

    def _semantic_progress(self, progress: SemanticProgress) -> None:
        if self._closing:
            return
        device = f" · {progress.device.upper()}" if progress.device else ""
        stages = {
            "unavailable": "模型未找到，使用现有检索",
            "waiting": "等待工作区",
            "queued": "等待后台建索引",
            "scanning": "正在统计已选代码片段",
            "comparing": "正在核对已有向量",
            "loading": "正在加载本地模型",
            "encoding": f"正在编码 {progress.completed}/{progress.total} 片段（复用 {progress.reused}）",
            "cleaning": "正在清理旧片段",
            "ready": f"已就绪 · {progress.total} 片段（复用 {progress.reused}）",
            "failed": "建索引失败，使用现有检索（详见终端）",
        }
        self._index_text = "本地向量索引：" + stages[progress.stage] + device
        if progress.stage in {"queued", "scanning"}:
            self._index_started = perf_counter()
        elif progress.stage in {"ready", "failed", "unavailable", "waiting"}:
            self._index_started = None
        if progress.stage == "encoding" and progress.total:
            self.index_progress.setRange(0, progress.total)
            self.index_progress.setValue(progress.completed)
            self.index_progress.show()
        else:
            self.index_progress.hide()
        self._refresh_activity()

    def _turn_progress(self, epoch: int, stage: str) -> None:
        if not self._closing and self._ready and epoch == self._active_epoch:
            self._answer_text = "回答：" + stage
            self._refresh_activity()

    def _open_settings(self) -> None:
        if self.worker is not None:
            self.test_mode.setChecked(self.worker.options.test_mode)
        self._show_settings()

    def _show_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _configuration_closed(self) -> None:
        if not self._ready and self.worker is None:
            self.close()

    def _transcript(self, role: str, text: str, current_question: bool = False) -> None:
        if not text.strip() or self._closing or not self._ready:
            return
        self.feed.add_message("computer" if role == "面试官" else "microphone", text)
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
                self.feed.remove_message(self._active_message)
                self._active_message = None

    def _response_started(self, epoch: int) -> None:
        if self._closing or not self._ready:
            return
        self._end_reply("已中断")
        self._active_epoch = epoch
        self._active_message = None
        self._answer_started = perf_counter()
        self._answer_text = "回答：正在准备问题与代码上下文…"
        self._refresh_activity()
        if self.composer is not None and self.worker is not None:
            self.composer.details.setText(
                f"{self.worker.options.settings.model} · 生成中…"
            )

    def _response(self, epoch: int, text: str) -> None:
        if epoch == self._active_epoch and self._ready and not self._closing and text:
            self._ensure_reply().append_text(text)

    def _response_finished(self, epoch: int, state: str) -> None:
        if epoch == self._active_epoch:
            self._end_reply(state)
            self._answer_started = None
            self._answer_text = "回答：" + state
            self._refresh_activity()

    def _failed(self, text: str) -> None:
        self._failure = text
        self._end_reply("失败")
        self._active_epoch = -1
        self._answer_started = None
        self._answer_text = "回答：失败"
        self._refresh_activity()

    def _worker_finished(self) -> None:
        worker, self.worker = self.worker, None
        self._ready = False
        self._active_epoch = -1
        self.loading.hide()
        self.hide()
        if worker is not None:
            worker.deleteLater()
        if self._closing:
            self.close()
        elif self._restart_options is not None:
            options, self._restart_options = self._restart_options, None
            self.loading.show()
            self._start_worker(options)
        else:
            self.error.setText(self._failure or "音频监听已结束，请检查设备后重试。")
            self.error.show()
            self._show_settings()

    def _save_preferences(self) -> None:
        for key, value in {
            "env": self.env_file.text(),
            "asr": self.asr_model.text(),
            "vad": self.vad_model.text(),
            "microphone_name": self.microphone.currentText(),
            "system_name": self.system.currentText(),
            "test_mode": self.test_mode.isChecked(),
        }.items():
            self.preferences.setValue(key, value)
        self.preferences.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        workspace_stopped = self.workspace.shutdown()
        self._ready = False
        self._active_epoch = -1
        self.settings_dialog.hide()
        self.loading.hide()
        if self.worker is not None:
            self.worker.request_stop()
            event.ignore()
            return
        if not workspace_stopped:
            event.ignore()
            return
        event.accept()
        if self._manage_application:
            QTimer.singleShot(0, QApplication.instance().quit)
