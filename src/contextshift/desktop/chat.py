"""Dark terminal-inspired conversation feed with painted role avatars.

The scene is a quiet evening desktop: ink surfaces, teal microphone, blue
computer and lilac robot. Each mascot is drawn inside its own rounded frame;
the mic rests on its stand, the monitor on its base, the robot antenna on its
head. Icons use Qt geometry and the system font, with no external assets.
"""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QInputMethodEvent,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QTextOption,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

Role = Literal["microphone", "computer", "ai"]


class TestInput(QPlainTextEdit):
    submitted = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._composing = False
        self.setFixedHeight(82)
        self.setAccessibleName("测试消息")

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:
        self._composing = bool(event.preeditString())
        super().inputMethodEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and not self._composing
        ):
            if not event.isAutoRepeat():
                self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class TestComposer(QWidget):
    """Only test sessions expose typed role simulation and diagnostics."""

    submitted = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 8, 8, 0)
        column.setSpacing(8)
        row = QHBoxLayout()
        self.role = QComboBox()
        self.role.setAccessibleName("模拟发言角色")
        self.role.addItem("对方提问", "interviewer")
        self.role.addItem("我的发言", "candidate")
        self.role.setToolTip("对方提问触发回答；我的发言只补充上下文。")
        self.details = QLabel()
        self.details.setObjectName("testDetails")
        self.details.setWordWrap(True)
        row.addWidget(self.role)
        row.addWidget(self.details, 1)
        self.send = QPushButton("发送 ↵")
        self.send.setObjectName("primary")
        row.addWidget(self.send)
        column.addLayout(row)
        self.editor = TestInput()
        self.editor.setToolTip("Enter 发送 · Shift+Enter 换行")
        column.addWidget(self.editor)
        self.role.currentIndexChanged.connect(self._role_changed)
        self.editor.textChanged.connect(self._update_send)
        self.editor.submitted.connect(self.submitted)
        self.send.clicked.connect(self.submitted)
        self._role_changed()

    def _role_changed(self) -> None:
        self.editor.setPlaceholderText(
            "输入对方的问题，测试回答…"
            if self.role.currentData() == "interviewer"
            else "输入你已说过的话，作为下一次回答的上下文…"
        )
        self._update_send()

    def _update_send(self) -> None:
        self.send.setEnabled(bool(self.editor.toPlainText().strip()))


@dataclass(frozen=True)
class RoleStyle:
    title: str
    accent: str
    background: str


ROLES = {
    "microphone": RoleStyle("我的发言", "#8bd5ca", "#182a2b"),
    "computer": RoleStyle("对方说话", "#8aadf4", "#1b2439"),
    "ai": RoleStyle("AI 分析", "#c6a0f6", "#292139"),
}


class Avatar(QWidget):
    """Small cartoon glyphs remain crisp at Retina display scale."""

    def __init__(self, role: Role, parent=None) -> None:
        super().__init__(parent)
        self.role = role
        self.setFixedSize(44, 44)
        self.setAccessibleName(ROLES[role].title + "头像")

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(ROLES[self.role].accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ROLES[self.role].background))
        painter.drawRoundedRect(QRectF(0, 0, 44, 44), 13, 13)
        pen = QPen(
            color,
            1.7,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        painter.setBrush(QColor("#161b27"))
        if self.role == "microphone":
            painter.drawRoundedRect(QRectF(15, 7, 14, 22), 7, 7)
            highlight = QColor(color)
            highlight.setAlpha(40)
            painter.fillRect(QRectF(18, 10, 3, 8), highlight)
            path = QPainterPath(QPointF(11, 22))
            path.cubicTo(11, 36, 33, 36, 33, 22)
            painter.drawPath(path)
            painter.drawLine(QPointF(22, 33), QPointF(22, 38))
            painter.drawLine(QPointF(16, 38), QPointF(28, 38))
            self._face(painter, 19, 22, 6, color)
        elif self.role == "computer":
            painter.drawRoundedRect(QRectF(7, 8, 30, 24), 4, 4)
            painter.setPen(QPen(QColor("#435574"), 1))
            painter.drawRoundedRect(QRectF(10, 11, 24, 16), 2, 2)
            painter.setPen(pen)
            painter.drawLine(QPointF(22, 32), QPointF(22, 37))
            painter.drawLine(QPointF(15, 38), QPointF(29, 38))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(30, 29, 2, 2))
            self._face(painter, 18, 18, 8, color)
        else:
            painter.drawLine(QPointF(22, 12), QPointF(22, 7))
            painter.setBrush(color)
            painter.drawEllipse(QRectF(20, 4, 4, 4))
            painter.setBrush(QColor("#161b27"))
            painter.drawRoundedRect(QRectF(10, 13, 24, 23), 7, 7)
            painter.drawRoundedRect(QRectF(6, 20, 4, 10), 2, 2)
            painter.drawRoundedRect(QRectF(34, 20, 4, 10), 2, 2)
            painter.drawLine(QPointF(18, 31), QPointF(26, 31))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(15, 21, 4, 5), 2, 2)
            painter.drawRoundedRect(QRectF(25, 21, 4, 5), 2, 2)

    @staticmethod
    def _face(painter, x: float, y: float, spacing: float, color: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QRectF(x - 1, y - 1, 2.4, 2.4))
        painter.drawEllipse(QRectF(x + spacing - 1, y - 1, 2.4, 2.4))
        painter.setPen(QPen(color, 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(
            QRectF(x + 1, y + 1, max(3, spacing - 1), 4), 190 * 16, 160 * 16
        )


class MessageText(QTextBrowser):
    """Grow with content and wrap even long paths and unbroken identifiers."""

    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("message")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setReadOnly(True)
        self.setOpenLinks(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(0)
        option = self.document().defaultTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().setDefaultTextOption(option)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit)
        self.document().contentsChanged.connect(lambda: self._fit_timer.start(0))

    def text(self) -> str:
        return self.toPlainText()

    def textFormat(self):
        return Qt.TextFormat.PlainText

    def setText(self, text: str) -> None:
        self.setPlainText(text)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self.textCursor().hasSelection()
        ):
            self.clicked.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_fit_timer"):
            self._fit_timer.start(0)

    def _fit(self) -> None:
        self.document().setTextWidth(max(1, self.viewport().width()))
        height = max(22, ceil(self.document().size().height()) + 2)
        if self.height() != height:
            self.setFixedHeight(height)
            self.updateGeometry()


class ChatMessage(QWidget):
    changed = Signal()

    def __init__(
        self,
        role: Role,
        text: str = "",
        *,
        title: str | None = None,
        state: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.role = role
        self.text = text
        self.state = state
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(Avatar(role), 0, Qt.AlignmentFlag.AlignTop)
        self.bubble = QFrame()
        self.bubble.setObjectName("bubbleAI" if role == "ai" else "bubble")
        column = QVBoxLayout(self.bubble)
        column.setContentsMargins(17, 12, 17, 14)
        column.setSpacing(7)
        self.setAccessibleName(title or ROLES[role].title)
        self.body = MessageText()
        self.body.clicked.connect(self._open_primary_source)
        self._references = ()
        self._open_source = None
        column.addWidget(self.body)
        self.sources = QWidget()
        self._source_layout = QGridLayout(self.sources)
        self._source_layout.setContentsMargins(0, 0, 0, 0)
        self._source_layout.setSpacing(4)
        self.sources.hide()
        column.addWidget(self.sources)
        row.addWidget(self.bubble, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.set_text(text)

    def set_sources(self, references, open_source):
        self._references = tuple(references)
        self._open_source = open_source
        if self.role == "ai" and self._references:
            self.body.setCursor(Qt.CursorShape.PointingHandCursor)
            self.body.setToolTip("点击回答打开主要来源；点击下方来源定位具体代码")
        while self._source_layout.count():
            item = self._source_layout.takeAt(0)
            item.widget().deleteLater()
        for i, reference in enumerate(references):
            symbol = reference.symbol.split(",", 1)[0].strip()
            root = Path(reference.workspace).name if reference.workspace else ""
            name = reference.path.rsplit("/", 1)[-1]
            label = (
                f"{root}/{name}:{reference.start_line}"
                if root
                else f"{name}:{reference.start_line}"
            )
            if symbol and symbol != "模块声明":
                label += f" · {symbol}"
            button = QPushButton(label)
            button.setObjectName("sourceLink")
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            button.setMinimumWidth(0)
            button.setToolTip(
                f"{reference.path}:{reference.start_line}-{reference.end_line}\n{reference.symbol}"
            )
            button.clicked.connect(
                lambda checked=False, ref=reference: open_source(ref)
            )
            self._source_layout.addWidget(button, i // 2, i % 2)
        self.sources.setVisible(bool(references))
        self.changed.emit()

    def _open_primary_source(self) -> None:
        if self.role == "ai" and self._references and self._open_source is not None:
            self._open_source(self._references[0])

    @property
    def pinned(self) -> bool:
        return self.state in {"分析中", "生成中"}

    def set_text(self, text: str) -> None:
        self.text = text
        self.body.setText(text if text else "•••")
        self.body.setStyleSheet("color: #a6adc8;" if not text else "")
        self.body.updateGeometry()
        self.updateGeometry()
        self.changed.emit()

    def append_text(self, text: str) -> None:
        self.set_state("生成中")
        self.set_text(self.text + text)

    def set_state(self, state: str) -> None:
        self.state = state
        self.setToolTip(state)
        self.changed.emit()


class ChatFeed(QScrollArea):
    def __init__(self, parent=None, *, max_messages: int = 300) -> None:
        super().__init__(parent)
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        self.max_messages = max_messages
        self.messages: list[ChatMessage] = []
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setObjectName("feed")
        self.content = QWidget()
        self.content.setObjectName("feedContent")
        self.column = QVBoxLayout(self.content)
        self.column.setContentsMargins(24, 24, 24, 24)
        self.column.setSpacing(18)
        self.welcome = self._welcome()
        self.column.addWidget(self.welcome)
        self.column.addStretch(1)
        self.setWidget(self.content)
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._scroll_to_end)
        self.verticalScrollBar().rangeChanged.connect(self.scroll_to_latest)

    def _welcome(self) -> QWidget:
        welcome = QWidget()
        welcome.setMinimumHeight(180)
        column = QVBoxLayout(welcome)
        column.setSpacing(17)
        column.addStretch()
        column.addWidget(Avatar("ai"), 0, Qt.AlignmentFlag.AlignHCenter)
        column.addStretch()
        return welcome

    def add_message(
        self, role: Role, text: str = "", *, title: str | None = None, state: str = ""
    ) -> ChatMessage:
        self.welcome.hide()
        message = ChatMessage(role, text, title=title, state=state)
        message.changed.connect(self.scroll_to_latest)
        self.column.insertWidget(self.column.count() - 1, message)
        self.messages.append(message)
        while len(self.messages) > self.max_messages:
            oldest = next((m for m in self.messages if not m.pinned), None)
            if oldest is None:
                break
            self.messages.remove(oldest)
            self.column.removeWidget(oldest)
            oldest.deleteLater()
        self.scroll_to_latest()
        return message

    def scroll_to_latest(self, *_args) -> None:
        if not self._scroll_timer.isActive():
            self._scroll_timer.start(0)

    def _scroll_to_end(self) -> None:
        self.column.activate()
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def remove_message(self, message: ChatMessage) -> None:
        if message in self.messages:
            self.messages.remove(message)
            self.column.removeWidget(message)
            message.deleteLater()
            self.scroll_to_latest()

    def clear(self) -> None:
        for message in self.messages:
            self.column.removeWidget(message)
            message.deleteLater()
        self.messages.clear()
        self.welcome.show()
        self.scroll_to_latest()
