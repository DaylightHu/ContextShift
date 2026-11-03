"""Dark terminal-inspired conversation feed with painted role avatars.

The scene is a quiet evening desktop: ink surfaces, teal microphone, blue
computer and lilac robot. Each mascot is drawn inside its own rounded frame;
the mic rests on its stand, the monitor on its base, the robot antenna on its
head. Icons use Qt geometry and the system font, with no external assets.
"""

from dataclasses import dataclass
from math import ceil
from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

Role = Literal["microphone", "computer", "ai"]


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
        column.addWidget(self.body)
        row.addWidget(self.bubble, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.set_text(text)

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
