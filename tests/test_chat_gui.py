"""Real Qt widgets: role messages, streaming, auto-scroll, and worker lifecycle."""

import asyncio
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from contextshift.desktop import worker as worker_module
from contextshift.desktop.chat import Avatar, ChatFeed, Composer
from contextshift.desktop.window import MainWindow
from contextshift.speech.events import SpeechStarted, TranscriptReady


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def wait_until(application, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.005)
    application.processEvents()
    assert predicate()


def settle(application):
    for _ in range(12):
        application.processEvents()


@pytest.fixture
def window(application, tmp_path):
    settings = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    widget = MainWindow(preferences=settings, discover_devices=False)
    widget.audio_enabled.setChecked(False)
    widget.env_file.setText(str(tmp_path / ".env"))
    widget.show()
    settle(application)
    yield widget
    widget.close()
    wait_until(application, lambda: widget.worker is None)
    widget.close()


def test_empty_chat_is_dark_and_ready_for_configuration(window):
    assert window.feed.welcome.isVisible()
    assert not window.question.isEnabled()
    assert not window.ask_button.isEnabled()
    assert "#151722" in window.styleSheet()
    assert not window.settings_dialog.isVisible()
    window.settings_button.click()
    assert window.settings_dialog.isVisible()


def test_role_messages_have_distinct_cartoon_avatars(window, application):
    window._transcript("候选人", "我主要负责缓存模块。")
    window._transcript("面试官", "更新失败时怎么办？")
    window._response_started(1)
    window._response(1, "保留旧缓存。")
    window._response_finished(1, "已完成")
    assert [m.role for m in window.feed.messages] == ["microphone", "computer", "ai"]
    assert [m.findChild(Avatar).role for m in window.feed.messages] == [
        "microphone",
        "computer",
        "ai",
    ]
    assert window.feed.messages[-1].state == "已完成"
    settle(application)
    icons = [m.findChild(Avatar).grab().toImage() for m in window.feed.messages]
    assert icons[0] != icons[1] and icons[1] != icons[2]


def test_question_precedes_pending_analysis_and_chunks_share_one_bubble(window):
    window._response_started(1)
    assert not window.feed.messages
    window._transcript("面试官", "问题", True)
    assert [m.role for m in window.feed.messages] == ["computer", "ai"]
    for part in ("结论：", "保留", "旧缓存。"):
        window._response(1, part)
    assert len(window.feed.messages) == 2
    assert window.feed.messages[-1].text == "结论：保留旧缓存。"


def test_interruption_preserves_old_text_and_drops_late_tokens(window):
    window._response_started(1)
    window._response(1, "旧提示")
    old = window.feed.messages[-1]
    window._response_started(2)
    window._response(1, "过期 token")
    window._transcript("面试官", "新问题", True)
    window._response(2, "新提示")
    assert old.text == "旧提示" and old.state == "已中断"
    assert window.feed.messages[-1].text == "新提示"


def test_cancel_and_clear_reject_queued_output(window):
    window._response_started(1)
    window._response(1, "提示")
    window._cancel_answer()
    window._response(1, "晚到的文字")
    assert window.feed.messages[-1].text == "提示"
    assert window.feed.messages[-1].state == "已取消"
    window._clear_chat()
    window._response(1, "不能重新出现")
    assert not window.feed.messages
    assert window.feed.welcome.isVisible()


@pytest.mark.parametrize(
    "text", ["多行\n消息\n保持换行", "<b>这不是 HTML</b>", "界" * 600, "x" * 600]
)
def test_message_content_wraps_without_horizontal_overflow(window, application, text):
    window.resize(740, 630)
    message = window.feed.add_message("computer", text)
    settle(application)
    assert message.body.text() == text
    assert message.body.textFormat() == Qt.TextFormat.PlainText
    assert window.feed.widget().width() <= window.feed.viewport().width()


def test_new_messages_and_streaming_always_scroll_to_bottom(window, application):
    for index in range(25):
        window.feed.add_message("microphone", f"第 {index} 段发言。\n" * 3)
    settle(application)
    bar = window.feed.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    bar.setValue(0)
    settle(application)
    assert bar.value() == 0
    window.feed.add_message("computer", "新的问题")
    settle(application)
    assert bar.value() == bar.maximum()
    message = window.feed.add_message("ai", state="分析中")
    settle(application)
    for part in ("流式内容\n" * 20, "继续生成\n" * 20):
        bar.setValue(0)
        message.append_text(part)
        settle(application)
        assert bar.value() == bar.maximum()


def test_chat_limit_preserves_active_reply(application):
    feed = ChatFeed(max_messages=3)
    active = feed.add_message("ai", state="生成中")
    for number in range(10):
        feed.add_message("microphone", str(number))
    assert len(feed.messages) == 3
    assert active in feed.messages
    active.append_text("仍可更新")
    feed.close()
    with pytest.raises(ValueError):
        ChatFeed(max_messages=0)


def test_enter_sends_and_shift_enter_inserts_newline(application):
    composer = Composer()
    composer.setPlainText("你好")
    composer.moveCursor(QTextCursor.MoveOperation.End)
    spy = QSignalSpy(composer.submitted)
    QTest.keyClick(composer, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert composer.toPlainText() == "你好\n" and spy.count() == 0
    QTest.keyClick(composer, Qt.Key.Key_Return)
    assert spy.count() == 1
    composer.close()


def test_invalid_settings_do_not_launch_worker_or_echo_key(window, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    window._toggle_running()
    assert window.worker is None
    assert "API Key" in window.error.text()
    assert window.settings_dialog.isVisible()


def test_real_worker_manual_question_streams_and_stops(
    window, application, monkeypatch
):
    requests = []
    closed = threading.Event()

    class Provider:
        def __init__(self, settings):
            pass

        async def stream(self, request):
            requests.append(request.input)
            yield "第一段"
            await asyncio.Event().wait()

        async def close(self):
            closed.set()

    monkeypatch.setattr(worker_module, "DeepSeekProvider", Provider)
    window.api_key.setText("test-key-not-persisted")
    window._toggle_running()
    wait_until(application, window.ask_button.isEnabled)
    window.question.setPlainText("解释缓存失效")
    window._ask()
    wait_until(
        application,
        lambda: window.feed.messages and window.feed.messages[-1].text == "第一段",
    )
    assert [m.role for m in window.feed.messages] == ["microphone", "ai"]
    assert requests == ["解释缓存失效"]
    window._stop_assistant()
    assert not window.ask_button.isEnabled()
    wait_until(application, lambda: window.worker is None)
    assert closed.is_set()
    assert window.feed.messages[-1].state == "已停止"
    window.preferences.sync()
    assert (
        "test-key-not-persisted" not in Path(window.preferences.fileName()).read_text()
    )


def test_real_worker_routes_both_audio_roles_into_chat(
    window, application, monkeypatch, tmp_path
):
    class Provider:
        def __init__(self, settings):
            pass

        async def stream(self, request):
            yield "保留旧缓存。"

        async def close(self):
            pass

    async def audio(**kwargs):
        for source, text in (("microphone", "我负责缓存"), ("system", "失败怎么办")):
            kwargs["on_speech_started"](SpeechStarted(1, 0, 0, source=source))
            kwargs["on_transcript"](
                TranscriptReady(1, 0, 100, text, 0.01, 0.1, source=source)
            )
        await asyncio.Event().wait()

    monkeypatch.setattr(worker_module, "DeepSeekProvider", Provider)
    monkeypatch.setattr(worker_module, "run_dual_audio_listener", audio)
    window.api_key.setText("test-only")
    window.microphone.addItem("Mic", 1)
    window.microphone.setCurrentIndex(1)
    window.system.addItem("BlackHole", 0)
    window.system.setCurrentIndex(1)
    window.audio_enabled.setChecked(True)
    window.asr_model.setText(str(tmp_path))
    window.vad_model.setText(str(tmp_path))
    window._toggle_running()
    wait_until(
        application,
        lambda: (
            window.feed.messages and window.feed.messages[-1].text == "保留旧缓存。"
        ),
    )
    assert [m.role for m in window.feed.messages] == ["microphone", "computer", "ai"]
    assert window.feed.messages[-1].state == "已完成"


def test_close_waits_for_worker_cleanup(window, application, monkeypatch):
    closed = threading.Event()

    class Provider:
        def __init__(self, settings):
            pass

        async def close(self):
            closed.set()

    monkeypatch.setattr(worker_module, "DeepSeekProvider", Provider)
    window.api_key.setText("test-only")
    window._toggle_running()
    wait_until(application, window.ask_button.isEnabled)
    window.close()
    wait_until(application, lambda: window.worker is None)
    assert closed.is_set()
    assert not window.isVisible()
