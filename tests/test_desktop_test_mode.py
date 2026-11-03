"""Exercise the typed path through the real Qt worker/session/history chain."""

import asyncio
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QInputMethodEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from contextshift.desktop import worker as worker_module
from contextshift.desktop.window import MainWindow
from contextshift.speech.events import SpeechStarted, TranscriptReady


def wait(app, predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(requests=[], providers=[], handler=None, streams_closed=[])

    class Provider:
        def __init__(self, settings):
            self.closed = False
            state.providers.append(self)

        async def prepare(self):
            pass

        async def stream(self, request):
            state.requests.append(request)
            index = len(state.requests)
            try:
                if state.handler is None:
                    yield "测试建议"
                else:
                    async for chunk in state.handler(index, request):
                        yield chunk
            finally:
                state.streams_closed.append(index)

        async def close(self):
            self.closed = True

    def forbidden(**kwargs):
        raise AssertionError("test mode must not initialize speech or capture")

    monkeypatch.setattr(worker_module, "DeepSeekProvider", Provider)
    monkeypatch.setattr(worker_module, "prepare_channels", forbidden)
    monkeypatch.setattr(worker_module, "run_dual_audio_listener", forbidden)
    return state


@pytest.fixture
def window(app, tmp_path, runtime):
    prefs = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    view = MainWindow(preferences=prefs, discover_devices=False)
    view.env_file.setText(str(tmp_path / "missing.env"))
    view.api_key.setText("test-only-key")
    view.asr_model.setText(str(tmp_path / "no-asr"))
    view.vad_model.setText(str(tmp_path / "no-vad"))
    view.test_mode.setChecked(True)
    yield view
    view.close()
    wait(app, lambda: view.worker is None)
    view.close()


def launch(app, window):
    window.launch()
    wait(app, lambda: window.isVisible() and window._ready)


def send(window, text, role="interviewer"):
    composer = window.composer
    composer.role.setCurrentIndex(composer.role.findData(role))
    composer.editor.setPlainText(text)
    composer.send.click()


def test_test_mode_needs_only_cloud_configuration(app, window):
    launch(app, window)
    assert window.worker.options.test_mode
    assert window.worker.options.microphone is None
    assert window.worker.options.system is None
    assert window.composer.isVisible()
    assert "音频已关闭" in window.state.toolTip()
    assert not window.asr_model.isEnabled() and not window.microphone.isEnabled()
    assert not window.composer.send.isEnabled()


def test_roles_use_actual_history_and_never_reuse_model_suggestions(
    app, window, runtime
):
    launch(app, window)
    send(window, "我负责 Rust 缓存模块", "candidate")
    wait(app, lambda: len(window.feed.messages) == 1)
    assert runtime.requests == []
    send(window, "更新失败怎么办？")
    wait(app, lambda: runtime.streams_closed == [1])
    assert "候选人：我负责 Rust 缓存模块" in runtime.requests[0].input
    assert "面试官问题：更新失败怎么办？" in runtime.requests[0].input
    send(window, "为什么这样处理？")
    wait(app, lambda: len(runtime.requests) == 2)
    assert "更新失败怎么办？" in runtime.requests[1].input
    assert "测试建议" not in runtime.requests[1].input
    assert [m.role for m in window.feed.messages[:3]] == [
        "microphone",
        "computer",
        "ai",
    ]


def test_enter_sends_shift_enter_and_ime_confirmation_do_not(app, window, runtime):
    launch(app, window)
    editor = window.composer.editor
    editor.setPlainText("第一行")
    editor.moveCursor(editor.textCursor().MoveOperation.End)
    QTest.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    assert editor.toPlainText() == "第一行\n"
    assert runtime.requests == []
    QApplication.sendEvent(editor, QInputMethodEvent("输入中", []))
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert runtime.requests == []
    QApplication.sendEvent(editor, QInputMethodEvent("", []))
    editor.setPlainText("真实提交")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    wait(app, lambda: len(runtime.requests) == 1)
    assert "真实提交" in runtime.requests[0].input
    assert editor.toPlainText() == ""


@pytest.mark.parametrize("text", ["", "  \n\t  "])
def test_blank_input_is_ignored(app, window, runtime, text):
    launch(app, window)
    send(window, text)
    window._send_test_message()
    assert runtime.requests == [] and window.feed.messages == []
    assert window.composer.editor.toPlainText() == text


def test_candidate_context_does_not_interrupt_but_new_question_does(
    app, window, runtime
):
    async def stream(index, request):
        if index == 1:
            yield "旧提示"
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                yield "过期 token"
        else:
            yield "新提示"

    runtime.handler = stream
    launch(app, window)
    send(window, "旧问题")
    wait(app, lambda: any(m.text == "旧提示" for m in window.feed.messages))
    old = window.feed.messages[-1]
    send(window, "我刚才补充的话", "candidate")
    wait(app, lambda: window.feed.messages[-1].role == "microphone")
    assert runtime.streams_closed == [] and old.pinned
    send(window, "新问题")
    wait(app, lambda: any(m.text == "新提示" for m in window.feed.messages))
    assert old.text == "旧提示" and old.state == "已中断"
    assert not any("过期 token" in m.text for m in window.feed.messages)
    assert "我刚才补充的话" in runtime.requests[-1].input


def test_request_error_stays_in_test_mode_and_can_retry(app, window, runtime):
    async def stream(index, request):
        if index == 1:
            raise RuntimeError("test failure sk-hidden-secret")
        yield "重试成功"

    runtime.handler = stream
    launch(app, window)
    worker = window.worker
    send(window, "第一次")
    wait(app, lambda: any("请求失败" in m.text for m in window.feed.messages))
    assert window.isVisible() and window.worker is worker
    assert "sk-hidden-secret" not in window.feed.messages[-1].text
    send(window, "第二次")
    wait(app, lambda: any(m.text == "重试成功" for m in window.feed.messages))
    assert len(runtime.providers) == 1
    assert "test failure" not in runtime.requests[-1].input


def test_cancelled_request_cleanup_error_cannot_fail_the_new_question(
    app, window, runtime
):
    async def stream(index, request):
        if index == 1:
            yield "旧回答"
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("旧连接关闭失败") from None
        else:
            await asyncio.sleep(0.02)
            yield "新回答"

    runtime.handler = stream
    launch(app, window)
    send(window, "旧问题")
    wait(app, lambda: any(m.text == "旧回答" for m in window.feed.messages))
    send(window, "新问题")
    wait(app, lambda: any(m.text == "新回答" for m in window.feed.messages))
    assert not any("请求失败" in m.text for m in window.feed.messages)


@pytest.mark.parametrize(
    "answer,reason", [("", "completed"), ("x" * 400, "local_limit")]
)
def test_metrics_and_limit_use_the_same_production_session(
    app, window, runtime, answer, reason
):
    async def stream(index, request):
        yield answer

    runtime.handler = stream
    launch(app, window)
    measured = []
    window.worker.metrics.connect(lambda epoch, metrics: measured.append(metrics))
    send(window, "检查边界输出")
    wait(app, lambda: bool(measured))
    result = measured[-1]
    assert result.finish_reason == reason
    assert result.elapsed_seconds >= 0
    if answer:
        assert 0 <= result.first_token_seconds <= result.elapsed_seconds
        assert 0 < result.output_characters < len(answer)
        assert "本地截断" in window.composer.details.text()
    else:
        assert result.first_token_seconds is None and result.output_characters == 0
        assert "无文本输出" in window.composer.details.text()


def test_switch_to_live_clears_test_history_and_removes_composer(
    app, window, runtime, monkeypatch, tmp_path
):
    launch(app, window)
    send(window, "测试模式专属上下文", "candidate")
    wait(app, lambda: len(window.feed.messages) == 1)
    old_provider = runtime.providers[0]
    callbacks = {}

    async def audio(**kwargs):
        callbacks.update(kwargs)
        kwargs["on_ready"]()
        await asyncio.Event().wait()

    monkeypatch.setattr(worker_module, "prepare_channels", lambda **kwargs: {})
    monkeypatch.setattr(worker_module, "run_dual_audio_listener", audio)
    window._open_settings()
    window.test_mode.setChecked(False)
    window.asr_model.setText(str(tmp_path))
    window.vad_model.setText(str(tmp_path))
    window.microphone.addItem("Microphone", 1)
    window.system.addItem("BlackHole", 0)
    window.microphone.setCurrentIndex(1)
    window.system.setCurrentIndex(1)
    window.apply_button.click()
    wait(app, lambda: window._ready and window.worker.options.test_mode is False)
    assert old_provider.closed and window.composer is None
    assert window.feed.messages == []
    callbacks_loop = window.worker._loop

    def question():
        callbacks["on_speech_started"](SpeechStarted(1, 0, 0, source="system"))
        callbacks["on_transcript"](
            TranscriptReady(1, 0, 1, "正式问题", 0, 0, source="system")
        )

    callbacks_loop.call_soon_threadsafe(question)
    wait(app, lambda: bool(runtime.requests))
    assert "测试模式专属上下文" not in runtime.requests[-1].input
    assert not window.worker.submit_text("禁止正式模式打字", "interviewer")


def test_saved_test_mode_skips_device_discovery(app, tmp_path, monkeypatch):
    prefs = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    prefs.setValue("test_mode", True)

    def forbidden():
        pytest.fail("saved text mode should not enumerate PortAudio devices")

    monkeypatch.setattr("contextshift.desktop.window.list_input_devices", forbidden)
    window = MainWindow(preferences=prefs)
    assert window.test_mode.isChecked() and window.microphone.currentData() is None
    window.close()


def test_submit_after_close_is_rejected_and_secrets_are_not_saved(app, window, runtime):
    launch(app, window)
    worker = window.worker
    window.close()
    assert not worker.submit_text("关闭后的输入", "interviewer")
    wait(app, lambda: window.worker is None)
    assert runtime.providers[0].closed
    text = Path(window.preferences.fileName()).read_text()
    assert "test-only-key" not in text and "test_mode=true" in text
