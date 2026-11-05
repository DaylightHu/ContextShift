"""Minimal GUI: ready-before-show, voice-only chat, and close-to-quit."""

import asyncio
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton

from contextshift.desktop import worker as worker_module
from contextshift.desktop import workspace as workspace_module
from contextshift.desktop.chat import Avatar, ChatFeed
from contextshift.desktop.window import MainWindow
from contextshift.repository import (
    RepositoryIndex,
    SourceFile,
    SourceReference,
    WorkspaceIndex,
)
from contextshift.semantic_index import SemanticProgress
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
    for _ in range(15):
        application.processEvents()


def test_activity_panel_separates_index_and_current_answer(window):
    window._semantic_progress(SemanticProgress("encoding", 64, 128, "mps", 32))
    assert "64/128" in window.index_activity.text()
    assert "MPS" in window.index_activity.text()
    assert window.index_progress.isVisibleTo(window.activity)
    window._ready = True
    window._response_started(3)
    window._turn_progress(2, "旧回答")
    assert "旧回答" not in window.answer_activity.text()
    window._turn_progress(3, "云端核实源码 · 第 2 轮")
    assert "第 2 轮" in window.answer_activity.text()
    assert "64/128" in window.index_activity.text()
    window._semantic_progress(SemanticProgress("ready", 128, 128, "mps", 32))
    assert "已就绪" in window.index_activity.text()
    assert not window.index_progress.isVisibleTo(window.activity)
    window._response_finished(3, "已完成")
    assert "已完成" in window.answer_activity.text()


@pytest.fixture
def window(application, tmp_path):
    preferences = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    widget = MainWindow(preferences=preferences, discover_devices=False)
    widget.env_file.setText(str(tmp_path / ".env"))
    widget.asr_model.setText(str(tmp_path))
    widget.vad_model.setText(str(tmp_path))
    widget.api_key.setText("test-only-key-not-saved")
    widget.microphone.addItem("MacBook Pro 麦克风", 1)
    widget.system.addItem("BlackHole 2ch", 0)
    widget.microphone.setCurrentIndex(1)
    widget.system.setCurrentIndex(1)
    yield widget
    widget.close()
    wait_until(application, lambda: widget.worker is None)
    widget.close()


def install_runtime(
    monkeypatch,
    *,
    cloud_gate=None,
    models_gate=None,
    audio_gate=None,
    fail_cloud=False,
    fail_models=False,
    fail_audio=False,
    emit_speech=False,
):
    events = []
    closed = threading.Event()

    class Provider:
        def __init__(self, settings):
            pass

        async def prepare(self):
            events.append("cloud_started")
            while cloud_gate is not None and not cloud_gate.is_set():
                await asyncio.sleep(0.005)
            if fail_cloud:
                raise RuntimeError("cloud unavailable")
            events.append("cloud_ready")

        async def stream(self, request):
            events.append("model_request")
            yield "保留旧缓存。"

        async def close(self):
            closed.set()

    def prepare(**kwargs):
        events.append("models_started")
        while models_gate is not None and not models_gate.is_set():
            if kwargs["cancelled"].wait(0.005):
                raise InterruptedError("cancelled")
        if fail_models:
            raise RuntimeError("checkpoint failed")
        events.append("models_ready")
        return {"prepared": True}

    async def audio(**kwargs):
        events.append("audio_opening")
        try:
            while audio_gate is not None and not audio_gate.is_set():
                await asyncio.sleep(0.005)
            if fail_audio:
                raise RuntimeError("audio open failed")
            assert kwargs["prepared_channels"] == {"prepared": True}
            kwargs["on_ready"]()
            events.append("audio_ready")
            if emit_speech:
                for source, text in (
                    ("microphone", "我负责缓存"),
                    ("system", "失败怎么办"),
                ):
                    kwargs["on_speech_started"](SpeechStarted(1, 0, 0, source=source))
                    kwargs["on_transcript"](
                        TranscriptReady(1, 0, 100, text, 0.01, 0.1, source=source)
                    )
            await asyncio.Event().wait()
        finally:
            events.append("audio_closed")

    monkeypatch.setattr(worker_module, "DeepSeekProvider", Provider)
    monkeypatch.setattr(worker_module, "prepare_channels", prepare)
    monkeypatch.setattr(worker_module, "run_dual_audio_listener", audio)
    return events, closed


def test_main_window_contains_only_icons_and_message_content(window):
    window._ready = True
    window.show()
    window._transcript("候选人", "我的真实发言")
    window._response_started(1)
    window._transcript("面试官", "对方的问题", True)
    window._response(1, "提示内容")
    window._response_finished(1, "已完成")
    assert [m.role for m in window.feed.messages] == ["microphone", "computer", "ai"]
    assert not window.centralWidget().findChildren(QLineEdit)
    assert not window.centralWidget().findChildren(QPushButton)
    visible_labels = [
        label.text()
        for label in window.findChildren(QLabel)
        if label.isVisible() and not window.workspace.isAncestorOf(label)
    ]
    assert visible_labels == [
        "●",
        "本地向量索引：等待工作区",
        "回答：已完成",
    ]
    assert [m.findChild(Avatar).role for m in window.feed.messages] == [
        "microphone",
        "computer",
        "ai",
    ]
    assert not hasattr(window, "question")


def test_answer_and_source_links_open_verified_file_in_vscode(
    application, window, tmp_path, monkeypatch
):
    source = tmp_path / "src" / "ApiService.java"
    source.parent.mkdir()
    source.write_text("line one\nline two\n")
    reference = SourceReference(
        "src/ApiService.java", 2, 2, "ApiService", "digest", str(tmp_path)
    )
    indexed_file = SourceFile(
        reference.path, source.read_text(), (0, 0, 0), "digest", (), ()
    )
    window.workspace.index = WorkspaceIndex(
        (RepositoryIndex(tmp_path, (), "", {reference.path: indexed_file}),)
    )
    commands = []
    monkeypatch.setattr(workspace_module.shutil, "which", lambda name: "/usr/bin/code")
    monkeypatch.setattr(
        workspace_module.subprocess,
        "Popen",
        lambda argv, **kwargs: commands.append(argv),
    )
    window._ready = True
    window.show()
    window._response_started(1)
    window._response(1, "ApiService 的处理入口")
    window._sources_ready(1, (reference,), 0.1)
    reply = window.feed.messages[-1]
    settle(application)
    QTest.mouseClick(reply.body.viewport(), Qt.MouseButton.LeftButton)
    reply.sources.findChildren(QPushButton)[0].click()
    assert commands == [
        ["/usr/bin/code", "--goto", f"{source}:2"],
        ["/usr/bin/code", "--goto", f"{source}:2"],
    ]

    outside = SourceReference("../outside.java", 1, 1, "", "digest", str(tmp_path))
    window.workspace.open_source_in_vscode(outside)
    assert len(commands) == 2
    assert "不在当前索引" in window.workspace.status.text()
    assert not hasattr(window, "start_button")


def test_chat_window_is_hidden_until_every_readiness_stage_completes(
    window, application, monkeypatch
):
    cloud, models, audio = threading.Event(), threading.Event(), threading.Event()
    events, _ = install_runtime(
        monkeypatch, cloud_gate=cloud, models_gate=models, audio_gate=audio
    )
    window.launch()
    wait_until(application, lambda: "cloud_started" in events)
    assert not window.isVisible() and window.loading.isVisible()
    assert "models_started" not in events
    cloud.set()
    wait_until(application, lambda: "models_started" in events)
    assert not window.isVisible() and "audio_opening" not in events
    models.set()
    wait_until(application, lambda: "audio_opening" in events)
    assert not window.isVisible()
    audio.set()
    wait_until(application, window.isVisible)
    assert window._ready and not window.loading.isVisible()
    assert events[:5] == [
        "cloud_started",
        "cloud_ready",
        "models_started",
        "models_ready",
        "audio_opening",
    ]


@pytest.mark.parametrize(
    "failure,message",
    [
        ("fail_cloud", "cloud unavailable"),
        ("fail_models", "checkpoint failed"),
        ("fail_audio", "audio open failed"),
    ],
)
def test_startup_failure_never_shows_an_unready_chat(
    window, application, monkeypatch, failure, message
):
    _events, closed = install_runtime(monkeypatch, **{failure: True})
    window.launch()
    wait_until(application, lambda: window.worker is None)
    assert not window.isVisible() and not window.loading.isVisible()
    assert window.settings_dialog.isVisible()
    assert message in window.error.text()
    assert closed.is_set()


@pytest.mark.parametrize("action", ["close", "reject"])
def test_closing_the_loading_window_cancels_startup(
    window, application, monkeypatch, action
):
    gate = threading.Event()
    events, closed = install_runtime(monkeypatch, models_gate=gate)
    window.launch()
    wait_until(application, lambda: "models_started" in events)
    getattr(window.loading, action)()
    wait_until(application, lambda: window.worker is None)
    assert not window.isVisible() and not window.settings_dialog.isVisible()
    assert "audio_opening" not in events and closed.is_set()


def test_closing_the_chat_stops_capture_and_provider(window, application, monkeypatch):
    events, closed = install_runtime(monkeypatch, emit_speech=True)
    window.launch()
    wait_until(
        application,
        lambda: (
            window.feed.messages and window.feed.messages[-1].text == "保留旧缓存。"
        ),
    )
    assert window.isVisible()
    window.close()
    wait_until(application, lambda: window.worker is None)
    assert "audio_closed" in events and closed.is_set()
    assert not window.isVisible() and not window.loading.isVisible()


def test_config_changes_restart_automatically_without_start_button(
    window, application, monkeypatch
):
    events, _ = install_runtime(monkeypatch)
    window.launch()
    wait_until(application, window.isVisible)
    first = window.worker
    window.launch()
    assert not window.isVisible()
    wait_until(application, lambda: window.isVisible() and window.worker is not first)
    assert events.count("models_ready") == 2


def test_invalid_configuration_shows_only_settings(window, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    window.api_key.clear()
    window.launch()
    assert window.worker is None and not window.isVisible()
    assert window.settings_dialog.isVisible()


def test_interruption_keeps_partial_text_and_discards_empty_analysis(window):
    window._ready = True
    window._response_started(1)
    window._response(1, "旧提示")
    old = window.feed.messages[-1]
    window._response_started(2)
    window._response(1, "过期 token")
    window._transcript("面试官", "新问题", True)
    window._response_finished(2, "失败")
    assert old.text == "旧提示" and old.state == "已中断"
    assert [m.role for m in window.feed.messages] == ["ai", "computer"]


@pytest.mark.parametrize("text", ["第一行\n第二行", "<b>原样显示</b>", "x" * 600])
def test_message_wrapping_and_auto_scroll(window, application, text):
    window._ready = True
    window.resize(540, 400)
    window.show()
    for _ in range(20):
        window.feed.add_message("computer", text)
    settle(application)
    bar = window.feed.verticalScrollBar()
    assert window.feed.widget().width() <= window.feed.viewport().width()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    bar.setValue(0)
    message = window.feed.add_message("ai", state="分析中")
    message.append_text("新的提示\n" * 12)
    settle(application)
    assert bar.value() == bar.maximum()


def test_api_key_is_not_saved(window, application, monkeypatch):
    install_runtime(monkeypatch)
    window.launch()
    wait_until(application, window.isVisible)
    window.preferences.sync()
    assert (
        "test-only-key-not-saved" not in Path(window.preferences.fileName()).read_text()
    )


def test_feed_capacity_still_keeps_active_reply(application):
    feed = ChatFeed(max_messages=3)
    active = feed.add_message("ai", state="生成中")
    for n in range(8):
        feed.add_message("microphone", str(n))
    assert len(feed.messages) == 3 and active in feed.messages
    feed.close()
