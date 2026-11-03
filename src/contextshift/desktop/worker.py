"""The desktop thread owns one asyncio session and optional local audio."""

import asyncio
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from contextshift.config import Settings
from contextshift.dialogue import DialogueHistory
from contextshift.live import LiveAssistant
from contextshift.llm import DeepSeekProvider
from contextshift.session import Session, TurnResult
from contextshift.speech.audio import AudioConfig
from contextshift.speech.events import SpeechStarted, TranscriptReady
from contextshift.speech.listen import run_dual_audio_listener


@dataclass(frozen=True, slots=True)
class DesktopOptions:
    settings: Settings
    audio_enabled: bool
    microphone: int | None = None
    system: int | None = None
    asr_model: Path = Path("models/asr/SenseVoiceSmall")
    vad_model: Path = Path("models/vad/fsmn-vad")


def safe_error(exc: BaseException) -> str:
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[API Key 已隐藏]", str(exc))


class AssistantWorker(QThread):
    ready = Signal()
    response_started = Signal(int)
    response = Signal(int, str)
    response_finished = Signal(int, str)
    transcript = Signal(str, str, bool)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, options: DesktopOptions, parent=None) -> None:
        super().__init__(parent)
        self.options = options
        self._loop: asyncio.AbstractEventLoop | None = None
        self._assistant: LiveAssistant | None = None
        self._stop: asyncio.Event | None = None
        self._stop_requested = threading.Event()
        self._epoch = 0

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._dispatch(self._stop_on_loop)

    def _dispatch(self, callback, *args) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(callback, *args)
            except RuntimeError:
                pass

    def _stop_on_loop(self) -> None:
        if self._assistant is not None:
            self._assistant.cancel_current()
        if self._stop is not None:
            self._stop.set()

    def ask(self, text: str) -> None:
        if not self._stop_requested.is_set():
            self._dispatch(self._ask_on_loop, text)

    def _ask_on_loop(self, text: str) -> None:
        if self._assistant is None or self._stop_requested.is_set():
            return
        self._new_response()
        self.transcript.emit("手动输入", text, True)
        self.status.emit("正在生成提示")
        self._assistant.handle_manual_question(text)

    def cancel_answer(self) -> None:
        self._dispatch(self._cancel_on_loop)

    def _cancel_on_loop(self) -> None:
        if self._assistant is not None:
            self._assistant.cancel_current()
            self.status.emit("已取消 · 等待下一句")

    def _new_response(self) -> None:
        self._epoch += 1
        self.response_started.emit(self._epoch)

    def run(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:  # noqa: BLE001 - keep GUI open on device/API errors
            self.failed.emit(safe_error(exc))
        finally:
            self._loop = None
            self._assistant = None

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        options = self.options
        session = Session(DeepSeekProvider(options.settings))

        def failed(exc: BaseException) -> None:
            self.failed.emit(safe_error(exc))
            self.response_finished.emit(self._epoch, "失败")
            self.status.emit("本次分析失败，可以重试")

        def finished(result: TurnResult) -> None:
            self.response_finished.emit(self._epoch, "已完成")
            self.status.emit("正在监听" if options.audio_enabled else "就绪 · 文字模式")

        self._assistant = LiveAssistant(
            session,
            lambda delta: self.response.emit(self._epoch, delta.text),
            history=DialogueHistory(
                max_turns=options.settings.history_max_turns,
                max_characters=options.settings.history_max_characters,
            ),
            on_error=failed,
            on_turn_finished=finished,
        )

        def started(event: SpeechStarted) -> None:
            self._assistant.handle_speech_started(event)
            if event.source == "system":
                self._new_response()
                self.status.emit("正在听对方讲话")

        def transcript(event: TranscriptReady) -> None:
            current_question = (
                event.source == "system"
                and event.utterance_id == self._assistant.latest_utterance_id
            )
            self.transcript.emit(
                "候选人" if event.role == "candidate" else "面试官",
                event.text,
                current_question,
            )
            self._assistant.handle_transcript(event)
            if current_question and event.text.strip():
                self.status.emit("正在生成提示")

        audio: asyncio.Task[None] | None = None
        try:
            if self._stop_requested.is_set():
                return
            if options.audio_enabled:
                if options.microphone is None or options.system is None:
                    raise ValueError("请选择麦克风与会议回环输入设备。")
                audio = asyncio.create_task(
                    run_dual_audio_listener(
                        asr_model_dir=options.asr_model,
                        vad_model_dir=options.vad_model,
                        microphone_config=AudioConfig(device=options.microphone),
                        system_config=AudioConfig(device=options.system),
                        on_speech_started=started,
                        on_transcript=transcript,
                    )
                )

                def audio_finished(task: asyncio.Task[None]) -> None:
                    if not task.cancelled() and not self._stop_requested.is_set():
                        exc = task.exception()
                        if exc is not None:
                            failed(exc)
                        self._stop.set()

                audio.add_done_callback(audio_finished)
            self.ready.emit()
            self.status.emit(
                "正在准备音频" if options.audio_enabled else "就绪 · 文字模式"
            )
            await self._stop.wait()
        finally:
            self._assistant.cancel_current()
            if audio is not None:
                audio.cancel()
                await asyncio.gather(audio, return_exceptions=True)
            await self._assistant.close()
            await session.close()
