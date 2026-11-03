"""Prepare the entire speech path before announcing a usable desktop session."""

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
from contextshift.speech.listen import prepare_channels, run_dual_audio_listener


@dataclass(frozen=True, slots=True)
class DesktopOptions:
    settings: Settings
    microphone: int
    system: int
    asr_model: Path
    vad_model: Path


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
        self._initialization: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._stop_requested = threading.Event()
        self._epoch = 0
        self._announced_ready = False

    def request_stop(self) -> None:
        self._stop_requested.set()
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._stop_on_loop)
            except RuntimeError:
                pass

    def _stop_on_loop(self) -> None:
        if self._assistant is not None:
            self._assistant.cancel_current()
        if self._initialization is not None:
            self._initialization.cancel()
        if self._stop is not None:
            self._stop.set()

    def _audio_ready(self) -> None:
        if not self._stop_requested.is_set() and not self._announced_ready:
            self._announced_ready = True
            self.status.emit("正在监听")
            self.ready.emit()

    def run(self) -> None:
        try:
            asyncio.run(self._run())
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - keep Qt alive on runtime errors
            if not self._stop_requested.is_set():
                self.failed.emit(safe_error(exc))
        finally:
            self._loop = None
            self._assistant = None

    async def _prepare(self, provider: DeepSeekProvider):
        self.status.emit("正在检查云端连接…")
        await provider.prepare()
        if self._stop_requested.is_set():
            raise asyncio.CancelledError
        options = self.options
        return await asyncio.to_thread(
            prepare_channels,
            asr_model_dir=options.asr_model,
            vad_model_dir=options.vad_model,
            microphone_config=AudioConfig(device=options.microphone),
            system_config=AudioConfig(device=options.system),
            cancelled=self._stop_requested,
            on_progress=self.status.emit,
        )

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        options = self.options
        provider = DeepSeekProvider(options.settings)
        session = Session(provider)

        def failed(exc: BaseException) -> None:
            self.response_finished.emit(self._epoch, "失败")
            self.failed.emit(safe_error(exc))
            self._stop.set()

        def finished(result: TurnResult) -> None:
            self.response_finished.emit(self._epoch, "已完成")

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
            if self._stop_requested.is_set():
                return
            self._assistant.handle_speech_started(event)
            if event.source == "system":
                self._epoch += 1
                self.response_started.emit(self._epoch)

        def transcript(event: TranscriptReady) -> None:
            if self._stop_requested.is_set():
                return
            current = (
                event.source == "system"
                and event.utterance_id == self._assistant.latest_utterance_id
            )
            self.transcript.emit(
                "候选人" if event.role == "candidate" else "面试官", event.text, current
            )
            self._assistant.handle_transcript(event)

        audio: asyncio.Task | None = None
        try:
            if self._stop_requested.is_set():
                return
            self._initialization = asyncio.create_task(self._prepare(provider))
            try:
                prepared = await self._initialization
            finally:
                self._initialization = None
            if self._stop_requested.is_set():
                return
            self.status.emit("正在打开音频输入…")
            audio = asyncio.create_task(
                run_dual_audio_listener(
                    asr_model_dir=options.asr_model,
                    vad_model_dir=options.vad_model,
                    microphone_config=AudioConfig(device=options.microphone),
                    system_config=AudioConfig(device=options.system),
                    prepared_channels=prepared,
                    on_ready=self._audio_ready,
                    on_speech_started=started,
                    on_transcript=transcript,
                )
            )

            def audio_finished(task: asyncio.Task) -> None:
                if not task.cancelled() and not self._stop_requested.is_set():
                    exc = task.exception()
                    if exc is not None:
                        failed(exc)
                    elif not self._announced_ready:
                        failed(RuntimeError("音频输入在就绪前退出。"))
                    self._stop.set()

            audio.add_done_callback(audio_finished)
            await self._stop.wait()
        finally:
            self._assistant.cancel_current()
            if audio is not None:
                audio.cancel()
                await asyncio.gather(audio, return_exceptions=True)
            await self._assistant.close()
            await session.close()
