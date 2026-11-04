"""Own streamed turns and prepare either speech capture or typed role tests."""

import asyncio
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PySide6.QtCore import QThread, Signal

from contextshift.config import Settings
from contextshift.dialogue import DialogueHistory
from contextshift.live import LiveAssistant
from contextshift.llm import DeepSeekProvider
from contextshift.llm.base import ModelRequest
from contextshift.repository import (
    PROJECT_INSTRUCTIONS,
    RepositoryIndex,
    SourceReference,
    WorkspaceIndex,
)
from contextshift.session import Session, TurnDelta, TurnResult
from contextshift.speech.audio import AudioConfig
from contextshift.speech.events import (
    AudioSource,
    SpeakerRole,
    SpeechStarted,
    TranscriptReady,
)
from contextshift.speech.listen import prepare_channels, run_dual_audio_listener


@dataclass(frozen=True, slots=True)
class DesktopOptions:
    settings: Settings
    microphone: int | None
    system: int | None
    asr_model: Path
    vad_model: Path
    test_mode: bool = False


@dataclass(frozen=True, slots=True)
class ReplyMetrics:
    first_token_seconds: float | None
    elapsed_seconds: float
    finish_reason: str
    output_characters: int
    retrieval_seconds: float = 0.0
    source_count: int = 0


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
    request_failed = Signal(int, str)
    metrics = Signal(int, object)
    sources = Signal(int, object, float)

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
        self._accepting_text = threading.Event()
        self._text_handler: Callable[[str, SpeakerRole], None] | None = None
        self._repository: RepositoryIndex | WorkspaceIndex | None = None
        self._repository_revision = 0
        self._code_focus: tuple[SourceReference, ...] = ()
        self._active_file = ""
        self._repository_pending = False
        self._repository_root: Path | None = None
        self._retrieval_metrics: tuple[int, float, int] = (-1, 0.0, 0)
        self._turn_sources: tuple[SourceReference, ...] = ()
        self._reply_text = ""

    def set_repository(
        self,
        index: RepositoryIndex | WorkspaceIndex | None,
        *,
        pending: bool = False,
        root: Path | None = None,
    ) -> None:
        """Change code scope without resetting actual conversation history."""

        def apply() -> None:
            new_root = index.root if index is not None else root
            self._repository = index
            self._repository_pending = pending
            self._repository_revision += 1
            if new_root is None or new_root != self._repository_root:
                self._code_focus = ()
                self._active_file = ""
            elif index is not None:
                self._code_focus = tuple(
                    reference
                    for reference in self._code_focus
                    if (
                        index.contains_reference(reference)
                        if isinstance(index, WorkspaceIndex)
                        else reference.path in index.files
                        and index.files[reference.path].digest == reference.digest
                    )
                )
            self._repository_root = new_root
            if self._assistant is not None:
                self._assistant.cancel_current()
                self.response_finished.emit(self._epoch, "上下文已更新")

        self._on_loop(apply)

    def set_active_file(self, relative: str) -> None:
        self._on_loop(lambda: setattr(self, "_active_file", relative))

    def _on_loop(self, function: Callable[[], None]) -> None:
        loop = self._loop
        if loop is None:
            function()
        elif not loop.is_closed():
            try:
                loop.call_soon_threadsafe(function)
            except RuntimeError:
                pass

    async def _resolve_context(self, request: ModelRequest) -> ModelRequest:
        index = self._repository
        revision, epoch = self._repository_revision, self._epoch
        if index is None:
            if self._repository_pending:
                return ModelRequest(
                    request.input
                    + "\n\n项目索引正在更新或不可用，本轮没有代码证据。涉及项目实现时明确说明暂时无法定位，不得猜测。",
                    request.instructions + PROJECT_INSTRUCTIONS,
                    request.query,
                )
            return request
        dialogue = (
            tuple(turn.text for turn in self._assistant.history.turns[-5:])
            if self._assistant
            else ()
        )
        cancelled = threading.Event()
        explorer = getattr(self._provider, "explore_code", None)
        if explorer is not None:
            bundle = await explorer(index, request.query, dialogue)
        else:
            try:
                bundle = await asyncio.to_thread(
                    index.retrieve,
                    request.query,
                    dialogue=dialogue,
                    focus=self._code_focus,
                    active_file=self._active_file,
                    cancelled=cancelled,
                )
            finally:
                cancelled.set()
        if (
            self._stop_requested.is_set()
            or revision != self._repository_revision
            or epoch != self._epoch
        ):
            raise asyncio.CancelledError
        self._turn_sources = bundle.references
        self._code_focus = bundle.references[:2]
        self._retrieval_metrics = (
            epoch,
            bundle.elapsed_seconds,
            len(bundle.references),
        )
        return ModelRequest(
            input=request.input + "\n\n" + bundle.text,
            instructions=request.instructions + PROJECT_INSTRUCTIONS,
            query=request.query,
        )

    def submit_text(self, text: str, role: SpeakerRole) -> bool:
        """Queue a test utterance on the same loop that owns live turns."""

        text = text.strip()
        loop = self._loop
        if (
            not text
            or role not in {"candidate", "interviewer"}
            or not self.options.test_mode
            or not self._accepting_text.is_set()
            or self._stop_requested.is_set()
            or loop is None
        ):
            return False

        def deliver() -> None:
            if not self._stop_requested.is_set() and self._text_handler is not None:
                self._text_handler(text, role)

        try:
            loop.call_soon_threadsafe(deliver)
        except RuntimeError:
            return False
        return True

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._accepting_text.clear()
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
            self.status.emit("测试模式" if self.options.test_mode else "正在监听")
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
        if options.test_mode:
            return None
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
        self._provider = provider
        session = Session(provider, context_resolver=self._resolve_context)
        request_started: float | None = None
        first_token: float | None = None

        def delta(event: TurnDelta) -> None:
            nonlocal first_token
            if first_token is None and request_started is not None:
                first_token = perf_counter() - request_started
            self.response.emit(self._epoch, event.text)
            self._reply_text += event.text

        def failed(exc: BaseException) -> None:
            self.response_finished.emit(self._epoch, "失败")
            if options.test_mode and self._announced_ready:
                self.request_failed.emit(self._epoch, safe_error(exc))
                return
            self.failed.emit(safe_error(exc))
            self._stop.set()

        def finished(result: TurnResult) -> None:
            cited = tuple(
                dict.fromkeys(
                    int(n) - 1 for n in re.findall(r"\[S(\d+)\]", self._reply_text)
                )
            )
            references = tuple(
                self._turn_sources[i] for i in cited if 0 <= i < len(self._turn_sources)
            )
            core_references = references or self._turn_sources[:3]
            if core_references:
                self._code_focus = core_references[:4]
            retrieval_seconds = (
                self._retrieval_metrics[1]
                if self._retrieval_metrics[0] == self._epoch
                else 0.0
            )
            self.sources.emit(self._epoch, core_references[:3], retrieval_seconds)
            self.response_finished.emit(
                self._epoch,
                "本地截断" if result.finish_reason == "local_limit" else "已完成",
            )
            if options.test_mode and request_started is not None:
                self.metrics.emit(
                    self._epoch,
                    ReplyMetrics(
                        first_token,
                        perf_counter() - request_started,
                        result.finish_reason,
                        result.output_characters,
                        self._retrieval_metrics[1]
                        if self._retrieval_metrics[0] == self._epoch
                        else 0.0,
                        len(core_references[:3]),
                    ),
                )

        self._assistant = LiveAssistant(
            session,
            delta,
            history=DialogueHistory(
                max_turns=options.settings.history_max_turns,
                max_characters=options.settings.history_max_characters,
            ),
            on_error=failed,
            on_turn_finished=finished,
        )

        def started(event: SpeechStarted) -> None:
            nonlocal request_started, first_token
            if self._stop_requested.is_set():
                return
            self._assistant.handle_speech_started(event)
            if event.source == "system":
                request_started, first_token = perf_counter(), None
                self._reply_text = ""
                self._turn_sources = ()
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

        ids: dict[AudioSource, int] = {"microphone": 0, "system": 0}

        def test_input(text: str, role: SpeakerRole) -> None:
            source: AudioSource = "microphone" if role == "candidate" else "system"
            ids[source] += 1
            started(SpeechStarted(ids[source], 0, perf_counter(), source=source))
            transcript(TranscriptReady(ids[source], 0, 0, text, 0, 0, source=source))

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
            if options.test_mode:
                self._text_handler = test_input
                self._accepting_text.set()
                self._audio_ready()
                await self._stop.wait()
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
            self._accepting_text.clear()
            self._text_handler = None
            self._assistant.cancel_current()
            if audio is not None:
                audio.cancel()
                await asyncio.gather(audio, return_exceptions=True)
            await self._assistant.close()
            await session.close()
