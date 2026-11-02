"""Coordinate speech events, actual dialogue history, and model turns."""

import asyncio
from collections.abc import Callable

from contextshift.dialogue import DialogueHistory
from contextshift.session import Session, TurnDelta, TurnResult
from contextshift.speech.events import SpeechStarted, TranscriptReady

TurnErrorHandler = Callable[[BaseException], None]


class LiveAssistant:
    """Turn speech events into preemptible model requests."""

    def __init__(
        self,
        session: Session,
        on_delta: Callable[[TurnDelta], None],
        *,
        history: DialogueHistory | None = None,
        on_error: TurnErrorHandler | None = None,
    ) -> None:
        self._session = session
        self._on_delta = on_delta
        self._history = history or DialogueHistory()
        self._on_error = on_error
        self._latest_utterance_id: int | None = None
        self._tasks: set[asyncio.Task[TurnResult]] = set()
        self._closed = False

    @property
    def history(self) -> DialogueHistory:
        """Return the actual-input history owned by this live session."""

        return self._history

    @property
    def latest_utterance_id(self) -> int | None:
        """Return the newest announced utterance identifier."""

        return self._latest_utterance_id

    def handle_speech_started(self, event: SpeechStarted) -> None:
        """Synchronously invalidate model output when new speech begins."""

        if self._closed:
            return
        if (
            self._latest_utterance_id is not None
            and event.utterance_id <= self._latest_utterance_id
        ):
            return

        self._latest_utterance_id = event.utterance_id
        self._session.cancel()

    def handle_transcript(self, event: TranscriptReady) -> None:
        """Record actual speech and answer only the newest utterance."""

        if self._closed:
            return

        text = event.text.strip()
        if not text:
            return

        model_input = self._history.build_model_input(text)
        self._history.append(text)

        if self._latest_utterance_id is None:
            self._latest_utterance_id = event.utterance_id
        if event.utterance_id != self._latest_utterance_id:
            return

        _, task = self._session.submit(model_input, self._on_delta)
        self._tasks.add(task)
        task.add_done_callback(self._turn_finished)

    async def close(self) -> None:
        """Cancel generation and wait for tasks owned by the coordinator."""

        if self._closed:
            return

        self._closed = True
        self._session.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def _turn_finished(self, task: asyncio.Task[TurnResult]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001 - surface provider failures
            if self._on_error is not None:
                self._on_error(exc)
