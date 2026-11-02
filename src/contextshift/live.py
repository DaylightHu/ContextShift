"""Coordinate speech events, actual dialogue history, and model turns."""

import asyncio
from collections.abc import Callable

from contextshift.dialogue import DialogueHistory
from contextshift.session import Session, TurnDelta, TurnResult
from contextshift.speech.events import AudioSource, SpeechStarted, TranscriptReady

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
        self._history = history if history is not None else DialogueHistory()
        self._on_error = on_error
        self._latest_ids: dict[AudioSource, int] = {}
        self._transcribed_ids: dict[AudioSource, int] = {}
        self._speaking: set[AudioSource] = set()
        self._tasks: set[asyncio.Task[TurnResult]] = set()
        self._closed = False

    @property
    def history(self) -> DialogueHistory:
        """Return the actual-input history owned by this live session."""

        return self._history

    @property
    def latest_utterance_id(self) -> int | None:
        """Return the newest announced interviewer utterance identifier."""

        return self._latest_ids.get("system")

    @property
    def candidate_is_speaking(self) -> bool:
        return "microphone" in self._speaking

    def handle_speech_started(self, event: SpeechStarted) -> None:
        """Track both speakers; synchronously preempt only on system speech."""

        if self._closed:
            return
        if event.utterance_id <= self._latest_ids.get(event.source, 0):
            return

        self._latest_ids[event.source] = event.utterance_id
        self._speaking.add(event.source)
        if event.source == "system":
            self._session.cancel()

    def handle_transcript(self, event: TranscriptReady) -> None:
        """Consume source-ordered final transcripts and answer system audio."""

        if self._closed:
            return

        if event.utterance_id <= self._transcribed_ids.get(event.source, 0):
            return
        self._transcribed_ids[event.source] = event.utterance_id

        latest = self._latest_ids.get(event.source, 0)
        if event.utterance_id > latest:
            # Also support adapters delivering only final transcripts.
            self._latest_ids[event.source] = event.utterance_id
        if event.utterance_id == self._latest_ids[event.source]:
            self._speaking.discard(event.source)

        text = event.text.strip()
        if not text:
            return

        model_input = (
            self._history.build_model_input(text, role="interviewer")
            if event.source == "system"
            else None
        )
        self._history.append(text, role=event.role)

        if model_input is None or event.utterance_id != self._latest_ids.get("system"):
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
