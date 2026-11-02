"""Tests for speech-to-turn coordination."""

import asyncio
from collections.abc import Callable
from typing import cast

import pytest

from contextshift.live import LiveAssistant
from contextshift.session import Session, TurnDelta, TurnResult
from contextshift.speech import SpeechStarted, TranscriptReady


class FakeSession:
    def __init__(self) -> None:
        self.cancel_calls = 0
        self.questions: list[str] = []

    def cancel(self) -> None:
        self.cancel_calls += 1

    def submit(
        self,
        question: str,
        on_delta: Callable[[TurnDelta], None],
    ) -> tuple[int, asyncio.Task[TurnResult]]:
        del on_delta
        self.questions.append(question)

        async def complete() -> TurnResult:
            return TurnResult(
                turn_id=len(self.questions),
                finish_reason="completed",
                output_characters=0,
            )

        return len(self.questions), asyncio.create_task(complete())


def started(utterance_id: int) -> SpeechStarted:
    return SpeechStarted(utterance_id, utterance_id * 100, 1.0)


def transcript(utterance_id: int, text: str) -> TranscriptReady:
    return TranscriptReady(
        utterance_id=utterance_id,
        start_ms=utterance_id * 100,
        end_ms=utterance_id * 100 + 80,
        text=text,
        asr_elapsed=0.01,
        rtf=0.1,
    )


@pytest.mark.asyncio
async def test_speech_start_cancels_once_per_utterance() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)

    assistant.handle_speech_started(started(1))
    assistant.handle_speech_started(started(1))
    assistant.handle_speech_started(started(2))

    assert session.cancel_calls == 2
    assert assistant.latest_utterance_id == 2
    await assistant.close()


@pytest.mark.asyncio
async def test_empty_transcript_is_ignored() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)
    assistant.handle_speech_started(started(1))

    assistant.handle_transcript(transcript(1, "  "))

    assert session.questions == []
    assert assistant.history.entries == ()
    await assistant.close()


@pytest.mark.asyncio
async def test_superseded_transcript_is_history_but_not_a_request() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)
    assistant.handle_speech_started(started(1))
    assistant.handle_speech_started(started(2))

    assistant.handle_transcript(transcript(1, "first actual question"))
    assistant.handle_transcript(transcript(2, "second actual question"))

    assert assistant.history.entries == (
        "first actual question",
        "second actual question",
    )
    assert len(session.questions) == 1
    assert "first actual question" in session.questions[0]
    assert session.questions[0].endswith("当前问题：second actual question")
    await assistant.close()
