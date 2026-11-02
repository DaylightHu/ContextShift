"""Tests for speech-to-turn coordination."""

import asyncio
from collections.abc import Callable
from itertools import combinations, product
from typing import cast

import pytest

from contextshift.dialogue import DialogueHistory
from contextshift.live import LiveAssistant
from contextshift.session import Session, TurnDelta, TurnResult
from contextshift.speech import SpeechStarted, TranscriptReady
from contextshift.speech.events import AudioSource


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


def started(utterance_id: int, source: AudioSource = "system") -> SpeechStarted:
    return SpeechStarted(utterance_id, utterance_id * 100, 1.0, source=source)


def transcript(
    utterance_id: int, text: str, source: AudioSource = "system"
) -> TranscriptReady:
    return TranscriptReady(
        utterance_id=utterance_id,
        start_ms=utterance_id * 100,
        end_ms=utterance_id * 100 + 80,
        text=text,
        asr_elapsed=0.01,
        rtf=0.1,
        source=source,
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
    assert "面试官问题：second actual question" in session.questions[0]
    await assistant.close()


@pytest.mark.asyncio
async def test_microphone_only_records_candidate_context() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)

    assistant.handle_speech_started(started(1, "microphone"))
    assert assistant.candidate_is_speaking
    assistant.handle_transcript(transcript(1, "我负责缓存优化", "microphone"))

    assert not assistant.candidate_is_speaking
    assert assistant.latest_utterance_id is None
    assert session.cancel_calls == 0
    assert session.questions == []
    assert [(turn.role, turn.text) for turn in assistant.history.turns] == [
        ("candidate", "我负责缓存优化")
    ]
    await assistant.close()


@pytest.mark.asyncio
async def test_source_local_ids_do_not_override_interviewer_question() -> None:
    session = FakeSession()
    history = DialogueHistory()
    assistant = LiveAssistant(cast(Session, session), lambda _: None, history=history)

    assistant.handle_speech_started(started(1))
    assistant.handle_speech_started(started(10, "microphone"))
    assistant.handle_transcript(transcript(10, "我负责缓存优化", "microphone"))
    assistant.handle_transcript(transcript(1, "具体怎么做的？"))

    assert assistant.history is history
    assert session.cancel_calls == 1
    assert assistant.latest_utterance_id == 1
    assert len(session.questions) == 1
    assert "候选人：我负责缓存优化" in session.questions[0]
    assert "面试官问题：具体怎么做的？" in session.questions[0]
    assert "interviewer" == history.turns[-1].role
    await assistant.close()


@pytest.mark.asyncio
async def test_same_id_on_each_source_and_duplicate_events() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)

    for source in ("microphone", "system"):
        assistant.handle_speech_started(started(1, source))
        assistant.handle_speech_started(started(1, source))
        assistant.handle_transcript(transcript(1, source, source))
        assistant.handle_transcript(transcript(1, source, source))

    assert session.cancel_calls == 1
    assert len(session.questions) == 1
    assert assistant.history.entries == ("microphone", "system")
    await assistant.close()


@pytest.mark.asyncio
async def test_transcript_only_adapter_can_trigger_latest_question() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)
    assistant.handle_transcript(transcript(1, "first"))
    assistant.handle_transcript(transcript(2, "second"))
    assert len(session.questions) == 2
    await assistant.close()


@pytest.mark.asyncio
async def test_empty_old_and_closed_events_do_not_corrupt_state() -> None:
    session = FakeSession()
    assistant = LiveAssistant(cast(Session, session), lambda _: None)
    assistant.handle_speech_started(started(1, "microphone"))
    assistant.handle_speech_started(started(2, "microphone"))
    assistant.handle_transcript(transcript(1, "older", "microphone"))
    assert assistant.candidate_is_speaking
    assistant.handle_transcript(transcript(2, "  ", "microphone"))
    assert not assistant.candidate_is_speaking
    await assistant.close()
    assistant.handle_speech_started(started(3))
    assistant.handle_transcript(transcript(3, "after shutdown"))
    assert assistant.history.entries == ("older",)
    assert session.questions == []
    assert session.cancel_calls == 1  # close() only


@pytest.mark.asyncio
async def test_mic_keeps_streaming_and_system_start_filters_late_tokens() -> None:
    from collections.abc import AsyncGenerator

    from contextshift.llm import ModelRequest

    class Provider:
        def __init__(self) -> None:
            self.first = asyncio.Event()
            self.continue_old = asyncio.Event()
            self.second = asyncio.Event()
            self.interrupted = asyncio.Event()
            self.inputs: list[str] = []

        async def stream(self, request: ModelRequest) -> AsyncGenerator[str, None]:
            self.inputs.append(request.input)
            if len(self.inputs) == 1:
                yield "旧建议"
                self.first.set()
                try:
                    await self.continue_old.wait()
                    yield "继续"
                    self.second.set()
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.interrupted.set()
                    yield "过期 token"
            else:
                yield "新建议"

        async def close(self) -> None:
            pass

    provider = Provider()
    session = Session(provider)
    rendered: list[TurnDelta] = []
    assistant = LiveAssistant(session, rendered.append)
    try:
        assistant.handle_speech_started(started(1))
        assistant.handle_transcript(transcript(1, "解释缓存"))
        await asyncio.wait_for(provider.first.wait(), 1)

        assistant.handle_speech_started(started(1, "microphone"))
        assistant.handle_transcript(transcript(1, "我做过缓存", "microphone"))
        provider.continue_old.set()
        await asyncio.wait_for(provider.second.wait(), 1)
        assert [delta.text for delta in rendered] == ["旧建议", "继续"]

        assistant.handle_speech_started(started(2))
        # Cancellation/filtering happens at SpeechStarted, before ASR completes.
        await asyncio.wait_for(provider.interrupted.wait(), 1)
        assert "过期 token" not in [delta.text for delta in rendered]
        assistant.handle_transcript(transcript(2, "如何失效？"))
        await asyncio.sleep(0)
        assert rendered[-1].text == "新建议"
        assert "候选人：我做过缓存" in provider.inputs[-1]
        assert assistant.history.entries == ("解释缓存", "我做过缓存", "如何失效？")
        assert "旧建议" not in provider.inputs[-1]
        assert "新建议" not in assistant.history.entries
    finally:
        await assistant.close()
        await session.close()


@pytest.mark.asyncio
async def test_all_source_interleavings_preserve_role_routing() -> None:
    # Each source keeps transcript order, but VAD can start a second utterance
    # before ASR has completed the first. Enumerate cross-source interleavings.
    orders = (
        (("start", 1), ("text", 1), ("start", 2), ("text", 2)),
        (("start", 1), ("start", 2), ("text", 1), ("text", 2)),
    )
    for mic_order, system_order in product(orders, repeat=2):
        for mic_positions in combinations(range(8), 4):
            session = FakeSession()
            assistant = LiveAssistant(cast(Session, session), lambda _: None)
            mic_events = iter(mic_order)
            system_events = iter(system_order)
            try:
                for position in range(8):
                    source = "microphone" if position in mic_positions else "system"
                    kind, utterance_id = next(
                        mic_events if source == "microphone" else system_events
                    )
                    if kind == "start":
                        assistant.handle_speech_started(started(utterance_id, source))
                    else:
                        assistant.handle_transcript(
                            transcript(utterance_id, f"{source}-{utterance_id}", source)
                        )
                assert session.cancel_calls == 2
                expected_questions = 2 if system_order == orders[0] else 1
                assert len(session.questions) == expected_questions
                assert "面试官问题：system-2" in session.questions[-1]
                assert sorted((e.role, e.text) for e in assistant.history.turns) == [
                    ("candidate", "microphone-1"),
                    ("candidate", "microphone-2"),
                    ("interviewer", "system-1"),
                    ("interviewer", "system-2"),
                ]
            finally:
                await assistant.close()
