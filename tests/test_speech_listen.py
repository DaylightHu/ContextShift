"""Tests for ordered speech events and serial transcription."""

import asyncio
from typing import Any, cast

import pytest

from contextshift.speech import SenseVoiceTranscriber, VadSegment
from contextshift.speech.listen import (
    _run_transcription_worker,
    _SpeechEventTracker,
    _TranscriptionJob,
)


class RecordingTranscriber:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def transcribe(self, samples: Any) -> str:
        self.calls.append(samples)
        return f"text-{samples}"


def test_complete_segment_emits_start_once_before_close() -> None:
    events = []
    tracker = _SpeechEventTracker(events.append)

    first_id = tracker.ensure_started(100)
    duplicate_id = tracker.ensure_started(100)
    closed_id = tracker.close(VadSegment(100, 500))

    assert first_id == duplicate_id == closed_id == 1
    assert [(event.utterance_id, event.start_ms) for event in events] == [(1, 100)]


def test_complete_segment_without_prior_boundary_emits_start() -> None:
    events = []
    tracker = _SpeechEventTracker(events.append)

    utterance_id = tracker.close(VadSegment(200, 600))

    assert utterance_id == 1
    assert [(event.utterance_id, event.start_ms) for event in events] == [(1, 200)]


@pytest.mark.asyncio
async def test_transcription_worker_preserves_queue_order() -> None:
    queue: asyncio.Queue[_TranscriptionJob | None] = asyncio.Queue()
    transcriber = RecordingTranscriber()
    transcripts = []
    worker = asyncio.create_task(
        _run_transcription_worker(
            queue,
            cast(SenseVoiceTranscriber, transcriber),
            transcripts.append,
        )
    )

    queue.put_nowait(_TranscriptionJob(1, VadSegment(0, 100), "one", 0.1))
    queue.put_nowait(_TranscriptionJob(2, VadSegment(100, 200), "two", 0.1))
    queue.put_nowait(None)
    await worker

    assert transcriber.calls == ["one", "two"]
    assert [event.utterance_id for event in transcripts] == [1, 2]
    assert [event.text for event in transcripts] == ["text-one", "text-two"]
