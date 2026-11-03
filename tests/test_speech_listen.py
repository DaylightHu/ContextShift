"""Tests for ordered speech events and serial transcription."""

import asyncio
import queue
from types import SimpleNamespace
from typing import Any, cast

import pytest

from contextshift.speech import AudioConfig, SenseVoiceTranscriber, VadSegment, listen
from contextshift.speech.errors import SpeechConfigurationError
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


@pytest.fixture(autouse=True)
def isolate_native_echo_backend(monkeypatch):
    # Channel tests inspect scheduling/routing; DSP is exercised in test_echo.py.
    class Echo:
        def __init__(self, sample_rate):
            pass

        def publish(self, packet):
            pass

        def clean(self, packet):
            return packet.samples * 0.5

        def close(self):
            pass

    monkeypatch.setattr(listen, "EchoProcessor", Echo)


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


def test_each_source_has_its_own_tracker_and_role() -> None:
    events = []
    mic = _SpeechEventTracker(events.append, "microphone")
    system = _SpeechEventTracker(events.append, "system")
    assert mic.ensure_started(0) == system.ensure_started(0) == 1
    assert [(event.source, event.role) for event in events] == [
        ("microphone", "candidate"),
        ("system", "interviewer"),
    ]


@pytest.mark.asyncio
async def test_two_asr_workers_do_not_block_or_relabel_each_other() -> None:
    mic_queue: asyncio.Queue[_TranscriptionJob | None] = asyncio.Queue()
    system_queue: asyncio.Queue[_TranscriptionJob | None] = asyncio.Queue()
    mic_entered = asyncio.Event()
    release_mic = asyncio.Event()
    transcripts = []

    async def on_mic(event) -> None:
        mic_entered.set()
        await release_mic.wait()
        transcripts.append(event)

    mic_transcriber = RecordingTranscriber()
    system_transcriber = RecordingTranscriber()
    mic_queue.put_nowait(_TranscriptionJob(1, VadSegment(0, 100), "mic", 0.1))
    mic_queue.put_nowait(None)
    system_queue.put_nowait(
        _TranscriptionJob(1, VadSegment(0, 100), "system", 0.1, source="system")
    )
    system_queue.put_nowait(None)
    mic_worker = asyncio.create_task(
        _run_transcription_worker(mic_queue, mic_transcriber, on_mic)
    )
    try:
        await asyncio.wait_for(mic_entered.wait(), 1)
        await asyncio.wait_for(
            _run_transcription_worker(
                system_queue, system_transcriber, transcripts.append
            ),
            1,
        )
        assert [(e.source, e.role, e.utterance_id) for e in transcripts] == [
            ("system", "interviewer", 1)
        ]
    finally:
        release_mic.set()
        await mic_worker
    assert [(e.source, e.text) for e in transcripts] == [
        ("system", "text-system"),
        ("microphone", "text-mic"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("mic_device", "system_device"), [(0, None), (0, 0)])
async def test_dual_listener_rejects_missing_or_identical_devices(
    mic_device, system_device
) -> None:
    with pytest.raises(SpeechConfigurationError):
        await listen.run_dual_audio_listener(
            asr_model_dir="unused",
            vad_model_dir="unused",
            microphone_config=AudioConfig(device=mic_device),
            system_config=AudioConfig(device=system_device),
        )


@pytest.mark.asyncio
async def test_default_mic_cannot_be_the_system_loopback(monkeypatch) -> None:
    from contextshift.speech import audio

    monkeypatch.setattr(
        audio,
        "_sounddevice",
        lambda: SimpleNamespace(query_devices=lambda *args: {"index": 4}),
    )
    with pytest.raises(SpeechConfigurationError, match="默认麦克风"):
        await listen.run_dual_audio_listener(
            asr_model_dir="unused",
            vad_model_dir="unused",
            microphone_config=AudioConfig(),
            system_config=AudioConfig(device=4),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_source", ["microphone", "system"])
async def test_channel_failure_stops_the_other_channel(
    monkeypatch, fail_source
) -> None:
    entered = {source: asyncio.Event() for source in ("microphone", "system")}
    stopped = set()
    configs = {}

    async def fake_listener(*, source, audio_config, **kwargs):
        configs[source] = audio_config
        entered[source].set()
        try:
            await entered["microphone"].wait()
            await entered["system"].wait()
            if source == fail_source:
                raise SpeechConfigurationError("capture failed")
            await asyncio.Event().wait()
        finally:
            stopped.add(source)

    monkeypatch.setattr(listen, "run_audio_listener", fake_listener)
    with pytest.raises(SpeechConfigurationError, match="capture failed"):
        await asyncio.wait_for(
            listen.run_dual_audio_listener(
                asr_model_dir="unused",
                vad_model_dir="unused",
                microphone_config=AudioConfig(device=0),
                system_config=AudioConfig(device=1),
            ),
            1,
        )
    assert stopped == {"microphone", "system"}
    assert configs["microphone"].device == 0
    assert configs["system"].device == 1


@pytest.mark.asyncio
async def test_cancelling_dual_listener_closes_both_channels(monkeypatch) -> None:
    entered = {source: asyncio.Event() for source in ("microphone", "system")}
    stopped = set()

    async def fake_listener(*, source, **kwargs):
        entered[source].set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.add(source)

    monkeypatch.setattr(listen, "run_audio_listener", fake_listener)
    task = asyncio.create_task(
        listen.run_dual_audio_listener(
            asr_model_dir="unused",
            vad_model_dir="unused",
            microphone_config=AudioConfig(device=0),
            system_config=AudioConfig(device=1),
        )
    )
    await asyncio.wait_for(entered["microphone"].wait(), 1)
    await asyncio.wait_for(entered["system"].wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped == {"microphone", "system"}


@pytest.mark.asyncio
async def test_capture_to_transcript_keeps_channels_isolated(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    streams = {}
    vads = []
    transcribers = []
    closed = set()
    starts = []
    transcripts = []
    both_transcribed = asyncio.Event()

    class Stream:
        def __init__(self, config, **kwargs):
            self.device = config.device
            self.blocks = queue.Queue()
            self.blocks.put(
                np.full(config.block_size, self.device + 1, dtype="float32")
            )
            streams[self.device] = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.add(self.device)

        def next_block(self):
            try:
                # Match the real capture timeout so a scheduled ASR task can
                # finish before this one-block fixture reports input loss.
                return self.blocks.get(timeout=1.0)
            except queue.Empty as exc:
                raise TimeoutError from exc

        def next_packet(self):
            from contextshift.speech.audio import AudioPacket

            return AudioPacket(self.next_block(), 0.0)

    class Vad:
        pending_start_ms = None

        def __init__(self, *args, **kwargs):
            self.samples = []
            vads.append(self)

        def accept(self, block):
            self.samples.append(float(block[0]))
            return [VadSegment(0, 200)]

    class Transcriber:
        def __init__(self, *args, **kwargs):
            self.samples = []
            transcribers.append(self)

        def transcribe(self, samples):
            self.samples.append(float(samples[0]))
            return "candidate words" if samples[0] == 0.5 else "interviewer question"

    def on_transcript(event):
        transcripts.append(event)
        if len(transcripts) == 2:
            both_transcribed.set()

    monkeypatch.setattr(listen, "MicrophoneStream", Stream)
    monkeypatch.setattr(listen, "FsmnVad", Vad)
    monkeypatch.setattr(listen, "SenseVoiceTranscriber", Transcriber)
    task = asyncio.create_task(
        listen.run_dual_audio_listener(
            asr_model_dir="unused",
            vad_model_dir="unused",
            microphone_config=AudioConfig(device=0),
            system_config=AudioConfig(device=1),
            on_speech_started=starts.append,
            on_transcript=on_transcript,
        )
    )
    try:
        await asyncio.wait_for(both_transcribed.wait(), 2)
        assert sorted((e.source, e.role, e.utterance_id) for e in starts) == [
            ("microphone", "candidate", 1),
            ("system", "interviewer", 1),
        ]
        assert sorted((e.source, e.role, e.text) for e in transcripts) == [
            ("microphone", "candidate", "candidate words"),
            ("system", "interviewer", "interviewer question"),
        ]
        assert len(vads) == len(transcribers) == len(streams) == 2
        assert sorted(vad.samples for vad in vads) == [[0.5], [2.0]]
        assert sorted(asr.samples for asr in transcribers) == [[0.5], [2.0]]
        assert {e.source: e.audio_processing for e in transcripts} == {
            "microphone": "aec",
            "system": "raw",
        }
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed == {0, 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("processing", ["raw", "text"])
async def test_live_boundary_rejects_unprocessed_mic_before_ui_or_history(
    monkeypatch,
    processing,
):
    from contextshift.speech.events import TranscriptReady

    delivered = []
    stopped = set()

    async def listener(*, source, on_transcript, **kwargs):
        try:
            if source == "microphone":
                await on_transcript(
                    TranscriptReady(
                        1,
                        0,
                        200,
                        "speaker echo",
                        0.01,
                        0.1,
                        audio_processing=processing,
                    )
                )
            await asyncio.Event().wait()
        finally:
            stopped.add(source)

    monkeypatch.setattr(listen, "run_audio_listener", listener)
    with pytest.raises(SpeechConfigurationError, match="已被拦截"):
        await listen.run_dual_audio_listener(
            asr_model_dir="unused",
            vad_model_dir="unused",
            microphone_config=AudioConfig(device=0),
            system_config=AudioConfig(device=1),
            on_transcript=delivered.append,
        )
    assert delivered == []
    assert stopped == {"microphone", "system"}


@pytest.mark.asyncio
async def test_mic_is_ready_only_after_aec_has_produced_audio(monkeypatch):
    import numpy as np

    from contextshift.speech.audio import AudioPacket

    ready = asyncio.Event()
    calls = []

    class Stream:
        def __init__(self, config, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def next_packet(self):
            return AudioPacket(np.zeros(3200, dtype=np.float32), 0.0)

    class Echo:
        def clean(self, packet):
            calls.append("clean")
            return None if len(calls) == 1 else packet.samples

    class Vad:
        pending_start_ms = None

        def accept(self, block):
            calls.append("vad")
            return []

    def on_ready():
        assert calls[:2] == ["clean", "clean"]
        ready.set()

    monkeypatch.setattr(listen, "MicrophoneStream", Stream)
    config = AudioConfig(device=0)
    task = asyncio.create_task(
        listen.run_audio_listener(
            source="microphone",
            asr_model_dir="unused",
            vad_model_dir="unused",
            audio_config=config,
            prepared=listen.PreparedChannel(config, Vad(), RecordingTranscriber()),
            echo=Echo(),
            on_ready=on_ready,
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert calls[-1] == "closed"
