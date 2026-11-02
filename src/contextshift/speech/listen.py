"""Microphone -> FSMN-VAD -> SenseVoice listener."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from time import perf_counter
from typing import Any

from contextshift.speech.asr import SenseVoiceTranscriber
from contextshift.speech.audio import AudioConfig, MicrophoneStream
from contextshift.speech.events import SpeechStarted, TranscriptReady
from contextshift.speech.vad import FsmnVad, VadSegment

SpeechStartedHandler = Callable[[SpeechStarted], None]
TranscriptHandler = Callable[[TranscriptReady], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class _TranscriptionJob:
    utterance_id: int
    segment: VadSegment
    samples: Any
    duration: float


class _SpeechEventTracker:
    """Assign stable IDs and emit exactly one start event per utterance."""

    def __init__(self, handler: SpeechStartedHandler | None) -> None:
        self._handler = handler
        self._next_id = 0
        self._active_id: int | None = None
        self._active_start_ms: int | None = None

    def ensure_started(self, start_ms: int) -> int:
        if self._active_id is not None and self._active_start_ms == start_ms:
            return self._active_id

        self._next_id += 1
        self._active_id = self._next_id
        self._active_start_ms = start_ms
        event = SpeechStarted(
            utterance_id=self._active_id,
            start_ms=start_ms,
            detected_at=perf_counter(),
        )
        if self._handler is not None:
            self._handler(event)
        return self._active_id

    def close(self, segment: VadSegment) -> int:
        utterance_id = self.ensure_started(segment.start_ms)
        self._active_id = None
        self._active_start_ms = None
        return utterance_id


async def _run_transcription_worker(
    queue: asyncio.Queue[_TranscriptionJob | None],
    transcriber: SenseVoiceTranscriber,
    on_transcript: TranscriptHandler | None,
) -> None:
    """Transcribe queued utterances serially and preserve callback order."""

    while True:
        job = await queue.get()
        try:
            if job is None:
                return

            asr_started = perf_counter()
            text = await asyncio.to_thread(transcriber.transcribe, job.samples)
            asr_elapsed = perf_counter() - asr_started
            rtf = asr_elapsed / job.duration if job.duration > 0 else 0.0
            event = TranscriptReady(
                utterance_id=job.utterance_id,
                start_ms=job.segment.start_ms,
                end_ms=job.segment.end_ms,
                text=text,
                asr_elapsed=asr_elapsed,
                rtf=rtf,
            )
            if on_transcript is not None:
                result = on_transcript(event)
                if isawaitable(result):
                    await result
            else:
                print(
                    f"\n[Speech {job.utterance_id}] audio={job.duration:.2f}s "
                    f"ASR={asr_elapsed:.3f}s RTF={rtf:.3f}\n{text}",
                    flush=True,
                )
        finally:
            queue.task_done()


async def run_microphone_listener(
    *,
    asr_model_dir: str | Path,
    vad_model_dir: str | Path,
    audio_config: AudioConfig | None = None,
    asr_device: str = "cpu",
    on_speech_started: SpeechStartedHandler | None = None,
    on_transcript: TranscriptHandler | None = None,
) -> None:
    """Capture until cancelled and transcribe closed VAD segments in order."""

    config = audio_config or AudioConfig()
    vad = FsmnVad(
        vad_model_dir, sample_rate=config.sample_rate, chunk_ms=config.block_ms
    )
    transcriber = SenseVoiceTranscriber(asr_model_dir, device=asr_device)
    blocks: list[Any] = []
    tracker = _SpeechEventTracker(on_speech_started)
    transcription_queue: asyncio.Queue[_TranscriptionJob | None] = asyncio.Queue()
    worker = asyncio.create_task(
        _run_transcription_worker(
            transcription_queue,
            transcriber,
            on_transcript,
        ),
        name="contextshift-asr-worker",
    )

    def schedule(segment: VadSegment, utterance_id: int) -> None:
        import numpy as np

        if not blocks:
            return
        samples = np.concatenate(blocks)
        start = max(0, int(segment.start_ms * config.sample_rate / 1000))
        end = min(len(samples), int(segment.end_ms * config.sample_rate / 1000))
        if end <= start:
            return
        duration = (end - start) / config.sample_rate
        transcription_queue.put_nowait(
            _TranscriptionJob(
                utterance_id=utterance_id,
                segment=segment,
                samples=samples[start:end],
                duration=duration,
            )
        )

    print("开始监听麦克风，按 Ctrl-C 停止。", flush=True)
    try:
        with MicrophoneStream(config) as microphone:
            while True:
                try:
                    block = await asyncio.to_thread(microphone.next_block)
                except TimeoutError:
                    continue
                blocks.append(block)
                segments = await asyncio.to_thread(vad.accept, block)
                for segment in segments:
                    utterance_id = tracker.close(segment)
                    schedule(segment, utterance_id)
                if vad.pending_start_ms is not None:
                    tracker.ensure_started(vad.pending_start_ms)
    finally:
        if vad.pending_start_ms is not None and blocks:
            import numpy as np

            samples = np.concatenate(blocks)
            start = max(0, int(vad.pending_start_ms * config.sample_rate / 1000))
            end = len(samples)
            if end > start:
                duration = (end - start) / config.sample_rate
                end_ms = int(end * 1000 / config.sample_rate)
                segment = VadSegment(vad.pending_start_ms, end_ms)
                utterance_id = tracker.close(segment)
                transcription_queue.put_nowait(
                    _TranscriptionJob(
                        utterance_id=utterance_id,
                        segment=segment,
                        samples=samples[start:end],
                        duration=duration,
                    )
                )
        transcription_queue.put_nowait(None)
        await worker
