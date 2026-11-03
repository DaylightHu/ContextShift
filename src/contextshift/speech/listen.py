"""Independent local capture -> FSMN-VAD -> SenseVoice channels."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from time import perf_counter
from typing import Any

from contextshift.speech.asr import SenseVoiceTranscriber
from contextshift.speech.audio import AudioConfig, MicrophoneStream
from contextshift.speech.errors import SpeechConfigurationError
from contextshift.speech.events import (
    AudioSource,
    SpeechStarted,
    TranscriptReady,
    role_for_source,
)
from contextshift.speech.vad import FsmnVad, VadSegment

SpeechStartedHandler = Callable[[SpeechStarted], None]
TranscriptHandler = Callable[[TranscriptReady], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class PreparedChannel:
    config: AudioConfig
    vad: FsmnVad
    transcriber: SenseVoiceTranscriber


def prepare_channels(
    *,
    asr_model_dir: str | Path,
    vad_model_dir: str | Path,
    microphone_config: AudioConfig,
    system_config: AudioConfig,
    asr_device: str = "cpu",
    cancelled: threading.Event | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[AudioSource, PreparedChannel]:
    """Prepare independent models without opening audio or emitting speech."""

    result = {}

    def check_cancelled() -> None:
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("启动已取消。")

    for source, config in (
        ("microphone", microphone_config),
        ("system", system_config),
    ):
        check_cancelled()
        if on_progress is not None:
            on_progress(
                "正在预热麦克风模型…"
                if source == "microphone"
                else "正在预热会议音频模型…"
            )
        vad = FsmnVad(
            vad_model_dir, sample_rate=config.sample_rate, chunk_ms=config.block_ms
        )
        transcriber = SenseVoiceTranscriber(asr_model_dir, device=asr_device)
        vad.warm_up()
        check_cancelled()
        transcriber.warm_up()
        check_cancelled()
        result[source] = PreparedChannel(config, vad, transcriber)
    return result


@dataclass(frozen=True, slots=True)
class _TranscriptionJob:
    utterance_id: int
    segment: VadSegment
    samples: Any
    duration: float
    source: AudioSource = "microphone"


class _SpeechEventTracker:
    """Assign stable IDs and emit exactly one start event per utterance."""

    def __init__(
        self,
        handler: SpeechStartedHandler | None,
        source: AudioSource = "microphone",
    ) -> None:
        self._handler = handler
        self._source = source
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
            source=self._source,
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
                source=job.source,
            )
            if on_transcript is not None:
                result = on_transcript(event)
                if isawaitable(result):
                    await result
            else:
                print(
                    f"\n[{job.source} Speech {job.utterance_id}] "
                    f"audio={job.duration:.2f}s "
                    f"ASR={asr_elapsed:.3f}s RTF={rtf:.3f}\n{text}",
                    flush=True,
                )
        finally:
            queue.task_done()


async def run_audio_listener(
    *,
    source: AudioSource,
    asr_model_dir: str | Path,
    vad_model_dir: str | Path,
    audio_config: AudioConfig | None = None,
    asr_device: str = "cpu",
    on_speech_started: SpeechStartedHandler | None = None,
    on_transcript: TranscriptHandler | None = None,
    prepared: PreparedChannel | None = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Capture until cancelled and transcribe closed VAD segments in order."""

    role_for_source(source)
    config = audio_config or AudioConfig()
    if prepared is not None:
        if prepared.config != config:
            raise SpeechConfigurationError("预热模型与音频设备配置不一致。")
        vad, transcriber = prepared.vad, prepared.transcriber
    else:
        vad = FsmnVad(
            vad_model_dir, sample_rate=config.sample_rate, chunk_ms=config.block_ms
        )
        transcriber = SenseVoiceTranscriber(asr_model_dir, device=asr_device)
    blocks: list[Any] = []
    tracker = _SpeechEventTracker(on_speech_started, source)
    transcription_queue: asyncio.Queue[_TranscriptionJob | None] = asyncio.Queue()
    worker = asyncio.create_task(
        _run_transcription_worker(
            transcription_queue,
            transcriber,
            on_transcript,
        ),
        name=f"contextshift-{source}-asr-worker",
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
                source=source,
            )
        )

    print(f"开始监听 {source}，按 Ctrl-C 停止。", flush=True)
    try:
        with MicrophoneStream(config) as microphone:
            if on_ready is not None:
                on_ready()
            while True:
                if worker.done():
                    await worker  # Surface ASR failures instead of capturing forever.
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
        # Shutdown must not deliver queued transcripts and start fresh model turns.
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def run_microphone_listener(
    *,
    asr_model_dir: str | Path,
    vad_model_dir: str | Path,
    audio_config: AudioConfig | None = None,
    asr_device: str = "cpu",
    on_speech_started: SpeechStartedHandler | None = None,
    on_transcript: TranscriptHandler | None = None,
) -> None:
    """Keep the standalone microphone transcription command available."""

    await run_audio_listener(
        source="microphone",
        asr_model_dir=asr_model_dir,
        vad_model_dir=vad_model_dir,
        audio_config=audio_config,
        asr_device=asr_device,
        on_speech_started=on_speech_started,
        on_transcript=on_transcript,
    )


async def run_dual_audio_listener(
    *,
    asr_model_dir: str | Path,
    vad_model_dir: str | Path,
    microphone_config: AudioConfig,
    system_config: AudioConfig,
    asr_device: str = "cpu",
    on_speech_started: SpeechStartedHandler | None = None,
    on_transcript: TranscriptHandler | None = None,
    prepared_channels: dict[AudioSource, PreparedChannel] | None = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Run isolated VAD/ASR queues for mic and an explicit loopback input.

    The system device must already expose meeting audio as a PortAudio input.
    Native macOS system-audio capture is a separate adapter, not implemented here.
    """

    if system_config.device is None:
        raise SpeechConfigurationError("系统音频必须显式指定回环输入设备。")
    if microphone_config.device == system_config.device:
        raise SpeechConfigurationError("麦克风和系统音频必须使用不同的输入设备。")
    # Resolve the default mic before opening streams; it may be the loopback input.
    if microphone_config.device is None:
        from contextshift.speech.audio import _sounddevice

        sd = _sounddevice()
        mic = sd.query_devices(None, "input")
        system = sd.query_devices(system_config.device, "input")
        if mic["index"] == system["index"]:
            raise SpeechConfigurationError(
                "默认麦克风与系统音频相同，请用 --input-device 指定独立麦克风。"
            )
    # Validate both configs before either task opens a capture stream.
    _ = microphone_config.block_size, system_config.block_size
    if prepared_channels is not None and set(prepared_channels) != {
        "microphone",
        "system",
    }:
        raise SpeechConfigurationError("两路模型尚未全部预热。")
    opened: set[AudioSource] = set()

    def channel_opened(source: AudioSource) -> None:
        if source in opened:
            return
        opened.add(source)
        if len(opened) == 2 and on_ready is not None:
            on_ready()

    tasks = [
        asyncio.create_task(
            run_audio_listener(
                source=source,
                asr_model_dir=asr_model_dir,
                vad_model_dir=vad_model_dir,
                audio_config=config,
                asr_device=asr_device,
                on_speech_started=on_speech_started,
                on_transcript=on_transcript,
                **(
                    {"prepared": prepared_channels[source]}
                    if prepared_channels is not None
                    else {}
                ),
                **(
                    {"on_ready": lambda source=source: channel_opened(source)}
                    if on_ready is not None
                    else {}
                ),
            ),
            name=f"contextshift-{source}-listener",
        )
        for source, config in (
            ("microphone", microphone_config),
            ("system", system_config),
        )
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
