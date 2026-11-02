"""Microphone -> FSMN-VAD -> SenseVoice listener."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from contextshift.speech.asr import SenseVoiceTranscriber
from contextshift.speech.audio import AudioConfig, MicrophoneStream
from contextshift.speech.vad import FsmnVad, VadSegment

TranscriptHandler = Callable[[str, float, float], Awaitable[None] | None]


async def run_microphone_listener(
    *,
    asr_model_dir: str | Path,
    vad_model_dir: str | Path,
    audio_config: AudioConfig | None = None,
    asr_device: str = "cpu",
    on_transcript: TranscriptHandler | None = None,
) -> None:
    """Capture until cancelled and transcribe closed VAD segments in parallel."""

    config = audio_config or AudioConfig()
    vad = FsmnVad(
        vad_model_dir, sample_rate=config.sample_rate, chunk_ms=config.block_ms
    )
    transcriber = SenseVoiceTranscriber(asr_model_dir, device=asr_device)
    blocks: list[Any] = []
    tasks: set[asyncio.Task[None]] = set()
    speech_number = 0
    started_at: float | None = None

    async def transcribe_segment(number: int, segment: Any, duration: float) -> None:
        asr_started = perf_counter()
        text = await asyncio.to_thread(transcriber.transcribe, segment)
        asr_elapsed = perf_counter() - asr_started
        rtf = asr_elapsed / duration if duration > 0 else 0.0
        if on_transcript is not None:
            result = on_transcript(text, asr_elapsed, rtf)
            if asyncio.iscoroutine(result):
                await result
        else:
            print(
                f"\n[Speech {number}] audio={duration:.2f}s "
                f"ASR={asr_elapsed:.3f}s RTF={rtf:.3f}\n{text}",
                flush=True,
            )

    def schedule(segment: VadSegment) -> None:
        nonlocal speech_number
        import numpy as np

        if not blocks:
            return
        samples = np.concatenate(blocks)
        start = max(0, int(segment.start_ms * config.sample_rate / 1000))
        end = min(len(samples), int(segment.end_ms * config.sample_rate / 1000))
        if end <= start:
            return
        speech_number += 1
        duration = (end - start) / config.sample_rate
        task = asyncio.create_task(
            transcribe_segment(speech_number, samples[start:end], duration),
            name=f"contextshift-speech-{speech_number}",
        )
        tasks.add(task)
        task.add_done_callback(tasks.discard)

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
                if vad.pending_start_ms is not None and started_at is None:
                    started_at = perf_counter()
                    print("检测到讲话", flush=True)
                for segment in segments:
                    schedule(segment)
                    started_at = None
    finally:
        if vad.pending_start_ms is not None and blocks:
            import numpy as np

            samples = np.concatenate(blocks)
            start = max(0, int(vad.pending_start_ms * config.sample_rate / 1000))
            end = len(samples)
            if end > start:
                speech_number += 1
                duration = (end - start) / config.sample_rate
                task = asyncio.create_task(
                    transcribe_segment(speech_number, samples[start:end], duration),
                    name=f"contextshift-speech-{speech_number}",
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tuple(tasks), return_exceptions=True)
