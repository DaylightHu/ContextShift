"""Timestamp-aligned speaker reference and local WebRTC AEC3 preprocessing."""

import threading
from collections import deque
from time import monotonic
from typing import Any

from contextshift.speech.audio import AudioPacket
from contextshift.speech.errors import SpeechConfigurationError, SpeechDependencyError


class EchoProcessor:
    """One processor per session, called serially by the microphone listener.

    Reference packets come directly from the system capture callback, before
    system VAD/ASR can stall them. Actual zero-valued packets are valid silence;
    missing packets are an error, never substituted with invented silence.
    """

    def __init__(self, sample_rate: int, *, reference_timeout: float = 1.0) -> None:
        import numpy as np

        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError as exc:
            raise SpeechDependencyError(
                "缺少回声消除依赖。请运行：uv sync --extra speech"
            ) from exc
        if sample_rate != 16_000:
            raise SpeechConfigurationError("当前回声消除链路要求两路均为 16 kHz。")
        self.sample_rate = sample_rate
        self._timeout = reference_timeout
        self._condition = threading.Condition()
        self._packets: deque[AudioPacket] = deque()
        self._closed = False
        self._started = False
        self._last_mic_end: float | None = None
        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            echo_cancellation=True,
            noise_suppression=False,
            auto_gain_control=False,
        )
        # Load the native processor before the UI can announce readiness.
        silence = np.zeros(sample_rate // 100, dtype=np.float32)
        self._processor.process(silence, silence)
        self._processor.reset()

    def publish(self, packet: AudioPacket) -> None:
        """Called by the system PortAudio callback; never waits for recognition."""
        with self._condition:
            if self._closed:
                return
            if self._packets:
                previous = self._packets[-1]
                expected = (
                    previous.captured_at + len(previous.samples) / self.sample_rate
                )
                if abs(packet.captured_at - expected) > 0.002:
                    raise SpeechConfigurationError(
                        "系统音频参考流不连续，请重新启动监听。"
                    )
            self._packets.append(packet)
            while (
                len(self._packets) > 1
                and packet.captured_at - self._packets[0].captured_at > 5.0
            ):
                self._packets.popleft()
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def clean(self, packet: AudioPacket) -> Any | None:
        """Return cleaned float32 mic samples, or skip pre-readiness overlap.

        Both streams use their ADC timestamps mapped to the same monotonic
        clock. Interpolation follows the reference clock; AEC3 estimates the
        remaining speaker/room delay. No speech/content-based mic muting occurs.
        """
        import numpy as np

        start = packet.captured_at
        count = len(packet.samples)
        end = start + count / self.sample_rate
        if self._last_mic_end is not None and abs(start - self._last_mic_end) > 0.002:
            raise SpeechConfigurationError("麦克风音频不连续，请重新启动监听。")
        self._last_mic_end = end
        deadline = monotonic() + self._timeout
        with self._condition:
            while True:
                if self._closed:
                    raise SpeechConfigurationError("回声消除已停止。")
                if self._packets:
                    first = self._packets[0]
                    if start < first.captured_at - 1 / self.sample_rate:
                        if not self._started:
                            return None  # Only before the microphone ready gate.
                        raise SpeechConfigurationError("麦克风已落后于系统音频参考。")
                    last = self._packets[-1]
                    available_end = (
                        last.captured_at + len(last.samples) / self.sample_rate
                    )
                    if available_end >= end - 1 / self.sample_rate:
                        packets = [
                            p
                            for p in self._packets
                            if p.captured_at <= end
                            and p.captured_at + len(p.samples) / self.sample_rate
                            > start
                        ]
                        break
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise SpeechConfigurationError(
                        "系统音频参考中断，已停止监听；未处理的麦克风内容不会写入历史。"
                    )
                self._condition.wait(remaining)

        # Use relative coordinates to retain precision for long running sessions.
        reference_times = np.concatenate(
            [
                (p.captured_at - start) + np.arange(len(p.samples)) / self.sample_rate
                for p in packets
            ]
        )
        reference = np.interp(
            np.arange(count) / self.sample_rate,
            reference_times,
            np.concatenate([p.samples for p in packets]),
        ).astype(np.float32)
        near = np.ascontiguousarray(packet.samples, dtype=np.float32)
        if not np.isfinite(near).all() or not np.isfinite(reference).all():
            raise SpeechConfigurationError("音频包含无效采样值。")
        result = self._processor.process(near, reference)
        if result.shape != near.shape or not np.isfinite(result).all():
            raise SpeechConfigurationError("回声消除返回无效音频。")
        self._started = True
        return result
