"""Microphone discovery and 16 kHz block capture."""

import queue
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from types import TracebackType
from typing import Any, Self

from contextshift.speech.errors import SpeechConfigurationError, SpeechDependencyError


@dataclass(frozen=True, slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    block_ms: int = 200
    device: int | str | None = None

    @property
    def block_size(self) -> int:
        if self.sample_rate < 1 or self.channels != 1 or self.block_ms < 1:
            raise SpeechConfigurationError("音频配置必须是正数，并且当前只支持单声道。")
        block_size = self.sample_rate * self.block_ms
        if block_size % 1000:
            raise SpeechConfigurationError(
                "block_ms 必须能在采样率下换算为整数采样点。"
            )
        return block_size // 1000


@dataclass(frozen=True, slots=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float


@dataclass(frozen=True, slots=True)
class AudioPacket:
    samples: Any
    captured_at: float


def _sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SpeechDependencyError(
            "缺少麦克风依赖。请运行：uv sync --extra speech"
        ) from exc
    return sd


def list_input_devices() -> list[InputDevice]:
    """Return input-capable PortAudio devices without opening a stream."""

    devices = _sounddevice().query_devices()
    return [
        InputDevice(
            index=index,
            name=str(device["name"]),
            max_input_channels=int(device["max_input_channels"]),
            default_sample_rate=float(device["default_samplerate"]),
        )
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]


class MicrophoneStream:
    """Queue-based PortAudio input stream for async consumers."""

    def __init__(
        self,
        config: AudioConfig,
        *,
        strict: bool = False,
        on_packet: Callable[[AudioPacket], None] | None = None,
    ) -> None:
        self.config = config
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=16)
        self._stream: Any | None = None
        self._strict = strict
        self._on_packet = on_packet
        self._error: BaseException | None = None

    def __enter__(self) -> Self:
        sd = _sounddevice()
        clock_offset: float | None = None

        def callback(indata: Any, frames: int, time: Any, status: Any) -> None:
            nonlocal clock_offset
            if self._error is not None:
                return
            try:
                if status and self._strict:
                    raise SpeechConfigurationError(f"音频采集不连续：{status}")
                if clock_offset is None:
                    clock_offset = perf_counter() - time.currentTime
                packet = AudioPacket(
                    indata.copy().reshape(-1), clock_offset + time.inputBufferAdcTime
                )
                if self._strict and frames != self.config.block_size:
                    raise SpeechConfigurationError(
                        "音频采集块长度变化，请重新启动监听。"
                    )
                if self._on_packet is not None:
                    self._on_packet(packet)
                try:
                    self._queue.put_nowait(packet)
                except queue.Full:
                    if self._strict:
                        raise SpeechConfigurationError(
                            "音频处理落后，已停止监听以防止上下文错位。"
                        )
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(packet)
            except Exception as exc:  # noqa: BLE001 - callback errors reach the consumer
                self._error = exc

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.block_size,
            dtype="float32",
            device=self.config.device,
            callback=callback,
        )
        try:
            self._stream.start()
        except BaseException:
            self._stream.close()
            self._stream = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
            finally:
                stream.close()

    def next_block(self, timeout: float = 1.0) -> Any:
        """Block until the next float32 mono audio block is available."""

        return self.next_packet(timeout).samples

    def next_packet(self, timeout: float = 1.0) -> AudioPacket:
        if self._error is not None:
            raise self._error
        try:
            packet = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            if self._error is not None:
                raise self._error
            raise TimeoutError from exc
        if self._error is not None:
            raise self._error
        return packet
