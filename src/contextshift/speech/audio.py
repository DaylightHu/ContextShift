"""Microphone discovery and 16 kHz block capture."""

import queue
from dataclasses import dataclass
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

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=16)
        self._stream: Any | None = None

    def __enter__(self) -> Self:
        sd = _sounddevice()

        def callback(indata: Any, frames: int, time: Any, status: Any) -> None:
            del frames, time
            if status:
                # Keep capture alive; the caller receives the audio blocks.
                pass
            block = indata.copy().reshape(-1)
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                # Dropping the oldest block is safer than blocking PortAudio's
                # callback thread and causing an input overflow.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(block)

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.block_size,
            dtype="float32",
            device=self.config.device,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def next_block(self, timeout: float = 1.0) -> Any:
        """Block until the next float32 mono audio block is available."""

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError from exc
