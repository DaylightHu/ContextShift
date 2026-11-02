"""Streaming FSMN-VAD adapter and event parsing."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextshift.speech.errors import SpeechConfigurationError, SpeechDependencyError


@dataclass(frozen=True, slots=True)
class VadSegment:
    """A complete speech interval in milliseconds from stream start."""

    start_ms: int
    end_ms: int


def parse_vad_events(
    events: Iterable[Iterable[int]],
    pending_start_ms: int | None = None,
) -> tuple[int | None, list[VadSegment]]:
    """Convert FSMN-VAD boundary events into complete intervals.

    Streaming FSMN-VAD emits ``[start, -1]``, ``[-1, end]`` or a complete
    ``[start, end]`` pair.  Timestamps are absolute milliseconds.
    """

    segments: list[VadSegment] = []
    pending = pending_start_ms

    for raw_event in events:
        event = list(raw_event)
        if len(event) != 2:
            continue
        start, end = int(event[0]), int(event[1])

        if start >= 0 and end == -1:
            pending = start
        elif start == -1 and end >= 0:
            if pending is not None and end > pending:
                segments.append(VadSegment(pending, end))
            pending = None
        elif start >= 0 and end >= 0 and end > start:
            segments.append(VadSegment(start, end))
            pending = None

    return pending, segments


class FsmnVad:
    """Run a downloaded FSMN-VAD checkpoint chunk by chunk."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        sample_rate: int = 16_000,
        chunk_ms: int = 200,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._model: Any | None = None
        self._cache: dict[str, Any] = {}
        self._pending_start_ms: int | None = None

        if not self.model_dir.is_dir():
            raise SpeechConfigurationError(
                f"VAD 模型目录不存在：{self.model_dir}. "
                "请先用 ModelScope 下载 FSMN-VAD。"
            )
        if sample_rate != 16_000:
            raise SpeechConfigurationError("当前 FSMN-VAD 接口要求 16 kHz 音频。")
        if chunk_ms < 1:
            raise SpeechConfigurationError("chunk_ms 必须大于 0。")

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise SpeechDependencyError(
                "缺少语音依赖。请运行：uv sync --extra speech"
            ) from exc

        self._model = AutoModel(
            model=str(self.model_dir),
            device="cpu",
            disable_update=True,
            disable_pbar=True,
            trust_remote_code=False,
        )
        return self._model

    def accept(self, samples: Any, *, is_final: bool = False) -> list[VadSegment]:
        """Feed one non-empty float32 block and return newly closed segments."""

        model = self._load()
        result = model.generate(
            input=samples,
            fs=self.sample_rate,
            cache=self._cache,
            chunk_size=self.chunk_ms,
            is_streaming_input=True,
            is_final=is_final,
            batch_size=1,
            dynamic_silence=False,
        )
        events: list[Iterable[int]] = []
        for item in result or []:
            events.extend(item.get("value", []))

        self._pending_start_ms, segments = parse_vad_events(
            events,
            self._pending_start_ms,
        )
        return segments

    @property
    def pending_start_ms(self) -> int | None:
        return self._pending_start_ms

    def reset(self) -> None:
        """Discard the per-recording streaming cache."""

        self._cache = {}
        self._pending_start_ms = None
