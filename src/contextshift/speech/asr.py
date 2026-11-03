"""Lazy SenseVoiceSmall wrapper.

FunASR is optional so the normal text-only ContextShift commands stay small and
usable without PyTorch.  Importing this module therefore never imports FunASR.
"""

from pathlib import Path
from typing import Any

from contextshift.speech.errors import SpeechConfigurationError, SpeechDependencyError


class SenseVoiceTranscriber:
    """Transcribe local 16 kHz mono audio with a downloaded SenseVoice model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        batch_size_s: int = 60,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.device = device
        self.batch_size_s = batch_size_s
        self._model: Any | None = None

        if not self.model_dir.is_dir():
            raise SpeechConfigurationError(
                f"ASR 模型目录不存在：{self.model_dir}. "
                "请先用 ModelScope 下载 SenseVoiceSmall。"
            )
        if batch_size_s < 1:
            raise SpeechConfigurationError("batch_size_s 必须大于 0。")

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
            device=self.device,
            disable_update=True,
            disable_pbar=True,
            trust_remote_code=False,
        )
        return self._model

    def transcribe(self, audio: str | Path | Any) -> str:
        """Return normalized text for a WAV path or a float32 NumPy waveform."""

        model = self._load()
        result = model.generate(
            input=str(audio) if isinstance(audio, Path) else audio,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=self.batch_size_s,
        )
        text = ""
        if result:
            text = str(result[0].get("text", ""))

        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError:
            return text.strip()
        return rich_transcription_postprocess(text).strip()

    def warm_up(self) -> None:
        """Load weights and run one discarded local inference before capture."""

        import numpy as np

        self.transcribe(np.zeros(16_000, dtype="float32"))
