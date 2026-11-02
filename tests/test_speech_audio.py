import pytest

from contextshift.speech.audio import AudioConfig
from contextshift.speech.errors import SpeechConfigurationError


def test_audio_config_converts_200ms_at_16khz() -> None:
    assert AudioConfig().block_size == 3200


def test_audio_config_rejects_non_mono_or_non_integral_blocks() -> None:
    with pytest.raises(SpeechConfigurationError):
        _ = AudioConfig(channels=2).block_size
    with pytest.raises(SpeechConfigurationError):
        _ = AudioConfig(block_ms=1, sample_rate=44_100).block_size
