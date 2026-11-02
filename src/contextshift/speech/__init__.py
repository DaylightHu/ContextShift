"""Optional local speech input for ContextShift."""

from contextshift.speech.asr import SenseVoiceTranscriber
from contextshift.speech.audio import AudioConfig, InputDevice, list_input_devices
from contextshift.speech.errors import SpeechConfigurationError, SpeechDependencyError
from contextshift.speech.events import SpeechStarted, TranscriptReady
from contextshift.speech.vad import FsmnVad, VadSegment, parse_vad_events

__all__ = [
    "AudioConfig",
    "FsmnVad",
    "InputDevice",
    "SenseVoiceTranscriber",
    "SpeechConfigurationError",
    "SpeechDependencyError",
    "SpeechStarted",
    "TranscriptReady",
    "VadSegment",
    "list_input_devices",
    "parse_vad_events",
]
