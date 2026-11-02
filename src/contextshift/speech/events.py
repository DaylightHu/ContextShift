"""Typed events emitted by the local speech pipeline."""

from dataclasses import dataclass, field
from typing import Literal

AudioSource = Literal["microphone", "system"]
SpeakerRole = Literal["candidate", "interviewer"]


def role_for_source(source: AudioSource) -> SpeakerRole:
    """Keep source and speaker identity inseparable at the event boundary."""

    if source == "microphone":
        return "candidate"
    if source == "system":
        return "interviewer"
    raise ValueError(f"Unknown audio source: {source!r}")


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """A source-local utterance start; only system speech preempts output."""

    utterance_id: int
    start_ms: int
    detected_at: float
    source: AudioSource = "microphone"
    role: SpeakerRole = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", role_for_source(self.source))


@dataclass(frozen=True, slots=True)
class TranscriptReady:
    """The final local transcription for one complete utterance."""

    utterance_id: int
    start_ms: int
    end_ms: int
    text: str
    asr_elapsed: float
    rtf: float
    source: AudioSource = "microphone"
    role: SpeakerRole = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", role_for_source(self.source))
