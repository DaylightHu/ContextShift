"""Typed events emitted by the local speech pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """A newly detected utterance that must preempt stale model output."""

    utterance_id: int
    start_ms: int
    detected_at: float


@dataclass(frozen=True, slots=True)
class TranscriptReady:
    """The final local transcription for one complete utterance."""

    utterance_id: int
    start_ms: int
    end_ms: int
    text: str
    asr_elapsed: float
    rtf: float
