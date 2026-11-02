"""Source and role are a single invariant, never independently configured."""

from dataclasses import asdict

import pytest

from contextshift.speech.events import SpeechStarted, TranscriptReady


@pytest.mark.parametrize(
    ("source", "role"), [("microphone", "candidate"), ("system", "interviewer")]
)
def test_both_events_expose_matching_source_and_role(source, role) -> None:
    events = [
        SpeechStarted(1, 0, 1.0, source=source),
        TranscriptReady(1, 0, 100, "actual speech", 0.01, 0.1, source=source),
    ]
    assert all(asdict(event)["role"] == role for event in events)
    assert all(event.source == source for event in events)


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown audio source"):
        SpeechStarted(1, 0, 1.0, source="other")
    with pytest.raises(ValueError, match="Unknown audio source"):
        TranscriptReady(1, 0, 100, "speech", 0.01, 0.1, source="other")
