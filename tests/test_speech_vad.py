from contextshift.speech.vad import VadSegment, parse_vad_events


def test_parse_streaming_start_and_end_events() -> None:
    pending, segments = parse_vad_events([[120, -1], [-1, 980]])

    assert pending is None
    assert segments == [VadSegment(120, 980)]


def test_parse_complete_event_and_keep_open_start() -> None:
    pending, segments = parse_vad_events([[0, 400], [700, -1]])

    assert pending == 700
    assert segments == [VadSegment(0, 400)]


def test_parse_ignores_invalid_or_unmatched_events() -> None:
    pending, segments = parse_vad_events([[10, 5], [-1, 200], [1]])

    assert pending is None
    assert segments == []
