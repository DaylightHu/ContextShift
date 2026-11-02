"""Tests for hard local response limits."""

import pytest

from contextshift.runtime import ResponseLimiter, ResponseLimits


def test_character_limit_truncates_across_deltas() -> None:
    limiter = ResponseLimiter(
        ResponseLimits(
            max_lines=3,
            max_characters=5,
            max_complete_sentences=3,
        )
    )

    first = limiter.push("abc")
    second = limiter.push("def")
    stale = limiter.push("ignored")

    assert first.text == "abc"
    assert not first.reached_limit
    assert second.text == "de"
    assert second.reached_limit
    assert second.reason == "max_characters"
    assert stale.text == ""
    assert stale.reached_limit
    assert limiter.characters == 5


def test_line_limit_excludes_the_extra_newline() -> None:
    limiter = ResponseLimiter(
        ResponseLimits(
            max_lines=3,
            max_characters=100,
            max_complete_sentences=10,
        )
    )

    result = limiter.push("line 1\nline 2\nline 3\nline 4")

    assert result.text == "line 1\nline 2\nline 3"
    assert result.reached_limit
    assert result.reason == "max_lines"
    assert limiter.lines == 3


def test_sentence_limit_includes_the_final_terminator() -> None:
    limiter = ResponseLimiter(
        ResponseLimits(
            max_lines=10,
            max_characters=100,
            max_complete_sentences=3,
        )
    )

    result = limiter.push("第一句。第二句！第三句？第四句。")

    assert result.text == "第一句。第二句！第三句？"
    assert result.reached_limit
    assert result.reason == "max_complete_sentences"
    assert limiter.complete_sentences == 3


def test_consecutive_terminators_count_as_one_sentence() -> None:
    limiter = ResponseLimiter(
        ResponseLimits(
            max_lines=10,
            max_characters=100,
            max_complete_sentences=2,
        )
    )

    result = limiter.push("Really?! Yes!")

    assert result.text == "Really?! Yes!"
    assert result.reached_limit
    assert limiter.complete_sentences == 2


def test_technical_periods_do_not_count_as_sentences() -> None:
    limiter = ResponseLimiter(
        ResponseLimits(
            max_lines=10,
            max_characters=100,
            max_complete_sentences=1,
        )
    )

    result = limiter.push("manager.py uses obj.method() on Python 3.11")

    assert result.text == "manager.py uses obj.method() on Python 3.11"
    assert not result.reached_limit
    assert limiter.complete_sentences == 0


def test_ascii_period_is_confirmed_by_following_whitespace() -> None:
    limiter = ResponseLimiter(
        ResponseLimits(
            max_lines=10,
            max_characters=100,
            max_complete_sentences=2,
        )
    )

    first = limiter.push("First sentence. ")
    second = limiter.push("Second sentence... Next")

    assert first.text == "First sentence. "
    assert not first.reached_limit
    assert second.text == "Second sentence..."
    assert second.reached_limit
    assert second.reason == "max_complete_sentences"
    assert limiter.complete_sentences == 2


@pytest.mark.parametrize(
    "field_name",
    ["max_lines", "max_characters", "max_complete_sentences"],
)
def test_limits_must_be_positive(field_name: str) -> None:
    values = {
        "max_lines": 3,
        "max_characters": 240,
        "max_complete_sentences": 3,
    }
    values[field_name] = 0

    with pytest.raises(ValueError, match=f"{field_name} must be at least 1"):
        ResponseLimits(**values)
