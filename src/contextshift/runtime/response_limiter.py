"""Hard local limits for streamed model responses."""

from dataclasses import dataclass
from typing import Literal

LimitReason = Literal[
    "max_lines",
    "max_characters",
    "max_complete_sentences",
]

IMMEDIATE_SENTENCE_TERMINATORS = frozenset("。!?！？")


@dataclass(frozen=True, slots=True)
class ResponseLimits:
    """Maximum rendered size for one model response."""

    max_lines: int | None = None
    max_characters: int | None = None
    max_complete_sentences: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_lines",
            "max_characters",
            "max_complete_sentences",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be at least 1")


@dataclass(frozen=True, slots=True)
class LimitedChunk:
    """The safe prefix of a delta and its limiting state."""

    text: str
    reached_limit: bool
    reason: LimitReason | None = None


class ResponseLimiter:
    """Incrementally enforce line, character, and sentence limits."""

    def __init__(self, limits: ResponseLimits | None = None) -> None:
        self._limits = limits or ResponseLimits()
        self._characters = 0
        self._lines = 1
        self._complete_sentences = 0
        self._in_terminator_run = False
        self._pending_ascii_period = False
        self._reason: LimitReason | None = None

    @property
    def characters(self) -> int:
        return self._characters

    @property
    def lines(self) -> int:
        return self._lines

    @property
    def complete_sentences(self) -> int:
        return self._complete_sentences

    @property
    def reached_limit(self) -> bool:
        return self._reason is not None

    @property
    def reason(self) -> LimitReason | None:
        return self._reason

    def push(self, delta: str) -> LimitedChunk:
        """Return the renderable prefix of one streamed text delta."""

        if self._reason is not None:
            return LimitedChunk(text="", reached_limit=True, reason=self._reason)

        accepted: list[str] = []

        for character in delta:
            if self._pending_ascii_period and character != ".":
                if character.isspace():
                    self._complete_sentences += 1
                    if (
                        self._limits.max_complete_sentences is not None
                        and self._complete_sentences
                        >= self._limits.max_complete_sentences
                    ):
                        self._pending_ascii_period = False
                        return self._stop(
                            accepted,
                            "max_complete_sentences",
                        )
                self._pending_ascii_period = False

            if (
                character == "\n"
                and self._limits.max_lines is not None
                and self._lines >= self._limits.max_lines
            ):
                return self._stop(accepted, "max_lines")

            accepted.append(character)
            self._characters += 1

            if character == "\n":
                self._lines += 1

            if character == ".":
                self._pending_ascii_period = True
                self._in_terminator_run = False
            elif character in IMMEDIATE_SENTENCE_TERMINATORS:
                if not self._in_terminator_run:
                    self._complete_sentences += 1
                self._in_terminator_run = True
            else:
                self._in_terminator_run = False

            if (
                self._limits.max_complete_sentences is not None
                and self._complete_sentences >= self._limits.max_complete_sentences
            ):
                return self._stop(accepted, "max_complete_sentences")

            if (
                self._limits.max_characters is not None
                and self._characters >= self._limits.max_characters
            ):
                return self._stop(accepted, "max_characters")

        return LimitedChunk(text="".join(accepted), reached_limit=False)

    def _stop(
        self,
        accepted: list[str],
        reason: LimitReason,
    ) -> LimitedChunk:
        self._reason = reason
        return LimitedChunk(
            text="".join(accepted),
            reached_limit=True,
            reason=reason,
        )
