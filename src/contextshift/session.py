"""Streaming session orchestration with turn-aware output filtering."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from contextshift.llm import ModelProvider, ModelRequest
from contextshift.runtime import (
    LimitReason,
    ResponseLimiter,
    ResponseLimits,
    TurnController,
)

DEFAULT_INSTRUCTIONS = "直接回答，不复述问题。最多三行。不要使用开场白、总结句或比喻。"


@dataclass(frozen=True, slots=True)
class TurnDelta:
    """A rendered text fragment associated with its originating turn."""

    turn_id: int
    text: str


TurnFinishReason = Literal["completed", "local_limit", "interrupted"]


@dataclass(frozen=True, slots=True)
class TurnResult:
    """How a turn finished and how much text reached the renderer."""

    turn_id: int
    finish_reason: TurnFinishReason
    output_characters: int
    limit_reason: LimitReason | None = None


DeltaHandler = Callable[[TurnDelta], None]


class Session:
    """Submit questions while preventing stale turns from rendering output."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        turn_controller: TurnController | None = None,
        response_limits: ResponseLimits | None = None,
    ) -> None:
        self._provider = provider
        self._instructions = instructions
        self._turn_controller = turn_controller or TurnController()
        self._response_limits = response_limits or ResponseLimits()
        self._tasks: set[asyncio.Task[TurnResult]] = set()
        self._closed = False

    def submit(
        self,
        question: str,
        on_delta: DeltaHandler,
    ) -> tuple[int, asyncio.Task[TurnResult]]:
        """Start a turn, cancelling and invalidating the previous turn."""

        if self._closed:
            raise RuntimeError("Session is closed")

        request = ModelRequest(
            input=question,
            instructions=self._instructions,
        )

        async def run(turn_id: int) -> TurnResult:
            limiter = ResponseLimiter(self._response_limits)
            stream = self._provider.stream(request)

            try:
                async for text in stream:
                    if not self._turn_controller.is_current(turn_id):
                        return TurnResult(
                            turn_id=turn_id,
                            finish_reason="interrupted",
                            output_characters=limiter.characters,
                        )

                    limited = limiter.push(text)

                    # Keep the current-turn check, limiting, and rendering
                    # synchronous so a new turn cannot be scheduled between them.
                    if limited.text:
                        on_delta(TurnDelta(turn_id=turn_id, text=limited.text))

                    if limited.reached_limit:
                        return TurnResult(
                            turn_id=turn_id,
                            finish_reason="local_limit",
                            output_characters=limiter.characters,
                            limit_reason=limited.reason,
                        )

                return TurnResult(
                    turn_id=turn_id,
                    finish_reason="completed",
                    output_characters=limiter.characters,
                )
            finally:
                await stream.aclose()

        turn_id, task = self._turn_controller.start(run)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return turn_id, task

    def cancel(self) -> None:
        """Cancel active generation and invalidate its remaining deltas."""

        self._turn_controller.cancel()

    async def close(self) -> None:
        """Stop active work, wait for cleanup, and close the provider."""

        if self._closed:
            return

        self._closed = True
        self._turn_controller.cancel()

        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

        await self._provider.close()
