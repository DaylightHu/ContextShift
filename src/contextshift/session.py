"""Streaming session orchestration with turn-aware output filtering."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from contextshift.llm import ModelProvider, ModelRequest
from contextshift.runtime import TurnController

DEFAULT_INSTRUCTIONS = "直接回答，不复述问题。最多三行。不要使用开场白、总结句或比喻。"


@dataclass(frozen=True, slots=True)
class TurnDelta:
    """A rendered text fragment associated with its originating turn."""

    turn_id: int
    text: str


DeltaHandler = Callable[[TurnDelta], None]


class Session:
    """Submit questions while preventing stale turns from rendering output."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        turn_controller: TurnController | None = None,
    ) -> None:
        self._provider = provider
        self._instructions = instructions
        self._turn_controller = turn_controller or TurnController()
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def submit(
        self,
        question: str,
        on_delta: DeltaHandler,
    ) -> tuple[int, asyncio.Task[None]]:
        """Start a turn, cancelling and invalidating the previous turn."""

        if self._closed:
            raise RuntimeError("Session is closed")

        request = ModelRequest(
            input=question,
            instructions=self._instructions,
        )

        async def run(turn_id: int) -> None:
            async for text in self._provider.stream(request):
                if not self._turn_controller.is_current(turn_id):
                    break

                # Keep the current-turn check and rendering synchronous so a new
                # turn cannot be scheduled between them on the event loop.
                on_delta(TurnDelta(turn_id=turn_id, text=text))

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
