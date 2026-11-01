"""Turn lifecycle and cancellation management."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class TurnController:
    """Run at most one active turn and invalidate stale output."""

    def __init__(self) -> None:
        self._turn_id = 0
        self._active_task: asyncio.Task[Any] | None = None

    @property
    def current_turn_id(self) -> int:
        """Return the newest turn identifier."""

        return self._turn_id

    def is_current(self, turn_id: int) -> bool:
        """Return whether output still belongs to the active turn."""

        return turn_id == self._turn_id

    def start(
        self,
        operation: Callable[[int], Coroutine[Any, Any, T]],
    ) -> tuple[int, asyncio.Task[T]]:
        """Cancel the previous task and immediately launch a new turn."""

        self._turn_id += 1
        self._cancel_active_task()

        turn_id = self._turn_id
        task = asyncio.create_task(
            operation(turn_id),
            name=f"contextshift-turn-{turn_id}",
        )
        self._active_task = task
        task.add_done_callback(self._on_task_done)

        return turn_id, task

    def cancel(self) -> None:
        """Cancel active work and invalidate any output already in flight."""

        self._turn_id += 1
        self._cancel_active_task()

    def _cancel_active_task(self) -> None:
        task = self._active_task

        if task is not None and not task.done():
            task.cancel()

        self._active_task = None

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        if self._active_task is task:
            self._active_task = None

        if not task.cancelled():
            task.exception()
