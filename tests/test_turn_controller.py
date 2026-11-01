import asyncio

import pytest

from contextshift.runtime import TurnController


@pytest.mark.asyncio
async def test_start_assigns_monotonic_turn_ids() -> None:
    controller = TurnController()

    async def return_turn_id(turn_id: int) -> int:
        return turn_id

    first_id, first_task = controller.start(return_turn_id)
    assert await first_task == first_id

    second_id, second_task = controller.start(return_turn_id)
    assert await second_task == second_id

    assert first_id == 1
    assert second_id == 2
    assert controller.is_current(second_id)


@pytest.mark.asyncio
async def test_new_turn_cancels_previous_task() -> None:
    controller = TurnController()
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()

    async def wait_forever(_: int) -> None:
        first_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def finish_immediately(turn_id: int) -> int:
        return turn_id

    first_id, first_task = controller.start(wait_forever)
    await first_started.wait()

    second_id, second_task = controller.start(finish_immediately)

    assert await second_task == second_id
    await first_cancelled.wait()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    assert not controller.is_current(first_id)
    assert controller.is_current(second_id)


@pytest.mark.asyncio
async def test_cancel_invalidates_current_turn() -> None:
    controller = TurnController()
    started = asyncio.Event()

    async def wait_forever(_: int) -> None:
        started.set()
        await asyncio.Event().wait()

    turn_id, task = controller.start(wait_forever)
    await started.wait()

    controller.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not controller.is_current(turn_id)
