"""Tests for turn-aware streaming session orchestration."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from contextshift.llm import ModelRequest
from contextshift.runtime import ResponseLimits
from contextshift.session import Session, TurnDelta


class InterruptResistantProvider:
    """Keep yielding stale output after cancellation to test the safety net."""

    def __init__(self) -> None:
        self.first_delta_rendered = asyncio.Event()
        self.closed = False

    async def stream(self, request: ModelRequest) -> AsyncGenerator[str, None]:
        if request.input == "A":
            yield "A1"
            self.first_delta_rendered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                yield "A2"
                yield "A3"
            return

        yield "B1"
        yield "B2"

    async def close(self) -> None:
        self.closed = True


class LimitAwareProvider:
    def __init__(self) -> None:
        self.stream_closed = asyncio.Event()
        self.closed = False

    async def stream(self, _: ModelRequest) -> AsyncGenerator[str, None]:
        try:
            yield "abcdef"
            await asyncio.Event().wait()
        finally:
            self.stream_closed.set()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_new_turn_never_renders_stale_deltas() -> None:
    provider = InterruptResistantProvider()
    session = Session(provider)
    rendered: list[TurnDelta] = []

    first_id, first_task = session.submit("A", rendered.append)
    await provider.first_delta_rendered.wait()

    second_id, second_task = session.submit("B", rendered.append)

    await asyncio.gather(first_task, second_task)
    await session.close()

    assert [(delta.turn_id, delta.text) for delta in rendered] == [
        (first_id, "A1"),
        (second_id, "B1"),
        (second_id, "B2"),
    ]
    assert provider.closed


@pytest.mark.asyncio
async def test_close_cancels_active_turn_and_closes_provider() -> None:
    provider = InterruptResistantProvider()
    session = Session(provider)
    rendered: list[TurnDelta] = []

    _, task = session.submit("A", rendered.append)
    await provider.first_delta_rendered.wait()

    await session.close()
    await task

    assert [delta.text for delta in rendered] == ["A1"]
    assert provider.closed


def test_submit_rejects_closed_session() -> None:
    async def run() -> None:
        provider = InterruptResistantProvider()
        session = Session(provider)
        await session.close()

        with pytest.raises(RuntimeError, match="Session is closed"):
            session.submit("question", lambda _: None)

    asyncio.run(run())


@pytest.mark.asyncio
async def test_local_limit_stops_rendering_and_closes_provider_stream() -> None:
    provider = LimitAwareProvider()
    session = Session(
        provider,
        response_limits=ResponseLimits(
            max_lines=3,
            max_characters=3,
            max_complete_sentences=3,
        ),
    )
    rendered: list[TurnDelta] = []

    turn_id, task = session.submit("question", rendered.append)
    result = await task

    assert [(delta.turn_id, delta.text) for delta in rendered] == [(turn_id, "abc")]
    assert result.turn_id == turn_id
    assert result.finish_reason == "local_limit"
    assert result.limit_reason == "max_characters"
    assert result.output_characters == 3
    assert provider.stream_closed.is_set()

    await session.close()
    assert provider.closed
