"""Tests for turn-aware streaming session orchestration."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from contextshift.llm import ModelRequest
from contextshift.llm.base import ProviderOutputLimit
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
async def test_cancelled_context_resolution_cannot_start_a_stale_provider_request():
    entered = asyncio.Event()
    sent = []

    class Provider:
        async def stream(self, request):
            sent.append(request)
            yield request.input

        async def close(self):
            pass

    async def resolve(request):
        if request.query == "A":
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return ModelRequest("stale code", request.instructions)
        return ModelRequest("B with current code", request.instructions)

    session = Session(Provider(), context_resolver=resolve)
    rendered = []
    _, first = session.submit("first", rendered.append, query="A")
    await asyncio.wait_for(entered.wait(), 1)
    _, second = session.submit("second", rendered.append, query="B")
    await asyncio.gather(first, second)
    assert [request.input for request in sent] == ["B with current code"]
    assert [delta.text for delta in rendered] == ["B with current code"]
    await session.close()


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


@pytest.mark.asyncio
async def test_provider_output_limit_keeps_partial_text_and_reports_truncation() -> (
    None
):
    class Provider:
        async def stream(self, request):
            yield "尚未结束的回答"
            raise ProviderOutputLimit("cloud limit")

        async def close(self):
            pass

    session = Session(Provider())
    rendered = []
    _, task = session.submit("问题", rendered.append)
    result = await task
    assert result.finish_reason == "provider_limit"
    assert "".join(delta.text for delta in rendered) == "尚未结束的回答"
    await session.close()
