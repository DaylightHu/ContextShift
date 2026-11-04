"""Tests for the reusable DeepSeek provider."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from contextshift.config import Settings
from contextshift.llm import DeepSeekProvider, ModelRequest


@dataclass(frozen=True)
class FakeEvent:
    type: str
    delta: str = ""


class FakeStream:
    def __init__(self, events: list[FakeEvent]) -> None:
        self._events = events
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[FakeEvent]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []

    async def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        stream = FakeStream(
            [
                FakeEvent("response.created"),
                FakeEvent("response.output_text.delta", "answer"),
            ]
        )
        self.streams.append(stream)
        return stream


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_reuses_client_and_closes_each_response_stream() -> None:
    settings = Settings(DEEPSEEK_API_KEY="test-key")
    assert settings.max_output_tokens == 2048
    fake_client = FakeClient()
    provider = DeepSeekProvider(
        settings,
        client=cast(AsyncOpenAI, fake_client),
    )
    request = ModelRequest(input="question", instructions="be brief")

    first = [delta async for delta in provider.stream(request)]
    second = [delta async for delta in provider.stream(request)]
    await provider.close()
    await provider.close()

    assert first == ["answer"]
    assert second == ["answer"]
    assert len(fake_client.responses.calls) == 2
    assert all(stream.closed for stream in fake_client.responses.streams)
    assert fake_client.close_calls == 1

    call = fake_client.responses.calls[0]
    assert call == {
        "model": settings.model,
        "instructions": "be brief",
        "input": "question",
        "reasoning": {"effort": settings.reasoning_effort},
        "max_output_tokens": settings.max_output_tokens,
        "stream": True,
    }


def test_output_budget_can_be_raised_above_previous_ceiling() -> None:
    settings = Settings(
        DEEPSEEK_API_KEY="test-key", CONTEXTSHIFT_MAX_OUTPUT_TOKENS=4096
    )
    assert settings.max_output_tokens == 4096


@pytest.mark.asyncio
async def test_rejects_requests_after_close() -> None:
    settings = Settings(DEEPSEEK_API_KEY="test-key")
    fake_client = FakeClient()
    provider = DeepSeekProvider(
        settings,
        client=cast(AsyncOpenAI, fake_client),
    )
    request = ModelRequest(input="question", instructions="be brief")
    await provider.close()

    with pytest.raises(RuntimeError, match="DeepSeekProvider is closed"):
        await anext(provider.stream(request))


@pytest.mark.asyncio
async def test_prepare_checks_model_on_same_client_without_generating_text() -> None:
    settings = Settings(DEEPSEEK_API_KEY="test-key")
    client = FakeClient()
    calls = []

    async def list_models(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(id=settings.model)])

    client.models = SimpleNamespace(list=list_models)
    provider = DeepSeekProvider(settings, client=cast(AsyncOpenAI, client))
    await provider.prepare()
    assert calls == [{"timeout": 10}]
    assert not client.responses.calls
    await provider.close()


@pytest.mark.asyncio
async def test_prepare_rejects_unknown_model_before_audio_start() -> None:
    client = FakeClient()

    async def list_models(**kwargs):
        return SimpleNamespace(data=[])

    client.models = SimpleNamespace(list=list_models)
    provider = DeepSeekProvider(
        Settings(DEEPSEEK_API_KEY="test-key"), client=cast(AsyncOpenAI, client)
    )
    with pytest.raises(ValueError, match="所选模型"):
        await provider.prepare()
    assert not client.responses.calls
    await provider.close()
