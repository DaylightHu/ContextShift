"""Tests for latency benchmark collection and aggregation."""

from collections.abc import AsyncGenerator

import pytest

from contextshift.benchmark import (
    BenchmarkResult,
    BenchmarkSummary,
    run_latency_benchmark,
)
from contextshift.llm import ModelRequest
from contextshift.runtime import ResponseLimits
from contextshift.session import Session


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class BenchmarkProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def stream(self, _: ModelRequest) -> AsyncGenerator[str, None]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("provider unavailable")

        yield "first"
        yield " second"

    async def close(self) -> None:
        self.closed = True


class EmptyProvider:
    async def stream(self, _: ModelRequest) -> AsyncGenerator[str, None]:
        if False:
            yield ""

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_benchmark_collects_successes_and_errors() -> None:
    provider = BenchmarkProvider()
    session = Session(provider)
    reported: list[BenchmarkResult] = []
    clock = SequenceClock([10.0, 10.2, 10.8, 20.0, 20.5])

    summary = await run_latency_benchmark(
        session,
        "question",
        2,
        on_result=reported.append,
        clock=clock,
    )
    await session.close()

    assert provider.calls == 2
    assert provider.closed
    assert reported == list(summary.results)
    assert summary.runs == 2
    assert summary.successes == 1
    assert summary.errors == 1
    assert summary.local_limits == 0

    success, failure = summary.results
    assert success.ttft == pytest.approx(0.2)
    assert success.total == pytest.approx(0.8)
    assert success.output_characters == 12
    assert success.error is None
    assert success.finish_reason == "completed"
    assert failure.ttft is None
    assert failure.total == pytest.approx(0.5)
    assert failure.output_characters == 0
    assert failure.error == "RuntimeError: provider unavailable"
    assert failure.finish_reason == "error"


def test_summary_calculates_interpolated_percentiles() -> None:
    summary = BenchmarkSummary(
        results=tuple(
            BenchmarkResult(
                run_number=index,
                ttft=float(index),
                total=float(index * 2),
                output_characters=index * 10,
            )
            for index in range(1, 5)
        )
    )

    assert summary.ttft_p50 == pytest.approx(2.5)
    assert summary.ttft_p95 == pytest.approx(3.85)
    assert summary.total_p50 == pytest.approx(5.0)
    assert summary.total_p95 == pytest.approx(7.7)
    assert summary.average_output_characters == pytest.approx(25.0)


def test_summary_handles_no_successful_runs() -> None:
    summary = BenchmarkSummary(
        results=(
            BenchmarkResult(
                run_number=1,
                ttft=None,
                total=0.5,
                output_characters=0,
                error="Empty response",
            ),
        )
    )

    assert summary.successes == 0
    assert summary.errors == 1
    assert summary.ttft_p50 is None
    assert summary.ttft_p95 is None
    assert summary.total_p50 is None
    assert summary.total_p95 is None
    assert summary.average_output_characters is None


@pytest.mark.asyncio
async def test_benchmark_records_empty_response_as_error() -> None:
    session = Session(EmptyProvider())

    summary = await run_latency_benchmark(
        session,
        "question",
        1,
        clock=SequenceClock([1.0, 1.4]),
    )
    await session.close()

    result = summary.results[0]
    assert not result.succeeded
    assert result.ttft is None
    assert result.total == pytest.approx(0.4)
    assert result.error == "Empty response"
    assert result.finish_reason == "empty_response"


@pytest.mark.asyncio
async def test_benchmark_records_local_limit_as_success() -> None:
    provider = BenchmarkProvider()
    session = Session(
        provider,
        response_limits=ResponseLimits(
            max_lines=3,
            max_characters=5,
            max_complete_sentences=3,
        ),
    )

    summary = await run_latency_benchmark(
        session,
        "question",
        1,
        clock=SequenceClock([1.0, 1.1, 1.2]),
    )
    await session.close()

    result = summary.results[0]
    assert result.succeeded
    assert result.finish_reason == "local_limit"
    assert result.limit_reason == "max_characters"
    assert result.output_characters == 5
    assert summary.local_limits == 1


@pytest.mark.asyncio
async def test_benchmark_rejects_non_positive_run_count() -> None:
    provider = BenchmarkProvider()
    session = Session(provider)

    with pytest.raises(ValueError, match="runs must be at least 1"):
        await run_latency_benchmark(session, "question", 0)

    await session.close()
