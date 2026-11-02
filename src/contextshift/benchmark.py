"""Latency benchmark orchestration and summary statistics."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, floor
from time import perf_counter
from typing import Literal

from contextshift.runtime import LimitReason
from contextshift.session import Session, TurnDelta

BenchmarkFinishReason = Literal[
    "completed",
    "local_limit",
    "empty_response",
    "error",
]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing and output metadata for one benchmark request."""

    run_number: int
    ttft: float | None
    total: float
    output_characters: int
    error: str | None = None
    finish_reason: BenchmarkFinishReason = "completed"
    limit_reason: LimitReason | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the request produced text without an error."""

        return self.error is None and self.ttft is not None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Aggregate latency statistics for a benchmark run."""

    results: tuple[BenchmarkResult, ...]

    @property
    def runs(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> int:
        return sum(result.succeeded for result in self.results)

    @property
    def errors(self) -> int:
        return self.runs - self.successes

    @property
    def local_limits(self) -> int:
        return sum(result.finish_reason == "local_limit" for result in self.results)

    @property
    def ttft_p50(self) -> float | None:
        return _percentile(self._successful_ttfts, 0.50)

    @property
    def ttft_p95(self) -> float | None:
        return _percentile(self._successful_ttfts, 0.95)

    @property
    def total_p50(self) -> float | None:
        return _percentile(self._successful_totals, 0.50)

    @property
    def total_p95(self) -> float | None:
        return _percentile(self._successful_totals, 0.95)

    @property
    def average_output_characters(self) -> float | None:
        lengths = [
            result.output_characters for result in self.results if result.succeeded
        ]
        if not lengths:
            return None
        return sum(lengths) / len(lengths)

    @property
    def _successful_ttfts(self) -> list[float]:
        return [
            result.ttft
            for result in self.results
            if result.succeeded and result.ttft is not None
        ]

    @property
    def _successful_totals(self) -> list[float]:
        return [result.total for result in self.results if result.succeeded]


ResultHandler = Callable[[BenchmarkResult], None]
Clock = Callable[[], float]


async def run_latency_benchmark(
    session: Session,
    prompt: str,
    runs: int,
    *,
    on_result: ResultHandler | None = None,
    clock: Clock = perf_counter,
) -> BenchmarkSummary:
    """Run sequential requests through one reusable streaming session."""

    if runs < 1:
        raise ValueError("runs must be at least 1")

    results: list[BenchmarkResult] = []

    for run_number in range(1, runs + 1):
        started_at = clock()
        first_token_at: float | None = None
        output_characters = 0
        error: str | None = None
        finish_reason: BenchmarkFinishReason = "completed"
        limit_reason: LimitReason | None = None

        def record_delta(delta: TurnDelta) -> None:
            nonlocal first_token_at, output_characters

            if first_token_at is None:
                first_token_at = clock()
            output_characters += len(delta.text)

        try:
            _, task = session.submit(prompt, record_delta)
            turn_result = await task
            if turn_result.finish_reason == "local_limit":
                finish_reason = "local_limit"
                limit_reason = turn_result.limit_reason
        except Exception as exc:  # noqa: BLE001 - each failed run must be recorded
            error = f"{type(exc).__name__}: {exc}"
            finish_reason = "error"

        finished_at = clock()
        if error is None and first_token_at is None:
            error = "Empty response"
            finish_reason = "empty_response"

        result = BenchmarkResult(
            run_number=run_number,
            ttft=(first_token_at - started_at if first_token_at is not None else None),
            total=finished_at - started_at,
            output_characters=output_characters,
            error=error,
            finish_reason=finish_reason,
            limit_reason=limit_reason,
        )
        results.append(result)

        if on_result is not None:
            on_result(result)

    return BenchmarkSummary(results=tuple(results))


def _percentile(values: list[float], percentile: float) -> float | None:
    """Calculate an interpolated percentile for a non-empty numeric sample."""

    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (position - lower_index)
