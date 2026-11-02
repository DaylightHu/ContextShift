import asyncio
from time import perf_counter

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from contextshift.benchmark import (
    BenchmarkResult,
    BenchmarkSummary,
    run_latency_benchmark,
)
from contextshift.config import Settings
from contextshift.llm import DeepSeekProvider
from contextshift.session import Session, TurnDelta

app = typer.Typer(
    name="contextshift",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def main() -> None:
    """ContextShift command-line interface."""


def load_settings() -> Settings:
    """Load application settings or exit with a concise CLI error."""

    try:
        return Settings()
    except ValidationError as exc:
        console.print("[red]配置读取失败。请检查 .env 中的 DEEPSEEK_API_KEY。[/red]")
        console.print(str(exc))
        raise typer.Exit(code=2) from exc


async def run_smoke_test(prompt: str) -> None:
    settings = load_settings()
    provider = DeepSeekProvider(settings)
    session = Session(provider)

    started_at = perf_counter()
    first_token_at: float | None = None

    console.print(f"[dim]model={settings.model}[/dim]")
    console.print("[dim]streaming...[/dim]\n")

    def render(delta: TurnDelta) -> None:
        nonlocal first_token_at

        if first_token_at is None:
            first_token_at = perf_counter()

        console.print(
            delta.text,
            end="",
            markup=False,
            soft_wrap=True,
        )

    try:
        _, task = session.submit(prompt, render)
        await task
    finally:
        await session.close()

    finished_at = perf_counter()
    console.print("\n")

    if first_token_at is not None:
        console.print(f"[green]TTFT: {first_token_at - started_at:.3f}s[/green]")

    console.print(f"[green]Total: {finished_at - started_at:.3f}s[/green]")


async def run_benchmark_command(prompt: str, runs: int) -> None:
    """Benchmark sequential requests using one persistent provider client."""

    settings = load_settings()
    provider = DeepSeekProvider(settings)
    session = Session(provider)

    console.print(f"[dim]model={settings.model} runs={runs}[/dim]")
    console.print("[dim]reusing one client for all runs...[/dim]\n")

    def show_result(result: BenchmarkResult) -> None:
        if result.succeeded:
            console.print(
                f"Run {result.run_number}/{runs}: "
                f"TTFT={result.ttft:.3f}s "
                f"Total={result.total:.3f}s "
                f"Characters={result.output_characters}"
            )
            return

        console.print(
            f"Run {result.run_number}/{runs}: ERROR {result.error}",
            markup=False,
        )

    try:
        summary = await run_latency_benchmark(
            session,
            prompt,
            runs,
            on_result=show_result,
        )
    finally:
        await session.close()

    console.print()
    _show_benchmark_summary(summary)


def _show_benchmark_summary(summary: BenchmarkSummary) -> None:
    table = Table(title="Latency benchmark", show_header=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Runs", str(summary.runs))
    table.add_row("Success", str(summary.successes))
    table.add_row("Errors", str(summary.errors))
    table.add_row("TTFT P50", _format_seconds(summary.ttft_p50))
    table.add_row("TTFT P95", _format_seconds(summary.ttft_p95))
    table.add_row("Total P50", _format_seconds(summary.total_p50))
    table.add_row("Total P95", _format_seconds(summary.total_p95))
    table.add_row(
        "Average characters",
        (
            f"{summary.average_output_characters:.1f}"
            if summary.average_output_characters is not None
            else "n/a"
        ),
    )
    console.print(table)


def _format_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "n/a"


@app.command()
def smoke(
    prompt: str = typer.Argument("为什么高并发缓存通常需要读写锁？"),
) -> None:
    """测试 DeepSeek 流式输出和首 token 延迟。"""

    try:
        asyncio.run(run_smoke_test(prompt))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=130) from None


@app.command()
def benchmark(
    prompt: str = typer.Argument("用两行解释 Python 的 GIL"),
    runs: int = typer.Option(20, "--runs", min=1, max=100),
) -> None:
    """在同一进程中复用连接并统计 TTFT 与总延迟。"""

    try:
        asyncio.run(run_benchmark_command(prompt, runs))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=130) from None
