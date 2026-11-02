import asyncio
from pathlib import Path
from time import perf_counter
from typing import Annotated

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
from contextshift.interactive import run_interactive
from contextshift.live import LiveAssistant
from contextshift.llm import DeepSeekProvider
from contextshift.session import Session, TurnDelta
from contextshift.speech import (
    AudioConfig,
    SenseVoiceTranscriber,
    SpeechConfigurationError,
    SpeechDependencyError,
    SpeechStarted,
    TranscriptReady,
    list_input_devices,
)
from contextshift.speech.listen import run_dual_audio_listener, run_microphone_listener

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
        turn_result = await task
    finally:
        await session.close()

    finished_at = perf_counter()
    console.print("\n")

    if first_token_at is not None:
        console.print(f"[green]TTFT: {first_token_at - started_at:.3f}s[/green]")

    console.print(f"[green]Total: {finished_at - started_at:.3f}s[/green]")

    if turn_result.finish_reason == "local_limit":
        console.print(
            "[dim]finish_reason=local_limit "
            f"limit_reason={turn_result.limit_reason}[/dim]"
        )


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
                f"Characters={result.output_characters} "
                f"Finish={result.finish_reason}"
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


async def run_interactive_command() -> None:
    """Run a persistent text session with real model preemption."""

    settings = load_settings()
    provider = DeepSeekProvider(settings)
    session = Session(provider)

    console.print(f"[dim]model={settings.model}[/dim]")

    try:
        await run_interactive(session, console)
    finally:
        await session.close()


async def run_live_command(
    *,
    asr_model: Path,
    vad_model: Path,
    input_device: int | None,
    system_device: int,
    block_ms: int,
    asr_device: str,
) -> None:
    """Use mic speech as context and system speech as model questions."""

    settings = load_settings()
    session = Session(DeepSeekProvider(settings))

    def render(delta: TurnDelta) -> None:
        console.print(delta.text, end="", markup=False, soft_wrap=True)

    def show_error(exc: BaseException) -> None:
        console.print(
            f"\n{type(exc).__name__}: {exc}",
            style="red",
            markup=False,
        )

    assistant = LiveAssistant(session, render, on_error=show_error)

    def speech_started(event: SpeechStarted) -> None:
        assistant.handle_speech_started(event)
        console.print(
            f"\n[{event.source}/{event.role} Speech {event.utterance_id} started; "
            f"{'old turn cancelled' if event.source == 'system' else 'context only'}]",
            style="dim",
            markup=False,
        )

    def transcript_ready(event: TranscriptReady) -> None:
        console.print(
            f"\n[{event.source}/{event.role} Speech {event.utterance_id}] {event.text}",
            markup=False,
        )
        console.print(f"[dim]ASR={event.asr_elapsed:.3f}s RTF={event.rtf:.3f}[/dim]")
        assistant.handle_transcript(event)

    console.print(f"[dim]model={settings.model}[/dim]")
    try:
        await run_dual_audio_listener(
            asr_model_dir=asr_model,
            vad_model_dir=vad_model,
            microphone_config=AudioConfig(block_ms=block_ms, device=input_device),
            system_config=AudioConfig(block_ms=block_ms, device=system_device),
            asr_device=asr_device,
            on_speech_started=speech_started,
            on_transcript=transcript_ready,
        )
    finally:
        await assistant.close()
        await session.close()


def _show_benchmark_summary(summary: BenchmarkSummary) -> None:
    table = Table(title="Latency benchmark", show_header=False)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Runs", str(summary.runs))
    table.add_row("Success", str(summary.successes))
    table.add_row("Errors", str(summary.errors))
    table.add_row("Local limits", str(summary.local_limits))
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


def _speech_error(exc: SpeechConfigurationError | SpeechDependencyError) -> None:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(code=2) from exc


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


@app.command()
def interactive() -> None:
    """连续输入问题，并用新问题实时抢占旧回答。"""

    try:
        asyncio.run(run_interactive_command())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        raise typer.Exit(code=130) from None


@app.command("audio-devices")
def audio_devices() -> None:
    """列出可用的麦克风与回环输入设备。"""

    try:
        devices = list_input_devices()
    except (SpeechConfigurationError, SpeechDependencyError) as exc:
        _speech_error(exc)
    if not devices:
        console.print("[yellow]没有检测到可用的输入设备。[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Index")
    table.add_column("Name")
    table.add_column("Inputs", justify="right")
    table.add_column("Default Hz", justify="right")
    for device in devices:
        table.add_row(
            str(device.index),
            device.name,
            str(device.max_input_channels),
            f"{device.default_sample_rate:.0f}",
        )
    console.print(table)


@app.command()
def transcribe(
    audio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    asr_model: Annotated[
        Path,
        typer.Option("--asr-model"),
    ] = Path("models/asr/SenseVoiceSmall"),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """用本地 SenseVoiceSmall 转写一个音频文件。"""

    try:
        text = SenseVoiceTranscriber(asr_model, device=device).transcribe(audio_file)
    except (SpeechConfigurationError, SpeechDependencyError) as exc:
        _speech_error(exc)
    console.print(text, markup=False)


@app.command()
def listen(
    asr_model: Annotated[
        Path,
        typer.Option("--asr-model"),
    ] = Path("models/asr/SenseVoiceSmall"),
    vad_model: Annotated[
        Path,
        typer.Option("--vad-model"),
    ] = Path("models/vad/fsmn-vad"),
    input_device: Annotated[
        int | None,
        typer.Option(min=0),
    ] = None,
    block_ms: Annotated[int, typer.Option(min=100, max=1000)] = 200,
    asr_device: Annotated[str, typer.Option("--asr-device")] = "cpu",
) -> None:
    """麦克风 → FSMN-VAD 切句 → SenseVoice 本地转写。"""

    try:
        asyncio.run(
            run_microphone_listener(
                asr_model_dir=asr_model,
                vad_model_dir=vad_model,
                audio_config=AudioConfig(block_ms=block_ms, device=input_device),
                asr_device=asr_device,
            )
        )
    except (SpeechConfigurationError, SpeechDependencyError) as exc:
        _speech_error(exc)
    except KeyboardInterrupt:
        console.print("\n[yellow]已停止监听。[/yellow]")


@app.command()
def live(
    system_device: Annotated[
        int,
        typer.Option("--system-device", min=0, help="承载会议声音的回环输入设备编号"),
    ],
    asr_model: Annotated[
        Path,
        typer.Option("--asr-model"),
    ] = Path("models/asr/SenseVoiceSmall"),
    vad_model: Annotated[
        Path,
        typer.Option("--vad-model"),
    ] = Path("models/vad/fsmn-vad"),
    input_device: Annotated[
        int | None,
        typer.Option(min=0),
    ] = None,
    block_ms: Annotated[int, typer.Option(min=100, max=1000)] = 200,
    asr_device: Annotated[str, typer.Option("--asr-device")] = "cpu",
) -> None:
    """麦克风记录候选人；系统回环音频触发提问并抢占旧回答。"""

    try:
        asyncio.run(
            run_live_command(
                asr_model=asr_model,
                vad_model=vad_model,
                input_device=input_device,
                system_device=system_device,
                block_ms=block_ms,
                asr_device=asr_device,
            )
        )
    except (SpeechConfigurationError, SpeechDependencyError) as exc:
        _speech_error(exc)
    except KeyboardInterrupt:
        console.print("\n[yellow]已停止实时助手。[/yellow]")
