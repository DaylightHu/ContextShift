import asyncio
from time import perf_counter

import typer
from pydantic import ValidationError
from rich.console import Console

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


async def run_smoke_test(prompt: str) -> None:
    try:
        settings = Settings()
    except ValidationError as exc:
        console.print("[red]配置读取失败。请检查 .env 中的 DEEPSEEK_API_KEY。[/red]")
        console.print(str(exc))
        raise typer.Exit(code=2) from exc

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
