"""User-facing interactive prompt with real turn preemption."""

import asyncio
from dataclasses import dataclass
from time import perf_counter

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from contextshift.dialogue import DialogueHistory
from contextshift.session import Session, TurnDelta, TurnResult


@dataclass(slots=True)
class ActiveTurn:
    """A submitted turn and the time at which it was superseded."""

    turn_id: int
    task: asyncio.Task[TurnResult]
    interrupted_at: float | None = None


async def run_interactive(
    session: Session,
    console: Console,
) -> None:
    """Read questions continuously while model turns run in the background."""

    prompt_session: PromptSession[str] = PromptSession()
    active_turn: ActiveTurn | None = None
    watchers: set[asyncio.Task[None]] = set()
    shutting_down = False
    history = DialogueHistory()

    console.print("[bold cyan]ContextShift interactive[/bold cyan]")
    console.print("输入问题后按回车。新问题会立即抢占旧问题。")
    console.print(
        "命令：[bold]/cancel[/bold] 取消当前回答，"
        "[bold]/history[/bold] 查看上下文，"
        "[bold]/clear[/bold] 清空上下文，"
        "[bold]/quit[/bold] 退出。"
    )
    console.print("上下文仅保存在当前进程内存；模型回答不会进入历史。")
    console.print("[yellow]不要输入真实密码、API Key 或其他敏感信息。[/yellow]\n")

    def render(delta: TurnDelta) -> None:
        console.print(
            delta.text,
            end="",
            markup=False,
            soft_wrap=True,
        )

    async def watch_turn(turn: ActiveTurn) -> None:
        nonlocal active_turn

        try:
            result = await turn.task
        except asyncio.CancelledError:
            if not shutting_down and turn.interrupted_at is not None:
                elapsed_ms = (perf_counter() - turn.interrupted_at) * 1000
                console.print(
                    f"\n[dim][Turn {turn.turn_id} interrupted "
                    f"in {elapsed_ms:.1f}ms][/dim]"
                )
        except Exception as exc:  # noqa: BLE001 - surface provider failures to user
            if not shutting_down:
                console.print(
                    f"\n[Turn {turn.turn_id} error] {type(exc).__name__}: {exc}",
                    style="red",
                    markup=False,
                )
        else:
            if not shutting_down and result.finish_reason == "interrupted":
                elapsed_ms = (
                    (perf_counter() - turn.interrupted_at) * 1000
                    if turn.interrupted_at is not None
                    else 0.0
                )
                console.print(
                    f"\n[dim][Turn {turn.turn_id} interrupted "
                    f"in {elapsed_ms:.1f}ms][/dim]"
                )
            elif not shutting_down and result.finish_reason == "local_limit":
                console.print(
                    f"\n[dim][Turn {turn.turn_id} stopped: {result.limit_reason}][/dim]"
                )
            elif not shutting_down:
                console.print(f"\n[dim][Turn {turn.turn_id} completed][/dim]")
        finally:
            if active_turn is turn:
                active_turn = None

    def start_watcher(turn: ActiveTurn) -> None:
        watcher = asyncio.create_task(
            watch_turn(turn),
            name=f"contextshift-watch-turn-{turn.turn_id}",
        )
        watchers.add(watcher)
        watcher.add_done_callback(watchers.discard)

    try:
        with patch_stdout(raw=True):
            while True:
                try:
                    question = (
                        await prompt_session.prompt_async("contextshift> ")
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not question:
                    continue

                if question == "/quit":
                    break

                if question == "/cancel":
                    if active_turn is None or active_turn.task.done():
                        console.print("[dim]没有正在生成的回答。[/dim]")
                        continue

                    active_turn.interrupted_at = perf_counter()
                    session.cancel()
                    continue

                if question == "/clear":
                    history.clear()
                    console.print("[dim]会话上下文已清空。[/dim]")
                    continue

                if question == "/history":
                    if not history.entries:
                        console.print("[dim]当前没有会话上下文。[/dim]")
                        continue

                    console.print(f"[dim]当前保留 {len(history)} 条真实输入：[/dim]")
                    for index, entry in enumerate(history.entries, start=1):
                        console.print(
                            f"{index}. {entry}",
                            markup=False,
                        )
                    continue

                if active_turn is not None and not active_turn.task.done():
                    active_turn.interrupted_at = perf_counter()

                model_input = history.build_model_input(question)
                turn_id, task = session.submit(model_input, render)
                history.append(question)
                active_turn = ActiveTurn(turn_id=turn_id, task=task)
                console.print(f"\n[bold cyan][Turn {turn_id}][/bold cyan]")
                start_watcher(active_turn)
    finally:
        shutting_down = True
        session.cancel()
        if watchers:
            await asyncio.gather(*tuple(watchers), return_exceptions=True)
