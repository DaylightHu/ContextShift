"""History limits are configurable and applied by both session entry points."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from contextshift import cli
from contextshift.config import Settings
from contextshift.dialogue import DialogueHistory


def test_history_settings_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTSHIFT_HISTORY_MAX_TURNS", "96")
    monkeypatch.setenv("CONTEXTSHIFT_HISTORY_MAX_CHARACTERS", "16000")
    settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-only")
    assert settings.history_max_turns == 96
    assert settings.history_max_characters == 16000


@pytest.mark.parametrize(
    "key", ["CONTEXTSHIFT_HISTORY_MAX_TURNS", "CONTEXTSHIFT_HISTORY_MAX_CHARACTERS"]
)
@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_history_settings_reject_invalid_limits(monkeypatch, key, value) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DEEPSEEK_API_KEY="test-only")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["live", "interactive"])
async def test_cli_applies_history_settings_to_both_modes(monkeypatch, mode) -> None:
    settings = Settings(
        _env_file=None,
        DEEPSEEK_API_KEY="test-only",
        CONTEXTSHIFT_HISTORY_MAX_TURNS=2,
        CONTEXTSHIFT_HISTORY_MAX_CHARACTERS=5,
    )
    histories = []
    closed = []

    class Provider:
        async def close(self):
            closed.append(True)

    class Assistant:
        def __init__(self, *args, history, **kwargs):
            histories.append(history)

        async def close(self):
            pass

    async def listener(**kwargs):
        pass

    async def interactive(*args, history):
        histories.append(history)

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda _: Provider())
    monkeypatch.setattr(cli, "LiveAssistant", Assistant)
    monkeypatch.setattr(cli, "run_dual_audio_listener", listener)
    monkeypatch.setattr(cli, "run_interactive", interactive)
    if mode == "live":
        await cli.run_live_command(
            asr_model=Path("unused"),
            vad_model=Path("unused"),
            input_device=1,
            system_device=0,
            block_ms=200,
            asr_device="cpu",
        )
    else:
        await cli.run_interactive_command()
    assert closed == [True]
    assert len(histories) == 1
    history: DialogueHistory = histories[0]
    history.append("evicted")
    history.append("abcdef", role="candidate")
    history.append("xy", role="interviewer")
    assert history.entries == ("abcdef", "xy")
    prompt = history.build_model_input("next", role="interviewer")
    assert "候选人：def\n2. 面试官：xy" in prompt
