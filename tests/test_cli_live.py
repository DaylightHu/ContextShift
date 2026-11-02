"""Live CLI requires a separate, explicit system-audio input."""

from typer.testing import CliRunner

from contextshift import cli


def test_live_requires_system_device() -> None:
    result = CliRunner().invoke(cli.app, ["live"])
    assert result.exit_code == 2
    assert "--system-device" in result.output


def test_live_passes_both_devices_to_runner(monkeypatch) -> None:
    received = {}

    async def fake_run(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr(cli, "run_live_command", fake_run)
    result = CliRunner().invoke(
        cli.app, ["live", "--input-device", "0", "--system-device", "1"]
    )
    assert result.exit_code == 0, result.output
    assert received["input_device"] == 0
    assert received["system_device"] == 1
