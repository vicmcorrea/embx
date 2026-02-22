import json
from pathlib import Path

from typer.testing import CliRunner

from embx.cli import app


runner = CliRunner()


def test_config_set_bool_and_int_non_interactive() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.config.set.types.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        first = runner.invoke(
            app,
            ["config", "set", "--key", "cache_enabled", "--value", "false", "--non-interactive"],
            env=env,
        )
        assert first.exit_code == 0

        second = runner.invoke(
            app,
            ["config", "set", "--key", "retry_attempts", "--value", "3", "--non-interactive"],
            env=env,
        )
        assert second.exit_code == 0

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["cache_enabled"] is False
        assert data["retry_attempts"] == 3


def test_config_set_invalid_key_fails() -> None:
    result = runner.invoke(
        app,
        ["config", "set", "--key", "does_not_exist", "--value", "x", "--non-interactive"],
    )
    assert result.exit_code == 2
    assert "Unknown config key" in result.output
