from __future__ import annotations

from typing import Any

import typer

from embx.commands.shared import emit_json, fail
from embx.config import DEFAULT_CONFIG, init_config, masked_config, resolve_config, upsert_config
from embx.exceptions import ConfigurationError


def _coerce_value(raw_value: str, sample: Any) -> Any:
    if isinstance(sample, bool):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        fail("Invalid boolean value. Use true/false.", code=2)
    if isinstance(sample, int):
        try:
            return int(raw_value)
        except ValueError:
            fail("Invalid integer value.", code=2)
    if isinstance(sample, float):
        try:
            return float(raw_value)
        except ValueError:
            fail("Invalid float value.", code=2)
    return raw_value


def _interactive_key_choice(keys: list[str]) -> str:
    typer.echo("Select config key:")
    for idx, key in enumerate(keys, start=1):
        typer.echo(f"  {idx}. {key}")
    raw = typer.prompt("Key number").strip()
    try:
        index = int(raw)
    except ValueError:
        fail("Key selection must be a number.", code=2)
    if index < 1 or index > len(keys):
        fail("Key selection is out of range.", code=2)
    return keys[index - 1]


def register_config_commands(config_app: typer.Typer) -> None:
    @config_app.command("init")
    def config_init(
        force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    ) -> None:
        try:
            path = init_config(force=force)
            typer.secho(f"Config created at {path}", fg=typer.colors.GREEN)
        except ConfigurationError as exc:
            fail(str(exc), code=2)

    @config_app.command("show")
    def config_show(
        json_output: bool = typer.Option(False, "--json", help="Print as JSON"),
    ) -> None:
        try:
            cfg = masked_config(resolve_config())
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        if json_output:
            emit_json(cfg)
            return
        for key in sorted(cfg):
            typer.echo(f"{key}={cfg[key]}")

    @config_app.command("set")
    def config_set(
        key: str | None = typer.Option(None, "--key", help="Config key to set"),
        value: str | None = typer.Option(None, "--value", help="Config value to set"),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Fail instead of prompting for missing key/value.",
        ),
    ) -> None:
        try:
            cfg = resolve_config()
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        all_keys = sorted(DEFAULT_CONFIG.keys())

        if key is None:
            if non_interactive:
                fail("--key is required in non-interactive mode.", code=2)
            key_name = _interactive_key_choice(all_keys)
        else:
            key_name = key.strip()

        if key_name not in all_keys:
            fail(f"Unknown config key '{key_name}'.", code=2)

        if value is None:
            if non_interactive:
                fail("--value is required in non-interactive mode.", code=2)
            default_value = str(cfg.get(key_name, DEFAULT_CONFIG[key_name]))
            value = typer.prompt(f"Value for {key_name}", default=default_value)

        coerced = _coerce_value(value, DEFAULT_CONFIG[key_name])
        path = upsert_config({key_name: coerced})
        typer.secho(f"Updated {key_name} in {path}", fg=typer.colors.GREEN)
