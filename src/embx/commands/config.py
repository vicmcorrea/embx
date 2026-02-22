from __future__ import annotations

import typer

from embx.commands.shared import emit_json, fail
from embx.config import init_config, masked_config, resolve_config
from embx.exceptions import ConfigurationError


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
