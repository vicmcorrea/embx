from __future__ import annotations

import typer

from embx import __version__
from embx.commands import register_all_commands


app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    help="Multi-provider embeddings CLI",
)
config_app = typer.Typer(help="Manage embx configuration")
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"embx {__version__}")
        raise typer.Exit()


@app.callback()
def _main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    _ = version


register_all_commands(app=app, config_app=config_app)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        typer.secho("Cancelled.", fg=typer.colors.YELLOW, err=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
