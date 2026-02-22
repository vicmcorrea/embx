from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from embx.commands.shared import emit_json
from embx.providers import available_provider_metadata


def register_providers_command(app: typer.Typer) -> None:
    @app.command("providers")
    def providers(
        json_output: bool = typer.Option(False, "--json", help="Print as JSON"),
    ) -> None:
        rows = available_provider_metadata()

        if json_output:
            emit_json(rows)
            return

        console = Console(no_color=not sys.stdout.isatty())
        table = Table(title="Available providers")
        table.add_column("Provider")
        table.add_column("Default Model")
        table.add_column("Required Config")
        for row in rows:
            table.add_row(row["name"], row["default_model"], row["requires"])
        console.print(table)
