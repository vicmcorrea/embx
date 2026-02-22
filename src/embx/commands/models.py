from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import typer

from embx.commands.shared import emit_csv, emit_json, emit_markdown, fail


def _normalize_model_rows(
    provider_name: str, raw_models: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in raw_models:
        model_id = str(row.get("id") or row.get("name") or row.get("model") or "")
        if not model_id:
            continue
        display_name = str(row.get("name") or model_id)
        context_length = (
            row.get("context_length") or row.get("max_input_tokens") or row.get("num_ctx")
        )
        dimensions = (
            row.get("embedding_dimension")
            or row.get("dimensions")
            or row.get("dim")
            or row.get("size")
        )
        rows.append(
            {
                "provider": provider_name,
                "id": model_id,
                "name": display_name,
                "context_length": context_length,
                "dimensions": dimensions,
            }
        )
    return rows


def register_models_command(app: typer.Typer) -> None:
    @app.command("models")
    def models(
        provider: str | None = typer.Option(
            None,
            "--provider",
            "-p",
            help="Provider name. Defaults to configured default_provider.",
        ),
        output_format: str = typer.Option("pretty", "--format", help="pretty, json, csv, or md"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file"),
        timeout_seconds: int = typer.Option(10, "--timeout-seconds", min=1, help="Request timeout"),
    ) -> None:
        from embx.config import resolve_config
        from embx.exceptions import ConfigurationError, ProviderError, ValidationError
        from embx.providers.discovery import list_embedding_models

        if output_format not in {"pretty", "json", "csv", "md"}:
            fail("--format must be one of: pretty, json, csv, md", code=2)

        try:
            cfg = resolve_config()
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        provider_name = (provider or str(cfg.get("default_provider", "openai"))).strip().lower()

        try:
            raw_models = asyncio.run(
                list_embedding_models(
                    provider_name=provider_name, config=cfg, timeout_seconds=timeout_seconds
                )
            )
        except (ConfigurationError, ProviderError, ValidationError) as exc:
            fail(str(exc), code=2)

        rows = _normalize_model_rows(provider_name, raw_models)
        if output_format == "json" or output is not None:
            emit_json(rows, output)
            return
        if output_format == "csv":
            emit_csv(rows, output)
            return
        if output_format == "md":
            emit_markdown(rows, output)
            return

        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Embedding models ({provider_name})")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Context")
        table.add_column("Dimensions")
        for row in rows:
            table.add_row(
                str(row["id"]),
                str(row["name"]),
                "" if row["context_length"] is None else str(row["context_length"]),
                "" if row["dimensions"] is None else str(row["dimensions"]),
            )
        Console(no_color=not sys.stdout.isatty()).print(table)
