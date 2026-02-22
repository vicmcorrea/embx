from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import typer

from embx.commands.shared import emit_csv, emit_json, emit_markdown, fail


MODEL_SOURCES = ("remote", "local", "all")


def _select_provider_interactively(options: list[str]) -> str:
    typer.echo("Select provider for model discovery:")
    for idx, name in enumerate(options, start=1):
        typer.echo(f"  {idx}. {name}")

    raw = typer.prompt("Provider number").strip()
    try:
        index = int(raw)
    except ValueError:
        fail("Provider selection must be a number.", code=2)
    if index < 1 or index > len(options):
        fail("Provider selection is out of range.", code=2)
    return options[index - 1]


def _select_source_interactively() -> str:
    typer.echo("Choose model source:")
    typer.echo("  1. remote")
    typer.echo("  2. local")
    typer.echo("  3. all")
    raw = typer.prompt("Source number", default="1").strip()
    mapping = {"1": "remote", "2": "local", "3": "all"}
    selected = mapping.get(raw)
    if selected is None:
        fail("Source selection must be 1, 2, or 3.", code=2)
    return selected


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
                "source": str(row.get("source", "remote")),
                "local_path": str(row.get("local_path", "")),
                "context_length": context_length,
                "dimensions": dimensions,
            }
        )
    return rows


def _filter_rows(
    rows: list[dict[str, Any]], search: str | None, limit: int | None
) -> list[dict[str, Any]]:
    filtered = rows
    if search:
        pattern = search.lower()
        filtered = [
            row
            for row in filtered
            if pattern in str(row.get("id", "")).lower()
            or pattern in str(row.get("name", "")).lower()
        ]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def _choose_row_interactively(rows: list[dict[str, Any]]) -> dict[str, Any]:
    typer.echo("Select model:")
    for idx, row in enumerate(rows, start=1):
        typer.echo(f"  {idx}. {row['id']}")

    raw = typer.prompt("Model number").strip()
    try:
        index = int(raw)
    except ValueError:
        fail("Model selection must be a number.", code=2)
    if index < 1 or index > len(rows):
        fail("Model selection is out of range.", code=2)
    return rows[index - 1]


def register_models_command(app: typer.Typer) -> None:
    @app.command("models")
    def models(
        provider: str | None = typer.Option(
            None,
            "--provider",
            "-p",
            help="Provider name. Defaults to configured default_provider.",
        ),
        source: str | None = typer.Option(
            None,
            "--source",
            help="Model source: remote, local, or all. Defaults to remote.",
        ),
        search: str | None = typer.Option(None, "--search", help="Filter model IDs by substring."),
        limit: int | None = typer.Option(
            25, "--limit", min=1, help="Maximum number of rows to return."
        ),
        output_format: str = typer.Option("pretty", "--format", help="pretty, json, csv, or md"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file"),
        timeout_seconds: int = typer.Option(10, "--timeout-seconds", min=1, help="Request timeout"),
        choose: bool = typer.Option(
            False,
            "--choose",
            help="Interactively pick one model and output only its id.",
        ),
        pick: int | None = typer.Option(
            None,
            "--pick",
            min=1,
            help="Pick 1-based model index and output only its id.",
        ),
        interactive: bool = typer.Option(
            False,
            "--interactive",
            help="Prompt for provider/source/filter values interactively.",
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Fail if required values are missing instead of prompting.",
        ),
    ) -> None:
        from embx.config import resolve_config
        from embx.exceptions import ConfigurationError, ProviderError, ValidationError
        from embx.providers import available_provider_metadata
        from embx.providers.discovery import list_embedding_models

        if output_format not in {"pretty", "json", "csv", "md"}:
            fail("--format must be one of: pretty, json, csv, md", code=2)
        if interactive and non_interactive:
            fail("--interactive and --non-interactive cannot be used together.", code=2)
        if choose and pick is not None:
            fail("--choose and --pick cannot be used together.", code=2)
        if choose and non_interactive:
            fail("--choose cannot be used with --non-interactive. Use --pick instead.", code=2)

        try:
            cfg = resolve_config()
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        options = [row["name"] for row in available_provider_metadata()]

        provider_name: str
        if provider is not None:
            provider_name = provider.strip().lower()
        elif interactive:
            provider_name = _select_provider_interactively(options)
        else:
            provider_name = str(cfg.get("default_provider", "openai")).strip().lower()

        if provider_name not in options:
            fail(f"Unknown provider '{provider_name}'. Available: {', '.join(options)}", code=2)

        if source is None:
            if provider_name == "huggingface" and interactive:
                source_name = _select_source_interactively()
            else:
                source_name = "remote"
        else:
            source_name = source.strip().lower()

        if source_name not in MODEL_SOURCES:
            fail("--source must be one of: remote, local, all", code=2)

        if interactive and search is None:
            search = typer.prompt("Search text (optional)", default="").strip() or None

        try:
            raw_models = asyncio.run(
                list_embedding_models(
                    provider_name=provider_name,
                    config=cfg,
                    timeout_seconds=timeout_seconds,
                    source=source_name,
                )
            )
        except (ConfigurationError, ProviderError, ValidationError) as exc:
            fail(str(exc), code=2)

        rows = _normalize_model_rows(provider_name, raw_models)
        rows = _filter_rows(rows, search=search, limit=limit)

        if choose or pick is not None:
            if not rows:
                fail("No models available for selection.", code=2)

            selected: dict[str, Any]
            if choose:
                selected = _choose_row_interactively(rows)
            else:
                assert pick is not None
                if pick > len(rows):
                    fail("--pick is out of range for the filtered model list.", code=2)
                selected = rows[pick - 1]

            selected_id = str(selected["id"])
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(selected_id + "\n", encoding="utf-8")
                typer.secho(f"Wrote output to {output}", fg=typer.colors.GREEN, err=True)
                return
            typer.echo(selected_id)
            return

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

        table = Table(title=f"Embedding models ({provider_name}, {source_name})")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Source")
        table.add_column("Context")
        table.add_column("Dimensions")
        table.add_column("Local Path")
        for row in rows:
            table.add_row(
                str(row["id"]),
                str(row["name"]),
                str(row["source"]),
                "" if row["context_length"] is None else str(row["context_length"]),
                "" if row["dimensions"] is None else str(row["dimensions"]),
                str(row["local_path"]),
            )
        Console(no_color=not sys.stdout.isatty()).print(table)
