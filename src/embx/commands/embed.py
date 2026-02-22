from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from embx.commands.shared import collect_single_text, emit_csv, emit_json, fail, safe_vector_preview
from embx.config import resolve_config
from embx.engine import EmbeddingEngine
from embx.exceptions import ConfigurationError, ProviderError, ValidationError


def register_embed_command(app: typer.Typer) -> None:
    @app.command("embed")
    def embed(
        text: str | None = typer.Argument(
            None,
            help="Text to embed. If omitted, stdin is used or prompt is shown.",
        ),
        provider: str | None = typer.Option(None, "--provider", "-p", help="Embedding provider"),
        model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
        dimensions: int | None = typer.Option(
            None, "--dimensions", min=1, help="Output dimensions"
        ),
        output_format: str = typer.Option("pretty", "--format", help="pretty, json, or csv"),
        output: Path | None = typer.Option(
            None, "--output", "-o", help="Write JSON result to file"
        ),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call"),
        retries: int | None = typer.Option(None, "--retries", min=0, help="Retry attempts"),
        retry_backoff: float | None = typer.Option(
            None,
            "--retry-backoff",
            min=0.0,
            help="Initial retry backoff in seconds",
        ),
    ) -> None:
        if output_format not in {"pretty", "json", "csv"}:
            fail("--format must be one of: pretty, json, csv", code=2)

        try:
            input_text = collect_single_text(text)
            overrides = {
                "default_provider": provider,
                "default_model": model,
                "retry_attempts": retries,
                "retry_backoff_seconds": retry_backoff,
            }
            cfg = resolve_config(overrides)
            provider_name = provider or str(cfg.get("default_provider"))
            engine = EmbeddingEngine(cfg)

            results = asyncio.run(
                engine.embed_texts(
                    texts=[input_text],
                    provider_name=provider_name,
                    model=model,
                    dimensions=dimensions,
                    use_cache=not no_cache,
                )
            )
            result = results[0]
        except (ValidationError, ConfigurationError, ProviderError) as exc:
            fail(str(exc), code=2)

        payload = result.to_dict()
        if output_format == "csv":
            emit_csv([payload], output)
            return
        if output_format == "json" or output is not None:
            emit_json(payload, output)
            return

        table = Table(title="Embedding")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("provider", result.provider)
        table.add_row("model", result.model)
        table.add_row("dimensions", str(len(result.vector)))
        table.add_row("cached", str(result.cached))
        if result.input_tokens is not None:
            table.add_row("input_tokens", str(result.input_tokens))
        if result.cost_usd is not None:
            table.add_row("cost_usd", f"{result.cost_usd:.8f}")
        table.add_row("vector_preview", safe_vector_preview(result.vector))
        Console(no_color=not sys.stdout.isatty()).print(table)
