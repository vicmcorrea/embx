from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import typer

from embx.commands.shared import emit_csv, emit_json, emit_markdown, fail


def register_ping_command(app: typer.Typer) -> None:
    @app.command("ping")
    def ping(
        provider: str | None = typer.Option(
            None,
            "--provider",
            "-p",
            help="Provider name. Defaults to configured default_provider.",
        ),
        model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
        text: str = typer.Option("ping", "--text", help="Input text used for ping request"),
        dimensions: int | None = typer.Option(
            None, "--dimensions", min=1, help="Output dimensions"
        ),
        output_format: str = typer.Option("pretty", "--format", help="pretty, json, csv, or md"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file"),
        no_cache: bool = typer.Option(
            True, "--no-cache/--use-cache", help="Disable cache by default"
        ),
        retries: int | None = typer.Option(None, "--retries", min=0, help="Retry attempts"),
        retry_backoff: float | None = typer.Option(
            None,
            "--retry-backoff",
            min=0.0,
            help="Initial retry backoff in seconds",
        ),
        timeout_seconds: int | None = typer.Option(
            None,
            "--timeout-seconds",
            min=1,
            help="Request timeout override",
        ),
    ) -> None:
        from embx.config import resolve_config
        from embx.engine import EmbeddingEngine
        from embx.exceptions import ConfigurationError, ProviderError, ValidationError

        if output_format not in {"pretty", "json", "csv", "md"}:
            fail("--format must be one of: pretty, json, csv, md", code=2)

        try:
            cfg = resolve_config(
                {
                    "default_provider": provider,
                    "default_model": model,
                    "retry_attempts": retries,
                    "retry_backoff_seconds": retry_backoff,
                    "timeout_seconds": timeout_seconds,
                }
            )
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        provider_name = provider or str(cfg.get("default_provider", "openai"))
        engine = EmbeddingEngine(cfg)

        started = time.perf_counter()
        try:
            result = asyncio.run(
                engine.embed_texts(
                    texts=[text],
                    provider_name=provider_name,
                    model=model,
                    dimensions=dimensions,
                    use_cache=not no_cache,
                )
            )[0]
        except (ValidationError, ConfigurationError, ProviderError) as exc:
            fail(f"Ping failed for provider '{provider_name}': {exc}", code=2)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        row = {
            "provider": provider_name,
            "model": result.model,
            "status": "ok",
            "latency_ms": elapsed_ms,
            "dimensions": len(result.vector),
            "cached": result.cached,
            "input_tokens": result.input_tokens,
            "cost_usd": result.cost_usd,
        }

        if output_format == "json" or output is not None:
            emit_json(row, output)
            return
        if output_format == "csv":
            emit_csv([row], output)
            return
        if output_format == "md":
            emit_markdown([row], output)
            return

        from rich.console import Console
        from rich.table import Table

        table = Table(title="embx ping")
        table.add_column("Field")
        table.add_column("Value")
        for key in [
            "provider",
            "model",
            "status",
            "latency_ms",
            "dimensions",
            "cached",
            "input_tokens",
            "cost_usd",
        ]:
            value = row[key]
            table.add_row(key, "" if value is None else str(value))
        Console(no_color=not sys.stdout.isatty()).print(table)
