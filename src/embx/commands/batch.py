from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from embx.commands.shared import emit_csv, fail


def register_batch_command(app: typer.Typer) -> None:
    @app.command("batch")
    def batch(
        input_file: Path = typer.Argument(
            ..., exists=True, dir_okay=False, help="Text file, one item per line"
        ),
        provider: str | None = typer.Option(None, "--provider", "-p", help="Embedding provider"),
        model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
        dimensions: int | None = typer.Option(
            None, "--dimensions", min=1, help="Output dimensions"
        ),
        output_format: str = typer.Option("jsonl", "--format", help="jsonl, json, or csv"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file"),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call"),
        retries: int | None = typer.Option(None, "--retries", min=0, help="Retry attempts"),
        retry_backoff: float | None = typer.Option(
            None,
            "--retry-backoff",
            min=0.0,
            help="Initial retry backoff in seconds",
        ),
    ) -> None:
        from embx.config import resolve_config
        from embx.engine import EmbeddingEngine
        from embx.exceptions import ConfigurationError, ProviderError, ValidationError

        if output_format not in {"json", "jsonl", "csv"}:
            fail("--format must be one of: jsonl, json, csv", code=2)

        try:
            lines = input_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            fail(f"Unable to read {input_file}: {exc}", code=2)

        texts = [line.strip() for line in lines if line.strip()]
        if not texts:
            fail("Input file has no non-empty lines.", code=2)

        try:
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
                    texts=texts,
                    provider_name=provider_name,
                    model=model,
                    dimensions=dimensions,
                    use_cache=not no_cache,
                )
            )
        except (ValidationError, ConfigurationError, ProviderError) as exc:
            fail(str(exc), code=2)

        rows = [item.to_dict() for item in results]
        if output_format == "json":
            payload = json.dumps(rows, indent=2)
        elif output_format == "csv":
            emit_csv(rows, output)
            return
        else:
            payload = "\n".join(json.dumps(row) for row in rows)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n", encoding="utf-8")
            typer.secho(
                f"Wrote {len(rows)} embeddings to {output}", fg=typer.colors.GREEN, err=True
            )
            return
        typer.echo(payload)
