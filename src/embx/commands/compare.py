from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from embx.commands.shared import (
    collect_single_text,
    emit_csv,
    emit_json,
    emit_markdown,
    fail,
    is_provider_configured,
    parse_provider_list,
    safe_vector_preview,
)
from embx.config import resolve_config
from embx.engine import EmbeddingEngine
from embx.exceptions import ConfigurationError, ProviderError, ValidationError
from embx.ranking import apply_ranking, strip_private_fields, supported_rankings


async def _compare_provider(
    *,
    engine: EmbeddingEngine,
    input_text: str,
    provider_name: str,
    model: str | None,
    dimensions: int | None,
    use_cache: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        results = await engine.embed_texts(
            texts=[input_text],
            provider_name=provider_name,
            model=model,
            dimensions=dimensions,
            use_cache=use_cache,
        )
        result = results[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "provider": provider_name,
            "status": "ok",
            "model": result.model,
            "dimensions": len(result.vector),
            "cached": result.cached,
            "latency_ms": round(elapsed_ms, 3),
            "cost_usd": result.cost_usd,
            "input_tokens": result.input_tokens,
            "vector_preview": safe_vector_preview(result.vector, size=6),
            "quality_score": None,
            "error": None,
            "_vector": result.vector,
        }
    except (ValidationError, ConfigurationError, ProviderError, Exception) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "provider": provider_name,
            "status": "error",
            "model": model,
            "dimensions": None,
            "cached": False,
            "latency_ms": round(elapsed_ms, 3),
            "cost_usd": None,
            "input_tokens": None,
            "vector_preview": None,
            "quality_score": None,
            "error": str(exc),
        }


def register_compare_command(app: typer.Typer) -> None:
    @app.command("compare")
    def compare(
        text: str | None = typer.Argument(
            None,
            help="Text to embed. If omitted, stdin is used or prompt is shown.",
        ),
        providers: str | None = typer.Option(
            None,
            "--providers",
            help="Comma-separated providers. Defaults to all registered providers.",
        ),
        only_configured: bool = typer.Option(
            False,
            "--only-configured/--include-unconfigured",
            help="Skip providers with missing required credentials.",
        ),
        model: str | None = typer.Option(
            None,
            "--model",
            "-m",
            help="Model name to force for all providers.",
        ),
        dimensions: int | None = typer.Option(
            None, "--dimensions", min=1, help="Output dimensions"
        ),
        output_format: str = typer.Option("pretty", "--format", help="pretty, json, csv, or md"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Write result to file"),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call"),
        rank_by: str = typer.Option(
            "none",
            "--rank-by",
            help="Rank successful providers by none, latency, cost, or quality.",
        ),
        top: int | None = typer.Option(
            None,
            "--top",
            min=1,
            help="Limit number of successful providers shown. Requires rank mode.",
        ),
        include_errors: bool = typer.Option(
            True,
            "--include-errors/--hide-errors",
            help="Include failed provider rows in output.",
        ),
        continue_on_error: bool = typer.Option(
            True,
            "--continue-on-error/--fail-fast",
            help="Continue comparing other providers if one fails.",
        ),
        retries: int | None = typer.Option(None, "--retries", min=0, help="Retry attempts"),
        retry_backoff: float | None = typer.Option(
            None,
            "--retry-backoff",
            min=0.0,
            help="Initial retry backoff in seconds",
        ),
    ) -> None:
        if output_format not in {"pretty", "json", "csv", "md"}:
            fail("--format must be one of: pretty, json, csv, md", code=2)

        ranking_options = supported_rankings()
        if rank_by not in ranking_options:
            options_text = ", ".join(ranking_options)
            fail(f"--rank-by must be one of: {options_text}", code=2)
        if top is not None and rank_by == "none":
            fail("--top requires --rank-by latency, cost, or quality", code=2)

        input_text = collect_single_text(text)
        provider_names = parse_provider_list(providers)

        try:
            cfg = resolve_config(
                {
                    "default_model": model,
                    "retry_attempts": retries,
                    "retry_backoff_seconds": retry_backoff,
                }
            )
            engine = EmbeddingEngine(cfg)
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        if only_configured:
            provider_names = [
                provider_name
                for provider_name in provider_names
                if is_provider_configured(provider_name, cfg)
            ]
            if not provider_names:
                fail(
                    "No configured providers available. Add credentials or use --include-unconfigured.",
                    code=2,
                )

        rows: list[dict[str, Any]]
        if continue_on_error:

            async def _run_parallel() -> list[dict[str, Any]]:
                tasks = [
                    _compare_provider(
                        engine=engine,
                        input_text=input_text,
                        provider_name=provider_name,
                        model=model,
                        dimensions=dimensions,
                        use_cache=not no_cache,
                    )
                    for provider_name in provider_names
                ]
                return await asyncio.gather(*tasks)

            rows = asyncio.run(_run_parallel())
        else:
            rows = []
            for provider_name in provider_names:
                row = asyncio.run(
                    _compare_provider(
                        engine=engine,
                        input_text=input_text,
                        provider_name=provider_name,
                        model=model,
                        dimensions=dimensions,
                        use_cache=not no_cache,
                    )
                )
                rows.append(row)
                if row["status"] == "error":
                    fail(f"Provider '{provider_name}' failed: {row['error']}", code=2)

        success_count = sum(1 for row in rows if row["status"] == "ok")

        ranking_result = apply_ranking(rows, rank_by)
        successful_rows = list(ranking_result.successful_rows)
        error_rows = list(ranking_result.error_rows)

        if top is not None:
            successful_rows = successful_rows[:top]

        output_rows = list(successful_rows)
        if include_errors:
            output_rows.extend(error_rows)

        if top is None and rank_by == "none":
            output_rows = list(ranking_result.ranked_rows)
            if not include_errors:
                output_rows = [row for row in output_rows if row["status"] == "ok"]

        public_rows = strip_private_fields(output_rows)

        if output_format == "csv":
            emit_csv(public_rows, output)
        elif output_format == "json" or output is not None:
            emit_json(public_rows, output)
        elif output_format == "md":
            emit_markdown(public_rows, output)
        else:
            table = Table(title="Embedding comparison")
            table.add_column("Rank")
            table.add_column("Provider")
            table.add_column("Status")
            table.add_column("Model")
            table.add_column("Dim")
            table.add_column("Cached")
            table.add_column("Latency ms")
            table.add_column("Cost USD")
            table.add_column("Quality")
            table.add_column("Message")

            for row in public_rows:
                message = row["error"] or row["vector_preview"] or ""
                cost = "" if row["cost_usd"] is None else f"{row['cost_usd']:.8f}"
                quality = (
                    "" if row["quality_score"] is None else f"{float(row['quality_score']):.6f}"
                )
                table.add_row(
                    "" if row["rank"] is None else str(row["rank"]),
                    str(row["provider"]),
                    str(row["status"]),
                    str(row["model"] or ""),
                    "" if row["dimensions"] is None else str(row["dimensions"]),
                    str(row["cached"]),
                    f"{row['latency_ms']:.3f}",
                    cost,
                    quality,
                    str(message),
                )
            Console(no_color=not sys.stdout.isatty()).print(table)

            if rank_by != "none" and successful_rows:
                best = successful_rows[0]
                if rank_by == "cost":
                    if best["cost_usd"] is None:
                        detail = "cost unavailable"
                    else:
                        detail = f"${best['cost_usd']:.8f}"
                elif rank_by == "quality":
                    detail = f"score {float(best['quality_score']):.6f}"
                else:
                    detail = f"{best['latency_ms']:.3f} ms"
                typer.secho(
                    f"Best by {rank_by}: {best['provider']} ({detail})",
                    fg=typer.colors.GREEN,
                    err=True,
                )

        if success_count == 0:
            fail("All compared providers failed. See output for details.", code=2)
