from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from embx import __version__
from embx.config import init_config, masked_config, resolve_config
from embx.engine import EmbeddingEngine
from embx.exceptions import ConfigurationError, ProviderError, ValidationError
from embx.providers import available_provider_metadata


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


def _fail(message: str, code: int = 1) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _collect_single_text(maybe_text: str | None) -> str:
    if maybe_text:
        return maybe_text

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
        _fail("No input found on stdin.", code=2)

    return typer.prompt("Text to embed").strip()


def _safe_vector_preview(vector: list[float], size: int = 8) -> str:
    items = vector[:size]
    rendered = ", ".join(f"{value:.5f}" for value in items)
    suffix = " ..." if len(vector) > size else ""
    return f"[{rendered}{suffix}]"


def _emit_json(data: Any, output: Path | None = None) -> None:
    serialized = json.dumps(data, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        typer.secho(f"Wrote output to {output}", fg=typer.colors.GREEN, err=True)
        return
    typer.echo(serialized)


def _all_provider_names() -> list[str]:
    return [row["name"] for row in available_provider_metadata()]


def _parse_provider_list(raw: str | None) -> list[str]:
    if raw is None:
        return _all_provider_names()

    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        _fail("--providers must contain at least one provider name.", code=2)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


@app.command("providers")
def providers(
    json_output: bool = typer.Option(False, "--json", help="Print as JSON"),
) -> None:
    rows = available_provider_metadata()

    if json_output:
        _emit_json(rows)
        return

    console = Console(no_color=not sys.stdout.isatty())
    table = Table(title="Available providers")
    table.add_column("Provider")
    table.add_column("Default Model")
    table.add_column("Required Config")
    for row in rows:
        table.add_row(row["name"], row["default_model"], row["requires"])
    console.print(table)


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
) -> None:
    try:
        path = init_config(force=force)
        typer.secho(f"Config created at {path}", fg=typer.colors.GREEN)
    except ConfigurationError as exc:
        _fail(str(exc), code=2)


@config_app.command("show")
def config_show(
    json_output: bool = typer.Option(False, "--json", help="Print as JSON"),
) -> None:
    try:
        cfg = masked_config(resolve_config())
    except ConfigurationError as exc:
        _fail(str(exc), code=2)

    if json_output:
        _emit_json(cfg)
        return
    for key in sorted(cfg):
        typer.echo(f"{key}={cfg[key]}")


@app.command("embed")
def embed(
    text: str | None = typer.Argument(
        None,
        help="Text to embed. If omitted, stdin is used or prompt is shown.",
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Embedding provider"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
    dimensions: int | None = typer.Option(None, "--dimensions", min=1, help="Output dimensions"),
    output_format: str = typer.Option("pretty", "--format", help="pretty or json"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON result to file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call"),
) -> None:
    if output_format not in {"pretty", "json"}:
        _fail("--format must be one of: pretty, json", code=2)

    try:
        input_text = _collect_single_text(text)
        overrides = {
            "default_provider": provider,
            "default_model": model,
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
        _fail(str(exc), code=2)

    payload = result.to_dict()
    if output_format == "json" or output is not None:
        _emit_json(payload, output)
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
    table.add_row("vector_preview", _safe_vector_preview(result.vector))
    Console(no_color=not sys.stdout.isatty()).print(table)


@app.command("batch")
def batch(
    input_file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Text file, one item per line"
    ),
    provider: str | None = typer.Option(None, "--provider", "-p", help="Embedding provider"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
    dimensions: int | None = typer.Option(None, "--dimensions", min=1, help="Output dimensions"),
    output_format: str = typer.Option("jsonl", "--format", help="jsonl or json"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call"),
) -> None:
    if output_format not in {"json", "jsonl"}:
        _fail("--format must be one of: jsonl, json", code=2)

    try:
        lines = input_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _fail(f"Unable to read {input_file}: {exc}", code=2)

    texts = [line.strip() for line in lines if line.strip()]
    if not texts:
        _fail("Input file has no non-empty lines.", code=2)

    try:
        overrides = {
            "default_provider": provider,
            "default_model": model,
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
        _fail(str(exc), code=2)

    rows = [item.to_dict() for item in results]
    if output_format == "json":
        payload = json.dumps(rows, indent=2)
    else:
        payload = "\n".join(json.dumps(row) for row in rows)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        typer.secho(f"Wrote {len(rows)} embeddings to {output}", fg=typer.colors.GREEN, err=True)
        return
    typer.echo(payload)


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
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name to force for all providers.",
    ),
    dimensions: int | None = typer.Option(None, "--dimensions", min=1, help="Output dimensions"),
    output_format: str = typer.Option("pretty", "--format", help="pretty or json"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON result to file"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call"),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help="Continue comparing other providers if one fails.",
    ),
) -> None:
    if output_format not in {"pretty", "json"}:
        _fail("--format must be one of: pretty, json", code=2)

    input_text = _collect_single_text(text)
    provider_names = _parse_provider_list(providers)

    try:
        cfg = resolve_config({"default_model": model})
        engine = EmbeddingEngine(cfg)
    except ConfigurationError as exc:
        _fail(str(exc), code=2)

    rows: list[dict[str, Any]] = []
    success_count = 0

    for provider_name in provider_names:
        started = time.perf_counter()
        try:
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
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "provider": provider_name,
                    "status": "ok",
                    "model": result.model,
                    "dimensions": len(result.vector),
                    "cached": result.cached,
                    "latency_ms": round(elapsed_ms, 3),
                    "cost_usd": result.cost_usd,
                    "input_tokens": result.input_tokens,
                    "vector_preview": _safe_vector_preview(result.vector, size=6),
                    "error": None,
                }
            )
            success_count += 1
        except (ValidationError, ConfigurationError, ProviderError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "provider": provider_name,
                    "status": "error",
                    "model": model,
                    "dimensions": None,
                    "cached": False,
                    "latency_ms": round(elapsed_ms, 3),
                    "cost_usd": None,
                    "input_tokens": None,
                    "vector_preview": None,
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                _fail(f"Provider '{provider_name}' failed: {exc}", code=2)

    if output_format == "json" or output is not None:
        _emit_json(rows, output)
    else:
        table = Table(title="Embedding comparison")
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Model")
        table.add_column("Dim")
        table.add_column("Cached")
        table.add_column("Latency ms")
        table.add_column("Cost USD")
        table.add_column("Message")

        for row in rows:
            message = row["error"] or row["vector_preview"] or ""
            cost = "" if row["cost_usd"] is None else f"{row['cost_usd']:.8f}"
            table.add_row(
                str(row["provider"]),
                str(row["status"]),
                str(row["model"] or ""),
                "" if row["dimensions"] is None else str(row["dimensions"]),
                str(row["cached"]),
                f"{row['latency_ms']:.3f}",
                cost,
                str(message),
            )
        Console(no_color=not sys.stdout.isatty()).print(table)

    if success_count == 0:
        _fail("All compared providers failed. See output for details.", code=2)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        typer.secho("Cancelled.", fg=typer.colors.YELLOW, err=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
