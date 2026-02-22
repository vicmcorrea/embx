from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import typer

from embx.commands.shared import (
    all_provider_names,
    collect_single_text,
    emit_csv,
    emit_json,
    fail,
    is_provider_configured,
    safe_vector_preview,
)


MODEL_SOURCES = ("remote", "local", "all")


def _select_provider_interactively(options: list[str]) -> str:
    typer.echo("Select provider:")
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


def _select_hf_source_interactively(default_source: str) -> str:
    source_map = {"remote": "1", "local": "2", "all": "3"}
    default_choice = source_map.get(default_source, "1")

    typer.echo("Choose HuggingFace model source:")
    typer.echo("  1. remote (models from HuggingFace API)")
    typer.echo("  2. local (models from local cache)")
    typer.echo("  3. all (combined remote and local)")

    raw = typer.prompt("Source number", default=default_choice).strip().lower()
    mapping = {"1": "remote", "2": "local", "3": "all"}
    selected = mapping.get(raw) or (raw if raw in MODEL_SOURCES else None)
    if selected is None:
        fail("Source selection must be remote, local, all, or 1/2/3.", code=2)
    return selected


def _normalize_model_rows(raw_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in raw_models:
        model_id = str(row.get("id") or row.get("name") or row.get("model") or "")
        if not model_id:
            continue
        rows.append(
            {
                "id": model_id,
                "name": str(row.get("name") or model_id),
                "source": str(row.get("source", "remote")),
            }
        )
    return rows


def _select_model_interactively(rows: list[dict[str, Any]]) -> str:
    typer.echo("Select model:")
    for idx, row in enumerate(rows, start=1):
        typer.echo(f"  {idx}. {row['id']} ({row['source']})")

    raw = typer.prompt("Model number").strip()
    try:
        index = int(raw)
    except ValueError:
        fail("Model selection must be a number.", code=2)
    if index < 1 or index > len(rows):
        fail("Model selection is out of range.", code=2)
    return str(rows[index - 1]["id"])


def register_quickstart_command(app: typer.Typer) -> None:
    @app.command("quickstart")
    def quickstart(
        text: str | None = typer.Argument(
            None,
            help="Text to embed. If omitted, stdin is used or prompt is shown.",
        ),
        provider: str | None = typer.Option(
            None,
            "--provider",
            "-p",
            help="Provider to use. If omitted, interactive selection is used.",
        ),
        model: str | None = typer.Option(
            None,
            "--model",
            "-m",
            help="Model ID to use. If omitted, models are discovered and selected.",
        ),
        source: str | None = typer.Option(
            None,
            "--source",
            help="Model source: remote, local, or all (HuggingFace only).",
        ),
        connect: bool = typer.Option(
            False,
            "--connect",
            help="Run guided connection/setup for the selected provider.",
        ),
        api_key: str | None = typer.Option(
            None,
            "--api-key",
            help="API key used when --connect is enabled for key-based providers.",
        ),
        base_url: str | None = typer.Option(
            None,
            "--base-url",
            help="Base URL override used during provider connect setup.",
        ),
        cache_dir: str | None = typer.Option(
            None,
            "--cache-dir",
            help="HuggingFace cache directory used during connect setup.",
        ),
        model_source: str | None = typer.Option(
            None,
            "--model-source",
            help="HuggingFace default model source to save (remote, local, all).",
        ),
        referer: str | None = typer.Option(
            None,
            "--referer",
            help="OpenRouter HTTP-Referer header value used during connect setup.",
        ),
        title: str | None = typer.Option(
            None,
            "--title",
            help="OpenRouter X-Title header value used during connect setup.",
        ),
        set_default: bool = typer.Option(
            True,
            "--set-default/--keep-default",
            help="Save selected provider/model as defaults.",
        ),
        dimensions: int | None = typer.Option(
            None,
            "--dimensions",
            min=1,
            help="Output dimensions override.",
        ),
        no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this call."),
        timeout_seconds: int = typer.Option(
            10,
            "--timeout-seconds",
            min=1,
            help="Timeout for model discovery.",
        ),
        output_format: str = typer.Option("pretty", "--format", help="pretty, json, or csv"),
        output: Path | None = typer.Option(None, "--output", "-o", help="Write result to file"),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Fail instead of prompting for missing values.",
        ),
    ) -> None:
        from embx.commands.connect import _collect_provider_updates
        from embx.config import resolve_config, upsert_config
        from embx.engine import EmbeddingEngine
        from embx.exceptions import ConfigurationError, ProviderError, ValidationError
        from embx.providers import get_provider
        from embx.providers.discovery import list_embedding_models

        if output_format not in {"pretty", "json", "csv"}:
            fail("--format must be one of: pretty, json, csv", code=2)

        cfg = resolve_config()
        providers = all_provider_names()

        if provider is None:
            if non_interactive:
                provider_name = str(cfg.get("default_provider", "openai")).strip().lower()
            else:
                provider_name = _select_provider_interactively(providers)
        else:
            provider_name = provider.strip().lower()

        if provider_name not in providers:
            fail(f"Unknown provider '{provider_name}'. Available: {', '.join(providers)}", code=2)

        configured = is_provider_configured(provider_name, cfg)
        should_connect = connect

        if configured and not non_interactive and not should_connect:
            should_connect = typer.confirm(
                f"{provider_name} is already configured. Reconfigure now?", default=False
            )

        if not configured and not should_connect:
            if non_interactive:
                fail(
                    f"Provider '{provider_name}' is not configured. Rerun with --connect and required credentials.",
                    code=2,
                )
            typer.echo(f"Provider '{provider_name}' is not configured yet.")
            should_connect = typer.confirm("Run guided connect now?", default=True)
            if not should_connect:
                fail("Cannot continue without provider setup.", code=2)

        if should_connect:
            updates = _collect_provider_updates(
                provider=provider_name,
                cfg=cfg,
                api_key=api_key,
                base_url=base_url,
                cache_dir=cache_dir,
                model_source=model_source,
                referer=referer,
                title=title,
                non_interactive=non_interactive,
            )
            if set_default:
                updates["default_provider"] = provider_name
            path = upsert_config(updates)
            typer.secho(f"Saved configuration to {path}", fg=typer.colors.GREEN, err=True)
            cfg = resolve_config()

        if source is None:
            if provider_name == "huggingface":
                default_source = str(cfg.get("huggingface_model_source", "remote")).strip().lower()
                if non_interactive:
                    source_name = default_source
                else:
                    source_name = _select_hf_source_interactively(default_source)
            else:
                source_name = "remote"
        else:
            source_name = source.strip().lower()

        if source_name not in MODEL_SOURCES:
            fail("--source must be one of: remote, local, all", code=2)
        if provider_name != "huggingface" and source_name != "remote":
            fail(f"provider '{provider_name}' supports source=remote only", code=2)

        selected_model = model
        if not selected_model:
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
                raw_models = []
                if not non_interactive:
                    typer.secho(
                        f"Model discovery warning ({provider_name}): {exc}",
                        fg=typer.colors.YELLOW,
                        err=True,
                    )

            rows = _normalize_model_rows(raw_models)
            if rows:
                if non_interactive:
                    selected_model = str(rows[0]["id"])
                else:
                    selected_model = _select_model_interactively(rows)
            else:
                selected_model = get_provider(provider_name).default_model
                if not non_interactive:
                    typer.echo(f"Using provider default model: {selected_model}")

        if set_default:
            upsert_config({"default_provider": provider_name, "default_model": selected_model})
            cfg = resolve_config()

        try:
            input_text = collect_single_text(text)
            engine = EmbeddingEngine(cfg)
            started = perf_counter()
            result = asyncio.run(
                engine.embed_texts(
                    texts=[input_text],
                    provider_name=provider_name,
                    model=selected_model,
                    dimensions=dimensions,
                    use_cache=not no_cache,
                )
            )[0]
            latency_ms = round((perf_counter() - started) * 1000, 3)
        except (ValidationError, ConfigurationError, ProviderError) as exc:
            fail(str(exc), code=2)

        payload = result.to_dict()
        payload["latency_ms"] = latency_ms

        if output_format == "csv":
            emit_csv([payload], output)
            return
        if output_format == "json" or output is not None:
            emit_json(payload, output)
            return

        from rich.console import Console
        from rich.table import Table

        table = Table(title="Quickstart embedding")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("provider", result.provider)
        table.add_row("model", result.model)
        table.add_row("dimensions", str(len(result.vector)))
        table.add_row("cached", str(result.cached))
        table.add_row("latency_ms", str(latency_ms))
        if result.input_tokens is not None:
            table.add_row("input_tokens", str(result.input_tokens))
        if result.cost_usd is not None:
            table.add_row("cost_usd", f"{result.cost_usd:.8f}")
        table.add_row("vector_preview", safe_vector_preview(result.vector))
        Console(no_color=not sys.stdout.isatty()).print(table)
