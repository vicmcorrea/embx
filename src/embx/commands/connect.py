from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from embx.commands.shared import fail


PROVIDER_ORDER = ["openai", "openrouter", "voyage", "ollama"]
PROVIDER_KEY_MAP = {
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
    "voyage": "voyage_api_key",
}


def _select_provider_interactively() -> str:
    typer.echo("Select provider:")
    for idx, name in enumerate(PROVIDER_ORDER, start=1):
        typer.echo(f"  {idx}. {name}")

    raw = typer.prompt("Provider number").strip()
    try:
        index = int(raw)
    except ValueError:
        fail("Provider selection must be a number.", code=2)

    if index < 1 or index > len(PROVIDER_ORDER):
        fail("Provider selection is out of range.", code=2)
    return PROVIDER_ORDER[index - 1]


def _collect_provider_updates(
    *,
    provider: str,
    cfg: dict[str, Any],
    api_key: str | None,
    base_url: str | None,
    referer: str | None,
    title: str | None,
    non_interactive: bool,
) -> dict[str, str]:
    updates: dict[str, str] = {}

    if provider in PROVIDER_KEY_MAP:
        key_field = PROVIDER_KEY_MAP[provider]
        resolved_key = api_key
        if resolved_key is None:
            if non_interactive:
                fail("--api-key is required in non-interactive mode for this provider.", code=2)
            resolved_key = typer.prompt(
                f"Paste {provider} API key",
                hide_input=True,
            ).strip()
        if not resolved_key:
            fail("API key cannot be empty.", code=2)
        updates[key_field] = resolved_key

    if provider == "openrouter":
        if base_url is None and not non_interactive:
            default_base = str(cfg.get("openrouter_base_url", "https://openrouter.ai/api/v1"))
            base_url = typer.prompt("OpenRouter base URL", default=default_base).strip()
        if base_url:
            updates["openrouter_base_url"] = base_url

        if referer is None and not non_interactive:
            referer = typer.prompt("OpenRouter HTTP-Referer (optional)", default="").strip()
        if title is None and not non_interactive:
            title = typer.prompt("OpenRouter X-Title (optional)", default="").strip()
        if referer is not None:
            updates["openrouter_referer"] = referer
        if title is not None:
            updates["openrouter_title"] = title

    if provider == "ollama":
        if base_url is None and not non_interactive:
            default_base = str(cfg.get("ollama_base_url", "http://localhost:11434"))
            base_url = typer.prompt("Ollama base URL", default=default_base).strip()
        if base_url:
            updates["ollama_base_url"] = base_url

    return updates


def _run_connect_test(
    provider_name: str, cfg: dict[str, Any], timeout_seconds: int
) -> tuple[bool, str]:
    from embx.providers.discovery import test_provider_connection

    return asyncio.run(
        test_provider_connection(
            provider_name=provider_name,
            config=cfg,
            timeout_seconds=timeout_seconds,
        )
    )


def register_connect_command(app: typer.Typer) -> None:
    @app.command("connect")
    def connect(
        provider: str | None = typer.Option(
            None,
            "--provider",
            "-p",
            help="Provider to configure. If omitted, starts interactive selection.",
        ),
        api_key: str | None = typer.Option(
            None,
            "--api-key",
            help="API key. If omitted, prompts securely for supported providers.",
        ),
        base_url: str | None = typer.Option(
            None,
            "--base-url",
            help="Base URL for Ollama/OpenRouter configuration.",
        ),
        referer: str | None = typer.Option(
            None,
            "--referer",
            help="OpenRouter HTTP-Referer header value.",
        ),
        title: str | None = typer.Option(
            None,
            "--title",
            help="OpenRouter X-Title header value.",
        ),
        set_default: bool = typer.Option(
            True,
            "--set-default/--keep-default",
            help="Set selected provider as default_provider.",
        ),
        connect_all: bool = typer.Option(
            False,
            "--all",
            help="Interactive wizard to configure multiple providers in one run.",
        ),
        test: bool = typer.Option(
            False,
            "--test",
            help="Test provider connectivity after saving configuration.",
        ),
        timeout_seconds: int = typer.Option(
            10,
            "--timeout-seconds",
            min=1,
            help="Timeout for --test checks.",
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Fail instead of prompting for missing values.",
        ),
    ) -> None:
        from embx.config import resolve_config, upsert_config

        cfg = resolve_config()
        updates: dict[str, str] = {}
        configured: list[str] = []

        if connect_all:
            if non_interactive:
                fail("--all cannot be used with --non-interactive.", code=2)
            if provider is not None:
                fail("--provider cannot be used with --all.", code=2)

            for option in PROVIDER_ORDER:
                should_configure = typer.confirm(f"Configure {option}?", default=False)
                if not should_configure:
                    continue

                option_updates = _collect_provider_updates(
                    provider=option,
                    cfg=cfg,
                    api_key=None,
                    base_url=None,
                    referer=None,
                    title=None,
                    non_interactive=False,
                )
                updates.update(option_updates)
                configured.append(option)

            if not configured:
                fail("No providers configured in wizard run.", code=2)
        else:
            if provider is None:
                if non_interactive:
                    fail("--provider is required in non-interactive mode.", code=2)
                provider = _select_provider_interactively()

            provider = provider.strip().lower()
            if provider not in PROVIDER_ORDER:
                available = ", ".join(PROVIDER_ORDER)
                fail(f"Unknown provider '{provider}'. Available: {available}", code=2)

            option_updates = _collect_provider_updates(
                provider=provider,
                cfg=cfg,
                api_key=api_key,
                base_url=base_url,
                referer=referer,
                title=title,
                non_interactive=non_interactive,
            )
            updates.update(option_updates)
            configured.append(provider)

        if set_default and configured:
            updates["default_provider"] = configured[0]

        path: Path = upsert_config(updates)
        typer.secho(f"Saved configuration to {path}", fg=typer.colors.GREEN)
        typer.echo(f"Providers configured: {', '.join(configured)}")

        if test:
            latest_cfg = resolve_config()
            failed: list[str] = []
            for name in configured:
                ok, message = _run_connect_test(name, latest_cfg, timeout_seconds)
                if ok:
                    typer.secho(f"[{name}] OK: {message}", fg=typer.colors.GREEN)
                else:
                    typer.secho(f"[{name}] FAIL: {message}", fg=typer.colors.RED, err=True)
                    failed.append(name)
            if failed:
                fail(f"Connectivity test failed for: {', '.join(failed)}", code=2)

        typer.echo("Run: embx doctor --only-configured")
