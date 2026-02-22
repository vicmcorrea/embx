from __future__ import annotations

from pathlib import Path

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
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            help="Fail instead of prompting for missing values.",
        ),
    ) -> None:
        from embx.config import resolve_config, upsert_config

        if provider is None:
            if non_interactive:
                fail("--provider is required in non-interactive mode.", code=2)
            provider = _select_provider_interactively()

        provider = provider.strip().lower()
        if provider not in PROVIDER_ORDER:
            available = ", ".join(PROVIDER_ORDER)
            fail(f"Unknown provider '{provider}'. Available: {available}", code=2)

        cfg = resolve_config()
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

        if set_default:
            updates["default_provider"] = provider

        path: Path = upsert_config(updates)
        typer.secho(f"Saved configuration to {path}", fg=typer.colors.GREEN)
        typer.echo(f"Provider configured: {provider}")
        typer.echo("Run: embx doctor --only-configured")
