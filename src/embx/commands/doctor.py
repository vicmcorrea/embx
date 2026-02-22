from __future__ import annotations

import asyncio
import sys
from typing import Any

import typer

from embx.commands.shared import check_ollama_endpoint, emit_json, fail, is_provider_configured


def register_doctor_command(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor(
        json_output: bool = typer.Option(False, "--json", help="Print as JSON"),
        only_configured: bool = typer.Option(
            False,
            "--only-configured",
            help="Show only providers that are configured",
        ),
        check_network: bool = typer.Option(
            False,
            "--check-network",
            help="Run lightweight network checks where possible",
        ),
        check_auth: bool = typer.Option(
            False,
            "--check-auth",
            help="Run provider connectivity/auth checks when possible.",
        ),
        timeout_seconds: int = typer.Option(
            3,
            "--timeout-seconds",
            min=1,
            help="Timeout for network/auth checks",
        ),
    ) -> None:
        from embx.config import resolve_config
        from embx.exceptions import ConfigurationError
        from embx.providers import available_provider_metadata
        from embx.providers.discovery import test_provider_connection

        try:
            cfg = resolve_config()
        except ConfigurationError as exc:
            fail(str(exc), code=2)

        rows: list[dict[str, Any]] = []
        for metadata in available_provider_metadata():
            provider_name = metadata["name"]
            configured = is_provider_configured(provider_name, cfg)
            if only_configured and not configured:
                continue

            network_status = "skipped"
            network_detail = ""
            if check_network and provider_name == "ollama":
                base_url = str(cfg.get("ollama_base_url", "http://localhost:11434"))
                network_status, network_detail = check_ollama_endpoint(base_url, timeout_seconds)

            auth_status = "skipped"
            auth_detail = ""
            if check_auth:
                if not configured:
                    auth_status = "skipped"
                    auth_detail = "provider not configured"
                else:
                    ok, message = asyncio.run(
                        test_provider_connection(
                            provider_name=provider_name,
                            config=cfg,
                            timeout_seconds=timeout_seconds,
                        )
                    )
                    auth_status = "ok" if ok else "error"
                    auth_detail = message

            rows.append(
                {
                    "provider": provider_name,
                    "configured": configured,
                    "required": metadata["requires"],
                    "network_status": network_status,
                    "network_detail": network_detail,
                    "auth_status": auth_status,
                    "auth_detail": auth_detail,
                }
            )

        if not rows:
            fail("No providers matched current filters.", code=2)

        if json_output:
            emit_json(rows)
            return

        from rich.console import Console
        from rich.table import Table

        table = Table(title="embx doctor")
        table.add_column("Provider")
        table.add_column("Configured")
        table.add_column("Required")
        table.add_column("Network")
        table.add_column("Network Detail")
        table.add_column("Auth")
        table.add_column("Auth Detail")
        for row in rows:
            table.add_row(
                str(row["provider"]),
                str(row["configured"]),
                str(row["required"]),
                str(row["network_status"]),
                str(row["network_detail"]),
                str(row["auth_status"]),
                str(row["auth_detail"]),
            )
        Console(no_color=not sys.stdout.isatty()).print(table)
