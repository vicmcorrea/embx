from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import httpx
import typer


def fail(message: str, code: int = 1) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def collect_single_text(maybe_text: str | None) -> str:
    if maybe_text:
        return maybe_text

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
        fail("No input found on stdin.", code=2)

    return typer.prompt("Text to embed").strip()


def safe_vector_preview(vector: list[float], size: int = 8) -> str:
    items = vector[:size]
    rendered = ", ".join(f"{value:.5f}" for value in items)
    suffix = " ..." if len(vector) > size else ""
    return f"[{rendered}{suffix}]"


def emit_json(data: Any, output: Path | None = None) -> None:
    serialized = json.dumps(data, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        typer.secho(f"Wrote output to {output}", fg=typer.colors.GREEN, err=True)
        return
    typer.echo(serialized)


def emit_csv(rows: list[dict[str, Any]], output: Path | None = None) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    if not fieldnames:
        fieldnames = ["result"]
        rows = [{"result": ""}]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    payload = buffer.getvalue()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.secho(f"Wrote output to {output}", fg=typer.colors.GREEN, err=True)
        return
    typer.echo(payload, nl=False)


def emit_markdown(rows: list[dict[str, Any]], output: Path | None = None) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    if not fieldnames:
        fieldnames = ["result"]
        rows = [{"result": ""}]

    def _escape(value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join("---" for _ in fieldnames) + " |"
    lines = [header, divider]
    for row in rows:
        line = "| " + " | ".join(_escape(row.get(key, "")) for key in fieldnames) + " |"
        lines.append(line)

    payload = "\n".join(lines) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.secho(f"Wrote output to {output}", fg=typer.colors.GREEN, err=True)
        return
    typer.echo(payload, nl=False)


def all_provider_names() -> list[str]:
    from embx.providers import available_provider_metadata

    return [row["name"] for row in available_provider_metadata()]


def parse_provider_list(raw: str | None) -> list[str]:
    if raw is None:
        return all_provider_names()

    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        fail("--providers must contain at least one provider name.", code=2)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def is_provider_configured(provider_name: str, config: dict[str, Any]) -> bool:
    from embx.providers import get_provider

    provider = get_provider(provider_name)
    required_keys = getattr(provider, "required_config_keys", ())
    if not required_keys:
        return True
    return all(str(config.get(key, "")).strip() for key in required_keys)


def check_ollama_endpoint(base_url: str, timeout_seconds: int) -> tuple[str, str]:
    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=timeout_seconds,
        )
        if response.status_code < 400:
            return "ok", f"HTTP {response.status_code}"
        return "error", f"HTTP {response.status_code}"
    except Exception as exc:
        return "error", str(exc)
