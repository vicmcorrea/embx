from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from embx.exceptions import ConfigurationError


DEFAULT_CONFIG: dict[str, Any] = {
    "default_provider": "openai",
    "default_model": "text-embedding-3-small",
    "timeout_seconds": 30,
    "retry_attempts": 0,
    "retry_backoff_seconds": 0.25,
    "cache_enabled": True,
    "openai_api_key": "",
    "openrouter_api_key": "",
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    "openrouter_referer": "",
    "openrouter_title": "",
    "voyage_api_key": "",
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "nomic-embed-text",
}


def _config_path() -> Path:
    override = os.getenv("EMBX_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "embx" / "config.json"


def load_file_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read config file at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Config file is invalid JSON at {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigurationError(f"Config file must be a JSON object: {path}")
    return parsed


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def load_env_config() -> dict[str, Any]:
    env_map: dict[str, tuple[str, type]] = {
        "default_provider": ("EMBX_PROVIDER", str),
        "default_model": ("EMBX_MODEL", str),
        "timeout_seconds": ("EMBX_TIMEOUT_SECONDS", int),
        "retry_attempts": ("EMBX_RETRY_ATTEMPTS", int),
        "retry_backoff_seconds": ("EMBX_RETRY_BACKOFF_SECONDS", float),
        "cache_enabled": ("EMBX_CACHE_ENABLED", bool),
        "openai_api_key": ("EMBX_OPENAI_API_KEY", str),
        "openrouter_api_key": ("EMBX_OPENROUTER_API_KEY", str),
        "openrouter_base_url": ("EMBX_OPENROUTER_BASE_URL", str),
        "openrouter_referer": ("EMBX_OPENROUTER_REFERER", str),
        "openrouter_title": ("EMBX_OPENROUTER_TITLE", str),
        "voyage_api_key": ("EMBX_VOYAGE_API_KEY", str),
        "ollama_base_url": ("EMBX_OLLAMA_BASE_URL", str),
        "ollama_model": ("EMBX_OLLAMA_MODEL", str),
    }

    out: dict[str, Any] = {}
    for key, (env_var, expected_type) in env_map.items():
        value = os.getenv(env_var)
        if value is None:
            continue
        if expected_type is bool:
            out[key] = _parse_bool(value)
        elif expected_type is int:
            try:
                out[key] = int(value)
            except ValueError as exc:
                raise ConfigurationError(f"{env_var} must be an integer") from exc
        elif expected_type is float:
            try:
                out[key] = float(value)
            except ValueError as exc:
                raise ConfigurationError(f"{env_var} must be a float") from exc
        else:
            out[key] = value
    return out


def resolve_config(cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config.update(load_file_config())
    config.update(load_env_config())
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                config[key] = value
    return config


def init_config(force: bool = False) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise ConfigurationError(f"Config already exists at {path}. Use --force to overwrite.")
    content = json.dumps(DEFAULT_CONFIG, indent=2) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def upsert_config(values: dict[str, Any]) -> Path:
    path = _config_path()

    current: dict[str, Any] = {}
    if path.exists():
        current = load_file_config()

    merged = DEFAULT_CONFIG.copy()
    merged.update(current)
    for key, value in values.items():
        if value is not None:
            merged[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return path


def masked_config(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(config)
    for key in list(out.keys()):
        if "key" in key.lower() or "token" in key.lower():
            value = str(out[key])
            out[key] = "" if not value else f"{value[:4]}..."
    return out
