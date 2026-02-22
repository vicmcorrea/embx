from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx

from embx.exceptions import ConfigurationError, ProviderError, ValidationError


def _openai_key(config: dict[str, Any]) -> str:
    return (
        str(config.get("openai_api_key", ""))
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("EMBX_OPENAI_API_KEY", "")
    )


def _openrouter_key(config: dict[str, Any]) -> str:
    return (
        str(config.get("openrouter_api_key", ""))
        or os.getenv("OPENROUTER_API_KEY", "")
        or os.getenv("EMBX_OPENROUTER_API_KEY", "")
    )


def _voyage_key(config: dict[str, Any]) -> str:
    return (
        str(config.get("voyage_api_key", ""))
        or os.getenv("VOYAGE_API_KEY", "")
        or os.getenv("EMBX_VOYAGE_API_KEY", "")
    )


def _huggingface_key(config: dict[str, Any]) -> str:
    return (
        str(config.get("huggingface_api_key", ""))
        or os.getenv("HF_TOKEN", "")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        or os.getenv("EMBX_HUGGINGFACE_API_KEY", "")
    )


def _huggingface_cache_root(config: dict[str, Any]) -> Path:
    explicit = str(config.get("huggingface_cache_dir", "")).strip()
    if explicit:
        return Path(explicit).expanduser()

    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_id_from_cache_dir_name(dir_name: str) -> str | None:
    if not dir_name.startswith("models--"):
        return None
    raw = dir_name[len("models--") :]
    if not raw:
        return None
    return raw.replace("--", "/")


def _latest_snapshot_path(model_dir: Path) -> Path | None:
    refs_dir = model_dir / "refs"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists() or not snapshots_dir.is_dir():
        return None

    if refs_dir.exists() and refs_dir.is_dir():
        ref_main = refs_dir / "main"
        if ref_main.exists() and ref_main.is_file():
            try:
                revision = ref_main.read_text(encoding="utf-8").strip()
            except OSError:
                revision = ""
            if revision:
                candidate = snapshots_dir / revision
                if candidate.exists() and candidate.is_dir():
                    return candidate

    snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    return snapshots[0]


async def list_embedding_models(
    provider_name: str,
    config: dict[str, Any],
    timeout_seconds: int,
    source: str = "remote",
) -> list[dict[str, Any]]:
    if provider_name == "openrouter":
        if source != "remote":
            raise ValidationError("openrouter model discovery supports source=remote only")
        return await _list_openrouter_models(config, timeout_seconds)

    if provider_name == "openai":
        if source != "remote":
            raise ValidationError("openai model discovery supports source=remote only")
        return await _list_openai_models(config, timeout_seconds)

    if provider_name == "ollama":
        if source != "remote":
            raise ValidationError("ollama model discovery supports source=remote only")
        return await _list_ollama_models(config, timeout_seconds)

    if provider_name == "voyage":
        raise ValidationError("Model discovery for voyage is not available yet.")

    if provider_name == "huggingface":
        if source == "remote":
            return await _list_huggingface_remote_models(config, timeout_seconds)
        if source == "local":
            return _list_huggingface_local_models(config)
        if source == "all":
            remote = await _list_huggingface_remote_models(config, timeout_seconds)
            local = _list_huggingface_local_models(config)
            seen: set[str] = set()
            out: list[dict[str, Any]] = []
            for item in local + remote:
                identifier = str(item.get("id", ""))
                if not identifier or identifier in seen:
                    continue
                seen.add(identifier)
                out.append(item)
            return out
        raise ValidationError("source must be one of: remote, local, all")

    raise ValidationError(f"Unknown provider '{provider_name}'")


async def test_provider_connection(
    provider_name: str,
    config: dict[str, Any],
    timeout_seconds: int,
) -> tuple[bool, str]:
    try:
        if provider_name == "voyage":
            await _test_voyage_embeddings(config, timeout_seconds)
            return True, "Voyage embeddings request succeeded"

        models = await list_embedding_models(
            provider_name=provider_name,
            config=config,
            timeout_seconds=timeout_seconds,
            source="remote",
        )
        return True, f"Fetched {len(models)} models"
    except (ConfigurationError, ProviderError, ValidationError) as exc:
        return False, str(exc)


async def _list_openrouter_models(
    config: dict[str, Any], timeout_seconds: int
) -> list[dict[str, Any]]:
    api_key = _openrouter_key(config)
    if not api_key:
        raise ConfigurationError("Missing OpenRouter API key")

    base_url = str(config.get("openrouter_base_url", "https://openrouter.ai/api/v1")).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    referer = str(config.get("openrouter_referer", "") or os.getenv("OPENROUTER_REFERER", ""))
    title = str(config.get("openrouter_title", "") or os.getenv("OPENROUTER_TITLE", ""))
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.get(f"{base_url}/embeddings/models", headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter model discovery failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderError(
            f"OpenRouter model discovery failed ({response.status_code}): {response.text}"
        )
    body = response.json()
    return list(body.get("data", []))


async def _list_openai_models(config: dict[str, Any], timeout_seconds: int) -> list[dict[str, Any]]:
    api_key = _openai_key(config)
    if not api_key:
        raise ConfigurationError("Missing OpenAI API key")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.get("https://api.openai.com/v1/models", headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI model discovery failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderError(
            f"OpenAI model discovery failed ({response.status_code}): {response.text}"
        )
    body = response.json()
    data = list(body.get("data", []))
    return [row for row in data if "embedding" in str(row.get("id", ""))]


async def _list_ollama_models(config: dict[str, Any], timeout_seconds: int) -> list[dict[str, Any]]:
    base_url = (
        str(config.get("ollama_base_url", ""))
        or os.getenv("OLLAMA_BASE_URL", "")
        or "http://localhost:11434"
    ).rstrip("/")

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.get(f"{base_url}/api/tags")
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama model discovery failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderError(
            f"Ollama model discovery failed ({response.status_code}): {response.text}"
        )
    body = response.json()
    return list(body.get("models", []))


async def _list_huggingface_remote_models(
    config: dict[str, Any], timeout_seconds: int
) -> list[dict[str, Any]]:
    token = _huggingface_key(config)
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {
        "pipeline_tag": "feature-extraction",
        "sort": "downloads",
        "direction": "-1",
        "limit": "200",
    }

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.get(
                "https://huggingface.co/api/models", params=params, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"HuggingFace model discovery failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderError(
            f"HuggingFace model discovery failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    if not isinstance(data, list):
        raise ProviderError("Unexpected HuggingFace model discovery response")
    return data


def _list_huggingface_local_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    cache_root = _huggingface_cache_root(config)
    if not cache_root.exists() or not cache_root.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    for entry in cache_root.iterdir():
        if not entry.is_dir():
            continue
        repo_id = _repo_id_from_cache_dir_name(entry.name)
        if repo_id is None:
            continue
        snapshot = _latest_snapshot_path(entry)
        rows.append(
            {
                "id": repo_id,
                "name": repo_id,
                "source": "local",
                "local_path": str(snapshot or entry),
            }
        )

    rows.sort(key=lambda row: str(row["id"]))
    return rows


async def _test_voyage_embeddings(config: dict[str, Any], timeout_seconds: int) -> None:
    api_key = _voyage_key(config)
    if not api_key:
        raise ConfigurationError("Missing Voyage API key")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": "voyage-3-lite", "input": ["ping"]}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.post(
                "https://api.voyageai.com/v1/embeddings", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Voyage test request failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderError(f"Voyage test request failed ({response.status_code}): {response.text}")
