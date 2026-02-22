from __future__ import annotations

import os
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


async def list_embedding_models(
    provider_name: str,
    config: dict[str, Any],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    if provider_name == "openrouter":
        return await _list_openrouter_models(config, timeout_seconds)
    if provider_name == "openai":
        return await _list_openai_models(config, timeout_seconds)
    if provider_name == "ollama":
        return await _list_ollama_models(config, timeout_seconds)
    if provider_name == "voyage":
        raise ValidationError("Model discovery for voyage is not available yet.")
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

        models = await list_embedding_models(provider_name, config, timeout_seconds)
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
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
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

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

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
    embedding_models = [row for row in data if "embedding" in str(row.get("id", ""))]
    return embedding_models


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


async def _test_voyage_embeddings(config: dict[str, Any], timeout_seconds: int) -> None:
    api_key = _voyage_key(config)
    if not api_key:
        raise ConfigurationError("Missing Voyage API key")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "voyage-3-lite",
        "input": ["ping"],
    }

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.post(
                "https://api.voyageai.com/v1/embeddings", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Voyage test request failed: {exc}") from exc

    if response.status_code >= 400:
        raise ProviderError(f"Voyage test request failed ({response.status_code}): {response.text}")
