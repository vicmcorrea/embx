from __future__ import annotations

import os
from typing import Any

import httpx

from embx.exceptions import ConfigurationError, ProviderError
from embx.models import EmbeddingResult
from embx.providers.base import EmbeddingProvider


class OpenRouterProvider(EmbeddingProvider):
    name = "openrouter"
    default_model = "openai/text-embedding-3-small"
    required_config_keys = ("openrouter_api_key",)

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict[str, Any],
    ) -> list[EmbeddingResult]:
        api_key = (
            str(config.get("openrouter_api_key", ""))
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("EMBX_OPENROUTER_API_KEY", "")
        )
        if not api_key:
            raise ConfigurationError(
                "Missing OpenRouter API key. Set EMBX_OPENROUTER_API_KEY or openrouter_api_key in config."
            )

        base_url = (
            str(config.get("openrouter_base_url", ""))
            or os.getenv("OPENROUTER_BASE_URL", "")
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")

        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = str(config.get("openrouter_referer", "") or os.getenv("OPENROUTER_REFERER", ""))
        title = str(config.get("openrouter_title", "") or os.getenv("OPENROUTER_TITLE", ""))
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            try:
                response = await client.post(
                    f"{base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"OpenRouter embeddings request failed ({response.status_code}): {response.text}"
            )

        body = response.json()
        data = body.get("data", [])
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        cost_usd = usage.get("cost")

        if len(data) != len(texts):
            raise ProviderError("OpenRouter response size does not match request size")

        out: list[EmbeddingResult] = []
        for text, row in zip(texts, data, strict=True):
            out.append(
                EmbeddingResult(
                    text=text,
                    vector=row["embedding"],
                    provider=self.name,
                    model=model,
                    input_tokens=int(prompt_tokens)
                    if isinstance(prompt_tokens, (int, float))
                    else None,
                    cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
                    cached=False,
                )
            )
        return out
