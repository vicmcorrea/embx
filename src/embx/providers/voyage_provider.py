from __future__ import annotations

import os
from typing import Any

import httpx

from embx.exceptions import ConfigurationError, ProviderError
from embx.models import EmbeddingResult
from embx.providers.base import EmbeddingProvider


class VoyageProvider(EmbeddingProvider):
    name = "voyage"
    default_model = "voyage-3-lite"
    required_config_keys = ("voyage_api_key",)

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict[str, Any],
    ) -> list[EmbeddingResult]:
        api_key = (
            str(config.get("voyage_api_key", ""))
            or os.getenv("VOYAGE_API_KEY", "")
            or os.getenv("EMBX_VOYAGE_API_KEY", "")
        )
        if not api_key:
            raise ConfigurationError(
                "Missing Voyage API key. Set EMBX_VOYAGE_API_KEY or voyage_api_key in config."
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        if dimensions is not None:
            payload["output_dimension"] = dimensions

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            try:
                response = await client.post(
                    "https://api.voyageai.com/v1/embeddings",
                    json=payload,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Voyage request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"Voyage embeddings request failed ({response.status_code}): {response.text}"
            )

        body = response.json()
        data = body.get("data", [])
        if len(data) != len(texts):
            raise ProviderError("Voyage response size does not match request size")

        out: list[EmbeddingResult] = []
        for text, row in zip(texts, data, strict=True):
            out.append(
                EmbeddingResult(
                    text=text,
                    vector=row["embedding"],
                    provider=self.name,
                    model=model,
                    cached=False,
                )
            )
        return out
