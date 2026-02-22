from __future__ import annotations

import os
from typing import Any

import httpx

from embx.exceptions import ConfigurationError, ProviderError
from embx.models import EmbeddingResult
from embx.providers.base import EmbeddingProvider


_PRICE_PER_1M_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


class OpenAIProvider(EmbeddingProvider):
    name = "openai"
    default_model = "text-embedding-3-small"
    required_config_keys = ("openai_api_key",)

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict[str, Any],
    ) -> list[EmbeddingResult]:
        api_key = (
            str(config.get("openai_api_key", ""))
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("EMBX_OPENAI_API_KEY", "")
        )
        if not api_key:
            raise ConfigurationError(
                "Missing OpenAI API key. Set EMBX_OPENAI_API_KEY or openai_api_key in config."
            )

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

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                json=payload,
                headers=headers,
            )

        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI embeddings request failed ({response.status_code}): {response.text}"
            )

        body = response.json()
        data = body.get("data", [])
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")

        if len(data) != len(texts):
            raise ProviderError("OpenAI response size does not match request size")

        per_item_cost = None
        if isinstance(prompt_tokens, int):
            price = _PRICE_PER_1M_TOKENS.get(model)
            if price is not None and len(texts) > 0:
                per_item_tokens = prompt_tokens / len(texts)
                per_item_cost = (per_item_tokens / 1_000_000) * price

        out: list[EmbeddingResult] = []
        for text, row in zip(texts, data, strict=True):
            out.append(
                EmbeddingResult(
                    text=text,
                    vector=row["embedding"],
                    provider=self.name,
                    model=model,
                    input_tokens=prompt_tokens,
                    cost_usd=per_item_cost,
                    cached=False,
                )
            )
        return out
