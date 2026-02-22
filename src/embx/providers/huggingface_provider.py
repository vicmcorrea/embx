from __future__ import annotations

import os
from typing import Any

import httpx

from embx.exceptions import ConfigurationError, ProviderError
from embx.models import EmbeddingResult
from embx.providers.base import EmbeddingProvider


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _is_vector(value: Any) -> bool:
    return isinstance(value, list) and all(_is_number(item) for item in value)


def _normalize_response(data: Any, expected_count: int) -> list[list[float]]:
    if expected_count == 1 and _is_vector(data):
        return [[float(item) for item in data]]

    if (
        isinstance(data, list)
        and len(data) == expected_count
        and all(_is_vector(item) for item in data)
    ):
        return [[float(v) for v in item] for item in data]

    if isinstance(data, dict):
        embeddings = data.get("embeddings") or data.get("data")
        if isinstance(embeddings, list):
            vectors: list[list[float]] = []
            for item in embeddings:
                if isinstance(item, dict) and _is_vector(item.get("embedding")):
                    vectors.append([float(v) for v in item["embedding"]])
                elif _is_vector(item):
                    vectors.append([float(v) for v in item])
            if len(vectors) == expected_count:
                return vectors

    raise ProviderError(
        "Unexpected HuggingFace embedding response shape. Use a sentence embedding model "
        "compatible with feature-extraction."
    )


class HuggingFaceProvider(EmbeddingProvider):
    name = "huggingface"
    default_model = "sentence-transformers/all-MiniLM-L6-v2"
    required_config_keys = ("huggingface_api_key",)

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict[str, Any],
    ) -> list[EmbeddingResult]:
        _ = dimensions
        api_key = (
            str(config.get("huggingface_api_key", ""))
            or os.getenv("HF_TOKEN", "")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
            or os.getenv("EMBX_HUGGINGFACE_API_KEY", "")
        )
        if not api_key:
            raise ConfigurationError(
                "Missing HuggingFace API key. Set EMBX_HUGGINGFACE_API_KEY or huggingface_api_key in config."
            )

        base_url = (
            str(config.get("huggingface_base_url", ""))
            or os.getenv("EMBX_HUGGINGFACE_BASE_URL", "")
            or "https://router.huggingface.co/hf-inference/models"
        ).rstrip("/")

        url = f"{base_url}/{model}/pipeline/feature-extraction"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": texts,
        }

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise ProviderError(f"HuggingFace request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"HuggingFace embeddings request failed ({response.status_code}): {response.text}"
            )

        vectors = _normalize_response(response.json(), expected_count=len(texts))
        out: list[EmbeddingResult] = []
        for text, vector in zip(texts, vectors, strict=True):
            out.append(
                EmbeddingResult(
                    text=text,
                    vector=vector,
                    provider=self.name,
                    model=model,
                    cached=False,
                )
            )
        return out
