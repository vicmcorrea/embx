from __future__ import annotations

import os
from typing import Any

import httpx

from embx.exceptions import ProviderError
from embx.models import EmbeddingResult
from embx.providers.base import EmbeddingProvider


class OllamaProvider(EmbeddingProvider):
    name = "ollama"
    default_model = "nomic-embed-text"

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict[str, Any],
    ) -> list[EmbeddingResult]:
        _ = dimensions
        base_url = (
            str(config.get("ollama_base_url", ""))
            or os.getenv("OLLAMA_BASE_URL", "")
            or "http://localhost:11434"
        ).rstrip("/")

        out: list[EmbeddingResult] = []
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for text in texts:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                if response.status_code >= 400:
                    raise ProviderError(
                        f"Ollama embeddings request failed ({response.status_code}): {response.text}"
                    )
                body = response.json()
                vector = body.get("embedding")
                if not isinstance(vector, list):
                    raise ProviderError("Ollama response missing embedding vector")
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
