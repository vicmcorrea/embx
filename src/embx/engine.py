from __future__ import annotations

import asyncio

from embx.cache import EmbeddingCache
from embx.exceptions import ConfigurationError, ProviderError
from embx.models import EmbeddingResult
from embx.providers.base import EmbeddingProvider
from embx.providers.registry import get_provider


class EmbeddingEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.cache = EmbeddingCache(enabled=bool(config.get("cache_enabled", True)))

    async def embed_texts(
        self,
        texts: list[str],
        provider_name: str,
        model: str | None = None,
        dimensions: int | None = None,
        use_cache: bool = True,
    ) -> list[EmbeddingResult]:
        provider = get_provider(provider_name)
        resolved_model = model or provider.default_model

        ordered: list[EmbeddingResult | None] = [None for _ in texts]
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, text in enumerate(texts):
            if use_cache:
                cached_vector = self.cache.get(provider_name, resolved_model, dimensions, text)
                if cached_vector is not None:
                    ordered[idx] = EmbeddingResult(
                        text=text,
                        vector=cached_vector,
                        provider=provider_name,
                        model=resolved_model,
                        cached=True,
                    )
                    continue

            missing_indices.append(idx)
            missing_texts.append(text)

        if missing_texts:
            fetched = await self._embed_with_retry(
                provider=provider,
                texts=missing_texts,
                model=resolved_model,
                dimensions=dimensions,
            )
            for idx, item in zip(missing_indices, fetched, strict=True):
                ordered[idx] = item
                if use_cache:
                    self.cache.set(
                        provider=provider_name,
                        model=resolved_model,
                        dimensions=dimensions,
                        text=item.text,
                        vector=item.vector,
                    )

        return [item for item in ordered if item is not None]

    async def _embed_with_retry(
        self,
        *,
        provider: EmbeddingProvider,
        texts: list[str],
        model: str,
        dimensions: int | None,
    ) -> list[EmbeddingResult]:
        retry_attempts = int(self.config.get("retry_attempts", 0))
        backoff_seconds = float(self.config.get("retry_backoff_seconds", 0.25))
        timeout_seconds = int(self.config.get("timeout_seconds", 30))

        attempt = 0
        while True:
            try:
                return await provider.embed(
                    texts=texts,
                    model=model,
                    dimensions=dimensions,
                    timeout_seconds=timeout_seconds,
                    config=self.config,
                )
            except ConfigurationError:
                raise
            except ProviderError:
                if attempt >= retry_attempts:
                    raise
                await asyncio.sleep(max(0.0, backoff_seconds) * (2**attempt))
                attempt += 1
            except Exception as exc:
                if attempt >= retry_attempts:
                    raise ProviderError(f"Provider request failed: {exc}") from exc
                await asyncio.sleep(max(0.0, backoff_seconds) * (2**attempt))
                attempt += 1
