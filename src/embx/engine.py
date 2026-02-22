from __future__ import annotations

from embx.cache import EmbeddingCache
from embx.models import EmbeddingResult
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
            fetched = await provider.embed(
                texts=missing_texts,
                model=resolved_model,
                dimensions=dimensions,
                timeout_seconds=int(self.config.get("timeout_seconds", 30)),
                config=self.config,
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
