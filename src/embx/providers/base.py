from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from embx.models import EmbeddingResult


class EmbeddingProvider(ABC):
    name: str
    default_model: str
    required_config_keys: tuple[str, ...] = ()

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict[str, Any],
    ) -> list[EmbeddingResult]:
        raise NotImplementedError
