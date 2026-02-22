from __future__ import annotations

from embx.exceptions import ValidationError
from embx.providers.base import EmbeddingProvider
from embx.providers.ollama_provider import OllamaProvider
from embx.providers.openai_provider import OpenAIProvider
from embx.providers.openrouter_provider import OpenRouterProvider
from embx.providers.voyage_provider import VoyageProvider


_PROVIDER_TYPES: dict[str, type[EmbeddingProvider]] = {
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "voyage": VoyageProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str) -> EmbeddingProvider:
    provider_type = _PROVIDER_TYPES.get(name)
    if provider_type is None:
        available = ", ".join(sorted(_PROVIDER_TYPES))
        raise ValidationError(f"Unknown provider '{name}'. Available providers: {available}")
    return provider_type()


def available_provider_metadata() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, provider_type in sorted(_PROVIDER_TYPES.items()):
        required = getattr(provider_type, "required_config_keys", ())
        rows.append(
            {
                "name": name,
                "default_model": provider_type.default_model,
                "requires": ", ".join(required) if required else "none",
            }
        )
    return rows
