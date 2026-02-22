import asyncio

from embx.exceptions import ConfigurationError
from embx.providers.discovery import list_embedding_models


def test_openrouter_model_discovery_requires_key() -> None:
    try:
        asyncio.run(
            list_embedding_models(
                provider_name="openrouter",
                config={"openrouter_api_key": ""},
                timeout_seconds=5,
            )
        )
    except ConfigurationError as exc:
        assert "Missing OpenRouter API key" in str(exc)
    else:
        raise AssertionError("Expected ConfigurationError")


def test_openrouter_model_discovery_success(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"id": "openai/text-embedding-3-small"}]}

        @property
        def text(self) -> str:
            return "ok"

    class FakeAsyncClient:
        def __init__(self, timeout: int):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        async def get(self, url: str, headers: dict):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("embx.providers.discovery.httpx.AsyncClient", FakeAsyncClient)

    rows = asyncio.run(
        list_embedding_models(
            provider_name="openrouter",
            config={"openrouter_api_key": "sk-openrouter"},
            timeout_seconds=7,
        )
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/embeddings/models"
    assert captured["headers"]["Authorization"] == "Bearer sk-openrouter"
    assert rows[0]["id"] == "openai/text-embedding-3-small"
