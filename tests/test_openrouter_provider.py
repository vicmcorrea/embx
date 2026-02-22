import asyncio

from embx.exceptions import ConfigurationError
from embx.providers.openrouter_provider import OpenRouterProvider


def test_openrouter_requires_api_key() -> None:
    provider = OpenRouterProvider()

    try:
        asyncio.run(
            provider.embed(
                texts=["hello"],
                model="openai/text-embedding-3-small",
                dimensions=None,
                timeout_seconds=5,
                config={"openrouter_api_key": ""},
            )
        )
    except ConfigurationError as exc:
        assert "Missing OpenRouter API key" in str(exc)
    else:
        raise AssertionError("Expected ConfigurationError")


def test_openrouter_success_parses_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 8, "total_tokens": 8, "cost": 0.0000016},
            }

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

        async def post(self, url: str, json: dict, headers: dict):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("embx.providers.openrouter_provider.httpx.AsyncClient", FakeAsyncClient)

    provider = OpenRouterProvider()
    results = asyncio.run(
        provider.embed(
            texts=["hello"],
            model="openai/text-embedding-3-small",
            dimensions=3,
            timeout_seconds=7,
            config={
                "openrouter_api_key": "sk-test",
                "openrouter_base_url": "https://openrouter.ai/api/v1",
                "openrouter_referer": "https://example.com",
                "openrouter_title": "embx",
            },
        )
    )

    assert captured["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert captured["json"]["dimensions"] == 3
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["HTTP-Referer"] == "https://example.com"
    assert captured["headers"]["X-Title"] == "embx"

    assert len(results) == 1
    assert results[0].provider == "openrouter"
    assert results[0].input_tokens == 8
    assert results[0].cost_usd == 0.0000016
