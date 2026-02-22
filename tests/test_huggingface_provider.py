import asyncio

from embx.exceptions import ConfigurationError
from embx.providers.huggingface_provider import HuggingFaceProvider


def test_huggingface_requires_api_key() -> None:
    provider = HuggingFaceProvider()

    try:
        asyncio.run(
            provider.embed(
                texts=["hello"],
                model="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=None,
                timeout_seconds=5,
                config={"huggingface_api_key": ""},
            )
        )
    except ConfigurationError as exc:
        assert "Missing HuggingFace API key" in str(exc)
    else:
        raise AssertionError("Expected ConfigurationError")


def test_huggingface_success_parses_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]

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

    monkeypatch.setattr("embx.providers.huggingface_provider.httpx.AsyncClient", FakeAsyncClient)

    provider = HuggingFaceProvider()
    results = asyncio.run(
        provider.embed(
            texts=["a", "b"],
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=None,
            timeout_seconds=9,
            config={"huggingface_api_key": "hf_test"},
        )
    )

    assert captured["url"].endswith(
        "/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    )
    assert captured["headers"]["Authorization"] == "Bearer hf_test"
    assert captured["json"]["inputs"] == ["a", "b"]

    assert len(results) == 2
    assert results[0].provider == "huggingface"
    assert len(results[0].vector) == 3
