import asyncio
from pathlib import Path

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


def test_huggingface_local_model_discovery(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    model_dir = cache / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = model_dir / "snapshots" / "rev123"
    snapshot.mkdir(parents=True)
    refs_main = model_dir / "refs" / "main"
    refs_main.parent.mkdir(parents=True)
    refs_main.write_text("rev123\n", encoding="utf-8")

    rows = asyncio.run(
        list_embedding_models(
            provider_name="huggingface",
            config={"huggingface_cache_dir": str(cache)},
            timeout_seconds=5,
            source="local",
        )
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert rows[0]["source"] == "local"


def test_huggingface_remote_model_discovery(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return [
                {
                    "id": "sentence-transformers/all-MiniLM-L6-v2",
                    "pipeline_tag": "feature-extraction",
                }
            ]

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

        async def get(self, url: str, params: dict, headers: dict):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr("embx.providers.discovery.httpx.AsyncClient", FakeAsyncClient)

    rows = asyncio.run(
        list_embedding_models(
            provider_name="huggingface",
            config={"huggingface_api_key": "hf_test"},
            timeout_seconds=5,
            source="remote",
        )
    )

    assert captured["url"] == "https://huggingface.co/api/models"
    assert captured["params"]["pipeline_tag"] == "feature-extraction"
    assert captured["headers"]["Authorization"] == "Bearer hf_test"
    assert rows[0]["id"] == "sentence-transformers/all-MiniLM-L6-v2"
