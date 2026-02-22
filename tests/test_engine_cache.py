import asyncio
from pathlib import Path

from embx.cache import EmbeddingCache
from embx.engine import EmbeddingEngine
from embx.models import EmbeddingResult


class DummyProvider:
    default_model = "dummy-model"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def embed(
        self,
        texts: list[str],
        model: str,
        dimensions: int | None,
        timeout_seconds: int,
        config: dict,
    ) -> list[EmbeddingResult]:
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "dimensions": dimensions,
                "timeout_seconds": timeout_seconds,
                "config": dict(config),
            }
        )
        return [
            EmbeddingResult(
                text=text,
                vector=[float(len(text)), 1.0],
                provider="dummy",
                model=model,
                cached=False,
            )
            for text in texts
        ]


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    cache = EmbeddingCache(enabled=True, path=cache_path)

    cache.set(
        provider="dummy",
        model="m",
        dimensions=2,
        text="hello",
        vector=[0.1, 0.2],
    )

    assert cache.get("dummy", "m", 2, "hello") == [0.1, 0.2]
    assert cache.get("dummy", "m", 2, "missing") is None


def test_embedding_cache_disabled_is_noop(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.db"
    cache = EmbeddingCache(enabled=False, path=cache_path)

    cache.set(
        provider="dummy",
        model="m",
        dimensions=2,
        text="hello",
        vector=[0.1, 0.2],
    )

    assert cache.get("dummy", "m", 2, "hello") is None
    assert not cache_path.exists()


def test_engine_uses_cache_between_calls(monkeypatch, tmp_path: Path) -> None:
    provider = DummyProvider()
    monkeypatch.setattr("embx.engine.get_provider", lambda _: provider)
    monkeypatch.setenv("EMBX_CACHE_PATH", str(tmp_path / "cache.db"))

    engine = EmbeddingEngine({"cache_enabled": True, "timeout_seconds": 12})

    first = asyncio.run(
        engine.embed_texts(
            texts=["alpha", "beta"],
            provider_name="dummy",
            model="m",
            dimensions=2,
            use_cache=True,
        )
    )
    second = asyncio.run(
        engine.embed_texts(
            texts=["alpha", "beta"],
            provider_name="dummy",
            model="m",
            dimensions=2,
            use_cache=True,
        )
    )

    assert len(provider.calls) == 1
    assert all(not item.cached for item in first)
    assert all(item.cached for item in second)


def test_engine_can_bypass_cache(monkeypatch, tmp_path: Path) -> None:
    provider = DummyProvider()
    monkeypatch.setattr("embx.engine.get_provider", lambda _: provider)
    monkeypatch.setenv("EMBX_CACHE_PATH", str(tmp_path / "cache.db"))

    engine = EmbeddingEngine({"cache_enabled": True, "timeout_seconds": 12})

    asyncio.run(
        engine.embed_texts(
            texts=["alpha"],
            provider_name="dummy",
            model="m",
            dimensions=2,
            use_cache=True,
        )
    )
    asyncio.run(
        engine.embed_texts(
            texts=["alpha"],
            provider_name="dummy",
            model="m",
            dimensions=2,
            use_cache=False,
        )
    )

    assert len(provider.calls) == 2
