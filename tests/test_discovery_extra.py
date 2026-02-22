import asyncio

from embx.providers import discovery


def test_huggingface_all_merges_local_and_remote(monkeypatch) -> None:
    async def fake_remote(config, timeout_seconds):
        _ = (config, timeout_seconds)
        return [{"id": "a/model"}, {"id": "b/model"}]

    def fake_local(config):
        _ = config
        return [{"id": "b/model", "source": "local"}, {"id": "c/model", "source": "local"}]

    monkeypatch.setattr(discovery, "_list_huggingface_remote_models", fake_remote)
    monkeypatch.setattr(discovery, "_list_huggingface_local_models", fake_local)

    rows = asyncio.run(
        discovery.list_embedding_models(
            provider_name="huggingface",
            config={},
            timeout_seconds=3,
            source="all",
        )
    )
    ids = [row["id"] for row in rows]
    assert ids == ["b/model", "c/model", "a/model"]


def test_test_provider_connection_voyage_branch(monkeypatch) -> None:
    async def fake_test(config, timeout_seconds):
        _ = (config, timeout_seconds)
        return None

    monkeypatch.setattr(discovery, "_test_voyage_embeddings", fake_test)

    ok, message = asyncio.run(
        discovery.test_provider_connection(
            provider_name="voyage",
            config={"voyage_api_key": "x"},
            timeout_seconds=2,
        )
    )
    assert ok is True
    assert "Voyage embeddings request succeeded" in message
