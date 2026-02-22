import json
from pathlib import Path

from typer.testing import CliRunner

from embx.cli import app
from embx.models import EmbeddingResult


runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "embx" in result.stdout


def test_providers_json() -> None:
    result = runner.invoke(app, ["providers", "--json"])
    assert result.exit_code == 0
    assert "openai" in result.stdout
    assert "ollama" in result.stdout


def test_embed_without_input_fails() -> None:
    result = runner.invoke(app, ["embed"], input="")
    assert result.exit_code == 2
    assert "No input found" in result.output


def test_config_init_and_show() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.test.config.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        init_result = runner.invoke(app, ["config", "init", "--force"], env=env)
        assert init_result.exit_code == 0
        assert config_path.exists()

        show_result = runner.invoke(app, ["config", "show", "--json"], env=env)
        assert show_result.exit_code == 0
        assert "default_provider" in show_result.stdout


def test_compare_json_success(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (dimensions, use_cache)
        return [
            EmbeddingResult(
                text=texts[0],
                vector=[0.1, 0.2, 0.3],
                provider=provider_name,
                model=model or "mock-model",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.cli.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["compare", "hello world", "--providers", "openai,ollama", "--format", "json"],
    )
    assert result.exit_code == 0
    assert '"provider": "openai"' in result.stdout
    assert '"provider": "ollama"' in result.stdout
    assert '"status": "ok"' in result.stdout


def test_compare_empty_provider_list_fails() -> None:
    result = runner.invoke(app, ["compare", "hello", "--providers", " , , "])
    assert result.exit_code == 2
    assert "--providers must contain at least one provider" in result.output


def test_compare_rank_by_cost_orders_results(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        cost_map = {"openai": 0.12, "voyage": 0.03}
        return [
            EmbeddingResult(
                text="x",
                vector=[0.0, 1.0],
                provider=provider_name,
                model="mock",
                cached=False,
                cost_usd=cost_map.get(provider_name),
            )
        ]

    monkeypatch.setattr("embx.cli.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "openai,voyage",
            "--rank-by",
            "cost",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["provider"] == "voyage"
    assert payload[0]["rank"] == 1
    assert payload[1]["provider"] == "openai"
    assert payload[1]["rank"] == 2


def test_compare_invalid_rank_by_fails() -> None:
    result = runner.invoke(app, ["compare", "hello", "--rank-by", "speed"])
    assert result.exit_code == 2
    assert "--rank-by must be one of: none, latency, cost" in result.output


def test_compare_continue_on_error_keeps_success(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        if provider_name == "openai":
            raise RuntimeError("boom")
        return [
            EmbeddingResult(
                text="x",
                vector=[0.0, 1.0],
                provider=provider_name,
                model="mock",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.cli.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["compare", "hello", "--providers", "openai,voyage", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    statuses = {item["provider"]: item["status"] for item in payload}
    assert statuses["openai"] == "error"
    assert statuses["voyage"] == "ok"


def test_compare_fail_fast_stops_on_error(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, provider_name, model, dimensions, use_cache)
        raise RuntimeError("forced")

    monkeypatch.setattr("embx.cli.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["compare", "hello", "--providers", "openai,voyage", "--fail-fast"],
    )
    assert result.exit_code == 2
    assert "failed" in result.output
