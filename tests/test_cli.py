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
