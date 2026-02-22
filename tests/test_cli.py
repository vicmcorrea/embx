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
    assert "openrouter" in result.stdout
    assert "huggingface" in result.stdout
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


def test_connect_interactive_openai() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.connect.config.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        result = runner.invoke(app, ["connect"], input="1\nsk-openai\n", env=env)
        assert result.exit_code == 0
        assert config_path.exists()

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["openai_api_key"] == "sk-openai"
        assert data["default_provider"] == "openai"


def test_connect_non_interactive_requires_values() -> None:
    result = runner.invoke(app, ["connect", "--provider", "openai", "--non-interactive"])
    assert result.exit_code == 2
    assert "--api-key is required in non-interactive mode" in result.output


def test_connect_openrouter_flags() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.connect.config.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        result = runner.invoke(
            app,
            [
                "connect",
                "--provider",
                "openrouter",
                "--api-key",
                "sk-openrouter",
                "--referer",
                "https://example.com",
                "--title",
                "embx-app",
                "--non-interactive",
            ],
            env=env,
        )
        assert result.exit_code == 0

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["openrouter_api_key"] == "sk-openrouter"
        assert data["openrouter_referer"] == "https://example.com"
        assert data["openrouter_title"] == "embx-app"
        assert data["default_provider"] == "openrouter"


def test_connect_all_wizard_multiple_providers() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.connect.config.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        result = runner.invoke(
            app,
            ["connect", "--all"],
            input="y\nsk-openai\nn\nn\ny\nsk-voyage\nn\n",
            env=env,
        )
        assert result.exit_code == 0

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["openai_api_key"] == "sk-openai"
        assert data["voyage_api_key"] == "sk-voyage"
        assert data["default_provider"] == "openai"


def test_connect_test_flag_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "embx.commands.connect._run_connect_test",
        lambda provider_name, cfg, timeout_seconds: (True, "ok"),
    )

    with runner.isolated_filesystem():
        config_path = Path("embx.connect.config.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        result = runner.invoke(
            app,
            [
                "connect",
                "--provider",
                "openai",
                "--api-key",
                "sk-openai",
                "--non-interactive",
                "--test",
            ],
            env=env,
        )
        assert result.exit_code == 0


def test_connect_test_flag_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "embx.commands.connect._run_connect_test",
        lambda provider_name, cfg, timeout_seconds: (False, "bad key"),
    )

    with runner.isolated_filesystem():
        config_path = Path("embx.connect.config.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        result = runner.invoke(
            app,
            [
                "connect",
                "--provider",
                "openai",
                "--api-key",
                "sk-openai",
                "--non-interactive",
                "--test",
            ],
            env=env,
        )
        assert result.exit_code == 2
        assert "Connectivity test failed" in result.output


def test_models_json_output(monkeypatch) -> None:
    async def fake_list_embedding_models(
        provider_name: str,
        config: dict,
        timeout_seconds: int,
        source: str = "remote",
    ):
        _ = (provider_name, config, timeout_seconds, source)
        return [{"id": "openai/text-embedding-3-small", "name": "Text Embedding 3 Small"}]

    monkeypatch.setattr(
        "embx.providers.discovery.list_embedding_models",
        fake_list_embedding_models,
    )

    result = runner.invoke(app, ["models", "--provider", "openrouter", "--format", "json"])
    assert result.exit_code == 0
    assert "openai/text-embedding-3-small" in result.stdout


def test_models_interactive_huggingface_local(monkeypatch) -> None:
    async def fake_list_embedding_models(
        provider_name: str,
        config: dict,
        timeout_seconds: int,
        source: str = "remote",
    ):
        _ = (config, timeout_seconds)
        assert provider_name == "huggingface"
        assert source == "local"
        return [{"id": "sentence-transformers/all-MiniLM-L6-v2", "name": "all-MiniLM"}]

    monkeypatch.setattr(
        "embx.providers.discovery.list_embedding_models", fake_list_embedding_models
    )

    result = runner.invoke(
        app,
        ["models", "--interactive", "--format", "json"],
        input="1\n2\nmini\n",
    )
    assert result.exit_code == 0
    assert "sentence-transformers/all-MiniLM-L6-v2" in result.stdout


def test_models_pick_outputs_id(monkeypatch) -> None:
    async def fake_list_embedding_models(
        provider_name: str,
        config: dict,
        timeout_seconds: int,
        source: str = "remote",
    ):
        _ = (provider_name, config, timeout_seconds, source)
        return [
            {"id": "model/a", "name": "A"},
            {"id": "model/b", "name": "B"},
        ]

    monkeypatch.setattr(
        "embx.providers.discovery.list_embedding_models", fake_list_embedding_models
    )

    result = runner.invoke(app, ["models", "--provider", "openrouter", "--pick", "2"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "model/b"


def test_models_choose_interactive_outputs_id(monkeypatch) -> None:
    async def fake_list_embedding_models(
        provider_name: str,
        config: dict,
        timeout_seconds: int,
        source: str = "remote",
    ):
        _ = (provider_name, config, timeout_seconds, source)
        return [
            {"id": "model/a", "name": "A"},
            {"id": "model/b", "name": "B"},
        ]

    monkeypatch.setattr(
        "embx.providers.discovery.list_embedding_models", fake_list_embedding_models
    )

    result = runner.invoke(app, ["models", "--provider", "openrouter", "--choose"], input="1\n")
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("model/a")


def test_models_choose_non_interactive_fails() -> None:
    result = runner.invoke(
        app,
        ["models", "--provider", "openrouter", "--choose", "--non-interactive"],
    )
    assert result.exit_code == 2
    assert "--choose cannot be used with --non-interactive" in result.output


def test_config_set_non_interactive() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.config.set.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        result = runner.invoke(
            app,
            [
                "config",
                "set",
                "--key",
                "default_provider",
                "--value",
                "huggingface",
                "--non-interactive",
            ],
            env=env,
        )
        assert result.exit_code == 0
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["default_provider"] == "huggingface"


def test_config_set_interactive() -> None:
    with runner.isolated_filesystem():
        config_path = Path("embx.config.set.json")
        env = {"EMBX_CONFIG_PATH": str(config_path)}

        init_result = runner.invoke(app, ["config", "init", "--force"], env=env)
        assert init_result.exit_code == 0

        result = runner.invoke(
            app,
            ["config", "set", "--key", "default_provider"],
            input="openrouter\n",
            env=env,
        )
        assert result.exit_code == 0
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["default_provider"] == "openrouter"


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

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

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

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

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
    assert "--rank-by must be one of: none, latency, cost, quality" in result.output


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

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

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

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["compare", "hello", "--providers", "openai,voyage", "--fail-fast"],
    )
    assert result.exit_code == 2
    assert "failed" in result.output


def test_compare_csv_output(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        return [
            EmbeddingResult(
                text="x",
                vector=[0.0, 1.0],
                provider=provider_name,
                model="mock",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["compare", "hello", "--providers", "openai", "--format", "csv"],
    )
    assert result.exit_code == 0
    assert "provider,status" in result.stdout
    assert "openai,ok" in result.stdout


def test_compare_markdown_output(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        return [
            EmbeddingResult(
                text="x",
                vector=[0.0, 1.0],
                provider=provider_name,
                model="mock",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["compare", "hello", "--providers", "openai", "--format", "md"],
    )
    assert result.exit_code == 0
    assert "| provider |" in result.stdout
    assert "| openai |" in result.stdout


def test_embed_retry_flags_reach_engine_config(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, provider_name, model, dimensions, use_cache)
        assert self.config["retry_attempts"] == 2
        assert self.config["retry_backoff_seconds"] == 0.1
        return [
            EmbeddingResult(
                text="x",
                vector=[0.0, 1.0],
                provider="mock",
                model="mock",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        ["embed", "hello", "--format", "json", "--retries", "2", "--retry-backoff", "0.1"],
    )
    assert result.exit_code == 0


def test_batch_csv_output(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (provider_name, model, dimensions, use_cache)
        return [
            EmbeddingResult(
                text=value,
                vector=[0.1, 0.2],
                provider="mock",
                model="mock-model",
                cached=False,
            )
            for value in texts
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    with runner.isolated_filesystem():
        input_path = Path("inputs.txt")
        input_path.write_text("first\nsecond\n", encoding="utf-8")
        result = runner.invoke(app, ["batch", str(input_path), "--format", "csv"])

    assert result.exit_code == 0
    assert "text,vector,provider,model" in result.stdout
    assert "first" in result.stdout
    assert "second" in result.stdout


def test_embed_csv_output(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (provider_name, model, dimensions, use_cache)
        return [
            EmbeddingResult(
                text=texts[0],
                vector=[0.5, 0.25],
                provider="mock",
                model="mock-model",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(app, ["embed", "hello", "--format", "csv"])
    assert result.exit_code == 0
    assert "text,vector,provider,model" in result.stdout
    assert "hello" in result.stdout


def test_compare_rank_by_quality_orders_results(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        vectors = {
            "openai": [1.0, 0.0, 0.0],
            "voyage": [0.9, 0.1, 0.0],
            "ollama": [-1.0, 0.0, 0.0],
        }
        return [
            EmbeddingResult(
                text="x",
                vector=vectors[provider_name],
                provider=provider_name,
                model="mock-model",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "openai,voyage,ollama",
            "--rank-by",
            "quality",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload[0]["provider"] in {"openai", "voyage"}
    assert payload[-1]["provider"] == "ollama"
    assert payload[0]["quality_score"] is not None


def test_compare_only_configured_filters_missing_keys(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        return [
            EmbeddingResult(
                text="x",
                vector=[0.1, 0.2],
                provider=provider_name,
                model="mock-model",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "openai,ollama",
            "--only-configured",
            "--format",
            "json",
        ],
        env={
            "EMBX_OPENAI_API_KEY": "",
            "EMBX_VOYAGE_API_KEY": "",
        },
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    providers = [item["provider"] for item in payload]
    assert providers == ["ollama"]


def test_compare_only_configured_with_none_fails() -> None:
    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "openai,voyage",
            "--only-configured",
        ],
        env={
            "EMBX_OPENAI_API_KEY": "",
            "EMBX_VOYAGE_API_KEY": "",
        },
    )
    assert result.exit_code == 2
    assert "No configured providers available" in result.output


def test_compare_top_requires_rank_mode() -> None:
    result = runner.invoke(app, ["compare", "hello", "--top", "1"])
    assert result.exit_code == 2
    assert "--top requires --rank-by latency, cost, or quality" in result.output


def test_compare_top_limits_successful_results(monkeypatch) -> None:
    async def fake_embed_texts(
        self,
        texts,
        provider_name,
        model=None,
        dimensions=None,
        use_cache=True,
    ):
        _ = (texts, model, dimensions, use_cache)
        latency_vectors = {
            "openai": [0.1, 0.2],
            "voyage": [0.2, 0.3],
            "ollama": [0.3, 0.4],
        }
        return [
            EmbeddingResult(
                text="x",
                vector=latency_vectors[provider_name],
                provider=provider_name,
                model="mock-model",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "openai,voyage,ollama",
            "--rank-by",
            "latency",
            "--top",
            "2",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload) == 2
    assert payload[0]["rank"] == 1
    assert payload[1]["rank"] == 2


def test_compare_hide_errors_excludes_failed_rows(monkeypatch) -> None:
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
            raise RuntimeError("forced")
        return [
            EmbeddingResult(
                text="x",
                vector=[0.1, 0.2],
                provider=provider_name,
                model="mock-model",
                cached=False,
            )
        ]

    monkeypatch.setattr("embx.engine.EmbeddingEngine.embed_texts", fake_embed_texts)

    result = runner.invoke(
        app,
        [
            "compare",
            "hello",
            "--providers",
            "openai,voyage",
            "--hide-errors",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["provider"] == "voyage"
    assert payload[0]["status"] == "ok"


def test_doctor_json_lists_providers() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    providers = {item["provider"] for item in payload}
    assert {"openai", "openrouter", "huggingface", "voyage", "ollama"}.issubset(providers)


def test_doctor_only_configured_filters_results() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--json", "--only-configured"],
        env={
            "EMBX_OPENAI_API_KEY": "",
            "EMBX_VOYAGE_API_KEY": "",
        },
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    providers = [item["provider"] for item in payload]
    assert providers == ["ollama"]


def test_doctor_network_check_for_ollama(monkeypatch) -> None:
    class Response:
        status_code = 200

    def fake_get(url: str, timeout: int):
        _ = timeout
        assert url.endswith("/api/tags")
        return Response()

    monkeypatch.setattr("embx.commands.shared.httpx.get", fake_get)

    result = runner.invoke(
        app,
        ["doctor", "--json", "--only-configured", "--check-network"],
        env={
            "EMBX_OPENAI_API_KEY": "",
            "EMBX_VOYAGE_API_KEY": "",
            "EMBX_OLLAMA_BASE_URL": "http://localhost:11434",
        },
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["network_status"] == "ok"
    assert payload[0]["network_detail"] == "HTTP 200"


def test_doctor_check_auth_uses_discovery_probe(monkeypatch) -> None:
    async def fake_test_provider_connection(provider_name: str, config: dict, timeout_seconds: int):
        _ = (config, timeout_seconds)
        if provider_name == "openrouter":
            return False, "bad token"
        return True, "ok"

    monkeypatch.setattr(
        "embx.providers.discovery.test_provider_connection",
        fake_test_provider_connection,
    )

    result = runner.invoke(
        app,
        ["doctor", "--json", "--check-auth", "--only-configured"],
        env={
            "EMBX_OPENAI_API_KEY": "key",
            "EMBX_OPENROUTER_API_KEY": "key",
            "EMBX_HUGGINGFACE_API_KEY": "key",
            "EMBX_VOYAGE_API_KEY": "key",
        },
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    row_map = {row["provider"]: row for row in payload}
    assert row_map["openai"]["auth_status"] == "ok"
    assert row_map["openrouter"]["auth_status"] == "error"
    assert row_map["openrouter"]["auth_detail"] == "bad token"
