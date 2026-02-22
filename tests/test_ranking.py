from embx.ranking import apply_ranking, supported_rankings


def test_supported_rankings_contains_expected_strategies() -> None:
    assert supported_rankings() == ("none", "latency", "cost", "quality")


def test_apply_ranking_quality_assigns_expected_order() -> None:
    rows = [
        {
            "provider": "a",
            "status": "ok",
            "latency_ms": 5.0,
            "cost_usd": 0.1,
            "quality_score": None,
            "_vector": [1.0, 0.0, 0.0],
        },
        {
            "provider": "b",
            "status": "ok",
            "latency_ms": 6.0,
            "cost_usd": 0.1,
            "quality_score": None,
            "_vector": [0.9, 0.1, 0.0],
        },
        {
            "provider": "c",
            "status": "ok",
            "latency_ms": 7.0,
            "cost_usd": 0.1,
            "quality_score": None,
            "_vector": [-1.0, 0.0, 0.0],
        },
    ]

    result = apply_ranking(rows, "quality")

    assert result.ranked_rows[0]["provider"] in {"a", "b"}
    assert result.ranked_rows[-1]["provider"] == "c"
    assert result.ranked_rows[0]["rank"] == 1
    assert result.ranked_rows[1]["rank"] == 2
    assert result.ranked_rows[2]["rank"] == 3


def test_apply_ranking_none_preserves_order_and_no_rank() -> None:
    rows = [
        {"provider": "x", "status": "ok", "_vector": [1.0], "quality_score": None},
        {"provider": "y", "status": "error", "_vector": None, "quality_score": None},
    ]

    result = apply_ranking(rows, "none")

    assert [row["provider"] for row in result.ranked_rows] == ["x", "y"]
    assert all(row["rank"] is None for row in result.ranked_rows)
