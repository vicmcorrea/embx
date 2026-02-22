from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from embx.exceptions import ValidationError


RankRow = dict[str, Any]
RankingFn = Callable[[list[RankRow]], list[RankRow]]

RANKING_FACTORY: dict[str, RankingFn] = {}


def register_ranking(name: str) -> Callable[[RankingFn], RankingFn]:
    def decorator(func: RankingFn) -> RankingFn:
        RANKING_FACTORY[name] = func
        return func

    return decorator


def ranking_factory(name: str) -> RankingFn:
    ranking_fn = RANKING_FACTORY.get(name)
    if ranking_fn is None:
        available = ", ".join(sorted(RANKING_FACTORY))
        raise ValidationError(f"Unknown rank strategy '{name}'. Available: {available}")
    return ranking_fn


def supported_rankings() -> tuple[str, ...]:
    return tuple(RANKING_FACTORY.keys())


def _aligned_cosine_similarity(a: list[float], b: list[float]) -> float:
    size = min(len(a), len(b))
    if size == 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for index in range(size):
        left = a[index]
        right = b[index]
        dot += left * right
        norm_a += left * left
        norm_b += right * right

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


def _assign_quality_scores(rows: list[RankRow]) -> None:
    if not rows:
        return
    if len(rows) == 1:
        rows[0]["quality_score"] = 1.0
        return

    for row in rows:
        vector = row.get("_vector")
        if not isinstance(vector, list):
            row["quality_score"] = 0.0
            continue

        scores: list[float] = []
        for other in rows:
            if other is row:
                continue
            other_vector = other.get("_vector")
            if not isinstance(other_vector, list):
                continue
            scores.append(_aligned_cosine_similarity(vector, other_vector))

        row["quality_score"] = 0.0 if not scores else sum(scores) / len(scores)


@register_ranking("none")
def _rank_none(rows: list[RankRow]) -> list[RankRow]:
    return list(rows)


@register_ranking("latency")
def _rank_latency(rows: list[RankRow]) -> list[RankRow]:
    return sorted(rows, key=lambda row: float(row["latency_ms"]))


@register_ranking("cost")
def _rank_cost(rows: list[RankRow]) -> list[RankRow]:
    return sorted(
        rows,
        key=lambda row: float(row["cost_usd"]) if row["cost_usd"] is not None else float("inf"),
    )


@register_ranking("quality")
def _rank_quality(rows: list[RankRow]) -> list[RankRow]:
    return sorted(rows, key=lambda row: float(row["quality_score"]), reverse=True)


@dataclass(slots=True)
class RankingResult:
    ranked_rows: list[RankRow]
    successful_rows: list[RankRow]
    error_rows: list[RankRow]


def apply_ranking(rows: list[RankRow], rank_by: str) -> RankingResult:
    successful_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]

    _assign_quality_scores(successful_rows)

    if rank_by == "none":
        ranked_rows = list(rows)
        for row in ranked_rows:
            row["rank"] = None
        return RankingResult(
            ranked_rows=ranked_rows,
            successful_rows=successful_rows,
            error_rows=error_rows,
        )

    ranking_fn = ranking_factory(rank_by)
    ranked_successful_rows = ranking_fn(successful_rows)
    for index, row in enumerate(ranked_successful_rows, start=1):
        row["rank"] = index
    for row in error_rows:
        row["rank"] = None

    return RankingResult(
        ranked_rows=ranked_successful_rows + error_rows,
        successful_rows=ranked_successful_rows,
        error_rows=error_rows,
    )


def strip_private_fields(rows: list[RankRow]) -> list[RankRow]:
    cleaned: list[RankRow] = []
    for row in rows:
        public_row = dict(row)
        public_row.pop("_vector", None)
        cleaned.append(public_row)
    return cleaned
