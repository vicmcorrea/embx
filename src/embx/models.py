from dataclasses import asdict, dataclass


@dataclass(slots=True)
class EmbeddingResult:
    text: str
    vector: list[float]
    provider: str
    model: str
    input_tokens: int | None = None
    cost_usd: float | None = None
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
