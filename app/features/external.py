"""External data adapter interfaces with deterministic neutral stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExternalDataPoint:
    """Normalized external signal for a symbol."""

    provider: str
    score: float = 0.0
    confidence: float = 0.0
    label: str = "neutral"

    def as_dict(self) -> dict[str, float | str]:
        return {
            "provider": self.provider,
            "score": self.score,
            "confidence": self.confidence,
            "label": self.label,
        }


class ExternalDataAdapter(Protocol):
    """Protocol for future external data providers."""

    name: str

    def get_signal(self, symbol: str) -> ExternalDataPoint:
        ...


class NeutralExternalDataAdapter:
    """Deterministic no-network adapter used until real providers are configured."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_signal(self, symbol: str) -> ExternalDataPoint:
        return ExternalDataPoint(provider=self.name)


class CoinGeckoStubAdapter(NeutralExternalDataAdapter):
    def __init__(self) -> None:
        super().__init__("coingecko")


class SentimentStubAdapter(NeutralExternalDataAdapter):
    def __init__(self) -> None:
        super().__init__("sentiment")


class DuneStubAdapter(NeutralExternalDataAdapter):
    def __init__(self) -> None:
        super().__init__("dune")


class NansenStubAdapter(NeutralExternalDataAdapter):
    def __init__(self) -> None:
        super().__init__("nansen")


def default_external_adapters() -> list[ExternalDataAdapter]:
    """Return all v1.1 neutral external adapters."""

    return [
        CoinGeckoStubAdapter(),
        SentimentStubAdapter(),
        DuneStubAdapter(),
        NansenStubAdapter(),
    ]
