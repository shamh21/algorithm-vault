"""Typed production failure classes for high-risk execution boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AlgVaultError(Exception):
    """Base application error with structured, non-secret context."""

    message: str
    code: str = "algvault_error"
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)


class ProviderConnectionError(AlgVaultError):
    pass


class ProviderRateLimitError(ProviderConnectionError):
    pass


class ProviderOrderRejectedError(AlgVaultError):
    pass


class TradeExecutionError(AlgVaultError):
    pass


class WalletCustodyError(AlgVaultError):
    pass


class WalletBroadcastError(WalletCustodyError):
    pass


class WalletReconciliationError(WalletCustodyError):
    pass


class ModelPromotionError(AlgVaultError):
    pass


class StrategyLoopError(AlgVaultError):
    pass


class MigrationStateError(AlgVaultError):
    pass
