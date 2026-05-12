"""Typed runtime configuration validation with secret redaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SECRET_KEYWORDS = ("SECRET", "TOKEN", "KEY", "PASSWORD", "PRIVATE", "WEBHOOK", "FERNET")
PRODUCTION_TARGETS = {"vps", "production", "prod", "postgres", "staging", "vercel"}
APPROVED_PRODUCTION_CUSTODY_MODES = {"kms", "hsm", "mpc"}
LOCAL_CUSTODY_MODES = {"local_dev", "encrypted_db"}
WORKER_MODES = {"web", "worker", "local", "dev", "test"}


@dataclass(frozen=True)
class RuntimeConfigValidation:
    """Validated high-risk runtime profile."""

    profile: str
    deployment_target: str
    database_backend: str
    worker_mode: str
    custody_mode: str
    withdrawals_enabled: bool
    live_trading_enabled: bool
    ml_enabled: bool
    blockers: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.blockers


class RuntimeConfigError(RuntimeError):
    """Raised when production runtime configuration would fail open."""

    def __init__(self, validation: RuntimeConfigValidation) -> None:
        self.validation = validation
        super().__init__("Invalid runtime configuration: " + "; ".join(validation.blockers))


def database_backend(database_url: str) -> str:
    if database_url.startswith(("postgresql://", "postgresql+", "postgres://")):
        return "postgres"
    if database_url.startswith("sqlite"):
        return "sqlite"
    return "other"


def redact_config_value(key: str, value: Any) -> Any:
    normalized = key.upper()
    if any(marker in normalized for marker in SECRET_KEYWORDS):
        return "[configured]" if value else ""
    if isinstance(value, Mapping):
        return {str(item_key): redact_config_value(str(item_key), item_value) for item_key, item_value in value.items()}
    return value


def redacted_config_snapshot(config: Mapping[str, Any], keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = keys or tuple(str(key) for key in config)
    return {key: redact_config_value(key, config.get(key)) for key in selected}


def validate_runtime_config(config: Mapping[str, Any], *, strict: bool = False) -> RuntimeConfigValidation:
    deployment_target = str(config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
    profile = str(config.get("APP_ENV") or config.get("FLASK_ENV") or deployment_target or "local").strip().lower()
    backend = database_backend(str(config.get("SQLALCHEMY_DATABASE_URI", "") or ""))
    worker_mode = str(config.get("WORKER_MODE", "web") or "web").strip().lower()
    custody_mode = str(config.get("WALLET_CUSTODY_MODE", "local_dev") or "local_dev").strip().lower()
    withdrawals_enabled = bool(config.get("WALLET_WITHDRAWALS_ENABLED", False))
    live_trading_enabled = bool(config.get("ENABLE_LIVE_TRADING", False))
    ml_enabled = any(
        bool(config.get(key, False))
        for key in (
            "RAPID_ML_LIVE_ENABLED",
            "ML_AUTO_VAULT_LIVE_ENABLED",
            "ML_LIVE_VAULT_ONE_SHOT_ENABLED",
            "ML_VAULT_TICK_ENABLED",
            "ML_SIGNAL_MODEL_ENABLED",
        )
    )

    blockers: list[str] = []
    if worker_mode not in WORKER_MODES:
        blockers.append(f"WORKER_MODE must be one of {sorted(WORKER_MODES)}")
    if custody_mode not in APPROVED_PRODUCTION_CUSTODY_MODES | LOCAL_CUSTODY_MODES:
        blockers.append("WALLET_CUSTODY_MODE must be local_dev, encrypted_db, kms, hsm, or mpc")

    production_like = deployment_target in PRODUCTION_TARGETS
    if production_like:
        if backend != "postgres":
            blockers.append("production DEPLOYMENT_TARGET requires a PostgreSQL DATABASE_URL")
        if bool(config.get("SCHEMA_BOOTSTRAP_ENABLED", False)) and not bool(config.get("ALLOW_PRODUCTION_SCHEMA_BOOTSTRAP", False)):
            blockers.append("production schema bootstrap is disabled; run migrations explicitly")
        if bool(config.get("ENABLE_IN_PROCESS_WORKERS", False)) and worker_mode == "web":
            blockers.append("production web mode cannot enable in-process workers")
        if live_trading_enabled and worker_mode == "web" and not bool(config.get("WORKER_PROCESS_CONFIGURED", False)):
            blockers.append("production live trading requires a dedicated worker process")
        if custody_mode == "encrypted_db" and withdrawals_enabled:
            blockers.append("production withdrawals require kms, hsm, or mpc custody mode")
        if withdrawals_enabled:
            _validate_production_withdrawal_settings(config, custody_mode, blockers)
        if ml_enabled and bool(config.get("ML_SIGNAL_ALLOW_LIVE_OVERRIDE", False)):
            blockers.append("production ML live override must stay disabled")

    validation = RuntimeConfigValidation(
        profile=profile,
        deployment_target=deployment_target,
        database_backend=backend,
        worker_mode=worker_mode,
        custody_mode=custody_mode,
        withdrawals_enabled=withdrawals_enabled,
        live_trading_enabled=live_trading_enabled,
        ml_enabled=ml_enabled,
        blockers=tuple(blockers),
    )
    if strict and blockers:
        raise RuntimeConfigError(validation)
    return validation


def _validate_production_withdrawal_settings(config: Mapping[str, Any], custody_mode: str, blockers: list[str]) -> None:
    if custody_mode not in APPROVED_PRODUCTION_CUSTODY_MODES:
        blockers.append("production withdrawals require approved custody mode: kms, hsm, or mpc")
    if bool(config.get("WALLET_EMERGENCY_STOP", False)):
        blockers.append("WALLET_EMERGENCY_STOP is active")
    if bool(config.get("WALLET_SIGNER_ISOLATION_REQUIRED", True)) and not bool(config.get("WALLET_SIGNER_ISOLATION_CONFIRMED", False)):
        blockers.append("production withdrawals require confirmed signer isolation")
    if not bool(config.get("WALLET_SDK_CHECKS_PASSED", False)):
        blockers.append("production withdrawals require passing wallet SDK/integration checks")
    if float(config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET", 0.0) or 0.0) <= 0:
        blockers.append("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET must be configured")
    asset_limits = config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET") or {}
    if not isinstance(asset_limits, Mapping) or not asset_limits:
        blockers.append("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET_JSON must be configured")
    if float(config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION", 0.0) or 0.0) <= 0:
        blockers.append("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION must be configured")
    if float(config.get("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT", 0.0) or 0.0) <= 0:
        blockers.append("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT must be configured")
