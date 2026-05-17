"""Withdrawal enablement helpers shared by routes, services, and readiness checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .wallet_addresses import use_real_addresses

PRODUCTION_TARGETS = {"vps", "production", "prod", "postgres", "staging", "vercel"}
APPROVED_PRODUCTION_CUSTODY_MODES = {"kms", "hsm", "mpc"}


def automatic_withdrawal_blockers(config: Mapping[str, Any]) -> list[str]:
    """Return blockers that prevent safe automatic wallet withdrawal enablement."""

    blockers: list[str] = []
    if not bool(config.get("WALLET_AUTO_ENABLE_WITHDRAWALS", True)):
        blockers.append("WALLET_AUTO_ENABLE_WITHDRAWALS is disabled")
    if not bool(config.get("ENABLE_LIVE_TRADING", False)):
        blockers.append("live trading is disabled")
    if str(config.get("APP_MODE", "") or "").strip().lower() != "live":
        blockers.append("APP_MODE must be live")
    if str(config.get("APP_MODE", "") or "").strip().lower() == "live" and str(
        config.get("APP_ENV") or config.get("FLASK_ENV") or ""
    ).strip().lower() == "testing":
        blockers.append("testing environment cannot auto-enable withdrawals")
    if not use_real_addresses(dict(config)):
        blockers.append("USE_REAL_ADDRESSES is disabled")
    if not bool(config.get("WALLET_REAL_CUSTODY_ENABLED", False)):
        blockers.append("WALLET_REAL_CUSTODY_ENABLED is disabled")
    if not bool(config.get("WALLET_ALLOW_IN_APP_KEYGEN", False)):
        blockers.append("WALLET_ALLOW_IN_APP_KEYGEN is disabled")
    if bool(config.get("WALLET_EMERGENCY_STOP", False)):
        blockers.append("WALLET_EMERGENCY_STOP is active")

    deployment_target = str(config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
    custody_mode = str(config.get("WALLET_CUSTODY_MODE", "local_dev") or "local_dev").strip().lower()
    if deployment_target in PRODUCTION_TARGETS:
        if custody_mode not in APPROVED_PRODUCTION_CUSTODY_MODES:
            blockers.append("production withdrawals require kms, hsm, or mpc custody mode")
        if custody_mode == "mpc":
            if not str(config.get("WALLET_MPC_SIGNER_URL", "") or "").strip():
                blockers.append("WALLET_MPC_SIGNER_URL is not configured")
            if not str(config.get("WALLET_MPC_SIGNER_TOKEN", "") or "").strip():
                blockers.append("WALLET_MPC_SIGNER_TOKEN is not configured")
        if bool(config.get("WALLET_SIGNER_ISOLATION_REQUIRED", True)) and not bool(
            config.get("WALLET_SIGNER_ISOLATION_CONFIRMED", False)
        ):
            blockers.append("production withdrawals require confirmed signer isolation")
        if not bool(config.get("WALLET_SDK_CHECKS_PASSED", False)):
            blockers.append("wallet SDK/integration checks are not marked passing")
        if float(config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET", 0.0) or 0.0) <= 0:
            blockers.append("per-wallet daily withdrawal limit is not configured")
        asset_limits = config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET") or {}
        if not isinstance(asset_limits, Mapping) or not asset_limits:
            blockers.append("daily withdrawal limits by asset are not configured")
        if float(config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION", 0.0) or 0.0) <= 0:
            blockers.append("per-destination daily withdrawal limit is not configured")
        if float(config.get("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT", 0.0) or 0.0) <= 0:
            blockers.append("global daily withdrawal limit is not configured")

    return list(dict.fromkeys(blockers))


def wallet_withdrawals_enabled(config: Mapping[str, Any]) -> bool:
    """Return true when withdrawals are explicitly enabled or can be safely auto-enabled."""

    if bool(config.get("WALLET_WITHDRAWALS_ENABLED", False)):
        return True
    return not automatic_withdrawal_blockers(config)
