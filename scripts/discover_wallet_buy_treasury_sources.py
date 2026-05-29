#!/usr/bin/env python3
"""List redacted wallet-buy treasury source candidates from the active database."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.models import Setting, WalletAddress

REQUIRED_WALLET_BUY_ASSETS = ("ETH", "USDC", "USDT")
WALLET_BUY_NETWORK = "Ethereum"
SIGNER_KEY_STORE = "internal_mpc_signer_keys_v1"


def build_wallet_buy_treasury_report() -> dict[str, Any]:
    """Return active MPC source candidates without exposing signer key material."""

    key_store = Setting.get_json(SIGNER_KEY_STORE, {}) or {}
    signer_records = key_store.get("keys", {}) if isinstance(key_store, dict) else {}
    signer_records = signer_records if isinstance(signer_records, dict) else {}
    rows = (
        WalletAddress.query.filter(
            WalletAddress.status == "active",
            WalletAddress.network == WALLET_BUY_NETWORK,
            WalletAddress.asset.in_(REQUIRED_WALLET_BUY_ASSETS),
        )
        .order_by(WalletAddress.asset.asc(), WalletAddress.onchain_balance.desc(), WalletAddress.id.desc())
        .all()
    )

    candidates: dict[str, list[dict[str, Any]]] = {asset: [] for asset in REQUIRED_WALLET_BUY_ASSETS}
    for row in rows:
        candidate = _candidate_payload(row, signer_records)
        if candidate["custody"] == "mpc" and candidate["signer_key_id"]:
            candidates.setdefault(str(row.asset).upper(), []).append(candidate)

    selected: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for asset in REQUIRED_WALLET_BUY_ASSETS:
        asset_candidates = sorted(candidates.get(asset, []), key=_candidate_sort_key)
        candidates[asset] = asset_candidates
        winner = next((item for item in asset_candidates if item["funded"] and item["signer_record_matches"]), None)
        if winner is None:
            blockers.append(f"{asset} has no funded active MPC source wallet with a matching signer key")
            continue
        selected[asset] = {
            WALLET_BUY_NETWORK: {
                "source_address": winner["address"],
                "signer_key_id": winner["signer_key_id"],
                "signer_route": "evm",
            }
        }

    fee_candidates = [item for item in candidates.get("ETH", []) if item["signer_record_matches"]]
    return {
        "required_assets": list(REQUIRED_WALLET_BUY_ASSETS),
        "network": WALLET_BUY_NETWORK,
        "ready_for_env": not blockers,
        "blockers": blockers,
        "candidates": candidates,
        "recommended_env_name": "APPLE_PAY_TREASURY_SOURCE_WALLETS_JSON",
        "recommended_env_value": selected,
        "fee_address_env_name": "APPLE_PAY_TREASURY_FEE_ADDRESS",
        "fee_address_candidates": [
            {
                "wallet_address_id": item["wallet_address_id"],
                "address": item["address"],
                "onchain_balance": item["onchain_balance"],
                "onchain_status": item["onchain_status"],
                "onchain_checked_at": item["onchain_checked_at"],
            }
            for item in fee_candidates
        ],
        "notes": [
            "This report intentionally omits encrypted private keys, raw signer records, and secrets.",
            "Fund the selected ETH source with enough ETH for deliveries and token-transfer gas before enabling live buys.",
        ],
    }


def _candidate_payload(row: WalletAddress, signer_records: dict[str, Any]) -> dict[str, Any]:
    metadata = row.encrypted_metadata if isinstance(row.encrypted_metadata, dict) else {}
    signer_key_id = str(metadata.get("signer_key_id") or "").strip()
    signer_record = signer_records.get(signer_key_id) if signer_key_id else None
    signer_record_matches = _signer_record_matches(row, signer_record)
    return {
        "wallet_address_id": int(row.id),
        "asset": str(row.asset).upper(),
        "network": str(row.network),
        "address": str(row.address),
        "custody": str(metadata.get("custody") or ""),
        "signer_key_id": signer_key_id,
        "signer_key_found": isinstance(signer_record, dict),
        "signer_record_matches": signer_record_matches,
        "onchain_balance": float(row.onchain_balance or 0.0),
        "onchain_status": str(row.onchain_status or ""),
        "onchain_checked_at": _iso(row.onchain_checked_at),
        "funded": float(row.onchain_balance or 0.0) > 0.0,
    }


def _signer_record_matches(row: WalletAddress, signer_record: Any) -> bool:
    if not isinstance(signer_record, dict):
        return False
    return (
        str(signer_record.get("asset") or "").upper() == str(row.asset).upper()
        and str(signer_record.get("network") or "") == str(row.network)
        and str(signer_record.get("address") or "").lower() == str(row.address).lower()
    )


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[bool, bool, float, int]:
    return (
        not bool(candidate.get("signer_record_matches")),
        not bool(candidate.get("funded")),
        -float(candidate.get("onchain_balance") or 0.0),
        -int(candidate.get("wallet_address_id") or 0),
    )


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return ""


def main() -> int:
    from app import create_app

    app = create_app()
    with app.app_context():
        print(json.dumps(build_wallet_buy_treasury_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
