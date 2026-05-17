"""Protected server-side wallet signer compatibility endpoint."""

from __future__ import annotations

import hmac
import secrets
from datetime import datetime
from typing import Any

from cryptography.fernet import Fernet
from flask import Blueprint, current_app, jsonify, request

from ..extensions import db
from ..models import Setting, WalletAddress, WalletWithdrawal
from ..services.wallet_addresses import validate_withdraw_address
from ..services.wallet_custody import (
    BitcoinWalletAdapter,
    EvmWalletAdapter,
    SolanaWalletAdapter,
    XrpWalletAdapter,
)

internal_mpc_signer_bp = Blueprint("internal_mpc_signer", __name__, url_prefix="/_internal/mpc-signer")

_KEY_STORE_SETTING = "internal_mpc_signer_keys_v1"


@internal_mpc_signer_bp.post("/wallets")
def create_signer_wallet():
    auth_error = _auth_error()
    if auth_error is not None:
        return auth_error
    payload = request.get_json(silent=True) or {}
    try:
        user_id = int(payload.get("user_id") or 0)
    except (TypeError, ValueError):
        user_id = 0
    asset = _asset_key(str(payload.get("asset") or ""))
    network = str(payload.get("network") or _default_network(asset)).strip() or _default_network(asset)
    if user_id <= 0 or not asset:
        return jsonify({"ok": False, "error": "user_id and asset are required", "code": "invalid_request"}), 400

    store = _load_store()
    for signer_key_id, record in store.get("keys", {}).items():
        if (
            int(record.get("user_id") or 0) == user_id
            and str(record.get("asset") or "").upper() == asset
            and str(record.get("network") or "") == network
        ):
            return jsonify(_wallet_response(signer_key_id, record))

    adapter = _adapter_for(asset, network)
    if adapter is None:
        return jsonify({"ok": False, "error": "unsupported asset/network", "code": "unsupported_asset_network"}), 400
    generated = adapter.generate_wallet(asset, network)
    if not validate_withdraw_address(generated.address, asset, network):
        return jsonify({"ok": False, "error": "generated address failed validation", "code": "generated_address_invalid"}), 500

    signer_key_id = f"signer_{secrets.token_urlsafe(24).replace('-', '').replace('_', '')[:32]}"
    record = {
        "user_id": user_id,
        "asset": asset,
        "network": network,
        "address": generated.address,
        "public_key": generated.public_key,
        "key_type": generated.key_type,
        "provider": "algvault_internal_signer",
        "encrypted_private_key": _encrypt_private_key(generated.private_key),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    store.setdefault("keys", {})[signer_key_id] = record
    Setting.set_json(_KEY_STORE_SETTING, store)
    db.session.commit()
    return jsonify(_wallet_response(signer_key_id, record))


@internal_mpc_signer_bp.post("/withdrawals/sign-and-broadcast")
def sign_and_broadcast_withdrawal():
    auth_error = _auth_error()
    if auth_error is not None:
        return auth_error
    payload = request.get_json(silent=True) or {}
    signer_key_id = str(payload.get("signer_key_id") or "").strip()
    store = _load_store()
    record = store.get("keys", {}).get(signer_key_id)
    if not record:
        return jsonify({"ok": False, "error": "signer key was not found", "code": "signer_key_not_found"}), 404

    try:
        withdrawal_id = int(payload.get("withdrawal_id") or 0)
        user_id = int(payload.get("user_id") or 0)
        amount = float(payload.get("amount") or 0.0)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid withdrawal payload", "code": "invalid_request"}), 400
    asset = _asset_key(str(payload.get("asset") or ""))
    network = str(payload.get("network") or "").strip()
    destination = str(payload.get("destination_address") or "").strip()
    source_address = str(payload.get("source_address") or "").strip()
    if (
        user_id <= 0
        or amount <= 0
        or asset != str(record.get("asset") or "").upper()
        or network != str(record.get("network") or "")
        or source_address.lower() != str(record.get("address") or "").lower()
        or not validate_withdraw_address(destination, asset, network)
    ):
        return jsonify({"ok": False, "error": "withdrawal payload does not match signer key", "code": "payload_mismatch"}), 400

    source = WalletAddress.query.filter_by(user_id=user_id, asset=asset, network=network, address=source_address, status="active").first()
    if source is None:
        return jsonify({"ok": False, "error": "source wallet address is not active", "code": "source_wallet_not_active"}), 409

    withdrawal = db.session.get(WalletWithdrawal, withdrawal_id) if withdrawal_id > 0 else None
    if withdrawal is None:
        withdrawal = WalletWithdrawal(
            id=withdrawal_id or None,
            user_id=user_id,
            asset=asset,
            network=network,
            amount=amount,
            destination_address=destination,
            source_wallet_address_id=source.id,
        )
    withdrawal.user_id = user_id
    withdrawal.asset = asset
    withdrawal.network = network
    withdrawal.amount = amount
    withdrawal.destination_address = destination
    withdrawal.source_wallet_address_id = source.id

    adapter = _adapter_for(asset, network)
    if adapter is None:
        return jsonify({"ok": False, "error": "unsupported asset/network", "code": "unsupported_asset_network"}), 400
    try:
        result = adapter.sign_and_broadcast(withdrawal, _decrypt_private_key(str(record.get("encrypted_private_key") or "")))
    except Exception as exc:  # noqa: BLE001
        return (
            jsonify(
                {
                    "ok": False,
                    "status": "failed",
                    "provider_reference": "",
                    "error": f"internal signer broadcast failed: {exc}",
                    "code": "internal_signer_broadcast_failed",
                }
            ),
            502,
        )
    response = {"ok": True, "status": result.status, "provider_reference": result.provider_reference}
    response.update(result.raw)
    return jsonify(response)


def _auth_error():
    if not bool(current_app.config.get("WALLET_INTERNAL_MPC_SIGNER_ENABLED", False)):
        return jsonify({"ok": False, "error": "not found"}), 404
    expected = str(current_app.config.get("WALLET_MPC_SIGNER_TOKEN", "") or "").strip()
    header = str(request.headers.get("Authorization", "") or "")
    supplied = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        _fernet()
    except Exception:  # noqa: BLE001
        return jsonify({"ok": False, "error": "signer encryption key is not configured", "code": "signer_key_unavailable"}), 503
    return None


def _load_store() -> dict[str, Any]:
    store = Setting.get_json(_KEY_STORE_SETTING, {"keys": {}})
    if not isinstance(store, dict):
        return {"keys": {}}
    keys = store.get("keys")
    if not isinstance(keys, dict):
        store["keys"] = {}
    return store


def _wallet_response(signer_key_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "address": record.get("address"),
        "public_key": record.get("public_key"),
        "signer_key_id": signer_key_id,
        "key_type": record.get("key_type"),
        "provider": record.get("provider") or "algvault_internal_signer",
    }


def _encrypt_private_key(private_key: str) -> str:
    return _fernet().encrypt(str(private_key).encode("utf-8")).decode("utf-8")


def _decrypt_private_key(encrypted_private_key: str) -> str:
    return _fernet().decrypt(str(encrypted_private_key).encode("utf-8")).decode("utf-8")


def _fernet() -> Fernet:
    return Fernet(str(current_app.config.get("WALLET_MPC_SIGNER_ENCRYPTION_KEY", "") or "").encode("utf-8"))


def _adapter_for(asset: str, network: str):
    for adapter in (
        EvmWalletAdapter(current_app.config),
        BitcoinWalletAdapter(current_app.config),
        SolanaWalletAdapter(current_app.config),
        XrpWalletAdapter(current_app.config),
    ):
        if adapter.supports(asset, network):
            return adapter
    return None


def _asset_key(asset: str) -> str:
    return "".join(ch for ch in str(asset or "").upper() if ch.isalnum())


def _default_network(asset: str) -> str:
    if asset == "BTC":
        return "Bitcoin"
    if asset == "SOL":
        return "Solana"
    if asset == "XRP":
        return "XRP Ledger"
    return "Ethereum"
