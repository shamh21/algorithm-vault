"""Consumer wallet and vault routes."""

from __future__ import annotations

import math
import threading
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, url_for
from sqlalchemy import or_
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import joinedload

from ...auth import current_user, qr_code_data_uri, require_authenticated_user, verify_totp
from ...extensions import db
from ...ml.online_ranker import ONE_H10_HORIZON, extract_features, horizon_from_context, horizon_from_duration, outcome_from_result
from ...models import (
    AuditLog,
    DepositAddress,
    Fill,
    LeveragedMarket,
    Order,
    Setting,
    StrategyRanking,
    StrategyRun,
    TradingConnection,
    VaultAllocationLeg,
    VaultCycle,
    WalletAddress,
    WalletBalance,
    WalletTransaction,
)
from ...runtime import get_current_mode, get_service, market_mode_for
from ...services.connection_health import (
    build_connection_health,
    latest_connection_health,
    operator_connection_message,
    store_connection_health,
)
from ...services.db_retry import commit_with_retry, is_database_locked
from ...services.market_scanner import ScoredCandidate
from ...services.one_h10_quality import ONE_H10_HORIZON_SECONDS, one_h10_forecast_live_blockers
from ...services.provider_assets import normalize_provider, provider_collateral_asset, provider_feature_context
from ...services.response_envelope import action_envelope, exception_envelope, readiness_envelope
from ...services.vault_allocation_assets import (
    BASE_VAULT_ALLOCATION_ASSETS,
    DEFAULT_ASSET_NETWORKS,
    allocation_asset_views,
    default_vault_allocation_asset,
    functional_wallet_network,
    supported_vault_allocation_assets,
    vault_asset_networks,
)
from ...services.vault_allocation_assets import (
    asset_usd_price as shared_asset_usd_price,
)
from ...services.vault_coherence import cycle_coherence_payload_from_forecasts, extract_cycle_coherence_payload
from ...services.vault_readiness import get_vault_cycle_readiness
from ...services.wallet_addresses import generate_deposit_address, use_real_addresses, validate_withdraw_address
from ...services.withdrawal_config import wallet_withdrawals_enabled
from ...services.worker_lease import in_process_workers_enabled
from ...utils import format_duration_seconds

consumer_bp = Blueprint("consumer", __name__)

SUPPORTED_WALLET_ASSETS = BASE_VAULT_ALLOCATION_ASSETS
SETTLEMENT_ASSETS = ("ETH", "BTC", "USDT", "USDC")
VAULT_UI_PROVIDERS = ("hyperliquid", "kucoin")
VAULT_PROVIDER_LABELS = {
    "hyperliquid": "Hyperliquid",
    "kucoin": "KuCoin",
    "binance": "Binance",
    "bybit": "Bybit",
    "dydx": "dYdX",
    "uniswap": "Uniswap",
}
ASSET_NETWORKS = DEFAULT_ASSET_NETWORKS
_CYCLE_START_JOBS: dict[str, dict[str, object]] = {}
_CYCLE_START_IDEMPOTENCY: dict[tuple[int, str], str] = {}
_CYCLE_START_SYNC_IDEMPOTENCY: dict[tuple[int, str], int] = {}
_CYCLE_START_JOB_LOCK = threading.Lock()
_CYCLE_START_JOB_KEY_PREFIX = "vault_start_job"
_CYCLE_START_IDEMPOTENCY_KEY_PREFIX = "vault_start_idem"
_CYCLE_START_SYNC_IDEMPOTENCY_KEY_PREFIX = "vault_start_cycle_idem"
_LIVE_API_DELEGATED_ENDPOINTS = {
    "consumer.vault_readiness",
    "consumer.vault_preview_route",
    "consumer.vault_routing_preview",
    "consumer.start_cycle",
    "consumer.create_vault_cycle",
    "consumer.vault_cycle_status",
    "consumer.cycle_start_status",
}


@consumer_bp.before_request
def _protect_consumer():
    if request.method == "OPTIONS":
        return None
    if request.endpoint and request.endpoint.startswith("consumer.legacy_"):
        return None
    vault_diagnostic_endpoints = {
        "consumer.vault",
        "consumer.vault_readiness",
        "consumer.vault_routing_preview",
        "consumer.vault_preview_route",
        "consumer.start_cycle",
        "consumer.vault_start_cycle",
    }
    guard = require_authenticated_user()
    if guard is not None:
        return guard
    if request.endpoint in _LIVE_API_DELEGATED_ENDPOINTS and _vault_live_api_deferred_for_request():
        return jsonify(
            {
                "ok": False,
                "code": "live_api_origin_required",
                "message": "Vault live exchange checks must be sent to the configured live API origin.",
                "live_api_origin": _public_live_api_origin(),
            }
        ), 409
    user = current_user()
    if (
        user is not None
        and request.endpoint not in vault_diagnostic_endpoints
        and _live_connection_required()
        and get_service("trading_connections").active_tradable_connection(user.id) is None
    ):
        flash("Connect, verify, and activate a live-ready trading account before using wallet and vault features.", "warning")
        return redirect(url_for("settings.connections"))
    return None


def _origin_from_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(raw)
    except Exception:  # noqa: BLE001
        return raw.rstrip("/")
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}".rstrip("/")


def _request_origin() -> str:
    return _origin_from_url(request.host_url)


def _public_live_api_origin() -> str:
    return _origin_from_url(str(current_app.config.get("PUBLIC_LIVE_API_ORIGIN") or ""))


def _vault_live_api_deferred_for_request() -> bool:
    live_origin = _public_live_api_origin()
    if not live_origin:
        return False
    return _request_origin() != live_origin


@consumer_bp.get("/")
def home():
    user = current_user()
    try:
        _sync_completed_cycles(user)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Home cycle sync skipped: %s", exc)

    wallet_summary = None
    wallet_error = ""
    try:
        balances = _wallet_balances(user)
        wallet_summary = get_service("wallet_summary").summary_for_user(user, balances=balances)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Home wallet balance unavailable: %s", exc)
        wallet_error = "Total wallet balance is temporarily unavailable."

    wallet_overview = _home_wallet_balance_payload(wallet_summary, wallet_error)
    pnl_history = _home_account_pnl_payload(user)
    return render_template(
        "home.html",
        wallet_overview=wallet_overview,
        pnl_history=pnl_history,
    )


@consumer_bp.get("/wallet/", strict_slashes=False)
def wallet():
    user = current_user()
    balances = _wallet_balances_read_only(user)
    wallet_summary = get_service("wallet_summary").summary_for_user(user, balances=balances, sync_custody=False)
    activity_page = get_service("wallet_activity").page_for_user(user.id, page=_wallet_activity_page_number())
    return render_template(
        "wallet.html",
        balances=wallet_summary.balances,
        wallet_summary=wallet_summary,
        wallet_view=_wallet_view_model(wallet_summary, activity_page.items),
        portfolio_total=wallet_summary.portfolio_total_usd,
        allocation_chart=_wallet_allocation_payload(wallet_summary),
        portfolio_trend=_portfolio_trend_payload(user, wallet_summary),
        transactions=activity_page.items,
        activity_page=activity_page,
        networks=ASSET_NETWORKS,
    )


@consumer_bp.route("/convert/", methods=["GET", "POST"], strict_slashes=False)
def convert():
    user = current_user()
    balances = _wallet_balances(user)
    assets = _wallet_convert_asset_rows(balances)
    asset_keys = [row["asset"] for row in assets]
    default_from = next(
        (row["asset"] for row in assets if float(row["available_balance"] or 0.0) > 0), asset_keys[0] if asset_keys else "USDC"
    )
    default_to = next((asset for asset in asset_keys if asset != default_from), "USDT" if default_from != "USDT" else "USDC")
    source = "form" if request.method == "POST" else "args"
    form_values = {
        "from_asset": _normalize_convert_asset(_request_value("from_asset", default_from, source=source), asset_keys, default_from),
        "to_asset": _normalize_convert_asset(_request_value("to_asset", default_to, source=source), asset_keys, default_to),
        "amount": str(_request_value("amount", "", source=source) or "").strip(),
    }
    errors: dict[str, str] = {}
    quote = None
    if form_values["amount"]:
        quote, errors = _wallet_convert_quote(form_values, assets)
    convert_state = _wallet_convert_state(form_values, assets, quote, errors)

    if request.method == "POST":
        if quote is not None and not errors:
            try:
                result = _execute_wallet_conversion(user, quote)
                commit_with_retry()
                flash(
                    f"Converted {result['from_amount']:.8f} {result['from_asset']} to {result['to_amount']:.8f} {result['to_asset']}.",
                    "success",
                )
                return redirect(url_for("consumer.convert", from_asset=result["to_asset"], to_asset=result["from_asset"]))
            except ValueError as exc:
                db.session.rollback()
                errors["form"] = str(exc)
                convert_state = _wallet_convert_state(form_values, assets, quote, errors)
        for message in dict.fromkeys(errors.values()):
            flash(message, "danger")

    return render_template(
        "convert.html",
        assets=assets,
        quote=quote,
        errors=errors,
        form_values=form_values,
        convert_state=convert_state,
    )


@consumer_bp.get("/wallet/deposit/<asset>")
def deposit(asset: str):
    user = current_user()
    asset = asset.upper().strip()
    if not _is_supported_wallet_asset(asset):
        flash("Unsupported deposit asset.", "danger")
        return redirect(url_for("consumer.wallet"))
    balance = _wallet_balance_for(user, asset)
    network = _selected_network(asset)
    address = _ensure_deposit_address(user.id, asset, network, balance)
    if address is None:
        flash("No deposit address configured for this asset/network.", "warning")
    return render_template(
        "deposit.html",
        balance=balance,
        address=address,
        qr_code_uri=qr_code_data_uri(address.address) if address is not None else None,
        networks=_asset_networks(asset),
        selected_network=network,
    )


@consumer_bp.post("/wallet/rotate-address/<asset>")
def rotate_address(asset: str):
    user = current_user()
    asset = asset.upper().strip()
    if not _is_supported_wallet_asset(asset):
        flash("Unsupported deposit asset.", "danger")
        return redirect(url_for("consumer.wallet"))
    if request.form.get("confirm_rotate") != "on":
        flash("Confirm that the old address should be burned before rotating.", "warning")
        return redirect(url_for("consumer.deposit", asset=asset))
    balance = _wallet_balance_for(user, asset)
    network = _selected_network(asset)
    old_address = _active_deposit_address(user.id, asset, network)
    new_address = _new_deposit_address(user.id, asset, network, rotated_from=old_address)
    if new_address is None:
        commit_with_retry()
        flash("No replacement deposit address is configured for that asset/network.", "danger")
        return redirect(url_for("consumer.deposit", asset=asset, network=network))
    if old_address is not None:
        old_address.is_active = False
        old_address.expired_at = datetime.utcnow()
        _deactivate_wallet_address_for_deposit(old_address)
    balance.active_deposit_address_id = new_address.id
    try:
        withdrawal = get_service("self_custody_wallet").handle_rotated_address(
            user.id,
            asset,
            network,
            old_address,
            new_address,
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Rotated-address sweep workflow failed closed.")
        db.session.add(
            AuditLog(
                category="wallet",
                action="rotation_sweep_error",
                message=f"Rotated-address sweep workflow failed closed: {exc}",
            )
        )
        withdrawal = None
    db.session.add(
        AuditLog(
            category="wallet",
            action="rotate_deposit_address",
            message=f"Rotated {asset} deposit address for user {user.username}.",
        )
    )
    commit_with_retry()
    if withdrawal is not None:
        flash("Deposit address rotated. Old-address funds require the gated sweep workflow before movement.", "warning")
    else:
        flash("Deposit address rotated. The previous address is now inactive.", "success")
    return redirect(url_for("consumer.deposit", asset=asset, network=network))


@consumer_bp.route("/wallet/withdraw/<asset>", methods=["GET", "POST"])
def withdraw(asset: str):
    user = current_user()
    asset = asset.upper().strip()
    if not _is_supported_wallet_asset(asset):
        flash("Unsupported withdrawal asset.", "danger")
        return redirect(url_for("consumer.wallet"))
    balance = _wallet_balance_for(user, asset)
    network = _selected_network(asset)
    errors: dict[str, str] = {}
    max_withdrawal_amount = _max_withdrawal_amount(user, asset, network)
    form_values = {
        "withdraw_address": request.values.get("withdraw_address", "").strip(),
        "amount": _decimal_amount(max_withdrawal_amount) if request.values.get("max") == "1" and max_withdrawal_amount > 0 else "",
        "network": network,
    }

    if request.method == "POST":
        if not wallet_withdrawals_enabled(current_app.config):
            flash("Withdrawals are disabled until the explicit or automatic safety gates are ready.", "danger")
            return redirect(url_for("consumer.withdraw", asset=asset))
        if Setting.get_json("panic_lock", False):
            flash("Withdrawals are paused while the safety lock is active.", "danger")
            return redirect(url_for("consumer.withdraw", asset=asset))
        withdraw_address = request.form.get("withdraw_address", "").strip()
        form_values["withdraw_address"] = withdraw_address
        try:
            amount = float(request.form.get("amount", "0") or 0)
        except ValueError:
            amount = 0.0
        code = request.form.get("totp_code", "").strip()
        network = request.form.get("network", network).strip()
        if request.form.get("withdraw_max") == "1":
            max_withdrawal_amount = _max_withdrawal_amount(user, asset, network)
            amount = max_withdrawal_amount
        form_values["amount"] = _decimal_amount(amount) if request.form.get("withdraw_max") == "1" else request.form.get("amount", "")
        form_values["network"] = network
        real_wallet_mode = use_real_addresses(current_app.config)
        available_limit = max_withdrawal_amount if real_wallet_mode else float(balance.available_balance or 0.0)

        if network not in _asset_networks(asset):
            errors["network"] = "Select a supported network."
        if not validate_withdraw_address(withdraw_address, asset, network):
            errors["withdraw_address"] = "Enter a valid destination address for the selected asset and network."
        if amount <= 0:
            errors["amount"] = "Enter a withdrawal amount greater than zero."
        elif amount > available_limit + 1e-9:
            errors["amount"] = "Withdrawal amount exceeds available balance."
        if not verify_totp(user, code):
            errors["totp_code"] = "Invalid authenticator code. Try again."
        if real_wallet_mode and get_current_mode() != "live":
            errors["form"] = "Real wallet withdrawals can only be broadcast in live mode."
        max_by_asset = current_app.config.get("WALLET_MAX_WITHDRAWAL_BY_ASSET") or {}
        if isinstance(max_by_asset, dict):
            max_amount = float(max_by_asset.get(asset.lower(), max_by_asset.get(asset, 0.0)) or 0.0)
            if max_amount > 0 and amount > max_amount:
                errors["amount"] = f"Withdrawal amount exceeds configured {asset} cap."

        if not errors and real_wallet_mode:
            try:
                _materialize_onchain_surplus_for_operation(user, asset, network, amount)
                db.session.flush()
                db.session.refresh(balance)
            except Exception as exc:  # noqa: BLE001
                errors["amount"] = str(exc)

        if not errors:
            wallet_service = get_service("self_custody_wallet")
            connection = _active_trading_connection(user)
            reserved = False
            if real_wallet_mode:
                balance.available_balance = float(balance.available_balance or 0.0) - amount
                balance.locked_balance = float(balance.locked_balance or 0.0) + amount
                reserved = True
            withdrawal = wallet_service.create_manual_withdrawal(
                user_id=user.id,
                asset=asset,
                network=network,
                destination_address=withdraw_address,
                amount=amount,
                trading_connection_id=connection.id if connection is not None else None,
            )
            if real_wallet_mode and withdrawal.status == "queued_treasury_solvency":
                db.session.add(
                    WalletTransaction(
                        user_id=user.id,
                        asset=asset,
                        amount=amount,
                        transaction_type="withdrawal",
                        status="pending_withdrawal",
                        network=network,
                        withdraw_address=withdraw_address,
                        note=f"Withdrawal workflow {withdrawal.id}: queued_treasury_solvency.",
                    )
                )
                commit_with_retry()
                flash("Withdrawal queued until treasury gas reserve coverage recovers. Funds remain locked for the workflow.", "warning")
                return redirect(url_for("consumer.wallet"))
            if real_wallet_mode and bool(current_app.config.get("WALLET_REQUIRE_WITHDRAWAL_APPROVAL", True)):
                db.session.add(
                    WalletTransaction(
                        user_id=user.id,
                        asset=asset,
                        amount=amount,
                        transaction_type="withdrawal",
                        status="pending_approval",
                        network=network,
                        withdraw_address=withdraw_address,
                        note=f"Withdrawal workflow {withdrawal.id}: pending_approval.",
                    )
                )
                commit_with_retry()
                flash("Withdrawal request submitted for admin approval. Funds are locked until approval or rejection.", "success")
                return redirect(url_for("consumer.wallet"))
            withdrawal = wallet_service.submit_withdrawal(withdrawal, mode=get_current_mode())
            if withdrawal.status.startswith("failed"):
                if reserved:
                    get_service("wallet_custody").release_failed_withdrawal(withdrawal)
                current_app.logger.error("Withdrawal %s failed: %s", withdrawal.id, withdrawal.failure_reason)
                db.session.add(
                    WalletTransaction(
                        user_id=user.id,
                        asset=asset,
                        amount=amount,
                        transaction_type="withdrawal",
                        status="failed",
                        network=network,
                        withdraw_address=withdraw_address,
                        note=withdrawal.failure_reason or "Withdrawal failed.",
                    )
                )
                commit_with_retry()
                errors["form"] = withdrawal.failure_reason or "Withdrawal failed. Check logs for details."
            else:
                if not reserved:
                    balance.available_balance = float(balance.available_balance or 0.0) - amount
                transaction_status = "complete" if withdrawal.status == "complete" else "pending_withdrawal"
                if transaction_status == "pending_withdrawal" and not reserved:
                    balance.locked_balance = float(balance.locked_balance or 0.0) + amount
                db.session.add(
                    WalletTransaction(
                        user_id=user.id,
                        asset=asset,
                        amount=amount,
                        transaction_type="withdrawal",
                        status=transaction_status,
                        network=network,
                        withdraw_address=withdraw_address,
                        note=f"Withdrawal workflow {withdrawal.id}: {withdrawal.status}.",
                    )
                )
                commit_with_retry()
                message = (
                    "Withdrawal broadcast. Waiting for confirmation."
                    if withdrawal.status == "submitted"
                    else f"Withdrawal status: {withdrawal.status.replace('_', ' ')}."
                )
                flash(message, "success")
                return redirect(url_for("consumer.wallet"))
        else:
            for message in dict.fromkeys(errors.values()):
                flash(message, "danger")

    return render_template(
        "withdraw.html",
        balance=balance,
        networks=_asset_networks(asset),
        selected_network=network,
        errors=errors,
        form_values=form_values,
        max_withdrawal_amount=max_withdrawal_amount,
    )


@consumer_bp.get("/vault/", strict_slashes=False)
def vault():
    user = current_user()
    defer_live_api = _vault_live_api_deferred_for_request()
    if not defer_live_api:
        _sync_completed_cycles(user)
    balances = _wallet_balances(user)
    default_asset = _default_vault_asset(balances)
    active_cycles = _active_cycles(user, refresh=not defer_live_api)
    recovered_run_ids = [] if defer_live_api else _recover_active_one_h10_cycles(active_cycles)
    for cycle in active_cycles:
        cycle.cycle_summary = get_service("vault_cycle_reporting").status_payload(cycle) if defer_live_api else _cycle_summary(cycle)
    cycle_page = get_service("vault_activity").page_for_user(user.id, page=_vault_cycle_page_number())
    initial_routing_preview = (
        _deferred_live_api_routing_preview_payload(
            amount=0.0,
            deposit_asset=default_asset,
            settlement_asset=default_asset,
            providers=list(VAULT_UI_PROVIDERS),
        )
        if defer_live_api
        else _vault_routing_preview_payload(
            user=user,
            amount=0.0,
            deposit_asset=default_asset,
            settlement_asset=default_asset,
            providers=list(VAULT_UI_PROVIDERS),
        )
    )
    commit_with_retry()
    if not defer_live_api:
        _start_strategy_runs(recovered_run_ids)
    return render_template(
        "vault.html",
        balances=balances,
        active_cycle=active_cycles[0] if active_cycles else None,
        active_cycles=active_cycles,
        recent_cycles=cycle_page.items,
        cycle_page=cycle_page,
        settlement_assets=_wallet_assets(),
        vault_default_asset=default_asset,
        vault_provider_options=_vault_provider_options(),
        vault_cycle_options=_vault_cycle_options(),
        initial_routing_preview=initial_routing_preview,
    )


@consumer_bp.post("/consumer/start")
@consumer_bp.post("/vault/start")
@consumer_bp.post("/vault/start-cycle")
def start_cycle():
    user = current_user()
    async_enabled = bool(current_app.config.get("VAULT_START_ASYNC_ENABLED", False))
    idempotency_key = _request_idempotency_key()
    if async_enabled and idempotency_key:
        existing_job = _existing_cycle_start_job(user.id, idempotency_key)
        if existing_job is not None:
            if _wants_start_json_response():
                return jsonify(_with_cycle_start_runtime_metadata(existing_job)), 202
            flash("Cycle start already queued. Refresh cycle status in a moment.", "info")
            return redirect(url_for("consumer.vault"))
    if not async_enabled and idempotency_key:
        existing_cycle = _existing_cycle_start_cycle(user.id, idempotency_key)
        if existing_cycle is not None:
            if _wants_start_json_response():
                return jsonify(
                    action_envelope(
                        ok=True,
                        code="vault_cycle_duplicate",
                        message="Cycle start already submitted. Showing the existing cycle.",
                        ready=True,
                        created=False,
                        duplicate=True,
                        **_cycle_start_runtime_metadata(
                            cycle_id=existing_cycle.id,
                            run_ids=[leg.strategy_run_id for leg in existing_cycle.allocation_legs if leg.strategy_run_id],
                            status="duplicate",
                        ),
                    )
                ), 200
            flash("Cycle start already submitted. Showing the existing cycle.", "info")
            return redirect(url_for("consumer.cycle_detail", cycle_id=existing_cycle.id))

    _sync_completed_cycles(user)

    asset = str(_request_value("deposit_asset", "USDC")).upper().strip()
    wallet_assets = _wallet_assets()
    settlement_raw = str(_request_value("settlement_asset", "")).upper().strip()
    settlement_asset = asset if settlement_raw in {"", "AUTO", "__AUTO__"} else settlement_raw
    if asset not in wallet_assets or settlement_asset not in wallet_assets:
        return _vault_start_error_response(
            "settlement_asset_unsupported",
            "Unsupported asset",
            "Select a supported wallet and settlement asset.",
        )

    try:
        amount = float(_request_value("deposit_amount", "0") or 0)
    except ValueError:
        amount = 0.0

    duration_seconds = _requested_duration_seconds()
    if duration_seconds <= 0:
        return _vault_start_error_response("duration_invalid", "Invalid duration", "Select a valid lock duration.")
    duration_hours = max(1, math.ceil(duration_seconds / 3600))
    is_one_h10 = _is_one_h10_duration(duration_seconds, duration_hours)

    requested_providers = _requested_provider_keys() if is_one_h10 else []
    if is_one_h10 and _wants_start_json_response():
        readiness = get_vault_cycle_readiness(
            user.id,
            cycle="1H10",
            settlement_asset=settlement_asset,
            deposit_asset=asset,
            amount=amount,
            enabled_exchanges=requested_providers,
            live_acknowledged=_one_h10_live_acknowledged(),
            idempotency_key=idempotency_key,
            enforce_ml_gate=_wants_start_json_response(),
            require_market_metadata=_wants_start_json_response(),
        )
        if not bool(readiness.get("ready", False)):
            return _vault_start_blocked_response(readiness)
    if amount <= 0:
        return _vault_start_error_response("amount_required", "Amount required", "Enter an allocation amount greater than zero.")

    if _vault_cycle_engine_form_enabled(is_one_h10):
        return _start_vault_cycle_engine_from_route(
            user=user,
            amount=amount,
            deposit_asset=asset,
            settlement_asset=settlement_asset,
            duration_seconds=duration_seconds,
            providers=_requested_provider_keys(),
            allowed_symbols=_requested_allowed_symbols(),
            idempotency_key=idempotency_key,
            success_message="Vault Cycle started with dynamic exchange allocation.",
            wants_start_response=True,
        )

    connections = _cycle_trading_connections(user, is_one_h10, providers=requested_providers if is_one_h10 else None)
    connection = connections[0] if connections else None
    if _live_connection_required() and connection is None:
        return _vault_start_error_response(
            "verified_connection_missing",
            "Verified connection missing",
            "Connect and verify at least one trading account before starting 1H10."
            if is_one_h10
            else "Connect your trading account before starting a live vault cycle.",
        )
    if is_one_h10:
        one_h10_block = _one_h10_live_start_block_reason()
        if one_h10_block:
            flash(one_h10_block, "warning")
            return redirect(url_for("consumer.vault"))
        if _live_connection_required() and not _one_h10_live_acknowledged():
            flash("Confirm the 1H10 acknowledgement before starting.", "warning")
            return redirect(url_for("consumer.vault"))

    balances = _wallet_balances(user)
    balance = next((item for item in balances if item.asset == asset), None)
    network = _asset_networks(asset)[0]
    verified_spendable = _verified_spendable_amount(user, asset, network)
    if verified_spendable is not None:
        if verified_spendable + 1e-9 < amount:
            flash("That allocation is higher than the verified on-chain wallet balance.", "danger")
            return redirect(url_for("consumer.vault"))
        try:
            _materialize_onchain_surplus_for_operation(user, asset, network, amount)
            db.session.flush()
            balance = WalletBalance.query.filter_by(user_id=user.id, asset=asset).one_or_none()
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "danger")
            return redirect(url_for("consumer.vault"))
    if balance is None or float(balance.available_balance) + 1e-9 < amount:
        flash("That allocation is higher than the available wallet balance.", "danger")
        return redirect(url_for("consumer.vault"))

    price = _asset_usd_price(asset)
    if price <= 0:
        flash("Market estimate is unavailable for that asset. Try a stable settlement asset or retry later.", "warning")
        return redirect(url_for("consumer.vault"))

    reserve_block_reason = _available_reserve_block_reason(balance, amount, price)
    if reserve_block_reason:
        flash(reserve_block_reason, "warning")
        return redirect(url_for("consumer.vault"))

    starting_value_usd = amount * price
    healthy_connections, connection_blockers = _healthy_cycle_connections(user, connections, is_one_h10)
    if is_one_h10:
        connections = healthy_connections
        connection = connections[0] if connections else None
    live_block_reason = None
    if not is_one_h10:
        live_block_reason = _fresh_live_connection_block_reason(user, connection)
    elif _live_connection_required() and not connections:
        live_block_reason = str(
            (connection_blockers[0] if connection_blockers else {}).get("reason")
            or "No verified 1H10 trading connection is currently healthy enough for live execution."
        )
    if live_block_reason:
        flash(live_block_reason, "danger")
        if connection is not None:
            return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))
        return redirect(url_for("settings.connections"))

    allowed_symbols = _requested_allowed_symbols()
    market_discovery: list[dict[str, object]] = []
    if is_one_h10:
        persist_start_features = bool(current_app.config.get("ONE_H10_START_SYNC_FEATURES", False))
        try:
            market_discovery = list(
                get_service("leveraged_markets").sync_for_user(
                    user.id,
                    mode="live",
                    feature_scope="all",
                    persist_features=persist_start_features,
                )
            )
        except Exception as exc:  # noqa: BLE001
            market_discovery = [{"skipped": True, "reason": str(exc)}]
    selection = get_service("vault_strategy_selector").select(
        asset,
        duration_hours,
        get_current_mode(),
        starting_value_usd,
        allowed_symbols=allowed_symbols,
        provider=connection.provider if connection is not None else None,
    )
    if is_one_h10:
        one_h10_legs, allocation_history, allocation_blockers = _one_h10_provider_legs(
            user=user,
            selection=selection,
            connections=connections,
            starting_value_usd=starting_value_usd,
            settlement_asset=settlement_asset,
            allowed_symbols=allowed_symbols,
            connection_blockers=connection_blockers,
        )
        if not one_h10_legs:
            flash(_one_h10_allocation_failure_message(allocation_blockers), "warning")
            return redirect(url_for("consumer.vault"))
        selection.legs[:] = one_h10_legs
        coherence_payload = cycle_coherence_payload_from_forecasts(
            [
                dict((leg.get("parameters") or {}).get("one_h10_forecast") or leg.get("forecast") or {})
                for leg in one_h10_legs
                if isinstance(leg, dict)
            ]
        )
        selection.metadata.update(
            {
                "exchange_allocation_history": allocation_history,
                "provider_allocation_history": allocation_history,
                "provider_skip_reasons": allocation_blockers,
                "requested_provider_filter": requested_providers,
                "market_discovery": market_discovery,
                "ml_readiness": _one_h10_ml_readiness("global"),
                "blocker_categories": _blocker_categories_from_reasons(allocation_blockers),
                "objective": "one_h10",
                "ml_objective": "one_h10",
                "ml_policy_required": True,
                "ml_governed_risk": True,
                **coherence_payload,
            }
        )
    block_reason = _cycle_start_block_reason(user, asset, duration_hours, starting_value_usd, selection)
    if block_reason:
        flash(block_reason, "warning")
        return redirect(url_for("consumer.vault"))
    now = datetime.utcnow()

    balance.available_balance = float(balance.available_balance) - amount
    balance.locked_balance = float(balance.locked_balance) + amount
    balance.estimated_usd_value = balance.total_balance * price

    cycle = VaultCycle(
        user_id=user.id,
        trading_connection_id=connection.id if connection is not None else None,
        deposit_asset=asset,
        deposit_amount=amount,
        settlement_asset=settlement_asset,
        lock_duration_hours=duration_hours,
        lock_duration_seconds=duration_seconds,
        status="active",
        execution_substatus=selection.execution_substatus,
        execution_mode=selection.execution_mode,
        live_validation_status=selection.live_validation_status,
        validation_started_at=now if selection.live_validation_status == "pending" else None,
        validation_failure_reason=selection.metadata.get("fallback_reason"),
        algorithm_profile="1H10" if is_one_h10 else selection.profile,
        selected_strategy_name=selection.strategy_name,
        selected_timeframe=selection.timeframe,
        started_at=now,
        unlocks_at=now + timedelta(seconds=duration_seconds),
        starting_value_usd=starting_value_usd,
        current_estimated_value_usd=starting_value_usd,
    )
    safe_selection_metadata = _json_safe_metadata(selection.metadata)
    selection.metadata.clear()
    selection.metadata.update(safe_selection_metadata)
    cycle.selection_metadata = selection.metadata
    db.session.add(cycle)
    db.session.flush()

    common_parameters = {
        "vault_cycle_id": cycle.id,
        "consumer_vault": True,
        "algorithm_profile": "1H10" if is_one_h10 else selection.profile,
        "vault_cycle_name": "1H10" if is_one_h10 else selection.profile,
        "one_h10_vault": is_one_h10,
        "ml_horizon": ONE_H10_HORIZON if is_one_h10 else selection.metadata.get("ml_horizon"),
        "objective": "one_h10" if is_one_h10 else selection.metadata.get("objective"),
        "ml_objective": "one_h10" if is_one_h10 else selection.metadata.get("ml_objective"),
        "target_return_objective": "one_h10" if is_one_h10 else selection.metadata.get("target_return_objective"),
        "ml_policy_required": True if is_one_h10 else bool(selection.metadata.get("ml_policy_required", False)),
        "ml_governed_risk": True if is_one_h10 else bool(selection.metadata.get("ml_governed_risk", False)),
        "target_roi_pct": selection.metadata.get("target_roi_pct"),
        "target_multiplier": selection.metadata.get("target_multiplier"),
        "target_amount_usd": selection.metadata.get("target_amount_usd"),
        "user_input_amount_usd": starting_value_usd,
        "settlement_asset": settlement_asset,
        "execution_mode": selection.execution_mode,
        "live_validation_status": selection.live_validation_status,
        "live_validation_started_at": now.isoformat() if selection.live_validation_status == "pending" else None,
        "lock_duration_hours": duration_hours,
        "lock_duration_seconds": duration_seconds,
        "allowed_symbols": allowed_symbols,
        "provider_filter": requested_providers,
        "user_id": user.id,
        "trading_connection_id": connection.id if connection is not None else None,
        "provider": selection.metadata.get("provider"),
        "execution_venue": selection.metadata.get("execution_venue"),
        "collateral_asset": selection.metadata.get("collateral_asset"),
    }
    run_ids: list[int] = []
    legs = selection.legs or [
        {
            "strategy_name": selection.strategy_name,
            "symbol": selection.symbol,
            "timeframe": selection.timeframe,
            "parameters": selection.parameters,
            "allocation_cap_usd": starting_value_usd,
            "leverage": selection.parameters.get("leverage", 1.0),
            "optimizer_ranking_id": selection.metadata.get("optimizer_ranking_id"),
        }
    ]
    for index, leg in enumerate(legs):
        leg_parameters = dict(leg.get("parameters") or selection.parameters)
        leg_parameters.update(common_parameters)
        include_pair_metadata = not (is_one_h10 and bool((leg.get("parameters") or {}).get("one_h10_all_pairs")))
        leg_parameters.update(
            {
                "allocation_cap_usd": float(leg.get("allocation_cap_usd", starting_value_usd) or 0.0),
                "optimizer_ranking_id": leg.get("optimizer_ranking_id"),
                "optimizer_profile": leg.get("optimizer_profile") or selection.metadata.get("optimizer_profile"),
                "edge_score": leg.get("edge_score"),
                "execution_style": leg.get("execution_style", "market"),
                "universe_source": leg.get("universe_source", "configured"),
                "allocation_mode": leg.get("allocation_mode") or selection.metadata.get("allocation_mode"),
                "ensemble_id": leg.get("ensemble_id") or selection.metadata.get("ensemble_id"),
                "ensemble_version": leg.get("ensemble_version") or selection.metadata.get("ensemble_version"),
                "ensemble_adapter": leg.get("ensemble_adapter"),
                "ensemble_weight": float(leg.get("ensemble_weight", 0.0) or 0.0),
                "target_ensemble_weight": float(leg.get("target_ensemble_weight", leg.get("ensemble_weight", 0.0)) or 0.0),
                "effective_allocation_weight": float(leg.get("effective_allocation_weight", leg.get("ensemble_weight", 0.0)) or 0.0),
                "cap_limited": bool(leg.get("cap_limited", False)),
                "cap_limit_reason": leg.get("cap_limit_reason", ""),
                "duration_bucket": selection.metadata.get("duration_bucket"),
                "ml_rank_score": float(leg.get("ml_rank_score", 0.0) or 0.0),
                "multi_timeframe_confluence": leg.get("multi_timeframe_confluence")
                or selection.metadata.get("multi_timeframe_confluence", {}),
                "confluence_score": float(leg.get("confluence_score", selection.metadata.get("confluence_score", 0.0)) or 0.0),
                "fib_confluence": leg.get("fib_confluence") or selection.metadata.get("fib_confluence", {}),
                "fibonacci_confluence": leg.get("fib_confluence") or selection.metadata.get("fib_confluence", {}),
                "market_regime": leg.get("market_regime") or selection.metadata.get("market_regime"),
                "pair_group_id": (leg.get("pair_group_id") or selection.metadata.get("pair_group_id")) if include_pair_metadata else None,
                "pair_mode": (leg.get("pair_mode") or selection.metadata.get("pair_mode")) if include_pair_metadata else None,
                "pair_symbol": (leg.get("pair_symbol") or selection.metadata.get("pair_symbol")) if include_pair_metadata else None,
                "pair_role": leg.get("pair_role") if include_pair_metadata else None,
                "pair_forced_side": (leg.get("parameters") or {}).get("pair_forced_side") if include_pair_metadata else None,
                "hedge_ratio": (leg.get("hedge_ratio") or selection.metadata.get("hedge_ratio")) if include_pair_metadata else None,
                "spread_zscore": (leg.get("spread_zscore") or selection.metadata.get("spread_zscore")) if include_pair_metadata else None,
                "spread_half_life": (leg.get("spread_half_life") or selection.metadata.get("spread_half_life"))
                if include_pair_metadata
                else None,
                "pair_score": (leg.get("pair_score") or selection.metadata.get("pair_score")) if include_pair_metadata else None,
                "correlation": (leg.get("correlation") or selection.metadata.get("correlation")) if include_pair_metadata else None,
                "pair_signal": (leg.get("pair_signal") or selection.metadata.get("pair_signal", {})) if include_pair_metadata else {},
                "pair_skip_reason": (leg.get("pair_skip_reason") or selection.metadata.get("pair_skip_reason", ""))
                if include_pair_metadata
                else "",
                "skip_reason": leg.get("skip_reason", ""),
                "leverage": float(leg.get("leverage", leg_parameters.get("leverage", 1.0)) or 1.0),
                "provider": leg.get("provider", leg_parameters.get("provider", selection.metadata.get("provider"))),
                "execution_venue": leg.get(
                    "execution_venue", leg_parameters.get("execution_venue", selection.metadata.get("execution_venue"))
                ),
                "trading_connection_id": leg.get("trading_connection_id", leg_parameters.get("trading_connection_id")),
                "collateral_asset": leg.get(
                    "collateral_asset", leg_parameters.get("collateral_asset", selection.metadata.get("collateral_asset"))
                ),
                "settlement_asset": leg.get("settlement_asset", leg_parameters.get("settlement_asset", settlement_asset)),
                "allocation_weight": float(leg.get("allocation_weight", leg_parameters.get("allocation_weight", 0.0)) or 0.0),
                "available_margin_usd": float(leg.get("available_margin_usd", leg_parameters.get("available_margin_usd", 0.0)) or 0.0),
                "market_id": leg.get("market_id", leg_parameters.get("market_id")),
                "venue_symbol": leg.get("venue_symbol", leg_parameters.get("venue_symbol")),
                "app_symbol": leg.get("app_symbol", leg_parameters.get("app_symbol")),
                "market_status": leg.get("market_status", leg_parameters.get("market_status")),
                "one_h10_scanner_score": leg_parameters.get("one_h10_scanner_score"),
                "one_h10_scanner_source": leg_parameters.get("one_h10_scanner_source"),
                "scanner_score_breakdown": leg_parameters.get("scanner_score_breakdown", {}),
                "scanner_features": leg_parameters.get("scanner_features", {}),
                "one_h10_forecast": leg_parameters.get("one_h10_forecast", {}),
                "forecast_metadata": leg_parameters.get("forecast_metadata", leg_parameters.get("one_h10_forecast", {})),
                "forecast_blockers": leg_parameters.get("forecast_blockers", []),
                "forecast_advisory_blockers": leg_parameters.get("forecast_advisory_blockers", []),
                "forecast_predicted_side": leg_parameters.get("forecast_predicted_side"),
                "forecast_confidence": leg_parameters.get("forecast_confidence"),
                "forecast_expected_return_bps": leg_parameters.get("forecast_expected_return_bps"),
                "forecast_suggested_notional_usd": leg_parameters.get("forecast_suggested_notional_usd"),
                "forecast_suggested_leverage": leg_parameters.get("forecast_suggested_leverage"),
                "forecast_suggested_order_type": leg_parameters.get("forecast_suggested_order_type"),
                "forecast_suggested_stop_loss_pct": leg_parameters.get("forecast_suggested_stop_loss_pct"),
                "forecast_suggested_take_profit_pct": leg_parameters.get("forecast_suggested_take_profit_pct"),
            }
        )
        identity = _normalized_cycle_leg_identity(
            leg,
            leg_parameters,
            selection,
            fallback_connection_id=connection.id if connection is not None else None,
        )
        leg_parameters.update(identity)
        leg_connection_id = identity["trading_connection_id"]
        run_symbol = str(identity["app_symbol"])
        run = StrategyRun(
            strategy_name=str(leg.get("strategy_name") or selection.strategy_name),
            symbol=run_symbol,
            timeframe=str(leg.get("timeframe") or selection.timeframe),
            mode=selection.mode,
            user_id=user.id,
            trading_connection_id=leg_connection_id,
            status="starting",
            lock_duration_seconds=duration_seconds,
            manual_enabled=True,
        )
        run.parameters = leg_parameters
        db.session.add(run)
        db.session.flush()
        leg_model = VaultAllocationLeg(
            vault_cycle_id=cycle.id,
            strategy_run_id=run.id,
            optimizer_ranking_id=leg.get("optimizer_ranking_id"),
            symbol=run.symbol,
            timeframe=run.timeframe,
            provider=str(leg_parameters.get("provider") or "global"),
            trading_connection_id=leg_connection_id,
            allocation_cap_usd=leg_parameters["allocation_cap_usd"],
            leverage=leg_parameters["leverage"],
            status="active",
        )
        leg_model.details = {
            "optimizer_profile": leg_parameters.get("optimizer_profile"),
            "edge_score": leg_parameters.get("edge_score"),
            "execution_style": leg_parameters.get("execution_style"),
            "universe_source": leg_parameters.get("universe_source"),
            "allocation_mode": leg_parameters.get("allocation_mode"),
            "ensemble_id": leg_parameters.get("ensemble_id"),
            "ensemble_version": leg_parameters.get("ensemble_version"),
            "ensemble_adapter": leg_parameters.get("ensemble_adapter"),
            "ensemble_weight": leg_parameters.get("ensemble_weight"),
            "target_ensemble_weight": leg_parameters.get("target_ensemble_weight"),
            "effective_allocation_weight": leg_parameters.get("effective_allocation_weight"),
            "cap_limited": leg_parameters.get("cap_limited"),
            "cap_limit_reason": leg_parameters.get("cap_limit_reason"),
            "duration_bucket": leg_parameters.get("duration_bucket"),
            "ml_rank_score": leg_parameters.get("ml_rank_score"),
            "multi_timeframe_confluence": leg_parameters.get("multi_timeframe_confluence"),
            "confluence_score": leg_parameters.get("confluence_score"),
            "fib_confluence": leg_parameters.get("fib_confluence"),
            "market_regime": leg_parameters.get("market_regime"),
            "pair_group_id": leg_parameters.get("pair_group_id"),
            "pair_mode": leg_parameters.get("pair_mode"),
            "pair_symbol": leg_parameters.get("pair_symbol"),
            "pair_role": leg_parameters.get("pair_role"),
            "hedge_ratio": leg_parameters.get("hedge_ratio"),
            "spread_zscore": leg_parameters.get("spread_zscore"),
            "spread_half_life": leg_parameters.get("spread_half_life"),
            "pair_score": leg_parameters.get("pair_score"),
            "correlation": leg_parameters.get("correlation"),
            "pair_signal": leg_parameters.get("pair_signal"),
            "pair_skip_reason": leg_parameters.get("pair_skip_reason"),
            "skip_reason": leg_parameters.get("skip_reason"),
            "provider": leg_parameters.get("provider"),
            "execution_venue": leg_parameters.get("execution_venue"),
            "trading_connection_id": leg_connection_id,
            "collateral_asset": leg_parameters.get("collateral_asset"),
            "settlement_asset": leg_parameters.get("settlement_asset"),
            "allocation_weight": leg_parameters.get("allocation_weight"),
            "available_margin_usd": leg_parameters.get("available_margin_usd"),
            "market_id": leg_parameters.get("market_id"),
            "venue_symbol": leg_parameters.get("venue_symbol"),
            "provider_symbol": leg_parameters.get("provider_symbol"),
            "app_symbol": leg_parameters.get("app_symbol"),
            "market_status": leg_parameters.get("market_status"),
            "one_h10_scanner_score": leg_parameters.get("one_h10_scanner_score"),
            "one_h10_scanner_source": leg_parameters.get("one_h10_scanner_source"),
            "one_h10_allocation_score": leg_parameters.get("one_h10_allocation_score"),
            "one_h10_allocation_method": leg_parameters.get("one_h10_allocation_method"),
            "allocation_score": leg_parameters.get("one_h10_allocation_score"),
            "scanner_score_breakdown": leg_parameters.get("scanner_score_breakdown"),
            "scanner_features": leg_parameters.get("scanner_features"),
            "one_h10_forecast": leg_parameters.get("one_h10_forecast"),
            "forecast_metadata": leg_parameters.get("forecast_metadata"),
            "forecast_blockers": leg_parameters.get("forecast_blockers"),
            "forecast_advisory_blockers": leg_parameters.get("forecast_advisory_blockers"),
            "forecast_predicted_side": leg_parameters.get("forecast_predicted_side"),
            "forecast_confidence": leg_parameters.get("forecast_confidence"),
            "forecast_expected_return_bps": leg_parameters.get("forecast_expected_return_bps"),
            "forecast_suggested_notional_usd": leg_parameters.get("forecast_suggested_notional_usd"),
            "forecast_suggested_leverage": leg_parameters.get("forecast_suggested_leverage"),
            "forecast_suggested_order_type": leg_parameters.get("forecast_suggested_order_type"),
            "forecast_suggested_stop_loss_pct": leg_parameters.get("forecast_suggested_stop_loss_pct"),
            "forecast_suggested_take_profit_pct": leg_parameters.get("forecast_suggested_take_profit_pct"),
            "forecast_profitability_score": leg_parameters.get("forecast_profitability_score"),
            "forecast_allocation_score": leg_parameters.get("forecast_allocation_score"),
            "forecast_execution_adjusted_net_return_bps": leg_parameters.get("forecast_execution_adjusted_net_return_bps"),
            "ml_horizon": leg_parameters.get("ml_horizon"),
            "one_h10_vault": leg_parameters.get("one_h10_vault"),
            "objective": leg_parameters.get("objective"),
            "ml_objective": leg_parameters.get("ml_objective"),
            "ml_policy_required": leg_parameters.get("ml_policy_required"),
            "ml_governed_risk": leg_parameters.get("ml_governed_risk"),
            "target_roi_pct": leg_parameters.get("target_roi_pct"),
            "target_amount_usd": leg_parameters.get("target_amount_usd"),
        }
        db.session.add(leg_model)
        db.session.flush()
        leg_parameters["vault_leg_id"] = leg_model.id
        run.parameters = leg_parameters
        if index == 0:
            cycle.strategy_run_id = run.id
        run_ids.append(run.id)

    db.session.add(
        WalletTransaction(
            vault_cycle_id=cycle.id,
            user_id=user.id,
            asset=asset,
            amount=amount,
            transaction_type="allocation",
            status="complete",
            note=f"{selection.profile} algorithm cycle started.",
        )
    )
    commit_with_retry()
    if idempotency_key:
        _persist_cycle_start_cycle_idempotency(user_id=user.id, idempotency_key=idempotency_key, cycle_id=cycle.id)

    if async_enabled:
        job_id = _enqueue_cycle_start_job(
            run_ids=run_ids,
            cycle_id=cycle.id,
            user_id=user.id,
            idempotency_key=idempotency_key,
        )
        payload = {
            "ok": True,
            "status": "queued",
            "job_id": job_id,
            "cycle_id": cycle.id,
            "run_ids": run_ids,
        }
        payload = _with_cycle_start_runtime_metadata(payload)
        if _wants_start_json_response():
            return jsonify(payload), 202
        flash("Vault cycle queued. Strategy workers are starting in the background.", "success")
        return redirect(url_for("consumer.vault"))

    _start_strategy_runs(run_ids)
    if _wants_start_json_response():
        return jsonify(
            action_envelope(
                ok=True,
                code="vault_cycle_started",
                message="Vault cycle started.",
                ready=True,
                created=True,
                **_cycle_start_runtime_metadata(cycle_id=cycle.id, run_ids=run_ids, status="started"),
            )
        ), 201
    flash("Vault cycle started.", "success")
    return redirect(url_for("consumer.vault"))


@consumer_bp.post("/vault/cycles")
def create_vault_cycle():
    user = current_user()
    _sync_completed_cycles(user)
    payload = request.get_json(silent=True) if request.is_json else {}
    payload = payload if isinstance(payload, dict) else {}

    def field(name: str, default: object = "") -> object:
        if name in payload:
            return payload.get(name)
        return request.form.get(name, default)

    deposit_asset = str(field("deposit_asset", field("settlement_asset", "USDT")) or "USDT").upper().strip()
    settlement_asset = str(field("settlement_asset", deposit_asset) or deposit_asset).upper().strip()
    try:
        amount = float(field("amount", field("deposit_amount", 0)) or 0)
    except (TypeError, ValueError):
        amount = 0.0
    try:
        duration_seconds = int(float(field("duration_seconds", 0) or 0))
    except (TypeError, ValueError):
        duration_seconds = 0
    if duration_seconds <= 0:
        if request.is_json:
            try:
                duration_hours = float(field("duration_hours", 24) or 24)
            except (TypeError, ValueError):
                duration_hours = 24
            duration_seconds = max(60, int(duration_hours * 3600))
        else:
            duration_seconds = _requested_duration_seconds()
    providers = payload.get("providers") if isinstance(payload.get("providers"), list) else request.form.getlist("providers")
    if not providers:
        providers = [str(item).strip() for item in str(field("providers", "") or "").split(",") if str(item).strip()]
    allowed_symbols = payload.get("allowed_symbols") if isinstance(payload.get("allowed_symbols"), list) else _requested_allowed_symbols()
    try:
        max_leverage = float(field("max_leverage", 0) or 0) or None
    except (TypeError, ValueError):
        max_leverage = None
    try:
        max_positions = int(float(field("max_positions", 0) or 0)) or None
    except (TypeError, ValueError):
        max_positions = None
    idempotency_key = str(field("idempotency_key", _request_idempotency_key()) or "").strip()

    try:
        result = _start_vault_cycle_engine(
            user=user,
            amount=amount,
            deposit_asset=deposit_asset,
            settlement_asset=settlement_asset,
            duration_seconds=duration_seconds,
            providers=[str(provider) for provider in providers],
            allowed_symbols=[str(symbol) for symbol in allowed_symbols],
            max_leverage=max_leverage,
            max_positions=max_positions,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        if _wants_json_response():
            return jsonify(exception_envelope(exc, default_code="vault_cycle_start_failed")), 400
        flash(str(exc), "danger")
        return redirect(url_for("consumer.vault"))

    cycle = result["cycle"]
    created = bool(result.get("created", False))
    if _wants_json_response():
        return jsonify(
            action_envelope(
                ok=True,
                code="vault_cycle_started" if created else "vault_cycle_duplicate",
                message="Vault Cycle started with dynamic exchange allocation."
                if created
                else "Cycle start already submitted. Showing the existing cycle.",
                created=created,
                duplicate=not created,
                **_cycle_start_runtime_metadata(
                    cycle_id=cycle.id,
                    run_ids=result.get("run_ids", []),
                    status="started" if created else "duplicate",
                ),
            )
        ), 201 if created else 200
    flash(
        "Vault Cycle started with dynamic exchange allocation."
        if created
        else "Cycle start already submitted. Showing the existing cycle.",
        "success" if created else "info",
    )
    return redirect(url_for("consumer.cycle_detail", cycle_id=cycle.id))


def _vault_cycle_engine_form_enabled(is_one_h10: bool) -> bool:
    return bool(current_app.config.get("VAULT_CYCLE_ENGINE_ENABLED", False)) and not is_one_h10


def _start_vault_cycle_engine(
    *,
    user,
    amount: float,
    deposit_asset: str,
    settlement_asset: str,
    duration_seconds: int,
    providers: list[str],
    allowed_symbols: list[str],
    max_leverage: float | None = None,
    max_positions: int | None = None,
    idempotency_key: str = "",
) -> dict[str, object]:
    result = get_service("vault_cycle_orchestrator").start_cycle(
        user=user,
        amount=amount,
        deposit_asset=deposit_asset,
        settlement_asset=settlement_asset,
        duration_seconds=duration_seconds,
        providers=[str(provider) for provider in providers],
        allowed_symbols=[str(symbol) for symbol in allowed_symbols],
        max_leverage=max_leverage,
        max_positions=max_positions,
        idempotency_key=idempotency_key,
        start_strategy_runs=False,
    )
    commit_with_retry()
    _start_strategy_runs([int(run_id) for run_id in result.get("run_ids", []) if run_id])
    return result


def _start_vault_cycle_engine_from_route(
    *,
    user,
    amount: float,
    deposit_asset: str,
    settlement_asset: str,
    duration_seconds: int,
    providers: list[str],
    allowed_symbols: list[str],
    idempotency_key: str,
    success_message: str,
    wants_start_response: bool,
):
    try:
        result = _start_vault_cycle_engine(
            user=user,
            amount=amount,
            deposit_asset=deposit_asset,
            settlement_asset=settlement_asset,
            duration_seconds=duration_seconds,
            providers=providers,
            allowed_symbols=allowed_symbols,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        if wants_start_response and _wants_start_json_response():
            return jsonify(exception_envelope(exc, default_code="vault_cycle_start_failed")), 400
        flash(str(exc), "danger")
        return redirect(url_for("consumer.vault"))

    cycle = result["cycle"]
    created = bool(result.get("created", False))
    run_ids = result.get("run_ids", [])
    if wants_start_response and _wants_start_json_response():
        return jsonify(
            action_envelope(
                ok=True,
                code="vault_cycle_started" if created else "vault_cycle_duplicate",
                message=success_message if created else "Cycle start already submitted. Showing the existing cycle.",
                ready=True,
                created=created,
                duplicate=not created,
                **_cycle_start_runtime_metadata(
                    cycle_id=cycle.id,
                    run_ids=run_ids,
                    status="started" if created else "duplicate",
                ),
            )
        ), 201 if created else 200
    flash(success_message if created else "Cycle start already submitted. Showing the existing cycle.", "success" if created else "info")
    return redirect(url_for("consumer.cycle_detail", cycle_id=cycle.id))


def _vault_readiness_payload_from_request(*, source: str) -> dict[str, object]:
    user = current_user()
    if source == "args":
        deposit_asset = str(request.args.get("deposit_asset") or request.args.get("asset") or "USDC").upper().strip()
        settlement_asset = str(request.args.get("settlement_asset") or deposit_asset or "USDC").upper().strip()
        amount = _float_query_arg("amount", 0.0)
        use_max = str(request.args.get("max") or request.args.get("use_max") or "").lower() in {"1", "true", "yes", "on"}
    else:
        deposit_asset = str(_request_value("deposit_asset", _request_value("asset", "USDC")) or "USDC").upper().strip()
        settlement_asset = str(_request_value("settlement_asset", deposit_asset) or deposit_asset).upper().strip()
        try:
            amount = float(_request_value("amount", _request_value("deposit_amount", 0)) or 0)
        except (TypeError, ValueError):
            amount = 0.0
        use_max = str(_request_value("max", _request_value("use_max", ""))).lower() in {"1", "true", "yes", "on"}
    if use_max:
        balance = WalletBalance.query.filter_by(user_id=user.id, asset=deposit_asset).one_or_none()
        verified = _verified_spendable_amount(user, deposit_asset, _asset_networks(deposit_asset)[0])
        amount = verified if verified is not None else (float(balance.available_balance or 0.0) if balance is not None else 0.0)
    providers = _requested_provider_keys(source=source)
    cycle_value = (
        str(request.args.get("cycle") or request.args.get("cycle_type") or "1H10")
        if source == "args"
        else str(_request_value("cycle", _request_value("cycle_type", "1H10")) or "1H10")
    )
    ack_value = (
        str(request.args.get("one_h10_live_ack") or request.args.get("live_acknowledged") or "").strip().lower()
        if source == "args"
        else str(_request_value("one_h10_live_ack", _request_value("live_acknowledged", ""))).strip().lower()
    )
    return get_vault_cycle_readiness(
        user.id,
        cycle=cycle_value.upper(),
        settlement_asset=settlement_asset,
        deposit_asset=deposit_asset,
        amount=amount,
        enabled_exchanges=providers,
        live_acknowledged=ack_value in {"1", "true", "yes", "on", "acknowledged"},
        idempotency_key=_request_idempotency_key(),
        enforce_ml_gate=False,
    )


@consumer_bp.get("/vault/readiness")
@consumer_bp.get("/api/vault/readiness")
def vault_readiness():
    payload = _vault_readiness_payload_from_request(source="args")
    blocker_count = len(list(payload.get("active_blockers") or []))
    payload["message"] = (
        "1H10 vault cycle is ready."
        if payload.get("ready")
        else f"Vault cycle is blocked by {blocker_count} live gate{'s' if blocker_count != 1 else ''}."
    )
    return jsonify(readiness_envelope(payload, code="vault_cycle_readiness", message=str(payload.get("message") or "")))


@consumer_bp.post("/vault/preview-route")
def vault_preview_route():
    payload = _vault_readiness_payload_from_request(source="form")
    blocker_count = len(list(payload.get("active_blockers") or []))
    payload["message"] = (
        "1H10 vault cycle is ready."
        if payload.get("ready")
        else f"Vault cycle is blocked by {blocker_count} live gate{'s' if blocker_count != 1 else ''}."
    )
    status_code = (
        200
        if payload.get("ready")
        or any(item.get("code") == "amount_required" for item in list(payload.get("active_blockers") or []) if isinstance(item, dict))
        else 200
    )
    return jsonify(readiness_envelope(payload, code="vault_cycle_readiness", message=str(payload.get("message") or ""))), status_code


@consumer_bp.get("/api/vault/routing-preview")
def vault_routing_preview():
    user = current_user()
    amount = _float_query_arg("amount", 0.0)
    deposit_asset = str(request.args.get("deposit_asset") or request.args.get("asset") or "USDC").upper().strip()
    settlement_asset = str(request.args.get("settlement_asset") or deposit_asset or "USDC").upper().strip()
    providers = _requested_provider_keys(source="args")
    payload = _vault_routing_preview_payload(
        user=user,
        amount=amount,
        deposit_asset=deposit_asset,
        settlement_asset=settlement_asset,
        providers=providers,
    )
    return jsonify(payload)


@consumer_bp.get("/api/vault/cycles/<int:cycle_id>")
def vault_cycle_status(cycle_id: int):
    user = current_user()
    get_service("vault_cycle_orchestrator").resume_due_cycles(user.id)
    try:
        get_service("vault_cycle_trading_enforcer").enforce_active_cycles(user.id)
    except OperationalError as exc:
        if not is_database_locked(exc):
            raise
        db.session.rollback()
        current_app.logger.warning("Deferred Vault Cycle active trading enforcement because SQLite is locked: %s", exc)
    commit_with_retry()
    cycle = VaultCycle.query.filter_by(id=cycle_id, user_id=user.id).one_or_none()
    if cycle is None:
        return jsonify({"ok": False, "error": "Vault cycle was not found."}), 404
    if cycle.status in {"active", "settling"}:
        _refresh_cycle_performance(cycle)
        commit_with_retry()
    payload = get_service("vault_cycle_reporting").status_payload(cycle)
    orders = _cycle_orders(cycle)
    order_summaries = [_order_summary(order) for order in orders]
    legs = [_leg_summary(leg) for leg in cycle.allocation_legs]
    runtime_notice = _cycle_one_h10_runtime_notice(cycle)
    payload["trade_decision_legs"] = _cycle_trade_decision_legs(cycle, order_summaries, legs)
    payload["trade_decision"] = _cycle_trade_decision(
        cycle,
        order_summaries,
        payload["trade_decision_legs"],
        runtime_notice,
    )
    payload["worker"] = _cycle_worker_status(cycle, payload["trade_decision_legs"])
    payload["live_order_path"] = payload["worker"]["live_order_path"]
    payload["runtime_notice"] = runtime_notice
    payload["ok"] = True
    return jsonify(payload)


@consumer_bp.get("/consumer/start-status/<job_id>")
@consumer_bp.get("/vault/start-status/<job_id>")
def cycle_start_status(job_id: str):
    user = current_user()
    job = _load_cycle_start_job(str(job_id))
    if not job or user is None:
        return jsonify({"ok": False, "error": "job_not_found", "job_id": job_id}), 404
    if int(job.get("user_id") or 0) != int(user.id):
        return jsonify({"ok": False, "error": "job_not_found", "job_id": job_id}), 404
    return jsonify(_with_cycle_start_runtime_metadata(job))


@consumer_bp.get("/activity/", strict_slashes=False)
def activity():
    return redirect(url_for("consumer.home"))


@consumer_bp.get("/vault/cycles/<int:cycle_id>")
def cycle_detail(cycle_id: int):
    user = current_user()
    _sync_completed_cycles(user)
    cycle = VaultCycle.query.filter_by(id=cycle_id, user_id=user.id).one_or_none()
    if cycle is None:
        flash("Vault cycle was not found.", "danger")
        return redirect(url_for("consumer.vault"))
    performance = None
    if cycle.status in {"active", "settling"} and not _vault_live_api_deferred_for_request():
        performance = _refresh_cycle_performance(cycle)
        recovered_run_ids = _recover_active_one_h10_cycles([cycle])
        commit_with_retry()
        _start_strategy_runs(recovered_run_ids)
    if cycle.status in {"active", "settling"} and _vault_live_api_deferred_for_request():
        summary = get_service("vault_cycle_reporting").status_payload(cycle)
    else:
        summary = (
            _cycle_summary(cycle, performance=performance)
            if cycle.status in {"active", "settling"}
            else cycle.cycle_summary or _cycle_summary(cycle)
        )
    summary["chart_payload"] = _cycle_chart_payload(cycle, summary)
    return render_template(
        "cycle_detail.html",
        cycle=cycle,
        summary=summary,
    )


@consumer_bp.get("/dashboard")
def legacy_dashboard():
    return redirect(url_for("dashboard.index"))


@consumer_bp.get("/api/dashboard-data")
def legacy_dashboard_data():
    return redirect(url_for("dashboard.dashboard_data"))


@consumer_bp.route("/backtests/", methods=["GET"], strict_slashes=False)
def legacy_backtests():
    return redirect(url_for("backtests.index"))


@consumer_bp.post("/backtests/run")
def legacy_backtests_run():
    return redirect(url_for("backtests.run"), code=307)


@consumer_bp.post("/backtests/optimize")
def legacy_backtests_optimize():
    return redirect(url_for("backtests.optimize"), code=307)


@consumer_bp.route("/panic/", methods=["GET"], strict_slashes=False)
def legacy_panic():
    abort(404)


@consumer_bp.post("/panic/activate")
def legacy_panic_activate():
    abort(404)


def _wallet_convert_asset_rows(balances: list[WalletBalance]) -> list[dict[str, object]]:
    by_asset = {str(balance.asset or "").upper(): balance for balance in balances}
    rows: list[dict[str, object]] = []
    for asset in _wallet_assets():
        balance = by_asset.get(asset)
        available = max(0.0, _safe_float(getattr(balance, "available_balance", 0.0)))
        locked = max(0.0, _safe_float(getattr(balance, "locked_balance", 0.0)))
        total = available + locked
        price = max(0.0, _asset_usd_price(asset))
        estimated = total * price if price > 0 else max(0.0, _safe_float(getattr(balance, "estimated_usd_value", 0.0)))
        rows.append(
            {
                "asset": asset,
                "available_balance": available,
                "locked_balance": locked,
                "total_balance": total,
                "estimated_usd_value": estimated,
                "price_usd": price,
                "available_usd": available * price if price > 0 else 0.0,
                "price_available": price > 0,
            }
        )
    return rows


def _normalize_convert_asset(value: object, asset_keys: list[str], default: str) -> str:
    asset = str(value or "").strip().upper()
    return asset if asset in set(asset_keys) else default


def _wallet_convert_state(
    form_values: dict[str, object],
    assets: list[dict[str, object]],
    quote: dict[str, object] | None,
    errors: dict[str, str],
) -> dict[str, object]:
    by_asset = {str(row["asset"]): row for row in assets}
    from_asset = str(form_values.get("from_asset") or "").upper()
    to_asset = str(form_values.get("to_asset") or "").upper()
    from_row = by_asset.get(from_asset, {})
    to_row = by_asset.get(to_asset, {})
    funded_assets = [row for row in assets if float(row.get("available_balance") or 0.0) > 0 and bool(row.get("price_available"))]
    priced_assets = [row for row in assets if bool(row.get("price_available"))]
    can_convert = len(funded_assets) > 0 and len(priced_assets) > 1
    amount_value = str(form_values.get("amount") or "").strip()
    return {
        "can_convert": can_convert,
        "funded_count": len(funded_assets),
        "priced_count": len(priced_assets),
        "from_available": float(from_row.get("available_balance") or 0.0),
        "from_available_usd": float(from_row.get("available_usd") or 0.0),
        "from_price": float(from_row.get("price_usd") or 0.0),
        "to_price": float(to_row.get("price_usd") or 0.0),
        "has_amount": bool(amount_value),
        "has_quote": quote is not None and not errors,
        "has_errors": bool(errors),
    }


def _wallet_convert_quote(
    form_values: dict[str, object], assets: list[dict[str, object]]
) -> tuple[dict[str, object] | None, dict[str, str]]:
    errors: dict[str, str] = {}
    by_asset = {str(row["asset"]): row for row in assets}
    from_asset = str(form_values.get("from_asset") or "").upper()
    to_asset = str(form_values.get("to_asset") or "").upper()
    from_row = by_asset.get(from_asset)
    to_row = by_asset.get(to_asset)
    if from_row is None:
        errors["from_asset"] = "Choose a supported source asset."
    if to_row is None:
        errors["to_asset"] = "Choose a supported destination asset."
    if from_asset and to_asset and from_asset == to_asset:
        errors["to_asset"] = "Choose two different assets."

    amount = _positive_decimal(form_values.get("amount"))
    if amount is None:
        errors["amount"] = "Enter an amount greater than zero."
    if errors:
        return None, errors

    from_price = Decimal(str(from_row.get("price_usd") or 0))
    to_price = Decimal(str(to_row.get("price_usd") or 0))
    if from_price <= 0:
        errors["from_asset"] = f"{from_asset} price is unavailable."
    if to_price <= 0:
        errors["to_asset"] = f"{to_asset} price is unavailable."
    available = Decimal(str(from_row.get("available_balance") or 0))
    if amount is not None and amount > available + Decimal("0.000000000001"):
        errors["amount"] = f"Amount exceeds available {from_asset} balance."
    if errors or amount is None:
        return None, errors

    usd_value = amount * from_price
    converted_amount = usd_value / to_price
    if converted_amount <= 0:
        errors["amount"] = "Conversion amount is too small."
        return None, errors
    rate = from_price / to_price
    return (
        {
            "from_asset": from_asset,
            "to_asset": to_asset,
            "from_amount": float(amount),
            "to_amount": float(converted_amount),
            "usd_value": float(usd_value),
            "from_price": float(from_price),
            "to_price": float(to_price),
            "rate": float(rate),
            "fee_usd": 0.0,
        },
        {},
    )


def _execute_wallet_conversion(user, quote: dict[str, object]) -> dict[str, object]:
    from_asset = str(quote.get("from_asset") or "").upper()
    to_asset = str(quote.get("to_asset") or "").upper()
    amount = Decimal(str(quote.get("from_amount") or 0))
    converted_amount = Decimal(str(quote.get("to_amount") or 0))
    if amount <= 0 or converted_amount <= 0:
        raise ValueError("Conversion amount is invalid.")
    source_balance = WalletBalance.query.filter_by(user_id=user.id, asset=from_asset).one_or_none()
    destination_balance = WalletBalance.query.filter_by(user_id=user.id, asset=to_asset).one_or_none()
    if source_balance is None:
        raise ValueError(f"{from_asset} balance is unavailable.")
    if destination_balance is None:
        destination_balance = WalletBalance(user_id=user.id, asset=to_asset, available_balance=0.0, locked_balance=0.0)
        db.session.add(destination_balance)
        db.session.flush()
    source_available = Decimal(str(source_balance.available_balance or 0))
    if amount > source_available + Decimal("0.000000000001"):
        raise ValueError(f"Amount exceeds available {from_asset} balance.")

    source_balance.available_balance = float(max(Decimal("0"), source_available - amount))
    destination_balance.available_balance = float(Decimal(str(destination_balance.available_balance or 0)) + converted_amount)
    _refresh_wallet_balance_estimate(source_balance)
    _refresh_wallet_balance_estimate(destination_balance)

    conversion_id = f"wallet-convert-{uuid.uuid4().hex[:12]}"
    note = (
        f"{conversion_id}: converted {float(amount):.8f} {from_asset} to {float(converted_amount):.8f} {to_asset} "
        f"at ${float(quote.get('from_price') or 0):.8f}/${float(quote.get('to_price') or 0):.8f}; internal ledger conversion."
    )
    db.session.add(
        WalletTransaction(
            user_id=user.id,
            asset=from_asset,
            amount=-float(amount),
            transaction_type="conversion",
            status="complete",
            network="internal",
            note=note,
        )
    )
    db.session.add(
        WalletTransaction(
            user_id=user.id,
            asset=to_asset,
            amount=float(converted_amount),
            transaction_type="conversion",
            status="complete",
            network="internal",
            note=note,
        )
    )
    return {
        "from_asset": from_asset,
        "to_asset": to_asset,
        "from_amount": float(amount),
        "to_amount": float(converted_amount),
    }


def _refresh_wallet_balance_estimate(balance: WalletBalance) -> None:
    price = _asset_usd_price(balance.asset)
    balance.estimated_usd_value = max(0.0, balance.total_balance * price) if price > 0 else 0.0


def _positive_decimal(value: object) -> Decimal | None:
    try:
        amount = Decimal(str(value or "").strip())
    except (InvalidOperation, ValueError):
        return None
    return amount if amount > 0 and amount.is_finite() else None


def _wallet_balances(user) -> list[WalletBalance]:
    _sync_real_wallet_balances(user)
    balances = (
        WalletBalance.query.options(joinedload(WalletBalance.active_deposit_address))
        .filter_by(user_id=user.id)
        .order_by(WalletBalance.asset.asc())
        .all()
    )
    wallet_assets = _wallet_assets()
    if not balances:
        for asset in wallet_assets:
            db.session.add(
                WalletBalance(
                    user_id=user.id,
                    asset=asset,
                    available_balance=0.0,
                    locked_balance=0.0,
                    estimated_usd_value=0.0,
                )
            )
        commit_with_retry()
        balances = (
            WalletBalance.query.options(joinedload(WalletBalance.active_deposit_address))
            .filter_by(user_id=user.id)
            .order_by(WalletBalance.asset.asc())
            .all()
        )
    else:
        existing = {balance.asset for balance in balances}
        for asset in wallet_assets:
            if asset not in existing:
                db.session.add(WalletBalance(user_id=user.id, asset=asset, estimated_usd_value=0.0))
        commit_with_retry()
        balances = (
            WalletBalance.query.options(joinedload(WalletBalance.active_deposit_address))
            .filter_by(user_id=user.id)
            .order_by(WalletBalance.asset.asc())
            .all()
        )

    changed = False
    for balance in balances:
        network = _asset_networks(balance.asset)[0]
        if balance.active_deposit_address_id is None:
            address = _ensure_deposit_address(user.id, balance.asset, network, balance, commit_link=False)
            if address is not None:
                balance.active_deposit_address_id = address.id
                changed = True
        if balance.total_balance <= 0:
            continue
        price = _asset_usd_price(balance.asset)
        if price > 0:
            estimate = balance.total_balance * price
            if not math.isclose(float(balance.estimated_usd_value or 0.0), estimate, rel_tol=1e-9, abs_tol=1e-9):
                balance.estimated_usd_value = estimate
                changed = True
    if changed:
        commit_with_retry()
    return balances


def _wallet_balances_read_only(user) -> list[WalletBalance]:
    return (
        WalletBalance.query.options(joinedload(WalletBalance.active_deposit_address))
        .filter_by(user_id=user.id)
        .order_by(WalletBalance.asset.asc())
        .all()
    )


def _default_vault_asset(balances: list[WalletBalance]) -> str:
    return default_vault_allocation_asset(
        allocation_asset_views(
            balances=balances,
            configured_assets=_configured_wallet_assets(),
            configured_networks=_configured_asset_networks,
        )
    )


def _sync_real_wallet_balances(user) -> None:
    custody = get_service("wallet_custody")
    if not getattr(custody, "enabled", False):
        return
    try:
        custody.sync_user(user.id)
        commit_with_retry()
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Real wallet sync failed closed for user %s: %s", user.id, exc)


def _wallet_activity_page_number() -> int:
    try:
        return max(1, int(request.args.get("activity_page", "1") or 1))
    except (TypeError, ValueError):
        return 1


def _wallet_view_model(wallet_summary, transactions: list[WalletTransaction]) -> dict[str, object]:
    balances = list(getattr(wallet_summary, "balances", []) or [])
    primary_balance = _primary_wallet_balance(balances)
    primary_asset = str(getattr(primary_balance, "asset", "USDC") or "USDC").upper()
    value_parts = [_wallet_balance_value_parts(balance) for balance in balances]
    total_available = sum(part["available_usd"] for part in value_parts)
    total_locked = sum(part["locked_usd"] for part in value_parts)
    checked_count = sum(1 for balance in balances if str(getattr(balance, "onchain_status", "")).lower() == "checked")
    issue_count = sum(1 for balance in balances if _wallet_balance_has_sync_issue(balance))
    panic_locked = bool(Setting.get_json("panic_lock", False))
    has_available_funds = any(_safe_float(getattr(balance, "available_balance", 0.0)) > 0 for balance in balances)
    withdrawals_ready = wallet_withdrawals_enabled(current_app.config) and not panic_locked and has_available_funds
    withdrawal_notice = ""
    if panic_locked:
        withdrawal_notice = "Withdrawals are paused while the safety lock is active."
    elif not has_available_funds:
        withdrawal_notice = "Add verified funds before requesting a withdrawal."
    elif not wallet_withdrawals_enabled(current_app.config):
        withdrawal_notice = "Withdrawals are waiting on server-side safety gates."

    return {
        "portfolio_total": _safe_float(getattr(wallet_summary, "portfolio_total_usd", 0.0)),
        "available_total": total_available,
        "locked_total": total_locked,
        "primary_asset": primary_asset,
        "asset_count": len(balances),
        "checked_count": checked_count,
        "issue_count": issue_count,
        "sync_label": "Review needed" if issue_count else "Verified On-chain",
        "sync_tone": "warning" if issue_count else "success",
        "last_checked_label": _wallet_last_checked_label(balances),
        "withdrawals_ready": withdrawals_ready,
        "withdrawal_notice": withdrawal_notice,
        "primary_deposit_href": url_for("consumer.deposit", asset=primary_asset),
        "primary_withdraw_href": url_for("consumer.withdraw", asset=primary_asset),
        "primary_convert_href": url_for("consumer.convert", from_asset=primary_asset),
        "asset_rows": [_wallet_asset_row(balance) for balance in balances],
        "transaction_rows": [_wallet_transaction_row(item) for item in transactions],
    }


def _primary_wallet_balance(balances: list[object]) -> object | None:
    funded = [
        balance
        for balance in balances
        if _safe_float(getattr(balance, "available_balance", 0.0)) + _safe_float(getattr(balance, "locked_balance", 0.0)) > 0
    ]
    preferred = ("USDC", "USDT", "ETH", "BTC", "SOL", "XRP")
    for asset in preferred:
        match = next((balance for balance in funded if str(getattr(balance, "asset", "")).upper() == asset), None)
        if match is not None:
            return match
    return funded[0] if funded else (balances[0] if balances else None)


def _wallet_asset_row(balance) -> dict[str, object]:
    asset = str(getattr(balance, "asset", "USDC") or "USDC").upper()
    total = _safe_float(getattr(balance, "total_balance", 0.0))
    available = _safe_float(getattr(balance, "available_balance", 0.0))
    locked = _safe_float(getattr(balance, "locked_balance", 0.0))
    wallet_address = getattr(getattr(balance, "active_deposit_address", None), "address", "") or ""
    sync_status = _wallet_sync_status(balance)
    return {
        "balance": balance,
        "asset": asset,
        "total": total,
        "available": available,
        "locked": locked,
        "value_usd": _safe_float(getattr(balance, "estimated_usd_value", 0.0)),
        "total_display": _format_asset_amount(total),
        "available_display": _format_asset_amount(available),
        "locked_display": _format_asset_amount(locked),
        "onchain_display": _format_asset_amount(_safe_float(getattr(balance, "onchain_balance", 0.0)))
        if str(getattr(balance, "onchain_status", "")).lower() == "checked"
        else "Pending",
        "address": wallet_address,
        "address_short": _short_wallet_address(wallet_address),
        "deposit_href": url_for("consumer.deposit", asset=asset),
        "withdraw_href": url_for("consumer.withdraw", asset=asset),
        "convert_href": url_for("consumer.convert", from_asset=asset),
        "status": sync_status,
    }


def _wallet_balance_value_parts(balance) -> dict[str, float]:
    available = _safe_float(getattr(balance, "available_balance", 0.0))
    locked = _safe_float(getattr(balance, "locked_balance", 0.0))
    total = max(0.0, available + locked)
    estimated = max(0.0, _safe_float(getattr(balance, "estimated_usd_value", 0.0)))
    price = (estimated / total) if total > 0 and estimated > 0 else max(0.0, _asset_usd_price(getattr(balance, "asset", "")))
    return {
        "available_usd": max(0.0, available) * price,
        "locked_usd": max(0.0, locked) * price,
    }


def _wallet_transaction_row(item: WalletTransaction) -> dict[str, object]:
    status = str(item.status or "pending").strip()
    status_label = status.replace("_", " ").title()
    transaction_type = str(item.transaction_type or "wallet_event").strip()
    return {
        "item": item,
        "type_label": transaction_type.replace("_", " ").title(),
        "amount_label": f"{_format_asset_amount(_safe_float(item.amount))} {item.asset}",
        "status_label": status_label,
        "status_tone": _wallet_transaction_status_tone(status),
        "status_icon": _wallet_transaction_status_icon(status),
        "meta_label": f"{item.network or item.asset} · {item.created_at.strftime('%b %d, %H:%M') if item.created_at else 'Pending'}",
    }


def _wallet_balance_has_sync_issue(balance) -> bool:
    status = str(getattr(balance, "onchain_status", "") or "").lower()
    mismatch = str(getattr(balance, "onchain_mismatch_status", "") or "").lower()
    sync_stale = bool(getattr(balance, "sync_stale", False))
    return sync_stale or status not in {"checked"} or mismatch in {"deficit_onchain", "surplus_onchain", "unavailable"}


def _wallet_sync_status(balance) -> dict[str, str]:
    status = str(getattr(balance, "onchain_status", "") or "").lower()
    mismatch = str(getattr(balance, "onchain_mismatch_status", "") or "").lower()
    if status == "checked" and mismatch in {"verified", "matched"}:
        return {"label": "On-chain Verified", "tone": "verified", "icon": "OK"}
    if status == "checked" and mismatch == "surplus_onchain":
        return {"label": "On-chain Surplus", "tone": "warning", "icon": "Review"}
    if status == "checked" and mismatch == "deficit_onchain":
        return {"label": "Needs Review", "tone": "danger", "icon": "Hold"}
    if bool(getattr(balance, "sync_stale", False)):
        return {"label": "Sync Pending", "tone": "warning", "icon": "Sync"}
    return {"label": "Verification Pending", "tone": "muted", "icon": "Pending"}


def _wallet_transaction_status_tone(status: str) -> str:
    normalized = status.lower()
    if normalized in {"complete", "confirmed", "settled", "success"}:
        return "success"
    if normalized in {"failed", "rejected", "cancelled", "canceled"}:
        return "danger"
    if normalized in {"pending", "pending_approval", "pending_withdrawal", "submitted", "queued_treasury_solvency"}:
        return "warning"
    return "muted"


def _wallet_transaction_status_icon(status: str) -> str:
    tone = _wallet_transaction_status_tone(status)
    if tone == "success":
        return "OK"
    if tone == "danger":
        return "Hold"
    if tone == "warning":
        return "Pending"
    return "Info"


def _wallet_last_checked_label(balances: list[object]) -> str:
    checked_values = [getattr(balance, "onchain_checked_at", None) for balance in balances if getattr(balance, "onchain_checked_at", None)]
    if not checked_values:
        return "Sync pending"
    latest = max(checked_values)
    if hasattr(latest, "strftime"):
        return f"Checked {latest.strftime('%b %d, %H:%M')}"
    return "Checked recently"


def _short_wallet_address(address: str) -> str:
    text = str(address or "").strip()
    if not text:
        return "Not generated"
    if len(text) <= 18:
        return text
    return f"{text[:10]}...{text[-6:]}"


def _format_asset_amount(value: object) -> str:
    amount = _safe_float(value)
    if abs(amount) < 0.0000005:
        return "0"
    if abs(amount) >= 100:
        return f"{amount:,.2f}".rstrip("0").rstrip(".")
    if abs(amount) >= 1:
        return f"{amount:,.4f}".rstrip("0").rstrip(".")
    return f"{amount:.6f}".rstrip("0").rstrip(".")


def _vault_cycle_page_number() -> int:
    try:
        return max(1, int(request.args.get("cycle_page", "1") or 1))
    except (TypeError, ValueError):
        return 1


def _wallet_balance_for(user, asset: str) -> WalletBalance:
    _wallet_balances(user)
    balance = WalletBalance.query.filter_by(user_id=user.id, asset=asset).one_or_none()
    if balance is None:
        balance = WalletBalance(user_id=user.id, asset=asset)
        db.session.add(balance)
        commit_with_retry()
    return balance


def _max_withdrawal_amount(user, asset: str, network: str) -> float:
    balance = WalletBalance.query.filter_by(user_id=user.id, asset=asset).one_or_none()
    app_balance = float(balance.available_balance or 0.0) if balance is not None else 0.0
    if not use_real_addresses(current_app.config):
        return app_balance
    verified = _verified_spendable_amount(user, asset, network)
    if verified is None:
        return app_balance
    return verified


def _verified_spendable_amount(user, asset: str, network: str) -> float | None:
    try:
        custody = get_service("wallet_custody")
        if not getattr(custody, "enabled", False) or not custody.supports(asset, network):
            return None
        return float(custody.verified_spendable_amount(user.id, asset, network) or 0.0)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Verified on-chain spendable check failed closed for %s/%s: %s", asset, network, exc)
        return 0.0


def _materialize_onchain_surplus_for_operation(user, asset: str, network: str, amount: float) -> dict[str, object] | None:
    custody = get_service("wallet_custody")
    if not getattr(custody, "enabled", False) or not custody.supports(asset, network):
        return None
    return custody.materialize_onchain_surplus(user.id, asset, network, amount)


def _decimal_amount(value: float) -> str:
    return f"{float(value or 0.0):.6f}".rstrip("0").rstrip(".")


def _portfolio_total(balances: list[WalletBalance]) -> float:
    return sum(float(balance.estimated_usd_value or 0.0) for balance in balances)


def _wallet_allocation_payload(wallet_summary) -> dict[str, object]:
    total = max(float(wallet_summary.portfolio_total_usd or 0.0), 0.0)
    rows: list[dict[str, object]] = []
    palette = ("#38bdf8", "#34d399", "#f59e0b", "#a78bfa", "#f43f5e", "#facc15", "#818cf8")
    for balance in wallet_summary.balances:
        value = max(float(balance.estimated_usd_value or 0.0), 0.0)
        rows.append(
            {
                "asset": balance.asset,
                "value": round(value, 2),
                "pct": round((value / total) * 100.0, 2) if total > 0 else 0.0,
                "total_balance": round(float(balance.total_balance or 0.0), 8),
                "available_balance": round(float(balance.available_balance or 0.0), 8),
                "locked_balance": round(float(balance.locked_balance or 0.0), 8),
            }
        )
    rows = sorted(rows, key=lambda item: float(item["value"]), reverse=True)
    gradient_segments: list[str] = []
    cursor = 0.0
    positive_rows = [row for row in rows if float(row["pct"]) > 0]
    for index, row in enumerate(rows):
        color = palette[index % len(palette)]
        row["color"] = color
        pct = max(float(row["pct"]), 0.0)
        if pct <= 0:
            continue
        end = 100.0 if index == len(positive_rows) - 1 else min(100.0, cursor + pct)
        gradient_segments.append(f"{color} {cursor:.2f}% {end:.2f}%")
        cursor = end
    return {
        "total": round(total, 2),
        "rows": rows,
        "gradient": ", ".join(gradient_segments) if gradient_segments else "rgba(148, 163, 184, 0.18) 0% 100%",
        "summary": "Vault allocation is weighted by current estimated wallet value."
        if total > 0
        else "No wallet balances are available yet.",
        "empty": total <= 0,
    }


def _portfolio_trend_payload(user, wallet_summary) -> dict[str, object]:
    cycles = VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.started_at.desc()).limit(8).all()
    points: list[dict[str, object]] = []
    for cycle in reversed(cycles):
        starting = max(float(cycle.starting_value_usd or 0.0), 0.0)
        current = max(
            float(cycle.current_estimated_value_usd or 0.0)
            or float(cycle.cycle_summary.get("final_settlement_value_usd", 0.0) or 0.0)
            or starting,
            0.0,
        )
        if cycle.started_at and starting > 0:
            points.append({"t": cycle.started_at.isoformat(), "value": round(starting, 2)})
        end_time = cycle.settled_at or cycle.updated_at or cycle.unlocks_at or cycle.started_at
        if end_time and current > 0:
            points.append({"t": end_time.isoformat(), "value": round(current, 2)})
    if len(points) == 1:
        current_total = max(float(wallet_summary.portfolio_total_usd or 0.0), float(points[0]["value"]))
        points.append({"t": datetime.utcnow().isoformat(), "value": round(current_total, 2)})
    return {
        "points": points[-16:],
        "summary": "Recent vault value path from started and settled cycle snapshots."
        if len(points) >= 2
        else "Portfolio trend appears after vault cycles generate value snapshots.",
        "empty": len(points) < 2,
    }


def _home_wallet_balance_payload(wallet_summary, error: str = "") -> dict[str, object]:
    if error or wallet_summary is None:
        return {
            "total_usd": 0.0,
            "state": "error",
            "error": error or "Total wallet balance is temporarily unavailable.",
            "sync_label": "Balance service unavailable",
        }

    total = max(_safe_float(getattr(wallet_summary, "portfolio_total_usd", 0.0)), 0.0)
    snapshot = getattr(wallet_summary, "cached_exchange_snapshot", {}) or {}
    sync_label = _home_sync_label(snapshot.get("synced_at") if isinstance(snapshot, dict) else "")
    return {
        "total_usd": round(total, 2),
        "state": "empty" if total <= 0 else "ready",
        "error": "",
        "sync_label": sync_label,
        "warning": "",
    }


def _home_sync_label(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Local wallet ledger"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return "Latest wallet snapshot"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return f"Updated {parsed.strftime('%b %d, %H:%M')} UTC"


def _home_account_pnl_payload(user) -> dict[str, object]:
    try:
        points, source = _home_fill_pnl_points(user)
        if len(points) < 2:
            points, source = _home_cycle_pnl_points(user)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Home account PnL unavailable: %s", exc)
        return {
            "points": [],
            "total_pnl": 0.0,
            "state": "error",
            "tone": "danger",
            "source": "unavailable",
            "summary": "",
            "error": "Past account P&L is temporarily unavailable.",
            "empty": False,
        }

    total_pnl = round(_safe_float(points[-1]["value"]) if points else 0.0, 2)
    empty = len(points) < 2
    return {
        "points": points,
        "total_pnl": total_pnl,
        "state": "empty" if empty else "ready",
        "tone": _money_tone(total_pnl),
        "source": source,
        "summary": _home_pnl_summary(source, empty),
        "error": "",
        "empty": empty,
    }


def _home_fill_pnl_points(user) -> tuple[list[dict[str, object]], str]:
    mode = get_current_mode()
    fills = (
        Fill.query.join(Fill.order)
        .filter(Order.user_id == user.id)
        .filter(Order.mode == mode)
        .filter(Fill.simulated == (mode == "paper"))
        .filter(Fill.realized_pnl_known.is_(True))
        .order_by(Fill.fill_time.desc(), Fill.id.desc())
        .limit(120)
        .all()
    )
    fills = list(reversed(fills))
    if not fills:
        return [], "trade_fills"

    first_time = fills[0].fill_time or datetime.utcnow()
    points = [_home_pnl_point(first_time - timedelta(minutes=1), 0.0, 0.0)]
    running = 0.0
    for fill in fills:
        delta = _safe_float(fill.pnl) - _safe_float(fill.fee) - _safe_float(getattr(fill, "funding_fee", 0.0))
        running += delta
        points.append(_home_pnl_point(fill.fill_time or datetime.utcnow(), running, delta))
    return points[-121:], "trade_fills"


def _home_cycle_pnl_points(user) -> tuple[list[dict[str, object]], str]:
    cycles = VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.started_at.desc(), VaultCycle.id.desc()).limit(80).all()
    cycles = sorted(cycles, key=lambda cycle: cycle.started_at or cycle.created_at or datetime.utcnow())
    if not cycles:
        return [], "vault_cycles"

    first_time = cycles[0].started_at or cycles[0].created_at or datetime.utcnow()
    points = [_home_pnl_point(first_time - timedelta(minutes=1), 0.0, 0.0)]
    running = 0.0
    for cycle in cycles:
        delta = _cycle_pnl_value(cycle)
        running += delta
        timestamp = cycle.settled_at or cycle.updated_at or cycle.unlocks_at or cycle.started_at or datetime.utcnow()
        points.append(_home_pnl_point(timestamp, running, delta))
    return points[-81:], "vault_cycles"


def _home_pnl_point(timestamp: datetime, value: float, pnl: float) -> dict[str, object]:
    return {
        "label": timestamp.strftime("%b %d"),
        "timestamp": timestamp.isoformat(),
        "value": round(_safe_float(value), 2),
        "pnl": round(_safe_float(pnl), 2),
    }


def _home_pnl_summary(source: str, empty: bool) -> str:
    if empty:
        return "No account P&L history yet."
    if source == "trade_fills":
        return "Realized account P&L from reconciled fills."
    return "Account P&L from vault cycle snapshots."


def _live_connection_required() -> bool:
    return bool(current_app.config.get("ENABLE_LIVE_TRADING", False)) and get_current_mode() == "live"


def _active_trading_connection(user) -> TradingConnection | None:
    return get_service("trading_connections").active_tradable_connection(user.id)


def _enabled_provider_states(user) -> list[dict[str, object]]:
    service = get_service("trading_connections")
    if hasattr(service, "enabled_tradable_connections"):
        connections = service.enabled_tradable_connections(user.id)
    else:
        connections = service.verified_tradable_connections(user.id)
    return [
        {
            "connection_id": connection.id,
            "provider": connection.provider,
            "verification_status": connection.verification_status,
            "is_active": bool(connection.is_active),
        }
        for connection in connections
    ]


def _home_command_center_payload(
    user, wallet_summary, balances: list[WalletBalance], enabled_provider_states: list[dict[str, object]]
) -> dict[str, object]:
    """Build the read-only mobile command center from existing product data."""

    now = datetime.utcnow()
    active_cycles = _active_cycles(user, refresh=False)
    cycle_page = get_service("vault_activity").page_for_user(user.id, page=1)
    activity_page = get_service("wallet_activity").page_for_user(user.id, page=1)
    recent_cycles = _dedupe_cycles([*active_cycles, *list(cycle_page.items or [])])
    strategy_runs = StrategyRun.query.order_by(StrategyRun.created_at.desc()).limit(8).all()
    strategy_rankings = (
        StrategyRanking.query.order_by(
            StrategyRanking.score.desc(),
            StrategyRanking.created_at.desc(),
        )
        .limit(8)
        .all()
    )
    risk_status = _home_risk_status(user)
    market_summary = _home_market_summary()
    portfolio_total = float(getattr(wallet_summary, "portfolio_total_usd", 0.0) or 0.0)
    active_value = sum(float(cycle.current_estimated_value_usd or cycle.starting_value_usd or 0.0) for cycle in active_cycles)
    exposure_pct = (active_value / portfolio_total * 100.0) if portfolio_total > 0 else 0.0
    blocked = bool(risk_status.get("panic_lock") or risk_status.get("live_trading_blocked"))

    return {
        "generated_at": now.isoformat(),
        "mode": get_current_mode(),
        "vault_pulse": {
            "balance_usd": portfolio_total,
            "active_cycles": len(active_cycles),
            "enabled_providers": sum(1 for item in enabled_provider_states if item.get("is_active")),
            "provider_total": len(enabled_provider_states),
            "risk_state": "Blocked" if blocked else "Monitoring",
            "risk_detail": _home_risk_detail(risk_status),
            "market_exposure_pct": exposure_pct,
            "active_value_usd": active_value,
        },
        "pnl": _home_pnl_cards(recent_cycles, now),
        "performance": _home_performance_points(recent_cycles, portfolio_total),
        "allocation": _home_allocation_items(balances, portfolio_total),
        "bots": _home_strategy_cards(strategy_rankings, strategy_runs),
        "bot_summary": _home_bot_summary(strategy_rankings, strategy_runs),
        "markets": market_summary,
        "insights": _home_insights(recent_cycles, active_cycles, risk_status, exposure_pct),
        "activity": _home_activity_items(activity_page.items, cycle_page.items),
    }


def _dedupe_cycles(cycles: list[VaultCycle]) -> list[VaultCycle]:
    seen: set[int] = set()
    unique: list[VaultCycle] = []
    for cycle in cycles:
        if cycle.id in seen:
            continue
        seen.add(cycle.id)
        unique.append(cycle)
    return unique


def _home_risk_status(user) -> dict[str, object]:
    try:
        mode = get_current_mode()
        connection = _active_trading_connection(user)
        status = get_service("risk_engine").status(
            mode,
            user_id=user.id if user else None,
            trading_connection_id=connection.id if connection else None,
        )
        return dict(status or {})
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Home risk status unavailable: %s", exc)
        return {"live_trading_blocked": True, "panic_lock": False, "reason": "Risk status unavailable"}


def _home_risk_detail(risk_status: dict[str, object]) -> str:
    if risk_status.get("panic_lock"):
        return "Safety lock active"
    if risk_status.get("live_trading_blocked"):
        return "Live gates require review"
    daily_pnl = _safe_float(risk_status.get("daily_realized_pnl"))
    daily_limit = _safe_float(risk_status.get("daily_loss_limit"))
    if daily_limit > 0:
        return f"Daily P&L {daily_pnl:.2f} / limit {daily_limit:.2f}"
    return "Risk gates nominal"


def _home_market_summary() -> list[dict[str, object]]:
    try:
        config = current_app.config
        symbols = config.get("ALLOWED_SYMBOLS", ["BTC", "ETH", "SOL", "XRP"])
        timeframe = config.get("DEFAULT_TIMEFRAME", "15m")
        market_mode = market_mode_for(get_current_mode())
        rows = get_service("market_data").get_dashboard_market_summary(symbols, timeframe, market_mode)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Home market summary unavailable: %s", exc)
        rows = []
    payload = []
    for row in list(rows or [])[:8]:
        payload.append(
            {
                "symbol": str(row.get("symbol") or "N/A"),
                "price": _safe_float(row.get("mid")),
                "change_pct": _safe_float(row.get("change_pct")),
                "volume": _safe_float(row.get("volume")),
                "status": str(row.get("status") or "live").replace("_", " ").title(),
            }
        )
    return payload


def _home_pnl_cards(cycles: list[VaultCycle], now: datetime) -> list[dict[str, object]]:
    windows = [
        ("Daily", timedelta(days=1)),
        ("Weekly", timedelta(days=7)),
        ("Monthly", timedelta(days=30)),
    ]
    cards = []
    for label, delta in windows:
        cutoff = now - delta
        value = sum(_cycle_pnl_value(cycle) for cycle in cycles if (cycle.started_at or now) >= cutoff)
        cards.append({"label": label, "value": value, "tone": _money_tone(value)})
    return cards


def _home_performance_points(cycles: list[VaultCycle], portfolio_total: float) -> list[dict[str, object]]:
    sorted_cycles = sorted(cycles, key=lambda cycle: cycle.started_at or datetime.utcnow())[-12:]
    points = []
    running_value = max(0.0, float(portfolio_total or 0.0) - sum(_cycle_pnl_value(cycle) for cycle in sorted_cycles))
    for cycle in sorted_cycles:
        running_value += _cycle_pnl_value(cycle)
        started_at = cycle.started_at or datetime.utcnow()
        points.append(
            {
                "label": started_at.strftime("%b %d"),
                "timestamp": started_at.isoformat(),
                "value": running_value,
                "pnl": _cycle_pnl_value(cycle),
            }
        )
    return points


def _home_allocation_items(balances: list[WalletBalance], portfolio_total: float) -> list[dict[str, object]]:
    items = []
    for balance in sorted(balances, key=lambda item: float(item.estimated_usd_value or 0.0), reverse=True)[:6]:
        value = float(balance.estimated_usd_value or 0.0)
        pct = (value / portfolio_total * 100.0) if portfolio_total > 0 else 0.0
        items.append(
            {
                "asset": balance.asset,
                "value_usd": value,
                "available": float(balance.available_balance or 0.0),
                "locked": float(balance.locked_balance or 0.0),
                "allocation_pct": pct,
            }
        )
    return items


def _home_strategy_cards(rankings: list[StrategyRanking], strategy_runs: list[StrategyRun]) -> list[dict[str, object]]:
    latest_runs: dict[tuple[str, str, str], StrategyRun] = {}
    for run in strategy_runs:
        key = (str(run.strategy_name), str(run.symbol), str(run.timeframe))
        latest_runs.setdefault(key, run)
    cards = []
    for ranking in rankings:
        key = (str(ranking.strategy_name), str(ranking.symbol), str(ranking.timeframe))
        run = latest_runs.get(key)
        last_signal = dict(run.last_signal or {}) if run is not None else {}
        rejected = bool(ranking.rejected)
        status = "Warning" if rejected else (str(run.status).replace("_", " ").title() if run is not None else "Monitoring")
        cards.append(
            {
                "name": ranking.strategy_name.replace("_", " ").title(),
                "status": status,
                "status_tone": "warning" if rejected else _status_tone(status),
                "pair": ranking.symbol,
                "timeframe": ranking.timeframe,
                "provider": ranking.provider.title(),
                "pnl_pct": _safe_float(ranking.recent_performance_score) * 100.0,
                "win_rate": _safe_float(ranking.win_rate) * 100.0,
                "drawdown": _safe_float(ranking.max_drawdown) * 100.0,
                "score": _safe_float(ranking.score),
                "risk_mode": ranking.risk_label or ranking.profile.replace("_", " ").title(),
                "last_action": str(last_signal.get("action") or ("Rejected" if rejected else "Monitoring")).replace("_", " ").title(),
                "detail": ranking.rejection_reason or "; ".join(ranking.warnings[:2]) or "Read-only automation candidate.",
            }
        )
    return cards


def _home_bot_summary(rankings: list[StrategyRanking], strategy_runs: list[StrategyRun]) -> dict[str, int]:
    running = sum(1 for run in strategy_runs if str(run.status).lower() in {"running", "active", "live", "monitoring"})
    warning = sum(1 for ranking in rankings if ranking.rejected or ranking.warnings)
    return {
        "running": running,
        "monitoring": max(0, len(rankings) - warning),
        "warning": warning,
        "paused": sum(1 for run in strategy_runs if str(run.status).lower() in {"paused", "disabled"}),
    }


def _home_insights(
    cycles: list[VaultCycle], active_cycles: list[VaultCycle], risk_status: dict[str, object], exposure_pct: float
) -> dict[str, object]:
    pnl_values = [_cycle_pnl_value(cycle) for cycle in cycles]
    drawdown = min(pnl_values) if pnl_values else 0.0
    wins = sum(1 for value in pnl_values if value > 0)
    completed = len([cycle for cycle in cycles if cycle.status not in {"active", "settling"}])
    confidence = 100.0
    if risk_status.get("panic_lock") or risk_status.get("live_trading_blocked"):
        confidence = 42.0
    elif exposure_pct > 80:
        confidence = 68.0
    elif active_cycles:
        confidence = 78.0
    return {
        "drawdown_usd": drawdown,
        "active_exposure_pct": exposure_pct,
        "automation_confidence": confidence,
        "completed_cycles": completed,
        "win_rate": (wins / len(pnl_values) * 100.0) if pnl_values else 0.0,
    }


def _home_activity_items(transactions: list[WalletTransaction], cycles: list[VaultCycle]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item in list(transactions or [])[:5]:
        items.append(
            {
                "title": str(item.transaction_type or "wallet").replace("_", " ").title(),
                "detail": f"{_safe_float(item.amount):.6f} {item.asset}",
                "status": str(item.status or "recorded").replace("_", " ").title(),
                "timestamp": item.created_at.isoformat() if item.created_at else "",
                "label": item.created_at.strftime("%b %d, %H:%M") if item.created_at else "Recent",
                "tone": "success" if _safe_float(item.amount) >= 0 else "neutral",
            }
        )
    for cycle in list(cycles or [])[:5]:
        items.append(
            {
                "title": f"Vault cycle {str(cycle.execution_substatus or cycle.status).replace('_', ' ')}",
                "detail": f"{_safe_float(cycle.deposit_amount):.6f} {cycle.deposit_asset} · {_safe_float(cycle.current_estimated_value_usd):.2f} USD",
                "status": str(cycle.status or "cycle").replace("_", " ").title(),
                "timestamp": cycle.started_at.isoformat() if cycle.started_at else "",
                "label": cycle.started_at.strftime("%b %d, %H:%M") if cycle.started_at else "Recent",
                "tone": _money_tone(_cycle_pnl_value(cycle)),
            }
        )
    items.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    return items[:6]


def _cycle_pnl_value(cycle: VaultCycle) -> float:
    metadata = cycle.selection_metadata
    if "total_pnl_usd" in metadata:
        return _safe_float(metadata.get("total_pnl_usd"))
    return _safe_float(cycle.current_estimated_value_usd) - _safe_float(cycle.starting_value_usd)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _money_tone(value: float) -> str:
    if value > 0:
        return "success"
    if value < 0:
        return "danger"
    return "neutral"


def _status_tone(status: str) -> str:
    normalized = str(status or "").lower()
    if any(token in normalized for token in ("error", "blocked", "failed", "warning")):
        return "danger" if "error" in normalized or "failed" in normalized else "warning"
    if any(token in normalized for token in ("running", "active", "live", "ready")):
        return "success"
    return "neutral"


def _vault_cycle_options() -> list[dict[str, object]]:
    horizon_seconds = _one_h10_horizon_seconds()
    return [
        {
            "key": "one_h10",
            "label": "1H10",
            "duration_hours": max(1, math.ceil(horizon_seconds / 3600)),
            "duration_seconds": horizon_seconds,
            "summary": "1 hour / 10x target objective",
            "enabled": True,
        }
    ]


def _vault_provider_options() -> list[dict[str, str]]:
    return [
        {
            "provider": provider,
            "label": VAULT_PROVIDER_LABELS.get(provider, provider.title()),
            "short_label": "".join(part[:1] for part in VAULT_PROVIDER_LABELS.get(provider, provider).replace("-", " ").split()).upper()[
                :3
            ],
        }
        for provider in VAULT_UI_PROVIDERS
    ]


def _float_query_arg(name: str, default: float = 0.0) -> float:
    try:
        return max(0.0, float(request.args.get(name, default) or default))
    except (TypeError, ValueError):
        return default


def _request_json_payload() -> dict[str, object]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _request_value(name: str, default: object = "", *, source: str = "form") -> object:
    if source == "args":
        return request.args.get(name, default)
    payload = _request_json_payload()
    if name in payload:
        return payload.get(name, default)
    return request.form.get(name, default)


def _request_values(name: str, *, source: str = "form") -> list[object]:
    if source == "args":
        return list(request.args.getlist(name))
    payload = _request_json_payload()
    value = payload.get(name)
    values: list[object] = []
    if isinstance(value, list):
        values.extend(value)
    elif value is not None:
        values.append(value)
    values.extend(request.form.getlist(name))
    return values


def _requested_provider_keys(source: str = "form") -> list[str]:
    values: list[str] = []
    if source == "args":
        values.extend(request.args.getlist("providers"))
        values.append(request.args.get("providers", ""))
    else:
        values.extend(_request_values("providers", source=source))
        values.append(_request_value("providers", "", source=source))

    providers: list[str] = []
    for raw in values:
        for item in str(raw or "").split(","):
            provider = normalize_provider(item)
            if provider:
                providers.append(provider)

    supported = set(VAULT_UI_PROVIDERS)
    selected = [provider for provider in dict.fromkeys(providers) if provider in supported]
    if source == "form" and _request_value("providers_submitted", "") and not selected:
        return []
    return selected or list(VAULT_UI_PROVIDERS)


def _deferred_live_api_routing_preview_payload(
    *,
    amount: float,
    deposit_asset: str,
    settlement_asset: str,
    providers: list[str] | None = None,
) -> dict[str, object]:
    horizon_seconds = _one_h10_horizon_seconds()
    objective = _one_h10_objective_payload()
    requested = [provider for provider in dict.fromkeys(providers or list(VAULT_UI_PROVIDERS)) if provider in VAULT_UI_PROVIDERS]
    requested = requested or list(VAULT_UI_PROVIDERS)
    amount = max(0.0, float(amount or 0.0))
    provider_rows = [
        {
            "provider": option["provider"],
            "label": option["label"],
            "short_label": option["short_label"],
            "enabled": option["provider"] in requested,
            "connected": False,
            "can_trade": False,
            "collateral_asset": provider_collateral_asset(option["provider"]),
            "available_margin_usd": 0.0,
            "allocation_weight": 0.0,
            "allocation_pct": 0.0,
            "target_amount": 0.0,
            "notional_usd": 0.0,
            "routing_score": 0.0,
            "score": 0.0,
            "status": "checking" if option["provider"] in requested else "disabled",
            "blockers": [],
            "warnings": [],
        }
        for option in _vault_provider_options()
    ]
    active_blockers = (
        [
            {
                "code": "amount_required",
                "title": "Amount required",
                "description": "Enter an amount greater than 0 before starting a 1H10 vault cycle.",
                "severity": "blocker",
                "fix_hint": "Use MAX or enter an amount within your available balance.",
            }
        ]
        if amount <= 0
        else []
    )
    return {
        "ok": False,
        "ready": False,
        "mode": "delegated_live_api",
        "state_label": "Amount Required" if amount <= 0 else "Checking Live API",
        "cycle": {
            "type": "one_h10",
            "label": "1H10",
            "duration_seconds": horizon_seconds,
            "duration_label": "1 hour",
        },
        "objective": objective,
        "providers": provider_rows,
        "summary": {
            "amount": amount,
            "deposit_asset": deposit_asset,
            "settlement_asset": settlement_asset,
            "allocation_engine": "1H10 Smart Router",
            "selected_provider_count": len(requested),
            "ready_provider_count": 0,
            "allocated_total": 0.0,
            "notional_usd": 0.0,
            "total_free_margin_usd": 0.0,
            "ml_readiness": {"display_status": "Live API check pending"},
            "routing_summary": "Enter amount to generate route." if amount <= 0 else "Checking live API route.",
        },
        "blockers": active_blockers,
        "active_blockers": active_blockers,
        "exchange_blockers": [],
        "hard_blockers": active_blockers,
        "advisory_blockers": [],
        "clearable_blockers": active_blockers,
        "can_start": False,
        "warnings": [],
        "exchange_status": {},
        "routing_preview": {
            "notional_usd": 0.0,
            "routes": [],
            "summary": "Enter amount to generate route." if amount <= 0 else "Checking live API route.",
        },
        "ready_exchange_count": 0,
        "total_exchange_count": len(requested),
        "live_api_origin": _public_live_api_origin(),
        "live_api_deferred": True,
    }


def _vault_routing_preview_payload(
    *,
    user,
    amount: float,
    deposit_asset: str,
    settlement_asset: str,
    providers: list[str] | None = None,
) -> dict[str, object]:
    horizon_seconds = _one_h10_horizon_seconds()
    requested = [provider for provider in dict.fromkeys(providers or list(VAULT_UI_PROVIDERS)) if provider in VAULT_UI_PROVIDERS]
    requested = requested or list(VAULT_UI_PROVIDERS)
    amount = max(0.0, float(amount or 0.0))
    readiness = get_vault_cycle_readiness(
        user.id if user is not None else None,
        cycle="1H10",
        settlement_asset=settlement_asset,
        deposit_asset=deposit_asset,
        amount=amount,
        enabled_exchanges=requested,
        live_acknowledged=_one_h10_live_acknowledged(),
        enforce_ml_gate=False,
    )
    exchange_status = {
        str(key).lower(): value for key, value in dict(readiness.get("exchange_status") or {}).items() if isinstance(value, dict)
    }
    rows: list[dict[str, object]] = []
    for option in _vault_provider_options():
        provider = str(option["provider"])
        state = exchange_status.get(provider, {})
        enabled = bool(state.get("enabled", provider in requested))
        blockers = list(state.get("blockers") or [])
        connected = bool(state.get("connected", False))
        status = str(state.get("status") or ("ready" if state.get("ready") else "disabled" if not enabled else "blocked"))
        rows.append(
            {
                "provider": provider,
                "label": option["label"],
                "short_label": option["short_label"],
                "enabled": enabled,
                "eligible": bool(state.get("eligible", state.get("ready", False))),
                "ready": bool(state.get("ready", False)),
                "connected": connected,
                "verified": bool(state.get("verified", False)),
                "can_trade": bool(state.get("can_trade", state.get("ready", False))),
                "collateral_asset": state.get("collateral_asset") or provider_collateral_asset(provider),
                "available_margin_usd": _one_h10_float(state.get("available_margin_usd"), 0.0),
                "allocation_weight": _one_h10_float(state.get("allocation_weight"), 0.0),
                "allocation_pct": _one_h10_float(state.get("allocation_pct"), 0.0),
                "target_amount": _one_h10_float(state.get("target_amount"), 0.0),
                "notional_usd": _one_h10_float(state.get("notional_usd"), 0.0),
                "routing_score": _one_h10_float(state.get("score"), 0.0) / 100.0,
                "score": _one_h10_float(state.get("score"), 0.0),
                "status": status,
                "readiness_state": str(state.get("readiness_state") or status),
                "funding_status": str(state.get("funding_status") or ""),
                "funding_label": str(state.get("funding_label") or ""),
                "funding_detail": str(state.get("funding_detail") or ""),
                "blockers": blockers if blockers else ([] if connected else ["connection_not_verified"]),
                "warnings": list(state.get("warnings") or []),
            }
        )

    summary = dict(readiness.get("routing_preview") or {})
    allocated_total = sum(float(row.get("target_amount") or 0.0) for row in rows if bool(row.get("enabled")))
    blockers = list(readiness.get("active_blockers") or [])

    return {
        "ok": True,
        "ready": bool(readiness.get("ready", False)),
        "mode": readiness.get("mode", "blocked"),
        "state_label": readiness.get("state_label", "Blocked"),
        "cycle": {
            "type": "one_h10",
            "label": "1H10",
            "duration_seconds": horizon_seconds,
            "duration_label": "1 hour",
        },
        "objective": readiness.get("objective", _one_h10_objective_payload()),
        "providers": rows,
        "summary": {
            "amount": amount,
            "deposit_asset": deposit_asset,
            "settlement_asset": settlement_asset,
            "allocation_engine": "1H10 Smart Router",
            "selected_provider_count": len(requested),
            "ready_provider_count": int(readiness.get("ready_exchange_count", 0) or 0),
            "allocated_total": allocated_total,
            "notional_usd": float(readiness.get("notional_usd", 0.0) or 0.0),
            "total_free_margin_usd": sum(float(row.get("available_margin_usd") or 0.0) for row in rows if bool(row.get("can_trade"))),
            "ml_readiness": readiness.get("ml_readiness", {}),
            "routing_summary": summary.get("summary", ""),
        },
        "blockers": blockers,
        "active_blockers": readiness.get("active_blockers", []),
        "exchange_blockers": readiness.get("exchange_blockers", []),
        "hard_blockers": readiness.get("hard_blockers", []),
        "advisory_blockers": readiness.get("advisory_blockers", []),
        "clearable_blockers": readiness.get("clearable_blockers", []),
        "can_start": bool(readiness.get("can_start", readiness.get("ready", False))),
        "warnings": readiness.get("warnings", []),
        "exchange_status": readiness.get("exchange_status", {}),
        "routing_preview": readiness.get("routing_preview", {}),
    }


def _is_one_h10_duration(duration_seconds: int, duration_hours: int) -> bool:
    horizon_seconds = _one_h10_horizon_seconds()
    return int(duration_seconds or 0) == horizon_seconds and int(duration_hours or 0) == max(1, math.ceil(horizon_seconds / 3600))


def _one_h10_horizon_seconds() -> int:
    return max(60, int(current_app.config.get("ONE_H10_HORIZON_SECONDS", ONE_H10_HORIZON_SECONDS) or ONE_H10_HORIZON_SECONDS))


def _one_h10_objective_payload() -> dict[str, object]:
    target_roi_pct = _one_h10_float(current_app.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0), 1000.0)
    return {
        "name": "1 hour / 10x target",
        "profile": "1H10",
        "target_multiplier": max(1.0, target_roi_pct / 100.0),
        "target_roi_pct": target_roi_pct,
        "horizon_seconds": _one_h10_horizon_seconds(),
        "horizon_label": "1 hour",
        "disclaimer": "Optimization target only. Live execution remains risk-gated.",
    }


def _one_h10_live_acknowledged() -> bool:
    value = str(_request_value("one_h10_live_ack", "")).strip().lower()
    if request.method == "GET":
        value = str(request.args.get("one_h10_live_ack", value)).strip().lower()
    return value in {"1", "true", "yes", "on", "acknowledged"}


def _one_h10_live_start_block_reason() -> str | None:
    if not _live_connection_required():
        return None
    if not bool(current_app.config.get("ONE_H10_LIVE_ENABLED", False)):
        return "1H10 live execution is disabled. Set ONE_H10_LIVE_ENABLED=true after paper/backtest validation."
    if not bool(current_app.config.get("EXPLICIT_LIVE_CONFIRMED", False)) or not bool(Setting.get_json("explicit_live_confirmed", False)):
        return "1H10 live execution requires explicit live trading confirmation."
    if not bool(current_app.config.get("SECONDARY_CONFIRMATION", False)) or not bool(Setting.get_json("secondary_confirmation", False)):
        return "1H10 live execution requires secondary live trading confirmation."
    return None


def _one_h10_live_context(user) -> dict[str, object]:
    connections = _cycle_trading_connections(user, True)
    providers: list[dict[str, object]] = []
    total_free_margin = 0.0
    blockers: list[dict[str, object]] = []
    for connection in connections:
        provider = normalize_provider(connection.provider)
        collateral = provider_collateral_asset(provider)
        cached_health = latest_connection_health(connection.id)
        provider_payload: dict[str, object] = {
            "provider": provider,
            "trading_connection_id": connection.id,
            "verified": connection.verification_status == "verified",
            "active": bool(connection.is_active),
            "collateral_asset": collateral,
            "available_margin_usd": 0.0,
            "can_trade": bool(cached_health.get("can_trade", False)) if cached_health else False,
            "health": cached_health,
            "blockers": [],
        }
        if provider not in {"hyperliquid", "kucoin"}:
            provider_payload["blockers"] = ["provider_discovery_not_implemented"]
            blockers.append({"provider": provider, "trading_connection_id": connection.id, "reason": "provider_discovery_not_implemented"})
            providers.append(provider_payload)
            continue
        backoff_notice = _one_h10_provider_market_data_backoff(provider, connection.id)
        if backoff_notice:
            reason = str(backoff_notice.get("message") or "Market data backoff active")
            health = cached_health or build_connection_health(
                connection,
                can_trade=False,
                alerts=[reason],
                failure_reason=reason,
            )
            provider_payload.update(
                {
                    "health": health,
                    "blockers": ["rate_limited"],
                    "can_trade": False,
                    "market_data_backoff": backoff_notice,
                }
            )
            blockers.append({"provider": provider, "trading_connection_id": connection.id, "reason": "rate_limited"})
            providers.append(provider_payload)
            continue
        try:
            snapshot = get_service("trading_connections").account_snapshot(user.id, "live", connection.id)
            alerts = [str(alert) for alert in (snapshot.alerts or []) if str(alert).strip()]
            available = _snapshot_free_margin_usd(snapshot, collateral)
            can_trade = bool(get_service("trading_connections").can_trade(user.id, "live", connection.id)) and not alerts
            health = build_connection_health(
                connection,
                can_trade=can_trade,
                alerts=alerts,
                failure_reason="; ".join(alerts) if alerts else None,
            )
            store_connection_health(connection, health)
            provider_payload.update(
                {
                    "available_margin_usd": available,
                    "can_trade": can_trade,
                    "health": health,
                    "blockers": alerts,
                }
            )
            if available <= 0:
                provider_payload["blockers"] = list(provider_payload.get("blockers", [])) + ["insufficient_free_margin"]
            if can_trade and available > 0:
                total_free_margin += available
            else:
                blockers.append(
                    {
                        "provider": provider,
                        "trading_connection_id": connection.id,
                        "reason": "; ".join(provider_payload.get("blockers", []) or ["provider_unhealthy"]),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            health = build_connection_health(connection, can_trade=False, alerts=[str(exc)], failure_reason=str(exc))
            store_connection_health(connection, health)
            provider_payload.update({"health": health, "blockers": [str(exc)], "can_trade": False})
            blockers.append({"provider": provider, "trading_connection_id": connection.id, "reason": str(exc)})
        providers.append(provider_payload)
    if providers:
        commit_with_retry()

    live_block = _one_h10_live_start_block_reason()
    ml_readiness = _one_h10_ml_readiness("global")
    blocker_categories = _blocker_categories_from_reasons(blockers)
    if live_block:
        blocker_categories.append(_blocker_category(live_block))
    if not bool(ml_readiness.get("ready", False)):
        blocker_categories.append("ml_not_ready")
    return {
        "enabled": bool(current_app.config.get("ONE_H10_LIVE_ENABLED", False)),
        "explicit_live_confirmed": bool(current_app.config.get("EXPLICIT_LIVE_CONFIRMED", False))
        and bool(Setting.get_json("explicit_live_confirmed", False)),
        "secondary_confirmation": bool(current_app.config.get("SECONDARY_CONFIRMATION", False))
        and bool(Setting.get_json("secondary_confirmation", False)),
        "ack_required": True,
        "target_copy": "1H10 high-upside objective",
        "providers": providers,
        "enabled_provider_count": sum(
            1 for item in providers if bool(item.get("can_trade")) and float(item.get("available_margin_usd", 0.0) or 0.0) > 0
        ),
        "total_free_margin_usd": total_free_margin,
        "max_dynamic_allocation_usd": total_free_margin,
        "ml_readiness": ml_readiness,
        "safety_blockers": list(dict.fromkeys(blocker_categories)),
        "live_block_reason": live_block,
        "poll_seconds": float(current_app.config.get("ONE_H10_POLL_SECONDS", 1.0) or 1.0),
        "rebalance_seconds": float(current_app.config.get("ONE_H10_REBALANCE_SECONDS", 15.0) or 15.0),
    }


def _one_h10_provider_market_data_backoff(provider: str, connection_id: int | str | None) -> dict[str, object] | None:
    provider_key = normalize_provider(provider)
    payload = Setting.get_json(f"one_h10_market_data_backoff:{provider_key}:{connection_id or 'global'}", {})
    if not isinstance(payload, dict):
        return None
    raw = str(payload.get("retry_after") or payload.get("backoff_until") or "")
    if not raw:
        return None
    try:
        until = datetime.fromisoformat(raw.replace("Z", ""))
    except ValueError:
        return None
    if until <= datetime.utcnow():
        return None
    reason = str(payload.get("reason") or "").strip()
    return {
        "provider": provider_key,
        "trading_connection_id": connection_id,
        "retry_after": until.isoformat(),
        "blocker_category": payload.get("blocker_category") or "rate_limited",
        "message": reason or f"{provider_key.title()} market data backoff active until {until.isoformat()}",
    }


def _normalized_cycle_leg_identity(
    leg: dict[str, object],
    parameters: dict[str, object],
    selection,
    *,
    fallback_connection_id: int | None = None,
) -> dict[str, object]:
    provider = normalize_provider(
        parameters.get("provider") or leg.get("provider") or getattr(selection, "metadata", {}).get("provider") or "global"
    )
    execution_venue = normalize_provider(parameters.get("execution_venue") or leg.get("execution_venue") or provider)
    app_symbol = (
        str(
            parameters.get("app_symbol")
            or leg.get("app_symbol")
            or leg.get("symbol")
            or parameters.get("symbol")
            or getattr(selection, "symbol", "")
            or "BTC"
        )
        .strip()
        .upper()
    )
    if not app_symbol:
        app_symbol = "BTC"
    raw_venue = (
        parameters.get("venue_symbol")
        or parameters.get("provider_symbol")
        or leg.get("venue_symbol")
        or leg.get("provider_symbol")
        or app_symbol
    )
    venue_symbol = str(raw_venue or app_symbol).strip()
    if not venue_symbol:
        venue_symbol = app_symbol
    if provider != "hyperliquid":
        venue_symbol = venue_symbol.upper()
    connection_raw = parameters.get("trading_connection_id") or leg.get("trading_connection_id") or fallback_connection_id
    try:
        trading_connection_id = int(connection_raw) if connection_raw is not None and str(connection_raw).strip() else None
    except (TypeError, ValueError):
        trading_connection_id = fallback_connection_id
    return {
        "symbol": app_symbol,
        "app_symbol": app_symbol,
        "venue_symbol": venue_symbol,
        "provider_symbol": venue_symbol,
        "provider": provider,
        "execution_venue": execution_venue,
        "trading_connection_id": trading_connection_id,
        "market_id": parameters.get("market_id") or leg.get("market_id"),
    }


def _json_safe_metadata(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _recover_active_one_h10_cycles(cycles: list[VaultCycle]) -> list[int]:
    start_run_ids: list[int] = []
    changed = False
    for cycle in cycles:
        changed = _refresh_one_h10_cycle_ml_state(cycle) or changed
        resumed = _resume_one_h10_active_runs(cycle)
        if resumed:
            start_run_ids.extend(resumed)
            changed = True
    if changed:
        commit_with_retry()
    return list(dict.fromkeys(start_run_ids))


def _is_one_h10_run_record(run: StrategyRun | None) -> bool:
    if run is None:
        return False
    params = run.parameters if isinstance(run.parameters, dict) else {}
    markers = {
        str(params.get("algorithm_profile") or "").strip().lower(),
        str(params.get("vault_cycle_name") or "").strip().lower(),
        str(params.get("ml_horizon") or "").strip().lower(),
        str(params.get("objective") or "").strip().lower(),
    }
    return bool(params.get("one_h10_vault")) or bool(markers & {"1h10", "one_h10", "one_hour_10x"})


def _mark_run_trade_decision(run: StrategyRun, *, stage: str, reason: str, message: str) -> None:
    payload = dict(run.last_signal or {}) if isinstance(run.last_signal, dict) else {}
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}
    metadata.update(
        {
            "trade_decision_stage": stage,
            "no_trade_reason": reason,
            "decision_reason_code": reason,
            "one_h10_vault": True,
            "ml_horizon": ONE_H10_HORIZON,
        }
    )
    run.last_signal = {
        "action": payload.get("action") or "hold",
        "rationale": payload.get("rationale") or message,
        "timeframe": payload.get("timeframe") or run.timeframe,
        "stop_loss": payload.get("stop_loss"),
        "take_profit": payload.get("take_profit"),
        "position_fraction": payload.get("position_fraction", 0.0),
        "metadata": metadata,
    }


def _start_strategy_runs(run_ids: list[int]) -> None:
    if not run_ids:
        return
    if not in_process_workers_enabled(current_app.config):
        queued_ids: list[int] = []
        for run_id in dict.fromkeys(run_ids):
            run = db.session.get(StrategyRun, int(run_id))
            if run is None:
                continue
            run.manual_enabled = True
            if run.status not in {"running", "starting"}:
                run.status = "queued"
            if _is_one_h10_run_record(run):
                _mark_run_trade_decision(
                    run,
                    stage="queued_for_worker",
                    reason="strategy_worker_pending",
                    message="1H10 strategy run is queued for the dedicated worker before signals can place orders.",
                )
            queued_ids.append(int(run.id))
        if queued_ids:
            audit = AuditLog(
                category="worker",
                action="strategy_runs_queued_for_worker",
                message="Strategy runs queued for dedicated worker startup.",
            )
            audit.details = {"run_ids": queued_ids, "worker_mode": current_app.config.get("WORKER_MODE", "web")}
            db.session.add(audit)
            commit_with_retry()
        return
    manager = get_service("strategy_manager")
    for run_id in dict.fromkeys(run_ids):
        manager.start(run_id)


def _request_idempotency_key() -> str:
    header_key = str(request.headers.get("Idempotency-Key", "")).strip()
    form_key = str(_request_value("idempotency_key", "")).strip()
    return header_key or form_key


def _wants_json_response() -> bool:
    requested = str(request.args.get("response", "")).strip().lower()
    if requested == "json":
        return True
    return request.accept_mimetypes.best == "application/json"


def _wants_start_json_response() -> bool:
    return request.path.rstrip("/").endswith("/vault/start-cycle") or _wants_json_response()


def _cycle_start_runtime_metadata(
    *,
    cycle_id: int | None,
    run_ids: list[int] | tuple[int, ...] | None,
    job_id: str | None = None,
    status: str | None = None,
) -> dict[str, object]:
    run_id_list = [int(run_id) for run_id in dict.fromkeys(run_ids or []) if run_id]
    cycle_id_value = int(cycle_id) if cycle_id else None
    job_id_value = str(job_id or "").strip()
    in_process = in_process_workers_enabled(current_app.config)
    worker_mode = str(current_app.config.get("WORKER_MODE", "web") or "web").strip().lower()
    metadata: dict[str, object] = {
        "status": status or ("queued" if job_id_value else "started"),
        "cycle_id": cycle_id_value,
        "run_ids": run_id_list,
        "worker_mode": worker_mode,
        "worker_process_configured": bool(current_app.config.get("WORKER_PROCESS_CONFIGURED", False)),
        "in_process_workers_enabled": bool(in_process),
        "strategy_run_queue": "in_process" if in_process else "dedicated_worker",
        "live_order_path": "VaultCycle -> StrategyRun -> Worker -> RiskEngine -> OrderManager",
    }
    if cycle_id_value:
        status_url = url_for("consumer.vault_cycle_status", cycle_id=cycle_id_value)
        detail_url = url_for("consumer.cycle_detail", cycle_id=cycle_id_value)
        metadata.update(
            {
                "cycle_status_url": status_url,
                "next_status_url": status_url,
                "cycle_detail_url": detail_url,
            }
        )
    if job_id_value:
        metadata["job_id"] = job_id_value
        metadata["start_status_url"] = url_for("consumer.cycle_start_status", job_id=job_id_value)
    return metadata


def _with_cycle_start_runtime_metadata(payload: dict[str, object]) -> dict[str, object]:
    enriched = dict(payload or {})
    try:
        cycle_id = int(enriched.get("cycle_id") or 0) or None
    except (TypeError, ValueError):
        cycle_id = None
    run_ids = [int(item) for item in list(enriched.get("run_ids") or []) if item]
    enriched.update(
        _cycle_start_runtime_metadata(
            cycle_id=cycle_id,
            run_ids=run_ids,
            job_id=str(enriched.get("job_id") or ""),
            status=str(enriched.get("status") or ""),
        )
    )
    return enriched


def _simple_start_blocker(code: str, title: str, description: str, *, severity: str = "blocker", fix_hint: str = "") -> dict[str, object]:
    return {
        "code": code,
        "title": title,
        "description": description,
        "severity": severity,
        "fix_hint": fix_hint or description,
    }


def _vault_start_blocked_response(readiness: dict[str, object], *, status_code: int = 400):
    blockers = list(readiness.get("active_blockers") or readiness.get("all_blockers") or readiness.get("blockers") or [])
    if not blockers:
        blockers = [
            _simple_start_blocker(
                "vault_not_ready",
                "Vault cycle not ready",
                str(readiness.get("message") or "Vault cycle readiness checks did not pass."),
            )
        ]
    count = len(blockers)
    message = str(readiness.get("message") or f"Vault cycle is blocked by {count} live gate{'s' if count != 1 else ''}.")
    if _wants_start_json_response():
        first = blockers[0] if isinstance(blockers[0], dict) else {}
        return jsonify(
            action_envelope(
                ok=False,
                code=str(first.get("code") or "vault_start_blocked"),
                message=message,
                blockers=blockers,
                details={"readiness": readiness},
                ready=False,
                mode=readiness.get("mode", "blocked"),
                readiness=readiness,
            )
        ), status_code
    first = blockers[0] if isinstance(blockers[0], dict) else {}
    flash(str(first.get("description") or first.get("title") or message), "warning")
    return redirect(url_for("consumer.vault"))


def _vault_start_error_response(code: str, title: str, description: str, *, status_code: int = 400):
    readiness = {
        "ready": False,
        "mode": "blocked",
        "message": description,
        "active_blockers": [_simple_start_blocker(code, title, description)],
    }
    return _vault_start_blocked_response(readiness, status_code=status_code)


def _existing_cycle_start_job(user_id: int, idempotency_key: str) -> dict[str, object] | None:
    idempotency_key = str(idempotency_key or "").strip()
    if not idempotency_key:
        return None
    lookup_key = (int(user_id), str(idempotency_key))
    with _CYCLE_START_JOB_LOCK:
        job_id = _CYCLE_START_IDEMPOTENCY.get(lookup_key)
    if not job_id:
        job_id = _load_cycle_start_idempotency_job_id(int(user_id), idempotency_key)
    if not job_id:
        return None
    payload = _load_cycle_start_job(job_id)
    if payload is None:
        return None
    with _CYCLE_START_JOB_LOCK:
        _CYCLE_START_IDEMPOTENCY[lookup_key] = str(job_id)
        _CYCLE_START_JOBS[str(job_id)] = dict(payload)
    return dict(payload)


def _enqueue_cycle_start_job(
    *,
    run_ids: list[int],
    cycle_id: int,
    user_id: int,
    idempotency_key: str,
) -> str:
    app = current_app._get_current_object()
    job_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    payload: dict[str, object] = {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "cycle_id": int(cycle_id),
        "run_ids": [int(run_id) for run_id in dict.fromkeys(run_ids)],
        "user_id": int(user_id),
        "queued_at": now,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    with _CYCLE_START_JOB_LOCK:
        _CYCLE_START_JOBS[job_id] = dict(payload)
        if idempotency_key:
            _CYCLE_START_IDEMPOTENCY[(int(user_id), str(idempotency_key))] = job_id
    _persist_cycle_start_job(payload)
    if idempotency_key:
        _persist_cycle_start_idempotency(user_id=int(user_id), idempotency_key=str(idempotency_key), job_id=job_id)
    if in_process_workers_enabled(current_app.config):
        worker = threading.Thread(
            target=_run_cycle_start_job,
            args=(app, job_id),
            daemon=True,
            name=f"vault-start-{job_id[:12]}",
        )
        worker.start()
    return job_id


def _run_cycle_start_job(app, job_id: str) -> None:
    with app.app_context():
        job: dict[str, object] | None
        with _CYCLE_START_JOB_LOCK:
            job = _CYCLE_START_JOBS.get(job_id)
        if not job:
            job = _load_cycle_start_job(job_id)
        if not job:
            return
        job = dict(job)
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat()
        with _CYCLE_START_JOB_LOCK:
            _CYCLE_START_JOBS[job_id] = dict(job)
        _persist_cycle_start_job(job)
        run_ids = [int(item) for item in list(job.get("run_ids") or [])]
        error_message = None
        try:
            _start_strategy_runs(run_ids)
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            current_app.logger.exception("Vault cycle async start failed for job_id=%s", job_id)
        job["completed_at"] = datetime.utcnow().isoformat()
        job["status"] = "failed" if error_message else "complete"
        job["error"] = error_message
        with _CYCLE_START_JOB_LOCK:
            _CYCLE_START_JOBS[job_id] = dict(job)
        _persist_cycle_start_job(job)


def _cycle_start_job_key(job_id: str) -> str:
    return f"{_CYCLE_START_JOB_KEY_PREFIX}:{str(job_id).strip()}"


def _cycle_start_idempotency_key(user_id: int, idempotency_key: str) -> str:
    return f"{_CYCLE_START_IDEMPOTENCY_KEY_PREFIX}:{int(user_id)}:{str(idempotency_key).strip()}"


def _cycle_start_sync_idempotency_key(user_id: int, idempotency_key: str) -> str:
    return f"{_CYCLE_START_SYNC_IDEMPOTENCY_KEY_PREFIX}:{int(user_id)}:{str(idempotency_key).strip()}"


def _existing_cycle_start_cycle(user_id: int, idempotency_key: str) -> VaultCycle | None:
    key = str(idempotency_key or "").strip()
    if not key:
        return None
    lookup_key = (int(user_id), key)
    with _CYCLE_START_JOB_LOCK:
        cycle_id = _CYCLE_START_SYNC_IDEMPOTENCY.get(lookup_key)
    if cycle_id is None:
        stored = Setting.get_json(_cycle_start_sync_idempotency_key(user_id, key), {})
        if isinstance(stored, dict):
            try:
                cycle_id = int(stored.get("cycle_id") or 0)
            except (TypeError, ValueError):
                cycle_id = None
        elif isinstance(stored, int):
            cycle_id = stored
    if not cycle_id:
        return None
    cycle = VaultCycle.query.filter_by(id=int(cycle_id), user_id=int(user_id)).one_or_none()
    if cycle is not None:
        with _CYCLE_START_JOB_LOCK:
            _CYCLE_START_SYNC_IDEMPOTENCY[lookup_key] = int(cycle.id)
    return cycle


def _persist_cycle_start_cycle_idempotency(*, user_id: int, idempotency_key: str, cycle_id: int) -> None:
    key = str(idempotency_key or "").strip()
    if not key:
        return
    with _CYCLE_START_JOB_LOCK:
        _CYCLE_START_SYNC_IDEMPOTENCY[(int(user_id), key)] = int(cycle_id)
    Setting.set_json(
        _cycle_start_sync_idempotency_key(user_id, key),
        {
            "cycle_id": int(cycle_id),
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    commit_with_retry()


def _persist_cycle_start_job(payload: dict[str, object]) -> None:
    job_id = str(payload.get("job_id", "")).strip()
    if not job_id:
        return
    Setting.set_json(_cycle_start_job_key(job_id), payload)
    commit_with_retry()


def _persist_cycle_start_idempotency(*, user_id: int, idempotency_key: str, job_id: str) -> None:
    key = str(idempotency_key).strip()
    if not key:
        return
    Setting.set_json(
        _cycle_start_idempotency_key(user_id, key),
        {
            "job_id": str(job_id),
            "updated_at": datetime.utcnow().isoformat(),
        },
    )
    commit_with_retry()


def _load_cycle_start_job(job_id: str) -> dict[str, object] | None:
    job_id = str(job_id).strip()
    if not job_id:
        return None
    with _CYCLE_START_JOB_LOCK:
        cached = _CYCLE_START_JOBS.get(job_id)
        if cached:
            return dict(cached)
    stored = Setting.get_json(_cycle_start_job_key(job_id), {})
    if not isinstance(stored, dict) or not stored:
        return None
    with _CYCLE_START_JOB_LOCK:
        _CYCLE_START_JOBS[job_id] = dict(stored)
    return dict(stored)


def _load_cycle_start_idempotency_job_id(user_id: int, idempotency_key: str) -> str | None:
    raw = Setting.get_json(_cycle_start_idempotency_key(user_id, idempotency_key), {})
    if isinstance(raw, dict):
        value = str(raw.get("job_id", "")).strip()
        return value or None
    if isinstance(raw, str):
        value = raw.strip()
        return value or None
    return None


def _refresh_one_h10_cycle_ml_state(cycle: VaultCycle) -> bool:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return False
    changed = False
    metadata = dict(cycle.selection_metadata or {})
    readiness = _json_safe_metadata(_one_h10_ml_readiness("global"))
    if metadata.get("ml_readiness") != readiness:
        metadata["ml_readiness"] = readiness
        changed = True
    if bool(readiness.get("ready", False)):
        for key in ("blocker_categories", "ml_blockers"):
            current = metadata.get(key)
            if isinstance(current, list):
                cleaned = [item for item in current if str(item) != "ml_not_ready"]
                if cleaned != current:
                    metadata[key] = cleaned
                    changed = True
    forecast_service = get_service("one_h10_forecast")
    for leg in cycle.allocation_legs:
        if leg.status not in {"active", "starting"}:
            continue
        details = dict(leg.details or {})
        forecast = details.get("one_h10_forecast") if isinstance(details.get("one_h10_forecast"), dict) else {}
        if not _one_h10_forecast_needs_refresh(forecast, details, readiness):
            continue
        refreshed = _refreshed_one_h10_forecast(forecast_service, leg, details)
        if not refreshed:
            continue
        safe_forecast = _json_safe_metadata(refreshed)
        _apply_one_h10_forecast(details, safe_forecast)
        leg.details = _json_safe_metadata(details)
        if leg.strategy_run is not None:
            params = dict(leg.strategy_run.parameters or {})
            _apply_one_h10_forecast(params, safe_forecast)
            if not params.get("scanner_features") and details.get("scanner_features"):
                params["scanner_features"] = details.get("scanner_features")
            leg.strategy_run.parameters = _json_safe_metadata(params)
        if _update_one_h10_history_forecast(metadata, leg, safe_forecast):
            changed = True
        changed = True
    forecasts = []
    for leg in cycle.allocation_legs:
        details = dict(leg.details or {})
        forecast = details.get("one_h10_forecast") if isinstance(details.get("one_h10_forecast"), dict) else {}
        if forecast:
            forecasts.append(forecast)
        if leg.strategy_run is not None and isinstance(leg.strategy_run.parameters, dict):
            run_forecast = leg.strategy_run.parameters.get("one_h10_forecast")
            if isinstance(run_forecast, dict) and run_forecast:
                forecasts.append(run_forecast)
    for provider_row in metadata.get("provider_allocation_history", metadata.get("exchange_allocation_history", [])) or []:
        if not isinstance(provider_row, dict):
            continue
        for history_leg in provider_row.get("legs", []) or []:
            if isinstance(history_leg, dict) and isinstance(history_leg.get("forecast"), dict):
                forecasts.append(history_leg["forecast"])
    missing_coherence = any(
        metadata.get(key) is None for key in ("cycle_status", "horizon_forecasts", "horizon_strategy_scores", "coherence_summary")
    )
    if forecasts and (changed or missing_coherence):
        coherence_payload = cycle_coherence_payload_from_forecasts(forecasts)
        for key, value in coherence_payload.items():
            if metadata.get(key) != value:
                metadata[key] = value
                changed = True
    if changed:
        metadata["one_h10_ml_state_refreshed_at"] = datetime.utcnow().isoformat()
        cycle.selection_metadata = _json_safe_metadata(metadata)
    return changed


def _resume_one_h10_active_runs(cycle: VaultCycle) -> list[int]:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return []
    if str(cycle.status or "").lower() != "active":
        return []
    if not bool(current_app.config.get("ONE_H10_AUTO_RESUME_ACTIVE_RUNS", True)):
        return []
    if Setting.get_json("panic_lock", False):
        return []
    if cycle.unlocks_at and cycle.unlocks_at <= datetime.utcnow():
        return []
    run_ids: list[int] = []
    for leg in cycle.allocation_legs:
        if str(leg.status or "").lower() != "active" or leg.strategy_run is None:
            continue
        run = leg.strategy_run
        status = str(run.status or "").lower()
        heartbeat_at = run.last_heartbeat_at
        heartbeat_stale = heartbeat_at is None or (datetime.utcnow() - heartbeat_at).total_seconds() > max(
            60.0, float(current_app.config.get("ONE_H10_POLL_SECONDS", 1.0) or 1.0) * 5.0
        )
        if run.manual_enabled and status in {"running", "starting"} and not heartbeat_stale:
            continue
        run.manual_enabled = True
        run.status = "starting"
        run.last_error = None
        run_ids.append(run.id)
    if run_ids:
        metadata = dict(cycle.selection_metadata or {})
        metadata["one_h10_auto_resumed_at"] = datetime.utcnow().isoformat()
        metadata["one_h10_auto_resumed_run_ids"] = list(dict.fromkeys(run_ids))
        cycle.selection_metadata = _json_safe_metadata(metadata)
    return run_ids


def _one_h10_forecast_needs_refresh(
    forecast: dict[str, object],
    details: dict[str, object],
    readiness: dict[str, object],
) -> bool:
    if not bool(readiness.get("ready", False)):
        return False
    blockers = set()
    for key in ("blockers", "advisory_blockers"):
        blockers.update(str(item) for item in (forecast.get(key, []) or []) if str(item))
    for key in ("forecast_blockers", "forecast_advisory_blockers"):
        blockers.update(str(item) for item in (details.get(key, []) or []) if str(item))
    if "ml_not_ready" in blockers:
        return True
    if not forecast:
        return True
    if bool(readiness.get("promoted_ready", False)) and not bool(forecast.get("ml_ready", False)):
        return True
    return str(forecast.get("source") or "") == "one_h10_bootstrap_forecast" and bool(readiness.get("promoted_ready", False))


def _refreshed_one_h10_forecast(
    forecast_service,
    leg: VaultAllocationLeg,
    details: dict[str, object],
) -> dict[str, object]:
    if forecast_service is None:
        return {}
    features = dict(details.get("scanner_features") or {})
    features.pop("one_h10_forecast", None)
    provider = str(details.get("provider") or leg.provider or "global")
    symbol = str(details.get("app_symbol") or leg.symbol or "").upper()
    market = None
    market_id = details.get("market_id")
    if market_id:
        try:
            market = db.session.get(LeveragedMarket, int(market_id))
        except (TypeError, ValueError):
            market = None
    try:
        return dict(
            forecast_service.forecast(
                features,
                provider=provider,
                symbol=symbol,
                allocation_cap_usd=float(leg.allocation_cap_usd or 0.0),
                available_margin_usd=float(details.get("available_margin_usd", 0.0) or 0.0),
                market=market,
            )
        )
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Failed to refresh 1H10 forecast for cycle leg %s: %s", leg.id, exc)
        return {}


def _apply_one_h10_forecast(payload: dict[str, object], forecast: dict[str, object]) -> None:
    payload["one_h10_forecast"] = forecast
    payload["forecast_metadata"] = forecast
    payload["forecast_blockers"] = list(forecast.get("blockers", []) or [])
    payload["forecast_advisory_blockers"] = list(forecast.get("advisory_blockers", []) or [])
    payload["forecast_predicted_side"] = forecast.get("predicted_side") or forecast.get("action")
    payload["forecast_confidence"] = forecast.get("confidence")
    payload["forecast_expected_return_bps"] = forecast.get("expected_return_bps")
    payload["forecast_suggested_notional_usd"] = forecast.get("suggested_notional_usd")
    payload["forecast_suggested_leverage"] = forecast.get("suggested_leverage")
    payload["forecast_suggested_order_type"] = forecast.get("suggested_order_type")
    payload["forecast_suggested_stop_loss_pct"] = forecast.get("suggested_stop_loss_pct")
    payload["forecast_suggested_take_profit_pct"] = forecast.get("suggested_take_profit_pct")


def _update_one_h10_history_forecast(
    metadata: dict[str, object],
    leg: VaultAllocationLeg,
    forecast: dict[str, object],
) -> bool:
    changed = False
    for key in ("provider_allocation_history", "exchange_allocation_history"):
        rows = metadata.get(key)
        if not isinstance(rows, list):
            continue
        for provider_row in rows:
            if not isinstance(provider_row, dict):
                continue
            for history_leg in provider_row.get("legs", []) or []:
                if not isinstance(history_leg, dict):
                    continue
                if not _one_h10_history_leg_matches(provider_row, history_leg, leg):
                    continue
                if history_leg.get("forecast") != forecast:
                    history_leg["forecast"] = forecast
                    changed = True
    return changed


def _one_h10_history_leg_matches(
    provider_row: dict[str, object],
    history_leg: dict[str, object],
    leg: VaultAllocationLeg,
) -> bool:
    provider = str(provider_row.get("provider") or "").lower()
    if provider and provider != str(leg.provider or "").lower():
        return False
    row_connection = provider_row.get("trading_connection_id")
    if row_connection is not None and leg.trading_connection_id is not None:
        try:
            if int(row_connection) != int(leg.trading_connection_id):
                return False
        except (TypeError, ValueError):
            return False
    history_market_id = history_leg.get("market_id")
    leg_market_id = leg.details.get("market_id")
    if history_market_id is not None and leg_market_id is not None:
        try:
            return int(history_market_id) == int(leg_market_id)
        except (TypeError, ValueError):
            return False
    return str(history_leg.get("symbol") or "").upper() == str(leg.symbol or "").upper()


def _one_h10_ml_readiness(provider: str = "global") -> dict[str, object]:
    required_families = (
        "pytorch_fibonacci",
        "pytorch_risk_policy",
        "pytorch_exit_policy",
        "pytorch_cap_policy",
        "pytorch_execution_policy",
        "pytorch_roi_target",
    )
    try:
        engine = get_service("ml_decision_engine")
        families = {family: dict(engine.family_readiness(family, ONE_H10_HORIZON, provider=provider)) for family in required_families}
        blockers: list[str] = []
        for family, payload in families.items():
            blockers.extend(f"{family}:{item}" for item in payload.get("blockers", []) or [])
        promoted_blockers = list(dict.fromkeys(blockers))
        promoted_ready = not promoted_blockers
        bootstrap_enabled = bool(current_app.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True)) and not bool(
            current_app.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)
        )
        execution_ready = promoted_ready or bootstrap_enabled
        readiness_mode = "promoted" if promoted_ready else "bootstrap" if bootstrap_enabled else "blocked"
        return {
            "ready": execution_ready,
            "execution_ready": execution_ready,
            "promoted_ready": promoted_ready,
            "bootstrap_enabled": bootstrap_enabled,
            "mode": readiness_mode,
            "display_status": "Ready" if promoted_ready else "Bootstrap Ready" if bootstrap_enabled else "Bootstrap / Not Ready",
            "enabled": bool(current_app.config.get("ML_ALL_AREAS_ENABLED", False)) or bootstrap_enabled,
            "horizon": ONE_H10_HORIZON,
            "provider": provider,
            "family": "one_h10_live_execution",
            "objective": "one_h10",
            "target_roi_pct": float(current_app.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0) or 1000.0),
            "families": families,
            "required_families": list(required_families),
            "blockers": [] if execution_ready else promoted_blockers,
            "advisory_blockers": promoted_blockers if execution_ready and not promoted_ready else [],
            "promoted_blockers": promoted_blockers,
            "ignored_optional_families": ["pytorch_gru_signal"],
            "source": "one_h10_live_execution_readiness",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "enabled": False,
            "horizon": ONE_H10_HORIZON,
            "provider": provider,
            "blockers": [str(exc)],
            "source": "ml_readiness_error",
        }


def _cycle_trading_connections(user, is_one_h10: bool, providers: list[str] | None = None) -> list[TradingConnection]:
    service = get_service("trading_connections")
    provider_filter = [normalize_provider(provider) for provider in (providers or []) if str(provider or "").strip()]
    if providers is not None and not provider_filter:
        return []
    if is_one_h10 and hasattr(service, "verified_tradable_connections"):
        return list(service.verified_tradable_connections(user.id, providers=provider_filter))
    if provider_filter:
        connection = service.active_tradable_connection(user.id, provider=provider_filter[0])
        return [connection] if connection is not None else []
    connection = service.active_tradable_connection(user.id)
    return [connection] if connection is not None else []


def _healthy_cycle_connections(
    user,
    connections: list[TradingConnection],
    is_one_h10: bool,
) -> tuple[list[TradingConnection], list[dict[str, object]]]:
    if not is_one_h10:
        return connections, []
    healthy: list[TradingConnection] = []
    blockers: list[dict[str, object]] = []
    for connection in connections:
        reason = _fresh_live_connection_block_reason(user, connection)
        if reason:
            blockers.append(
                {
                    "provider": connection.provider,
                    "trading_connection_id": connection.id,
                    "reason": reason,
                }
            )
            continue
        healthy.append(connection)
    return healthy, blockers


def _snapshot_free_margin_usd(snapshot, collateral_asset: str) -> float:
    collateral = str(collateral_asset or "").upper()
    stable_assets = {"USDC", "USDT"}
    best = 0.0
    fallback = 0.0
    for row in getattr(snapshot, "balances", []) or []:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset") or "").upper()
        amount = float(row.get("withdrawable", row.get("available", row.get("value", 0.0))) or 0.0)
        value = float(row.get("value", amount) or amount)
        free_value = amount if asset in stable_assets else value
        if asset == collateral:
            best = max(best, free_value)
        elif asset in stable_assets and collateral in stable_assets:
            fallback = max(fallback, free_value)
    return max(best, fallback, 0.0)


def _one_h10_provider_legs(
    *,
    user,
    selection,
    connections: list[TradingConnection],
    starting_value_usd: float,
    settlement_asset: str,
    allowed_symbols: list[str],
    connection_blockers: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    trading_connections = get_service("trading_connections")
    market_service = get_service("leveraged_markets")
    scanner = get_service("market_scanner")
    forecast_service = get_service("one_h10_forecast")
    base_legs = list(selection.legs or []) or [
        {
            "strategy_name": selection.strategy_name,
            "symbol": selection.symbol,
            "timeframe": selection.timeframe,
            "parameters": selection.parameters,
            "allocation_cap_usd": starting_value_usd,
            "leverage": selection.parameters.get("leverage", 1.0),
            "optimizer_ranking_id": selection.metadata.get("optimizer_ranking_id"),
        }
    ]
    provider_allocations: list[dict[str, object]] = []
    blockers = list(connection_blockers or [])
    for connection in connections:
        provider = normalize_provider(connection.provider)
        if provider not in {"hyperliquid", "kucoin"}:
            blockers.append({"provider": provider, "trading_connection_id": connection.id, "reason": "provider_discovery_not_implemented"})
            continue
        collateral = provider_collateral_asset(provider)
        snapshot = trading_connections.account_snapshot(user.id, "live", connection.id)
        alerts = [str(alert) for alert in (getattr(snapshot, "alerts", []) or []) if str(alert).strip()]
        if alerts:
            blockers.append({"provider": provider, "trading_connection_id": connection.id, "reason": "; ".join(alerts)})
            continue
        available = _snapshot_free_margin_usd(snapshot, collateral)
        if available <= 0:
            blockers.append({"provider": provider, "trading_connection_id": connection.id, "reason": "insufficient_free_margin"})
            continue
        markets = market_service.active_markets(provider=provider, symbols=None)
        ranked = scanner.score_one_h10_markets(
            markets,
            provider=provider,
            limit=int(current_app.config.get("ONE_H10_MAX_PROVIDER_LEGS", 3) or 3),
        )
        diagnostics = dict(getattr(scanner, "last_scan_diagnostics", {}) or {})
        if not ranked:
            fallback_symbols = [str(leg.get("symbol") or "").upper() for leg in base_legs if str(leg.get("symbol") or "").strip()]
            fallback_symbols.extend(str(symbol or "").upper() for symbol in allowed_symbols if str(symbol or "").strip())
            fallback_symbols.extend(
                str(symbol or "").upper()
                for symbol in current_app.config.get("ALLOWED_SYMBOLS", ["BTC", "ETH", "SOL"])
                if str(symbol or "").strip()
            )
            if not fallback_symbols:
                fallback_symbols = [str(selection.symbol or "BTC").upper(), "ETH", "SOL"]
            max_fallbacks = max(1, int(current_app.config.get("ONE_H10_MAX_PROVIDER_LEGS", 3) or 3))
            ranked = []
            for fallback_index, fallback_symbol in enumerate(list(dict.fromkeys(fallback_symbols))[:max_fallbacks]):
                ranked.append(
                    ScoredCandidate(
                        symbol=fallback_symbol,
                        score=max(1.0, float(max_fallbacks - fallback_index)),
                        technical_score=0.0,
                        ml_score=0.0,
                        hot_score=0.0,
                        source="one_h10_bootstrap_fallback",
                        features={
                            "provider": provider,
                            "scanner_source": "one_h10_bootstrap_fallback",
                            "fallback_reason": "one_h10_no_ranked_markets",
                            "ml_horizon": ONE_H10_HORIZON,
                            "objective": "one_h10",
                            "symbol": fallback_symbol,
                        },
                        score_breakdown={"fallback": 1.0},
                        rejection_reason="one_h10_no_ranked_markets",
                        stale_data=True,
                    )
                )
            diagnostics = {
                **diagnostics,
                "fallback_reason": "one_h10_no_ranked_markets",
                "fallback_symbols": [candidate.symbol for candidate in ranked],
            }
            blockers.append(
                {
                    "provider": provider,
                    "trading_connection_id": connection.id,
                    "reason": "one_h10_no_ranked_markets_bootstrap_fallback",
                    "scanner_diagnostics": diagnostics,
                }
            )
        provider_allocations.append(
            {
                "provider": provider,
                "trading_connection_id": connection.id,
                "collateral_asset": collateral,
                "available_margin_usd": available,
                "markets": markets,
                "ranked": ranked,
                "scanner_diagnostics": diagnostics,
            }
        )
    total_available = sum(float(item["available_margin_usd"] or 0.0) for item in provider_allocations)
    if total_available <= 0:
        return [], [], blockers

    selection_total = sum(max(float(leg.get("allocation_cap_usd", 0.0) or 0.0), 0.0) for leg in base_legs)
    if selection_total <= 0:
        selection_total = float(len(base_legs) or 1)

    generated: list[dict[str, object]] = []
    allocation_history: list[dict[str, object]] = []
    for provider_allocation in provider_allocations:
        provider = str(provider_allocation["provider"])
        connection_id = int(provider_allocation["trading_connection_id"])
        available = float(provider_allocation["available_margin_usd"] or 0.0)
        provider_cap = min(available, starting_value_usd * (available / total_available))
        ranked = list(provider_allocation.get("ranked") or [])
        markets = list(provider_allocation.get("markets") or [])
        market_by_id = {int(market.id): market for market in markets if isinstance(market, LeveragedMarket) and market.id is not None}
        market_by_venue_symbol = {
            str(market.venue_symbol).upper(): market
            for market in markets
            if isinstance(market, LeveragedMarket) and str(market.venue_symbol or "").strip()
        }
        markets_by_symbol: dict[str, list[LeveragedMarket]] = {}
        for market in markets:
            if isinstance(market, LeveragedMarket):
                markets_by_symbol.setdefault(str(market.symbol).upper(), []).append(market)
        provider_history = {
            "provider": provider,
            "trading_connection_id": connection_id,
            "collateral_asset": provider_allocation["collateral_asset"],
            "available_margin_usd": available,
            "allocated_usd": 0.0,
            "settlement_asset": settlement_asset,
            "scanner_diagnostics": provider_allocation.get("scanner_diagnostics", {}),
            "legs": [],
        }
        accepted_rows: list[dict[str, object]] = []
        for candidate in ranked:
            candidate_symbol = str(candidate.symbol or selection.symbol).upper()
            template_leg = next(
                (dict(item) for item in base_legs if str(item.get("symbol") or "").upper() == candidate_symbol),
                dict(base_legs[0] if base_legs else {}),
            )
            leg = dict(template_leg)
            leg_symbol = str(candidate.symbol or leg.get("symbol") or selection.symbol).upper()
            candidate_features = dict(candidate.features or {})
            candidate_market_id = candidate_features.get("market_id")
            market = None
            if candidate_market_id:
                try:
                    market = market_by_id.get(int(candidate_market_id))
                except (TypeError, ValueError):
                    market = None
            venue_symbol = str(candidate_features.get("venue_symbol") or "").strip()
            if market is None and venue_symbol:
                market = market_by_venue_symbol.get(venue_symbol.upper())
            if market is None:
                symbol_matches = markets_by_symbol.get(leg_symbol, [])
                if len(symbol_matches) == 1:
                    market = symbol_matches[0]
            if market is None:
                bootstrap_fallback = str(candidate.source or "") == "one_h10_bootstrap_fallback" and bool(
                    current_app.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True)
                )
                if not bootstrap_fallback:
                    provider_history["legs"].append(
                        {
                            "symbol": leg_symbol,
                            "app_symbol": leg_symbol,
                            "venue_symbol": venue_symbol or leg_symbol,
                            "provider_symbol": venue_symbol or leg_symbol,
                            "allocation_cap_usd": 0.0,
                            "strategy_name": leg.get("strategy_name") or selection.strategy_name,
                            "market_id": candidate_market_id,
                            "market_status": "candidate_market_missing",
                            "scanner_score": candidate.score,
                            "scanner_source": candidate.source,
                            "score_breakdown": candidate.score_breakdown or {},
                            "skip_reason": "one_h10_candidate_market_missing",
                        }
                    )
                    blockers.append(
                        {
                            "provider": provider,
                            "trading_connection_id": connection_id,
                            "symbol": leg_symbol,
                            "reason": "one_h10_candidate_market_missing",
                        }
                    )
                    continue
            symbol = str(getattr(market, "symbol", leg_symbol) or leg_symbol).upper()
            venue_symbol = str(getattr(market, "venue_symbol", venue_symbol) or venue_symbol or symbol).strip()
            market_status = getattr(market, "status", "fallback_configured") if market is not None else "fallback_configured"
            forecast = {}
            if forecast_service is not None:
                forecast = forecast_service.forecast(
                    candidate_features,
                    provider=provider,
                    symbol=symbol,
                    allocation_cap_usd=provider_cap,
                    available_margin_usd=available,
                    market=market,
                )
                candidate_features["one_h10_forecast"] = forecast
            live_blockers = _one_h10_forecast_live_blockers(forecast)
            if live_blockers:
                reason = "one_h10_forecast_blocked:" + ",".join(live_blockers)
                provider_history["legs"].append(
                    {
                        "symbol": symbol,
                        "app_symbol": symbol,
                        "venue_symbol": venue_symbol,
                        "provider_symbol": venue_symbol,
                        "allocation_cap_usd": 0.0,
                        "strategy_name": leg.get("strategy_name") or selection.strategy_name,
                        "market_id": getattr(market, "id", None),
                        "market_status": market_status,
                        "scanner_score": candidate.score,
                        "scanner_source": candidate.source,
                        "score_breakdown": candidate.score_breakdown or {},
                        "forecast": forecast,
                        "skip_reason": reason,
                    }
                )
                blockers.append(
                    {
                        "provider": provider,
                        "trading_connection_id": connection_id,
                        "symbol": symbol,
                        "venue_symbol": venue_symbol,
                        "reason": reason,
                        "forecast_blockers": live_blockers,
                    }
                )
                continue
            allocation_score = _one_h10_float(forecast.get("allocation_score") if isinstance(forecast, dict) else 0.0)
            if allocation_score <= 0:
                allocation_score = _one_h10_float(forecast.get("profitability_score") if isinstance(forecast, dict) else 0.0)
            if allocation_score <= 0:
                allocation_score = _one_h10_float(candidate_features.get("allocation_score"))
            if allocation_score <= 0:
                allocation_score = _one_h10_float(candidate.score)
            accepted_rows.append(
                {
                    "candidate": candidate,
                    "leg": leg,
                    "symbol": symbol,
                    "venue_symbol": venue_symbol,
                    "market": market,
                    "market_status": market_status,
                    "candidate_features": candidate_features,
                    "forecast": forecast,
                    "allocation_score": allocation_score,
                }
            )

        allocation_score_total = sum(max(_one_h10_float(row.get("allocation_score")), 0.0) for row in accepted_rows)
        if accepted_rows and allocation_score_total <= 0:
            allocation_score_total = float(len(accepted_rows))

        for row in accepted_rows:
            candidate = row["candidate"]
            leg = dict(row["leg"] if isinstance(row.get("leg"), dict) else {})
            symbol = str(row.get("symbol") or selection.symbol).upper()
            venue_symbol = str(row.get("venue_symbol") or symbol)
            market = row.get("market")
            market_status = str(row.get("market_status") or "fallback_configured")
            candidate_features = dict(row.get("candidate_features") or {})
            allocation_score = max(_one_h10_float(row.get("allocation_score")), 0.0)
            weight = allocation_score / allocation_score_total if allocation_score_total > 0 else 0.0
            if weight <= 0:
                weight = 1.0 / max(len(accepted_rows), 1)
            allocation_cap = provider_cap * weight
            if allocation_cap <= 0:
                continue
            forecast = dict(row.get("forecast") or {})
            if forecast_service is not None:
                forecast = forecast_service.forecast(
                    candidate_features,
                    provider=provider,
                    symbol=symbol,
                    allocation_cap_usd=allocation_cap,
                    available_margin_usd=available,
                    market=market if isinstance(market, LeveragedMarket) else None,
                )
                candidate_features["one_h10_forecast"] = forecast
            live_blockers = _one_h10_forecast_live_blockers(forecast)
            if live_blockers:
                reason = "one_h10_forecast_blocked:" + ",".join(live_blockers)
                provider_history["legs"].append(
                    {
                        "symbol": symbol,
                        "app_symbol": symbol,
                        "venue_symbol": venue_symbol,
                        "provider_symbol": venue_symbol,
                        "allocation_cap_usd": 0.0,
                        "strategy_name": leg.get("strategy_name") or selection.strategy_name,
                        "market_id": getattr(market, "id", None),
                        "market_status": market_status,
                        "scanner_score": getattr(candidate, "score", 0.0),
                        "scanner_source": getattr(candidate, "source", ""),
                        "allocation_score": allocation_score,
                        "score_breakdown": getattr(candidate, "score_breakdown", {}) or {},
                        "forecast": forecast,
                        "skip_reason": reason,
                    }
                )
                blockers.append(
                    {
                        "provider": provider,
                        "trading_connection_id": connection_id,
                        "symbol": symbol,
                        "venue_symbol": venue_symbol,
                        "reason": reason,
                        "forecast_blockers": live_blockers,
                    }
                )
                continue
            params = dict(leg.get("parameters") or selection.parameters)
            provider_context = provider_feature_context(provider)
            params.update(
                {
                    **provider_context,
                    "provider": provider,
                    "execution_venue": provider,
                    "trading_connection_id": connection_id,
                    "collateral_asset": provider_allocation["collateral_asset"],
                    "settlement_asset": settlement_asset,
                    "available_margin_usd": available,
                    "allocation_weight": allocation_cap / max(starting_value_usd, 1.0),
                    "one_h10_vault": True,
                    "ml_horizon": ONE_H10_HORIZON,
                    "objective": "one_h10",
                    "ml_objective": "one_h10",
                    "target_return_objective": "one_h10",
                    "ml_policy_required": True,
                    "ml_governed_risk": True,
                    "vault_cycle_name": "1H10",
                    "target_roi_pct": selection.metadata.get("target_roi_pct", current_app.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0)),
                    "target_amount_usd": starting_value_usd * 10.0,
                    "user_input_amount_usd": starting_value_usd,
                    "market_status": market_status,
                    "market_id": getattr(market, "id", None),
                    "venue_symbol": venue_symbol,
                    "app_symbol": symbol,
                    "one_h10_bootstrap_live": bool(current_app.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True)),
                    "one_h10_all_pairs": True,
                    "one_h10_scanner_score": candidate.score,
                    "one_h10_scanner_source": candidate.source,
                    "one_h10_allocation_score": allocation_score,
                    "one_h10_allocation_method": "net_expectancy",
                    "scanner_score_breakdown": candidate.score_breakdown or {},
                    "scanner_features": candidate_features,
                    "one_h10_forecast": forecast,
                    "forecast_metadata": forecast,
                    "forecast_blockers": list(forecast.get("blockers", []) if isinstance(forecast, dict) else []),
                    "forecast_advisory_blockers": list(forecast.get("advisory_blockers", []) if isinstance(forecast, dict) else []),
                    "forecast_predicted_side": forecast.get("predicted_side") if isinstance(forecast, dict) else None,
                    "forecast_confidence": forecast.get("confidence") if isinstance(forecast, dict) else None,
                    "forecast_expected_return_bps": forecast.get("expected_return_bps") if isinstance(forecast, dict) else None,
                    "forecast_suggested_notional_usd": forecast.get("suggested_notional_usd") if isinstance(forecast, dict) else None,
                    "forecast_suggested_leverage": forecast.get("suggested_leverage") if isinstance(forecast, dict) else None,
                    "forecast_suggested_order_type": forecast.get("suggested_order_type") if isinstance(forecast, dict) else None,
                    "forecast_suggested_stop_loss_pct": forecast.get("suggested_stop_loss_pct") if isinstance(forecast, dict) else None,
                    "forecast_suggested_take_profit_pct": forecast.get("suggested_take_profit_pct") if isinstance(forecast, dict) else None,
                    "forecast_profitability_score": forecast.get("profitability_score") if isinstance(forecast, dict) else None,
                    "forecast_allocation_score": forecast.get("allocation_score") if isinstance(forecast, dict) else None,
                    "forecast_execution_adjusted_net_return_bps": forecast.get("execution_adjusted_net_return_bps")
                    if isinstance(forecast, dict)
                    else None,
                    "liquidity_usd": candidate_features.get("liquidity_usd", getattr(market, "liquidity_usd", 0.0)),
                    "spread_bps": candidate_features.get("spread_bps", getattr(market, "spread_bps", 0.0)),
                    "funding_rate": candidate_features.get("funding_rate", getattr(market, "funding_rate", 0.0)),
                }
            )
            next_leg = dict(leg)
            next_leg.update(
                {
                    "provider": provider,
                    "execution_venue": provider,
                    "trading_connection_id": connection_id,
                    "collateral_asset": provider_allocation["collateral_asset"],
                    "settlement_asset": settlement_asset,
                    "symbol": symbol,
                    "venue_symbol": venue_symbol,
                    "app_symbol": symbol,
                    "allocation_cap_usd": allocation_cap,
                    "allocation_weight": allocation_cap / max(starting_value_usd, 1.0),
                    "parameters": params,
                    "market_id": getattr(market, "id", None),
                    "market_status": market_status,
                    "scanner_score": candidate.score,
                    "scanner_source": candidate.source,
                    "allocation_score": allocation_score,
                    "forecast": forecast,
                }
            )
            generated.append(next_leg)
            provider_history["allocated_usd"] = float(provider_history["allocated_usd"]) + allocation_cap
            provider_history["legs"].append(
                {
                    "symbol": symbol,
                    "app_symbol": symbol,
                    "venue_symbol": venue_symbol,
                    "provider_symbol": venue_symbol,
                    "allocation_cap_usd": allocation_cap,
                    "strategy_name": next_leg.get("strategy_name") or selection.strategy_name,
                    "market_id": getattr(market, "id", None),
                    "market_status": market_status,
                    "scanner_score": candidate.score,
                    "scanner_source": candidate.source,
                    "allocation_score": allocation_score,
                    "allocation_method": "net_expectancy",
                    "score_breakdown": candidate.score_breakdown or {},
                    "forecast": forecast,
                }
            )
        allocation_history.append(provider_history)
    return generated, allocation_history, blockers


def _one_h10_forecast_live_blockers(forecast: dict[str, object] | None) -> list[str]:
    return one_h10_forecast_live_blockers(forecast, current_app.config)


def _one_h10_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _one_h10_allocation_failure_message(blockers: list[dict[str, object]]) -> str:
    reasons = [
        str(row.get("reason") or "").replace("one_h10_forecast_blocked:", "")
        for row in blockers
        if isinstance(row, dict) and str(row.get("reason") or "").strip()
    ]
    if reasons:
        common = list(dict.fromkeys(reasons))[:3]
        return "1H10 did not start because no candidate passed live entry checks: " + "; ".join(common)
    return "1H10 could not allocate capital to a healthy, funded trading provider."


def _fresh_live_connection_block_reason(user, connection: TradingConnection | None) -> str | None:
    if not _live_connection_required():
        return None
    if connection is None:
        return "Connect your trading account before starting a live vault cycle."
    cached_health = latest_connection_health(connection.id)
    backoff_seconds = float(current_app.config.get("LIVE_CONNECTION_FAILURE_BACKOFF_SECONDS", 60.0) or 0.0)
    if _connection_health_backoff_active(cached_health, backoff_seconds):
        return operator_connection_message(cached_health)
    service = get_service("trading_connections")
    try:
        snapshot = service.account_snapshot(user.id, "live", connection.id)
        alerts = [str(alert) for alert in (snapshot.alerts or []) if str(alert).strip()]
        can_trade = bool(service.can_trade(user.id, "live", connection.id)) and not alerts
        health = build_connection_health(
            connection,
            can_trade=can_trade,
            alerts=alerts,
            failure_reason="; ".join(alerts) if alerts else None,
        )
    except Exception as exc:  # noqa: BLE001
        health = build_connection_health(connection, can_trade=False, alerts=[str(exc)], failure_reason=str(exc))
    store_connection_health(connection, health)
    commit_with_retry()
    if not bool(health.get("can_trade", False)):
        return operator_connection_message(health)
    return None


def _connection_health_backoff_active(health: dict[str, object], backoff_seconds: float) -> bool:
    if backoff_seconds <= 0 or not health or bool(health.get("can_trade", False)):
        return False
    if not bool(health.get("transient_failure", False)):
        return False
    checked_at = str(health.get("last_checked_at") or "")
    if not checked_at:
        return False
    try:
        parsed = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    age_seconds = (datetime.now(UTC) - parsed).total_seconds()
    return age_seconds < backoff_seconds


def _available_reserve_block_reason(balance: WalletBalance, amount: float, price: float) -> str | None:
    reserve_usd = float(current_app.config.get("VAULT_MIN_AVAILABLE_RESERVE_USD", 5.0) or 0.0)
    if reserve_usd <= 0:
        return None
    available = float(balance.available_balance or 0.0)
    locked = float(balance.locked_balance or 0.0)
    if locked <= 0:
        return None
    remaining_usd = max(available - amount, 0.0) * price
    if remaining_usd + 1e-9 >= reserve_usd:
        return None
    return f"Keep at least ${reserve_usd:.2f} available while another {balance.asset} vault cycle is locked. Wait for settlement or reduce this allocation."


def _active_cycle(user) -> VaultCycle | None:
    cycles = _active_cycles(user)
    return cycles[0] if cycles else None


def _active_cycles(user, *, refresh: bool = True) -> list[VaultCycle]:
    if refresh and _vault_live_api_deferred_for_request():
        refresh = False
    cycles = (
        VaultCycle.query.filter_by(user_id=user.id)
        .filter(VaultCycle.status.in_(["active", "settling"]))
        .order_by(VaultCycle.started_at.desc())
        .all()
    )
    if refresh:
        for cycle in cycles:
            if cycle.status == "active":
                _refresh_cycle_performance(cycle)
        commit_with_retry()
    return cycles


def _cycle_start_block_reason(
    user,
    asset: str,
    duration_hours: int,
    starting_value_usd: float,
    selection,
) -> str | None:
    active_cycles = _active_cycles(user, refresh=False)
    config = current_app.config
    max_active = int(config.get("VAULT_MAX_ACTIVE_CYCLES", 6) or 6)
    if len(active_cycles) >= max_active:
        return f"Maximum active vault cycles reached ({max_active}). Wait for a cycle to settle before adding another."

    per_asset = int(config.get("VAULT_MAX_ACTIVE_CYCLES_PER_ASSET", 4) or 4)
    asset_count = sum(1 for cycle in active_cycles if cycle.deposit_asset == asset)
    if asset_count >= per_asset:
        return f"Maximum active {asset} cycles reached ({per_asset})."

    balances = _wallet_balances(user)
    portfolio_total = max(_portfolio_total(balances), starting_value_usd, 1.0)
    duration_bucket = str(selection.metadata.get("ml_horizon") or horizon_from_duration(duration_hours)).lower()
    strategy_name = str(selection.strategy_name or "")
    symbols = {str(leg.get("symbol") or "").upper() for leg in (selection.legs or []) if leg.get("symbol")}
    if not symbols and selection.symbol:
        symbols.add(str(selection.symbol).upper())

    checks = [
        (
            "asset",
            config.get("VAULT_MAX_ASSET_EXPOSURE_PCT", 0.75),
            sum(float(cycle.starting_value_usd or 0.0) for cycle in active_cycles if cycle.deposit_asset == asset),
        ),
        (
            "duration",
            config.get("VAULT_MAX_DURATION_EXPOSURE_PCT", 0.70),
            sum(
                float(cycle.starting_value_usd or 0.0)
                for cycle in active_cycles
                if horizon_from_context(
                    {**cycle.selection_metadata, "algorithm_profile": cycle.algorithm_profile},
                    cycle.lock_duration_hours,
                )
                == duration_bucket
            ),
        ),
        (
            "strategy",
            config.get("VAULT_MAX_STRATEGY_EXPOSURE_PCT", 0.70),
            sum(
                float(cycle.starting_value_usd or 0.0)
                for cycle in active_cycles
                if str(cycle.selected_strategy_name or "") == strategy_name
            ),
        ),
        (
            "symbol",
            config.get("VAULT_MAX_SYMBOL_EXPOSURE_PCT", 0.70),
            _symbol_exposure_usd(active_cycles, symbols),
        ),
    ]
    for label, raw_cap, current_exposure in checks:
        cap = max(0.0, min(float(raw_cap or 0.0), 1.0))
        if cap <= 0:
            continue
        if (current_exposure + starting_value_usd) / portfolio_total > cap + 1e-9:
            return f"New cycle rejected because {label} exposure would exceed the configured vault concentration cap."

    min_score = float(config.get("VAULT_MIN_RISK_ADJUSTED_SCORE", 0.0) or 0.0)
    optimizer_score = selection.metadata.get("optimizer_score")
    edge_score = max(float(leg.get("edge_score") or 0.0) for leg in (selection.legs or [{"edge_score": 0.0}]))
    expected_score = max(float(optimizer_score or 0.0), edge_score)
    if min_score > 0 and expected_score < min_score:
        return "New cycle rejected because expected risk-adjusted return does not beat the configured vault threshold."

    return None


def _symbol_exposure_usd(active_cycles: list[VaultCycle], symbols: set[str]) -> float:
    if not symbols:
        return 0.0
    exposure = 0.0
    for cycle in active_cycles:
        cycle_symbols = {str(leg.symbol or "").upper() for leg in cycle.allocation_legs}
        if not cycle_symbols and cycle.selection_metadata.get("symbol"):
            cycle_symbols.add(str(cycle.selection_metadata.get("symbol")).upper())
        if cycle_symbols & symbols:
            exposure += float(cycle.starting_value_usd or 0.0)
    return exposure


def _sync_completed_cycles(user) -> None:
    if _vault_live_api_deferred_for_request():
        return
    now = datetime.utcnow()
    try:
        get_service("vault_cycle_trading_enforcer").enforce_active_cycles(user.id)
    except OperationalError as exc:
        if not is_database_locked(exc):
            raise
        db.session.rollback()
        current_app.logger.warning("Deferred Vault Cycle active trading enforcement because SQLite is locked: %s", exc)
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.warning("Deferred Vault Cycle active trading enforcement: %s", exc)
    try:
        engine_results = get_service("vault_cycle_orchestrator").resume_due_cycles(user.id)
        if engine_results:
            commit_with_retry()
    except OperationalError as exc:
        if not is_database_locked(exc):
            raise
        db.session.rollback()
        current_app.logger.warning("Deferred Vault Cycle engine settlement because SQLite is locked: %s", exc)
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.warning("Deferred Vault Cycle engine settlement: %s", exc)
    cycles = VaultCycle.query.filter_by(user_id=user.id).filter(VaultCycle.status == "active", VaultCycle.unlocks_at <= now).all()
    cycles = [cycle for cycle in cycles if not get_service("vault_cycle_settlement").is_vault_cycle_engine_cycle(cycle)]
    if not cycles:
        return
    manager = get_service("strategy_manager")
    run_ids: set[int] = set()
    for cycle in cycles:
        run_ids.update(leg.strategy_run_id for leg in cycle.allocation_legs if leg.strategy_run_id)
        if cycle.strategy_run_id:
            run_ids.add(cycle.strategy_run_id)

    try:
        for run_id in run_ids:
            manager.stop(run_id)
    except OperationalError as exc:
        if not is_database_locked(exc):
            raise
        db.session.rollback()
        current_app.logger.warning("Deferred vault cycle settlement because SQLite is locked: %s", exc)
        return

    for cycle in cycles:
        cycle.execution_substatus = "settling"
        for leg in cycle.allocation_legs:
            leg.status = "complete"
        deposit_balance = WalletBalance.query.filter_by(user_id=user.id, asset=cycle.deposit_asset).one_or_none()
        if deposit_balance is not None:
            deposit_balance.locked_balance = max(0.0, float(deposit_balance.locked_balance or 0.0) - cycle.deposit_amount)
            deposit_price = _asset_usd_price(cycle.deposit_asset)
            if deposit_price > 0:
                deposit_balance.estimated_usd_value = deposit_balance.total_balance * deposit_price

        settlement_balance = WalletBalance.query.filter_by(user_id=user.id, asset=cycle.settlement_asset).one_or_none()
        if settlement_balance is None:
            settlement_balance = WalletBalance(user_id=user.id, asset=cycle.settlement_asset)
            db.session.add(settlement_balance)
        _refresh_cycle_performance(cycle)
        settlement_price = _asset_usd_price(cycle.settlement_asset)
        cycle.final_settlement_amount = (
            cycle.current_estimated_value_usd / settlement_price if settlement_price > 0 else cycle.deposit_amount
        )
        settlement_balance.available_balance = float(settlement_balance.available_balance or 0.0) + cycle.final_settlement_amount
        if settlement_price > 0:
            settlement_balance.estimated_usd_value = settlement_balance.total_balance * settlement_price
        cycle.status = "complete"
        cycle.execution_substatus = "complete"
        cycle.settled_at = now
        cycle.cycle_summary = _cycle_summary(cycle)
        db.session.add(
            WalletTransaction(
                vault_cycle_id=cycle.id,
                user_id=user.id,
                asset=cycle.settlement_asset,
                amount=cycle.final_settlement_amount,
                transaction_type="settlement",
                status="complete",
                note="Vault cycle settled to the selected asset.",
            )
        )
        _learn_from_completed_cycle(cycle)
    try:
        commit_with_retry()
    except OperationalError as exc:
        if not is_database_locked(exc):
            raise
        db.session.rollback()
        current_app.logger.warning("Deferred vault cycle settlement because SQLite is locked: %s", exc)


def _selected_network(asset: str) -> str:
    requested = request.values.get("network", "").strip()
    networks = _asset_networks(asset)
    return requested if requested in networks else networks[0]


def _asset_networks(asset: str) -> tuple[str, ...]:
    return vault_asset_networks(asset, _configured_asset_networks(asset))


def _functional_wallet_network(asset: str, network: str) -> bool:
    return functional_wallet_network(asset, network)


def _wallet_assets() -> tuple[str, ...]:
    return supported_vault_allocation_assets(_configured_wallet_assets())


def _configured_wallet_assets() -> tuple[str, ...]:
    try:
        return tuple(get_service("wallet_address_service").configured_assets())
    except Exception:  # noqa: BLE001
        return ()


def _configured_asset_networks(asset: str) -> tuple[str, ...]:
    try:
        return tuple(get_service("wallet_address_service").configured_networks(asset))
    except Exception:  # noqa: BLE001
        return ()


def _is_supported_wallet_asset(asset: str) -> bool:
    return asset in _wallet_assets()


def _active_deposit_address(user_id: int, asset: str, network: str) -> DepositAddress | None:
    return (
        DepositAddress.query.filter_by(user_id=user_id, asset=asset, network=network, is_active=True)
        .order_by(DepositAddress.version.desc())
        .first()
    )


def _ensure_deposit_address(
    user_id: int,
    asset: str,
    network: str,
    balance: WalletBalance | None = None,
    *,
    commit_link: bool = True,
) -> DepositAddress | None:
    address = _active_deposit_address(user_id, asset, network)
    if address is None:
        address = _new_deposit_address(user_id, asset, network)
    if address is None:
        return None
    if balance is not None and balance.active_deposit_address_id != address.id:
        balance.active_deposit_address_id = address.id
        if commit_link:
            commit_with_retry()
    return address


def _new_deposit_address(
    user_id: int,
    asset: str,
    network: str,
    rotated_from: DepositAddress | None = None,
) -> DepositAddress | None:
    latest = DepositAddress.query.filter_by(user_id=user_id, asset=asset, network=network).order_by(DepositAddress.version.desc()).first()
    version = (latest.version if latest is not None else 0) + 1
    try:
        configured_address = generate_deposit_address(asset, user_id, network, force_new=rotated_from is not None)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Deposit address generation failed for %s/%s.", asset, network)
        db.session.add(
            AuditLog(
                category="wallet",
                action="deposit_address_generation_failed",
                message=f"Deposit address generation failed for {asset} on {network}: {exc}",
            )
        )
        return None
    existing_address = DepositAddress.query.filter_by(address=configured_address).one_or_none()
    if existing_address is not None:
        db.session.add(
            AuditLog(
                category="wallet",
                action="deposit_address_generation_duplicate",
                message=f"Deposit address generation returned an already registered {asset} address on {network}.",
            )
        )
        return None
    address = DepositAddress(
        user_id=user_id,
        asset=asset,
        network=network,
        version=version,
        address=configured_address,
        rotated_from_id=rotated_from.id if rotated_from is not None else None,
        is_active=True,
    )
    db.session.add(address)
    db.session.flush()
    wallet_address = None
    custody = get_service("wallet_custody")
    if custody.enabled and custody.supports(asset, network):
        wallet_address = WalletAddress.query.filter_by(
            user_id=user_id,
            asset=asset,
            network=network,
            address=configured_address,
        ).one_or_none()
    if wallet_address is not None:
        wallet_address.deposit_address_id = address.id
    try:
        get_service("self_custody_wallet").record_public_address(
            user_id,
            asset,
            network,
            configured_address,
            deposit_address_id=address.id,
        )
    except Exception:  # noqa: BLE001
        current_app.logger.exception("Failed to record public wallet address metadata.")
    return address


def _deactivate_wallet_address_for_deposit(deposit_address: DepositAddress) -> None:
    wallet_address = (
        WalletAddress.query.filter_by(
            user_id=deposit_address.user_id,
            asset=deposit_address.asset,
            network=deposit_address.network,
            address=deposit_address.address,
        )
        .order_by(WalletAddress.rotation_index.desc())
        .first()
    )
    if wallet_address is not None:
        wallet_address.status = "inactive"


def _asset_usd_price(asset: str) -> float:
    return shared_asset_usd_price(
        asset,
        lambda asset_key: float(get_service("market_data").get_mid_price(asset_key, market_mode_for(get_current_mode())) or 0.0),
    )


def _requested_duration_seconds() -> int:
    value = str(_request_value("lock_duration", "24"))
    if value.strip().lower() in {"1", "1h10", "one_h10"}:
        return _one_h10_horizon_seconds()
    if value == "custom":
        raw_value = _request_value("custom_duration_value", "") or _request_value("custom_duration_hours", "24")
        unit = str(_request_value("custom_duration_unit", "hours"))
        try:
            amount = float(raw_value)
        except ValueError:
            return 0
        multiplier = 60 if unit == "minutes" else 3600
        return max(60, min(int(amount * multiplier), 24 * 30 * 3600))
    try:
        hours = int(float(value))
    except ValueError:
        return 0
    return max(3600, min(hours * 3600, 24 * 30 * 3600))


def _requested_allowed_symbols() -> list[str]:
    raw_values = _request_values("allowed_symbols")
    if not raw_values:
        raw_values = [_request_value("allowed_symbols", "")]
    symbols: list[str] = []
    for raw in raw_values:
        for item in str(raw or "").split(","):
            symbol = item.strip().upper()
            if symbol:
                symbols.append(symbol)
    configured = [str(symbol).upper() for symbol in current_app.config.get("ALLOWED_SYMBOLS", ["BTC"]) if str(symbol).strip()]
    allowed = set(configured)
    if not symbols:
        return configured
    selected = [symbol for symbol in dict.fromkeys(symbols) if symbol in allowed]
    return selected or configured


def _estimated_cycle_value(cycle: VaultCycle, performance: dict[str, float | bool] | None = None) -> float:
    performance = performance if performance is not None else _cycle_performance(cycle)
    if performance["has_trading_data"]:
        return max(float(cycle.starting_value_usd or 0.0) + performance["total_pnl"], 0.0)
    symbol = cycle.selection_metadata.get("symbol") or cycle.selection_metadata.get("asset", cycle.deposit_asset)
    start = max(float(cycle.starting_value_usd or 0.0), 0.0)
    if start <= 0:
        return 0.0
    if str(symbol).upper() in {"USDC", "USDT"}:
        return start
    try:
        candles = get_service("market_data").get_candles(symbol, "15m", mode=market_mode_for(get_current_mode()), limit=24)
        closes = [float(candle["close"]) for candle in candles if float(candle.get("close", 0.0)) > 0]
    except Exception:  # noqa: BLE001
        return start
    if len(closes) < 2 or closes[0] <= 0:
        return start
    market_change = (closes[-1] - closes[0]) / closes[0]
    return max(start * (1 + market_change * 0.25), 0.0)


def _refresh_cycle_performance(cycle: VaultCycle) -> dict[str, float | bool]:
    performance = _cycle_performance(cycle)
    if _cycle_live_data_backoff_active(cycle):
        cycle.current_estimated_value_usd = max(
            float(cycle.current_estimated_value_usd or 0.0) or float(cycle.starting_value_usd or 0.0) + float(performance["total_pnl"]),
            0.0,
        )
    else:
        cycle.current_estimated_value_usd = _estimated_cycle_value(cycle, performance)
    metadata = cycle.selection_metadata
    metadata["realized_pnl_usd"] = performance["realized_pnl"]
    metadata["unrealized_pnl_usd"] = performance["unrealized_pnl"]
    metadata["total_pnl_usd"] = performance["total_pnl"]
    cycle.selection_metadata = metadata
    return performance


def _learn_from_completed_cycle(cycle: VaultCycle) -> None:
    config = getattr(get_service("online_ranker"), "config", {})
    if not bool(config.get("ML_RANKER_ENABLED", False)):
        return
    mode = cycle.strategy_run.mode if cycle.strategy_run else cycle.execution_mode
    ranker = get_service("online_ranker")
    if not ranker.should_update_from_mode(mode):
        return
    metadata = cycle.selection_metadata
    starting = max(float(cycle.starting_value_usd or 0.0), 1.0)
    total_pnl = float(metadata.get("total_pnl_usd", 0.0) or 0.0)
    return_after_costs = total_pnl / starting
    horizon = horizon_from_context(
        {
            **metadata,
            "algorithm_profile": cycle.algorithm_profile,
            "lock_duration_hours": cycle.lock_duration_hours,
            "lock_duration_seconds": cycle.lock_duration_seconds,
        },
        cycle.lock_duration_hours,
    )
    features = extract_features(
        {
            **metadata,
            "strategy_name": cycle.selected_strategy_name,
            "symbol": metadata.get("symbol") or cycle.deposit_asset,
            "timeframe": cycle.selected_timeframe,
            "optimizer_profile": metadata.get("optimizer_profile"),
            "lock_duration_hours": cycle.lock_duration_hours,
            "lock_duration_seconds": cycle.lock_duration_seconds,
            "horizon": horizon,
            "net_return_after_costs": return_after_costs,
            "total_return": return_after_costs,
            "starting_value_usd": cycle.starting_value_usd,
            "allocation_amount_usd": cycle.starting_value_usd,
            "trade_count": len(cycle.transactions or []),
        }
    )
    ranker.update(
        features,
        outcome_from_result(
            {
                "net_return_after_costs": return_after_costs,
                "total_return": return_after_costs,
                "recent_performance_score": return_after_costs,
                "profit_factor": 1.2 if total_pnl > 0 else 0.8,
                "consistency": 1.0 if total_pnl > 0 else 0.0,
                "window_stability": 1.0,
                "trade_count": max(len(cycle.allocation_legs or []), 1),
                "edge_score": metadata.get("edge_score", 0.0),
                "cost_drag_bps": metadata.get("estimated_slippage_bps", 0.0),
            }
        ),
        horizon=horizon,
        source="vault_cycle",
        source_id=cycle.id,
        mode=mode,
        metadata={
            "cycle_id": cycle.id,
            "strategy_name": cycle.selected_strategy_name,
            "profile": cycle.algorithm_profile,
            "total_pnl_usd": total_pnl,
        },
    )
    if bool(config.get("ENSEMBLE_LEARNING_ENABLED", True)) and metadata.get("ensemble_version"):
        decay = float(config.get("ENSEMBLE_LEARNING_DECAY", 0.8) or 0.8)
        outcome_payload = {
            "net_return_after_costs": return_after_costs,
            "total_return": return_after_costs,
            "recent_performance_score": return_after_costs,
            "profit_factor": 1.2 if total_pnl > 0 else 0.8,
            "consistency": 1.0 if total_pnl > 0 else 0.0,
            "window_stability": 1.0,
            "trade_count": max(len(cycle.allocation_legs or []), 1),
            "edge_score": metadata.get("edge_score", 0.0),
            "cost_drag_bps": metadata.get("estimated_slippage_bps", 0.0),
        }
        ranker.update_contextual_bandit(
            features,
            outcome_from_result(outcome_payload),
            horizon=horizon,
            source="vault_cycle_ensemble",
            source_id=cycle.id,
            mode=mode,
            decay=decay,
            metadata={
                "cycle_id": cycle.id,
                "ensemble_id": metadata.get("ensemble_id"),
                "ensemble_version": metadata.get("ensemble_version"),
                "total_pnl_usd": total_pnl,
            },
        )
        for leg in cycle.allocation_legs:
            leg_details = leg.details
            allocation = max(float(leg.allocation_cap_usd or 0.0), 1.0)
            leg_pnl = float(leg.realized_pnl_usd or 0.0) + float(leg.unrealized_pnl_usd or 0.0)
            leg_return = leg_pnl / allocation
            leg_features = extract_features(
                {
                    **metadata,
                    **leg_details,
                    "strategy_name": leg.strategy_run.strategy_name if leg.strategy_run else None,
                    "symbol": leg.symbol,
                    "timeframe": leg.timeframe,
                    "optimizer_profile": leg_details.get("optimizer_profile") or metadata.get("optimizer_profile"),
                    "lock_duration_hours": cycle.lock_duration_hours,
                    "horizon": horizon,
                    "net_return_after_costs": leg_return,
                    "total_return": leg_return,
                    "allocation_amount_usd": allocation,
                    "ensemble_weight": leg_details.get("ensemble_weight"),
                }
            )
            ranker.update_contextual_bandit(
                leg_features,
                outcome_from_result(
                    {
                        "net_return_after_costs": leg_return,
                        "total_return": leg_return,
                        "recent_performance_score": leg_return,
                        "profit_factor": 1.2 if leg_pnl > 0 else 0.8,
                        "consistency": 1.0 if leg_pnl > 0 else 0.0,
                        "window_stability": 1.0,
                        "trade_count": 1,
                        "edge_score": leg_details.get("edge_score", 0.0),
                    }
                ),
                horizon=horizon,
                source="vault_leg",
                source_id=leg.id,
                mode=mode,
                decay=decay,
                metadata={
                    "cycle_id": cycle.id,
                    "leg_id": leg.id,
                    "ensemble_id": metadata.get("ensemble_id"),
                    "ensemble_version": metadata.get("ensemble_version"),
                    "strategy_name": leg.strategy_run.strategy_name if leg.strategy_run else None,
                    "symbol": leg.symbol,
                    "leg_pnl_usd": leg_pnl,
                },
            )


def _cycle_realized_totals(cycle_orders: list[Order]) -> tuple[float, dict[int, float], dict[str, float]]:
    realized = 0.0
    leg_totals: dict[int, float] = {}
    symbol_totals: dict[str, float] = {}
    for order in cycle_orders:
        order_pnl = sum(
            float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0) for fill in order.fills
        )
        realized += order_pnl
        leg_id = _order_vault_leg_id(order)
        if leg_id:
            try:
                leg_totals[int(leg_id)] = leg_totals.get(int(leg_id), 0.0) + order_pnl
                continue
            except (TypeError, ValueError):
                pass
        symbol_key = str(order.symbol or "").upper()
        if symbol_key:
            symbol_totals[symbol_key] = symbol_totals.get(symbol_key, 0.0) + order_pnl
    return realized, leg_totals, symbol_totals


def _cycle_live_data_backoff_active(cycle: VaultCycle) -> bool:
    if str(cycle.execution_mode or "").lower() != "live":
        return False
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return False
    if _cycle_one_h10_runtime_notice(cycle):
        return True
    for leg in cycle.allocation_legs:
        provider = str(leg.details.get("provider") or leg.provider or "").strip()
        connection_id = _cycle_leg_connection_id(cycle, leg)
        if provider and _one_h10_provider_market_data_backoff(provider, connection_id):
            return True
    return False


def _cycle_leg_connection_id(cycle: VaultCycle, leg: VaultAllocationLeg | None) -> int | None:
    raw = None
    if leg is not None:
        raw = leg.trading_connection_id or leg.details.get("trading_connection_id")
    raw = raw or cycle.trading_connection_id
    try:
        return int(raw) if raw is not None and str(raw).strip() else None
    except (TypeError, ValueError):
        return None


def _position_lookup_keys(*values: object) -> list[str]:
    keys: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        keys.append(text)
        upper = text.upper()
        if upper != text:
            keys.append(upper)
    return list(dict.fromkeys(keys))


def _cycle_live_positions_by_connection(cycle: VaultCycle) -> dict[tuple[int, str], dict[str, object]]:
    positions: dict[tuple[int, str], dict[str, object]] = {}
    if str(cycle.execution_mode or "").lower() != "live":
        return positions
    service = get_service("trading_connections")
    connection_rows: dict[int, str] = {}
    for leg in cycle.allocation_legs:
        connection_id = _cycle_leg_connection_id(cycle, leg)
        if connection_id is None:
            continue
        provider = str(leg.details.get("provider") or leg.provider or "").strip()
        connection_rows[connection_id] = provider
    if not connection_rows and cycle.trading_connection_id:
        connection_rows[int(cycle.trading_connection_id)] = ""
    for connection_id, provider in connection_rows.items():
        if provider and _one_h10_provider_market_data_backoff(provider, connection_id):
            continue
        try:
            snapshot = service.account_snapshot(cycle.user_id, "live", connection_id)
        except Exception:  # noqa: BLE001
            continue
        if getattr(snapshot, "alerts", None) and not getattr(snapshot, "positions", None):
            continue
        for position in getattr(snapshot, "positions", []) or []:
            if not isinstance(position, dict):
                continue
            for key in _position_lookup_keys(
                position.get("symbol"),
                position.get("venue_symbol"),
                position.get("coin"),
            ):
                positions[(connection_id, key)] = position
    return positions


def _position_for_leg(
    leg: VaultAllocationLeg,
    positions_by_connection: dict[tuple[int, str], dict[str, object]],
    fallback_connection_id: int | None,
) -> tuple[dict[str, object] | None, int | None, str | None]:
    connection_id = _cycle_leg_connection_id(leg.vault_cycle, leg) or fallback_connection_id
    if connection_id is None:
        return None, None, None
    details = leg.details
    for key in _position_lookup_keys(
        details.get("venue_symbol"),
        details.get("provider_symbol"),
        details.get("app_symbol"),
        leg.symbol,
    ):
        position = positions_by_connection.get((connection_id, key))
        if position is not None:
            return position, connection_id, key
    return None, connection_id, None


def _last_known_cycle_performance(
    cycle: VaultCycle,
    cycle_orders: list[Order],
    realized: float,
    leg_totals: dict[int, float],
    symbol_totals: dict[str, float],
) -> dict[str, float | bool]:
    for leg in cycle.allocation_legs:
        leg.realized_pnl_usd = leg_totals.get(leg.id, symbol_totals.get(str(leg.symbol or "").upper(), 0.0))
    metadata = cycle.selection_metadata
    leg_unrealized = sum(float(leg.unrealized_pnl_usd or 0.0) for leg in cycle.allocation_legs)
    unrealized = float(metadata.get("unrealized_pnl_usd", leg_unrealized) or 0.0)
    return {
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "has_trading_data": bool(cycle_orders) or abs(unrealized) > 1e-12,
    }


def _cycle_performance(cycle: VaultCycle) -> dict[str, float | bool]:
    cycle_orders = _cycle_orders(cycle)
    realized, leg_totals, symbol_totals = _cycle_realized_totals(cycle_orders)
    if _cycle_live_data_backoff_active(cycle):
        return _last_known_cycle_performance(cycle, cycle_orders, realized, leg_totals, symbol_totals)

    unrealized = 0.0
    mode = cycle.strategy_run.mode if cycle.strategy_run else cycle.execution_mode
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        symbols = {leg.symbol for leg in cycle.allocation_legs if leg.symbol}
        if not symbols and cycle.strategy_run:
            symbols.add(cycle.strategy_run.symbol)
        for symbol in symbols:
            try:
                position = get_service("order_manager").current_position(
                    symbol,
                    mode,
                    cycle.user_id,
                    cycle.trading_connection_id,
                )
                unrealized += float(position.get("unrealized_pnl", 0.0) or 0.0)
            except Exception:  # noqa: BLE001
                continue
        for leg in cycle.allocation_legs:
            leg.realized_pnl_usd = leg_totals.get(leg.id, symbol_totals.get(str(leg.symbol or "").upper(), 0.0))
            try:
                position = get_service("order_manager").current_position(
                    leg.symbol,
                    mode,
                    cycle.user_id,
                    cycle.trading_connection_id,
                )
                leg.unrealized_pnl_usd = float(position.get("unrealized_pnl", 0.0) or 0.0)
            except Exception:  # noqa: BLE001
                leg.unrealized_pnl_usd = 0.0
        return {
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
            "has_trading_data": bool(cycle_orders) or abs(unrealized) > 1e-12,
        }

    positions_by_connection = _cycle_live_positions_by_connection(cycle) if str(mode or "").lower() == "live" else {}
    counted_positions: set[tuple[int, str]] = set()

    for leg in cycle.allocation_legs:
        leg.realized_pnl_usd = leg_totals.get(leg.id, symbol_totals.get(str(leg.symbol or "").upper(), 0.0))
        position, connection_id, matched_key = _position_for_leg(leg, positions_by_connection, cycle.trading_connection_id)
        if position is None and cycle_orders:
            connection_id = _cycle_leg_connection_id(cycle, leg)
            try:
                position = get_service("order_manager").current_position(
                    leg.symbol,
                    mode,
                    cycle.user_id,
                    connection_id,
                )
                matched_key = str(position.get("symbol") or leg.symbol or "").upper()
            except Exception:  # noqa: BLE001
                position = None
        if position is None:
            leg.unrealized_pnl_usd = 0.0
            continue
        leg.unrealized_pnl_usd = float(position.get("unrealized_pnl", 0.0) or 0.0)
        position_key = str(position.get("symbol") or matched_key or leg.symbol or "").upper()
        if connection_id is not None and (connection_id, position_key) not in counted_positions:
            unrealized += float(position.get("unrealized_pnl", 0.0) or 0.0)
            counted_positions.add((connection_id, position_key))
    return {
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "has_trading_data": bool(cycle_orders) or abs(unrealized) > 1e-12,
    }


def _cycle_orders(cycle: VaultCycle) -> list[Order]:
    query = Order.query.filter_by(user_id=cycle.user_id)
    if cycle.execution_mode:
        query = query.filter_by(mode=cycle.execution_mode)
    query = query.filter(or_(Order.vault_cycle_id == cycle.id, Order.vault_cycle_id.is_(None)))
    return [order for order in query.order_by(Order.created_at.asc()).all() if _order_vault_cycle_id(order) == cycle.id]


def _order_vault_cycle_id(order: Order) -> int | None:
    raw = order.vault_cycle_id
    if raw is None:
        raw = order.details.get("vault_cycle_id")
    try:
        return int(raw) if raw is not None and str(raw).strip() else None
    except (TypeError, ValueError):
        return None


def _order_vault_leg_id(order: Order) -> int | None:
    raw = order.vault_leg_id
    if raw is None:
        raw = order.details.get("vault_leg_id")
    try:
        return int(raw) if raw is not None and str(raw).strip() else None
    except (TypeError, ValueError):
        return None


def _cycle_summary(cycle: VaultCycle, *, performance: dict[str, float | bool] | None = None) -> dict[str, object]:
    performance = performance if performance is not None else _cycle_performance(cycle)
    orders = _cycle_orders(cycle)
    order_summaries = [_order_summary(order) for order in orders]
    fills = [fill for order in orders for fill in order.fills]
    fees = sum(float(fill.fee or 0.0) + float(getattr(fill, "funding_fee", 0.0) or 0.0) for fill in fills)
    slippage_values = [
        float(order.details.get("slippage_bps", 0.0) or 0.0) for order in orders if order.details.get("slippage_bps") is not None
    ]
    legs = [_leg_summary(leg) for leg in cycle.allocation_legs]
    leverages = [float(leg.get("leverage", 1.0) or 1.0) for leg in legs]
    leverages.extend(float(order.get("leverage", 1.0) or 1.0) for order in order_summaries)
    if not leverages:
        leverages = [1.0]
    reward_risks = [float(order["risk_reward"]) for order in order_summaries if order.get("risk_reward")]
    if not reward_risks:
        reward_risks = [float(leg.get("reward_risk", 0.0) or 0.0) for leg in legs if leg.get("reward_risk")]
    starting_value = float(cycle.starting_value_usd or 0.0)
    settlement_price = _asset_usd_price(cycle.settlement_asset) or 1.0
    final_settlement_amount = float(cycle.final_settlement_amount or 0.0)
    final_settlement_value = final_settlement_amount
    if final_settlement_value <= 0:
        final_settlement_value = float(cycle.current_estimated_value_usd or 0.0) / max(settlement_price, 1e-9)
    final_settlement_usd = final_settlement_value * settlement_price
    roi_pct = ((final_settlement_usd - starting_value) / max(starting_value, 1e-9)) * 100.0 if starting_value > 0 else 0.0
    target_amount_usd = float(
        cycle.selection_metadata.get(
            "target_amount_usd",
            starting_value * (10.0 if str(cycle.algorithm_profile).upper() == "1H10" else 1.0),
        )
        or 0.0
    )
    ml_readiness = _cycle_effective_ml_readiness(cycle)
    blocker_categories = _cycle_blocker_categories(cycle, orders)
    ranked_candidates = _cycle_ranked_candidates(cycle)
    rejected_intents = [order for order in order_summaries if str(order.get("status") or "").lower() in {"rejected", "failed"}]
    repairable_no_order = _cycle_repairable_no_order(cycle, orders)
    runtime_notice = _cycle_one_h10_runtime_notice(cycle)
    trade_decision_legs = _cycle_trade_decision_legs(cycle, order_summaries, legs)
    trade_decision = _cycle_trade_decision(cycle, order_summaries, trade_decision_legs, runtime_notice)
    worker_status = _cycle_worker_status(cycle, trade_decision_legs)
    raw_no_order_reason = cycle.selection_metadata.get("no_order_failure_reason") or cycle.validation_failure_reason
    no_order_failure_reason = _sanitize_cycle_reason(raw_no_order_reason) if repairable_no_order else None
    summary = {
        "cycle_id": cycle.id,
        "status": cycle.status,
        "execution_substatus": cycle.execution_substatus,
        "no_order_failure_reason": no_order_failure_reason,
        "repairable_no_order": repairable_no_order,
        "runtime_notice": runtime_notice,
        "trade_decision": trade_decision,
        "trade_decision_legs": trade_decision_legs,
        "worker": worker_status,
        "live_order_path": worker_status["live_order_path"],
        "deposit_asset": cycle.deposit_asset,
        "deposit_amount": float(cycle.deposit_amount or 0.0),
        "settlement_asset": cycle.settlement_asset,
        "lock_duration_seconds": int(cycle.lock_duration_seconds or cycle.lock_duration_hours * 3600),
        "lock_duration_label": format_duration_seconds(cycle.lock_duration_seconds or cycle.lock_duration_hours * 3600),
        "starting_value_usd": float(cycle.starting_value_usd or 0.0),
        "input_amount": float(cycle.deposit_amount or 0.0),
        "input_amount_usd": starting_value,
        "target_amount_usd": target_amount_usd,
        "target_amount": target_amount_usd / max(settlement_price, 1e-9) if target_amount_usd > 0 else 0.0,
        "target_roi_pct": float(cycle.selection_metadata.get("target_roi_pct", 0.0) or 0.0),
        "current_estimated_value_usd": float(cycle.current_estimated_value_usd or 0.0),
        "final_settlement_amount": final_settlement_value,
        "final_settlement_value_usd": final_settlement_usd,
        "roi_pct": roi_pct,
        "execution_mode": cycle.execution_mode,
        "algorithm_profile": cycle.algorithm_profile,
        "symbols": sorted({order.symbol for order in orders} | {leg.symbol for leg in cycle.allocation_legs}),
        "sides": sorted({order.side for order in orders}),
        "order_count": len(orders),
        "trades_taken": len(orders),
        "fill_count": len(fills),
        "fees_usd": fees,
        "realized_pnl_usd": float(performance["realized_pnl"]),
        "unrealized_pnl_usd": float(performance["unrealized_pnl"]),
        "total_pnl_usd": float(performance["total_pnl"]),
        "max_leverage": max(leverages),
        "avg_leverage": sum(leverages) / len(leverages),
        "risk_reward": (sum(reward_risks) / len(reward_risks)) if reward_risks else 0.0,
        "drawdown": float(cycle.selection_metadata.get("max_drawdown", 0.0) or 0.0),
        "slippage_bps": (sum(slippage_values) / len(slippage_values))
        if slippage_values
        else float(cycle.selection_metadata.get("estimated_slippage_bps", 0.0) or 0.0),
        "execution_styles": sorted({str(leg.get("execution_style") or "") for leg in legs if leg.get("execution_style")}),
        "provider_allocation_history": cycle.selection_metadata.get(
            "provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])
        ),
        "exchange_allocation_history": cycle.selection_metadata.get("exchange_allocation_history", []),
        "provider_skip_reasons": cycle.selection_metadata.get("provider_skip_reasons", []),
        "market_discovery": cycle.selection_metadata.get("market_discovery", []),
        "feature_diagnostics": _cycle_feature_diagnostics(cycle),
        "forecast_blockers": _cycle_forecast_blockers(cycle),
        "forecast_advisory_blockers": _cycle_forecast_advisory_blockers(cycle),
        "ml_readiness": ml_readiness,
        "ml_blockers": cycle.selection_metadata.get("ml_blockers", ml_readiness.get("blockers", [])),
        "risk_blockers": cycle.selection_metadata.get("risk_blockers", []),
        "blocker_categories": blocker_categories,
        "ranked_candidates": ranked_candidates,
        "skipped_symbols": _cycle_skipped_symbols(cycle),
        "rejected_intents": rejected_intents,
        "submitted_order_count": sum(
            1 for order in order_summaries if str(order.get("status") or "").lower() in {"submitted", "open", "filled"}
        ),
        "failed_order_count": sum(1 for order in order_summaries if str(order.get("status") or "").lower() == "failed"),
        "rejected_order_count": sum(1 for order in order_summaries if str(order.get("status") or "").lower() == "rejected"),
        "orders": order_summaries,
        "legs": legs,
        "generated_at": datetime.utcnow().isoformat(),
    }
    summary.update(extract_cycle_coherence_payload(cycle))
    if get_service("vault_cycle_settlement").is_vault_cycle_engine_cycle(cycle):
        vault_cycle_payload = get_service("vault_cycle_reporting").status_payload(cycle)
        summary["vault_cycle_engine"] = True
        summary["allocations"] = vault_cycle_payload.get("allocations", [])
        summary["transfers"] = vault_cycle_payload.get("transfers", [])
        summary["risk_events"] = vault_cycle_payload.get("risk_events", [])
        summary["settlement"] = vault_cycle_payload.get("settlement", {})
    return summary


def _cycle_ranked_candidates(cycle: VaultCycle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for provider in (
        cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []
    ):
        if not isinstance(provider, dict):
            continue
        for leg in provider.get("legs", []) or []:
            if not isinstance(leg, dict):
                continue
            rows.append(
                {
                    "provider": provider.get("provider"),
                    "trading_connection_id": provider.get("trading_connection_id"),
                    "symbol": leg.get("symbol"),
                    "venue_symbol": leg.get("venue_symbol"),
                    "market_id": leg.get("market_id"),
                    "allocation_cap_usd": float(leg.get("allocation_cap_usd", 0.0) or 0.0),
                    "scanner_score": float(leg.get("scanner_score", 0.0) or 0.0),
                    "scanner_source": leg.get("scanner_source"),
                    "score_breakdown": leg.get("score_breakdown", {}),
                    "forecast": leg.get("forecast", {}),
                }
            )
    return sorted(rows, key=lambda item: float(item.get("scanner_score", 0.0) or 0.0), reverse=True)


def _cycle_chart_payload(cycle: VaultCycle, summary: dict[str, object]) -> dict[str, object]:
    starting_value = max(float(summary.get("starting_value_usd", cycle.starting_value_usd) or 0.0), 0.0)
    current_value = max(float(summary.get("current_estimated_value_usd", cycle.current_estimated_value_usd) or 0.0), 0.0)
    value_points: list[dict[str, object]] = []
    pnl_points: list[dict[str, object]] = []
    cumulative_pnl = 0.0
    if cycle.started_at:
        value_points.append({"t": cycle.started_at.isoformat(), "value": round(starting_value, 2)})
        pnl_points.append({"t": cycle.started_at.isoformat(), "value": 0.0})

    orders = [order for order in (summary.get("orders") or []) if isinstance(order, dict)]
    for order in orders:
        order_time = order.get("created_at")
        if not order_time:
            continue
        cumulative_pnl += float(order.get("realized_pnl", 0.0) or 0.0)
        value_points.append({"t": order_time, "value": round(max(starting_value + cumulative_pnl, 0.0), 2)})
        pnl_points.append({"t": order_time, "value": round(cumulative_pnl, 2)})

    end_time = cycle.settled_at or cycle.updated_at or datetime.utcnow()
    value_points.append({"t": end_time.isoformat(), "value": round(current_value or starting_value + cumulative_pnl, 2)})
    pnl_points.append({"t": end_time.isoformat(), "value": round(float(summary.get("total_pnl_usd", cumulative_pnl) or 0.0), 2)})

    allocation_rows: list[dict[str, object]] = []
    raw_allocations = summary.get("allocations") or summary.get("legs") or []
    total_allocation = 0.0
    allocation_source = raw_allocations if isinstance(raw_allocations, list) else []
    for row in allocation_source:
        if not isinstance(row, dict):
            continue
        amount = float(row.get("allocated_amount", row.get("allocation_cap_usd", 0.0)) or 0.0)
        total_allocation += max(amount, 0.0)
        allocation_rows.append(
            {
                "label": str(row.get("provider") or row.get("symbol") or row.get("strategy_name") or "Allocation"),
                "asset": row.get("collateral_asset") or row.get("settlement_asset"),
                "value": round(max(amount, 0.0), 2),
                "score": round(float(row.get("risk_adjusted_score", row.get("scanner_score", 0.0)) or 0.0), 4),
            }
        )
    for row in allocation_rows:
        row["pct"] = round((float(row["value"]) / total_allocation) * 100.0, 2) if total_allocation > 0 else 0.0

    timeline = [
        {
            "t": order.get("created_at"),
            "label": f"{str(order.get('side') or '').upper()} {order.get('symbol')}",
            "status": order.get("status"),
            "pnl": round(float(order.get("realized_pnl", 0.0) or 0.0), 2),
        }
        for order in orders[:12]
    ]
    return {
        "value": {
            "points": value_points[-18:],
            "summary": "Cycle value path from allocation, order, and latest estimate snapshots.",
            "empty": len(value_points) < 2,
        },
        "pnl": {
            "points": pnl_points[-18:],
            "summary": "Realized and estimated cycle PnL over available order events.",
            "empty": len(pnl_points) < 2,
        },
        "allocations": allocation_rows,
        "timeline": timeline,
    }


def _cycle_repairable_no_order(cycle: VaultCycle, orders: list[Order]) -> bool:
    if orders:
        return False
    if not (cycle.selection_metadata.get("no_order_failure_reason") or cycle.validation_failure_reason):
        return False
    status = str(cycle.status or "").lower()
    substatus = str(cycle.execution_substatus or "").lower()
    return status in {"complete", "failed"} or substatus in {"limited", "failed_no_execution", "error"}


def _cycle_worker_status(cycle: VaultCycle, leg_decisions: list[dict[str, object]]) -> dict[str, object]:
    statuses = [str(leg.strategy_run.status or "").lower() for leg in cycle.allocation_legs if leg.strategy_run is not None]
    queued_count = sum(1 for status in statuses if status == "queued")
    starting_count = sum(1 for status in statuses if status == "starting")
    running_count = sum(1 for status in statuses if status == "running")
    decision_queued_count = sum(1 for row in leg_decisions if row.get("stage") == "queued_for_worker")
    in_process = in_process_workers_enabled(current_app.config)
    queue = "in_process" if in_process else "dedicated_worker"
    return {
        "worker_mode": str(current_app.config.get("WORKER_MODE", "web") or "web"),
        "worker_process_configured": bool(current_app.config.get("WORKER_PROCESS_CONFIGURED", False)),
        "in_process_workers_enabled": bool(in_process),
        "strategy_run_queue": queue,
        "strategy_run_count": len(statuses),
        "queued_run_count": max(queued_count, decision_queued_count),
        "starting_run_count": starting_count,
        "running_run_count": running_count,
        "live_order_path": "VaultCycle -> StrategyRun -> Worker -> RiskEngine -> OrderManager",
    }


def _cycle_trade_decision(
    cycle: VaultCycle,
    order_summaries: list[dict[str, object]],
    leg_decisions: list[dict[str, object]],
    runtime_notice: dict[str, object] | None = None,
) -> dict[str, object]:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return {
            "stage": "not_applicable",
            "label": "Standard cycle",
            "mode": cycle.execution_mode,
            "status": str(cycle.status or ""),
            "message": "Trade decision diagnostics are available for 1H10 cycles.",
        }
    runtime_notice = runtime_notice or {}
    stages = [str(row.get("stage") or "") for row in leg_decisions]
    submitted = [row for row in order_summaries if str(row.get("status") or "").lower() in {"submitted", "open", "filled"}]
    failed = [row for row in order_summaries if str(row.get("status") or "").lower() == "failed"]
    rejected = [row for row in order_summaries if str(row.get("status") or "").lower() == "rejected"]
    if submitted:
        stage = "placed"
    elif failed:
        stage = "failed"
    elif rejected or "blocked" in stages:
        stage = "blocked"
    elif "signal_generated" in stages:
        stage = "signal_generated"
    elif "skipped" in stages:
        stage = "skipped"
    elif "queued_for_worker" in stages:
        stage = "queued_for_worker"
    elif "waiting_for_signal" in stages:
        stage = "waiting_for_signal"
    elif runtime_notice:
        stage = "blocked"
    else:
        stage = "pending"

    reason = ""
    for row in [*leg_decisions, *order_summaries]:
        reason = str(row.get("reason") or row.get("risk_rejection_reason") or row.get("last_signal_reason") or "").strip()
        if reason:
            break
    if not reason and runtime_notice:
        reason = str(runtime_notice.get("message") or runtime_notice.get("blocker_category") or "")
    reason = _sanitize_cycle_reason(reason)
    labels = {
        "placed": "Order placed",
        "failed": "Order failed",
        "blocked": "Blocked by gate",
        "signal_generated": "Signal generated",
        "skipped": "Signal skipped",
        "queued_for_worker": "Queued for worker",
        "waiting_for_signal": "Waiting for signal",
        "pending": "Collecting data",
    }
    messages = {
        "placed": "At least one 1H10 order reached the broker submission path.",
        "failed": "A 1H10 order failed and live trading remains blocked until review.",
        "blocked": "A server-side risk, readiness, forecast, or market-data gate blocked order placement.",
        "signal_generated": "A directional signal exists and is awaiting server-side order/risk handling.",
        "skipped": "The latest 1H10 signal resolved to no trade.",
        "queued_for_worker": "The run is queued for the dedicated strategy worker; no signal can place an order until the worker starts it.",
        "waiting_for_signal": "The strategy worker is active and waiting for an actionable 1H10 signal.",
        "pending": "The cycle has not reached an order decision yet.",
    }
    return {
        "stage": stage,
        "label": labels.get(stage, "Trade decision"),
        "mode": cycle.execution_mode,
        "status": "success" if stage == "placed" else "error" if stage == "failed" else "blocked" if stage == "blocked" else "pending",
        "message": messages.get(stage, ""),
        "reason": reason,
        "order_count": len(order_summaries),
        "submitted_order_count": len(submitted),
        "rejected_order_count": len(rejected),
        "failed_order_count": len(failed),
        "leg_count": len(leg_decisions),
        "queued_run_count": sum(1 for row in leg_decisions if row.get("stage") == "queued_for_worker"),
        "signal_count": sum(1 for row in leg_decisions if row.get("has_signal")),
        "broker_order_submitted": bool(submitted),
        "runtime_notice": runtime_notice,
    }


def _cycle_trade_decision_legs(
    cycle: VaultCycle,
    order_summaries: list[dict[str, object]],
    legs: list[dict[str, object]],
) -> list[dict[str, object]]:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return []
    rows: list[dict[str, object]] = []
    for leg in legs:
        orders = _orders_for_trade_decision_leg(leg, order_summaries)
        order_statuses = [str(order.get("status") or "").lower() for order in orders]
        signal = leg.get("last_signal") if isinstance(leg.get("last_signal"), dict) else {}
        signal_action = str(leg.get("last_signal_action") or signal.get("action") or "").lower()
        signal_metadata = signal.get("metadata") if isinstance(signal.get("metadata"), dict) else {}
        explicit_stage = str(signal_metadata.get("trade_decision_stage") or "").strip()
        run_status = str(leg.get("run_status") or leg.get("strategy_run_status") or "").lower()
        if not run_status and leg.get("strategy_run_id"):
            run = db.session.get(StrategyRun, int(leg["strategy_run_id"]))
            run_status = str(run.status or "").lower() if run else ""
        runtime_backoff = leg.get("runtime_backoff") if isinstance(leg.get("runtime_backoff"), dict) else {}
        forecast_blockers = [str(item) for item in (leg.get("forecast_blockers") or []) if str(item)]
        if any(status in {"submitted", "open", "filled"} for status in order_statuses):
            stage = "placed"
            reason = ""
        elif "failed" in order_statuses:
            stage = "failed"
            reason = _first_order_reason(orders)
        elif "rejected" in order_statuses:
            stage = "blocked"
            reason = _first_order_reason(orders)
        elif explicit_stage in {"queued_for_worker", "waiting_for_signal", "signal_generated", "skipped", "blocked", "failed"}:
            stage = explicit_stage
            reason = str(signal_metadata.get("no_trade_reason") or leg.get("last_signal_reason") or "")
        elif runtime_backoff:
            stage = "blocked"
            reason = runtime_backoff.get("message") or runtime_backoff.get("blocker_category") or "market_data_backoff"
        elif run_status == "queued":
            stage = "queued_for_worker"
            reason = "strategy_worker_pending"
        elif run_status == "error":
            stage = "failed"
            reason = str(leg.get("last_signal_reason") or "strategy_run_error")
        elif signal_action in {"buy", "sell", "reduce"}:
            stage = "signal_generated"
            reason = str(signal.get("rationale") or "")
        elif signal_action == "hold" and (signal_metadata.get("no_trade_reason") or leg.get("last_signal_reason")):
            stage = "skipped"
            reason = str(signal_metadata.get("no_trade_reason") or leg.get("last_signal_reason") or "")
        elif forecast_blockers:
            stage = "blocked"
            reason = ",".join(forecast_blockers)
        elif run_status in {"running", "starting"}:
            stage = "waiting_for_signal"
            reason = ""
        else:
            stage = "pending"
            reason = ""
        rows.append(
            {
                "stage": stage,
                "status": "success"
                if stage == "placed"
                else "error"
                if stage == "failed"
                else "blocked"
                if stage == "blocked"
                else "pending",
                "reason": _sanitize_cycle_reason(reason),
                "symbol": leg.get("symbol"),
                "provider": leg.get("provider"),
                "strategy_run_id": leg.get("strategy_run_id"),
                "strategy_name": leg.get("strategy_name"),
                "timeframe": leg.get("timeframe"),
                "run_status": run_status,
                "has_signal": bool(signal_action),
                "signal_action": signal_action or "pending",
                "forecast_blockers": forecast_blockers,
                "order_count": len(orders),
                "order_statuses": list(dict.fromkeys(order_statuses)),
            }
        )
    return rows


def _orders_for_trade_decision_leg(leg: dict[str, object], orders: list[dict[str, object]]) -> list[dict[str, object]]:
    leg_id = leg.get("id")
    strategy_run_id = leg.get("strategy_run_id")
    symbol = str(leg.get("symbol") or "").upper()
    provider = str(leg.get("provider") or "").lower()
    matched = []
    for order in orders:
        if leg_id and order.get("vault_leg_id") == leg_id:
            matched.append(order)
            continue
        if strategy_run_id and order.get("strategy_run_id") == strategy_run_id:
            matched.append(order)
            continue
        order_symbol = str(order.get("symbol") or "").upper()
        order_provider = str(order.get("provider") or "").lower()
        if symbol and order_symbol == symbol and (not provider or not order_provider or provider == order_provider):
            matched.append(order)
    return matched


def _first_order_reason(orders: list[dict[str, object]]) -> str:
    for order in orders:
        reason = str(order.get("risk_rejection_reason") or order.get("exchange_error") or "").strip()
        if reason:
            return reason
    return ""


def _cycle_one_h10_runtime_notice(cycle: VaultCycle) -> dict[str, object]:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return {}
    notice = cycle.selection_metadata.get("one_h10_runtime_notice")
    if isinstance(notice, dict) and notice:
        result = dict(notice)
        if result.get("message"):
            result["message"] = _sanitize_cycle_reason(result.get("message"))
        if _runtime_notice_expired(result):
            return {}
        return result
    blocker = str(cycle.selection_metadata.get("one_h10_market_data_blocker") or "")
    backoff_until = str(cycle.selection_metadata.get("one_h10_market_data_backoff_until") or "")
    error = cycle.selection_metadata.get("one_h10_market_data_error")
    if not blocker and not error:
        return {}
    return (
        {
            "kind": "market_data_backoff",
            "message": _sanitize_cycle_reason(error or blocker),
            "blocker_category": blocker or _blocker_category(error),
            "retry_after": backoff_until,
        }
        if not _runtime_notice_expired({"retry_after": backoff_until})
        else {}
    )


def _runtime_notice_expired(notice: dict[str, object]) -> bool:
    raw = str(notice.get("retry_after") or notice.get("backoff_until") or "")
    if not raw:
        return False
    try:
        until = datetime.fromisoformat(raw.replace("Z", ""))
    except ValueError:
        return False
    return until <= datetime.utcnow()


def _sanitize_cycle_reason(reason: object) -> str:
    text = str(reason or "").strip()
    lower = text.lower()
    if not text:
        return ""
    if "429" in lower or "rate limit" in lower or "too many requests" in lower:
        return "Provider rate limited market data or account data; retrying after backoff."
    if "provider_market_data_unavailable" in lower:
        return "Provider-specific market data is unavailable for this symbol; waiting for a safe data source."
    if "400302" in lower or "currently unavailable in the u.s" in lower or "current ip:" in lower:
        return "KuCoin is unavailable from this runtime region; use a compliant non-restricted fixed-egress live API runtime."
    if "invalid request ip" in lower:
        return text
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _cycle_feature_diagnostics(cycle: VaultCycle) -> dict[str, object]:
    discovery = [row for row in cycle.selection_metadata.get("market_discovery", []) or [] if isinstance(row, dict)]
    return {
        "provider_count": len(discovery),
        "active_markets": sum(int(row.get("active", 0) or 0) for row in discovery),
        "disabled_markets": sum(int(row.get("disabled", 0) or 0) for row in discovery),
        "features_attempted": sum(int(row.get("features_attempted", 0) or 0) for row in discovery),
        "features_skipped": sum(int(row.get("features_skipped", 0) or 0) for row in discovery),
        "feature_skip_reasons": list(
            dict.fromkeys(str(reason) for row in discovery for reason in (row.get("feature_skip_reasons", []) or []) if str(reason))
        ),
    }


def _cycle_forecast_blockers(cycle: VaultCycle) -> list[str]:
    blockers: list[str] = []
    for leg in cycle.allocation_legs:
        details = leg.details
        blockers.extend(_hard_forecast_blockers(details.get("forecast_blockers", []) or [], cycle))
    for provider in (
        cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []
    ):
        if not isinstance(provider, dict):
            continue
        for leg in provider.get("legs", []) or []:
            if not isinstance(leg, dict):
                continue
            forecast = leg.get("forecast") if isinstance(leg.get("forecast"), dict) else {}
            blockers.extend(_hard_forecast_blockers(forecast.get("blockers", []) or [], cycle))
    return list(dict.fromkeys(blockers))


def _cycle_forecast_advisory_blockers(cycle: VaultCycle) -> list[str]:
    blockers: list[str] = []
    for leg in cycle.allocation_legs:
        details = leg.details
        blockers.extend(str(item) for item in (details.get("forecast_advisory_blockers", []) or []) if str(item))
        blockers.extend(_advisory_forecast_blockers(details.get("forecast_blockers", []) or [], cycle))
    for provider in (
        cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []
    ):
        if not isinstance(provider, dict):
            continue
        for leg in provider.get("legs", []) or []:
            if not isinstance(leg, dict):
                continue
            forecast = leg.get("forecast") if isinstance(leg.get("forecast"), dict) else {}
            blockers.extend(str(item) for item in (forecast.get("advisory_blockers", []) or []) if str(item))
            blockers.extend(_advisory_forecast_blockers(forecast.get("blockers", []) or [], cycle))
    return list(dict.fromkeys(blockers))


def _hard_forecast_blockers(blockers: list[object], cycle: VaultCycle) -> list[str]:
    advisory = _one_h10_advisory_blocker_set(cycle)
    return [str(item) for item in blockers if str(item) and str(item) not in advisory]


def _advisory_forecast_blockers(blockers: list[object], cycle: VaultCycle) -> list[str]:
    advisory = _one_h10_advisory_blocker_set(cycle)
    return [str(item) for item in blockers if str(item) and str(item) in advisory]


def _one_h10_advisory_blocker_set(cycle: VaultCycle) -> set[str]:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return set()
    configured = current_app.config.get("ONE_H10_ML_ADVISORY_BLOCKERS", [])
    if isinstance(configured, str):
        return {item.strip() for item in configured.split(",") if item.strip()}
    return {str(item).strip() for item in (configured or []) if str(item).strip()}


def _cycle_skipped_symbols(cycle: VaultCycle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for provider in (
        cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []
    ):
        if not isinstance(provider, dict):
            continue
        diagnostics = provider.get("scanner_diagnostics", {}) if isinstance(provider.get("scanner_diagnostics"), dict) else {}
        for skipped in diagnostics.get("rejected", diagnostics.get("skipped_symbols", [])) or []:
            if isinstance(skipped, dict):
                rows.append({"provider": provider.get("provider"), **skipped})
            else:
                rows.append({"provider": provider.get("provider"), "symbol": str(skipped), "reason": "scanner_rejected"})
    return rows


def _cycle_effective_ml_readiness(cycle: VaultCycle) -> dict[str, object]:
    readiness = cycle.selection_metadata.get("ml_readiness", {})
    if not isinstance(readiness, dict):
        return {}
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return readiness
    if bool(readiness.get("ready", False)):
        return readiness
    if readiness.get("source") != "one_h10_live_execution_readiness":
        return readiness
    current_readiness = _one_h10_ml_readiness("global")
    return current_readiness if bool(current_readiness.get("ready", False)) else readiness


def _cycle_blocker_categories(cycle: VaultCycle, orders: list[Order]) -> list[str]:
    categories: list[str] = list(cycle.selection_metadata.get("blocker_categories", []) or [])
    categories.extend(_blocker_categories_from_reasons(cycle.selection_metadata.get("provider_skip_reasons", []) or []))
    ml_readiness = _cycle_effective_ml_readiness(cycle)
    if ml_readiness and not bool(ml_readiness.get("ready", False)):
        categories.append("ml_not_ready")
    for blocker in cycle.selection_metadata.get("ml_blockers", []) or []:
        categories.append(_blocker_category(blocker))
    for blocker in cycle.selection_metadata.get("risk_blockers", []) or []:
        categories.append(_blocker_category(blocker))
    for blocker in _cycle_forecast_blockers(cycle):
        categories.append(_blocker_category(blocker))
    for order in orders:
        details = dict(order.details or {})
        if details.get("blocker_category"):
            categories.append(str(details["blocker_category"]))
        if order.rejection_reason:
            categories.append(_blocker_category(order.rejection_reason))
        if details.get("risk_rejection_reason"):
            categories.append(_blocker_category(details.get("risk_rejection_reason")))
        if details.get("exchange_error"):
            categories.append(_blocker_category(details.get("exchange_error")))
    if not orders and str(cycle.algorithm_profile or "").upper() == "1H10" and not categories:
        categories.append("ml_hold")
    categories = list(dict.fromkeys(item for item in categories if item))
    if _one_h10_ml_hold_is_advisory(cycle):
        categories = [item for item in categories if item != "ml_hold"]
    return categories


def _one_h10_ml_hold_is_advisory(cycle: VaultCycle) -> bool:
    if str(cycle.algorithm_profile or "").upper() != "1H10":
        return False
    if bool(current_app.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)):
        return False
    advisory = set(_cycle_forecast_advisory_blockers(cycle))
    ml_hold_advisory = {"forecast_hold", "low_confidence", "ml_fibonacci_confidence_below_minimum"}
    return bool(advisory.intersection(ml_hold_advisory))


def _blocker_categories_from_reasons(rows: list[dict[str, object]] | list[object]) -> list[str]:
    categories: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            categories.append(_blocker_category(row.get("reason") or row.get("failure_reason") or row.get("blocker") or ""))
        else:
            categories.append(_blocker_category(row))
    return list(dict.fromkeys(item for item in categories if item))


def _blocker_category(reason: object) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return ""
    checks = [
        ("rate_limited", ("429", "rate limit", "too many request")),
        (
            "provider_market_data_unavailable",
            ("provider_market_data_unavailable", "provider-specific market data", "market data unavailable"),
        ),
        ("provider_unhealthy", ("provider_unhealthy", "blocked live trading", "cannot trade", "unhealthy", "action needed")),
        ("insufficient_margin", ("insufficient_free_margin", "insufficient margin", "insufficient balance", "wallet_balance_insufficient")),
        ("no_active_markets", ("no_active_markets", "no ranked", "no_ranked_markets", "market unavailable")),
        ("features_stale", ("feature_backoff", "features_stale", "stale", "snapshot unavailable")),
        ("ml_not_ready", ("ml_not_ready", "not_ready", "promoted_", "ml_all_areas_enabled=false", "torch_missing")),
        ("ml_hold", ("ml_signal_hold", "selected hold", "zero_sizing", "low_confidence")),
        ("risk_rejected", ("risk_rejected", "risk", "safety_envelope", "policy_rejected")),
        ("leverage_cap", ("leverage", "max_leverage", "leverage_cap")),
        ("dynamic_cap_breach", ("dynamic_cap", "notional", "hard_cap")),
        ("liquidity_too_low", ("liquidity", "minimum liquidity")),
        ("slippage_too_high", ("slippage", "spread")),
        ("min_notional", ("min_notional", "minimum order", "min size")),
        ("missing_stop_take_profit", ("stop loss", "take profit", "missing_exit")),
        ("exchange_rejected", ("exchange rejected", "rejected")),
        ("connector_error", ("connector", "api", "timeout", "network", "failed")),
        ("worker_pending", ("strategy_worker_pending", "queued for dedicated worker", "worker startup")),
    ]
    for category, markers in checks:
        if any(marker in text for marker in markers):
            return category
    return "connector_error" if text else ""


def _order_summary(order: Order) -> dict[str, object]:
    fills = list(order.fills)
    fill_quantity = sum(float(fill.quantity or 0.0) for fill in fills)
    weighted_fill = sum(float(fill.quantity or 0.0) * float(fill.price or 0.0) for fill in fills)
    average_fill = float(order.average_fill_price or 0.0)
    if average_fill <= 0 and fill_quantity > 0:
        average_fill = weighted_fill / fill_quantity
    fees = sum(float(fill.fee or 0.0) + float(getattr(fill, "funding_fee", 0.0) or 0.0) for fill in fills)
    realized = sum(float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0) for fill in fills)
    risk_reward = _risk_reward_ratio(order, average_fill)
    return {
        "id": order.id,
        "client_order_id": order.client_order_id,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "symbol": order.symbol,
        "side": order.side,
        "status": order.status,
        "order_type": order.order_type,
        "provider": order.details.get("provider") or order.details.get("execution_venue"),
        "trading_connection_id": order.trading_connection_id,
        "vault_cycle_id": _order_vault_cycle_id(order),
        "vault_leg_id": _order_vault_leg_id(order),
        "settlement_asset": order.details.get("settlement_asset"),
        "quantity": float(order.quantity or 0.0),
        "filled_quantity": float(order.filled_quantity or 0.0),
        "average_fill": average_fill,
        "fees": fees,
        "realized_pnl": realized,
        "leverage": float(order.leverage or 1.0),
        "stop_loss": float(order.stop_loss or 0.0),
        "take_profit": float(order.take_profit or 0.0),
        "risk_reward": risk_reward,
        "risk_rejection_reason": order.details.get("risk_rejection_reason") or order.rejection_reason,
        "blocker_category": order.details.get("blocker_category")
        or _blocker_category(order.details.get("risk_rejection_reason") or order.rejection_reason),
        "ml_policy_authority": (
            order.details.get("risk_decision", {}).get("details", {}) if isinstance(order.details.get("risk_decision"), dict) else {}
        ).get("ml_policy_authority"),
        "ml_policy_decisions": order.details.get("ml_policy_decisions", {}),
        "exchange_error": order.details.get("exchange_error"),
        "exchange_latency_ms": float(order.details.get("exchange_latency_ms", 0.0) or 0.0),
        "risk_latency_ms": float(order.details.get("risk_latency_ms", 0.0) or 0.0),
        "slippage_bps": float(order.details.get("slippage_bps", 0.0) or 0.0),
        "edge_score": float(order.details.get("edge_score", 0.0) or 0.0),
        "cost_drag_bps": float(order.details.get("cost_drag_bps", 0.0) or 0.0),
        "signal_confidence": float(order.details.get("signal_confidence", 0.0) or 0.0),
        "quality_reasons": order.details.get("quality_reasons", []),
        "fibonacci_alignment": order.details.get("fibonacci_alignment", {}),
        "feature_confluence": order.details.get("feature_confluence", {}),
        "ml_signal_quality": order.details.get("ml_signal_quality", {}),
        "one_h10_forecast": order.details.get("one_h10_forecast", order.details.get("forecast_metadata", {})),
        "forecast_blockers": order.details.get("forecast_blockers", []),
        "forecast_predicted_side": order.details.get("forecast_predicted_side"),
        "forecast_confidence": order.details.get("forecast_confidence"),
        "forecast_expected_return_bps": order.details.get("forecast_expected_return_bps"),
        "market_source": order.details.get("market_source"),
        "signal_stability": order.details.get("signal_stability"),
        "execution_mode": order.mode,
        "fills": [
            {
                "id": fill.id,
                "side": fill.side,
                "quantity": float(fill.quantity or 0.0),
                "price": float(fill.price or 0.0),
                "fee": float(fill.fee or 0.0),
                "pnl": float(fill.pnl or 0.0),
                "funding_fee": float(getattr(fill, "funding_fee", 0.0) or 0.0),
                "fee_known": bool(getattr(fill, "fee_known", True)),
                "realized_pnl_known": bool(getattr(fill, "realized_pnl_known", True)),
                "reconciliation": getattr(fill, "details", {}),
                "simulated": bool(fill.simulated),
                "fill_time": fill.fill_time.isoformat() if fill.fill_time else None,
            }
            for fill in fills
        ],
    }


def _leg_summary(leg: VaultAllocationLeg) -> dict[str, object]:
    parameters = leg.strategy_run.parameters if leg.strategy_run else {}
    forecast = leg.details.get("one_h10_forecast") if isinstance(leg.details.get("one_h10_forecast"), dict) else {}
    provider = str(leg.details.get("provider") or leg.provider or "").strip()
    runtime_backoff = _one_h10_provider_market_data_backoff(provider, leg.trading_connection_id) if provider else None
    last_signal = leg.strategy_run.last_signal if leg.strategy_run and isinstance(leg.strategy_run.last_signal, dict) else {}
    signal_metadata = last_signal.get("metadata") if isinstance(last_signal.get("metadata"), dict) else {}
    run_status = str(leg.strategy_run.status or "").lower() if leg.strategy_run else ""
    stop = float(
        forecast.get("suggested_stop_loss_pct") or parameters.get("stop_loss_pct", parameters.get("fallback_stop_loss_pct", 0.0)) or 0.0
    )
    take = float(
        forecast.get("suggested_take_profit_pct")
        or parameters.get("take_profit_pct", parameters.get("fallback_take_profit_pct", 0.0))
        or 0.0
    )
    leverage = float(forecast.get("suggested_leverage") or leg.leverage or 1.0)
    return {
        "id": leg.id,
        "strategy_run_id": leg.strategy_run_id,
        "strategy_name": leg.strategy_run.strategy_name if leg.strategy_run else None,
        "symbol": leg.symbol,
        "timeframe": leg.timeframe,
        "provider": provider or leg.provider,
        "trading_connection_id": leg.trading_connection_id,
        "collateral_asset": leg.details.get("collateral_asset"),
        "settlement_asset": leg.details.get("settlement_asset"),
        "allocation_weight": float(leg.details.get("allocation_weight", 0.0) or 0.0),
        "available_margin_usd": float(leg.details.get("available_margin_usd", 0.0) or 0.0),
        "allocation_cap_usd": float(leg.allocation_cap_usd or 0.0),
        "leverage": leverage,
        "status": leg.status,
        "run_status": run_status,
        "realized_pnl_usd": float(leg.realized_pnl_usd or 0.0),
        "unrealized_pnl_usd": float(leg.unrealized_pnl_usd or 0.0),
        "stop_loss_pct": stop,
        "take_profit_pct": take,
        "reward_risk": (take / stop) if stop > 0 else 0.0,
        "execution_style": leg.details.get("execution_style"),
        "edge_score": leg.details.get("edge_score"),
        "optimizer_profile": leg.details.get("optimizer_profile"),
        "allocation_mode": leg.details.get("allocation_mode"),
        "ensemble_id": leg.details.get("ensemble_id"),
        "ensemble_version": leg.details.get("ensemble_version"),
        "ensemble_adapter": leg.details.get("ensemble_adapter"),
        "ensemble_weight": leg.details.get("ensemble_weight"),
        "target_ensemble_weight": leg.details.get("target_ensemble_weight"),
        "effective_allocation_weight": leg.details.get("effective_allocation_weight"),
        "cap_limited": leg.details.get("cap_limited"),
        "cap_limit_reason": leg.details.get("cap_limit_reason"),
        "duration_bucket": leg.details.get("duration_bucket"),
        "ml_rank_score": leg.details.get("ml_rank_score"),
        "confluence_score": leg.details.get("confluence_score"),
        "multi_timeframe_confluence": leg.details.get("multi_timeframe_confluence", {}),
        "pair_group_id": leg.details.get("pair_group_id"),
        "pair_mode": leg.details.get("pair_mode"),
        "pair_symbol": leg.details.get("pair_symbol"),
        "pair_role": leg.details.get("pair_role"),
        "hedge_ratio": leg.details.get("hedge_ratio"),
        "spread_zscore": leg.details.get("spread_zscore"),
        "spread_half_life": leg.details.get("spread_half_life"),
        "pair_score": leg.details.get("pair_score"),
        "correlation": leg.details.get("correlation"),
        "pair_skip_reason": leg.details.get("pair_skip_reason"),
        "market_id": leg.details.get("market_id"),
        "venue_symbol": leg.details.get("venue_symbol"),
        "app_symbol": leg.details.get("app_symbol"),
        "market_status": leg.details.get("market_status"),
        "ml_horizon": leg.details.get("ml_horizon"),
        "one_h10_vault": bool(leg.details.get("one_h10_vault", False)),
        "target_roi_pct": leg.details.get("target_roi_pct"),
        "target_amount_usd": leg.details.get("target_amount_usd"),
        "scanner_score": leg.details.get("one_h10_scanner_score", leg.details.get("scanner_score")),
        "scanner_source": leg.details.get("one_h10_scanner_source", leg.details.get("scanner_source")),
        "scanner_score_breakdown": leg.details.get("scanner_score_breakdown", {}),
        "scanner_features": leg.details.get("scanner_features", {}),
        "one_h10_forecast": leg.details.get("one_h10_forecast", leg.details.get("forecast_metadata", {})),
        "forecast_blockers": leg.details.get("forecast_blockers", []),
        "forecast_predicted_side": leg.details.get("forecast_predicted_side"),
        "forecast_confidence": leg.details.get("forecast_confidence"),
        "forecast_expected_return_bps": leg.details.get("forecast_expected_return_bps"),
        "forecast_suggested_notional_usd": leg.details.get("forecast_suggested_notional_usd"),
        "forecast_suggested_leverage": leg.details.get("forecast_suggested_leverage"),
        "forecast_suggested_order_type": leg.details.get("forecast_suggested_order_type"),
        "last_signal": last_signal,
        "last_signal_action": last_signal.get("action"),
        "last_signal_reason": signal_metadata.get("no_trade_reason") or last_signal.get("rationale"),
        "runtime_backoff": runtime_backoff or {},
        "eligible_to_trade_now": (
            str(leg.status or "").lower() == "active"
            and (not leg.strategy_run or run_status in {"running", "starting"})
            and not runtime_backoff
        ),
    }


def _risk_reward_ratio(order: Order, average_fill: float) -> float:
    if average_fill <= 0 or not order.stop_loss or not order.take_profit:
        return 0.0
    risk = abs(average_fill - float(order.stop_loss))
    reward = abs(float(order.take_profit) - average_fill)
    return reward / risk if risk > 0 else 0.0
