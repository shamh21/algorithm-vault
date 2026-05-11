"""Consumer wallet and vault routes."""

from __future__ import annotations

import math
import threading
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import joinedload

from ..auth import current_user, qr_code_data_uri, require_authenticated_user, verify_totp
from ..extensions import db
from ..ml.online_ranker import ONE_H10_HORIZON, extract_features, horizon_from_context, horizon_from_duration, outcome_from_result
from ..models import AuditLog, DepositAddress, Fill, LeveragedMarket, Order, Setting, StrategyRun, TradingConnection, VaultAllocationLeg, VaultCycle, WalletAddress, WalletBalance, WalletTransaction
from ..runtime import get_current_mode, get_service, market_mode_for
from ..services.connection_health import build_connection_health, latest_connection_health, operator_connection_message, store_connection_health
from ..services.db_retry import commit_with_retry, is_database_locked
from ..services.market_scanner import ScoredCandidate
from ..services.provider_assets import normalize_provider, provider_collateral_asset, provider_feature_context
from ..services.wallet_addresses import generate_deposit_address, use_real_addresses, validate_withdraw_address
from ..utils import format_duration_seconds

consumer_bp = Blueprint("consumer", __name__)

SUPPORTED_WALLET_ASSETS = ("USDC", "USDT", "BTC", "ETH", "SOL", "XRP")
SETTLEMENT_ASSETS = ("ETH", "BTC", "USDT", "USDC")
ASSET_NETWORKS = {
    "BTC": ("Bitcoin",),
    "ETH": ("Ethereum",),
    "SOL": ("Solana",),
    "XRP": ("XRP Ledger",),
    "USDC": ("Ethereum",),
    "USDT": ("Ethereum",),
}
_CYCLE_START_JOBS: dict[str, dict[str, object]] = {}
_CYCLE_START_IDEMPOTENCY: dict[tuple[int, str], str] = {}
_CYCLE_START_JOB_LOCK = threading.Lock()
_CYCLE_START_JOB_KEY_PREFIX = "vault_start_job"
_CYCLE_START_IDEMPOTENCY_KEY_PREFIX = "vault_start_idem"


@consumer_bp.before_request
def _protect_consumer():
    if request.endpoint and request.endpoint.startswith("consumer.legacy_"):
        return None
    guard = require_authenticated_user()
    if guard is not None:
        return guard
    user = current_user()
    if user is not None and _live_connection_required() and get_service("trading_connections").active_tradable_connection(user.id) is None:
        flash("Connect, verify, and activate a live-ready trading account before using wallet and vault features.", "warning")
        return redirect(url_for("settings.connections"))
    return None


@consumer_bp.get("/")
def home():
    user = current_user()
    _sync_completed_cycles(user)
    balances = _wallet_balances(user)
    wallet_summary = get_service("wallet_summary").summary_for_user(user, balances=balances)
    exchange_snapshot = _exchange_balance_snapshot(user)
    active_cycles = _active_cycles(user, refresh=False)
    active_cycle = active_cycles[0] if active_cycles else None
    recent_transactions = WalletTransaction.query.filter_by(user_id=user.id).order_by(WalletTransaction.created_at.desc()).limit(5).all()
    return render_template(
        "home.html",
        balances=balances,
        wallet_summary=wallet_summary,
        exchange_snapshot=exchange_snapshot,
        portfolio_total=wallet_summary.portfolio_total_usd,
        active_cycle=active_cycle,
        recent_transactions=recent_transactions,
    )


@consumer_bp.get("/wallet/", strict_slashes=False)
def wallet():
    user = current_user()
    _sync_completed_cycles(user)
    balances = _wallet_balances(user)
    wallet_summary = get_service("wallet_summary").summary_for_user(user, balances=balances)
    exchange_snapshot = _exchange_balance_snapshot(user, refresh=_refresh_exchange_requested())
    transactions = WalletTransaction.query.filter_by(user_id=user.id).order_by(WalletTransaction.created_at.desc()).limit(20).all()
    return render_template(
        "wallet.html",
        balances=wallet_summary.balances,
        wallet_summary=wallet_summary,
        exchange_snapshot=exchange_snapshot,
        portfolio_total=wallet_summary.portfolio_total_usd,
        transactions=transactions,
        settlement_assets=_wallet_assets(),
        networks=ASSET_NETWORKS,
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
        if not bool(current_app.config.get("WALLET_WITHDRAWALS_ENABLED", False)):
            flash("Withdrawals are disabled until explicitly enabled by configuration.", "danger")
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

        if network not in _asset_networks(asset):
            errors["network"] = "Select a supported network."
        if not validate_withdraw_address(withdraw_address, asset, network):
            errors["withdraw_address"] = "Enter a valid destination address for the selected asset and network."
        if amount <= 0:
            errors["amount"] = "Enter a withdrawal amount greater than zero."
        elif amount > max(float(balance.available_balance or 0.0), max_withdrawal_amount) + 1e-9:
            errors["amount"] = "Withdrawal amount exceeds available balance."
        if not verify_totp(user, code):
            errors["totp_code"] = "Invalid authenticator code. Try again."
        real_wallet_mode = use_real_addresses(current_app.config)
        if real_wallet_mode and get_current_mode() != "live":
            errors["form"] = "Real wallet withdrawals can only be broadcast in live mode."
        max_by_asset = current_app.config.get("WALLET_MAX_WITHDRAWAL_BY_ASSET") or {}
        if isinstance(max_by_asset, dict):
            max_amount = float(max_by_asset.get(asset.lower(), max_by_asset.get(asset, 0.0)) or 0.0)
            if max_amount > 0 and amount > max_amount:
                errors["amount"] = f"Withdrawal amount exceeds configured {asset} cap."

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
                return redirect(url_for("consumer.activity"))
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
                message = "Withdrawal broadcast. Waiting for confirmation." if withdrawal.status == "submitted" else f"Withdrawal status: {withdrawal.status.replace('_', ' ')}."
                flash(message, "success")
                return redirect(url_for("consumer.activity"))
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
    _sync_completed_cycles(user)
    balances = _wallet_balances(user)
    active_cycles = _active_cycles(user)
    recovered_run_ids = _recover_active_one_h10_cycles(active_cycles)
    for cycle in active_cycles:
        cycle.cycle_summary = _cycle_summary(cycle)
    commit_with_retry()
    _start_strategy_runs(recovered_run_ids)
    recent_cycles = VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.started_at.desc()).limit(8).all()
    return render_template(
        "vault.html",
        balances=balances,
        active_cycle=active_cycles[0] if active_cycles else None,
        active_cycles=active_cycles,
        recent_cycles=recent_cycles,
        one_h10_live_context=_one_h10_live_context(user),
        settlement_assets=_wallet_assets(),
    )


@consumer_bp.post("/consumer/start")
@consumer_bp.post("/vault/start")
def start_cycle():
    user = current_user()
    async_enabled = bool(current_app.config.get("VAULT_START_ASYNC_ENABLED", False))
    idempotency_key = _request_idempotency_key()
    if async_enabled and idempotency_key:
        existing_job = _existing_cycle_start_job(user.id, idempotency_key)
        if existing_job is not None:
            if _wants_json_response():
                return jsonify(existing_job), 202
            flash("Cycle start already queued. Refresh cycle status in a moment.", "info")
            return redirect(url_for("consumer.vault"))

    _sync_completed_cycles(user)

    asset = request.form.get("deposit_asset", "USDC").upper().strip()
    settlement_asset = request.form.get("settlement_asset", "USDC").upper().strip()
    wallet_assets = _wallet_assets()
    if asset not in wallet_assets or settlement_asset not in wallet_assets:
        flash("Select a supported wallet and settlement asset.", "danger")
        return redirect(url_for("consumer.vault"))

    try:
        amount = float(request.form.get("deposit_amount", "0") or 0)
    except ValueError:
        amount = 0.0
    if amount <= 0:
        flash("Enter an allocation amount greater than zero.", "danger")
        return redirect(url_for("consumer.vault"))

    duration_seconds = _requested_duration_seconds()
    if duration_seconds <= 0:
        flash("Select a valid lock duration.", "danger")
        return redirect(url_for("consumer.vault"))
    duration_hours = max(1, math.ceil(duration_seconds / 3600))
    is_one_h10 = _is_one_h10_duration(duration_seconds, duration_hours)

    connections = _cycle_trading_connections(user, is_one_h10)
    connection = connections[0] if connections else None
    if _live_connection_required() and connection is None:
        flash("Connect and verify at least one trading account before starting 1H10." if is_one_h10 else "Connect your trading account before starting a live vault cycle.", "warning")
        return redirect(url_for("settings.connections"))
    if is_one_h10:
        one_h10_block = _one_h10_live_start_block_reason()
        if one_h10_block:
            flash(one_h10_block, "warning")
            return redirect(url_for("consumer.vault"))
        if _live_connection_required() and not _one_h10_live_acknowledged():
            flash("Confirm the 1H10 live leveraged-order acknowledgement before starting.", "warning")
            return redirect(url_for("consumer.vault"))

    balances = _wallet_balances(user)
    balance = next((item for item in balances if item.asset == asset), None)
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
        live_block_reason = str((connection_blockers[0] if connection_blockers else {}).get("reason") or "No verified 1H10 trading connection is currently healthy enough for live execution.")
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
        selection.metadata.update(
            {
                "exchange_allocation_history": allocation_history,
                "provider_allocation_history": allocation_history,
                "provider_skip_reasons": allocation_blockers,
                "market_discovery": market_discovery,
                "ml_readiness": _one_h10_ml_readiness("global"),
                "blocker_categories": _blocker_categories_from_reasons(allocation_blockers),
                "objective": "one_h10",
                "ml_objective": "one_h10",
                "ml_policy_required": True,
                "ml_governed_risk": True,
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
        include_pair_metadata = not (
            is_one_h10 and bool((leg.get("parameters") or {}).get("one_h10_all_pairs"))
        )
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
                "multi_timeframe_confluence": leg.get("multi_timeframe_confluence") or selection.metadata.get("multi_timeframe_confluence", {}),
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
                "spread_half_life": (leg.get("spread_half_life") or selection.metadata.get("spread_half_life")) if include_pair_metadata else None,
                "pair_score": (leg.get("pair_score") or selection.metadata.get("pair_score")) if include_pair_metadata else None,
                "correlation": (leg.get("correlation") or selection.metadata.get("correlation")) if include_pair_metadata else None,
                "pair_signal": (leg.get("pair_signal") or selection.metadata.get("pair_signal", {})) if include_pair_metadata else {},
                "pair_skip_reason": (leg.get("pair_skip_reason") or selection.metadata.get("pair_skip_reason", "")) if include_pair_metadata else "",
                "skip_reason": leg.get("skip_reason", ""),
                "leverage": float(leg.get("leverage", leg_parameters.get("leverage", 1.0)) or 1.0),
                "provider": leg.get("provider", leg_parameters.get("provider", selection.metadata.get("provider"))),
                "execution_venue": leg.get("execution_venue", leg_parameters.get("execution_venue", selection.metadata.get("execution_venue"))),
                "trading_connection_id": leg.get("trading_connection_id", leg_parameters.get("trading_connection_id")),
                "collateral_asset": leg.get("collateral_asset", leg_parameters.get("collateral_asset", selection.metadata.get("collateral_asset"))),
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
        if _wants_json_response():
            return jsonify(payload), 202
        flash("Vault cycle queued. Strategy workers are starting in the background.", "success")
        return redirect(url_for("consumer.vault"))

    for run_id in run_ids:
        get_service("strategy_manager").start(run_id)
    flash("Vault cycle started. Estimated performance is not guaranteed and execution remains risk-gated.", "success")
    return redirect(url_for("consumer.vault"))


@consumer_bp.get("/consumer/start-status/<job_id>")
@consumer_bp.get("/vault/start-status/<job_id>")
def cycle_start_status(job_id: str):
    user = current_user()
    job = _load_cycle_start_job(str(job_id))
    if not job or user is None:
        return jsonify({"ok": False, "error": "job_not_found", "job_id": job_id}), 404
    if int(job.get("user_id") or 0) != int(user.id):
        return jsonify({"ok": False, "error": "job_not_found", "job_id": job_id}), 404
    return jsonify(job)


@consumer_bp.get("/activity/", strict_slashes=False)
def activity():
    user = current_user()
    _sync_completed_cycles(user)
    return render_template(
        "activity.html",
        transactions=WalletTransaction.query.filter_by(user_id=user.id).order_by(WalletTransaction.created_at.desc()).limit(50).all(),
        cycles=VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.started_at.desc()).limit(50).all(),
    )


@consumer_bp.get("/vault/cycles/<int:cycle_id>")
def cycle_detail(cycle_id: int):
    user = current_user()
    _sync_completed_cycles(user)
    cycle = VaultCycle.query.filter_by(id=cycle_id, user_id=user.id).one_or_none()
    if cycle is None:
        flash("Vault cycle was not found.", "danger")
        return redirect(url_for("consumer.activity"))
    performance = None
    if cycle.status in {"active", "settling"}:
        performance = _refresh_cycle_performance(cycle)
        recovered_run_ids = _recover_active_one_h10_cycles([cycle])
        commit_with_retry()
        _start_strategy_runs(recovered_run_ids)
    summary = _cycle_summary(cycle, performance=performance) if cycle.status in {"active", "settling"} else cycle.cycle_summary or _cycle_summary(cycle)
    return render_template(
        "cycle_detail.html",
        cycle=cycle,
        summary=summary,
        explanation=_cycle_profit_explanation(),
    )


@consumer_bp.get("/dashboard")
def legacy_dashboard():
    return redirect(url_for("dashboard.index"))


@consumer_bp.get("/api/dashboard-data")
def legacy_dashboard_data():
    return redirect(url_for("dashboard.dashboard_data"))


@consumer_bp.post("/strategies/start")
def legacy_start_strategy():
    return redirect(url_for("dashboard.start_strategy"), code=307)


@consumer_bp.post("/strategies/<int:run_id>/stop")
def legacy_stop_strategy(run_id: int):
    return redirect(url_for("dashboard.stop_strategy", run_id=run_id), code=307)


@consumer_bp.route("/orders/", methods=["GET"], strict_slashes=False)
def legacy_orders():
    return redirect(url_for("orders.index"))


@consumer_bp.post("/orders/place")
def legacy_order_place():
    return redirect(url_for("orders.place"), code=307)


@consumer_bp.post("/orders/<int:order_id>/cancel")
def legacy_order_cancel(order_id: int):
    return redirect(url_for("orders.cancel", order_id=order_id), code=307)


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
    return redirect(url_for("panic.index"))


@consumer_bp.post("/panic/activate")
def legacy_panic_activate():
    return redirect(url_for("panic.activate"), code=307)


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
            address = _ensure_deposit_address(user.id, balance.asset, network, balance)
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


def _sync_real_wallet_balances(user) -> None:
    custody = get_service("wallet_custody")
    if not getattr(custody, "enabled", False):
        return
    try:
        custody.sync_user(user.id)
        commit_with_retry()
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("Real wallet sync failed closed for user %s: %s", user.id, exc)


def _sync_connection_balances(user) -> None:
    if not _live_connection_required():
        return
    connection = _active_trading_connection(user)
    if connection is None:
        return
    get_service("wallet_summary").refresh_exchange_snapshot(user, get_service("trading_connections"), mode="live", connection_id=connection.id)


def _exchange_balance_snapshot(user, *, refresh: bool = False) -> dict:
    if refresh:
        _sync_connection_balances(user)
    return get_service("wallet_summary").cached_exchange_snapshot(user.id)


def _refresh_exchange_requested() -> bool:
    return str(request.args.get("refresh_exchange", "")).strip().lower() in {"1", "true", "yes", "on"}


def _exchange_balance_snapshot_key(user_id: int) -> str:
    return f"exchange_balance_snapshot:{int(user_id)}"


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
    if asset not in {"USDT", "USDC", "ETH"} or network != "Ethereum":
        return app_balance
    try:
        custody = get_service("wallet_custody")
        source = (
            WalletAddress.query.filter_by(user_id=user.id, asset=asset, network=network, status="active")
            .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc())
            .first()
        )
        if source is None:
            return app_balance
        snapshot = custody._require_adapter(asset, network).get_balance(source.address, asset, network)  # noqa: SLF001
        if snapshot.checked:
            return max(float(snapshot.amount or 0.0), 0.0)
    except Exception:
        return app_balance
    return app_balance


def _decimal_amount(value: float) -> str:
    return f"{float(value or 0.0):.6f}".rstrip("0").rstrip(".")


def _portfolio_total(balances: list[WalletBalance]) -> float:
    return sum(float(balance.estimated_usd_value or 0.0) for balance in balances)


def _live_connection_required() -> bool:
    return bool(current_app.config.get("ENABLE_LIVE_TRADING", False)) and get_current_mode() == "live"


def _active_trading_connection(user) -> TradingConnection | None:
    return get_service("trading_connections").active_tradable_connection(user.id)


def _is_one_h10_duration(duration_seconds: int, duration_hours: int) -> bool:
    return int(duration_seconds or 0) == 3600 and int(duration_hours or 0) == 1


def _one_h10_live_acknowledged() -> bool:
    value = str(request.form.get("one_h10_live_ack", "")).strip().lower()
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
        "target_copy": "1H10 targets 10x the user's input amount in 1 hour. This is an objective, not a guaranteed return.",
        "providers": providers,
        "enabled_provider_count": sum(1 for item in providers if bool(item.get("can_trade")) and float(item.get("available_margin_usd", 0.0) or 0.0) > 0),
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
        parameters.get("provider")
        or leg.get("provider")
        or getattr(selection, "metadata", {}).get("provider")
        or "global"
    )
    execution_venue = normalize_provider(
        parameters.get("execution_venue")
        or leg.get("execution_venue")
        or provider
    )
    app_symbol = str(
        parameters.get("app_symbol")
        or leg.get("app_symbol")
        or leg.get("symbol")
        or parameters.get("symbol")
        or getattr(selection, "symbol", "")
        or "BTC"
    ).strip().upper()
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


def _start_strategy_runs(run_ids: list[int]) -> None:
    if not run_ids:
        return
    manager = get_service("strategy_manager")
    for run_id in dict.fromkeys(run_ids):
        manager.start(run_id)


def _request_idempotency_key() -> str:
    header_key = str(request.headers.get("Idempotency-Key", "")).strip()
    form_key = str(request.form.get("idempotency_key", "")).strip()
    return header_key or form_key


def _wants_json_response() -> bool:
    requested = str(request.args.get("response", "")).strip().lower()
    if requested == "json":
        return True
    return request.accept_mimetypes.best == "application/json"


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
        heartbeat_stale = heartbeat_at is None or (
            datetime.utcnow() - heartbeat_at
        ).total_seconds() > max(60.0, float(current_app.config.get("ONE_H10_POLL_SECONDS", 1.0) or 1.0) * 5.0)
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
        families = {
            family: dict(engine.family_readiness(family, ONE_H10_HORIZON, provider=provider))
            for family in required_families
        }
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


def _cycle_trading_connections(user, is_one_h10: bool) -> list[TradingConnection]:
    service = get_service("trading_connections")
    if is_one_h10 and hasattr(service, "verified_tradable_connections"):
        return list(service.verified_tradable_connections(user.id))
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
            fallback_symbols = [
                str(leg.get("symbol") or "").upper()
                for leg in base_legs
                if str(leg.get("symbol") or "").strip()
            ]
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
        score_total = sum(max(float(candidate.score or 0.0), 0.0) for candidate in ranked)
        if score_total <= 0:
            score_total = float(len(ranked) or 1)
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
        for candidate in ranked:
            candidate_symbol = str(candidate.symbol or selection.symbol).upper()
            template_leg = next(
                (
                    dict(item)
                    for item in base_legs
                    if str(item.get("symbol") or "").upper() == candidate_symbol
                ),
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
            weight = max(float(candidate.score or 0.0), 0.0) / score_total if score_total > 0 else 0.0
            if weight <= 0:
                weight = 1.0 / max(len(ranked), 1)
            allocation_cap = provider_cap * weight
            if allocation_cap <= 0:
                continue
            params = dict(leg.get("parameters") or selection.parameters)
            provider_context = provider_feature_context(provider)
            forecast = {}
            if forecast_service is not None:
                forecast = forecast_service.forecast(
                    candidate_features,
                    provider=provider,
                    symbol=symbol,
                    allocation_cap_usd=allocation_cap,
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
                    "venue_symbol": venue_symbol,
                    "app_symbol": symbol,
                    "market_status": market_status,
                    "scanner_score": candidate.score,
                    "scanner_source": candidate.source,
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
                    "score_breakdown": candidate.score_breakdown or {},
                    "forecast": forecast,
                }
            )
        allocation_history.append(provider_history)
    return generated, allocation_history, blockers


def _one_h10_forecast_live_blockers(forecast: dict[str, object] | None) -> list[str]:
    if not isinstance(forecast, dict) or not forecast:
        return ["forecast_unavailable"]
    blockers = {str(item) for item in (forecast.get("blockers", []) or []) if str(item)}
    blockers.update(str(item) for item in (forecast.get("advisory_blockers", []) or []) if str(item))
    hard = {
        "cost_drag_above_threshold",
        "low_edge_after_costs",
        "low_liquidity_capacity",
        "stale_market_data",
        "features_stale",
    }
    selected = sorted(blockers.intersection(hard))
    if forecast.get("expected_net_edge_passed") is False and "low_edge_after_costs" not in selected:
        selected.append("low_edge_after_costs")
    if str(forecast.get("predicted_side") or forecast.get("action") or "hold").lower() not in {"buy", "sell"}:
        selected.append("forecast_hold")
    if _one_h10_float(forecast.get("suggested_notional_usd")) <= 0:
        selected.append("forecast_zero_sizing")
    return list(dict.fromkeys(selected))


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
    return "1H10 could not allocate capital to a healthy, funded leveraged provider."


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
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - parsed).total_seconds()
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
    now = datetime.utcnow()
    cycles = VaultCycle.query.filter_by(user_id=user.id).filter(VaultCycle.status == "active", VaultCycle.unlocks_at <= now).all()
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
            cycle.current_estimated_value_usd / settlement_price
            if settlement_price > 0
            else cycle.deposit_amount
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
    configured = tuple(get_service("wallet_address_service").configured_networks(asset))
    networks = tuple(dict.fromkeys((*ASSET_NETWORKS.get(asset, ("native",)), *configured)))
    functional = tuple(network for network in networks if _functional_wallet_network(asset, network))
    return functional or ASSET_NETWORKS.get(asset, ("native",))


def _functional_wallet_network(asset: str, network: str) -> bool:
    asset_key = asset.upper().strip()
    network_key = "".join(ch for ch in str(network or "").upper() if ch.isalnum())
    if asset_key in {"ETH", "USDC", "USDT"}:
        return network_key in {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}
    if asset_key == "BTC":
        return network_key == "BITCOIN"
    if asset_key == "SOL":
        return network_key == "SOLANA"
    if asset_key == "XRP":
        return network_key == "XRPLEDGER"
    return False


def _wallet_assets() -> tuple[str, ...]:
    configured = tuple(get_service("wallet_address_service").configured_assets())
    return tuple(dict.fromkeys((*SUPPORTED_WALLET_ASSETS, *configured)))


def _is_supported_wallet_asset(asset: str) -> bool:
    return asset in _wallet_assets()


def _active_deposit_address(user_id: int, asset: str, network: str) -> DepositAddress | None:
    return (
        DepositAddress.query.filter_by(user_id=user_id, asset=asset, network=network, is_active=True)
        .order_by(DepositAddress.version.desc())
        .first()
    )


def _ensure_deposit_address(user_id: int, asset: str, network: str, balance: WalletBalance | None = None) -> DepositAddress | None:
    address = _active_deposit_address(user_id, asset, network)
    if address is None:
        address = _new_deposit_address(user_id, asset, network)
    if address is None:
        return None
    if balance is not None and balance.active_deposit_address_id != address.id:
        balance.active_deposit_address_id = address.id
        commit_with_retry()
    return address


def _new_deposit_address(
    user_id: int,
    asset: str,
    network: str,
    rotated_from: DepositAddress | None = None,
) -> DepositAddress | None:
    latest = (
        DepositAddress.query.filter_by(user_id=user_id, asset=asset, network=network)
        .order_by(DepositAddress.version.desc())
        .first()
    )
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
    asset = asset.upper()
    if asset in {"USDC", "USDT"}:
        return 1.0
    try:
        price = float(get_service("market_data").get_mid_price(asset, market_mode_for(get_current_mode())))
    except Exception:  # noqa: BLE001
        return 0.0
    return price if price > 0 else 0.0


def _requested_duration_seconds() -> int:
    value = request.form.get("lock_duration", "24")
    if value == "custom":
        raw_value = request.form.get("custom_duration_value") or request.form.get("custom_duration_hours", "24")
        unit = request.form.get("custom_duration_unit", "hours")
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
    raw_values = request.form.getlist("allowed_symbols")
    if not raw_values:
        raw_values = [request.form.get("allowed_symbols", "")]
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
            float(cycle.current_estimated_value_usd or 0.0)
            or float(cycle.starting_value_usd or 0.0) + float(performance["total_pnl"]),
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
            float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0)
            for fill in order.fills
        )
        realized += order_pnl
        leg_id = order.details.get("vault_leg_id")
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
    return [
        order
        for order in query.order_by(Order.created_at.asc()).all()
        if order.details.get("vault_cycle_id") == cycle.id
    ]


def _cycle_summary(cycle: VaultCycle, *, performance: dict[str, float | bool] | None = None) -> dict[str, object]:
    performance = performance if performance is not None else _cycle_performance(cycle)
    orders = _cycle_orders(cycle)
    order_summaries = [_order_summary(order) for order in orders]
    fills = [fill for order in orders for fill in order.fills]
    fees = sum(float(fill.fee or 0.0) + float(getattr(fill, "funding_fee", 0.0) or 0.0) for fill in fills)
    slippage_values = [
        float(order.details.get("slippage_bps", 0.0) or 0.0)
        for order in orders
        if order.details.get("slippage_bps") is not None
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
    rejected_intents = [
        order
        for order in order_summaries
        if str(order.get("status") or "").lower() in {"rejected", "failed"}
    ]
    repairable_no_order = _cycle_repairable_no_order(cycle, orders)
    runtime_notice = _cycle_one_h10_runtime_notice(cycle)
    raw_no_order_reason = cycle.selection_metadata.get("no_order_failure_reason") or cycle.validation_failure_reason
    no_order_failure_reason = _sanitize_cycle_reason(raw_no_order_reason) if repairable_no_order else None
    summary = {
        "cycle_id": cycle.id,
        "status": cycle.status,
        "execution_substatus": cycle.execution_substatus,
        "no_order_failure_reason": no_order_failure_reason,
        "repairable_no_order": repairable_no_order,
        "runtime_notice": runtime_notice,
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
        "slippage_bps": (sum(slippage_values) / len(slippage_values)) if slippage_values else float(
            cycle.selection_metadata.get("estimated_slippage_bps", 0.0) or 0.0
        ),
        "execution_styles": sorted({str(leg.get("execution_style") or "") for leg in legs if leg.get("execution_style")}),
        "provider_allocation_history": cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])),
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
        "submitted_order_count": sum(1 for order in order_summaries if str(order.get("status") or "").lower() in {"submitted", "open", "filled"}),
        "failed_order_count": sum(1 for order in order_summaries if str(order.get("status") or "").lower() == "failed"),
        "rejected_order_count": sum(1 for order in order_summaries if str(order.get("status") or "").lower() == "rejected"),
        "orders": order_summaries,
        "legs": legs,
        "generated_at": datetime.utcnow().isoformat(),
    }
    return summary


def _cycle_ranked_candidates(cycle: VaultCycle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for provider in cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []:
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


def _cycle_repairable_no_order(cycle: VaultCycle, orders: list[Order]) -> bool:
    if orders:
        return False
    if not (cycle.selection_metadata.get("no_order_failure_reason") or cycle.validation_failure_reason):
        return False
    status = str(cycle.status or "").lower()
    substatus = str(cycle.execution_substatus or "").lower()
    return status in {"complete", "failed"} or substatus in {"limited", "failed_no_execution", "error"}


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
    return {
        "kind": "market_data_backoff",
        "message": _sanitize_cycle_reason(error or blocker),
        "blocker_category": blocker or _blocker_category(error),
        "retry_after": backoff_until,
    } if not _runtime_notice_expired({"retry_after": backoff_until}) else {}


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
            dict.fromkeys(
                str(reason)
                for row in discovery
                for reason in (row.get("feature_skip_reasons", []) or [])
                if str(reason)
            )
        ),
    }


def _cycle_forecast_blockers(cycle: VaultCycle) -> list[str]:
    blockers: list[str] = []
    for leg in cycle.allocation_legs:
        details = leg.details
        blockers.extend(_hard_forecast_blockers(details.get("forecast_blockers", []) or [], cycle))
    for provider in cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []:
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
    for provider in cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []:
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
    for provider in cycle.selection_metadata.get("provider_allocation_history", cycle.selection_metadata.get("exchange_allocation_history", [])) or []:
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
        ("provider_market_data_unavailable", ("provider_market_data_unavailable", "provider-specific market data", "market data unavailable")),
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
    realized = sum(
        float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0)
        for fill in fills
    )
    risk_reward = _risk_reward_ratio(order, average_fill)
    return {
        "id": order.id,
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "status": order.status,
        "order_type": order.order_type,
        "provider": order.details.get("provider") or order.details.get("execution_venue"),
        "trading_connection_id": order.trading_connection_id,
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
        "blocker_category": order.details.get("blocker_category") or _blocker_category(order.details.get("risk_rejection_reason") or order.rejection_reason),
        "ml_policy_authority": (order.details.get("risk_decision", {}).get("details", {}) if isinstance(order.details.get("risk_decision"), dict) else {}).get("ml_policy_authority"),
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
        forecast.get("suggested_stop_loss_pct")
        or parameters.get("stop_loss_pct", parameters.get("fallback_stop_loss_pct", 0.0))
        or 0.0
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


def _cycle_profit_explanation() -> list[str]:
    return [
        "Allocated tokens move from available balance to locked balance for each cycle.",
        "The selected strategy can trade only up to that cycle allocation cap.",
        "Fills update realized PnL; open positions update unrealized PnL.",
        "Estimated value is starting value plus realized and unrealized PnL after fees.",
        "Settlement converts final estimated value into the selected settlement asset.",
        "Profit can improve through better entries, lower fees and slippage, tighter sizing, stronger edge, and better strategy selection; losses remain possible.",
    ]
