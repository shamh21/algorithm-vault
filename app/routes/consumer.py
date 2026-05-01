"""Consumer wallet and vault routes."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from sqlalchemy.exc import OperationalError

from ..auth import current_user, qr_code_data_uri, require_authenticated_user, verify_totp
from ..extensions import db
from ..ml.online_ranker import extract_features, horizon_from_duration, outcome_from_result
from ..models import AuditLog, DepositAddress, Fill, Order, Setting, StrategyRun, TradingConnection, VaultAllocationLeg, VaultCycle, WalletAddress, WalletBalance, WalletTransaction
from ..runtime import get_current_mode, get_service, market_mode_for
from ..services.db_retry import commit_with_retry, is_database_locked
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
    exchange_snapshot = _exchange_balance_snapshot(user)
    active_cycles = _active_cycles(user)
    active_cycle = active_cycles[0] if active_cycles else None
    recent_transactions = WalletTransaction.query.filter_by(user_id=user.id).order_by(WalletTransaction.created_at.desc()).limit(5).all()
    return render_template(
        "home.html",
        balances=balances,
        exchange_snapshot=exchange_snapshot,
        portfolio_total=_portfolio_total(balances),
        active_cycle=active_cycle,
        recent_transactions=recent_transactions,
    )


@consumer_bp.get("/wallet/", strict_slashes=False)
def wallet():
    user = current_user()
    _sync_completed_cycles(user)
    balances = _wallet_balances(user)
    exchange_snapshot = _exchange_balance_snapshot(user)
    transactions = WalletTransaction.query.filter_by(user_id=user.id).order_by(WalletTransaction.created_at.desc()).limit(20).all()
    return render_template(
        "wallet.html",
        balances=balances,
        exchange_snapshot=exchange_snapshot,
        portfolio_total=_portfolio_total(balances),
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
    form_values = {"withdraw_address": "", "amount": "", "network": network}

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
        form_values["amount"] = request.form.get("amount", "")
        code = request.form.get("totp_code", "").strip()
        network = request.form.get("network", network).strip()
        form_values["network"] = network

        if network not in _asset_networks(asset):
            errors["network"] = "Select a supported network."
        if not validate_withdraw_address(withdraw_address, asset, network):
            errors["withdraw_address"] = "Enter a valid destination address for the selected asset and network."
        if amount <= 0:
            errors["amount"] = "Enter a withdrawal amount greater than zero."
        elif amount > float(balance.available_balance or 0.0) + 1e-9:
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
            if withdrawal.status == "failed":
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
                flash("Withdrawal submitted. Check Activity for the latest status.", "success")
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
    )


@consumer_bp.get("/vault/", strict_slashes=False)
def vault():
    user = current_user()
    _sync_completed_cycles(user)
    balances = _wallet_balances(user)
    active_cycles = _active_cycles(user)
    recent_cycles = VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.started_at.desc()).limit(8).all()
    return render_template(
        "vault.html",
        balances=balances,
        active_cycle=active_cycles[0] if active_cycles else None,
        active_cycles=active_cycles,
        recent_cycles=recent_cycles,
        settlement_assets=_wallet_assets(),
    )


@consumer_bp.post("/vault/start")
def start_cycle():
    user = current_user()
    _sync_completed_cycles(user)
    connection = _active_trading_connection(user)
    if _live_connection_required() and connection is None:
        flash("Connect your trading account before starting a live vault cycle.", "warning")
        return redirect(url_for("settings.connections"))

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

    balances = _wallet_balances(user)
    balance = next((item for item in balances if item.asset == asset), None)
    if balance is None or float(balance.available_balance) + 1e-9 < amount:
        flash("That allocation is higher than the available wallet balance.", "danger")
        return redirect(url_for("consumer.vault"))

    price = _asset_usd_price(asset)
    if price <= 0:
        flash("Market estimate is unavailable for that asset. Try a stable settlement asset or retry later.", "warning")
        return redirect(url_for("consumer.vault"))

    starting_value_usd = amount * price
    allowed_symbols = _requested_allowed_symbols()
    selection = get_service("vault_strategy_selector").select(
        asset,
        duration_hours,
        get_current_mode(),
        starting_value_usd,
        allowed_symbols=allowed_symbols,
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
        algorithm_profile=selection.profile,
        selected_strategy_name=selection.strategy_name,
        selected_timeframe=selection.timeframe,
        started_at=now,
        unlocks_at=now + timedelta(seconds=duration_seconds),
        starting_value_usd=starting_value_usd,
        current_estimated_value_usd=starting_value_usd,
    )
    cycle.selection_metadata = selection.metadata
    db.session.add(cycle)
    db.session.flush()

    common_parameters = {
        "vault_cycle_id": cycle.id,
        "consumer_vault": True,
        "algorithm_profile": selection.profile,
        "execution_mode": selection.execution_mode,
        "live_validation_status": selection.live_validation_status,
        "live_validation_started_at": now.isoformat() if selection.live_validation_status == "pending" else None,
        "lock_duration_hours": duration_hours,
        "lock_duration_seconds": duration_seconds,
        "allowed_symbols": allowed_symbols,
        "user_id": user.id,
        "trading_connection_id": connection.id if connection is not None else None,
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
                "pair_group_id": leg.get("pair_group_id") or selection.metadata.get("pair_group_id"),
                "pair_mode": leg.get("pair_mode") or selection.metadata.get("pair_mode"),
                "pair_symbol": leg.get("pair_symbol") or selection.metadata.get("pair_symbol"),
                "pair_role": leg.get("pair_role"),
                "pair_forced_side": (leg.get("parameters") or {}).get("pair_forced_side"),
                "hedge_ratio": leg.get("hedge_ratio") or selection.metadata.get("hedge_ratio"),
                "spread_zscore": leg.get("spread_zscore") or selection.metadata.get("spread_zscore"),
                "spread_half_life": leg.get("spread_half_life") or selection.metadata.get("spread_half_life"),
                "pair_score": leg.get("pair_score") or selection.metadata.get("pair_score"),
                "correlation": leg.get("correlation") or selection.metadata.get("correlation"),
                "pair_signal": leg.get("pair_signal") or selection.metadata.get("pair_signal", {}),
                "pair_skip_reason": leg.get("pair_skip_reason") or selection.metadata.get("pair_skip_reason", ""),
                "skip_reason": leg.get("skip_reason", ""),
                "leverage": float(leg.get("leverage", leg_parameters.get("leverage", 1.0)) or 1.0),
            }
        )
        run = StrategyRun(
            strategy_name=str(leg.get("strategy_name") or selection.strategy_name),
            symbol=str(leg.get("symbol") or selection.symbol),
            timeframe=str(leg.get("timeframe") or selection.timeframe),
            mode=selection.mode,
            user_id=user.id,
            trading_connection_id=connection.id if connection is not None else None,
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

    for run_id in run_ids:
        get_service("strategy_manager").start(run_id)
    flash("Vault cycle started. Estimated performance is not guaranteed and execution remains risk-gated.", "success")
    return redirect(url_for("consumer.vault"))


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
    if cycle.status in {"active", "settling"}:
        _refresh_cycle_performance(cycle)
        commit_with_retry()
    summary = cycle.cycle_summary or _cycle_summary(cycle)
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
    balances = WalletBalance.query.filter_by(user_id=user.id).order_by(WalletBalance.asset.asc()).all()
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
        balances = WalletBalance.query.filter_by(user_id=user.id).order_by(WalletBalance.asset.asc()).all()
    else:
        existing = {balance.asset for balance in balances}
        for asset in wallet_assets:
            if asset not in existing:
                db.session.add(WalletBalance(user_id=user.id, asset=asset, estimated_usd_value=0.0))
        commit_with_retry()
        balances = WalletBalance.query.filter_by(user_id=user.id).order_by(WalletBalance.asset.asc()).all()

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
    snapshot = get_service("trading_connections").account_snapshot(user.id, "live", connection.id)
    balances: list[dict[str, float | str]] = []
    for item in snapshot.balances:
        asset = str(item.get("asset", "") or "").upper()
        if not asset:
            continue
        value = float(item.get("value", 0.0) or 0.0)
        withdrawable = float(item.get("withdrawable", value) or 0.0)
        balances.append(
            {
                "asset": asset,
                "type": str(item.get("type", "margin") or "margin"),
                "value": value,
                "withdrawable": max(withdrawable, 0.0),
                "estimated_usd_value": value if asset in {"USDC", "USDT", "USD"} else value * _asset_usd_price(asset),
            }
        )
    if balances or snapshot.alerts:
        Setting.set_json(
            _exchange_balance_snapshot_key(user.id),
            {
                "mode": "live",
                "connection_id": connection.id,
                "provider": connection.provider,
                "balances": balances,
                "positions_count": len(snapshot.positions or []),
                "open_orders_count": len(snapshot.open_orders or []),
                "alerts": snapshot.alerts or [],
                "synced_at": datetime.utcnow().isoformat() + "Z",
            },
        )
        commit_with_retry()


def _exchange_balance_snapshot(user) -> dict:
    _sync_connection_balances(user)
    value = Setting.get_json(_exchange_balance_snapshot_key(user.id), {})
    return value if isinstance(value, dict) else {}


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


def _portfolio_total(balances: list[WalletBalance]) -> float:
    return sum(float(balance.estimated_usd_value or 0.0) for balance in balances)


def _live_connection_required() -> bool:
    return bool(current_app.config.get("ENABLE_LIVE_TRADING", False)) and get_current_mode() == "live"


def _active_trading_connection(user) -> TradingConnection | None:
    return get_service("trading_connections").active_tradable_connection(user.id)


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
    duration_bucket = horizon_from_duration(duration_hours)
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
                if horizon_from_duration(cycle.lock_duration_hours) == duration_bucket
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


def _estimated_cycle_value(cycle: VaultCycle) -> float:
    performance = _cycle_performance(cycle)
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
    cycle.current_estimated_value_usd = _estimated_cycle_value(cycle)
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
    horizon = horizon_from_duration(cycle.lock_duration_hours)
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


def _cycle_performance(cycle: VaultCycle) -> dict[str, float | bool]:
    cycle_orders = _cycle_orders(cycle)
    realized = 0.0
    for order in cycle_orders:
        realized += sum(float(fill.pnl or 0.0) - float(fill.fee or 0.0) for fill in order.fills)
    unrealized = 0.0
    symbols = {leg.symbol for leg in cycle.allocation_legs if leg.symbol}
    if not symbols and cycle.strategy_run:
        symbols.add(cycle.strategy_run.symbol)
    mode = cycle.strategy_run.mode if cycle.strategy_run else cycle.execution_mode
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
        leg_orders = [order for order in cycle_orders if order.details.get("vault_leg_id") == leg.id or order.symbol == leg.symbol]
        leg.realized_pnl_usd = sum(
            float(fill.pnl or 0.0) - float(fill.fee or 0.0)
            for order in leg_orders
            for fill in order.fills
        )
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


def _cycle_orders(cycle: VaultCycle) -> list[Order]:
    query = Order.query.filter_by(user_id=cycle.user_id)
    if cycle.execution_mode:
        query = query.filter_by(mode=cycle.execution_mode)
    return [
        order
        for order in query.order_by(Order.created_at.asc()).all()
        if order.details.get("vault_cycle_id") == cycle.id
    ]


def _cycle_summary(cycle: VaultCycle) -> dict[str, object]:
    performance = _cycle_performance(cycle)
    orders = _cycle_orders(cycle)
    order_summaries = [_order_summary(order) for order in orders]
    fills = [fill for order in orders for fill in order.fills]
    fees = sum(float(fill.fee or 0.0) for fill in fills)
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
    summary = {
        "cycle_id": cycle.id,
        "status": cycle.status,
        "deposit_asset": cycle.deposit_asset,
        "deposit_amount": float(cycle.deposit_amount or 0.0),
        "settlement_asset": cycle.settlement_asset,
        "lock_duration_seconds": int(cycle.lock_duration_seconds or cycle.lock_duration_hours * 3600),
        "lock_duration_label": format_duration_seconds(cycle.lock_duration_seconds or cycle.lock_duration_hours * 3600),
        "starting_value_usd": float(cycle.starting_value_usd or 0.0),
        "current_estimated_value_usd": float(cycle.current_estimated_value_usd or 0.0),
        "final_settlement_amount": float(cycle.final_settlement_amount or 0.0),
        "execution_mode": cycle.execution_mode,
        "algorithm_profile": cycle.algorithm_profile,
        "symbols": sorted({order.symbol for order in orders} | {leg.symbol for leg in cycle.allocation_legs}),
        "sides": sorted({order.side for order in orders}),
        "order_count": len(orders),
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
        "orders": order_summaries,
        "legs": legs,
        "generated_at": datetime.utcnow().isoformat(),
    }
    return summary


def _order_summary(order: Order) -> dict[str, object]:
    fills = list(order.fills)
    fill_quantity = sum(float(fill.quantity or 0.0) for fill in fills)
    weighted_fill = sum(float(fill.quantity or 0.0) * float(fill.price or 0.0) for fill in fills)
    average_fill = float(order.average_fill_price or 0.0)
    if average_fill <= 0 and fill_quantity > 0:
        average_fill = weighted_fill / fill_quantity
    fees = sum(float(fill.fee or 0.0) for fill in fills)
    realized = sum(float(fill.pnl or 0.0) - float(fill.fee or 0.0) for fill in fills)
    risk_reward = _risk_reward_ratio(order, average_fill)
    return {
        "id": order.id,
        "client_order_id": order.client_order_id,
        "symbol": order.symbol,
        "side": order.side,
        "status": order.status,
        "order_type": order.order_type,
        "quantity": float(order.quantity or 0.0),
        "filled_quantity": float(order.filled_quantity or 0.0),
        "average_fill": average_fill,
        "fees": fees,
        "realized_pnl": realized,
        "leverage": float(order.leverage or 1.0),
        "stop_loss": float(order.stop_loss or 0.0),
        "take_profit": float(order.take_profit or 0.0),
        "risk_reward": risk_reward,
        "slippage_bps": float(order.details.get("slippage_bps", 0.0) or 0.0),
        "edge_score": float(order.details.get("edge_score", 0.0) or 0.0),
        "cost_drag_bps": float(order.details.get("cost_drag_bps", 0.0) or 0.0),
        "signal_confidence": float(order.details.get("signal_confidence", 0.0) or 0.0),
        "quality_reasons": order.details.get("quality_reasons", []),
        "fibonacci_alignment": order.details.get("fibonacci_alignment", {}),
        "feature_confluence": order.details.get("feature_confluence", {}),
        "ml_signal_quality": order.details.get("ml_signal_quality", {}),
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
                "simulated": bool(fill.simulated),
                "fill_time": fill.fill_time.isoformat() if fill.fill_time else None,
            }
            for fill in fills
        ],
    }


def _leg_summary(leg: VaultAllocationLeg) -> dict[str, object]:
    parameters = leg.strategy_run.parameters if leg.strategy_run else {}
    stop = float(parameters.get("stop_loss_pct", parameters.get("fallback_stop_loss_pct", 0.0)) or 0.0)
    take = float(parameters.get("take_profit_pct", parameters.get("fallback_take_profit_pct", 0.0)) or 0.0)
    return {
        "id": leg.id,
        "strategy_run_id": leg.strategy_run_id,
        "strategy_name": leg.strategy_run.strategy_name if leg.strategy_run else None,
        "symbol": leg.symbol,
        "timeframe": leg.timeframe,
        "allocation_cap_usd": float(leg.allocation_cap_usd or 0.0),
        "leverage": float(leg.leverage or 1.0),
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
