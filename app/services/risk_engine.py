"""Pre-trade risk controls enforced before every order."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..extensions import db
from ..ml.offline_ranker import OfflineRanker
from ..models import AuditLog, Fill, Order, PositionSnapshot, RiskEvent, Setting, StrategyValidation


VALID_MODES = {"live"}


@dataclass(slots=True)
class RiskDecision:
    """Structured result for a risk evaluation."""

    approved: bool
    rule_name: str = "ok"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "details": dict(self.details),
        }


class RiskEngine:
    """Evaluates hard risk rules before order submission."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def evaluate(self, intent: Any, market_price: float | None, has_trading_access: bool) -> RiskDecision:
        mode = str(getattr(intent, "mode", "")).lower()
        symbol = str(getattr(intent, "symbol", "")).upper()
        quantity = self._safe_float(getattr(intent, "quantity", 0.0))
        leverage = self._safe_float(getattr(intent, "leverage", 1.0), 1.0)
        limit_price = self._optional_positive_float(getattr(intent, "limit_price", None))
        reduce_only = bool(getattr(intent, "reduce_only", False))
        metadata = dict(getattr(intent, "metadata", {}) or {})
        user_id = getattr(intent, "user_id", None)
        trading_connection_id = getattr(intent, "trading_connection_id", None)
        shadow_validation_id: int | None = None

        if mode not in VALID_MODES:
            return self._reject("invalid_mode", f"Unsupported trading mode: {mode}")

        if not symbol:
            return self._reject("invalid_symbol", "Order symbol is required.")

        if Setting.get_json("panic_lock", False):
            return self._reject("panic_lock", "Trading is disabled until panic lock is manually reset.")

        if mode == "live" and not self._config_bool("ENABLE_LIVE_TRADING"):
            return self._reject("live_disabled", "Live trading is disabled by configuration.")

        if mode == "live" and self._is_max_return_objective(metadata) and not bool(self.config.get("MAX_RETURN_LIVE_ELIGIBLE", False)):
            return self._reject(
                "max_return_live_disabled",
                "Max-return v3 live orders require MAX_RETURN_LIVE_ELIGIBLE=true.",
                {"profit_objective_version": metadata.get("profit_objective_version")},
            )

        if mode == "live" and self._is_pair_trade(metadata) and not bool(self.config.get("PAIR_LIVE_ELIGIBLE", False)):
            return self._reject(
                "pair_live_disabled",
                "Pair trades require PAIR_LIVE_ELIGIBLE=true before live orders are allowed.",
                {
                    "pair_group_id": metadata.get("pair_group_id"),
                    "pair_mode": metadata.get("pair_mode"),
                    "pair_symbol": metadata.get("pair_symbol"),
                },
            )

        if mode == "live" and self._is_dynamic_intraday(metadata) and not bool(self.config.get("DYNAMIC_INTRADAY_LIVE_ELIGIBLE", False)):
            return self._reject(
                "dynamic_intraday_live_disabled",
                "Dynamic intraday live orders require DYNAMIC_INTRADAY_LIVE_ELIGIBLE=true.",
                {"optimizer_profile": metadata.get("optimizer_profile")},
            )

        if mode == "live" and self._is_high_upside_profile(metadata):
            high_upside_decision = self._evaluate_high_upside_gate(metadata)
            if not high_upside_decision.approved:
                return high_upside_decision

        if mode == "live":
            live_decision = self._evaluate_live_gate(intent)
            if not live_decision.approved:
                return live_decision

        if mode in {"testnet", "live"} and not has_trading_access:
            return self._reject(
                "credentials_missing",
                "Exchange trading credentials are missing or disabled for this mode.",
            )

        allowed_symbols = {str(item).upper() for item in self.config.get("ALLOWED_SYMBOLS", [])}
        if allowed_symbols and symbol not in allowed_symbols:
            return self._reject("symbol_not_allowed", f"{symbol} is not in the approved symbol list.")

        if quantity <= 0:
            return self._reject("invalid_size", "Order quantity must be greater than zero.")

        max_leverage = self._config_float("MAX_LEVERAGE", 1.0)
        if leverage <= 0 or leverage > max_leverage:
            return self._reject(
                "max_leverage",
                f"Requested leverage exceeds configured cap of {max_leverage}.",
                {"requested_leverage": leverage, "max_leverage": max_leverage},
            )

        if mode == "live" and self._uses_experimental_live_caps(metadata):
            aggressive_live_cap = min(
                max_leverage,
                self._config_float("AGGRESSIVE_MAX_LIVE_LEVERAGE", max_leverage),
            )
            if leverage > aggressive_live_cap:
                return self._reject(
                    "aggressive_live_leverage_cap",
                    f"Aggressive live leverage exceeds configured cap of {aggressive_live_cap}.",
                    {"requested_leverage": leverage, "max_leverage": aggressive_live_cap},
                )

        reference_price = self._resolve_reference_price(market_price, limit_price)
        if reference_price <= 0:
            return self._reject("price_unavailable", "A reference price is required to perform risk checks.")

        notional = reference_price * quantity
        max_notional = (
            self._config_float("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD", 0.0)
            if mode == "live" and self._is_high_upside_profile(metadata)
            else self._config_float("MAX_POSITION_NOTIONAL", 0.0)
        )

        if max_notional > 0 and notional > max_notional:
            return self._reject(
                "high_upside_max_notional" if self._is_high_upside_profile(metadata) else "max_notional",
                f"Order notional {notional:.2f} exceeds cap {max_notional:.2f}.",
                {"notional": notional, "max_notional": max_notional},
            )

        if mode == "live" and not reduce_only and self._uses_experimental_live_caps(metadata):
            cap_decision = self._validate_aggressive_live_cap(metadata, notional, user_id, trading_connection_id)
            if not cap_decision.approved:
                return cap_decision

        stop_loss = self._optional_positive_float(getattr(intent, "stop_loss", None))
        take_profit = self._optional_positive_float(getattr(intent, "take_profit", None))

        if not reduce_only:
            if stop_loss is None:
                return self._reject("stop_loss_required", "Opening trades must include a stop loss.")

            stop_decision = self._validate_stop_loss(intent, reference_price, stop_loss)
            if not stop_decision.approved:
                return stop_decision

        if mode == "live" and not reduce_only and self._requires_aggressive_shadow_validation(metadata):
            validation = self._fresh_shadow_live_validation(intent)
            if validation is None:
                return self._reject(
                    "shadow_live_validation_required",
                    "Aggressive 1H live orders require a recent passed pre-live validation.",
                    {
                        "strategy_name": getattr(intent, "strategy_name", None),
                        "symbol": symbol,
                        "timeframe": getattr(intent, "timeframe", None),
                        "max_age_hours": self._config_float("ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS", 24.0),
                    },
                )
            shadow_validation_id = int(validation.id)
            metadata["shadow_validation_id"] = shadow_validation_id

        if mode == "live" and not reduce_only and self._requires_pair_shadow_validation(metadata):
            validation = self._fresh_shadow_live_validation(intent)
            if validation is None:
                return self._reject(
                    "pair_shadow_live_validation_required",
                    "Live pair trades require a recent passed pair pre-live validation.",
                    {
                        "strategy_name": getattr(intent, "strategy_name", None),
                        "symbol": symbol,
                        "timeframe": getattr(intent, "timeframe", None),
                        "pair_group_id": metadata.get("pair_group_id"),
                        "pair_mode": metadata.get("pair_mode"),
                        "pair_symbol": metadata.get("pair_symbol"),
                        "max_age_hours": self._config_float("ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS", 24.0),
                    },
                )
            shadow_validation_id = int(validation.id)
            metadata["pair_shadow_validation_id"] = shadow_validation_id

        if not reduce_only and metadata.get("rule_based_signal"):
            rr_decision = self._validate_rule_based_reward_risk(metadata, take_profit)
            if not rr_decision.approved:
                return rr_decision

        slippage_pct = self._projected_slippage_pct(intent, reference_price, limit_price)
        max_slippage = self._config_float("MAX_SLIPPAGE_PCT", 0.0)

        if max_slippage >= 0 and slippage_pct > max_slippage:
            return self._reject(
                "slippage_cap",
                f"Projected slippage {slippage_pct:.4f} exceeds threshold {max_slippage:.4f}.",
                {"slippage_pct": slippage_pct, "max_slippage_pct": max_slippage},
            )

        loss_used = abs(min(0.0, self.daily_realized_pnl(mode, user_id=user_id, trading_connection_id=trading_connection_id)))
        daily_loss_limit = (
            self._config_float("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0)
            if mode == "live" and self._is_high_upside_profile(metadata)
            else self._config_float("MAX_DAILY_LOSS_USDC", 0.0)
        )

        if daily_loss_limit > 0 and loss_used >= daily_loss_limit:
            if mode == "live" and self._is_high_upside_profile(metadata):
                self._disable_high_upside(
                    "daily_loss_limit_breach",
                    {"loss_used": loss_used, "daily_loss_limit": daily_loss_limit},
                )
            return self._reject(
                "daily_loss_limit",
                "Daily loss limit reached; new trades are blocked until the next day.",
                {"loss_used": loss_used, "daily_loss_limit": daily_loss_limit},
            )

        cooldown = self.cooldown_remaining(mode, user_id=user_id, trading_connection_id=trading_connection_id)
        if cooldown > 0 and not reduce_only:
            return self._reject(
                "loss_cooldown",
                f"Recent losses triggered a cooldown. Wait {cooldown} more minute(s).",
                {"cooldown_minutes_remaining": cooldown},
            )

        return RiskDecision(
            approved=True,
            details={
                "mode": mode,
                "symbol": symbol,
                "reference_price": reference_price,
                "notional": notional,
                "slippage_pct": slippage_pct,
                "fibonacci_alignment": metadata.get("fibonacci_alignment", {}),
                "reward_risk": metadata.get("risk_reward"),
                "shadow_validation_id": shadow_validation_id,
                "pair_shadow_validation_id": metadata.get("pair_shadow_validation_id"),
            },
        )

    def _evaluate_live_gate(self, intent: Any) -> RiskDecision:
        if Setting.get_json("live_trading_blocked", False):
            return self._reject("live_blocked", "Live trading is blocked after a prior failure; review and reset first.")

        return RiskDecision(approved=True)

    def _validate_aggressive_live_cap(
        self,
        metadata: dict[str, Any],
        requested_notional: float,
        user_id: int | None,
        trading_connection_id: int | None,
    ) -> RiskDecision:
        duration_bucket = self._duration_bucket(metadata.get("duration_hours") or metadata.get("lock_duration_hours") or metadata.get("duration_bucket"))
        if self._is_high_upside_profile(metadata):
            cap_usdc = max(0.0, self._duration_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", duration_bucket, 0.0))
            cap_pct = max(0.0, self._duration_cap("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION", duration_bucket, 0.0))
            rule_name = "high_upside_live_cap"
            message = "High-upside live exposure exceeds configured cap."
        else:
            cap_usdc = max(0.0, self._duration_cap("EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION", duration_bucket, self._config_float("AGGRESSIVE_1H_LIVE_CAP_USDC", 25.0)))
            cap_pct = max(0.0, self._duration_cap("EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION", duration_bucket, self._config_float("AGGRESSIVE_1H_LIVE_CAP_PCT", 0.02)))
            rule_name = "aggressive_1h_live_cap"
            message = "Aggressive 1H live exposure exceeds configured cap."
        account_equity = self._safe_float(metadata.get("account_equity_usd"))
        equity_cap = account_equity * cap_pct if account_equity > 0 else cap_usdc
        effective_cap = min(cap_usdc, equity_cap) if cap_usdc > 0 else equity_cap
        current_exposure = self._aggressive_live_exposure(user_id, trading_connection_id)
        projected = current_exposure + max(0.0, requested_notional)

        if effective_cap > 0 and projected > effective_cap + 1e-9:
            return self._reject(
                rule_name,
                message,
                {
                    "requested_notional": requested_notional,
                    "current_exposure": current_exposure,
                    "projected_exposure": projected,
                    "effective_cap": effective_cap,
                    "cap_usdc": cap_usdc,
                    "cap_pct": cap_pct,
                    "duration_bucket": duration_bucket,
                    "account_equity_usd": account_equity,
                },
            )

        return RiskDecision(approved=True)

    def _aggressive_live_exposure(self, user_id: int | None, trading_connection_id: int | None) -> float:
        query = Order.query.filter_by(mode="live")
        if user_id is not None:
            query = query.filter_by(user_id=int(user_id))
        if trading_connection_id is not None:
            query = query.filter_by(trading_connection_id=int(trading_connection_id))
        aggressive_orders = [order for order in query.order_by(Order.created_at.desc()).limit(250).all() if self._uses_experimental_live_caps(order.details)]
        symbols = {order.symbol for order in aggressive_orders}
        exposure = 0.0

        for order in aggressive_orders:
            if order.reduce_only or order.status not in {"open", "submitted", "pending"}:
                continue
            price = self._safe_float(order.average_fill_price) or self._safe_float(order.limit_price)
            exposure += abs(self._safe_float(order.quantity) * price)

        for symbol in symbols:
            snapshot = (
                self._position_query("live", symbol, user_id, trading_connection_id)
                .order_by(PositionSnapshot.snapshot_time.desc())
                .first()
            )
            if snapshot is not None:
                exposure += abs(self._safe_float(snapshot.notional))

        return exposure

    @staticmethod
    def _is_aggressive_1h(metadata: dict[str, Any]) -> bool:
        return str((metadata or {}).get("optimizer_profile", "")).lower() == "aggressive_1h"

    @staticmethod
    def _is_aggressive_experimental(metadata: dict[str, Any]) -> bool:
        return str((metadata or {}).get("optimizer_profile", "")).lower() in {
            "aggressive_1h",
            "extreme_roi_experimental",
        }

    def _is_experimental_duration_ensemble(self, metadata: dict[str, Any]) -> bool:
        if not bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE", False)):
            return False
        payload = metadata or {}
        return (
            str(payload.get("ensemble_version", "")).lower() == "duration_experimental_v1"
            or str(payload.get("allocation_mode", "")).lower() == "duration_experimental_ensemble"
            or bool(payload.get("duration_ensemble_enabled", False))
        )

    @staticmethod
    def _is_max_return_objective(metadata: dict[str, Any]) -> bool:
        payload = metadata or {}
        return str(payload.get("profit_objective_version", "")).lower() == "max_return_v3"

    def _is_max_return_live_eligible(self, metadata: dict[str, Any]) -> bool:
        return bool(self.config.get("MAX_RETURN_LIVE_ELIGIBLE", False)) and self._is_max_return_objective(metadata)

    @staticmethod
    def _is_pair_trade(metadata: dict[str, Any]) -> bool:
        payload = metadata or {}
        return bool(payload.get("pair_group_id") or payload.get("pair_mode") or payload.get("pair_symbol"))

    def _is_pair_live_eligible(self, metadata: dict[str, Any]) -> bool:
        return bool(self.config.get("PAIR_LIVE_ELIGIBLE", False)) and self._is_pair_trade(metadata)

    @staticmethod
    def _is_dynamic_intraday(metadata: dict[str, Any]) -> bool:
        return str((metadata or {}).get("optimizer_profile", "")).lower() == "dynamic_intraday"

    def _is_dynamic_intraday_live_eligible(self, metadata: dict[str, Any]) -> bool:
        return bool(self.config.get("DYNAMIC_INTRADAY_LIVE_ELIGIBLE", False)) and self._is_dynamic_intraday(metadata)

    @staticmethod
    def _is_high_upside_profile(metadata: dict[str, Any]) -> bool:
        value = (metadata or {}).get("high_upside_profile", False)
        return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _uses_experimental_live_caps(self, metadata: dict[str, Any]) -> bool:
        return (
            self._is_aggressive_experimental(metadata)
            or self._is_experimental_duration_ensemble(metadata)
            or self._is_max_return_live_eligible(metadata)
            or self._is_pair_live_eligible(metadata)
            or self._is_dynamic_intraday_live_eligible(metadata)
            or self._is_high_upside_profile(metadata)
        )

    def _requires_aggressive_shadow_validation(self, metadata: dict[str, Any]) -> bool:
        return (
            (
                bool(self.config.get("ENSEMBLE_1H_REQUIRE_SHADOW_VALIDATION", True))
                or (self._is_dynamic_intraday(metadata) and bool(self.config.get("DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION", True)))
            )
            and self._uses_experimental_live_caps(metadata)
            and not self._is_pair_live_eligible(metadata)
        )

    def _requires_pair_shadow_validation(self, metadata: dict[str, Any]) -> bool:
        return (
            self._is_pair_live_eligible(metadata)
            and bool(self.config.get("PAIR_REQUIRE_SHADOW_VALIDATION", True))
        )

    @staticmethod
    def _position_query(
        mode: str,
        symbol: str,
        user_id: int | None,
        trading_connection_id: int | None,
    ):
        query = PositionSnapshot.query.filter_by(mode=mode, symbol=symbol)
        if user_id is not None:
            query = query.filter_by(user_id=int(user_id))
        if trading_connection_id is not None:
            query = query.filter_by(trading_connection_id=int(trading_connection_id))
        return query

    def _has_shadow_live_validation(self, intent: Any) -> bool:
        return self._fresh_shadow_live_validation(intent) is not None

    def _fresh_shadow_live_validation(self, intent: Any) -> StrategyValidation | None:
        strategy_name = getattr(intent, "strategy_name", None)
        if not strategy_name:
            return None

        query = StrategyValidation.query.filter_by(
            strategy_name=strategy_name,
            symbol=str(getattr(intent, "symbol", "")).upper(),
            stage="shadow_live",
            status="passed",
        )

        timeframe = getattr(intent, "timeframe", None)
        if timeframe:
            query = query.filter_by(timeframe=timeframe)

        validation = query.order_by(StrategyValidation.completed_at.desc()).first()
        if validation is None or validation.completed_at is None:
            return None

        max_age_hours = self._config_float("ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS", 24.0)
        if max_age_hours > 0:
            completed_at = self._as_aware_utc(validation.completed_at)
            if completed_at < datetime.now(timezone.utc) - timedelta(hours=max_age_hours):
                return None

        return validation

    def daily_realized_pnl(
        self,
        mode: str,
        *,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> float:
        since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        query = (
            Fill.query.join(Fill.order)
            .filter(Fill.fill_time >= since)
            .filter(Order.mode == mode)
            .filter(Fill.simulated == (mode == "paper"))
        )
        if user_id is not None:
            query = query.filter(Order.user_id == int(user_id))
        if trading_connection_id is not None:
            query = query.filter(Order.trading_connection_id == int(trading_connection_id))
        fills = query.all()

        return sum(self._safe_float(fill.pnl) - self._safe_float(fill.fee) for fill in fills)

    def cooldown_remaining(
        self,
        mode: str,
        *,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> int:
        cooldown_minutes = max(0, self._config_int("LOSS_COOLDOWN_MINUTES", 0))
        if cooldown_minutes <= 0:
            return 0

        now = datetime.now(timezone.utc)
        window = now - timedelta(minutes=cooldown_minutes)

        query = (
            Fill.query.join(Fill.order)
            .filter(Fill.fill_time >= window)
            .filter(Fill.pnl < 0)
            .filter(Order.mode == mode)
            .filter(Fill.simulated == (mode == "paper"))
        )
        if user_id is not None:
            query = query.filter(Order.user_id == int(user_id))
        if trading_connection_id is not None:
            query = query.filter(Order.trading_connection_id == int(trading_connection_id))
        fill = query.order_by(Fill.fill_time.desc()).first()

        if fill is None:
            return 0

        fill_time = self._as_aware_utc(fill.fill_time)
        remaining = timedelta(minutes=cooldown_minutes) - (now - fill_time)

        return max(0, int(remaining.total_seconds() // 60))

    def status(
        self,
        mode: str,
        *,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> dict[str, Any]:
        return {
            "panic_lock": bool(Setting.get_json("panic_lock", False)),
            "live_trading_blocked": bool(Setting.get_json("live_trading_blocked", False)),
            "explicit_live_confirmed": True,
            "secondary_confirmation": True,
            "daily_realized_pnl": self.daily_realized_pnl(mode, user_id=user_id, trading_connection_id=trading_connection_id),
            "daily_loss_limit": self._config_float("MAX_DAILY_LOSS_USDC", 0.0),
            "cooldown_minutes_remaining": self.cooldown_remaining(mode, user_id=user_id, trading_connection_id=trading_connection_id),
            "max_leverage": self._config_float("MAX_LEVERAGE", 1.0),
            "max_position_notional": self._config_float("MAX_POSITION_NOTIONAL", 0.0),
            "max_slippage_pct": self._config_float("MAX_SLIPPAGE_PCT", 0.0),
            "high_upside": {
                "profile_enabled": bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)),
                "live_eligible": bool(self.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)),
                "auto_disabled": bool(Setting.get_json("high_upside_live_disabled", False)),
                "disabled_reason": Setting.get_json("high_upside_live_disabled_reason", {}),
                "max_position_notional": self._config_float("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD", 0.0),
                "max_daily_loss": self._config_float("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0),
                "requires_promoted_offline_ml": bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)),
                "offline_ml_readiness": OfflineRanker(self.config).readiness(self._duration_bucket("1h"), require_blend=False),
            },
        }

    def _evaluate_high_upside_gate(self, metadata: dict[str, Any]) -> RiskDecision:
        if Setting.get_json("high_upside_live_disabled", False):
            return self._reject(
                "high_upside_auto_disabled",
                "High-upside live profile is auto-disabled until operator review.",
                {"disabled_reason": Setting.get_json("high_upside_live_disabled_reason", {})},
            )
        if not bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)):
            return self._reject("high_upside_profile_disabled", "High-upside profile requires HIGH_UPSIDE_PROFILE_ENABLED=true.")
        if not bool(self.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)):
            return self._reject("high_upside_live_disabled", "High-upside live orders require HIGH_UPSIDE_LIVE_ELIGIBLE=true.")
        missing: list[str] = []
        if self._config_float("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD", 0.0) <= 0:
            missing.append("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD")
        if self._config_float("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0) <= 0:
            missing.append("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC")
        duration_bucket = self._duration_bucket(metadata.get("duration_hours") or metadata.get("lock_duration_hours") or metadata.get("duration_bucket"))
        if self._duration_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", duration_bucket, 0.0) <= 0:
            missing.append("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION_JSON")
        if self._duration_cap("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION", duration_bucket, 0.0) <= 0:
            missing.append("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION_JSON")
        if missing:
            return self._reject(
                "high_upside_caps_missing",
                "High-upside profile is missing required live cap configuration.",
                {"missing": missing, "duration_bucket": duration_bucket},
            )
        ml_decision = self._evaluate_high_upside_ml_gate(duration_bucket)
        if not ml_decision.approved:
            return ml_decision
        drawdown = self._safe_float(metadata.get("max_drawdown"))
        drawdown_cap = abs(self._config_float("AGGRESSIVE_1H_MAX_DRAWDOWN_PCT", 0.35))
        if drawdown < 0 and drawdown <= -drawdown_cap:
            self._disable_high_upside("drawdown_breach", {"max_drawdown": drawdown, "drawdown_cap": drawdown_cap})
            return self._reject(
                "high_upside_drawdown_breach",
                "High-upside profile drawdown diagnostics breached the configured cap.",
                {"max_drawdown": drawdown, "drawdown_cap": drawdown_cap},
            )
        return RiskDecision(approved=True)

    def _evaluate_high_upside_ml_gate(self, duration_bucket: str) -> RiskDecision:
        if not bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)):
            return RiskDecision(approved=True)
        readiness = OfflineRanker(self.config).readiness(duration_bucket, require_blend=False)
        if bool(readiness.get("ready", False)):
            return RiskDecision(approved=True)
        blockers = list(readiness.get("blockers", []))
        diagnostic_breach = any(
            blocker in blockers
            for blocker in {
                "validation_loss_above_threshold",
                "negative_error_rate_above_threshold",
                "calibration_error_above_threshold",
                "top_decile_precision_below_threshold",
                "false_positive_high_upside_rate_above_threshold",
                "prediction_drift_above_threshold",
                "model_age_above_threshold",
                "feature_schema_version_mismatch",
            }
        )
        if diagnostic_breach:
            self._disable_high_upside(
                "offline_model_diagnostic_breach",
                {"duration_bucket": duration_bucket, "blockers": blockers},
            )
        return self._reject(
            "high_upside_promoted_ml_required",
            "High-upside live orders require a promoted offline ML model that passes readiness checks.",
            {"duration_bucket": duration_bucket, "offline_ml_readiness": readiness},
        )

    def _disable_high_upside(self, reason: str, details: dict[str, Any]) -> None:
        Setting.set_json("high_upside_live_disabled", True)
        Setting.set_json("high_upside_live_disabled_reason", {"reason": reason, **dict(details or {})})
        db.session.add(
            AuditLog(
                category="risk",
                action="high_upside_auto_disabled",
                message=f"High-upside live profile auto-disabled: {reason}",
            )
        )
        db.session.flush()

    def log_rejection(self, decision: RiskDecision, payload: dict[str, Any], order_id: int | None = None) -> None:
        details = {**dict(payload or {}), **dict(decision.details or {})}

        event = RiskEvent(
            order_id=order_id,
            user_id=details.get("user_id"),
            trading_connection_id=details.get("trading_connection_id"),
            rule_name=decision.rule_name,
            reason=decision.reason,
        )
        event.payload = details
        db.session.add(event)

        audit = AuditLog(
            category="risk",
            action=decision.rule_name,
            message=decision.reason,
            user_id=details.get("user_id"),
            trading_connection_id=details.get("trading_connection_id"),
        )
        audit.details = details
        db.session.add(audit)

        db.session.commit()

    def _validate_stop_loss(self, intent: Any, reference_price: float, stop_loss: float) -> RiskDecision:
        side = str(getattr(intent, "side", "")).lower()

        if side == "buy" and stop_loss >= reference_price:
            return self._reject(
                "invalid_stop_loss",
                "Buy orders require stop loss below the reference price.",
                {"reference_price": reference_price, "stop_loss": stop_loss},
            )

        if side == "sell" and stop_loss <= reference_price:
            return self._reject(
                "invalid_stop_loss",
                "Sell orders require stop loss above the reference price.",
                {"reference_price": reference_price, "stop_loss": stop_loss},
            )

        return RiskDecision(approved=True)

    def _validate_rule_based_reward_risk(
        self,
        metadata: dict[str, Any],
        take_profit: float | None,
    ) -> RiskDecision:
        reward_risk = self._safe_float(metadata.get("risk_reward"))
        minimum_reward_risk = self._config_float("MIN_REWARD_RISK", 1.0)

        if take_profit is None:
            return self._reject("take_profit_required", "Rule-based opening trades must include take profit.")

        if reward_risk < minimum_reward_risk:
            return self._reject(
                "invalid_reward_risk",
                f"Rule-based reward/risk {reward_risk:.2f} is below minimum {minimum_reward_risk:.2f}.",
                {"reward_risk": reward_risk, "minimum_reward_risk": minimum_reward_risk},
            )

        return RiskDecision(approved=True)

    def _projected_slippage_pct(self, intent: Any, reference_price: float, limit_price: float | None) -> float:
        if limit_price is not None:
            return abs(limit_price - reference_price) / reference_price

        return max(0.0, self._safe_float(getattr(intent, "slippage_pct", 0.0)))

    def _duration_cap(self, key: str, bucket: str, default: float) -> float:
        mapping = self.config.get(key) or {}
        if not isinstance(mapping, dict):
            return default
        for candidate in (bucket, str(bucket).lower()):
            if candidate in mapping:
                return self._safe_float(mapping.get(candidate), default)
        return default

    @staticmethod
    def _duration_bucket(value: Any) -> str:
        raw = str(value or "").lower()
        if raw in {"1h", "24h", "48h", "72h", "7d"}:
            return raw
        try:
            duration = max(1, int(float(raw)))
        except (TypeError, ValueError):
            return "1h"
        if duration <= 1:
            return "1h"
        if duration <= 24:
            return "24h"
        if duration <= 48:
            return "48h"
        if duration <= 72:
            return "72h"
        return "7d"

    @staticmethod
    def _resolve_reference_price(market_price: float | None, limit_price: float | None) -> float:
        market = RiskEngine._safe_float(market_price)
        if market > 0:
            return market

        return RiskEngine._safe_float(limit_price)

    @staticmethod
    def _optional_positive_float(value: Any) -> float | None:
        parsed = RiskEngine._safe_float(value)
        return parsed if parsed > 0 else None

    @staticmethod
    def _as_aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    def _config_float(self, key: str, default: float = 0.0) -> float:
        return self._safe_float(self.config.get(key), default)

    def _config_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _config_bool(self, key: str) -> bool:
        return bool(self.config.get(key, False))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _reject(rule_name: str, reason: str, details: dict[str, Any] | None = None) -> RiskDecision:
        return RiskDecision(approved=False, rule_name=rule_name, reason=reason, details=details or {})
