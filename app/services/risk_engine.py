"""Pre-trade risk controls enforced before every order."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..extensions import db
from ..ml.decision_engine import MLDecisionEngine
from ..ml.offline_ranker import OfflineRanker
from ..ml.signal_model import MLSignalModel
from ..models import AuditLog, Fill, LeveragedMarket, Order, PositionSnapshot, RiskEvent, Setting, StrategyValidation
from .one_h10_quality import first_one_h10_reason_code, one_h10_forecast_live_blockers
from .provider_assets import normalize_provider

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


@dataclass(slots=True)
class SafetyEnvelope:
    """Non-bypassable live checks that stay outside ML authority."""

    ready: bool
    blockers: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
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
        is_one_h10 = self._is_one_h10(metadata)
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

        if mode == "live" and is_one_h10 and not reduce_only:
            metadata.update(
                {
                    "one_h10_vault": True,
                    "ml_horizon": "1h10",
                    "objective": metadata.get("objective") or "one_h10",
                    "ml_objective": metadata.get("ml_objective") or "one_h10",
                    "ml_policy_required": True,
                    "ml_governed_risk": True,
                }
            )
            try:
                intent.metadata = metadata
            except Exception:  # noqa: BLE001
                pass

        if mode == "live" and is_one_h10:
            one_h10_decision = self._evaluate_one_h10_gate(metadata)
            if not one_h10_decision.approved:
                return one_h10_decision

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

        ml_policy_active = mode == "live" and self._ml_policy_active(metadata)
        if mode == "live" and reduce_only:
            ml_policy_active = False
        elif mode == "live" and is_one_h10 and not reduce_only:
            ml_policy_active = True
        safety_envelope_payload: dict[str, Any] = {}
        ml_policy_payload: dict[str, Any] = {}

        allowed_symbols = {str(item).upper() for item in self.config.get("ALLOWED_SYMBOLS", [])}
        one_h10_all_pairs = is_one_h10 and bool(self.config.get("ONE_H10_ALL_PAIRS_ENABLED", True))
        high_upside_all_pairs = (
            self._is_high_upside_profile(metadata)
            and normalize_provider(metadata.get("provider") or metadata.get("execution_venue")) == "hyperliquid"
            and bool(self.config.get("HIGH_UPSIDE_ALL_PAIRS_ENABLED", True))
            and bool(metadata.get("venue_symbol") or metadata.get("provider_symbol") or symbol)
        )
        rapid_ml_all_futures = self._is_rapid_ml_active_futures_market(metadata, symbol)
        if (
            not reduce_only
            and allowed_symbols
            and symbol not in allowed_symbols
            and not one_h10_all_pairs
            and not high_upside_all_pairs
            and not rapid_ml_all_futures
        ):
            return self._reject("symbol_not_allowed", f"{symbol} is not in the approved symbol list.")

        if quantity <= 0:
            return self._reject("invalid_size", "Order quantity must be greater than zero.")

        max_leverage = self._exchange_max_leverage(metadata, symbol)
        if leverage <= 0 or leverage > max_leverage:
            return self._reject(
                "max_leverage",
                f"Requested leverage exceeds exchange cap of {max_leverage}.",
                {"requested_leverage": leverage, "max_leverage": max_leverage},
            )
        if mode == "live" and is_one_h10:
            one_h10_leverage_cap = min(max_leverage, self._config_float("ONE_H10_MAX_LEVERAGE", max_leverage))
            if leverage > one_h10_leverage_cap:
                return self._reject(
                    "one_h10_leverage_cap",
                    f"1H10 leverage exceeds configured cap of {one_h10_leverage_cap}.",
                    {"requested_leverage": leverage, "max_leverage": one_h10_leverage_cap},
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
        provider_key = normalize_provider(metadata.get("provider") or metadata.get("execution_venue"))
        min_notional = self._config_float("HYPERLIQUID_MIN_ORDER_VALUE_USD", 10.0)
        if mode == "live" and not reduce_only and provider_key == "hyperliquid" and min_notional > 0 and notional + 1e-9 < min_notional:
            return self._reject(
                "hyperliquid_min_order_value",
                f"Hyperliquid requires minimum order value of ${min_notional:g}.",
                {"notional": notional, "min_notional": min_notional, "provider": provider_key},
            )
        if mode == "live" and is_one_h10 and not reduce_only:
            one_h10_quality_decision = self._evaluate_one_h10_forecast_quality(metadata)
            if not one_h10_quality_decision.approved:
                return one_h10_quality_decision
        if ml_policy_active:
            safety_envelope = self._safety_envelope(
                intent,
                reference_price=reference_price,
                notional=notional,
                has_trading_access=has_trading_access,
            )
            safety_envelope_payload = safety_envelope.as_dict()
            if not safety_envelope.ready:
                return self._reject(
                    "safety_envelope_blocked",
                    "Non-bypassable live safety envelope blocked the ML-governed order.",
                    safety_envelope_payload,
                )
            ml_policy_decision = self._evaluate_ml_policy_bundle(
                intent,
                reference_price=reference_price,
                notional=notional,
                safety_envelope=safety_envelope,
            )
            ml_policy_payload = dict(ml_policy_decision.details or {})
            if not ml_policy_decision.approved:
                return ml_policy_decision

        stop_loss = self._optional_positive_float(getattr(intent, "stop_loss", None))
        take_profit = self._optional_positive_float(getattr(intent, "take_profit", None))

        if not reduce_only:
            if stop_loss is None:
                return self._reject("stop_loss_required", "Opening trades must include a stop loss.")
            if ml_policy_active and take_profit is None:
                return self._reject(
                    "take_profit_required",
                    "ML-governed opening trades must include take profit.",
                    {"ml_policy_authority": self.config.get("ML_POLICY_LIVE_AUTHORITY", "guarded")},
                )

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
        adaptive_slippage = self.adaptive_slippage_metrics(metadata, slippage_pct=slippage_pct)
        adaptive_limit = self._safe_float(adaptive_slippage.get("max_acceptable_pct"), 0.0)

        if adaptive_limit > 0 and slippage_pct > adaptive_limit:
            return self._reject(
                "adaptive_slippage_cap",
                f"Projected slippage {slippage_pct:.4f} exceeds adaptive ML limit {adaptive_limit:.4f}.",
                {"slippage_pct": slippage_pct, "adaptive_slippage": adaptive_slippage},
            )

        loss_used = abs(min(0.0, self.daily_realized_pnl(mode, user_id=user_id, trading_connection_id=trading_connection_id)))
        daily_loss_limit = self._daily_loss_limit_usdc(mode, metadata)

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
                "adaptive_slippage": adaptive_slippage,
                "fibonacci_alignment": metadata.get("fibonacci_alignment", {}),
                "reward_risk": metadata.get("risk_reward"),
                "shadow_validation_id": shadow_validation_id,
                "pair_shadow_validation_id": metadata.get("pair_shadow_validation_id"),
                "safety_envelope": safety_envelope_payload,
                "ml_policy_decisions": ml_policy_payload,
                "ml_policy_authority": ml_policy_payload.get(
                    "ml_policy_authority",
                    self.config.get("ML_POLICY_LIVE_AUTHORITY", "guarded") if ml_policy_active else "disabled",
                ),
            },
        )

    def _ml_policy_active(self, metadata: dict[str, Any]) -> bool:
        payload = metadata or {}
        if bool(payload.get("ml_policy_required", False)) or bool(payload.get("ml_governed_risk", False)):
            return True
        return any(
            bool(self.config.get(key, False))
            for key in (
                "ML_RISK_POLICY_ENABLED",
                "ML_EXIT_POLICY_ENABLED",
                "ML_CAP_POLICY_ENABLED",
                "ML_ORDER_POLICY_ENABLED",
                "ML_ROI_TARGET_POLICY_ENABLED",
            )
        )

    def risk_controls(self) -> dict[str, Any]:
        """Return persisted admin risk controls with conservative defaults."""

        raw = Setting.get_json("risk_controls", {})
        payload = raw if isinstance(raw, dict) else {}
        default_pct = self._safe_float(payload.get("daily_loss_limit_pct"), 5.0)
        return {
            "daily_loss_limit_pct": max(0.0, min(default_pct, 100.0)),
            "daily_loss_unlimited": bool(payload.get("daily_loss_unlimited", False)),
            "max_leverage": max(0.0, self._safe_float(payload.get("max_leverage"), self._config_float("MAX_LEVERAGE", 1.0))),
            "profile": self._risk_profile(payload.get("profile")),
        }

    def save_risk_controls(self, payload: dict[str, Any]) -> dict[str, Any]:
        controls = {
            "daily_loss_limit_pct": max(0.0, min(self._safe_float(payload.get("daily_loss_limit_pct"), 5.0), 100.0)),
            "daily_loss_unlimited": bool(payload.get("daily_loss_unlimited", False)),
            "max_leverage": max(0.0, self._safe_float(payload.get("max_leverage"), self._config_float("MAX_LEVERAGE", 1.0))),
            "profile": self._risk_profile(payload.get("profile")),
        }
        Setting.set_json("risk_controls", controls)
        return controls

    def daily_loss_unlimited(self) -> bool:
        return bool(self.risk_controls().get("daily_loss_unlimited", False))

    def _daily_loss_limit_usdc(self, mode: str, metadata: dict[str, Any]) -> float:
        if self.daily_loss_unlimited():
            return 0.0
        raw_controls = Setting.get_json("risk_controls", {})
        controls_saved = isinstance(raw_controls, dict) and "daily_loss_limit_pct" in raw_controls
        controls = self.risk_controls()
        pct = self._safe_float(controls.get("daily_loss_limit_pct"), 0.0)
        capital = self._risk_capital_usdc(metadata)
        if controls_saved and pct > 0 and capital > 0:
            return capital * pct / 100.0
        if mode == "live" and self._is_high_upside_profile(metadata):
            return self._config_float("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0)
        return self._config_float("MAX_DAILY_LOSS_USDC", 0.0)

    def _ml_hard_daily_loss_limit_usdc(self, metadata: dict[str, Any]) -> float:
        if self.daily_loss_unlimited():
            return 0.0
        raw_controls = Setting.get_json("risk_controls", {})
        controls_saved = isinstance(raw_controls, dict) and "daily_loss_limit_pct" in raw_controls
        capital = self._risk_capital_usdc(metadata)
        pct = self._safe_float(self.risk_controls().get("daily_loss_limit_pct"), 0.0)
        if controls_saved and pct > 0 and capital > 0:
            return capital * pct / 100.0
        return self._config_float("ML_LIVE_HARD_DAILY_LOSS_USDC", 0.50)

    def _risk_capital_usdc(self, metadata: dict[str, Any]) -> float:
        payload = metadata or {}
        for key in (
            "account_equity_usd",
            "allocation_amount_usd",
            "allocation_cap_usd",
            "available_margin_usd",
            "user_input_amount_usd",
            "starting_value_usd",
            "capital_usd",
        ):
            value = self._safe_float(payload.get(key), 0.0)
            if value > 0:
                return value
        return 0.0

    def _exchange_max_leverage(self, metadata: dict[str, Any], symbol: str) -> float:
        payload = metadata or {}
        provider = normalize_provider(payload.get("provider") or payload.get("execution_venue"))
        venue_symbol = str(payload.get("venue_symbol") or payload.get("provider_symbol") or symbol or "").upper()
        symbol_key = str(symbol or "").upper()
        exchange_cap = 0.0
        query = LeveragedMarket.query.filter_by(status="active")
        if provider and provider != "global":
            query = query.filter_by(provider=provider)
        if symbol_key or venue_symbol:
            match = (
                query.filter((LeveragedMarket.symbol == symbol_key) | (LeveragedMarket.venue_symbol == venue_symbol))
                .order_by(LeveragedMarket.max_leverage.desc())
                .first()
            )
            if match is not None:
                exchange_cap = self._safe_float(match.max_leverage, 0.0)
        if exchange_cap <= 0:
            match = query.order_by(LeveragedMarket.max_leverage.desc()).first()
            if match is not None:
                exchange_cap = self._safe_float(match.max_leverage, 0.0)
        if exchange_cap <= 0:
            exchange_cap = self._config_float("MAX_LEVERAGE", 1.0)
        admin_cap = self._safe_float(self.risk_controls().get("max_leverage"), exchange_cap)
        if admin_cap <= 0:
            return 0.0
        return max(0.0, min(exchange_cap, admin_cap))

    def adaptive_slippage_metrics(self, metadata: dict[str, Any], *, slippage_pct: float | None = None) -> dict[str, Any]:
        payload = metadata or {}
        spread_bps = max(0.0, self._safe_float(payload.get("spread_bps"), self._safe_float(payload.get("observed_spread_bps"), 8.0)))
        liquidity_usd = max(0.0, self._safe_float(payload.get("liquidity_usd"), self._safe_float(payload.get("orderbook_depth_usd"), 0.0)))
        latency_ms = max(0.0, self._safe_float(payload.get("exchange_latency_ms"), self._safe_float(payload.get("latency_ms"), 0.0)))
        volatility_pct = max(0.0, self._safe_float(payload.get("volatility_pct"), self._safe_float(payload.get("recent_volatility_pct"), 0.0)))
        historical_pct = self._recent_reported_slippage_pct(payload)
        requested_pct = max(0.0, self._safe_float(slippage_pct, self._safe_float(payload.get("slippage_pct"), 0.0)))

        spread_component = max(spread_bps * 1.35, 2.0)
        latency_component = min(latency_ms / 55.0, 35.0)
        volatility_component = min(volatility_pct * 1_000.0, 45.0)
        history_component = historical_pct * 10_000.0 * 1.15
        estimate_bps = max(spread_component + latency_component + volatility_component, history_component, requested_pct * 10_000.0)
        if liquidity_usd > 0:
            estimate_bps *= 0.92 if liquidity_usd >= 1_000_000 else 1.08
        estimate_bps = max(2.0, min(estimate_bps, 500.0))
        max_acceptable_pct = min(max((estimate_bps * 1.35) / 10_000.0, 0.0002), 0.05)

        quality = 100.0
        quality -= min(spread_bps * 1.8, 34.0)
        quality -= min(latency_ms / 35.0, 24.0)
        quality -= min(volatility_pct * 350.0, 28.0)
        if liquidity_usd > 0:
            quality += min(liquidity_usd / 250_000.0, 12.0)
        market_quality = max(0.0, min(quality, 100.0))
        confidence = max(30.0, min(96.0, 55.0 + (12.0 if spread_bps > 0 else 0.0) + (12.0 if liquidity_usd > 0 else 0.0) + (10.0 if historical_pct > 0 else 0.0) - min(latency_ms / 120.0, 18.0)))
        execution_health = max(0.0, min((market_quality * 0.58) + (confidence * 0.42), 100.0))
        volatility_state = "Extreme" if volatility_pct >= 0.08 or spread_bps >= 45 else "Elevated" if volatility_pct >= 0.025 or spread_bps >= 18 else "Calm"

        return {
            "estimate_pct": estimate_bps / 10_000.0,
            "estimate_bps": estimate_bps,
            "max_acceptable_pct": max_acceptable_pct,
            "confidence": confidence,
            "market_quality": market_quality,
            "execution_health": execution_health,
            "spread_bps": spread_bps,
            "latency_ms": latency_ms,
            "liquidity_usd": liquidity_usd,
            "volatility_state": volatility_state,
            "model": "adaptive_ml",
        }

    def _safety_envelope(
        self,
        intent: Any,
        *,
        reference_price: float,
        notional: float,
        has_trading_access: bool,
    ) -> SafetyEnvelope:
        blockers: list[str] = []
        mode = str(getattr(intent, "mode", "")).lower()
        metadata = dict(getattr(intent, "metadata", {}) or {})
        is_one_h10 = self._is_one_h10(metadata)
        hard_cap = self._ml_live_hard_cap(metadata)
        hard_daily_loss = self._ml_hard_daily_loss_limit_usdc(metadata)
        unlimited_loss = self.daily_loss_unlimited()
        user_id = getattr(intent, "user_id", None)
        trading_connection_id = getattr(intent, "trading_connection_id", None)
        dynamic_budget, dynamic_budget_details = self._one_h10_dynamic_allocation_budget(
            metadata,
            leverage=self._safe_float(getattr(intent, "leverage", 1.0), 1.0),
        )

        if mode != "live":
            blockers.append("safety_mode_not_live")
        if not self._config_bool("ENABLE_LIVE_TRADING"):
            blockers.append("safety_live_trading_disabled")
        if Setting.get_json("panic_lock", False):
            blockers.append("safety_panic_lock")
        if not has_trading_access:
            blockers.append("safety_verified_connection_missing")
        if not bool(self.config.get("EXPLICIT_LIVE_CONFIRMED", False)) or not bool(Setting.get_json("explicit_live_confirmed", False)):
            blockers.append("safety_explicit_live_confirmation_missing")
        if not bool(self.config.get("SECONDARY_CONFIRMATION", False)) or not bool(Setting.get_json("secondary_confirmation", False)):
            blockers.append("safety_secondary_confirmation_missing")
        if reference_price <= 0:
            blockers.append("safety_reference_price_missing")
        if is_one_h10:
            if dynamic_budget <= 0:
                blockers.append("safety_one_h10_dynamic_cap_missing")
                blockers.append("safety_one_h10_allocation_budget_missing")
            elif notional > dynamic_budget + 1e-9:
                blockers.append("safety_one_h10_dynamic_cap_breached")
                blockers.append("safety_one_h10_allocation_budget_breached")
        elif hard_cap <= 0:
            if not bool(metadata.get("rapid_ml")):
                blockers.append("safety_ml_live_hard_cap_missing")
        elif notional > hard_cap + 1e-9:
            blockers.append("safety_ml_live_hard_cap_breached")
        loss_used = abs(min(0.0, self.daily_realized_pnl(mode, user_id=user_id, trading_connection_id=trading_connection_id)))
        if unlimited_loss:
            pass
        elif hard_daily_loss <= 0:
            blockers.append("safety_ml_live_hard_daily_loss_missing")
        elif loss_used >= hard_daily_loss:
            blockers.append("safety_ml_live_hard_daily_loss_reached")
        if str(self.config.get("ML_POLICY_LIVE_AUTHORITY", "guarded") or "guarded").lower() != "guarded":
            blockers.append("safety_ml_live_authority_not_guarded")

        return SafetyEnvelope(
            ready=not blockers,
            blockers=list(dict.fromkeys(blockers)),
            details={
                "mode": mode,
                "symbol": str(getattr(intent, "symbol", "")).upper(),
                "reference_price": reference_price,
                "notional": notional,
                "hard_cap_usdc": hard_cap,
                "one_h10_dynamic_cap_usd": dynamic_budget if is_one_h10 else None,
                "one_h10_dynamic_cap": dynamic_budget_details if is_one_h10 else {},
                "one_h10_allocation_budget_usd": dynamic_budget if is_one_h10 else None,
                "one_h10_allocation_budget": dynamic_budget_details if is_one_h10 else {},
                "hard_daily_loss_usdc": hard_daily_loss,
                "daily_loss_unlimited": unlimited_loss,
                "loss_used_usdc": loss_used,
                "policy_live_authority": self.config.get("ML_POLICY_LIVE_AUTHORITY", "guarded"),
                "ml_policy_required": bool(metadata.get("ml_policy_required", False)),
                "has_trading_access": bool(has_trading_access),
                "explicit_live_confirmed": bool(self.config.get("EXPLICIT_LIVE_CONFIRMED", False))
                and bool(Setting.get_json("explicit_live_confirmed", False)),
                "secondary_confirmation": bool(self.config.get("SECONDARY_CONFIRMATION", False))
                and bool(Setting.get_json("secondary_confirmation", False)),
            },
        )

    def _evaluate_one_h10_forecast_quality(self, metadata: dict[str, Any]) -> RiskDecision:
        forecast = metadata.get("one_h10_forecast") if isinstance(metadata.get("one_h10_forecast"), dict) else {}
        blockers = one_h10_forecast_live_blockers(forecast, self.config)
        if not blockers:
            return RiskDecision(approved=True, details={"one_h10_signal_quality": "passed"})
        return self._reject(
            "one_h10_signal_quality_blocked",
            "1H10 opening orders require a forecast that passes cost, confidence, liquidity, staleness, and risk/reward gates.",
            {
                "blockers": blockers,
                "decision_reason_code": first_one_h10_reason_code(blockers),
                "forecast": forecast,
            },
        )

    def _evaluate_ml_policy_bundle(
        self,
        intent: Any,
        *,
        reference_price: float,
        notional: float,
        safety_envelope: SafetyEnvelope,
    ) -> RiskDecision:
        metadata = dict(getattr(intent, "metadata", {}) or {})
        is_one_h10 = self._is_one_h10(metadata)
        leverage = self._safe_float(getattr(intent, "leverage", 1.0), 1.0)
        dynamic_budget, dynamic_budget_details = self._one_h10_dynamic_allocation_budget(metadata, leverage=leverage)
        ml_live_cap = dynamic_budget if is_one_h10 and dynamic_budget > 0 else self._ml_live_hard_cap(metadata)
        horizon = self._ml_policy_horizon(metadata)
        provider_key = normalize_provider(metadata.get("provider") or metadata.get("execution_venue"))
        context = {
            **metadata,
            "provider": provider_key,
            "execution_venue": provider_key,
            "symbol": str(getattr(intent, "symbol", "")).upper(),
            "side": str(getattr(intent, "side", "")).lower(),
            "horizon": horizon,
            "objective": metadata.get("objective") or metadata.get("ml_objective") or "risk_adjusted",
            "notional": notional,
            "reference_price": reference_price,
            "leverage": leverage,
            "stop_loss": self._safe_float(getattr(intent, "stop_loss", 0.0)),
            "take_profit": self._safe_float(getattr(intent, "take_profit", 0.0)),
            "stop_loss_pct": self._stop_loss_pct(intent, reference_price),
            "take_profit_pct": self._take_profit_pct(intent, reference_price),
            "ml_live_hard_cap_usdc": ml_live_cap,
            "one_h10_dynamic_cap_usd": dynamic_budget,
            "one_h10_dynamic_cap": dynamic_budget_details,
            "one_h10_allocation_budget_usd": dynamic_budget,
            "one_h10_allocation_budget": dynamic_budget_details,
            "ml_live_hard_daily_loss_usdc": self._ml_hard_daily_loss_limit_usdc(metadata),
            "hard_max_leverage": self._config_float("MAX_LEVERAGE", 1.0),
            "safety_envelope_ready": safety_envelope.ready,
        }
        engine = MLDecisionEngine(self.config, signal_model=MLSignalModel(self.config))
        if is_one_h10 and bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)):
            forecast = metadata.get("one_h10_forecast") if isinstance(metadata.get("one_h10_forecast"), dict) else {}
            forecast_blockers = [str(item) for item in (forecast.get("blockers", []) or []) if str(item)]
            if not bool(forecast.get("ml_ready", False)) or str(forecast.get("ml_horizon") or "").lower() != "1h10":
                return self._reject(
                    "one_h10_promoted_ml_required",
                    "1H10 opening live orders require a promoted 1h10 ML forecast.",
                    {"horizon": horizon, "forecast": forecast, "blockers": list(dict.fromkeys([*forecast_blockers, "ml_not_ready"]))},
                )
            required_feature_blockers = {"features_stale", "missing_fibonacci_features"}
            if required_feature_blockers.intersection(forecast_blockers):
                return self._reject(
                    "one_h10_required_features_missing",
                    "1H10 opening live orders require fresh higher-timeframe Fibonacci and indicator features.",
                    {"horizon": horizon, "forecast": forecast, "blockers": list(dict.fromkeys(forecast_blockers))},
                )
            fib_readiness = self._ml_family_readiness(engine, "pytorch_fibonacci", horizon, provider_key)
            if not bool(fib_readiness.get("ready", False)):
                return self._reject(
                    "one_h10_fibonacci_ml_not_ready",
                    "1H10 opening live orders require promoted pytorch_fibonacci readiness for horizon 1h10.",
                    {"horizon": horizon, "readiness": fib_readiness, "blockers": list(fib_readiness.get("blockers", []) or [])},
                )
        required_families = {
            **({"pytorch_fibonacci": "ML_FIBONACCI_MODEL_ENABLED"} if is_one_h10 and bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)) else {}),
            "pytorch_risk_policy": "ML_RISK_POLICY_ENABLED",
            "pytorch_exit_policy": "ML_EXIT_POLICY_ENABLED",
            "pytorch_cap_policy": "ML_CAP_POLICY_ENABLED",
            "pytorch_execution_policy": "ML_ORDER_POLICY_ENABLED",
            "pytorch_roi_target": "ML_ROI_TARGET_POLICY_ENABLED",
        }
        enabled_families = [
            family for family, flag in required_families.items() if bool(self.config.get(flag, False))
        ]
        if is_one_h10 and bool(self.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True)) and not enabled_families:
            return RiskDecision(
                approved=True,
                details={
                    "ml_policy_decisions": {
                        "one_h10_bootstrap_policy": {
                            "ready": True,
                            "action": "approve",
                            "horizon": horizon,
                            "reason": "one_h10_bootstrap_live_enabled",
                            "blockers": [],
                        }
                    },
                    "safety_envelope": safety_envelope.as_dict(),
                    "ml_policy_authority": "one_h10_bootstrap",
                    "enabled_families": enabled_families,
                    "notional": notional,
                    "horizon": horizon,
                    "one_h10_dynamic_cap_usd": dynamic_budget,
                    "one_h10_allocation_budget_usd": dynamic_budget,
                    "bootstrap_live": True,
                },
            )
        if is_one_h10 and "pytorch_risk_policy" not in enabled_families:
            return self._reject(
                "one_h10_ml_policy_not_enabled",
                "1H10 opening live orders require the custom 1h10 ML risk policy to be enabled.",
                {"horizon": horizon, "safety_envelope": safety_envelope.as_dict()},
            )
        if bool(metadata.get("ml_policy_required", False)) and not enabled_families:
            return self._reject(
                "ml_policy_not_enabled",
                "ML-governed risk was required by metadata, but no ML policy families are enabled.",
                {"safety_envelope": safety_envelope.as_dict()},
            )
        decisions: dict[str, Any] = {}
        blockers: list[str] = []
        for family in enabled_families:
            readiness = self._ml_family_readiness(engine, family, horizon, provider_key)
            if not bool(readiness.get("ready", False)):
                blockers.append(f"{family}_not_ready")
                decisions[family] = {"ready": False, "blockers": readiness.get("blockers", [])}
                continue
            decision = dict(engine.decision(family, context, horizon=horizon))
            decisions[family] = decision
            raw = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
            if family == "pytorch_risk_policy" and str(decision.get("action") or "").lower() != "approve":
                blockers.append("ml_risk_policy_rejected")
            if family == "pytorch_fibonacci":
                raw_blockers = raw.get("blockers") if isinstance(raw.get("blockers"), list) else []
                blockers.extend(str(item) for item in raw_blockers if str(item))
                if str(decision.get("action") or "").lower() not in {"suggest", "approve"}:
                    blockers.append("ml_fibonacci_policy_rejected")
            if family == "pytorch_exit_policy" and raw.get("blockers"):
                blockers.extend(str(item) for item in list(raw.get("blockers") or []))
            if family == "pytorch_cap_policy":
                suggested = self._safe_float(raw.get("suggested_notional_usdc"), 0.0)
                suggested_leverage = self._safe_float(raw.get("suggested_leverage"), 0.0)
                leverage_cap = self._config_float("MAX_LEVERAGE", 1.0)
                if is_one_h10:
                    leverage_cap = min(leverage_cap, self._config_float("ONE_H10_MAX_LEVERAGE", leverage_cap))
                if suggested_leverage > leverage_cap + 1e-9:
                    blockers.append("ml_cap_policy_leverage_cap_breach")
                if is_one_h10:
                    if dynamic_budget > 0 and suggested > dynamic_budget + 1e-9:
                        blockers.append("ml_cap_policy_dynamic_cap_breach")
                        blockers.append("ml_cap_policy_allocation_budget_breach")
                else:
                    hard_cap = self._ml_live_hard_cap(metadata)
                    if hard_cap > 0 and suggested > hard_cap + 1e-9:
                        blockers.append("ml_cap_policy_hard_cap_breach")

        details = {
            "ml_policy_decisions": decisions,
            "safety_envelope": safety_envelope.as_dict(),
            "ml_policy_authority": self.config.get("ML_POLICY_LIVE_AUTHORITY", "guarded"),
            "enabled_families": enabled_families,
            "notional": notional,
            "horizon": horizon,
            "one_h10_dynamic_cap_usd": dynamic_budget if is_one_h10 else None,
            "one_h10_allocation_budget_usd": dynamic_budget if is_one_h10 else None,
        }
        if blockers:
            return self._reject(
                "ml_policy_rejected",
                "Promoted ML policy did not approve the live risk intent.",
                {**details, "blockers": list(dict.fromkeys(blockers))},
            )
        return RiskDecision(approved=True, details=details)

    @staticmethod
    def _ml_family_readiness(engine: Any, family: str, horizon: str, provider: str) -> dict[str, Any]:
        try:
            return dict(engine.family_readiness(family, horizon, provider=provider))
        except TypeError as exc:
            if "provider" not in str(exc):
                raise
            return dict(engine.family_readiness(family, horizon))

    def _ml_policy_horizon(self, metadata: dict[str, Any]) -> str:
        payload = metadata or {}
        if self._is_one_h10(payload):
            return "1h10"
        explicit = str(payload.get("ml_horizon") or payload.get("horizon") or "").strip().lower()
        if explicit:
            return explicit
        return self._duration_bucket(payload.get("duration_hours") or payload.get("lock_duration_hours") or "1h")

    def _one_h10_dynamic_allocation_budget(self, metadata: dict[str, Any], *, leverage: float) -> tuple[float, dict[str, Any]]:
        payload = metadata or {}
        if not self._is_one_h10(payload):
            return 0.0, {}
        margin_sources: dict[str, float] = {}
        for key in (
            "allocation_cap_usd",
            "available_margin_usd",
            "provider_free_margin_usd",
            "free_margin_usd",
            "account_equity_usd",
            "user_input_amount_usd",
            "starting_value_usd",
        ):
            value = self._safe_float(payload.get(key), 0.0)
            if value > 0:
                margin_sources[key] = value
        effective_leverage = max(1.0, self._safe_float(leverage, 1.0))
        margin_cap = min(margin_sources.values()) if margin_sources else 0.0
        leveraged_allocation_budget = margin_cap * effective_leverage if margin_cap > 0 else 0.0

        capacity_sources: dict[str, float] = {}
        for key in (
            "liquidity_capacity",
            "liquidity_capacity_usd",
            "orderbook_capacity_usd",
            "exchange_capacity_usd",
            "max_order_capacity_usd",
        ):
            value = self._safe_float(payload.get(key), 0.0)
            if value > 0:
                capacity_sources[key] = value

        candidates = []
        if leveraged_allocation_budget > 0:
            candidates.append(leveraged_allocation_budget)
        candidates.extend(capacity_sources.values())
        cap = min(candidates) if candidates else 0.0
        return cap, {
            "budget_usd": cap,
            "margin_cap_usd": margin_cap,
            "leveraged_allocation_budget_usd": leveraged_allocation_budget,
            "leverage": effective_leverage,
            "margin_sources": margin_sources,
            "capacity_sources": capacity_sources,
        }

    def _evaluate_live_gate(self, intent: Any) -> RiskDecision:
        if Setting.get_json("live_trading_blocked", False):
            return self._reject("live_blocked", "Live trading is blocked after a prior failure; review and reset first.")
        if (
            bool(self.config.get("LIVE_BLOCK_ON_UNRECONCILED_FILLS", True))
            and not bool(getattr(intent, "reduce_only", False))
        ):
            unreconciled_count = self.unreconciled_live_fill_count(
                user_id=getattr(intent, "user_id", None),
                trading_connection_id=getattr(intent, "trading_connection_id", None),
            )
            if unreconciled_count > 0:
                return self._reject(
                    "unreconciled_live_fills",
                    "New live entries are blocked until recent closing fills have reconciled realized PnL.",
                    {"unreconciled_fill_count": unreconciled_count},
                )

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
    def _is_one_h10(metadata: dict[str, Any]) -> bool:
        payload = metadata or {}
        markers = {
            str(payload.get("algorithm_profile") or "").strip().lower(),
            str(payload.get("vault_cycle_name") or "").strip().lower(),
            str(payload.get("ml_horizon") or "").strip().lower(),
            str(payload.get("objective") or "").strip().lower(),
        }
        return bool(payload.get("one_h10_vault")) or bool(markers & {"1h10", "one_h10", "one_hour_10x"})

    @staticmethod
    def _is_high_upside_profile(metadata: dict[str, Any]) -> bool:
        if RiskEngine._is_one_h10(metadata):
            return False
        value = (metadata or {}).get("high_upside_profile", False)
        return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _is_rapid_ml_active_futures_market(self, metadata: dict[str, Any], symbol: str) -> bool:
        payload = metadata or {}
        if not bool(payload.get("rapid_ml")) or not bool(payload.get("rapid_ml_all_futures_universe")):
            return False
        provider = normalize_provider(payload.get("provider") or payload.get("execution_venue"))
        if provider not in {"hyperliquid", "kucoin"}:
            return False
        symbol_key = str(symbol or "").upper()
        venue_symbol = str(payload.get("venue_symbol") or payload.get("provider_symbol") or symbol_key).upper()
        market_id = self._safe_float(payload.get("futures_market_id"), 0.0)
        query = LeveragedMarket.query.filter_by(provider=provider, status="active")
        if market_id > 0:
            if query.filter_by(id=int(market_id)).first() is not None:
                return True
        return (
            query.filter((LeveragedMarket.symbol == symbol_key) | (LeveragedMarket.venue_symbol == venue_symbol)).first()
            is not None
        )

    def _ml_live_hard_cap(self, metadata: dict[str, Any]) -> float:
        payload = metadata or {}
        if bool(payload.get("rapid_ml")):
            metadata_cap = self._safe_float(payload.get("ml_live_hard_cap_usdc"), 0.0)
            if metadata_cap > 0:
                return metadata_cap
            rapid_cap = self._config_float("RAPID_ML_HARD_CAP_USDC", 0.0)
            if rapid_cap > 0:
                return rapid_cap
            return 0.0
        return self._config_float("ML_LIVE_HARD_CAP_USDC", 10.0)

    def _uses_experimental_live_caps(self, metadata: dict[str, Any]) -> bool:
        return (
            self._is_aggressive_experimental(metadata)
            or self._is_experimental_duration_ensemble(metadata)
            or self._is_max_return_live_eligible(metadata)
            or self._is_pair_live_eligible(metadata)
            or self._is_dynamic_intraday_live_eligible(metadata)
            or self._is_high_upside_profile(metadata)
        )

    def _evaluate_one_h10_gate(self, metadata: dict[str, Any]) -> RiskDecision:
        if not bool(self.config.get("ONE_H10_LIVE_ENABLED", False)):
            return self._reject("one_h10_live_disabled", "1H10 live orders require ONE_H10_LIVE_ENABLED=true.")
        if not bool(self.config.get("ML_DETERMINISTIC_SAFETY_GATES_REQUIRED", True)):
            return self._reject(
                "one_h10_safety_gates_disabled",
                "1H10 ML-driven live orders require deterministic safety gates to remain enabled.",
            )
        return RiskDecision(approved=True, details={"ml_horizon": metadata.get("ml_horizon"), "vault_cycle_name": "1H10"})

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

        return sum(self._fill_net_pnl(fill) for fill in fills)

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
            .filter(Order.mode == mode)
            .filter(Fill.simulated == (mode == "paper"))
        )
        if user_id is not None:
            query = query.filter(Order.user_id == int(user_id))
        if trading_connection_id is not None:
            query = query.filter(Order.trading_connection_id == int(trading_connection_id))
        fill = next((row for row in query.order_by(Fill.fill_time.desc()).all() if self._fill_net_pnl(row) < 0), None)

        if fill is None:
            return 0

        fill_time = self._as_aware_utc(fill.fill_time)
        remaining = timedelta(minutes=cooldown_minutes) - (now - fill_time)

        return max(0, int(remaining.total_seconds() // 60))

    def unreconciled_live_fill_count(
        self,
        *,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> int:
        lookback_hours = max(0.0, self._config_float("LIVE_UNRECONCILED_FILL_LOOKBACK_HOURS", 24.0))
        since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours or 24.0)
        query = (
            Fill.query.join(Fill.order)
            .filter(Order.mode == "live")
            .filter(Fill.simulated.is_(False))
            .filter(Fill.fill_time >= since)
            .filter(Fill.realized_pnl_known.is_(False))
        )
        if user_id is not None:
            query = query.filter(Order.user_id == int(user_id))
        if trading_connection_id is not None:
            query = query.filter(Order.trading_connection_id == int(trading_connection_id))
        return int(query.count())

    def _fill_net_pnl(self, fill: Fill) -> float:
        return self._safe_float(fill.pnl) - self._safe_float(fill.fee) - self._safe_float(getattr(fill, "funding_fee", 0.0))

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
            "daily_loss_limit": self._daily_loss_limit_usdc(mode, {}),
            "daily_loss_unlimited": self.daily_loss_unlimited(),
            "risk_controls": self.risk_controls(),
            "cooldown_minutes_remaining": self.cooldown_remaining(mode, user_id=user_id, trading_connection_id=trading_connection_id),
            "max_leverage": self._config_float("MAX_LEVERAGE", 1.0),
            "adaptive_slippage": self.adaptive_slippage_metrics({}),
            "high_upside": {
                "profile_enabled": bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)),
                "live_eligible": bool(self.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)),
                "auto_live_enabled": bool(self.config.get("HIGH_UPSIDE_AUTO_LIVE_ENABLED", False)),
                "auto_disabled": bool(Setting.get_json("high_upside_live_disabled", False)),
                "disabled_reason": Setting.get_json("high_upside_live_disabled_reason", {}),
                "max_daily_loss": self._daily_loss_limit_usdc(mode, {"high_upside_profile": True}),
                "requires_promoted_offline_ml": bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)),
                "offline_ml_readiness": OfflineRanker(self.config).readiness(self._duration_bucket("1h"), require_blend=False),
                "requires_ml_signal": bool(self.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True)),
                "ml_signal_readiness": MLSignalModel(self.config).readiness(
                    self._duration_bucket("1h"),
                    require_promoted=bool(self.config.get("ML_SIGNAL_REQUIRE_PROMOTED", True)),
                ),
                "ml_suite_readiness": MLDecisionEngine(
                    self.config,
                    signal_model=MLSignalModel(self.config),
                ).readiness(self._duration_bucket("1h")),
                "deterministic_safety_gates_required": bool(
                    self.config.get("ML_DETERMINISTIC_SAFETY_GATES_REQUIRED", True)
                ),
                "ml_policy_controls": {
                    "risk_policy_enabled": bool(self.config.get("ML_RISK_POLICY_ENABLED", False)),
                    "exit_policy_enabled": bool(self.config.get("ML_EXIT_POLICY_ENABLED", False)),
                    "cap_policy_enabled": bool(self.config.get("ML_CAP_POLICY_ENABLED", False)),
                    "order_policy_enabled": bool(self.config.get("ML_ORDER_POLICY_ENABLED", False)),
                    "roi_target_policy_enabled": bool(self.config.get("ML_ROI_TARGET_POLICY_ENABLED", False)),
                    "live_authority": self.config.get("ML_POLICY_LIVE_AUTHORITY", "guarded"),
                    "sandbox_bypass_enabled": bool(self.config.get("ML_POLICY_SANDBOX_BYPASS_ENABLED", True)),
                    "live_hard_cap_usdc": self._config_float("ML_LIVE_HARD_CAP_USDC", 10.0),
                    "live_hard_daily_loss_usdc": self._ml_hard_daily_loss_limit_usdc({}),
                    "target_roi_1h_pct": self._config_float("ML_TARGET_ROI_1H_PCT", 1000.0),
                    "target_roi_1w_pct": self._config_float("ML_TARGET_ROI_1W_PCT", 100.0),
                },
                "continuous_controls": {
                    "adaptive_cadence_enabled": bool(self.config.get("HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED", False)),
                    "ml_continuous_vault_enabled": bool(self.config.get("ML_CONTINUOUS_VAULT_ENABLED", False)),
                    "ml_vault_tick_enabled": bool(self.config.get("ML_VAULT_TICK_ENABLED", False)),
                    "ml_vault_provider_scope": self.config.get("ML_VAULT_PROVIDER_SCOPE", "all"),
                    "ml_vault_max_cap_usdc": self._config_float("ML_VAULT_MAX_CAP_USDC", 10.0),
                    "ml_vault_max_daily_loss_usdc": 0.0 if self.daily_loss_unlimited() else self._config_float("ML_VAULT_MAX_DAILY_LOSS_USDC", 0.50),
                    "ml_vault_leverage_policy": self.config.get("ML_VAULT_LEVERAGE_POLICY", "exchange_max_gated"),
                    "ml_vault_min_liquidation_buffer_pct": self._config_float("ML_VAULT_MIN_LIQUIDATION_BUFFER_PCT", 0.20),
                    "max_live_cycles_per_day": self._config_int("HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY", 1),
                    "daily_live_order_count": self._high_upside_daily_live_order_count(),
                    "max_active_cycles": self._config_int("HIGH_UPSIDE_MAX_ACTIVE_CYCLES", 1),
                    "active_live_order_count": self._high_upside_active_live_order_count(),
                    "loss_cooldown_seconds": self._config_float("HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS", 3600.0),
                    "rejection_cooldown_seconds": self._config_float("HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS", 900.0),
                    "rate_limit_backoff_seconds": self._config_float("HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS", 300.0),
                },
            },
        }

    def _evaluate_high_upside_gate(self, metadata: dict[str, Any]) -> RiskDecision:
        if not bool(self.config.get("ML_DETERMINISTIC_SAFETY_GATES_REQUIRED", True)):
            return self._reject(
                "deterministic_safety_gates_disabled",
                "ML-driven live orders require deterministic safety gates to remain enabled.",
            )
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
        if not bool(self.config.get("HIGH_UPSIDE_AUTO_LIVE_ENABLED", False)):
            return self._reject(
                "high_upside_auto_live_disabled",
                "High-upside live orders require HIGH_UPSIDE_AUTO_LIVE_ENABLED=true.",
            )
        if str(self.config.get("APP_MODE", "paper") or "paper").lower() != "live":
            return self._reject("high_upside_app_mode_not_live", "High-upside live orders require APP_MODE=live.")
        if not bool(self.config.get("EXPLICIT_LIVE_CONFIRMED", False)) or not bool(Setting.get_json("explicit_live_confirmed", False)):
            return self._reject(
                "high_upside_explicit_live_confirmation_missing",
                "High-upside live orders require config and DB explicit live confirmation.",
            )
        if not bool(self.config.get("SECONDARY_CONFIRMATION", False)) or not bool(Setting.get_json("secondary_confirmation", False)):
            return self._reject(
                "high_upside_secondary_confirmation_missing",
                "High-upside live orders require config and DB secondary confirmation.",
            )
        missing: list[str] = []
        if not self.daily_loss_unlimited() and self._config_float("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0) <= 0:
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
        ml_decision = self._evaluate_high_upside_ml_gate(
            duration_bucket,
            provider=metadata.get("provider") or metadata.get("execution_venue"),
        )
        if not ml_decision.approved:
            return ml_decision
        signal_decision = self._evaluate_high_upside_signal_gate(metadata, duration_bucket)
        if not signal_decision.approved:
            return signal_decision
        continuous_decision = self._evaluate_high_upside_continuous_limits(metadata)
        if not continuous_decision.approved:
            return continuous_decision
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

    def _evaluate_high_upside_continuous_limits(self, metadata: dict[str, Any]) -> RiskDecision:
        max_daily = self._config_int("HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY", 1)
        daily_count = self._high_upside_daily_live_order_count()
        if max_daily >= 0 and daily_count >= max_daily:
            return self._reject(
                "high_upside_daily_live_cycle_limit",
                "High-upside daily live cycle limit has been reached.",
                {"daily_count": daily_count, "max_daily_live_cycles": max_daily},
            )

        max_active = self._config_int("HIGH_UPSIDE_MAX_ACTIVE_CYCLES", 1)
        active_count = self._high_upside_active_live_order_count()
        if max_active >= 0 and active_count >= max_active:
            return self._reject(
                "high_upside_active_cycle_limit",
                "High-upside active live cycle limit has been reached.",
                {"active_count": active_count, "max_active_cycles": max_active},
            )

        rejection_cooldown = self._high_upside_recent_rejection_cooldown_seconds()
        if rejection_cooldown > 0:
            return self._reject(
                "high_upside_rejection_cooldown",
                "Recent high-upside rejection triggered a cooldown.",
                {"cooldown_remaining_seconds": rejection_cooldown},
            )

        loss_cooldown = self._high_upside_loss_cooldown_seconds()
        if loss_cooldown > 0:
            return self._reject(
                "high_upside_loss_cooldown",
                "Recent high-upside loss triggered a cooldown.",
                {"cooldown_remaining_seconds": loss_cooldown},
            )

        backoff_seconds = self._high_upside_rate_limit_backoff_remaining_seconds()
        if backoff_seconds > 0:
            return self._reject(
                "high_upside_provider_rate_limit_backoff",
                "High-upside provider rate-limit backoff is active.",
                {"backoff_remaining_seconds": backoff_seconds},
            )

        return RiskDecision(approved=True, details={"high_upside_continuous_limits": dict(metadata or {})})

    def _evaluate_high_upside_ml_gate(self, duration_bucket: str, *, provider: Any = "global") -> RiskDecision:
        if not bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)):
            return RiskDecision(approved=True)
        provider_key = normalize_provider(provider)
        readiness = OfflineRanker(self.config).readiness(duration_bucket, require_blend=False, provider=provider_key)
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
                {"duration_bucket": duration_bucket, "provider": provider_key, "blockers": blockers},
            )
        return self._reject(
            "high_upside_promoted_ml_required",
            "High-upside live orders require a promoted offline ML model that passes readiness checks.",
            {"duration_bucket": duration_bucket, "provider": provider_key, "offline_ml_readiness": readiness},
        )

    def _evaluate_high_upside_signal_gate(self, metadata: dict[str, Any], duration_bucket: str) -> RiskDecision:
        if not bool(self.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True)):
            return RiskDecision(approved=True)
        if not bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False)):
            return self._reject(
                "high_upside_ml_signal_disabled",
                "High-upside live orders require ML_SIGNAL_MODEL_ENABLED=true.",
            )
        readiness = MLSignalModel(self.config).readiness(
            duration_bucket,
            require_promoted=bool(self.config.get("ML_SIGNAL_REQUIRE_PROMOTED", True)),
        )
        if not bool(readiness.get("ready", False)):
            return self._reject(
                "high_upside_promoted_ml_signal_required",
                "High-upside live orders require a promoted ML signal model that passes readiness checks.",
                {"duration_bucket": duration_bucket, "ml_signal_readiness": readiness},
            )
        signal_payload = metadata.get("ml_signal_model") if isinstance(metadata.get("ml_signal_model"), dict) else {}
        if not signal_payload:
            return self._reject(
                "high_upside_ml_signal_missing",
                "High-upside live order metadata is missing the promoted ML signal decision.",
            )
        confidence = self._safe_float(signal_payload.get("confidence"))
        min_confidence = self._config_float("ML_SIGNAL_MIN_CONFIDENCE", 0.60)
        if confidence < min_confidence:
            return self._reject(
                "high_upside_ml_signal_low_confidence",
                "High-upside ML signal confidence is below the configured threshold.",
                {"confidence": confidence, "min_confidence": min_confidence},
            )
        if not bool(signal_payload.get("ready_for_live", False)):
            return self._reject(
                "high_upside_ml_signal_not_ready",
                "High-upside ML signal was not marked live-ready.",
                {"ml_signal_model": signal_payload},
            )
        if str(signal_payload.get("status") or "") != "promoted":
            return self._reject(
                "high_upside_ml_signal_not_promoted",
                "High-upside ML signal must come from a promoted model.",
                {"ml_signal_model": signal_payload},
            )
        if str(signal_payload.get("action") or "").lower() not in {"buy", "sell"}:
            return self._reject(
                "high_upside_ml_signal_hold",
                "High-upside ML signal did not select an actionable side.",
                {"ml_signal_model": signal_payload},
            )
        return RiskDecision(approved=True)

    def _high_upside_daily_live_order_count(self) -> int:
        since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        count = 0
        for order in Order.query.filter(Order.mode == "live", Order.created_at >= since).order_by(Order.created_at.desc()).limit(500).all():
            if self._high_upside_order(order) and str(order.status or "").lower() not in {"rejected", "failed", "cancelled"}:
                count += 1
        return count

    def _high_upside_active_live_order_count(self) -> int:
        active_statuses = {"open", "submitted", "pending", "partially_filled"}
        count = 0
        for order in Order.query.filter_by(mode="live").order_by(Order.created_at.desc()).limit(500).all():
            if self._high_upside_order(order) and str(order.status or "").lower() in active_statuses:
                count += 1
        return count

    def _high_upside_order(self, order: Order) -> bool:
        details = dict(order.details or {})
        return self._is_high_upside_profile(details) or str(details.get("optimizer_profile", "")).lower() == "aggressive_1h"

    def _high_upside_recent_rejection_cooldown_seconds(self) -> float:
        cooldown = self._config_float("HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS", 900.0)
        if cooldown <= 0:
            return 0.0
        since = datetime.now(timezone.utc) - timedelta(seconds=cooldown)
        for order in (
            Order.query.filter(Order.mode == "live", Order.created_at >= since, Order.status.in_(["rejected", "failed"]))
            .order_by(Order.created_at.desc())
            .limit(100)
            .all()
        ):
            if self._high_upside_order(order):
                elapsed = (datetime.now(timezone.utc) - self._as_aware_utc(order.created_at)).total_seconds()
                return max(0.0, cooldown - elapsed)
        return 0.0

    def _high_upside_loss_cooldown_seconds(self) -> float:
        cooldown = self._config_float("HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS", 3600.0)
        if cooldown <= 0:
            return 0.0
        since = datetime.now(timezone.utc) - timedelta(seconds=cooldown)
        rows = (
            db.session.query(Order, Fill)
            .join(Fill, Fill.order_id == Order.id)
            .filter(Order.mode == "live", Fill.fill_time >= since)
            .order_by(Fill.fill_time.desc())
            .limit(100)
            .all()
        )
        for order, fill in rows:
            if self._high_upside_order(order) and self._fill_net_pnl(fill) < 0:
                elapsed = (datetime.now(timezone.utc) - self._as_aware_utc(fill.fill_time)).total_seconds()
                return max(0.0, cooldown - elapsed)
        return 0.0

    def _high_upside_rate_limit_backoff_remaining_seconds(self) -> float:
        until = Setting.get_json("high_upside_rate_limited_until", None)
        if not until:
            return 0.0
        try:
            until_dt = datetime.fromisoformat(str(until).replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        now = datetime.now(until_dt.tzinfo or timezone.utc)
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (until_dt - now).total_seconds())

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

    def _stop_loss_pct(self, intent: Any, reference_price: float) -> float:
        stop_loss = self._optional_positive_float(getattr(intent, "stop_loss", None))
        if stop_loss is None or reference_price <= 0:
            return 0.0
        return abs(reference_price - stop_loss) / reference_price

    def _take_profit_pct(self, intent: Any, reference_price: float) -> float:
        take_profit = self._optional_positive_float(getattr(intent, "take_profit", None))
        if take_profit is None or reference_price <= 0:
            return 0.0
        return abs(take_profit - reference_price) / reference_price

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

    @staticmethod
    def _risk_profile(value: Any) -> str:
        profile = str(value or "balanced").strip().lower().replace("_", "-")
        if profile in {"conservative", "balanced", "aggressive", "maximum-performance"}:
            return profile
        if profile in {"maximum performance", "max", "maximum"}:
            return "maximum-performance"
        return "balanced"

    def _recent_reported_slippage_pct(self, metadata: dict[str, Any]) -> float:
        provider = normalize_provider((metadata or {}).get("provider") or (metadata or {}).get("execution_venue"))
        values: list[float] = []
        query = Order.query.filter(Order.mode == "live").order_by(Order.created_at.desc(), Order.id.desc()).limit(40)
        for order in query.all():
            details = order.details or {}
            if provider and provider != "global" and normalize_provider(details.get("provider") or details.get("execution_venue")) != provider:
                continue
            value = self._safe_float(
                details.get("reported_slippage"),
                self._safe_float(details.get("submitted_slippage_pct"), self._safe_float(details.get("slippage_pct"), 0.0)),
            )
            if value > 0:
                values.append(value)
            if len(values) >= 8:
                break
        if not values:
            return 0.0
        return sum(values) / len(values)

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
