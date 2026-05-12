"""Shared market tradability calculations for dynamic candidate scoring."""

from __future__ import annotations

from statistics import mean
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def level_price_size(row: Any) -> tuple[float, float]:
    if isinstance(row, dict):
        return safe_float(row.get("px", row.get("price"))), safe_float(row.get("sz", row.get("size")))
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return safe_float(row[0]), safe_float(row[1])
    return 0.0, 0.0


def best_bid_ask(book: dict[str, Any]) -> tuple[float, float]:
    levels = book.get("levels", []) if isinstance(book, dict) else []
    try:
        return level_price_size(levels[0][0])[0], level_price_size(levels[1][0])[0]
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def spread_bps(book: dict[str, Any], mid: float = 0.0) -> float:
    bid, ask = best_bid_ask(book)
    if bid <= 0 or ask <= 0 or ask < bid:
        return 0.0
    resolved_mid = mid if mid > 0 else (bid + ask) / 2
    return ((ask - bid) / max(resolved_mid, 1e-9)) * 10_000


def book_liquidity_usd(book: dict[str, Any], *, depth: int) -> float:
    levels = book.get("levels", []) if isinstance(book, dict) else []
    if not isinstance(levels, list):
        return 0.0
    total = 0.0
    resolved_depth = max(1, int(depth or 1))
    for side in levels[:2]:
        if not isinstance(side, list):
            continue
        for row in side[:resolved_depth]:
            price, size = level_price_size(row)
            total += price * size
    return total


def volatility_pct(candles: list[dict[str, Any]], *, lookback: int | None = None) -> float:
    closes = [safe_float(row.get("close")) for row in candles if isinstance(row, dict)]
    closes = [close for close in closes if close > 0]
    if len(closes) < 3:
        return 0.0
    returns = [
        abs((closes[index] - closes[index - 1]) / closes[index - 1])
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    if lookback is not None:
        returns = returns[-max(1, int(lookback)) :]
    return mean(returns) * 100 if returns else 0.0


def volatility_regime(volatility: float) -> tuple[str, float]:
    if volatility <= 0:
        return "unknown", 0.0
    if volatility < 0.08:
        return "compressed", 0.45
    if volatility <= 1.5:
        return "tradable", 1.0
    if volatility <= 3.5:
        return "elevated", 0.65
    return "dislocated", 0.20


def cost_drag_bps(*, spread: float, fee_bps: float, slippage_bps: float) -> float:
    return max(0.0, float(spread or 0.0) + float(fee_bps or 0.0) * 2 + float(slippage_bps or 0.0))


def market_structure_score(
    *,
    liquidity_usd: float,
    spread: float,
    volatility_score: float,
    min_liquidity_usd: float,
    max_spread_bps: float,
) -> float:
    liquidity_score = min(float(liquidity_usd or 0.0) / max(float(min_liquidity_usd or 0.0) * 4, 1.0), 1.0)
    spread_score = max(0.0, 1.0 - float(spread or 0.0) / max(float(max_spread_bps or 0.0), 1.0))
    return max(0.0, min(liquidity_score * 0.45 + spread_score * 0.30 + float(volatility_score or 0.0) * 0.25, 1.0))
