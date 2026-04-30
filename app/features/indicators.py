"""Pure deterministic technical indicator helpers."""

from __future__ import annotations

from statistics import pstdev
from typing import Any


def ema_series(values: list[float], period: int) -> list[float]:
    """Return an EMA series for the supplied values."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]

    if not clean:
        return []

    period = max(_safe_int(period, 1), 1)
    multiplier = 2 / (period + 1)

    result = [clean[0]]

    for value in clean[1:]:
        result.append((value - result[-1]) * multiplier + result[-1])

    return result


def ema(values: list[float], period: int) -> float:
    """Return the latest EMA value."""

    result = ema_series(values, period)
    return result[-1] if result else 0.0


def sma(values: list[float], period: int) -> float:
    """Return the latest simple moving average value."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]
    period = max(_safe_int(period, 1), 1)

    if not clean:
        return 0.0

    window = clean[-period:]
    return sum(window) / len(window) if window else 0.0


def rsi(values: list[float], period: int) -> float:
    """Return simple RSI for the latest close."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]

    period = max(_safe_int(period, 1), 1)

    if len(clean) < period + 1:
        return 50.0

    gains = 0.0
    losses = 0.0

    for index in range(-period, 0):
        change = clean[index] - clean[index - 1]

        if change > 0:
            gains += change
        elif change < 0:
            losses += abs(change)

    if gains == 0 and losses == 0:
        return 50.0

    if losses == 0:
        return 100.0

    if gains == 0:
        return 0.0

    relative_strength = gains / losses
    return 100 - (100 / (1 + relative_strength))


def volatility(values: list[float], period: int) -> float:
    """Return close-to-close return volatility over the latest window."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]
    period = max(_safe_int(period, 2), 2)

    if len(clean) < 3:
        return 0.0

    window = clean[-(period + 1) :]
    returns = [
        (window[index] - window[index - 1]) / window[index - 1]
        for index in range(1, len(window))
        if window[index - 1] > 0
    ]

    if len(returns) < 2:
        return 0.0

    return pstdev(returns)


def macd(
    values: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, float]:
    """Return latest MACD line, signal line, and histogram."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]
    if not clean:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    fast_period = max(_safe_int(fast_period, 12), 1)
    slow_period = max(_safe_int(slow_period, 26), fast_period + 1)
    signal_period = max(_safe_int(signal_period, 9), 1)
    fast = ema_series(clean, fast_period)
    slow = ema_series(clean, slow_period)
    length = min(len(fast), len(slow))
    if length <= 0:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    macd_values = [fast[-length + index] - slow[-length + index] for index in range(length)]
    signal_values = ema_series(macd_values, signal_period)
    macd_line = macd_values[-1] if macd_values else 0.0
    signal_line = signal_values[-1] if signal_values else 0.0
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": macd_line - signal_line,
    }


def bollinger_bands(values: list[float], period: int = 20, stddev_multiplier: float = 2.0) -> dict[str, float]:
    """Return latest Bollinger Band values and normalized bandwidth."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]
    period = max(_safe_int(period, 20), 2)
    multiplier = max(_safe_float(stddev_multiplier, 2.0), 0.0)
    if not clean:
        return {"middle": 0.0, "upper": 0.0, "lower": 0.0, "bandwidth": 0.0, "percent_b": 0.5}

    window = clean[-period:]
    middle = sum(window) / len(window)
    deviation = pstdev(window) if len(window) > 1 else 0.0
    upper = middle + deviation * multiplier
    lower = middle - deviation * multiplier
    latest = clean[-1]
    width = upper - lower
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "bandwidth": width / middle if middle > 0 else 0.0,
        "percent_b": (latest - lower) / width if width > 0 else 0.5,
    }


def trend_strength(values: list[float], fast_period: int, slow_period: int) -> float:
    """Return EMA trend spread as a fraction of the latest close."""

    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value > 0]

    if len(clean) < 2:
        return 0.0

    fast = ema(clean, fast_period)
    slow = ema(clean, slow_period)
    latest = clean[-1]

    if latest <= 0:
        return 0.0

    return (fast - slow) / latest


def atr(candles: list[dict[str, Any]], period: int) -> float:
    """Return average true range for OHLC candles."""

    period = max(_safe_int(period, 1), 1)
    clean = [_normalize_candle(candle) for candle in candles]
    clean = [candle for candle in clean if candle is not None]

    if len(clean) < 2:
        return 0.0

    true_ranges: list[float] = []
    start = max(1, len(clean) - period)

    for index in range(start, len(clean)):
        candle = clean[index]
        previous_close = clean[index - 1]["close"]

        high = candle["high"]
        low = candle["low"]

        true_ranges.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0


def volume_spike(
    candles: list[dict[str, Any]],
    lookback: int,
    multiplier: float,
) -> dict[str, float | bool]:
    """Return deterministic volume spike status and ratio."""

    lookback = max(_safe_int(lookback, 1), 1)
    multiplier = max(_safe_float(multiplier, 1.0), 0.0)

    if len(candles) < 2:
        return _empty_volume_spike()

    current = max(_safe_float(candles[-1].get("volume")), 0.0)

    history = [
        max(_safe_float(row.get("volume")), 0.0)
        for row in candles[-lookback - 1 : -1]
    ]

    history = [value for value in history if value > 0]

    average = sum(history) / len(history) if history else 0.0
    ratio = current / average if average > 0 else 0.0

    return {
        "is_spike": bool(average > 0 and ratio >= multiplier),
        "ratio": ratio,
        "current": current,
        "average": average,
    }


def _normalize_candle(candle: dict[str, Any]) -> dict[str, float] | None:
    close = _safe_float(candle.get("close"))
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))

    if close <= 0:
        return None

    if high <= 0:
        high = close

    if low <= 0:
        low = close

    if low > high:
        low, high = high, low

    return {
        "high": high,
        "low": low,
        "close": close,
    }


def _empty_volume_spike() -> dict[str, float | bool]:
    return {
        "is_spike": False,
        "ratio": 0.0,
        "current": 0.0,
        "average": 0.0,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
