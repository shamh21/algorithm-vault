# 1H10 Validation Notes

## Signal Flow

1H10 uses persisted leveraged-market features and provider market data to rank candidates, then builds a `1h10` forecast through `OneH10ForecastService`. The selected forecast is attached to vault legs and strategy runs, applied again in the live strategy loop, and finally checked by `RiskEngine` before any opening live order is allowed.

## Decision Gates

Executable 1H10 signals now use shared quality gates for expected edge after costs, cost drag, confidence, risk/reward, liquidity, execution quality, stale data, and ML readiness. The gate preserves existing forecast blockers and adds structured reason codes such as `BELOW_EDGE_THRESHOLD`, `LOW_CONFIDENCE`, `POOR_RISK_REWARD`, `HIGH_SLIPPAGE`, `LOW_LIQUIDITY`, `STALE_MARKET_DATA`, `PROVIDER_DEGRADED`, and `RISK_ENGINE_BLOCKED`.

New configurable defaults:

- `ONE_H10_MIN_EDGE_AFTER_COST_BPS=4`
- `ONE_H10_MAX_COST_DRAG_BPS=18`
- `ONE_H10_MIN_RISK_REWARD=1.0`
- `ONE_H10_MAX_SIGNAL_AGE_SECONDS=4200`
- `ONE_H10_MIN_EXECUTION_QUALITY=0.60`

Existing live flags and risk controls remain authoritative. The optimized decision path does not auto-enable live trading.

## Evaluation

Run a minimal cost-aware report with:

```bash
flask one-h10-evaluation-report --symbol BTC --timeframe 1m --strategy scalping
```

The report uses existing backtest utilities and configured fees/slippage. It includes total return, fee/slippage-adjusted return, drawdown, Sharpe-like and Sortino-like metrics, win rate, profit factor, trade counts, long/short counts, exposure minutes, config used, data period, and small-sample warnings.

Backtest improvements are validation evidence only and do not guarantee future live performance.
