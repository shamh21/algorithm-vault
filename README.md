# Algorithm Vault Local Live Runbook

Algorithm Vault is configured as a local, live-only Flask application. Live execution is still gated by 2FA, a verified active trading connection, explicit risk controls, wallet readiness, withdrawal approval, panic lock, daily loss limits, notional caps, slippage caps, leverage caps, and stop-loss requirements.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill real local values. Do not commit `.env`.
4. Use a real Fernet key for `TOTP_ENCRYPTION_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Verification

Run the full suite:

```bash
python -m pytest -q
```

Check wallet readiness:

```bash
flask wallet-readiness
```

Check full local production readiness:

```bash
flask production-readiness --strict
```

Strict readiness exits nonzero until required live secrets, RPC/indexer URLs, token contracts, withdrawal caps, and safety settings are configured.

Check the local `sufyanh` profile and in-app wallet without touching exchange balances or submitting orders:

```bash
flask profile-wallet-check --username sufyanh
```

This command is read-only. It reports the local app wallet balance source, locked funds, order count, verified trading connections, cached exchange snapshot metadata, and reconciliation warnings such as duplicate completed settlement transactions. Normal wallet and dashboard pages render from local/cached data first; use `/wallet?refresh_exchange=1` or `/admin/dashboard?refresh_exchange=1` only when you want an explicit read-only provider snapshot refresh.

Run high-upside research diagnostics:

```bash
flask run-optimization --profile aggressive_1h --high-upside-profile
```

Optimizer research commands are bounded by `OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS` so slow market-data providers return JSON diagnostics instead of appearing hung. The command does not create live orders.

Discover hot multi-provider high-upside vault candidates without submitting orders:

```bash
flask discover-high-upside-vault-candidates --provider-scope multi_provider
```

The discovery command is research-only. It uses configured provider connections only for public-market discovery where supported, skips providers without a market-data adapter, scores hot/liquid candidates with scanner, Fibonacci, pair, and ML diagnostics, then runs bounded backtests before persisting accepted `StrategyRanking` rows. Live high-upside execution remains blocked unless `HIGH_UPSIDE_PROFILE_ENABLED=true`, `HIGH_UPSIDE_LIVE_ELIGIBLE=true`, `HIGH_UPSIDE_AUTO_LIVE_ENABLED=true`, promoted ML/readiness gates pass, and `RiskEngine` approves the final order intent.

Preview an operator-approved live canary from a promoted ranking without submitting:

```bash
flask live-canary-trade --ranking-id <id> --user-id <id>
```

`CANARY_PREVIEW_ONLY=true` is the safe default. While it is enabled, even an exact `--submit` command is blocked after running the full signal, sizing, net ROI v2, readiness, and risk evaluation. The command writes a preview audit record with `preview_only=true` and `real_order_submitted=false`.

Real canary submission is a later operator action only after strict readiness passes, a verified active live connection exists, all risk gates approve, and `CANARY_PREVIEW_ONLY=false` is set locally:

```bash
flask live-canary-trade --ranking-id <id> --user-id <id> --submit --confirm LIVE-CANARY-TRADE
```

The canary command computes size from existing live caps, allocation/risk config, stop distance, and current price; it does not accept arbitrary oversized quantity input.

## Exchange Egress IP

KuCoin and other IP-restricted exchange API keys must whitelist the machine's current public egress IP before any live vault cycle can start. For reliable production use, run the app behind a stable VPS, VPN, or static egress IP and update the exchange whitelist before activating live cycles. If a provider returns `Invalid request ip`, the app records the latest connection health and blocks new cycles before locking funds.

## KuCoin And Hyperliquid Live Validation

KuCoin futures order size is contract count, not the app's asset quantity. Keep `KUCOIN_SYMBOL_MAP_JSON` and `KUCOIN_CONTRACT_SPECS_JSON` configured for every KuCoin symbol you allow; missing or misaligned contract metadata fails closed before any order is submitted. The safe validation order is mocked tests first, then `flask production-readiness --strict`, then read-only connection verification/account snapshot. Only use `flask live-canary-trade ... --submit --confirm LIVE-CANARY-TRADE` after that staged validation and after lowering live caps to match the small test balance.

Hyperliquid remains routed through the official Python SDK and requires an API wallet/agent private key, not a recovery phrase or main wallet seed. Rejected SDK order responses are stored on the local order and audit details so the operator can see the provider-side reason.

For the first real funds test, keep the command path CLI-only:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
flask production-readiness --strict
flask live-funds-readiness --provider active --user-id <id> --ranking-id <id>
flask live-canary-trade --ranking-id <id> --user-id <id> --connection-id <connection-id>
```

To let the app search for a genuinely accepted ranking and preview both first-canary providers in one operator command:

```bash
flask live-auto-canary --user-id <id> --provider hyperliquid --research-budget-minutes 30
```

`live-auto-canary` uses live market data for live canary research and ML only to prioritize accepted-ranking sweeps; it does not clear `rejected=true`, remove `rejection_reason`, or submit without the existing exact confirmation path. When `CANARY_PREVIEW_ONLY=false` has been set locally and the preview output is clean, the same command can submit one guarded Hyperliquid canary:

```bash
CANARY_PREVIEW_ONLY=false flask live-auto-canary --user-id <id> --provider hyperliquid --research-budget-minutes 30 --submit --confirm LIVE-CANARY-TRADE
```

For rapid execution, first preview multiple accepted rankings without submitting. The default remains one submission; raising `--max-submissions` never bypasses accepted-ranking checks, preview readiness, live micro-canary daily order caps, or post-submit verification:

```bash
flask live-auto-canary --user-id <id> --provider hyperliquid --research-budget-minutes 30 --max-submissions 3
CANARY_PREVIEW_ONLY=false flask live-auto-canary --user-id <id> --provider hyperliquid --research-budget-minutes 30 --max-submissions 3 --submit --confirm LIVE-CANARY-TRADE
```

The separate rapid ML trader is ML-first and uses a user-provided total capital cap across verified Hyperliquid and KuCoin connections. By default it refreshes the active futures/perp universe from provider metadata instead of requiring `ALLOWED_SYMBOLS`, then ranks the discovered markets with promoted signal, allocator, risk, execution, exit, cap, ops-anomaly, ROI-target, and offline ranker models. It is preview-only by default and still refuses orders when provider health, stale credentials, open-order/position reconciliation, exchange minimum notional/contract sizing, stop-loss/take-profit, daily loss, slippage, or correlation guards fail:

```bash
flask ml-quality-report --horizon 1h --provider both --model-family all --candidate-limit 2 --compact
flask live-rapid-ml-trader --user-id <id> --capital-usd <amount> --provider both
RAPID_ML_LIVE_ENABLED=true RAPID_ML_PREVIEW_ONLY=false CANARY_PREVIEW_ONLY=false flask live-rapid-ml-trader --user-id <id> --capital-usd <amount> --provider both --submit --confirm RAPID-ML-LIVE
```

`ml-quality-report --compact` is read-only and shows blocked families, promoted/candidate model IDs, ready candidate IDs, and exact retrain/promote commands for each provider/family. Omit `--compact` when you need full metrics. `RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED` defaults to `true`; `RAPID_ML_MAX_SYMBOLS_PER_PROVIDER` defaults to `48` to keep live API usage bounded while selecting from the discovered futures universe. `RAPID_ML_ML_SIZING_ENABLED` defaults to `true`, so order notional is derived from promoted model confidence, edge, allocator, risk, ROI, and cap-policy signals; `RAPID_ML_HARD_CAP_USDC` defaults to `0`, meaning no fixed rapid notional cap is applied unless you explicitly set one. Sizing is still constrained by the user-provided capital, available balance, exchange minimum notional, daily loss, slippage, stop/take requirements, and open-position reconciliation. `RAPID_ML_MAX_DAILY_LOSS_PCT` defaults to `0.05` and is clamped at `0.10`; it cannot be disabled. `RAPID_ML_DECISION_INTERVAL_MS` defaults to `1000`, and `RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER` defaults to one opening order per provider per second. The loop does not force an order every interval; it submits only when promoted ML expected edge clears the conservative profitability gate: round-trip provider fees (`RAPID_ML_FEE_BPS_BY_PROVIDER_JSON`), expected slippage (`RAPID_ML_SLIPPAGE_BPS`), live spread or `RAPID_ML_UNKNOWN_SPREAD_BPS`, `RAPID_ML_COST_RESERVE_BPS`, `RAPID_ML_MIN_CONFIDENCE`, `RAPID_ML_MIN_EDGE_AGREEMENT`, and `RAPID_ML_MIN_EDGE_BPS`. Rapid scoring builds leakage-safe live candle features using `RAPID_ML_FEATURE_TIMEFRAME` and `RAPID_ML_FEATURE_CANDLE_LIMIT`; missing or stale features block submission. Provider circuit breakers use `RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES` so stale transient failures do not permanently block a now-healthy venue. Offline ranker scoring also respects `ML_OFFLINE_SAFE_SCORING_MODEL_TYPES`; keep it at `sklearn` unless XGBoost scoring has been separately validated in this runtime.

Activate KuCoin only after the Hyperliquid order, audit log, provider order status, balances, open orders, and positions are understood. Activating one verified connection deactivates the user's other trading connections:

```bash
flask activate-trading-connection --user-id <id> --connection-id <kucoin-connection-id> --confirm ACTIVATE-LIVE-CONNECTION
flask live-auto-canary --user-id <id> --provider kucoin --research-budget-minutes 30
```

The first canary cap defaults to `FIRST_CANARY_MAX_NOTIONAL_USDT=1`. If the venue minimum order size blocks the preferred cap, set `FIRST_CANARY_USE_MIN_SIZE_FALLBACK=true` only long enough to use `FIRST_CANARY_FALLBACK_MAX_NOTIONAL_USDT=2`. Keep `CANARY_PREVIEW_ONLY=true` until the readiness and preview output are reviewed. To submit exactly one canary:

```bash
CANARY_PREVIEW_ONLY=false flask live-canary-trade --ranking-id <id> --user-id <id> --connection-id <connection-id> --submit --confirm LIVE-CANARY-TRADE
```

After the canary attempt, restore `CANARY_PREVIEW_ONLY=true`, then review the order, audit log, provider open orders, and positions before doing anything else.

Train and promote offline ranker candidates only after reviewing guardrails:

```bash
flask train-offline-ranker --horizon 1h --model both --confirm TRAIN-OFFLINE-RANKER
flask promote-offline-ranker --horizon 1h --model-id <id> --confirm PROMOTE-OFFLINE-RANKER
```

Offline ML models affect rankings only when `ML_OFFLINE_MODELS_ENABLED=true`, a model has been promoted, and `ML_OFFLINE_BLEND_ENABLED=true`. High-upside live orders fail closed when `HIGH_UPSIDE_REQUIRE_PROMOTED_ML=true` and no promoted model passes readiness.

Train and promote the high-upside ML signal model separately. It is fail-closed by default and requires PyTorch plus a promoted `pytorch_gru` artifact before high-upside live signals can become actionable:

```bash
flask train-ml-signal-model --horizon 1h --model pytorch_gru --confirm TRAIN-ML-SIGNAL-MODEL
flask promote-ml-signal-model --horizon 1h --model-id <id> --confirm PROMOTE-ML-SIGNAL-MODEL
```

ML signal output may choose `buy`, `sell`, or `hold`, suggest stop-loss/take-profit percentages, and suggest a sizing score. Signal training is class-balanced by default and caps market-history rows with `ML_SIGNAL_MAX_TRAINING_ROWS=15000` and `ML_SIGNAL_TRAINING_EPOCHS=16` so large backfills do not stall operator workflows. Promotion also rejects degenerate signal models whose confidence-filtered validation `action_rate` is below `ML_SIGNAL_MIN_ACTION_RATE` or above `ML_SIGNAL_MAX_ACTION_RATE`, requires confidence-filtered `action_precision` above `ML_SIGNAL_MIN_ACTION_PRECISION`, uses `ML_SIGNAL_MAX_CLASSIFICATION_LOSS=1.10` for the 3-class classifier loss, and rejects fully abstaining candidates that do not beat the validation majority-class baseline. Use `flask sweep-ml-signal-model --provider both --confirm SWEEP-ML-SIGNAL-MODEL` to train bounded signal candidates across label/confidence overlays without promoting or submitting orders. It still cannot bypass vault caps, stop-loss/take-profit requirements, `OrderManager`, or `RiskEngine`.

The app-wide ML decision layer exposes a shared envelope for signal, Fibonacci, backtest-scorer, optimizer-policy, allocator, universe, ops-anomaly, risk-policy, exit-policy, cap-policy, execution-policy, and ROI-target model families. It is disabled by default and remains fail-closed unless models are explicitly enabled, trained, and promoted:

```bash
flask ml-readiness --horizon 1h
flask ml-shadow-evaluate --horizon 1h --provider hyperliquid
flask train-ml-suite --horizon 1h --model-family all --confirm TRAIN-ML-SUITE
flask train-ml-suite --horizon 1h --model-family all --objective extreme_roi_1h --use-market-history --confirm TRAIN-ML-SUITE
flask train-ml-suite --horizon 1w --model-family all --objective consistent_roi_1w --use-market-history --confirm TRAIN-ML-SUITE
flask ml-feedback-sync --horizon all --source all --confirm SYNC-ML-FEEDBACK
flask promote-ml-suite --horizon 1h --model-id <id> --confirm PROMOTE-ML-SUITE
```

`ML_ALL_AREAS_ENABLED=false` is the default. When the policy families are explicitly enabled, promoted ML can govern risk approve/reject decisions, exits, caps, leverage suggestions, slippage/execution style, and ROI target diagnostics. Live policy remains inside a non-bypassable SafetyEnvelope: live mode, live enablement, confirmations, panic lock, verified connection, auditability, provider constraints, and hard capital/loss ceilings must pass before any live order can continue through `StrategyRunner`, `OrderManager`, and `RiskEngine`.

Use `ml-risk-preview` to inspect the ML-governed policy layer for an accepted ranking without submitting an order:

```bash
flask ml-risk-preview --ranking-id <id> --horizon 1h --user-id 1
flask ml-risk-preview --ranking-id <id> --horizon 1w --user-id 1
```

`ml-feedback-sync` is research-only and disabled by default with `ML_FEEDBACK_SYNC_ENABLED=false`. When explicitly enabled, it converts historical rankings, backtests, orders, vault cycles, ops/risk events, and stored market history into deduplicated `MLTrainingEvent` rows so the app-wide suite and online learner can train from every major process without submitting orders.

`train-ml-suite` trains every PyTorch family from existing app data. The signal model can now consume stored market-history windows when `--use-market-history` is passed, using only candle data available at each cutoff before the target return. `pytorch_fibonacci` learns from Fibonacci-tagged rankings and backtest trades, `pytorch_backtest_scorer` from stored backtest results, `pytorch_optimizer_policy` from prior optimizer rankings, `pytorch_universe` from optimizer rankings and ML training events, `pytorch_allocator` from completed vault allocation legs, and `pytorch_ops_anomaly` from audit/risk/connection-health records. These artifacts reuse the `MLOfflineModel` table and must be promoted before live-facing advisory use.

The manual ML live-vault workflow is preview-first and capped for the 10 USDC test profile. The preview command runs strict readiness, funds readiness, promoted ML readiness, bounded high-upside discovery, vault selection, and a RiskEngine dry-run without starting a cycle:

```bash
flask ml-live-vault-preview --user-id 1 --provider active --cap-usdc 10 --horizon 1h
```

Only a fresh ready preview may be used by the one-shot command, and the command still fails unless `ML_LIVE_VAULT_ONE_SHOT_ENABLED=true`, live confirmations, high-upside caps, promoted ML, panic lock, connection, balance, and RiskEngine gates all pass. It starts one normal vault cycle through `StrategyManager`; it does not call `OrderManager` directly:

```bash
flask ml-live-vault-one-shot --preview-audit-id <id> --user-id 1 --confirm ML-LIVE-VAULT-10USDC
```

The extreme-upside objective is available as a research and preview target. `ML_EXTREME_UPSIDE_TARGET_ROI_PCT=1000` is an aspirational scoring target for 1-hour asymmetric upside; it is not a promise and does not weaken profit-factor, drawdown, trade-count, liquidity, spread, stop-loss, take-profit, cap, confirmation, or RiskEngine gates:

```bash
flask discover-high-upside-vault-candidates --objective extreme_upside --provider-scope hyperliquid --timeframe 1h --max-sweeps 1
flask ml-live-vault-preview --user-id 1 --provider active --cap-usdc 10 --horizon 1h --objective extreme_upside
```

`ML_DYNAMIC_CAPS_ENABLED=false`, `ML_EXTREME_UPSIDE_MODEL_ENABLED=false`, and `ML_AUTO_VAULT_LIVE_ENABLED=false` are the defaults. When explicitly enabled with promoted models, ML may suggest notional, leverage, risk, Fibonacci zones, hold duration, and cadence, but those suggestions are clipped by operator hard caps and `RiskEngine`. The auto-vault command still starts at most one normal vault cycle through `StrategyManager`; it never calls `OrderManager` directly:

```bash
flask ml-auto-vault-cycle --user-id 1 --provider active --cap-usdc 10 --horizon 1h --objective extreme_upside --confirm ML-AUTO-VAULT-LIVE
```

Continuous ML vault operation is implemented as a one-tick command for launchd or cron, not a long-running daemon. It is disabled by default, records `ml_vault_last_tick_at`, `ml_vault_last_decision`, provider backoff, and live-cycle timestamps in settings, and can start at most one 10 USDC cycle only through the existing strategy manager path:

```bash
flask ml-history-backfill --provider-scope all --timeframe 5m --timeframe 15m --timeframe 1h --max-symbols 50 --lookback-days 90 --confirm BACKFILL-ML-HISTORY
flask train-ml-suite --horizon 1h --model-family all --objective extreme_upside --use-market-history --confirm TRAIN-ML-SUITE
flask train-ml-signal-model --horizon 1h --objective extreme_upside --use-market-history --confirm TRAIN-ML-SIGNAL-MODEL
flask ml-vault-tick --user-id 1 --provider-scope all --cap-usdc 10 --objective extreme_upside --confirm ML-VAULT-TICK
```

`ML_VAULT_LEVERAGE_POLICY=exchange_max_gated` means ML may suggest leverage only up to a verified provider/exchange maximum and only when liquidation buffer, balance, configured caps, and `RiskEngine` all pass. If exchange maximum leverage or liquidation-buffer data is unavailable, the live tick blocks.

## Clean Slate

When tests and strict readiness pass, reset local state for a new account:

```bash
flask reset-local-state --confirm FULL-LIVE-RESET
```

The reset command backs up the SQLite database first and prints the backup path. After reset, register a new user, set up 2FA, connect and verify an exchange account, then activate it.

## Local Run

```bash
flask run
```

Use the browser to smoke-test registration, 2FA setup, settings/connections, wallet deposit readiness, admin live readiness, and admin withdrawal approval. Do not place orders or approve withdrawals unless real credentials and operator intent are confirmed.
