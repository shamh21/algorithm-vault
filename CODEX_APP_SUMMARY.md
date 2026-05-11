# Codex App Summary

Last reviewed: 2026-05-01.

This repository is a local, live-only Flask application named **Algorithm Vault**. It combines a consumer crypto wallet/vault UI, admin trading controls, live exchange connections, strategy execution, backtesting, optimization, ML ranking, and safety/readiness tooling. The app is intentionally gated for real-money operation through 2FA, verified live trading connections, risk controls, wallet readiness, withdrawal approval, panic lock, daily loss limits, notional caps, slippage caps, leverage caps, stop-loss requirements, and explicit CLI confirmations.

Do not read, print, or summarize `.env` secret values. Use `.env.example` and `app/config.py` for config shape.

## Core Runtime

- Entry point: `run.py`
- App factory: `app/__init__.py:create_app`
- Config: `app/config.py:BaseConfig`
- Database extension: `app/extensions.py`
- Default database: SQLite via `DATABASE_URL`, defaulting to `sqlite:///hyperliquid_dashboard.db`
- Templates: `templates/`
- Static assets: `static/css/app.css`, `static/js/*.js`
- Tests: `tests/`

`create_app()` loads `.env`, configures Flask-SQLAlchemy, registers services into `app.extensions["services"]`, registers all blueprints, installs CSRF protection, creates/migrates the local schema, seeds default settings and the admin user, migrates legacy wallet rows, and resets stale strategy runs.

Runtime mode is live-only:

- `app/runtime.py:get_current_mode()` forces `current_mode` to `"live"`.
- `available_modes()` returns only `["live"]`.
- `market_mode_for()` maps everything to `"live"`.
- `OrderManager.place_order()` rejects non-live modes.

## Main User Surfaces

Authentication routes live in `app/routes/auth.py`.

- `/register`: user signup, optional invite code.
- `/login`: password plus TOTP when enabled.
- `/setup-2fa`: creates encrypted TOTP secret and QR code.
- `/logout`: clears session.

Consumer wallet/vault routes live in `app/routes/consumer.py`.

- `/`: consumer home with wallet balances, exchange snapshot, active vault cycle, recent activity.
- `/wallet/`: wallet balances, portfolio total, transaction history.
- `/wallet/deposit/<asset>`: deposit address display and QR code.
- `/wallet/rotate-address/<asset>`: burns old deposit address and creates/relinks a replacement.
- `/wallet/withdraw/<asset>`: validates address, balance, 2FA, panic lock, withdrawal caps, and approval requirements.
- `/vault/`: allocation UI and active/recent cycles.
- `/vault/start`: starts a risk-gated vault cycle, locks wallet funds, creates one or more `StrategyRun` rows, creates `VaultAllocationLeg` rows, and starts strategy threads.
- `/activity/`: transaction and cycle history.
- `/vault/cycles/<id>`: cycle detail, orders, leg summary, and performance explanation.
- Legacy consumer routes redirect to the admin dashboard, orders, backtests, and panic routes.

Settings and live connection routes live in `app/routes/settings.py`.

- `/settings/`: mode, panic state, live confirmations, address mode, external/pattern model status, and trading connections.
- `/settings/connections`: provider-specific connection management.
- Save, verify, activate, and delete trading connections.
- Activation is restricted to verified tradable connections and deactivates the user's other connections.
- Additional POST routes reset live blocks, reset panic lock, confirm live settings, and set address mode.

Admin routes live in `app/routes/admin.py`, `app/routes/dashboard.py`, `app/routes/orders.py`, `app/routes/panic.py`, and `app/routes/backtests.py`.

- `/admin/dashboard`: balances, positions, open orders, recent trades, PnL, risk status, strategy runs, rankings, feature snapshot, audits, risk events, market summary.
- `/admin/api/dashboard-data`: JSON dashboard payload.
- `/admin/strategies/start` and `/admin/strategies/<run_id>/stop`: manual strategy control.
- `/admin/orders/`: local order history.
- `/admin/orders/place`: manual order entry through `OrderManager` and `RiskEngine`.
- `/admin/orders/<order_id>/cancel`: cancel local/live order.
- `/admin/risk`: risk status, risk events, audit log.
- `/admin/live-readiness`: live trading and wallet readiness view.
- `/admin/strategies`: strategy runs, rankings, validations, shadow observations, vault cycles.
- `/admin/deposit-addresses`: generated/configured deposit address review.
- `/admin/wallet-withdrawals`: approve or reject gated withdrawals.
- `/admin/panic/`: panic lock view.
- `/admin/panic/activate`: sets panic lock, stops strategies, cancels orders, attempts to flatten positions, and records audit details.
- `/admin/backtests/`: backtest/optimizer UI, rankings, scanner diagnostics, high-upside/ML status.
- `/admin/backtests/run`: candle-based backtest.
- `/admin/backtests/optimize`: walk-forward optimization and optional auto-deploy of top rankings.

## Service Graph

The app factory constructs and registers these services under `app.extensions["services"]`:

- `strategy_registry`: built-in strategy catalog.
- `hyperliquid_client`: low-level Hyperliquid SDK/HTTP client.
- `execution_venue`: execution abstraction for Hyperliquid.
- `market_data`: normalized candles, mids, order books, cache stats.
- `realtime_market`: realtime/live market helper.
- `market_structure`: execution-quality and market-structure features.
- `market_universe`: dynamic market universe discovery.
- `market_scanner`: candidate scoring and opportunity scanner.
- `pair_screening`: pair-trading candidate analysis.
- `feature_engine`: indicators, Fibonacci, multi-timeframe, pattern, and external feature snapshots.
- `risk_engine`: hard pre-trade risk rules.
- `trading_connections`: user-scoped encrypted credential storage and connector selection.
- `online_ranker`: online learning/ranking from strategy outcomes.
- `offline_ranker`: offline model training/scoring.
- `order_manager`: live order submission, cancellation, protective exits, local order records.
- `backtest_engine`: candle-driven simulation.
- `strategy_optimizer`: walk-forward optimizer and ranking writer.
- `vault_strategy_selector`: chooses strategies/legs for consumer vault allocations.
- `wallet_address_service`: deposit address generation/pool selection.
- `wallet_custody`: real custody adapters and deposit/withdrawal readiness.
- `self_custody_wallet`: manual withdrawal workflow, approval/rejection/submission.
- `strategy_manager`: in-process background strategy runner.

Use `app.runtime.get_service("name")` inside request/app contexts.

## Trading Providers

Provider metadata and encrypted connection handling live in `app/services/trading_connections.py`.

Supported provider keys:

- `hyperliquid`: API wallet/agent secret plus account address. Uses `HyperliquidClient`.
- `binance`: USD-M futures API key/secret. Uses `BinanceFuturesConnector`.
- `kucoin`: futures API key/secret/passphrase. Uses `KucoinFuturesConnector`.
- `uniswap`: wallet delegation metadata and Uniswap Trading API. Uses `UniswapDelegatedConnector`.
- `dydx`: permissioned trading key metadata. Uses `DydxV4Connector`.

The service rejects seed-phrase-shaped secrets, encrypts API credentials with Fernet, verifies connections, stores verification status/errors, and returns user-scoped connectors for account snapshots and live trading.

## Order and Risk Flow

Order input is normalized by `app/services/order_manager.py:OrderIntent`.

Order placement flow:

1. Reject non-live mode.
2. Enforce idempotency by `client_order_id`.
3. Get market price.
4. Check connection access through `TradingConnectionService`.
5. Evaluate `RiskEngine`.
6. Create a local `Order`.
7. If approved, submit to the active provider connector.
8. Store exchange result, fills, position snapshots, audit entries, or failure details.

`RiskEngine` in `app/services/risk_engine.py` is the central pre-trade gate. It checks:

- Live-only mode.
- Panic lock.
- `ENABLE_LIVE_TRADING`.
- Verified trading access.
- Allowed symbols.
- Positive size and valid leverage.
- Max leverage.
- Max notional.
- Stop-loss requirement for opening trades.
- Stop-loss validity.
- Experimental gates for high-upside, max-return, dynamic intraday, pair trading, and shadow validation.
- Projected slippage cap.
- Daily loss cap.
- Loss cooldown.
- Reward/risk rules for rule-based signals.
- Offline/promoted ML readiness where configured.

Opening trades must have a stop loss. Panic lock blocks trading until reset.

## Strategy Execution

Strategies implement `BaseStrategy.generate_signal()` and return `Signal` from `app/strategies/base.py`.

Supported actions:

- `buy`
- `sell`
- `hold`
- `reduce`

Supported timeframes:

- `1m`
- `5m`
- `15m`
- `1h`

Built-in strategies are registered in `app/strategies/registry.py`:

- `ema_crossover`
- `mean_reversion`
- `breakout`
- `rsi_mean_reversion`
- `rule_based`
- `volatility_breakout`
- `scalping`

`StrategyManager` in `app/services/strategy_runner.py` runs strategy loops in daemon threads. Each loop loads candles, builds the strategy, generates a signal, records heartbeat/signal state, handles vault live-gate shadow observations, computes edge/signal quality metadata, sizes positions from vault/risk parameters, and sends `OrderIntent` to `OrderManager`. It stops when `manual_enabled` is false or panic lock is active.

## Vault Flow

Consumer vault cycles are started through `consumer.start_cycle()`.

Key models:

- `WalletBalance`: available and locked balance per asset.
- `VaultCycle`: user allocation, duration, selected strategy, live validation status, starting/current value, final settlement.
- `VaultAllocationLeg`: per-symbol leg for a cycle, linked to strategy run and optimizer ranking.
- `WalletTransaction`: append-style wallet activity.

Start-cycle flow:

1. Require authenticated 2FA user.
2. Require active verified trading connection when live connection gating is enabled.
3. Validate deposit/settlement asset, amount, lock duration, balance, reserve, price, and connection health.
4. Ask `VaultStrategySelector` for strategy/leg selection.
5. Apply cycle-start blockers such as reserve, exposure, selection quality, and live readiness.
6. Move allocation from available to locked wallet balance.
7. Create `VaultCycle`, one or more `StrategyRun` records, and `VaultAllocationLeg` records.
8. Add wallet allocation transaction.
9. Start strategy manager threads for each run.

Completed cycles are synced by consumer helpers, with performance estimates refreshed from linked orders and legs.

## Wallet and Custody

Wallet routes support `USDC`, `USDT`, `BTC`, `ETH`, `SOL`, and `XRP`.

Wallet/custody services:

- `app/services/wallet_addresses.py`: configured address pools, generated addresses, active address linking, validation helpers.
- `app/services/wallet_custody.py`: real wallet custody, generated wallets, chain adapters for EVM, Bitcoin, Solana, and XRP Ledger, deposit sync/readiness, withdrawal release handling.
- `app/services/self_custody_wallet.py`: manual withdrawal workflow and provider bridge withdrawal.

Withdrawal flow requires:

- Authenticated user with valid 2FA session.
- Asset/network support.
- Enabled withdrawals config.
- Panic lock not active.
- Valid destination address.
- Sufficient available balance.
- Valid TOTP code.
- Configured per-asset withdrawal cap.
- Admin approval when `WALLET_REQUIRE_WITHDRAWAL_APPROVAL=true`.

Real custody mode can lock withdrawal funds until approval/submission or release them on rejection/failure.

## Backtesting, Optimization, and ML

Backtesting:

- `app/backtesting/engine.py:BacktestEngine`
- `BacktestConfig` controls symbol, timeframe, initial balance, fees, slippage, sizing, drawdown, daily loss, cooldown, trade windows, intrabar model, leverage, and funding cost.
- Results are stored in `BacktestRun`.

Optimization:

- `app/backtesting/optimizer.py:StrategyOptimizer`
- Runs walk-forward strategy/profile sweeps, stores `OptimizerRun` and `StrategyRanking`, supports dynamic universe, high-upside/aggressive profiles, ensemble allocation, pair screening, market structure filters, net ROI diagnostics, and live-readiness gates.

ML:

- `app/ml/online_ranker.py`: online ranker with horizon features and outcome learning.
- `app/ml/offline_ranker.py`: offline training and promoted model readiness.
- Offline models can affect rankings only when enabled and promoted by config/CLI guardrails.

Net ROI and edge-quality helpers live in `app/services/net_roi.py`.

## CLI Commands

CLI commands are registered in `app/cli.py`.

Primary operator commands:

- `flask run-backtest`
- `flask run-optimization`
- `flask find-live-canary-ranking`
- `flask live-auto-canary`
- `flask live-canary-trade`
- `flask live-funds-readiness`
- `flask wallet-readiness`
- `flask production-readiness --strict`
- `flask activate-trading-connection --user-id <id> --connection-id <id> --confirm ACTIVATE-LIVE-CONNECTION`
- `flask reset-local-state --confirm FULL-LIVE-RESET`
- `flask live-only-clean-slate --confirm ...`
- `flask recover-evm-token-deposit`
- `flask repair-limited-cycle`
- `flask promote-live-ranker`
- `flask train-offline-ranker --confirm TRAIN-OFFLINE-RANKER`
- `flask promote-offline-ranker --confirm PROMOTE-OFFLINE-RANKER`

Canary submission is intentionally guarded:

- `CANARY_PREVIEW_ONLY=true` is the safe default.
- Preview commands write audit records but do not submit real orders.
- Real submission requires `CANARY_PREVIEW_ONLY=false`, an accepted ranking, strict readiness, verified active connection, risk approval, and exact confirmation text `LIVE-CANARY-TRADE`.

## Data Model Overview

Models live in `app/models.py`.

Auth/settings:

- `Setting`
- `User`

Strategy/vault/wallet:

- `StrategyRun`
- `WalletBalance`
- `VaultCycle`
- `VaultAllocationLeg`
- `WalletTransaction`
- `WalletLedgerEvent`
- `TradingConnection`
- `DepositAddress`
- `WalletAccount`
- `WalletAddress`
- `WalletWithdrawal`
- `WalletAuditLog`

Trading:

- `Order`
- `Fill`
- `PositionSnapshot`

Research/ML:

- `BacktestRun`
- `OptimizerRun`
- `StrategyRanking`
- `MLModelState`
- `MLOfflineModel`
- `MLTrainingEvent`
- `StrategyValidation`
- `ShadowLiveObservation`

Audit/risk:

- `AuditLog`
- `RiskEvent`

Many model fields store structured metadata as JSON strings with helper properties.

## Configuration Shape

Config is environment-driven through `app/config.py` and `.env.example`.

Important categories:

- Flask/runtime: `FLASK_SECRET_KEY`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `TOTP_ENCRYPTION_KEY`, `DATABASE_URL`
- Live posture: `ENABLE_LIVE_TRADING`, `APP_MODE`, `EXPLICIT_LIVE_CONFIRMED`, `SECONDARY_CONFIRMATION`
- Provider credentials/endpoints: Hyperliquid, Binance, KuCoin, dYdX, Uniswap
- Market universe/scanning: dynamic universe, hot-token scan, market-structure features
- Risk: daily loss, position notional, leverage, slippage, risk per trade, reserve
- Optimizer and ML gates: high-upside, aggressive 1h, offline/online ML, canary preview, first-canary caps
- Wallet/custody: real address mode, EVM/BTC/Solana/XRP RPC/indexer URLs, token contracts, confirmations, withdrawal caps

Use `.env.example` for names and defaults. Do not expose real `.env` values.

## Security and Safety Notes

- User sessions are cookie-backed Flask sessions.
- Passwords use Werkzeug password hashing.
- TOTP secrets are encrypted using `TOTP_ENCRYPTION_KEY`; if invalid/missing, code falls back to a key derived from `SECRET_KEY`.
- CSRF protection is custom and enforced for unsafe HTTP methods unless disabled in tests.
- Admin access requires authenticated user, enabled 2FA, and `role == "admin"`.
- Trading credentials are encrypted at rest with Fernet.
- Seed phrases are rejected in trading connection input.
- Live order paths should fail closed on missing credentials, failed provider verification, bad market data, missing stop loss, panic lock, or risk violation.
- Withdrawals can require admin approval and should respect panic lock plus per-asset caps.

## Development Workflow

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Run:

```bash
flask run
```

Test:

```bash
python -m pytest -q
```

Readiness:

```bash
flask wallet-readiness
flask production-readiness --strict
```

## Test Coverage Map

The test suite includes coverage for:

- Live-only runtime behavior.
- UI smoke paths.
- Wallet custody and trading connections.
- Market structure and pair screening.
- Feature extraction and strategy improvements.
- Live safeguards.
- Optimizer behavior.
- Backtesting engine.
- Multi-cycle wallet realtime behavior.
- Production readiness.
- Vault duration rankings and UI copy.
- Online ranker.
- Database retry behavior.
- v11 integration scenarios.

Main test config is in `tests/conftest.py`, which builds the app with an in-memory SQLite database and live-only test settings.

## Guidance for Future Codex Work

- Check `git status --short` before editing; this repo may already contain user changes.
- Do not revert unrelated dirty files.
- Avoid reading or printing `.env`.
- Prefer app-factory tests using `create_app({...})`.
- Use `get_service()` rather than reconstructing services manually inside app contexts.
- Keep changes live-only unless the user explicitly asks to reintroduce paper/testnet paths.
- Preserve fail-closed behavior for live trading, wallet custody, and withdrawals.
- Any new order path must go through `OrderManager` and `RiskEngine`.
- Any new strategy should implement `BaseStrategy`, return `Signal`, and be registered in `StrategyRegistry`.
- Any new provider should implement the `TradingConnector` protocol and be wired through `TradingConnectionService`.
- Any route that writes data needs CSRF coverage through templates or headers.
- Use `commit_with_retry()` or `write_with_retry()` for SQLite write paths that may run concurrently with background threads.
- Add focused tests for changes that affect risk gates, live execution, wallet balances, connection verification, optimizer ranking, or strategy threading.
