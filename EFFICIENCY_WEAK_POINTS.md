# TradingBot Efficiency Weak-Point Report

## What Was Fixed

- `strategy/no_trade` audit bloat was caused by writing a full diagnostic JSON blob on repeated skipped strategy ticks. `StrategyManager` now writes compact no-trade details and throttles identical run/reason/category rows by default.
- Stale encrypted exchange credentials now surface as direct credential/decryption failures instead of being recorded as generic transient provider backoff.
- VPS readiness now treats `postgresql+psycopg://...` as the valid production database target and keeps local SQLite validation for local operation.
- SQLite repair/backfill logic is SQLite-only, so Postgres deployments do not run PRAGMA/ALTER repair paths meant for local databases.
- Orders now have explicit `vault_cycle_id` and `vault_leg_id` columns, with legacy JSON fallback kept for older records.

## Query And Storage Improvements

- Added composite indexes for strategy rankings, optimizer runs, orders, audit logs, wallet transactions, market-history windows, and live-cycle lookups.
- Vault order lookup paths now filter on indexed `Order.vault_cycle_id` / `Order.vault_leg_id` before falling back to legacy `Order.details` JSON.
- The protected cleanup command backs up SQLite first, dry-runs by default, requires `--protect-username sufyanh`, and only prunes noisy historical strategy diagnostics.

## Runtime Limits Still Present

- Strategy loops are still in-process daemon threads. A multi-worker Gunicorn deployment can duplicate those loops, so the first VPS default should stay at one Gunicorn worker with multiple threads.
- Postgres improves database concurrency, but it does not by itself coordinate strategy execution across processes. A larger refactor should move strategy runners into a dedicated worker with process-level locking or queue ownership.

## Larger Refactor Candidates

- `app/cli.py` is very large and owns many unrelated operational workflows; split it by command family before adding more production commands.
- Consumer routes are dense and mix wallet, vault, order, and presentation concerns; extract service/query helpers for easier profiling and regression tests.
- Strategy runner complexity remains high. The next performance pass should isolate market-data fetch, signal scoring, auditing, live gating, and order-intent creation into smaller units with clearer ownership and independent timing metrics.
