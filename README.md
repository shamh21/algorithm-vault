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

Run high-upside research diagnostics:

```bash
flask run-optimization --profile aggressive_1h --high-upside-profile
```

Optimizer research commands are bounded by `OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS` so slow market-data providers return JSON diagnostics instead of appearing hung. The command does not create live orders.

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

Train and promote offline ranker candidates only after reviewing guardrails:

```bash
flask train-offline-ranker --horizon 1h --model both --confirm TRAIN-OFFLINE-RANKER
flask promote-offline-ranker --horizon 1h --model-id <id> --confirm PROMOTE-OFFLINE-RANKER
```

Offline ML models affect rankings only when `ML_OFFLINE_MODELS_ENABLED=true`, a model has been promoted, and `ML_OFFLINE_BLEND_ENABLED=true`. High-upside live orders fail closed when `HIGH_UPSIDE_REQUIRE_PROMOTED_ML=true` and no promoted model passes readiness.

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
