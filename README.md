# Algorithm Vault Local Live Runbook

Algorithm Vault is configured as a local, live-only Flask application. Live execution is still gated by 2FA, a verified active trading connection, explicit risk controls, wallet readiness, withdrawal approval, panic lock, daily loss limits, adaptive ML slippage, exchange leverage limits, allocation budgets, and stop-loss requirements.

## Developer Quick Start

Algorithm Vault is a Python Flask/Jinja app with static PWA assets. It uses `pip` and `requirements*.txt`; there is no `package.json`, Capacitor, Expo, CocoaPods, or native iOS project in this repo.

Prerequisites:

- Python 3.10 or newer
- `pip`, `venv`, and `curl`
- Optional for local production preview: a browser with PWA/service-worker support

Create a local environment and install the fast default runtime:

```bash
python3 -m venv --copies .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install -r requirements.txt
```

Create a local `.env` and replace the placeholder secrets:

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))" # FLASK_SECRET_KEY
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" # TOTP_ENCRYPTION_KEY
```

Validate the setup:

```bash
scripts/check_local_setup.sh
```

Run the web app in Flask development mode:

```bash
scripts/run_local_dev.sh
```

Open `http://127.0.0.1:5000`. Override the bind without editing files:

```bash
PORT=5001 scripts/run_local_dev.sh
HOST=0.0.0.0 PORT=5000 scripts/run_local_dev.sh
```

Preview the production-style web/PWA stack locally:

```bash
scripts/run_local_production.sh
GUNICORN_BIND=127.0.0.1:8765 scripts/healthcheck.sh
```

Open `http://127.0.0.1:8765`. The PWA manifest and service worker are served from `/manifest.json` and `static/js/sw.js`.

For desktop-only HTTPS diagnostics, generate a local-only certificate and run the separate HTTPS Flask server:

```bash
scripts/create_local_https_cert.sh
scripts/run_local_https.sh
```

Open `https://127.0.0.1:5443` on the desktop. The HTTPS runner binds to `0.0.0.0`, keeps the normal HTTP dev command unchanged, enables secure cookies for that HTTPS origin, and reads cert paths from `LOCAL_HTTPS_CERT` and `LOCAL_HTTPS_KEY` or ignored `.env.local-https`. Generated local certs and keys stay under `.local-certs/` and are ignored by git.

Do not use self-signed certificates or private LAN IP HTTPS URLs for iPhone PWA install testing. iPhone Safari and installed PWAs must use `https://app.algvault.com` or a stable public HTTPS tunnel hostname with a valid trusted certificate.

## iPhone PWA HTTPS setup

Do not install AlgVault from `https://172.20.10.6` or any private LAN IP HTTPS URL. Safari shows certificate warnings before app code loads, so JavaScript, CSS, service workers, splash screens, and PWA code cannot hide or fix that warning.

Use one of these stable HTTPS origins:

- Production: `https://app.algvault.com`
- Development fallback: a static ngrok HTTPS domain or Cloudflare Tunnel hostname
- Hosted fallback: a Netlify, Vercel, or Cloudflare Pages HTTPS deployment URL

For local tunnel testing, run the Flask app locally and expose it through a stable public HTTPS hostname:

```bash
HOST=0.0.0.0 PORT=5000 scripts/run_local_dev.sh
ngrok http --domain=YOUR_STATIC_NGROK_DOMAIN 5000
```

Or:

```bash
cloudflared tunnel --url http://localhost:5000
```

Then reinstall the iPhone PWA from the trusted URL:

1. Delete the old AlgVault PWA icon from the iPhone Home Screen.
2. Clear Safari website data for the old `172.20.10.6` origin if needed.
3. Open `https://app.algvault.com` or the stable tunnel hostname in Safari.
4. Confirm Safari does not show "This connection is not private."
5. Use Safari Share > Add to Home Screen.
6. Launch AlgVault from the Home Screen.
7. Confirm no network requests go to `172.20.10.6`.
8. Confirm service worker scope is `/`.
9. Confirm manifest `start_url` is `/`.

Build and test checks:

```bash
python -m flask --app wsgi:app routes
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Troubleshooting:

- Missing env vars: run `scripts/check_local_setup.sh --strict`, then update `.env`.
- Wrong Python: use Python 3.10+ and recreate `.venv` if needed.
- Port already in use: change `PORT` or stop the existing listener.
- PWA looks stale: hard refresh or clear the browser service worker/cache for the local origin.
- Package mismatch: this repo uses `pip` requirements files, not npm/yarn/pnpm.
- Production readiness fails: `flask production-readiness --strict` is intentionally stricter than local dev and requires live secrets, RPC/indexer URLs, withdrawal caps, and safety settings.

## Setup

1. Create and activate a Python environment.
2. Install the fast default runtime:

```bash
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install -r requirements.txt
```

If an existing `.venv` reports a missing `.venv/bin/python3`, leave the active shell first and recreate it cleanly:

```bash
deactivate 2>/dev/null || true
mv .venv ".venv.broken-$(date +%Y%m%d%H%M%S)"
python3 -m venv --copies .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install -r requirements.txt
```

Optional dependency groups are split out so local startup is not blocked by heavyweight SDKs:

```bash
python -m pip install -r requirements-ml.txt        # sklearn/XGBoost ranker workflows
python -m pip install -r requirements-torch.txt     # PyTorch model families
python -m pip install -r requirements-wallets.txt   # BTC/Solana/XRPL wallet signing
python -m pip install -r requirements-full.txt      # non-conflicting local extras
python -m pip install -r requirements-exchanges.txt # dYdX SDK; use separately from wallet extras if pip reports httpx conflicts
```

3. Copy `.env.example` to `.env` and fill real local values. Do not commit `.env`.
4. Use a real Fernet key for `TOTP_ENCRYPTION_KEY`:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Verification

Run the full suite:

```bash
python -m pip install -r requirements-dev.txt
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

## Optimized Local Production Run

Use this on macOS or any local workstation when you want the production WSGI stack without Ubuntu Nginx/systemd:

```bash
scripts/run_local_production.sh
```

Defaults keep local startup convenient: one Gunicorn worker, four threads, SQLite WAL enabled, a persistent local database at `instance/algorithm_vault_local_production.db`, local in-process workers enabled, and bind `127.0.0.1:8765`. Override without editing files:

```bash
PORT=8000 scripts/run_local_production.sh
HOST=0.0.0.0 PORT=8765 scripts/run_local_production.sh
DATABASE_URL=postgresql+psycopg://tradingbot:<password>@127.0.0.1:5432/tradingbot scripts/run_local_production.sh
```

The local production runner intentionally keeps `SESSION_COOKIE_SECURE=false` because it previews the WSGI stack over plain HTTP. Use `scripts/run_local_https.sh` only for desktop HTTPS diagnostics. Use `https://app.algvault.com` or a stable public HTTPS tunnel for iPhone PWA testing, and use the VPS/Nginx path for production HTTPS cookies.

Healthcheck the same bind from another terminal:

```bash
GUNICORN_BIND=127.0.0.1:8765 scripts/healthcheck.sh
```

If the default port is already serving a healthy app, the runner exits cleanly and prints the URL instead of retrying Gunicorn. Stop local Gunicorn processes with:

```bash
scripts/stop_local_production.sh
```

The Nginx and systemd commands below are for an Ubuntu VPS, not macOS.

## Vercel Web + Dedicated Live API + Render Worker Runtime

Vercel serves the operator console through `server.py`. Vault live readiness, routing preview, start-cycle, and cycle-status JSON calls can be delegated to a fixed-egress live API service through `PUBLIC_LIVE_API_ORIGIN`; the always-on trading worker runs separately on Render through `render.yaml`. The Vercel app, live API, and worker must point at the same PostgreSQL database and use matching session/encryption secrets.

Prepare static assets for Vercel locally:

```bash
scripts/prepare_vercel_static.sh
```

Before first production traffic, provision a Vercel Marketplace Postgres database, set the Vercel env vars from `deploy/env.vercel.example`, set the live API env vars from `deploy/env.live-api.example`, set the Render worker env vars from `deploy/env.render-worker.example`, then run migrations against that database:

```bash
SKIP_SCHEMA_BOOTSTRAP=1 python -m flask --app wsgi:app db upgrade
```

If production is running in recovery SQLite because the configured Postgres target is unavailable, keep `ALGVAULT_RECOVERY_SQLITE_ENABLED=true` until the database is verified healthy, migrated, and imported. Use the safe cutover checklist in [docs/ops/postgres-recovery-cutover.md](docs/ops/postgres-recovery-cutover.md); do not change `DATABASE_URL` or disable recovery mode as part of a health-only deployment.

Vercel production defaults should include:

```bash
APP_ENV=production
DEPLOYMENT_TARGET=vercel
APP_MODE=live
ENABLE_LIVE_TRADING=true
WORKER_MODE=web
ENABLE_IN_PROCESS_WORKERS=false
WORKER_PROCESS_CONFIGURED=true
SCHEMA_BOOTSTRAP_ENABLED=false
```

Render worker defaults should match the same database and secrets, with `WORKER_MODE=worker` and start command `python -m app.workers.runner`.

Set `PUBLIC_LIVE_API_ORIGIN` on Vercel only when Vault live JSON calls should leave the Vercel runtime. Set `LIVE_API_CORS_ALLOWED_ORIGINS` on the live API to the exact operator-console origin. `SESSION_COOKIE_DOMAIN` is never derived automatically; set it explicitly to `.algvault.com` only when the console and API are on sibling subdomains such as `app.algvault.com` and `api.algvault.com`. A `vercel.app` console cannot share that cookie domain with `api.algvault.com`, so use a custom app domain before relying on shared session cookies.

For 1H10 live execution, configure the live API and worker in the same compliant non-restricted region with fixed outbound IPs or documented outbound ranges, then allowlist those addresses with each exchange. `PUBLIC_LIVE_API_ORIGIN` should point the browser's Vault calls at that live API; Vercel page rendering will not perform exchange probes when this origin is configured. KuCoin must remain disabled unless `KUCOIN_OPERATOR_REGION` truthfully identifies a non-restricted account/operator region and the runtime location is eligible under KuCoin terms. Keep `WALLET_WITHDRAWALS_ENABLED=false` unless KMS/HSM/MPC custody, signer isolation, SDK checks, and withdrawal caps are fully configured.

Deployment split checklist:

- Vercel/operator console: keep `PUBLIC_APP_ORIGIN` and `PUBLIC_API_ORIGIN` on the console origin for existing non-Vault browser calls, and set `PUBLIC_LIVE_API_ORIGIN=https://api.example.com` only for delegated Vault JSON calls.
- Render/live API: set `PUBLIC_APP_ORIGIN=https://app.example.com`, `PUBLIC_API_ORIGIN=https://api.example.com`, `PUBLIC_LIVE_API_ORIGIN=https://api.example.com`, and `LIVE_API_CORS_ALLOWED_ORIGINS=https://app.example.com`. Use the same database, `FLASK_SECRET_KEY`, and `TOTP_ENCRYPTION_KEY` as the console. Set `SESSION_COOKIE_DOMAIN=.example.com` only for sibling subdomains such as `app.example.com` and `api.example.com`.
- Live gates and secrets belong on the live API/worker only: `ENABLE_LIVE_TRADING=true`, `APP_MODE=live`, `ONE_H10_LIVE_ENABLED=true`, explicit/secondary live confirmations, promoted 1H10 ML readiness, verified exchange credentials, usable collateral, and fresh market metadata. Keep secret values out of docs and git.
- Do not use wildcard CORS with credentials. Do not rely on `SESSION_COOKIE_DOMAIN` for `*.vercel.app` preview domains. KuCoin region/account restrictions require an eligible account/operator and a compliant fixed-egress runtime region.

After deployment, run the no-secrets live API smoke from an operator machine:

```bash
export LIVE_API_BASE_URL=https://api.example.com
export FRONTEND_ORIGIN=https://app.example.com
export DISALLOWED_ORIGIN=https://evil.example
.venv/bin/python scripts/live_api_smoke.py
```

The smoke checks credentialed CORS on `/api/vault/routing-preview`, rejects wildcard/disallowed-origin CORS, accepts unauthenticated auth failures, and fails if the response body exposes raw KuCoin restricted-region text, backend IP text, provider JSON, or stack traces. It does not send cookies, tokens, API keys, or order requests.

## VPS/Postgres Runtime

For a VPS deployment, use Postgres, run Alembic migrations explicitly, and run live strategy/scanner/treasury work in the dedicated DB-lease worker instead of Gunicorn web workers. Production defaults set `ENABLE_IN_PROCESS_WORKERS=false`; queued strategy runs are picked up by `python -m app.workers.runner`.

```bash
export DEPLOYMENT_TARGET=vps
export DATABASE_URL=postgresql+psycopg://tradingbot:<password>@127.0.0.1:5432/tradingbot
export WEB_CONCURRENCY=2
export GUNICORN_THREADS=4
export WORKER_MODE=web
export ENABLE_IN_PROCESS_WORKERS=false
export WORKER_PROCESS_CONFIGURED=true
flask db upgrade
flask production-readiness --strict
gunicorn -c deploy/gunicorn.conf.py wsgi:app
```

Start the worker in a second process or via `deploy/systemd/algorithm-vault-worker.service`:

```bash
WORKER_MODE=worker python -m app.workers.runner
# or one-shot diagnostics:
flask worker start --once --job strategy_starter
```

Postgres is the supported VPS database backend. Local SQLite remains valid for local operation, but production startup no longer creates or patches schema implicitly; schema changes are versioned under `migrations/`.

Production deployment templates live under `deploy/`:

```bash
sudo install -d -o tradingbot -g tradingbot /etc/algorithm-vault /var/log/algorithm-vault
sudo cp deploy/env.production.example /etc/algorithm-vault/algorithm-vault.env
sudo cp deploy/nginx/algorithm-vault.conf /etc/nginx/sites-available/algorithm-vault
sudo ln -sfn /etc/nginx/sites-available/algorithm-vault /etc/nginx/sites-enabled/algorithm-vault
sudo cp deploy/systemd/algorithm-vault.service /etc/systemd/system/algorithm-vault.service
sudo cp deploy/systemd/algorithm-vault-worker.service /etc/systemd/system/algorithm-vault-worker.service
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now algorithm-vault algorithm-vault-worker
```

Use `wsgi:app` with `deploy/gunicorn.conf.py` for production. The default bind is `127.0.0.1:8000`; Nginx owns external ports, redirects HTTP to HTTPS, terminates TLS with a valid certificate for `app.algvault.com`, and proxies health, app, static, service-worker, manifest, icons, and SSE stream traffic. Confirm DNS for `app.algvault.com` points at the deployment host and that the certificate chain is valid before installing the iPhone PWA.

Production environment files should keep HTTPS posture enabled:

```bash
APP_ENV=production
DEPLOYMENT_TARGET=vps
PREFERRED_URL_SCHEME=https
PUBLIC_APP_ORIGIN=https://app.algvault.com
PUBLIC_API_ORIGIN=https://app.algvault.com
PUBLIC_LIVE_API_ORIGIN=https://api.algvault.com
PROXY_FIX_ENABLED=true
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Lax
SESSION_COOKIE_DOMAIN=.algvault.com
SECURE_HEADERS_HSTS_ENABLED=true
```

Browser API calls, auth redirects, manifest URLs, service-worker registration, and EventSource streams resolve through `PUBLIC_APP_ORIGIN` and `PUBLIC_API_ORIGIN`. Vault live JSON calls resolve through `PUBLIC_LIVE_API_ORIGIN` when set. For a split `app.algvault.com` / `api.algvault.com` topology, set `LIVE_API_CORS_ALLOWED_ORIGINS`, preserve CSRF/session secrets across both services, and explicitly set `SESSION_COOKIE_DOMAIN=.algvault.com` on both web/API services. Fill `/etc/algorithm-vault/algorithm-vault.env` with real values, then run:

```bash
scripts/production_bootstrap.sh
flask db upgrade
scripts/healthcheck.sh
flask production-readiness --strict
```

Before compacting a large local SQLite database, run the protected cleanup command in dry-run mode:

```bash
flask prune-efficiency-data --protect-username sufyanh
```

To delete old noisy `strategy/no_trade` and transient backoff diagnostics, require the exact confirmation phrase. The command backs up SQLite first and refuses to continue unless the protected `sufyanh` account can be verified. It does not delete users, wallet balances, wallet transactions, wallet/deposit addresses, ledger events, withdrawals, trading connections, vault cycles, orders, or fills.

```bash
flask prune-efficiency-data --protect-username sufyanh --confirm PRUNE-EFFICIENCY-DATA --vacuum
```

Check the local `sufyanh` profile and in-app wallet without touching exchange balances or submitting orders:

```bash
flask profile-wallet-check --username sufyanh
```

This command is read-only. It reports the local app wallet balance source, locked funds, order count, verified trading connections, cached exchange snapshot metadata, and reconciliation warnings such as duplicate completed settlement transactions. Normal wallet and dashboard pages render from local/cached data first; use `/wallet?refresh_exchange=1` or `/admin/dashboard?refresh_exchange=1` only when you want an explicit read-only provider snapshot refresh.

Before cutting over production traffic, export the local protected account wallet snapshot and verify it on the production database/host:

```bash
flask profile-wallet-check --username sufyanh > /secure-transfer/sufyanh-wallet-snapshot.json
flask production-account-readiness \
  --username sufyanh \
  --expected-origin https://app.algvault.com \
  --expected-wallet-snapshot /secure-transfer/sufyanh-wallet-snapshot.json
```

`production-account-readiness` is read-only. It fails closed unless `PUBLIC_APP_ORIGIN` and `PUBLIC_API_ORIGIN` both resolve to the expected public HTTPS production origin, the `sufyanh` profile exists, the in-app wallet has funds, and each asset total is at least the exported local wallet snapshot. Use `--allow-no-snapshot` only for a weaker existence/nonzero-funds smoke check when no local snapshot is available.

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

### KuCoin Spot Smoke Tests

Normal tests never trade and do not require KuCoin credentials. The KuCoin spot smoke tests use environment variables only and are skipped unless the guard values are present. Do not put API keys, passphrases, account IDs, or `.env` files in git. Use a key with General permission for read-only checks and add Spot permission only for test-order/create/cancel validation; do not enable Withdrawal permission.

```bash
export KUCOIN_API_KEY=...
export KUCOIN_API_SECRET=...
export KUCOIN_API_PASSPHRASE=...
export KUCOIN_TEST_ACCOUNT=sufyanh
export KUCOIN_TEST_SYMBOL=BTC-USDT
export KUCOIN_MAX_TEST_NOTIONAL_USDT=5
export KUCOIN_ENABLE_LIVE_TEST_TRADES=false
export KUCOIN_ENABLE_FILL_TEST=false
```

Run deterministic KuCoin coverage without credentials:

```bash
.venv/bin/python -m pytest tests/test_kucoin_spot_provider.py tests/test_kucoin_time_sync.py
```

Run the gated live smoke tests only after reviewing the printed preflight summary:

```bash
.venv/bin/python -m pytest tests/test_kucoin_spot_live_smoke.py -m "integration and live" -s
```

The live smoke prints account `sufyanh`, symbol, max notional, live-trading flag, fill-test flag, and redacted credential presence. The first live test reads server/account state and confirms available quote balance for `KUCOIN_TEST_SYMBOL`. The second uses KuCoin Spot `POST /api/v1/hf/orders/test`; this verifies auth, signature, and payload shape but does not enter the matching system and must not be queried afterward. The third places a tiny off-market post-only spot limit order and cancels it immediately, and it only runs with `KUCOIN_ENABLE_LIVE_TEST_TRADES=true`. The fourth performs a minimal market buy/sell round trip only when both `KUCOIN_ENABLE_LIVE_TEST_TRADES=true` and `KUCOIN_ENABLE_FILL_TEST=true`; it stays under `KUCOIN_MAX_TEST_NOTIONAL_USDT` and stops on unsupported minimums or unexpected fill state.

Hyperliquid remains routed through the official Python SDK and requires an API wallet/agent private key, not a recovery phrase or main wallet seed. Rejected SDK order responses are stored on the local order and audit details so the operator can see the provider-side reason.

### Hyperliquid Testnet Smoke Tests

Hyperliquid provider smoke tests use the saved `sufyanh` trading connection as the credential source, but signed tests are testnet-only and fail closed unless every guard is set:

```bash
export HYPERLIQUID_ACCOUNT=sufyanh
export HYPERLIQUID_ENV=testnet
export HYPERLIQUID_BASE_URL=https://api.hyperliquid-testnet.xyz
export RUN_HYPERLIQUID_LIVE_TESTS=1
# Optional, stronger IOC/fill/flatten test:
export RUN_HYPERLIQUID_FILL_TEST=1
# Optional cap. Leave unset or 0 to use the smallest practical valid notional:
export HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD=0
```

Run the normal suite first, then the Hyperliquid-specific tests:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_hyperliquid_client.py tests/test_trading_connections.py tests/test_hyperliquid_testnet_smoke.py
```

The read-only smoke checks testnet metadata, mids, order book, account state, balances, positions, open orders, and recent fills. The signed smoke computes a tiny valid BTC/ETH perp size from Hyperliquid metadata, submits an ALO/post-only limit with a `codex-hl-test-` local client order id, verifies it is open, cancels it, and verifies it is gone. Cleanup uses `finally` blocks to cancel created orders and flatten any resulting position. The optional fill smoke places the smallest practical IOC order, verifies fills/status, refreshes balances/positions, then immediately flattens reduce-only through the SDK.

Current local probing of the saved `sufyanh` testnet address shows it is queryable but has no positive testnet margin/withdrawable balance. Until testnet funds are available, signed smoke tests stop before order placement with an `insufficient_testnet_funds` message.

For the first real funds test, keep the command path CLI-only:

```bash
python3 -m pip install -r requirements-dev.txt
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

`ml-quality-report --compact` is read-only and shows blocked families, promoted/candidate model IDs, ready candidate IDs, and exact retrain/promote commands for each provider/family. Omit `--compact` when you need full metrics. `RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED` defaults to `true`; `RAPID_ML_MAX_SYMBOLS_PER_PROVIDER` defaults to `48` to keep live API usage bounded while selecting from the discovered futures universe. `RAPID_ML_ML_SIZING_ENABLED` defaults to `true`, so order size is derived from promoted model confidence, edge, allocator, risk, ROI, user-provided capital, and available balance. Sizing is still constrained by exchange minimum order value, daily loss, adaptive slippage, stop/take requirements, and open-position reconciliation. `RAPID_ML_MAX_DAILY_LOSS_PCT` defaults to `0.05` and is clamped at `0.10` unless the admin risk page explicitly enables unlimited daily-loss mode. `RAPID_ML_DECISION_INTERVAL_MS` defaults to `1000`, and `RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER` defaults to one opening order per provider per second. The loop does not force an order every interval; it submits only when promoted ML expected edge clears the conservative profitability gate: round-trip provider fees (`RAPID_ML_FEE_BPS_BY_PROVIDER_JSON`), expected slippage (`RAPID_ML_SLIPPAGE_BPS`), live spread or `RAPID_ML_UNKNOWN_SPREAD_BPS`, `RAPID_ML_COST_RESERVE_BPS`, `RAPID_ML_MIN_CONFIDENCE`, `RAPID_ML_MIN_EDGE_AGREEMENT`, and `RAPID_ML_MIN_EDGE_BPS`. Rapid scoring builds leakage-safe live candle features using `RAPID_ML_FEATURE_TIMEFRAME` and `RAPID_ML_FEATURE_CANDLE_LIMIT`; missing or stale features block submission. Provider circuit breakers use `RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES` so stale transient failures do not permanently block a now-healthy venue. Offline ranker scoring also respects `ML_OFFLINE_SAFE_SCORING_MODEL_TYPES`; keep it at `sklearn` unless XGBoost scoring has been separately validated in this runtime.

Activate KuCoin only after the Hyperliquid order, audit log, provider order status, balances, open orders, and positions are understood. Activating one verified connection deactivates the user's other trading connections:

```bash
flask activate-trading-connection --user-id <id> --connection-id <kucoin-connection-id> --confirm ACTIVATE-LIVE-CONNECTION
flask live-auto-canary --user-id <id> --provider kucoin --research-budget-minutes 30
```

The first canary allocation budget defaults to `FIRST_CANARY_ALLOCATION_BUDGET_USDT=1`. If the venue minimum order size blocks the preferred budget, set `FIRST_CANARY_USE_MIN_SIZE_FALLBACK=true` only long enough to use `FIRST_CANARY_FALLBACK_ALLOCATION_BUDGET_USDT=2`. Keep `CANARY_PREVIEW_ONLY=true` until the readiness and preview output are reviewed. To submit exactly one canary:

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

## Vercel Deployment Status

AlgVault uses Vercel's native GitHub integration, not a GitHub Actions deploy workflow. Local and Codex Cloud changes must be committed and pushed to `https://github.com/shamh21/algorithm-vault` before Vercel can build them.

- Main app: `https://algorithm-vault-chi.vercel.app/`
- Admin app: `https://admin-pwa-alpha.vercel.app/`
- Production branch: `main`, unless the Vercel dashboard is intentionally configured otherwise
- Preview behavior: non-production branches and pull requests should create Vercel Preview Deployments

See `DEPLOYMENT.md` for the exact root directories, build commands, environment checklist, and stale PWA cache troubleshooting.
