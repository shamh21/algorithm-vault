# Codex Cloud Readiness

## Detected Stack

AlgVault is a mixed Python and JavaScript repository.

- Root app: Flask 3, Jinja templates, SQLAlchemy/Flask-Migrate, static CSS/JS PWA assets, pytest, Ruff, mypy.
- Main PWA: served by the Flask root app with `static/manifest.json`, `static/manifest.webmanifest`, `static/js/sw.js`, iOS icon assets, and mobile viewport metadata in `templates/base.html`.
- Admin PWA: `admin-pwa/` uses Next.js 16, React 19, TypeScript, Tailwind CSS 4, ESLint, and npm.
- Deployment: `vercel.json` prepares static assets and routes Flask through `server.py`; `render.yaml` defines a live API and worker; `wsgi.py` supports Gunicorn/VPS deployment.

Python version signals are mixed. Use Python 3.12 for Codex Cloud because `.python-version`, `pyproject.toml`, `uv.lock`, and `render.yaml` point to 3.12. The existing GitHub Actions workflow currently uses Python 3.11, and the inspected local `.venv` was Python 3.10.

Node version is only constrained by the admin PWA dependencies. The installed Next.js package declares `node >=20.9.0`; use Node.js 20.9.0 or newer in Codex Cloud.

## Package Managers And Setup

The root app uses `pip` with `requirements*.txt`. The ignored `uv.lock` is present locally but is not authoritative for Codex Cloud setup because `.gitignore` excludes it and the README documents pip requirements. The admin PWA uses npm because `admin-pwa/package-lock.json` is present.

Recommended Codex Cloud setup script:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-dev.txt
cd admin-pwa
npm ci
```

Recommended maintenance commands:

```bash
PYTHON_BIN=.venv/bin/python scripts/check_local_setup.sh
.venv/bin/python -m pip install -r requirements-dev.txt
cd admin-pwa
npm ci
```

Keep agent internet access disabled by default after dependency installation. Enable it only for dependency fetches, official documentation, security advisory checks, or deployment diagnostics that explicitly need network access.

## Environment Variables

Use `.env.example` only as a safe placeholder template. Local secret files such as `.env`, `.env.local`, `.env.vercel.production.local`, and `.env*.local` must remain untracked.

Required non-secret/local-safe values for a development run:

- `FLASK_APP`
- `FLASK_DEBUG`
- `DATABASE_URL`
- `DEPLOYMENT_TARGET`
- `PUBLIC_APP_ORIGIN`
- `PUBLIC_API_ORIGIN`
- `PUBLIC_LIVE_API_ORIGIN`
- `LIVE_API_CORS_ALLOWED_ORIGINS`
- `SESSION_COOKIE_DOMAIN`
- `WORKER_MODE`
- `ENABLE_IN_PROCESS_WORKERS`
- `WORKER_PROCESS_CONFIGURED`
- `ENABLE_LIVE_TRADING`
- `APP_MODE`
- `CANARY_PREVIEW_ONLY`
- `WALLET_WITHDRAWALS_ENABLED`

Required secrets, listed by name only:

- `FLASK_SECRET_KEY`: Flask session/signing secret.
- `TOTP_ENCRYPTION_KEY`: Fernet key for 2FA and connection encryption.
- `ADMIN_PASSWORD`: Optional local admin bootstrap password.
- `SIGNUP_INVITE_CODE`: Optional registration gate.
- `DATABASE_URL`: Secret when it contains production database credentials.
- `HL_ACCOUNT_ADDRESS`: Hyperliquid account/address value for server-side connection use.
- `HL_SECRET_KEY`: Hyperliquid API wallet/private key for server-side signing.
- `KUCOIN_API_KEY`: KuCoin API key for server-side broker integration.
- `KUCOIN_API_SECRET`: KuCoin API secret for server-side broker integration.
- `KUCOIN_API_PASSPHRASE`: KuCoin API passphrase for server-side broker integration.
- `UNISWAP_API_KEY`: Optional server-side delegated trading API key.
- `TREASURY_ENCRYPTION_KEY`: Optional treasury encryption key.
- Wallet RPC/indexer URLs and token maps are deployment-specific and can expose infrastructure details. Keep real values out of git.

`NEXT_PUBLIC_API_BASE_URL` is public and exposed to browser JavaScript in `admin-pwa`. Do not place secrets in any `NEXT_PUBLIC_*` variable.

## Verification Commands

Run these from the repository root unless a command changes directory:

```bash
PYTHON_BIN=.venv/bin/python scripts/check_local_setup.sh
.venv/bin/python -m ruff check app/settings_validation.py app/services/audit_events.py app/services/failures.py app/services/response_envelope.py app/services/vault_cycle_orchestrator.py app/services/worker_lease.py app/services/model_registry.py app/workers tests/test_audit_events.py tests/test_production_hardening_migrations.py tests/test_vault_cycle_engine.py tests/test_worker_and_runtime_hardening.py
.venv/bin/python -m ruff format --check app/settings_validation.py app/services/audit_events.py app/services/failures.py app/services/response_envelope.py app/services/vault_cycle_orchestrator.py app/services/worker_lease.py app/services/model_registry.py app/workers tests/test_audit_events.py tests/test_production_hardening_migrations.py tests/test_vault_cycle_engine.py tests/test_worker_and_runtime_hardening.py
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
cd admin-pwa
npm run lint
npm run typecheck
npm run build
```

Commands intentionally unavailable:

- Root `npm`/`pnpm`/`yarn`/`bun` scripts: there is no root `package.json`.
- Admin PWA tests: no test framework or `npm test` script exists.

No current command failures were found during readiness inspection. The full Python test suite emitted a pre-existing Torch warning about NumPy not being installed, but the suite passed.

## PWA And iOS Readiness Notes

Root Flask PWA:

- Web app manifest exists at `static/manifest.json` and `static/manifest.webmanifest`.
- App name, short name, start URL, scope, display mode, theme color, background color, and portrait orientation are configured.
- iOS metadata and apple touch icon links are present in `templates/base.html`.
- Maskable 192 and 512 icons are present.
- Service worker exists at `static/js/sw.js` and is registered from `static/js/app-shell.js`.
- Offline navigation fallback clearly says AlgVault is offline and tells the user to reconnect before refreshing wallet, vault, or market data.
- API, auth, service-worker, and manifest requests are network-only, which avoids presenting cached trading actions as successful.

Admin PWA:

- Next metadata includes iOS web app support, viewport fit, icons, theme color, and a generated manifest route.
- There is no service worker or offline runtime. Treat it as an installable admin web app without offline behavior unless a future task explicitly adds one.

Trading screens should continue to block or clearly label actions when readiness checks fail, broker/API connectivity is disconnected, data is stale, or the browser is offline.

## Security And Trading Guardrails

Readiness inspection found no live-looking tracked secrets in the inspected tracked files. Dummy credential strings are present in tests only. Ignored local `.env*`, Vercel env, local cert, virtualenv, cache, and generated public files exist on disk and must remain untracked.

Keep these boundaries intact:

- Broker credentials, API keys, signing keys, webhooks, wallet keys, withdrawals, trade execution, and account-sensitive operations stay server-side.
- Frontend readiness previews and disabled states are advisory. Server-side risk/readiness checks must remain authoritative.
- Do not log secrets, raw provider responses containing credentials, wallet private material, or sensitive account data.
- Keep real values out of `.env.example`, docs, source files, tests, and frontend bundles.
- Treat `NEXT_PUBLIC_*` values as public browser data.
