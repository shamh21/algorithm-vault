# AlgVault Codex Instructions

## Project Overview

AlgVault is an automated trading platform with a mobile-first Flask/Jinja PWA at the repository root and a nested Next.js admin PWA in `admin-pwa/`.

- Root app: Python Flask, Jinja templates, static CSS/JS PWA assets, Alembic migrations, pytest, Ruff, mypy.
- Admin app: Next.js 16, React 19, TypeScript, ESLint, npm lockfile.
- Deployment assumptions: Vercel serves the Flask web/API entrypoint through `server.py` and prepared static assets from `public/`; Render can run the live API and worker from `render.yaml`; VPS/Gunicorn deployment uses `wsgi:app`.

Use Python 3.12 for new Codex Cloud environments because `.python-version`, `pyproject.toml`, `uv.lock`, and `render.yaml` point to 3.12. The current local `.venv` may differ.
Use Node.js 20.9.0 or newer for `admin-pwa`; the installed Next.js package declares `node >=20.9.0`.

## Setup And Commands

Root install:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-dev.txt
```

Root development and verification:

```bash
scripts/run_local_dev.sh
scripts/run_local_production.sh
PYTHON_BIN=.venv/bin/python scripts/check_local_setup.sh
.venv/bin/python -m ruff check app/settings_validation.py app/services/audit_events.py app/services/failures.py app/services/response_envelope.py app/services/vault_cycle_orchestrator.py app/services/worker_lease.py app/services/model_registry.py app/workers tests/test_audit_events.py tests/test_production_hardening_migrations.py tests/test_vault_cycle_engine.py tests/test_worker_and_runtime_hardening.py
.venv/bin/python -m ruff format --check app/settings_validation.py app/services/audit_events.py app/services/failures.py app/services/response_envelope.py app/services/vault_cycle_orchestrator.py app/services/worker_lease.py app/services/model_registry.py app/workers tests/test_audit_events.py tests/test_production_hardening_migrations.py tests/test_vault_cycle_engine.py tests/test_worker_and_runtime_hardening.py
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
scripts/prepare_vercel_static.sh
```

Admin PWA install and verification:

```bash
cd admin-pwa
npm ci
npm run dev
npm run lint
npm run typecheck
npm run build
npm run start
```

There is no root `package.json`, no root JavaScript package manager, and no admin test script unless a test framework is added later.

## Change Rules

- Inspect existing patterns before editing. Prefer nearby Flask, Jinja, static JS, service, and Next component conventions over new abstractions.
- Preserve routes, component APIs, request/response shapes, state management, styling system, and dependencies unless the task explicitly asks for a change.
- Prefer small, targeted changes over broad rewrites. Avoid unrelated formatting churn.
- Run the relevant checks before finishing. For root Python changes, use the CI-style Ruff, mypy, and pytest commands above. For `admin-pwa`, use lint, typecheck, and build.
- Summarize changed files after every task, and call out any commands that were not run or failed.

## Product And Design Rules

- AlgVault should feel secure, precise, modern, premium, trustworthy, and fintech-grade.
- Keep copy sober and clear. Do not use guaranteed-profit, risk-free, market-beating, or investment-advice language.
- Prefer language around automated execution, strategy monitoring, analytics, broker/API connectivity, user control, and risk visibility.
- Keep the UI dark-mode friendly, mobile-first, and optimized for iPhone/iOS/PWA usage.
- Maintain clear loading, empty, error, stale-data, disconnected broker/API, failed sync, failed trade, and recovery states.
- Do not redesign routes or trading workflows unless explicitly asked.

## Accessibility Rules

- Use semantic HTML and accurate form labels.
- Preserve keyboard support for controls, menus, dialogs, and navigation.
- Keep visible focus states and sufficient contrast.
- Respect reduced-motion preferences for animations and transitions.

## Security Rules

- Never expose broker credentials, API keys, webhooks, private tokens, trading secrets, wallet keys, or sensitive strategy execution logic in frontend code.
- Broker connections, trade execution, webhooks, credential storage, signing, withdrawals, and account-sensitive operations must stay server-side.
- Treat all `NEXT_PUBLIC_*` variables as public browser-exposed values.
- Do not log secrets, raw provider credentials, private wallet material, or account-sensitive trading data.
- Do not add real credentials to the repo. Use placeholders in examples and docs.
- Client-side readiness, routing previews, and button states are advisory only. Server-side validation and risk gates must remain authoritative.

## PWA Rules

- Preserve installability, mobile-first layouts, viewport-fit support, manifest links, icons, and service-worker behavior.
- Keep offline and poor-network states explicit and safe.
- Avoid making trading actions appear successful when data is stale, disconnected, offline, or blocked by server readiness.
- Do not add or replace the PWA framework unless a task explicitly requires it.
