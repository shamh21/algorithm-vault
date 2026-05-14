# AlgVault Vercel Deployment Workflow

AlgVault deploys from one GitHub repository into two Vercel projects:

- Main app: `https://algorithm-vault-chi.vercel.app/`
- Admin app: `https://admin-pwa-alpha.vercel.app/`
- GitHub repo: `https://github.com/shamh21/algorithm-vault`
- Production branch: `main`, unless the Vercel dashboard is intentionally configured otherwise

Use Vercel's native Git integration as the steady-state deployment path. Do not add Vercel tokens, broker credentials, webhook secrets, private environment values, or CLI deployment secrets to this repo.

## Workflow

1. Make changes locally or in Codex Cloud.
2. Run the relevant verification commands before publishing.
3. Commit the changes to a Git branch.
4. Push the branch to GitHub.
5. Use the Vercel Preview Deployment created for that branch or pull request to verify the change.
6. Merge the pull request into `main` only after checks and preview validation pass.
7. Vercel creates the Production Deployment from `main` and updates the production domains after a successful build.

Local machine changes and Codex Cloud changes follow the same rule: they do not reach Vercel until they are committed and pushed to GitHub.

## Vercel Project Settings

Main app project:

- Git provider: GitHub repo `shamh21/algorithm-vault`
- Root directory: repository root
- Framework preset: Flask or Other with the checked-in `vercel.json`
- Install command: default/blank unless the dashboard requires a Python install override
- Build command: `bash scripts/prepare_vercel_static.sh`
- Output directory: default/blank
- Production branch: `main`
- Required env vars: configure values from `deploy/env.vercel.example` in Vercel for Production and Preview

Admin app project:

- Git provider: GitHub repo `shamh21/algorithm-vault`
- Root directory: `admin-pwa`
- Framework preset: Next.js
- Install command: `npm ci`
- Build command: `npm run build`
- Output directory: default Next.js output, `.next`
- Node.js version: `20.9.0` or newer; use `20.19.0` or newer if npm reports package engine warnings
- Production branch: `main`
- Required env vars: configure `NEXT_PUBLIC_API_BASE_URL` and `BACKEND_ORIGIN` as needed in Vercel for Production and Preview

## Dashboard Checklist

Check both Vercel projects before relying on automatic deployments:

- Git provider is connected to `https://github.com/shamh21/algorithm-vault`.
- Production branch is `main`, unless a deliberate non-default production branch is documented.
- Auto-deployments are enabled for Git pushes.
- Ignored Build Step is set to Automatic or otherwise does not skip normal Preview and Production deployments.
- Main app uses the repository root and `bash scripts/prepare_vercel_static.sh`.
- Admin app uses `admin-pwa`, `npm ci`, and `npm run build`.
- Production and Preview environment variables are configured in Vercel, not committed to git.
- Preview deployments use Preview-scoped env vars and production deployments use Production-scoped env vars.
- Production domains are assigned to the intended project and environment.

## Verification

Root app:

```bash
PYTHON_BIN=.venv/bin/python scripts/check_local_setup.sh
.venv/bin/python -m ruff check app/settings_validation.py app/services/audit_events.py app/services/failures.py app/services/response_envelope.py app/services/vault_cycle_orchestrator.py app/services/worker_lease.py app/services/model_registry.py app/workers tests/test_audit_events.py tests/test_production_hardening_migrations.py tests/test_vault_cycle_engine.py tests/test_worker_and_runtime_hardening.py
.venv/bin/python -m ruff format --check app/settings_validation.py app/services/audit_events.py app/services/failures.py app/services/response_envelope.py app/services/vault_cycle_orchestrator.py app/services/worker_lease.py app/services/model_registry.py app/workers tests/test_audit_events.py tests/test_production_hardening_migrations.py tests/test_vault_cycle_engine.py tests/test_worker_and_runtime_hardening.py
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
bash scripts/prepare_vercel_static.sh
```

Admin app:

```bash
cd admin-pwa
npm ci
npm run lint
npm run typecheck
npm run build
```

## Deployment Troubleshooting

- Change is not visible: confirm the change was committed and pushed to GitHub.
- Change is visible only on Preview: merge the branch into `main` or verify the configured production branch.
- Vercel deployment failed: inspect the deployment logs for build, lint, typecheck, or test failures.
- Deployment was skipped: check the project's Ignored Build Step setting.
- Wrong app deployed: verify the Vercel project root directory, especially root vs `admin-pwa`.
- Runtime/build config missing: compare Vercel environment variables against `.env.example`, `deploy/env.vercel.example`, and the admin app variables above.
- iPhone or installed PWA looks stale: close/reopen the installed PWA, refresh Safari, clear website data for the old origin, or reinstall the PWA after the production deployment is ready.
