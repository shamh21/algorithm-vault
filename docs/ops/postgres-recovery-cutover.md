# Postgres Recovery Cutover Runbook

Production must stay on recovery SQLite until a healthy Postgres database has been verified, migrated, and imported. Do not set `ALGVAULT_RECOVERY_SQLITE_ENABLED=false` in Vercel Production while the configured Postgres target is unavailable.

## Current Guardrails

- Recovery SQLite is intentional while Prisma Postgres at `db.prisma.io` is unavailable or returning `planLimitReached`.
- Live trading, withdrawals, schema bootstrap, and in-process workers must remain disabled while `RECOVERY_SQLITE_ACTIVE=true`.
- Do not change `DATABASE_URL` unless an approved healthy Postgres target is provided.
- Do not use Neon unless explicitly authorized.
- Do not attempt Supabase or Nile provisioning until Vercel Marketplace terms have been accepted.
- Do not print or paste database credentials, broker credentials, API keys, webhooks, wallet keys, or private tokens.

## Healthy-Recovery Verification

Run these checks before any database cutover work:

```bash
curl -fsS https://algorithm-vault-chi.vercel.app/readyz
curl -fsS https://algorithm-vault-chi.vercel.app/ops/status
curl -fsSI https://algorithm-vault-chi.vercel.app/login
```

Expected recovery state:

- `/readyz` returns `ok=true`, `database=sqlite`, `checks.database_recovery_mode=true`, and a `configured_postgres` diagnostic without credentials.
- `/ops/status` returns `runtime_config.database_recovery_mode=true`, `runtime_config.live_trading_enabled=false`, `database_recovery.active=true`, and live-operation blockers for PostgreSQL and recovery SQLite.
- `/login` returns `200`.

## Pre-Cutover Checklist

Do not perform the cutover until every item below is complete:

1. Restore or replace the Prisma Postgres resource, or provide another explicitly approved healthy Postgres target.
2. Confirm `DATABASE_URL` points to that healthy Postgres target and direct connection succeeds without `planLimitReached`.
3. Confirm Vercel Production can reach the same target through deployment logs or the non-secret `/readyz` and `/ops/status` diagnostics.
4. Pull Vercel Production envs to a temporary file outside the repo:

```bash
tmp_env="$(mktemp /tmp/algvault-vercel-prod.XXXXXX)"
npx --yes vercel env pull "$tmp_env" --environment=production --yes
```

5. Load the temp env only in the current shell, then verify the target without printing credentials:

```bash
set -a
. "$tmp_env"
set +a
ALGVAULT_RECOVERY_SQLITE_ENABLED=false FLASK_SKIP_DOTENV=1 VERCEL=1 SKIP_SCHEMA_BOOTSTRAP=1 \
  .venv/bin/python -m flask --app wsgi:app db upgrade
```

6. Dry-run imports from the recovery bundle for restored accounts and provider rows:

```bash
for username in sufyanh debugmax2 smoke_admin smoke_admin_persist; do
  .venv/bin/python scripts/import_account_to_postgres.py \
    --source app/recovery/algvault_sufyanh.seed \
    --target-url "$DATABASE_URL" \
    --username "$username" \
    --include-connections \
    --dry-run
done
```

7. If the dry-runs are clean, run the same imports without `--dry-run`. Stop on any ID conflict, missing table, or unexpected count.
8. Verify auth, dashboard, admin routes, wallet balances, broker connection state, API health routes, and trading-disabled states against Postgres.
9. Keep live trading blocked until strict readiness passes:

```bash
FLASK_SKIP_DOTENV=1 VERCEL=1 .venv/bin/python -m flask --app wsgi:app production-readiness --strict
FLASK_SKIP_DOTENV=1 VERCEL=1 .venv/bin/python -m flask --app wsgi:app wallet-readiness
FLASK_SKIP_DOTENV=1 VERCEL=1 .venv/bin/python -m flask --app wsgi:app platform-treasury status
```

10. Only after all database, import, auth, wallet, treasury, provider, worker, and risk checks pass, update Vercel Production to `ALGVAULT_RECOVERY_SQLITE_ENABLED=false`.
11. Deploy and verify:

```bash
npx --yes vercel deploy --prod --force --yes
curl -fsS https://algorithm-vault-chi.vercel.app/readyz
curl -fsS https://algorithm-vault-chi.vercel.app/ops/status
```

Expected post-cutover state:

- `/readyz` reports `database=postgres` and does not report `database_recovery_mode=true`.
- `/ops/status` no longer has recovery database blockers.
- Live trading remains blocked unless every explicit live-trading, treasury, custody, worker, provider, and risk safety check passes.

## Rollback

If Postgres health, migrations, imports, auth, wallet balances, or broker checks fail, leave or restore `ALGVAULT_RECOVERY_SQLITE_ENABLED=true` in Vercel Production and redeploy. Do not enable live trading during rollback.
