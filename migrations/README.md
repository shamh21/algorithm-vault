# Database Migration Notes

AlgVault uses Flask-Migrate/Alembic for versioned schema changes.

Local or test database:

```bash
flask db upgrade
```

Staging/production:

```bash
export APP_ENV=production
export DEPLOYMENT_TARGET=vps
export DATABASE_URL=postgresql+psycopg://...
flask db upgrade
flask production-readiness --strict
```

The baseline migration matches the current SQLAlchemy model schema. New worker
lease/idempotency and model-governance fields are in follow-on migrations.
Production startup verifies the Alembic version table instead of silently
creating or patching tables. Local/test startup can still create a fresh schema
when `SCHEMA_BOOTSTRAP_ENABLED=true`.

To generate a reviewed migration after model changes:

```bash
SKIP_SCHEMA_BOOTSTRAP=1 flask db migrate -m "describe schema change"
flask db upgrade
```
