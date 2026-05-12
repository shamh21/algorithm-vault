#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

export FLASK_APP="${FLASK_APP:-wsgi:app}"

echo "Algorithm Vault bootstrap: initializing application schema"

if [[ -f "$ROOT_DIR/migrations/env.py" ]]; then
  "$PYTHON_BIN" -m flask db upgrade
else
  "$PYTHON_BIN" - <<'PY'
from app import create_app

app = create_app()
with app.app_context():
    print("Application schema bootstrap completed.")
PY
fi

if [[ "${STRICT_BOOTSTRAP_READINESS:-false}" =~ ^(1|true|yes|on)$ ]]; then
  "$PYTHON_BIN" -m flask production-readiness --strict
fi
