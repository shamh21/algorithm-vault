#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

export FLASK_APP="${FLASK_APP:-wsgi:app}"
export FLASK_DEBUG="${FLASK_DEBUG:-1}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Warning: .env is missing. Copy .env.example to .env and replace local placeholders." >&2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found. Create .venv or set PYTHON_BIN." >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use." >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
  echo "Use another port: PORT=$((PORT + 1)) scripts/run_local_dev.sh" >&2
  exit 1
fi

echo "Starting Algorithm Vault development server"
echo "URL: http://$HOST:$PORT"
echo "FLASK_APP: $FLASK_APP"

exec "$PYTHON_BIN" -m flask run --host "$HOST" --port "$PORT"
