#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

GUNICORN_BIN="${GUNICORN_BIN:-$ROOT_DIR/.venv/bin/gunicorn}"
if [[ -x "$GUNICORN_BIN" ]]; then
  GUNICORN_CMD=("$GUNICORN_BIN")
else
  GUNICORN_CMD=("$PYTHON_BIN" -m gunicorn)
fi

mkdir -p "$ROOT_DIR/instance"

export APP_ENV="${APP_ENV:-production}"
export DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET:-local}"
export WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
export GUNICORN_THREADS="${GUNICORN_THREADS:-4}"
export GUNICORN_BIND="${GUNICORN_BIND:-${HOST:-127.0.0.1}:${PORT:-8765}}"
export GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"
export GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-45}"
export GUNICORN_KEEPALIVE="${GUNICORN_KEEPALIVE:-5}"
export GUNICORN_MAX_REQUESTS="${GUNICORN_MAX_REQUESTS:-1000}"
export GUNICORN_MAX_REQUESTS_JITTER="${GUNICORN_MAX_REQUESTS_JITTER:-100}"
export SQLITE_BUSY_TIMEOUT_MS="${SQLITE_BUSY_TIMEOUT_MS:-30000}"
export SQLITE_ENABLE_WAL="${SQLITE_ENABLE_WAL:-true}"
export RATELIMIT_ENABLED="${RATELIMIT_ENABLED:-true}"
export PREFERRED_URL_SCHEME="${PREFERRED_URL_SCHEME:-http}"
export SESSION_COOKIE_SECURE="${SESSION_COOKIE_SECURE:-false}"
export SECURE_HEADERS_HSTS_ENABLED="${SECURE_HEADERS_HSTS_ENABLED:-false}"

bind_host="${GUNICORN_BIND%:*}"
bind_port="${GUNICORN_BIND##*:}"
health_host="$bind_host"
if [[ "$health_host" == "0.0.0.0" || "$health_host" == "::" || "$health_host" == "[::]" || -z "$health_host" ]]; then
  health_host="127.0.0.1"
fi
HEALTH_BASE_URL="http://$health_host:$bind_port"

if [[ "${ALLOW_ALREADY_RUNNING:-true}" =~ ^(1|true|yes|on)$ ]]; then
  if curl --fail --silent --show-error "$HEALTH_BASE_URL/healthz" >/dev/null 2>&1; then
    echo "Algorithm Vault is already running at $HEALTH_BASE_URL."
    echo "Healthcheck: BASE_URL=$HEALTH_BASE_URL scripts/healthcheck.sh"
    echo "Stop it: scripts/stop_local_production.sh"
    echo "Use another port: PORT=$((bind_port + 1)) scripts/run_local_production.sh"
    exit 0
  fi
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$bind_port" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $bind_port is already in use, but $HEALTH_BASE_URL/healthz did not pass." >&2
  lsof -nP -iTCP:"$bind_port" -sTCP:LISTEN >&2 || true
  echo "Stop Algorithm Vault listeners with: scripts/stop_local_production.sh" >&2
  echo "Or choose another port with: PORT=$((bind_port + 1)) scripts/run_local_production.sh" >&2
  exit 1
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  export DATABASE_URL="sqlite:///$ROOT_DIR/instance/algorithm_vault_local_production.db"
fi

if [[ "${RUN_BOOTSTRAP:-true}" =~ ^(1|true|yes|on)$ ]]; then
  "$ROOT_DIR/scripts/production_bootstrap.sh"
fi

echo "Algorithm Vault optimized local production run"
echo "Bind: $GUNICORN_BIND"
echo "Workers: $WEB_CONCURRENCY web worker, $GUNICORN_THREADS threads"
echo "Healthcheck: BASE_URL=$HEALTH_BASE_URL scripts/healthcheck.sh"

exec "${GUNICORN_CMD[@]}" -c deploy/gunicorn.conf.py wsgi:app
