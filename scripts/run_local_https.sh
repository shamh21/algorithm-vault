#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_HTTPS_ENV="${LOCAL_HTTPS_ENV:-$ROOT_DIR/.env.local-https}"
if [[ -f "$LOCAL_HTTPS_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_HTTPS_ENV"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

export FLASK_APP="${FLASK_APP:-wsgi:app}"
export FLASK_DEBUG="${FLASK_DEBUG:-1}"
export PREFERRED_URL_SCHEME="${PREFERRED_URL_SCHEME:-https}"
export SESSION_COOKIE_SECURE="${SESSION_COOKIE_SECURE:-true}"
export SESSION_COOKIE_HTTPONLY="${SESSION_COOKIE_HTTPONLY:-true}"
export SESSION_COOKIE_SAMESITE="${SESSION_COOKIE_SAMESITE:-Lax}"
export SECURE_HEADERS_HSTS_ENABLED="${SECURE_HEADERS_HSTS_ENABLED:-false}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5443}"
LOCAL_HTTPS_CERT="${LOCAL_HTTPS_CERT:-$ROOT_DIR/.local-certs/algorithm-vault-local.crt}"
LOCAL_HTTPS_KEY="${LOCAL_HTTPS_KEY:-$ROOT_DIR/.local-certs/algorithm-vault-local.key}"

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Warning: .env is missing. Copy .env.example to .env and replace local placeholders." >&2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found. Create .venv or set PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -r "$LOCAL_HTTPS_CERT" || ! -r "$LOCAL_HTTPS_KEY" ]]; then
  echo "Local HTTPS certificate/key not found or not readable." >&2
  echo "Expected cert: $LOCAL_HTTPS_CERT" >&2
  echo "Expected key:  $LOCAL_HTTPS_KEY" >&2
  echo "Generate ignored local files with: scripts/create_local_https_cert.sh" >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use." >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
  echo "Use another port: PORT=$((PORT + 1)) scripts/run_local_https.sh" >&2
  exit 1
fi

LAN_IP="${LOCAL_HTTPS_HOST:-}"
if [[ -z "$LAN_IP" ]] && command -v ipconfig >/dev/null 2>&1; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi
if [[ -z "$LAN_IP" ]] && command -v hostname >/dev/null 2>&1; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi

echo "Starting Algorithm Vault HTTPS development server for desktop diagnostics"
echo "Local desktop URL: https://127.0.0.1:$PORT"
if [[ -n "$LAN_IP" ]]; then
  echo "LAN diagnostic URL: https://$LAN_IP:$PORT"
fi
echo "Do not install the iPhone PWA from a self-signed LAN/IP HTTPS URL."
echo "Use https://app.algvault.com or a stable public HTTPS tunnel for iPhone PWA testing."
echo "Bind: $HOST:$PORT"
echo "Cert: $LOCAL_HTTPS_CERT"
echo "FLASK_APP: $FLASK_APP"

exec "$PYTHON_BIN" -m flask run --host "$HOST" --port "$PORT" --cert "$LOCAL_HTTPS_CERT" --key "$LOCAL_HTTPS_KEY"
