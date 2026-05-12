#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STRICT=false
for arg in "$@"; do
  case "$arg" in
    --strict)
      STRICT=true
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/check_local_setup.sh [--strict]

Checks the local Flask/PWA developer setup without starting the app.

Environment:
  ENV_FILE   Path to the env file to check. Defaults to .env.
  PYTHON_BIN Python executable. Defaults to .venv/bin/python, then python3.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

failures=0
warnings=0

info() {
  echo "ok: $*"
}

warn() {
  warnings=$((warnings + 1))
  echo "warning: $*" >&2
}

fail() {
  failures=$((failures + 1))
  echo "error: $*" >&2
}

strict_fail_or_warn() {
  if [[ "$STRICT" == "true" ]]; then
    fail "$*"
  else
    warn "$*"
  fi
}

env_value() {
  local key="$1"
  if [[ ! -f "$ENV_FILE" ]]; then
    return 0
  fi
  awk -v key="$key" '
    BEGIN { FS = "=" }
    $0 ~ "^[[:space:]]*#" || $0 ~ "^[[:space:]]*$" { next }
    $1 == key {
      sub(/^[^=]*=/, "")
      print
      exit
    }
  ' "$ENV_FILE"
}

is_placeholder() {
  local value="$1"
  [[ -z "$value" ]] && return 0
  [[ "$value" == "change-me" ]] && return 0
  [[ "$value" == replace-with-* ]] && return 0
  [[ "$value" == *"<required"* ]] && return 0
  [[ "$value" == "CHANGE_ME"* ]] && return 0
  return 1
}

echo "Algorithm Vault local setup check"
echo "Repo: $ROOT_DIR"
echo "Env:  $ENV_FILE"

if command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]]; then
  python_version="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
  if "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    info "Python $python_version is available"
  else
    fail "Python 3.10 or newer is required; found ${python_version:-unknown}"
  fi
else
  fail "Python executable not found. Create .venv or set PYTHON_BIN."
fi

if [[ -f "$ENV_FILE" ]]; then
  info "env file exists"
else
  strict_fail_or_warn "env file is missing. Run: cp .env.example .env"
fi

if [[ -f ".env.example" ]]; then
  info ".env.example exists"
else
  fail ".env.example is missing"
fi

if [[ -f ".gitignore" ]] && grep -Eq '^\.env$' .gitignore && grep -Eq '^!\.env\.example$' .gitignore; then
  info ".gitignore ignores .env and keeps .env.example"
else
  fail ".gitignore must ignore .env and unignore .env.example"
fi

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  fail ".env is tracked by Git; remove it from the index before committing"
else
  info ".env is not tracked by Git"
fi

if [[ -f "$ENV_FILE" ]]; then
  flask_secret="$(env_value FLASK_SECRET_KEY)"
  totp_key="$(env_value TOTP_ENCRYPTION_KEY)"
  database_url="$(env_value DATABASE_URL)"
  app_mode="$(env_value APP_MODE)"
  live_enabled="$(env_value ENABLE_LIVE_TRADING)"

  if is_placeholder "$flask_secret" || [[ "${#flask_secret}" -lt 32 ]]; then
    strict_fail_or_warn "FLASK_SECRET_KEY should be a non-placeholder value with at least 32 characters"
  else
    info "FLASK_SECRET_KEY is configured"
  fi

  if is_placeholder "$totp_key"; then
    strict_fail_or_warn "TOTP_ENCRYPTION_KEY must be a generated Fernet key"
  elif "$PYTHON_BIN" - "$totp_key" <<'PY' >/dev/null 2>&1
import sys
from cryptography.fernet import Fernet

Fernet(sys.argv[1].encode("utf-8"))
PY
  then
    info "TOTP_ENCRYPTION_KEY is a valid Fernet key"
  else
    strict_fail_or_warn "TOTP_ENCRYPTION_KEY is not a valid Fernet key"
  fi

  if [[ -z "$database_url" ]]; then
    strict_fail_or_warn "DATABASE_URL is missing"
  elif [[ "$database_url" == sqlite://* || "$database_url" == postgresql://* || "$database_url" == postgresql+* || "$database_url" == postgres://* ]]; then
    info "DATABASE_URL uses a supported scheme"
  else
    strict_fail_or_warn "DATABASE_URL should use sqlite or PostgreSQL"
  fi

  normalized_mode="$(printf '%s' "${app_mode:-paper}" | tr '[:upper:]' '[:lower:]')"
  normalized_live="$(printf '%s' "${live_enabled:-false}" | tr '[:upper:]' '[:lower:]')"
  if [[ "$normalized_mode" == "live" && ! "$normalized_live" =~ ^(1|true|yes|on)$ ]]; then
    fail "APP_MODE=live requires ENABLE_LIVE_TRADING=true"
  elif [[ "$normalized_mode" != "live" && "$normalized_live" =~ ^(1|true|yes|on)$ ]]; then
    strict_fail_or_warn "ENABLE_LIVE_TRADING=true while APP_MODE is not live"
  elif [[ "$normalized_mode" == "live" ]]; then
    warn "local env is in live mode; keep real order/withdrawal preview gates enabled unless intentionally testing live flows"
  else
    info "runtime posture is local-safe paper mode"
  fi

  for key in FLASK_APP DEPLOYMENT_TARGET WEB_CONCURRENCY GUNICORN_THREADS ADMIN_USERNAME ADMIN_PASSWORD; do
    value="$(env_value "$key")"
    if [[ -n "$value" ]]; then
      info "$key is set"
    elif [[ "$key" == "ADMIN_PASSWORD" ]]; then
      warn "ADMIN_PASSWORD is empty; first local user registration remains available"
    else
      warn "$key is not set in $ENV_FILE; local scripts/app defaults can still provide it"
    fi
  done
fi

if [[ -x "$PYTHON_BIN" || "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]]; then
  missing_imports="$("$PYTHON_BIN" - <<'PY'
import importlib.util

modules = {
    "cryptography": "cryptography",
    "dotenv": "python-dotenv",
    "flask": "Flask",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "gunicorn": "gunicorn",
    "hyperliquid": "hyperliquid-python-sdk",
    "requests": "requests",
    "sqlalchemy": "SQLAlchemy",
}
missing = [package for module, package in modules.items() if importlib.util.find_spec(module) is None]
print(",".join(missing))
PY
)"
  if [[ -n "$missing_imports" ]]; then
    strict_fail_or_warn "missing runtime package(s): $missing_imports. Run: python -m pip install -r requirements.txt"
  else
    info "required runtime imports are available"
  fi
fi

for path in \
  static/manifest.json \
  static/manifest.webmanifest \
  static/js/sw.js \
  static/js/app-shell.js \
  static/icons/icon-192.png \
  static/icons/icon-512.png \
  static/icons/apple-touch-icon.png; do
  if [[ -f "$path" ]]; then
    info "PWA asset exists: $path"
  else
    fail "PWA asset is missing: $path"
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo "Local setup check failed with $failures error(s) and $warnings warning(s)." >&2
  exit 1
fi

echo "Local setup check passed with $warnings warning(s)."
