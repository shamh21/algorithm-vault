#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASE_URL:-}" ]]; then
  bind="${GUNICORN_BIND:-127.0.0.1:8000}"
  if [[ "$bind" == unix:* ]]; then
    echo "Set BASE_URL when Gunicorn is bound to a Unix socket." >&2
    exit 2
  fi
  host="${bind%:*}"
  port="${bind##*:}"
  if [[ "$host" == "0.0.0.0" || "$host" == "::" || "$host" == "[::]" || -z "$host" ]]; then
    host="127.0.0.1"
  fi
  BASE_URL="http://$host:$port"
fi

curl --fail --silent --show-error "$BASE_URL/healthz" >/dev/null
curl --fail --silent --show-error "$BASE_URL/readyz" >/dev/null
echo "Algorithm Vault health checks passed at $BASE_URL."
