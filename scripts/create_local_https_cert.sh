#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to create a local HTTPS certificate." >&2
  exit 1
fi

CERT_DIR="${LOCAL_HTTPS_CERT_DIR:-$ROOT_DIR/.local-certs}"
CERT_PATH="${LOCAL_HTTPS_CERT:-$CERT_DIR/algorithm-vault-local.crt}"
KEY_PATH="${LOCAL_HTTPS_KEY:-$CERT_DIR/algorithm-vault-local.key}"
CERT_DAYS="${LOCAL_HTTPS_CERT_DAYS:-825}"
LAN_HOST="${LOCAL_HTTPS_HOST:-}"

if [[ -z "$LAN_HOST" ]] && command -v ipconfig >/dev/null 2>&1; then
  LAN_HOST="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi
if [[ -z "$LAN_HOST" ]] && command -v hostname >/dev/null 2>&1; then
  LAN_HOST="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi

mkdir -p "$CERT_DIR" "$(dirname "$CERT_PATH")" "$(dirname "$KEY_PATH")"

SAN_ENTRIES="DNS:localhost,IP:127.0.0.1,IP:::1"
if [[ -n "$LAN_HOST" ]]; then
  if [[ "$LAN_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ || "$LAN_HOST" == *:* ]]; then
    SAN_ENTRIES="$SAN_ENTRIES,IP:$LAN_HOST"
  else
    SAN_ENTRIES="$SAN_ENTRIES,DNS:$LAN_HOST"
  fi
fi

openssl req \
  -x509 \
  -newkey rsa:2048 \
  -nodes \
  -sha256 \
  -days "$CERT_DAYS" \
  -keyout "$KEY_PATH" \
  -out "$CERT_PATH" \
  -subj "/CN=Algorithm Vault Local HTTPS" \
  -addext "subjectAltName=$SAN_ENTRIES"

chmod 600 "$KEY_PATH"
chmod 644 "$CERT_PATH"

echo "Created local HTTPS certificate:"
echo "  Cert: $CERT_PATH"
echo "  Key:  $KEY_PATH"
if [[ -n "$LAN_HOST" ]]; then
  echo "  LAN SAN: $LAN_HOST"
fi
echo
echo "These files are local-only and ignored by git."
echo "Do not use this self-signed certificate for iPhone PWA install testing."
echo "Use https://app.algvault.com or a stable public HTTPS tunnel for iPhone PWA testing."
echo "Run: LOCAL_HTTPS_CERT=\"$CERT_PATH\" LOCAL_HTTPS_KEY=\"$KEY_PATH\" scripts/run_local_https.sh"
