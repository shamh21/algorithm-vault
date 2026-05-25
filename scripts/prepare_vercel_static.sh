#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_DIR="$ROOT_DIR/public"

rm -rf "$PUBLIC_DIR"
mkdir -p "$PUBLIC_DIR"

cp -R "$ROOT_DIR/static" "$PUBLIC_DIR/static"

if [[ -d "$ROOT_DIR/static/icons" ]]; then
  cp -R "$ROOT_DIR/static/icons" "$PUBLIC_DIR/icons"
fi

if [[ -f "$ROOT_DIR/static/manifest.json" ]]; then
  cp "$ROOT_DIR/static/manifest.json" "$PUBLIC_DIR/manifest.json"
fi

if [[ -f "$ROOT_DIR/static/manifest.webmanifest" ]]; then
  cp "$ROOT_DIR/static/manifest.webmanifest" "$PUBLIC_DIR/manifest.webmanifest"
fi

if [[ -f "$ROOT_DIR/static/icons/favicon.ico" ]]; then
  cp "$ROOT_DIR/static/icons/favicon.ico" "$PUBLIC_DIR/favicon.ico"
fi

find "$PUBLIC_DIR" -name ".DS_Store" -delete
