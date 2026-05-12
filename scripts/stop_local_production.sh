#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-8765}"
pattern="gunicorn -c deploy/gunicorn.conf.py wsgi:app"

pids=()
while IFS= read -r pid; do
  [[ -n "$pid" ]] && pids+=("$pid")
done < <(
  {
    pgrep -f "$pattern" 2>/dev/null || true
    if command -v lsof >/dev/null 2>&1; then
      lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
    fi
  } | sort -u
)

targets=()
for pid in "${pids[@]}"; do
  [[ -n "$pid" ]] || continue
  command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command_line" == *"gunicorn"* && "$command_line" == *"wsgi:app"* ]]; then
    targets+=("$pid")
  fi
done

if [[ "${#targets[@]}" -eq 0 ]]; then
  echo "No local Algorithm Vault Gunicorn listeners found."
  exit 0
fi

echo "Stopping local Algorithm Vault Gunicorn processes: ${targets[*]}"
kill -TERM "${targets[@]}" 2>/dev/null || true

deadline=$((SECONDS + 15))
while (( SECONDS < deadline )); do
  still_running=()
  for pid in "${targets[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_running+=("$pid")
    fi
  done
  if [[ "${#still_running[@]}" -eq 0 ]]; then
    echo "Algorithm Vault stopped."
    exit 0
  fi
  sleep 1
done

echo "Some processes did not stop after SIGTERM: ${still_running[*]}" >&2
echo "Force stop them with: kill -KILL ${still_running[*]}" >&2
exit 1
