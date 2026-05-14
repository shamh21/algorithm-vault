#!/usr/bin/env bash
set -euo pipefail

PRODUCTION_URL="${VERCEL_PRODUCTION_URL:-https://algorithm-vault-chi.vercel.app}"
TOKEN="${VERCEL_TOKEN:-}"
SCOPE="${VERCEL_SCOPE:-}"

missing=()
[[ -n "${TOKEN}" ]] || missing+=("VERCEL_TOKEN")
[[ -n "${VERCEL_ORG_ID:-}" ]] || missing+=("VERCEL_ORG_ID")
[[ -n "${VERCEL_PROJECT_ID:-}" ]] || missing+=("VERCEL_PROJECT_ID")

if (( ${#missing[@]} )); then
  printf 'Missing required Vercel deployment environment variables: %s\n' "${missing[*]}" >&2
  printf 'Set them locally or as GitHub Actions secrets for the algorithm-vault-chi Vercel project.\n' >&2
  exit 2
fi

scope_args=()
if [[ -n "${SCOPE}" ]]; then
  scope_args+=(--scope "${SCOPE}")
fi

printf 'Preparing production deployment for %s\n' "${PRODUCTION_URL}"
npx --yes vercel@latest pull --yes --environment=production --token "${TOKEN}" "${scope_args[@]}"
deploy_output="$(npx --yes vercel@latest deploy --prod --token "${TOKEN}" "${scope_args[@]}")"
printf '%s\n' "${deploy_output}"

if ! grep -q "${PRODUCTION_URL}" <<<"${deploy_output}"; then
  printf 'Deployment completed, but Vercel did not echo %s. Confirm the project production domain points to this deployment.\n' "${PRODUCTION_URL}" >&2
fi
