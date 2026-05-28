#!/usr/bin/env python3
"""Verify production wallet-buy readiness without reading secret values."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any
from urllib.request import urlopen

REQUIRED_BUY_ASSETS = {"ETH", "USDC", "USDT"}
REQUIRED_APPLE_PAY_ENV_NAMES = {
    "APPLE_PAY_DIRECT_ENABLED",
    "APPLE_PAY_CRYPTO_SALE_APPROVED",
    "APPLE_PAY_MERCHANT_ID",
    "APPLE_PAY_DOMAIN",
    "APPLE_PAY_DOMAIN_ASSOCIATION",
    "APPLE_PAY_MERCHANT_CERT_PEM",
    "APPLE_PAY_MERCHANT_KEY_PEM",
    "APPLE_PAY_GATEWAY_AUTHORIZE_URL",
    "APPLE_PAY_GATEWAY_API_KEY",
    "APPLE_PAY_GATEWAY_WEBHOOK_SECRET",
    "APPLE_PAY_BUY_ALLOWED_ASSETS_JSON",
    "APPLE_PAY_TREASURY_SOURCE_WALLETS_JSON",
    "APPLE_PAY_TREASURY_SIGNER_URL",
    "APPLE_PAY_TREASURY_SIGNER_TOKEN",
    "WORKER_APPLE_PAY_FULFILLMENT_ENABLED",
}
REQUIRED_CARD_ENV_NAMES = {
    "CARD_BUY_ENABLED",
    "CARD_GATEWAY_TOKENIZATION_URL",
    "CARD_GATEWAY_AUTHORIZE_URL",
    "CARD_GATEWAY_API_KEY",
    "CARD_GATEWAY_WEBHOOK_SECRET",
    "CARD_GATEWAY_PUBLIC_CONFIG_JSON",
    "APPLE_PAY_BUY_ALLOWED_ASSETS_JSON",
    "APPLE_PAY_TREASURY_SOURCE_WALLETS_JSON",
    "APPLE_PAY_TREASURY_FEE_ADDRESS",
    "APPLE_PAY_TREASURY_SIGNER_URL",
    "APPLE_PAY_TREASURY_SIGNER_TOKEN",
    "WORKER_APPLE_PAY_FULFILLMENT_ENABLED",
}
APPLE_DOMAIN_ASSOCIATION_PATH = "/.well-known/apple-developer-merchantid-domain-association"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="https://algvault.app", help="Production origin to verify.")
    parser.add_argument("--scope", default="sufyan-h-s-projects", help="Vercel scope/team slug.")
    parser.add_argument(
        "--mode",
        choices=["card", "apple-pay", "both"],
        default="card",
        help="Verify card buys, direct Apple Pay buys, or both.",
    )
    parser.add_argument("--skip-vercel-env", action="store_true", help="Skip Vercel env-name inspection.")
    return parser.parse_args()


def fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=20) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return payload


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=20) as response:  # noqa: S310
        return response.read().decode("utf-8")


def vercel_env_names(*, scope: str) -> set[str]:
    proc = subprocess.run(
        ["npx", "--yes", "vercel@latest", "env", "ls", "production", "--scope", scope],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "vercel env ls failed").strip())
    names: set[str] = set()
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith((">", "name ", "Common next commands", "-", "Retrieving project")):
            continue
        first = line.split(maxsplit=1)[0]
        if first.replace("_", "").isalnum():
            names.add(first)
    return names


def verify_buy_payload(
    *,
    label: str,
    payload: dict[str, Any],
    failures: list[str],
    require_treasury_fee_address: bool = False,
) -> None:
    if payload.get("ready") is not True:
        failures.append(f"{label}.ready is not true")
    if payload.get("enabled") is not True:
        failures.append(f"{label}.enabled is not true")
    if float(payload.get("treasury_fee_bps") or 0.0) != 250.0:
        failures.append(f"{label}.treasury_fee_bps is not 250")
    if payload.get("treasury_fee_asset") not in {None, "ETH"}:
        failures.append(f"{label}.treasury_fee_asset is not ETH")
    if require_treasury_fee_address and payload.get("treasury_fee_address_configured") is False:
        failures.append(f"{label} treasury fee address is not configured")
    fulfillment_worker = payload.get("fulfillment_worker") if isinstance(payload.get("fulfillment_worker"), dict) else {}
    if fulfillment_worker and fulfillment_worker.get("required") is True and fulfillment_worker.get("recent") is not True:
        failures.append(f"{label} fulfillment worker heartbeat is not recent")
    allowed_assets = payload.get("allowed_assets") if isinstance(payload.get("allowed_assets"), dict) else {}
    configured_assets = {str(asset).upper() for asset in allowed_assets}
    if not REQUIRED_BUY_ASSETS.issubset(configured_assets):
        failures.append(f"{label}.allowed_assets does not include ETH, USDC, and USDT")
    unexpected_assets = configured_assets - REQUIRED_BUY_ASSETS
    if unexpected_assets:
        failures.append(f"{label}.allowed_assets includes non-v1 assets: " + ", ".join(sorted(unexpected_assets)))
    for asset, networks in allowed_assets.items():
        if not networks:
            failures.append(f"{label}.allowed_assets has no networks for {asset}")


def required_env_names(mode: str) -> set[str]:
    if mode == "apple-pay":
        return set(REQUIRED_APPLE_PAY_ENV_NAMES)
    if mode == "both":
        return set(REQUIRED_APPLE_PAY_ENV_NAMES | REQUIRED_CARD_ENV_NAMES)
    return set(REQUIRED_CARD_ENV_NAMES)


def _internal_treasury_signer_active(apple_pay: dict[str, Any], mode: str) -> bool:
    payload = apple_pay
    if mode == "card":
        card = apple_pay.get("card_buy") if isinstance(apple_pay.get("card_buy"), dict) else {}
        payload = card
    return payload.get("treasury_signer_provider") == "internal_mpc" and payload.get("treasury_signer_configured") is True


def _list_value(payload: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def main() -> int:
    args = parse_args()
    origin = str(args.origin or "").rstrip("/")
    failures: list[str] = []

    readyz = fetch_json(f"{origin}/readyz")
    if readyz.get("ok") is not True:
        failures.append("/readyz is not ok")

    ops = fetch_json(f"{origin}/ops/status?detail=full")
    apple_pay = ops.get("apple_pay_purchase") if isinstance(ops.get("apple_pay_purchase"), dict) else {}
    if args.mode in {"apple-pay", "both"}:
        verify_buy_payload(label="apple_pay_purchase", payload=apple_pay, failures=failures)
        if apple_pay.get("domain_association_configured") is not True:
            failures.append("apple_pay_purchase domain association is not configured")
        supported_networks = apple_pay.get("supported_networks") if isinstance(apple_pay.get("supported_networks"), list) else []
        if "mastercard" in supported_networks and "masterCard" not in supported_networks:
            failures.append("apple_pay_purchase.supported_networks must preserve masterCard casing")
        try:
            if not fetch_text(f"{origin}{APPLE_DOMAIN_ASSOCIATION_PATH}").strip():
                failures.append("Apple Pay domain association file is empty")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Apple Pay domain association file is not reachable: {exc}")

    if args.mode in {"card", "both"}:
        card = (apple_pay.get("card_buy") or {}) if isinstance(apple_pay, dict) else {}
        verify_buy_payload(label="card_buy", payload=card, failures=failures, require_treasury_fee_address=True)

    workers = ops.get("workers") if isinstance(ops.get("workers"), dict) else {}
    expected_leases = _list_value(workers, "expected_leases", "expected_lease_names")
    if "apple_pay_fulfillment:singleton" not in expected_leases:
        failures.append("apple_pay_fulfillment worker lease is not expected")
    missing_leases = _list_value(workers, "missing_expected_leases", "missing_expected_lease_names")
    stale_leases = _list_value(workers, "stale_expected_leases")
    if "apple_pay_fulfillment:singleton" in missing_leases:
        failures.append("apple_pay_fulfillment worker lease is missing")
    if "apple_pay_fulfillment:singleton" in stale_leases:
        failures.append("apple_pay_fulfillment worker lease is stale")

    if not args.skip_vercel_env:
        names = vercel_env_names(scope=args.scope)
        required = required_env_names(args.mode)
        if args.mode in {"apple-pay", "both"} and _internal_treasury_signer_active(apple_pay, "apple-pay"):
            required -= {"APPLE_PAY_TREASURY_SIGNER_URL", "APPLE_PAY_TREASURY_SIGNER_TOKEN"}
        if args.mode in {"card", "both"} and _internal_treasury_signer_active(apple_pay, "card"):
            required -= {"APPLE_PAY_TREASURY_SIGNER_URL", "APPLE_PAY_TREASURY_SIGNER_TOKEN"}
        missing_env = sorted(required - names)
        if missing_env:
            failures.append("Vercel production env is missing: " + ", ".join(missing_env))

    if failures:
        print(json.dumps({"ok": False, "failures": failures}, indent=2))
        return 1
    print(json.dumps({"ok": True, "origin": origin, "mode": args.mode, "apple_pay_purchase": apple_pay}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
