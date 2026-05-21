"""Server-side 1inch conversion path for Hyperliquid vault funding."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import WalletAddress
from .wallet_custody import EVM_NETWORKS, EvmWalletAdapter

HYPERLIQUID_BRIDGE2_CONTRACT = "0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7"
KNOWN_EVM_CHAIN_IDS = {
    "ETHEREUM": 1,
    "ARBITRUM": 42161,
    "OPTIMISM": 10,
    "BASE": 8453,
    "POLYGON": 137,
    "AVALANCHE": 43114,
    "BSC": 56,
}
ONEINCH_PROVIDER_KEYS = {"1inch", "oneinch", "one_inch"}


@dataclass(frozen=True, slots=True)
class OneInchRouteCheck:
    ready: bool
    blockers: list[str]
    network: str
    source_wallet_address: str = ""


class OneInchFundingConnector:
    """Converts stablecoins through 1inch and sends USDC to Hyperliquid Bridge2."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        user_id: int,
        wallet_custody: Any | None = None,
        hyperliquid_account_address: str = "",
        http_get: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.user_id = int(user_id)
        self.wallet_custody = wallet_custody
        self.hyperliquid_account_address = str(hyperliquid_account_address or "").strip()
        self._http_get = http_get
        self._sleep = sleep
        self._last_source_wallet_id: int | None = None
        self._last_network = ""

    @classmethod
    def enabled(cls, config: dict[str, Any]) -> bool:
        provider = str(config.get("VAULT_CYCLE_CONVERSION_PROVIDER") or "").strip().lower()
        wallet_kind = str(config.get("WALLET_CONVERSION_PROVIDER_KIND") or "").strip().lower()
        return (
            bool(config.get("VAULT_CYCLE_ONEINCH_AUTO_CONVERSION_ENABLED", False))
            or provider in ONEINCH_PROVIDER_KEYS
            or (bool(config.get("WALLET_CONVERSION_PROVIDER_ENABLED", False)) and wallet_kind in ONEINCH_PROVIDER_KEYS)
        )

    def route_check(
        self,
        *,
        from_asset: str,
        to_asset: str,
        amount: float,
        hyperliquid_account_address: str = "",
    ) -> OneInchRouteCheck:
        network = self.network()
        blockers = self.readiness_blockers(from_asset=from_asset, to_asset=to_asset, network=network)
        source_wallet = None
        if not blockers:
            try:
                source_wallet = self.source_wallet_for(from_asset, network, amount)
            except Exception as exc:  # noqa: BLE001
                blockers.append(str(exc))
        account = str(hyperliquid_account_address or self.hyperliquid_account_address or "").strip()
        if (
            source_wallet is not None
            and bool(self.config.get("VAULT_CYCLE_HYPERLIQUID_BRIDGE_REQUIRES_SOURCE_ACCOUNT", True))
            and account
            and source_wallet.address.lower() != account.lower()
        ):
            blockers.append("hyperliquid_bridge_source_wallet_mismatch")
        return OneInchRouteCheck(
            ready=not blockers,
            blockers=list(dict.fromkeys(blockers)),
            network=network,
            source_wallet_address=source_wallet.address if source_wallet is not None else "",
        )

    def readiness_blockers(self, *, from_asset: str, to_asset: str, network: str) -> list[str]:
        blockers: list[str] = []
        if not self.enabled(self.config):
            blockers.append("oneinch_auto_conversion_disabled")
        if not self.api_key():
            blockers.append("oneinch_api_key_missing")
        if self.chain_id(network) <= 0:
            blockers.append("oneinch_network_unsupported")
        if self.network_key(network) not in EVM_NETWORKS:
            blockers.append("oneinch_network_not_evm")
        if not self.token_identifier(from_asset, network):
            blockers.append(f"oneinch_{self.asset_key(from_asset).lower()}_contract_missing")
        if not self.token_identifier(to_asset, network):
            blockers.append(f"oneinch_{self.asset_key(to_asset).lower()}_contract_missing")
        signing_enabled = bool(self.config.get("WALLET_CONVERSION_SIGNER_TRANSACTIONS_ENABLED", False)) or bool(
            self.config.get("VAULT_CYCLE_SWAP_BRIDGE_SIGNER_TRANSACTIONS_ENABLED", False)
        )
        if not signing_enabled:
            blockers.append("oneinch_signer_transactions_disabled")
        if not bool(self.config.get("WALLET_REAL_CUSTODY_ENABLED", False)):
            blockers.append("wallet_real_custody_disabled")
        try:
            custody = self.wallet_custody_service()
        except Exception as exc:  # noqa: BLE001
            blockers.append(str(exc))
        else:
            supports = getattr(custody, "supports", None)
            if callable(supports) and not supports(from_asset, network):
                blockers.append("oneinch_source_wallet_network_unsupported")
        return list(dict.fromkeys(blockers))

    def supports_route(self, *, from_asset: str, from_network: str, to_asset: str, to_network: str) -> bool:
        if self.network_key(from_network) != self.network_key(to_network):
            return False
        return not self.readiness_blockers(from_asset=from_asset, to_asset=to_asset, network=from_network)

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        if str(mode or "").lower() != "live":
            raise RuntimeError("oneinch_live_mode_required")
        from_asset_key = self.asset_key(from_asset)
        to_asset_key = self.asset_key(to_asset)
        if from_asset_key == to_asset_key:
            return {
                "status": "confirmed",
                "provider_reference": str(client_reference or ""),
                "confirmed_amount": max(0.0, float(amount or 0.0)),
                "source": "same_asset",
            }
        network = self.network()
        requested = max(0.0, float(amount or 0.0))
        if requested <= 0:
            raise RuntimeError("oneinch_amount_required")
        route = self.route_check(from_asset=from_asset_key, to_asset=to_asset_key, amount=requested)
        if not route.ready:
            raise RuntimeError("; ".join(route.blockers))
        source_wallet = self.source_wallet_for(from_asset_key, network, requested)
        self._last_source_wallet_id = source_wallet.id
        self._last_network = network
        amount_units = self.amount_units(requested, from_asset_key, network)
        quote = self._quote(
            chain_id=self.chain_id(network),
            from_asset=from_asset_key,
            to_asset=to_asset_key,
            amount_units=amount_units,
        )
        min_output = self._minimum_output_amount(quote, to_asset_key, network, max_slippage_bps)
        self._approve_if_needed(source_wallet, from_asset_key, network, amount_units)
        swap = self._swap(
            chain_id=self.chain_id(network),
            from_asset=from_asset_key,
            to_asset=to_asset_key,
            amount_units=amount_units,
            from_address=source_wallet.address,
            slippage_bps=max_slippage_bps,
        )
        tx = self._transaction_from_response(swap)
        broadcast = self.wallet_custody_service().sign_and_broadcast_evm_transaction(
            user_id=self.user_id,
            source_wallet_address_id=source_wallet.id,
            network=network,
            transaction=tx,
            mode="live",
        )
        status = self._poll_receipt(broadcast.provider_reference, network)
        if status != "confirmed":
            return {
                "status": "submitted",
                "provider_reference": broadcast.provider_reference,
                "confirmed_amount": 0.0,
                "client_reference": client_reference,
                "raw": {"quote": quote, "swap": swap, "broadcast": broadcast.raw},
            }
        return {
            "status": "confirmed",
            "provider": "1inch",
            "provider_reference": broadcast.provider_reference,
            "confirmed_amount": min_output,
            "client_reference": client_reference,
            "from_network": network,
            "to_network": network,
            "raw": {"quote": quote, "swap": swap, "broadcast": broadcast.raw},
        }

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        if str(mode or "").lower() != "live":
            raise RuntimeError("oneinch_live_mode_required")
        asset_key = self.asset_key(asset)
        if asset_key != "USDC":
            raise RuntimeError("oneinch_hyperliquid_funding_supports_usdc_only")
        network_name = str(network or self._last_network or self.network()).strip()
        requested = max(0.0, float(amount or 0.0))
        source_wallet = self._last_source_wallet(network_name) or self.source_wallet_for(asset_key, network_name, requested)
        account_address = str(destination or self.hyperliquid_account_address or "").strip()
        if bool(self.config.get("VAULT_CYCLE_HYPERLIQUID_BRIDGE_REQUIRES_SOURCE_ACCOUNT", True)) and (
            not account_address or source_wallet.address.lower() != account_address.lower()
        ):
            raise RuntimeError("hyperliquid_bridge_source_wallet_mismatch")
        bridge_address = self.hyperliquid_bridge2_address()
        tx = {
            "to": self.token_identifier(asset_key, network_name),
            "value": "0x0",
            "data": self.erc20_transfer_data(bridge_address, self.amount_units(requested, asset_key, network_name)),
            "chainId": self.chain_id(network_name),
        }
        broadcast = self.wallet_custody_service().sign_and_broadcast_evm_transaction(
            user_id=self.user_id,
            source_wallet_address_id=source_wallet.id,
            network=network_name,
            transaction=tx,
            mode="live",
        )
        return {
            "status": broadcast.status,
            "provider": "1inch",
            "provider_reference": broadcast.provider_reference,
            "confirmed_amount": 0.0,
            "asset": asset_key,
            "network": network_name,
            "destination": bridge_address,
            "hyperliquid_account_address": account_address,
            "memo": memo or "",
            "client_reference": client_reference,
            "raw": broadcast.raw,
        }

    def source_wallet_for(self, asset: str, network: str, amount: float) -> WalletAddress:
        asset_key = self.asset_key(asset)
        network_name = str(network or "").strip()
        adapter = EvmWalletAdapter(self.config)
        candidates = (
            WalletAddress.query.filter_by(user_id=self.user_id, asset=asset_key, network=network_name, status="active")
            .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc(), WalletAddress.id.desc())
            .all()
        )
        for candidate in candidates:
            snapshot = adapter.get_balance(candidate.address, asset_key, network_name)
            if bool(snapshot.checked) and float(snapshot.amount or 0.0) + 1e-12 >= float(amount or 0.0):
                return candidate
        raise RuntimeError(f"oneinch_source_wallet_insufficient:{asset_key}:{network_name}")

    def wallet_custody_service(self) -> Any:
        if self.wallet_custody is not None:
            return self.wallet_custody
        if has_app_context():
            service = current_app.extensions.get("services", {}).get("wallet_custody")
            if service is not None:
                return service
        raise RuntimeError("wallet_custody_service_missing")

    def network(self) -> str:
        return str(self.config.get("VAULT_CYCLE_ONEINCH_NETWORK") or "Arbitrum").strip() or "Arbitrum"

    def api_key(self) -> str:
        return str(self.config.get("VAULT_CYCLE_ONEINCH_API_KEY") or "").strip()

    def api_url(self) -> str:
        return str(self.config.get("VAULT_CYCLE_ONEINCH_API_URL") or "https://api.1inch.dev/swap/v6.1").strip().rstrip("/")

    def hyperliquid_bridge2_address(self) -> str:
        address = str(self.config.get("HYPERLIQUID_BRIDGE2_CONTRACT_ADDRESS") or HYPERLIQUID_BRIDGE2_CONTRACT).strip()
        if not self.is_evm_address(address):
            raise RuntimeError("hyperliquid_bridge2_contract_invalid")
        return address

    def amount_units(self, amount: float, asset: str, network: str) -> int:
        decimals = self.token_decimals(asset, network)
        value = Decimal(str(max(0.0, float(amount or 0.0)))) * (Decimal(10) ** decimals)
        return int(value.quantize(Decimal("1"), rounding=ROUND_DOWN))

    def token_identifier(self, asset: str, network: str) -> str:
        return EvmWalletAdapter(self.config)._token_contract(self.asset_key(asset), network)  # noqa: SLF001

    def token_decimals(self, asset: str, network: str) -> int:
        return EvmWalletAdapter(self.config)._token_decimals(self.asset_key(asset), network)  # noqa: SLF001

    def chain_id(self, network: str) -> int:
        network_key = self.network_key(network)
        mapping = self.config.get("WALLET_EVM_NETWORKS") or {}
        configured: Any = {}
        if isinstance(mapping, dict):
            configured = mapping.get(network_key) or mapping.get(network_key.lower()) or {}
        if isinstance(configured, dict) and configured.get("chain_id"):
            return int(configured["chain_id"])
        return KNOWN_EVM_CHAIN_IDS.get(network_key, 0)

    def _quote(self, *, chain_id: int, from_asset: str, to_asset: str, amount_units: int) -> dict[str, Any]:
        return self._request_json(
            f"/{chain_id}/quote",
            {
                "src": self.token_identifier(from_asset, self.network()),
                "dst": self.token_identifier(to_asset, self.network()),
                "amount": str(amount_units),
            },
        )

    def _swap(
        self,
        *,
        chain_id: int,
        from_asset: str,
        to_asset: str,
        amount_units: int,
        from_address: str,
        slippage_bps: float,
    ) -> dict[str, Any]:
        return self._request_json(
            f"/{chain_id}/swap",
            {
                "src": self.token_identifier(from_asset, self.network()),
                "dst": self.token_identifier(to_asset, self.network()),
                "amount": str(amount_units),
                "from": from_address,
                "slippage": self._slippage_percent(slippage_bps),
                "disableEstimate": "false",
                "allowPartialFill": "false",
            },
        )

    def _approve_if_needed(self, source_wallet: WalletAddress, asset: str, network: str, amount_units: int) -> None:
        chain_id = self.chain_id(network)
        token = self.token_identifier(asset, network)
        allowance = self._request_json(
            f"/{chain_id}/approve/allowance",
            {"tokenAddress": token, "walletAddress": source_wallet.address},
        )
        if self._int_value(allowance.get("allowance")) >= int(amount_units or 0):
            return
        approval = self._request_json(
            f"/{chain_id}/approve/transaction",
            {"tokenAddress": token, "amount": str(amount_units)},
        )
        tx = self._transaction_from_response(approval)
        broadcast = self.wallet_custody_service().sign_and_broadcast_evm_transaction(
            user_id=self.user_id,
            source_wallet_address_id=source_wallet.id,
            network=network,
            transaction=tx,
            mode="live",
        )
        status = self._poll_receipt(broadcast.provider_reference, network)
        if status != "confirmed":
            raise RuntimeError("oneinch_approval_not_confirmed")

    def _request_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.api_url()}{path}?{urllib.parse.urlencode(params)}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key()}",
            "User-Agent": "AlgVaultOneInchFunding/1.0",
        }
        if self._http_get is not None:
            return self._http_get(url, headers)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(2.0, float(self.config.get("VAULT_CYCLE_ONEINCH_API_TIMEOUT_SECONDS", 12.0) or 12.0)),
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"1inch request failed with HTTP {exc.code}: {detail}") from exc
        parsed = json.loads(body or "{}")
        if not isinstance(parsed, dict):
            raise RuntimeError("1inch returned a non-object response")
        return parsed

    def _transaction_from_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = payload.get("tx") if isinstance(payload.get("tx"), dict) else payload
        if not isinstance(raw, dict):
            raise RuntimeError("1inch response did not include a transaction object")
        tx = {key: raw.get(key) for key in ("to", "data", "value", "gas", "gasLimit", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas", "chainId")}
        return {key: value for key, value in tx.items() if value is not None and value != ""}

    def _minimum_output_amount(self, quote: dict[str, Any], asset: str, network: str, max_slippage_bps: float) -> float:
        raw_amount = quote.get("dstAmount") or quote.get("toAmount") or quote.get("amount") or 0
        quoted_units = self._int_value(raw_amount)
        if quoted_units <= 0:
            raise RuntimeError("oneinch_quote_empty")
        slippage_fraction = max(0.0, float(max_slippage_bps or 0.0)) / 10_000.0
        min_units = int(Decimal(quoted_units) * Decimal(str(max(0.0, 1.0 - slippage_fraction))))
        return float(Decimal(min_units) / (Decimal(10) ** self.token_decimals(asset, network)))

    def _poll_receipt(self, tx_hash: str, network: str) -> str:
        if not tx_hash:
            return "failed"
        adapter = EvmWalletAdapter(self.config)
        attempts = max(1, int(float(self.config.get("VAULT_CYCLE_ONEINCH_CONFIRMATION_ATTEMPTS", 3) or 3)))
        delay = max(0.0, float(self.config.get("VAULT_CYCLE_ONEINCH_CONFIRMATION_POLL_SECONDS", 2.0) or 2.0))
        for attempt in range(attempts):
            confirmation = adapter.confirm_transaction(tx_hash, "ETH", network)
            raw = confirmation.get("raw") if isinstance(confirmation, dict) else None
            if isinstance(raw, dict) and raw.get("status") == "0x1":
                return "confirmed"
            if isinstance(raw, dict) and raw.get("status") == "0x0":
                return "failed"
            if attempt < attempts - 1 and delay > 0:
                self._sleep(delay)
        return "submitted"

    def _last_source_wallet(self, network: str) -> WalletAddress | None:
        if not self._last_source_wallet_id:
            return None
        wallet = db.session.get(WalletAddress, int(self._last_source_wallet_id))
        if wallet is None or str(wallet.network or "").strip() != str(network or "").strip():
            return None
        return wallet

    @staticmethod
    def _slippage_percent(slippage_bps: float) -> str:
        return f"{max(0.01, float(slippage_bps or 0.0) / 100.0):.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def _int_value(value: Any) -> int:
        if isinstance(value, int):
            return max(0, value)
        text = str(value or "0").strip()
        if not text:
            return 0
        return max(0, int(text, 16) if text.lower().startswith("0x") else int(float(text)))

    @staticmethod
    def erc20_transfer_data(destination: str, amount_units: int) -> str:
        return "0xa9059cbb" + str(destination or "").lower().replace("0x", "").rjust(64, "0") + hex(int(amount_units or 0))[2:].rjust(64, "0")

    @staticmethod
    def asset_key(asset: str) -> str:
        return "".join(ch for ch in str(asset or "").upper() if ch.isalnum())

    @staticmethod
    def network_key(network: str) -> str:
        return "".join(ch for ch in str(network or "").upper() if ch.isalnum())

    @staticmethod
    def is_evm_address(address: str) -> bool:
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", str(address or "").strip()))
