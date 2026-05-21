"""App-wide server-side swap/bridge execution for Vault Cycle funding."""

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

from ..models import WalletAddress
from .wallet_custody import BroadcastResult, EvmWalletAdapter

NATIVE_EVM_TOKEN = "0x0000000000000000000000000000000000000000"
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


@dataclass(frozen=True, slots=True)
class SwapBridgeQuote:
    provider: str
    from_asset: str
    from_network: str
    from_chain_id: int
    from_amount: float
    from_amount_units: int
    to_asset: str
    to_network: str
    to_chain_id: int
    to_amount_estimate: float
    to_amount_min: float
    approval_address: str
    transaction_request: dict[str, Any]
    tool: str
    provider_reference: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SwapBridgeStatus:
    status: str
    provider_reference: str
    confirmed_amount: float
    receipt_hash: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SwapBridgeExecution:
    status: str
    quote: SwapBridgeQuote
    route_tx: BroadcastResult | None
    approval_tx: BroadcastResult | None
    provider_status: SwapBridgeStatus


class VaultCycleSwapBridgeService:
    """Executes the app-wide LI.FI EVM swap/bridge route from server-held wallets."""

    def __init__(
        self,
        config: dict[str, Any],
        wallet_custody: Any | None = None,
        *,
        http_get: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.wallet_custody = wallet_custody
        self._http_get = http_get
        self._sleep = sleep

    def readiness_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not bool(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_ENABLED", False)):
            blockers.append("VAULT_CYCLE_SWAP_BRIDGE_ENABLED is disabled")
        if str(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_PROVIDER", "lifi") or "").strip().lower() != "lifi":
            blockers.append("VAULT_CYCLE_SWAP_BRIDGE_PROVIDER must be lifi")
        if not str(self.config.get("LIFI_API_URL", "") or "").strip():
            blockers.append("LIFI_API_URL is not configured")
        signing_enabled = bool(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_SIGNER_TRANSACTIONS_ENABLED", False)) or bool(
            self.config.get("WALLET_CONVERSION_SIGNER_TRANSACTIONS_ENABLED", False)
        )
        if not signing_enabled:
            blockers.append("server-side conversion transaction signing is disabled")
        if not bool(self.config.get("WALLET_REAL_CUSTODY_ENABLED", False)):
            blockers.append("WALLET_REAL_CUSTODY_ENABLED is disabled")
        return blockers

    def supports_route(self, *, from_asset: str, from_network: str, to_asset: str, to_network: str) -> bool:
        if self.readiness_blockers():
            return False
        return (
            self.chain_id(from_network) > 0
            and self.chain_id(to_network) > 0
            and bool(self.token_identifier(from_asset, from_network))
            and bool(self.token_identifier(to_asset, to_network))
            and bool(EvmWalletAdapter(self.config)._rpc_url(from_network))  # noqa: SLF001
        )

    def source_wallet_for(self, *, user_id: int, asset: str, network: str, amount: float) -> WalletAddress:
        asset_key = self.asset_key(asset)
        network_name = str(network or "").strip()
        adapter = EvmWalletAdapter(self.config)
        candidates = (
            WalletAddress.query.filter_by(user_id=int(user_id), asset=asset_key, network=network_name, status="active")
            .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc(), WalletAddress.id.desc())
            .all()
        )
        for candidate in candidates:
            snapshot = adapter.get_balance(candidate.address, asset_key, network_name)
            if bool(snapshot.checked) and float(snapshot.amount or 0.0) + 1e-12 >= float(amount or 0.0):
                return candidate
        raise RuntimeError(f"app_wide_source_wallet_insufficient:{asset_key}:{network_name}")

    def quote(
        self,
        *,
        from_asset: str,
        from_network: str,
        to_asset: str,
        to_network: str,
        amount: float,
        from_address: str,
        to_address: str,
    ) -> SwapBridgeQuote:
        blockers = self.readiness_blockers()
        if blockers:
            raise RuntimeError("; ".join(blockers))
        if not self.supports_route(from_asset=from_asset, from_network=from_network, to_asset=to_asset, to_network=to_network):
            raise RuntimeError("app_wide_conversion_route_unsupported")
        if not self._is_evm_address(from_address) or not self._is_evm_address(to_address):
            raise RuntimeError("app-wide conversion route requires EVM source and destination addresses")

        from_asset_key = self.asset_key(from_asset)
        to_asset_key = self.asset_key(to_asset)
        from_units = self.amount_units(amount, from_asset_key, from_network)
        payload = self._request_json(
            "/quote",
            {
                "fromChain": str(self.chain_id(from_network)),
                "toChain": str(self.chain_id(to_network)),
                "fromToken": self.token_identifier(from_asset_key, from_network),
                "toToken": self.token_identifier(to_asset_key, to_network),
                "fromAmount": str(from_units),
                "fromAddress": from_address,
                "toAddress": to_address,
                "slippage": str(max(0.0001, float(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_SLIPPAGE", 0.003) or 0.003))),
                "order": str(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_ORDER", "FASTEST") or "FASTEST"),
            },
        )
        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        estimate = payload.get("estimate") if isinstance(payload.get("estimate"), dict) else {}
        transaction = payload.get("transactionRequest") if isinstance(payload.get("transactionRequest"), dict) else {}
        if not transaction:
            raise RuntimeError("LI.FI quote did not include a transactionRequest")
        to_decimals = self.token_decimals(to_asset_key, to_network)
        to_amount_units = self._int_value(estimate.get("toAmount") or estimate.get("toAmountMin") or 0)
        to_amount_min_units = self._int_value(estimate.get("toAmountMin") or estimate.get("toAmount") or 0)
        return SwapBridgeQuote(
            provider="lifi",
            from_asset=from_asset_key,
            from_network=str(from_network or "").strip(),
            from_chain_id=self.chain_id(from_network),
            from_amount=float(amount or 0.0),
            from_amount_units=self._int_value(action.get("fromAmount") or from_units),
            to_asset=to_asset_key,
            to_network=str(to_network or "").strip(),
            to_chain_id=self.chain_id(to_network),
            to_amount_estimate=self.units_amount(to_amount_units, to_decimals),
            to_amount_min=self.units_amount(to_amount_min_units, to_decimals),
            approval_address=str(estimate.get("approvalAddress") or payload.get("approvalAddress") or "").strip(),
            transaction_request=transaction,
            tool=str(payload.get("tool") or estimate.get("tool") or "").strip(),
            provider_reference=str(payload.get("id") or payload.get("integrator") or ""),
            raw=payload,
        )

    def execute(
        self,
        *,
        user_id: int,
        from_wallet: WalletAddress,
        from_asset: str,
        from_network: str,
        to_asset: str,
        to_network: str,
        amount: float,
        to_address: str,
        mode: str = "live",
    ) -> SwapBridgeExecution:
        quote = self.quote(
            from_asset=from_asset,
            from_network=from_network,
            to_asset=to_asset,
            to_network=to_network,
            amount=amount,
            from_address=from_wallet.address,
            to_address=to_address,
        )
        approval_tx = self._submit_approval_if_needed(user_id=user_id, from_wallet=from_wallet, quote=quote, mode=mode)
        if approval_tx is not None:
            approval_status = self.poll_evm_receipt(approval_tx.provider_reference, quote.from_network)
            if approval_status.status != "confirmed":
                return SwapBridgeExecution("pending_approval", quote, None, approval_tx, approval_status)
        route_tx = self.wallet_custody_service().sign_and_broadcast_evm_transaction(
            user_id=user_id,
            source_wallet_address_id=from_wallet.id,
            network=quote.from_network,
            transaction=quote.transaction_request,
            mode=mode,
        )
        provider_status = self.poll_status(quote, route_tx.provider_reference)
        return SwapBridgeExecution(provider_status.status, quote, route_tx, approval_tx, provider_status)

    def poll_status(self, quote: SwapBridgeQuote, tx_hash: str) -> SwapBridgeStatus:
        attempts = max(1, int(float(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_STATUS_POLL_ATTEMPTS", 1) or 1)))
        delay = max(0.0, float(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_STATUS_POLL_SECONDS", 5.0) or 0.0))
        status = SwapBridgeStatus("submitted", tx_hash, 0.0, "", {})
        for attempt in range(attempts):
            status = self.status(quote=quote, tx_hash=tx_hash)
            if status.status in {"confirmed", "failed"}:
                return status
            if attempt < attempts - 1 and delay > 0:
                self._sleep(delay)
        return status

    def status(self, *, quote: SwapBridgeQuote, tx_hash: str) -> SwapBridgeStatus:
        if not tx_hash:
            return SwapBridgeStatus("failed", "", 0.0, "", {"error": "missing transaction hash"})
        receipt_status = self.poll_evm_receipt(tx_hash, quote.from_network, attempts=1)
        try:
            params = {
                "txHash": tx_hash,
                "fromChain": str(quote.from_chain_id),
                "toChain": str(quote.to_chain_id),
            }
            if quote.tool:
                params["bridge"] = quote.tool
            payload = self._request_json("/status", params)
        except Exception as exc:  # noqa: BLE001
            return SwapBridgeStatus(
                receipt_status.status,
                tx_hash,
                0.0,
                tx_hash if receipt_status.status == "confirmed" else "",
                {"status_error": str(exc)},
            )
        raw_status = str(payload.get("status") or payload.get("substatus") or "").upper()
        if raw_status in {"DONE", "COMPLETED", "SUCCESS", "CONFIRMED"}:
            receiving = payload.get("receiving") if isinstance(payload.get("receiving"), dict) else {}
            to_amount = self.units_amount(
                self._int_value(receiving.get("amount") or 0),
                self.token_decimals(quote.to_asset, quote.to_network),
            )
            return SwapBridgeStatus(
                "confirmed",
                tx_hash,
                to_amount or quote.to_amount_min or quote.to_amount_estimate,
                str(receiving.get("txHash") or tx_hash),
                payload,
            )
        if raw_status in {"FAILED", "INVALID", "NOT_FOUND"}:
            return SwapBridgeStatus("failed", tx_hash, 0.0, "", payload)
        return SwapBridgeStatus("submitted", tx_hash, 0.0, tx_hash if receipt_status.status == "confirmed" else "", payload)

    def poll_evm_receipt(self, tx_hash: str, network: str, *, attempts: int | None = None) -> SwapBridgeStatus:
        max_attempts = max(1, int(attempts or float(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_RECEIPT_POLL_ATTEMPTS", 1) or 1)))
        delay = max(0.0, float(self.config.get("VAULT_CYCLE_SWAP_BRIDGE_RECEIPT_POLL_SECONDS", 2.0) or 0.0))
        adapter = EvmWalletAdapter(self.config)
        raw: dict[str, Any] = {}
        for attempt in range(max_attempts):
            confirmation = adapter.confirm_transaction(tx_hash, "ETH", network)
            raw_confirmation = confirmation.get("raw") if isinstance(confirmation, dict) else None
            raw = raw_confirmation if isinstance(raw_confirmation, dict) else {}
            if raw.get("status") == "0x1":
                return SwapBridgeStatus("confirmed", tx_hash, 0.0, tx_hash, raw)
            if raw.get("status") == "0x0":
                return SwapBridgeStatus("failed", tx_hash, 0.0, tx_hash, raw)
            if attempt < max_attempts - 1 and delay > 0:
                self._sleep(delay)
        return SwapBridgeStatus("submitted", tx_hash, 0.0, "", raw)

    def wallet_custody_service(self) -> Any:
        if self.wallet_custody is not None:
            return self.wallet_custody
        if has_app_context():
            service = current_app.extensions.get("services", {}).get("wallet_custody")
            if service is not None:
                return service
        raise RuntimeError("wallet custody service is not available")

    def hyperliquid_bridge2_address(self) -> str:
        address = str(self.config.get("HYPERLIQUID_BRIDGE2_CONTRACT_ADDRESS") or HYPERLIQUID_BRIDGE2_CONTRACT).strip()
        if not self._is_evm_address(address):
            raise RuntimeError("HYPERLIQUID_BRIDGE2_CONTRACT_ADDRESS is invalid")
        return address

    def amount_units(self, amount: float, asset: str, network: str) -> int:
        decimals = self.token_decimals(asset, network)
        value = Decimal(str(max(0.0, float(amount or 0.0)))) * (Decimal(10) ** decimals)
        return int(value.quantize(Decimal("1"), rounding=ROUND_DOWN))

    @staticmethod
    def units_amount(units: int, decimals: int) -> float:
        if int(units or 0) <= 0:
            return 0.0
        return float(Decimal(int(units)) / (Decimal(10) ** int(decimals or 0)))

    def token_identifier(self, asset: str, network: str) -> str:
        asset_key = self.asset_key(asset)
        if asset_key == "ETH":
            return NATIVE_EVM_TOKEN
        return EvmWalletAdapter(self.config)._token_contract(asset_key, network)  # noqa: SLF001

    def token_decimals(self, asset: str, network: str) -> int:
        asset_key = self.asset_key(asset)
        if asset_key == "ETH":
            return 18
        return EvmWalletAdapter(self.config)._token_decimals(asset_key, network)  # noqa: SLF001

    def chain_id(self, network: str) -> int:
        network_key = self.network_key(network)
        mapping = self.config.get("WALLET_EVM_NETWORKS") or {}
        configured: Any = {}
        if isinstance(mapping, dict):
            configured = mapping.get(network_key) or mapping.get(network_key.lower()) or {}
        if isinstance(configured, dict) and configured.get("chain_id"):
            return int(configured["chain_id"])
        return KNOWN_EVM_CHAIN_IDS.get(network_key, 0)

    def erc20_transfer_transaction(self, *, asset: str, network: str, destination: str, amount: float) -> dict[str, Any]:
        return {
            "to": self.token_identifier(asset, network),
            "value": "0x0",
            "data": self.erc20_transfer_data(destination, self.amount_units(amount, asset, network)),
            "chainId": self.chain_id(network),
        }

    @staticmethod
    def erc20_transfer_data(destination: str, amount_units: int) -> str:
        return "0xa9059cbb" + str(destination or "").lower().replace("0x", "").rjust(64, "0") + hex(int(amount_units or 0))[2:].rjust(64, "0")

    @staticmethod
    def asset_key(asset: str) -> str:
        return "".join(ch for ch in str(asset or "").upper() if ch.isalnum())

    @staticmethod
    def network_key(network: str) -> str:
        return "".join(ch for ch in str(network or "").upper() if ch.isalnum())

    def _submit_approval_if_needed(
        self,
        *,
        user_id: int,
        from_wallet: WalletAddress,
        quote: SwapBridgeQuote,
        mode: str,
    ) -> BroadcastResult | None:
        if quote.from_asset == "ETH" or not quote.approval_address:
            return None
        if not self._is_evm_address(quote.approval_address):
            raise RuntimeError("LI.FI quote returned an invalid approval address")
        adapter = EvmWalletAdapter(self.config)
        allowance = adapter.erc20_allowance(from_wallet.address, quote.from_asset, quote.from_network, quote.approval_address)
        if allowance >= int(quote.from_amount_units or 0):
            return None
        approval_tx = adapter.approval_transaction(quote.from_asset, quote.from_network, quote.approval_address, quote.from_amount_units)
        return self.wallet_custody_service().sign_and_broadcast_evm_transaction(
            user_id=user_id,
            source_wallet_address_id=from_wallet.id,
            network=quote.from_network,
            transaction=approval_tx,
            mode=mode,
        )

    def _request_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        base_url = str(self.config.get("LIFI_API_URL", "https://li.quest/v1") or "https://li.quest/v1").rstrip("/")
        url = f"{base_url}{path}?{urllib.parse.urlencode(params)}"
        headers = {"Accept": "application/json", "User-Agent": "AlgVaultVaultCycle/1.0"}
        api_key = str(self.config.get("LIFI_API_KEY", "") or "").strip()
        if api_key:
            headers["x-lifi-api-key"] = api_key
        if self._http_get is not None:
            return self._http_get(url, headers)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(2.0, float(self.config.get("LIFI_API_TIMEOUT_SECONDS", 12.0) or 12.0)),
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LI.FI request failed with HTTP {exc.code}: {detail}") from exc
        parsed = json.loads(body or "{}")
        if not isinstance(parsed, dict):
            raise RuntimeError("LI.FI returned a non-object response")
        return parsed

    @staticmethod
    def _is_evm_address(address: str) -> bool:
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", str(address or "").strip()))

    @staticmethod
    def _int_value(value: Any) -> int:
        if isinstance(value, int):
            return max(0, value)
        text = str(value or "0").strip()
        if not text:
            return 0
        return max(0, int(text, 16) if text.lower().startswith("0x") else int(float(text)))
