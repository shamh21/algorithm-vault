from __future__ import annotations

from pathlib import Path


def test_vault_script_supports_max_and_settlement_sync_without_duration_copy() -> None:
    source = Path("static/js/vault.js").read_text()

    assert "[data-vault-max]" in source
    assert "data-available-balance" in source or "availableBalance" in source
    assert "syncSettlementToAllocation" in source
    assert "settlementManuallyChanged" in source
    assert "data-provider-toggle" in source
    assert "AbortController" in source
    assert "scheduleRoutingPreview" in source
    assert "durationCopy" not in source
    assert "riskNotice" not in source
    assert "not a guaranteed return" not in source
    assert "24h cycle" not in source
    assert "48h cycle" not in source
    assert "7d cycle" not in source


def test_vault_template_is_minimal_one_h10_flow() -> None:
    source = Path("templates/vault.html").read_text()

    assert 'name="one_h10_live_ack"' in source
    assert 'data-vault-max' in source
    assert 'data-settlement-asset' in source
    assert 'name="lock_duration"' in source
    assert "cycle.duration_hours" in source
    assert source.count('data-vault-routing-form') == 1
    assert source.count('data-settlement-asset') == 1
    assert source.count('type="submit"') == 1
    assert 'url_for(\'consumer.start_cycle\')' in source
    assert 'name="providers"' in source
    assert "Enabled Exchanges" in source
    assert "Routing Preview" in source
    assert "Hyperliquid" not in source
    assert "KuCoin" not in source
    assert "Vault Activity" in source
    assert "Recent Cycles" in source
    assert "Full History" in source
    assert "1H10 Vault Cycle Status" in source
    assert "Next-Move Forecast" in source
    assert "Automation readiness" in source
    for removed_copy in [
        "Capital Routing",
        "Multi-Exchange",
        "Vault Cycle Multi-Exchange",
        "vault-cycle-engine-card",
        "create_vault_cycle",
        "Settlement Currency",
        "Max Leverage",
        "Max Positions",
        "Choose amount, duration, and settlement asset",
        "The profile is selected automatically",
        "Candidate Symbols",
        "Operator Readiness",
        "Providers",
        "Free Margin",
        "Dynamic Budget",
        "ML Readiness",
        "Poll",
        "Rebalance",
        "Current blockers",
        "Vault risk notice",
        "returns are not guaranteed",
        "leveraged",
        "24h",
        "48h",
        "7d",
        "Custom",
    ]:
        assert removed_copy not in source


def test_cycle_detail_forecast_copy_is_probability_oriented() -> None:
    source = Path("templates/cycle_detail.html").read_text()

    assert "Time Horizon Matrix" in source
    assert "Signal Reasoning" in source
    assert "Confidence" in source
    assert "Coherence" in source
    assert "Automation Readiness" in source
    for misleading_copy in [
        "guaranteed prediction",
        "always wins",
        "profit assured",
        "market-beating",
        "risk-free",
        "investment advice",
    ]:
        assert misleading_copy not in source.lower()
