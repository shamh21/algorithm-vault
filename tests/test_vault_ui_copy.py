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
    assert "data-vault-max" in source
    assert "data-settlement-asset" in source
    assert 'name="lock_duration"' in source
    assert "cycle.duration_hours" in source
    assert source.count("data-vault-routing-form") == 1
    assert source.count("data-settlement-asset") == 1
    assert source.count('type="submit"') == 1
    assert "url_for('consumer.start_cycle')" in source
    assert 'name="providers"' in source
    assert "Enabled Exchanges" in source
    assert "Routing Preview" in source
    assert "Hyperliquid" not in source
    assert "data-kucoin-diagnostics" in source
    assert "Server-side KuCoin diagnostics pending." in source
    assert "Vault Activity" in source
    assert "Recent Cycles" in source
    assert "Full History" not in source
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


def test_vault_provider_ui_supports_safe_geo_and_auto_funded_copy() -> None:
    source = Path("static/js/vault.js").read_text()

    assert "ready_auto_funded" in source
    assert "Auto-funded" in source
    assert "geo_restricted" in source
    assert "Provider restricted" in source
    assert "metricLabel" in source
    assert "98.84.12.34" not in source


def test_vault_assets_use_explicit_readiness_cache_busters() -> None:
    vault_source = Path("templates/vault.html").read_text()
    base_source = Path("templates/base.html").read_text()
    sw_source = Path("static/js/sw.js").read_text()

    assert "vault-shell-polish-9" in vault_source
    assert "algvault-vault-shell-polish-10-wallet-merge-1" in base_source
    assert "algvault-v21-vault-shell-polish-9" in sw_source


def test_pwa_shell_keeps_heavy_chart_libraries_off_precache_path() -> None:
    source = Path("static/js/sw.js").read_text()
    app_shell = source.split("const APP_SHELL = [", 1)[1].split("];", 1)[0]

    assert "vendor/chart.umd.min.js" not in app_shell
    assert "vendor/lightweight-charts.standalone.production.js" not in app_shell
    assert "dashboard.js" not in app_shell
    assert "backtests.js" not in app_shell
    assert "mini-charts.js" not in app_shell


def test_one_h10_copy_avoids_profit_claims() -> None:
    selector_source = Path("app/services/vault_selector_parts/legacy.py").read_text()
    route_source = Path("app/routes/consumer_parts/legacy.py").read_text()
    source = selector_source + route_source

    assert "aims to 10x" not in source
    assert "guaranteed" not in source.lower()
    assert "risk-free" not in source.lower()
    assert "high-upside one-hour objective" in source


def test_vault_template_server_renders_provider_readiness_fallbacks() -> None:
    source = Path("templates/vault.html").read_text()

    assert "preview_providers" in source
    assert "data-provider-status>{{ status_label }}" in source
    assert "data-provider-score>{{ metric_label }}" in source
    assert "data-provider-allocation>{{ allocation_label }}" in source
    assert "is-auto-funded" in source
    assert "is-restricted" in source
    assert "funding_detail" in source
