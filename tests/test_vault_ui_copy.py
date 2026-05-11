from __future__ import annotations

from pathlib import Path


def test_vault_duration_copy_updates_notice_and_button() -> None:
    source = Path("static/js/vault.js").read_text()

    assert "durationCopy" in source
    assert "1H10 aims to 10x the user's input amount in 1 hour" in source
    assert "not a guaranteed return" in source
    assert "24h cycle balances momentum" in source
    assert "48h cycle optimizes a balanced multi-factor scope" in source
    assert "7d cycle uses slower trend" in source
    assert "riskNotice.textContent" in source
    assert "oneH10Ack" in source
    assert 'if (selected.value === "1") return "1H10";' in source
    assert "Start ${label} Cycle" in source


def test_vault_template_includes_one_h10_live_acknowledgement() -> None:
    source = Path("templates/vault.html").read_text()

    assert 'name="one_h10_live_ack"' in source
    assert "returns are not guaranteed" in source
