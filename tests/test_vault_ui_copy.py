from __future__ import annotations

from pathlib import Path


def test_vault_duration_copy_updates_notice_and_button() -> None:
    source = Path("static/js/vault.js").read_text()

    assert "durationCopy" in source
    assert "1h cycle uses short-horizon allocation" in source
    assert "24h cycle balances momentum" in source
    assert "48h cycle optimizes a balanced multi-factor scope" in source
    assert "7d cycle uses slower trend" in source
    assert "riskNotice.textContent" in source
    assert "Start ${label} Cycle" in source
