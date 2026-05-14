from __future__ import annotations

from click.testing import CliRunner
from werkzeug.security import generate_password_hash

from app.cli_commands.legacy import _production_account_readiness_payload, register_cli
from app.extensions import db
from app.models import User, WalletBalance


def _seed_sufyanh() -> User:
    user = User(username="sufyanh", password_hash=generate_password_hash("not-used"), role="user")
    db.session.add(user)
    db.session.flush()
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=25.0, locked_balance=5.0, estimated_usd_value=30.0))
    db.session.commit()
    return user


def test_account_funds_readiness_matches_local_wallet_snapshot(app) -> None:
    with app.app_context():
        _seed_sufyanh()
        service = app.extensions["services"]["wallet_summary"]
        expected_snapshot = {
            "wallet": {
                "portfolio_total_usd": 30.0,
                "balances": [
                    {"asset": "USDC", "available_balance": 25.0, "locked_balance": 5.0, "total_balance": 30.0}
                ],
            }
        }

        result = service.account_funds_readiness(username="sufyanh", expected_snapshot=expected_snapshot)

        assert result["ready"] is True
        assert result["blockers"] == []
        assert result["funded_assets"] == ["USDC"]
        assert result["comparisons"][0]["total_ok"] is True


def test_account_funds_readiness_blocks_when_production_balance_is_below_snapshot(app) -> None:
    with app.app_context():
        _seed_sufyanh()
        service = app.extensions["services"]["wallet_summary"]
        expected_snapshot = {
            "wallet": {
                "portfolio_total_usd": 100.0,
                "balances": [{"asset": "USDC", "available_balance": 100.0, "locked_balance": 0.0}],
            }
        }

        result = service.account_funds_readiness(username="sufyanh", expected_snapshot=expected_snapshot)

        assert result["ready"] is False
        assert "balance_below_expected:USDC" in result["blockers"]
        assert "portfolio_total_usd_below_expected" in result["blockers"]


def test_production_account_readiness_checks_url_and_account_snapshot(app) -> None:
    app.config.update(
        DEPLOYMENT_TARGET="vps",
        PUBLIC_APP_ORIGIN="https://app.algvault.com",
        PUBLIC_API_ORIGIN="https://app.algvault.com",
    )
    with app.app_context():
        _seed_sufyanh()
        expected_snapshot = {
            "wallet": {
                "portfolio_total_usd": 30.0,
                "balances": [{"asset": "USDC", "available_balance": 25.0, "locked_balance": 5.0}],
            }
        }

        result = _production_account_readiness_payload(expected_wallet_snapshot=expected_snapshot)

        assert result["ready"] is True
        assert result["blockers"] == []
        assert result["production_url"]["public_app_origin"] == "https://app.algvault.com"
        assert result["account"]["username"] == "sufyanh"


def test_production_account_readiness_cli_exits_nonzero_without_snapshot(app) -> None:
    register_cli(app)
    app.config.update(
        DEPLOYMENT_TARGET="vps",
        PUBLIC_APP_ORIGIN="https://app.algvault.com",
        PUBLIC_API_ORIGIN="https://app.algvault.com",
    )
    with app.app_context():
        _seed_sufyanh()

    result = CliRunner().invoke(app.cli, ["production-account-readiness", "--username", "sufyanh"])

    assert result.exit_code == 1
    assert "expected_wallet_snapshot_required" in result.output
