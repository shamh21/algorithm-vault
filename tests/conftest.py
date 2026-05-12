from __future__ import annotations

import pytest

from app import create_app
from app.extensions import db


@pytest.fixture()
def app():
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "ENABLE_LIVE_TRADING": True,
            "APP_MODE": "live",
            "DEFAULT_PAPER_BALANCE": 1_000.0,
            "PAPER_BALANCE_MIN": 0.0,
            "PAPER_BALANCE_MAX": 1_000_000.0,
            "MAX_DAILY_LOSS_USDC": 100.0,
            "EXPLICIT_LIVE_CONFIRMED": False,
            "SECONDARY_CONFIRMATION": False,
            "ONE_H10_LIVE_ENABLED": True,
            "SHADOW_LIVE_MIN_TRADES": 1,
            "SHADOW_LIVE_MIN_HOURS": 0.0,
        }
    )
    with app.app_context():
        yield app
        db.session.remove()
        db.drop_all()
