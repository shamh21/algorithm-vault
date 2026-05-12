"""Runtime helpers for settings-backed mode and service access."""

from __future__ import annotations

from flask import current_app

from .models import Setting


def get_service(name: str):
    """Return a registered application service."""

    return current_app.extensions["services"][name]


def get_current_mode() -> str:
    """Return the only supported operating mode for runtime workflows."""

    if Setting.get_json("current_mode") != "live":
        Setting.set_json("current_mode", "live")
    return "live"


def available_modes() -> list[str]:
    """Return UI-selectable modes."""

    return ["live"]


def market_mode_for(mode: str) -> str:
    """Map local runtime modes to market data modes supported by the venue."""

    return "live"
