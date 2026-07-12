"""Compatibility module alias for consumer routes."""

from __future__ import annotations

import sys as _sys

from flask import redirect, url_for

from .consumer_parts import legacy as _legacy


@_legacy.consumer_bp.get("/overview", endpoint="public_overview")
def public_overview():
    """Compatibility endpoint used by shared templates for signed-out users."""
    return redirect(url_for("auth.login"))


consumer_bp = _legacy.consumer_bp
_sys.modules[__name__] = _legacy
