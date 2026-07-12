"""Compatibility module alias for consumer routes."""

from __future__ import annotations

import sys as _sys

from flask import redirect, url_for

from .consumer_parts import legacy as _legacy


def _public_page_redirect():
    """Compatibility target for legacy public-navigation endpoints."""
    return redirect(url_for("auth.login"))


_legacy.consumer_bp.add_url_rule("/overview", endpoint="public_overview", view_func=_public_page_redirect, methods=["GET"])
_legacy.consumer_bp.add_url_rule("/features", endpoint="public_features", view_func=_public_page_redirect, methods=["GET"])
_legacy.consumer_bp.add_url_rule("/pricing", endpoint="public_pricing", view_func=_public_page_redirect, methods=["GET"])
_legacy.consumer_bp.add_url_rule("/mobile", endpoint="public_mobile", view_func=_public_page_redirect, methods=["GET"])
_legacy.consumer_bp.add_url_rule("/connectivity", endpoint="public_connectivity", view_func=_public_page_redirect, methods=["GET"])
_legacy.consumer_bp.add_url_rule("/security", endpoint="public_security", view_func=_public_page_redirect, methods=["GET"])

consumer_bp = _legacy.consumer_bp
_sys.modules[__name__] = _legacy
