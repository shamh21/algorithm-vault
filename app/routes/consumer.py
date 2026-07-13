"""Compatibility module alias for consumer routes."""

from __future__ import annotations

import sys as _sys

from flask import redirect, render_template, request, url_for

from ..services.seo import PUBLIC_ENDPOINTS, public_page
from .consumer_parts import legacy as _legacy


_PUBLIC_ROUTES = (
    ("home", "/overview/", "public_overview"),
    ("features", "/features/", "public_features"),
    ("pricing", "/pricing/", "public_pricing"),
    ("mobile", "/mobile/", "public_mobile"),
    ("connectivity", "/connectivity/", "public_connectivity"),
    ("security", "/security/", "public_security"),
)


def _public_page_view(key: str):
    """Build a public marketing view without exposing private app helpers."""

    def view():
        html = render_template("marketing/page.html", page=public_page(key))
        theme_href = url_for("static", filename="css/algvault-theme.css")
        theme_link = f'<link rel="stylesheet" href="{theme_href}?v=redblack-20260712">'
        return html.replace("</head>", f"  {theme_link}\n  </head>", 1)

    view.__name__ = f"public_{key}_view"
    return view


_original_protect_consumer = _legacy._protect_consumer


def _protect_consumer_with_public_pages():
    """Preserve private guards while allowing canonical marketing routes."""
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if request.endpoint == "consumer.home" and _legacy.current_user() is None:
        return redirect(url_for("consumer.public_overview"))
    return _original_protect_consumer()


_before_request_funcs = _legacy.consumer_bp.before_request_funcs.get(None, [])
for _index, _handler in enumerate(_before_request_funcs):
    if _handler is _original_protect_consumer:
        _before_request_funcs[_index] = _protect_consumer_with_public_pages
        break


for _key, _path, _endpoint in _PUBLIC_ROUTES:
    _legacy.consumer_bp.add_url_rule(
        _path,
        endpoint=_endpoint,
        view_func=_public_page_view(_key),
        methods=["GET"],
    )


consumer_bp = _legacy.consumer_bp
_sys.modules[__name__] = _legacy
