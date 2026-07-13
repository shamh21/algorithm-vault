"""Compatibility module alias for consumer routes."""

from __future__ import annotations

import sys as _sys

from flask import Response, current_app, redirect, render_template, request, url_for

from ..services.seo import (
    PUBLIC_ENDPOINTS,
    PUBLIC_HTML_CACHE_CONTROL,
    SEO_ASSET_CACHE_CONTROL,
    is_public_indexable_path,
    is_seo_asset_path,
    public_navigation,
    public_page,
    robots_txt,
    seo_context,
    should_noindex_path,
    sitemap_xml,
)
from .consumer_parts import legacy as _legacy


_PUBLIC_ROUTES = (
    ("home", "/overview/", "public_overview"),
    ("features", "/features/", "public_features"),
    ("pricing", "/pricing/", "public_pricing"),
    ("mobile", "/mobile/", "public_mobile"),
    ("connectivity", "/connectivity/", "public_connectivity"),
    ("security", "/security/", "public_security"),
)
_PUBLIC_COMPAT_ENDPOINTS = {
    "consumer.public_robots",
    "consumer.public_sitemap",
    "consumer.legacy_mobile_pwa",
    "consumer.legacy_broker_connectivity",
}
_APPLE_TOUCH_ICON = '<link rel="apple-touch-icon" href="/icons/algvault-ios-180.png">'


def _public_page_view(key: str):
    """Build a public marketing view without exposing private app helpers."""

    def view():
        page = public_page(key)
        html = render_template(
            "marketing/page.html",
            page=page,
            seo=seo_context(current_app, endpoint=page.endpoint, path=page.path, authenticated=False),
            public_seo_pages=public_navigation(),
        )
        theme_href = url_for("static", filename="css/algvault-theme.css")
        theme_link = f'<link rel="stylesheet" href="{theme_href}?v=redblack-20260712">'
        return html.replace("</head>", f"  {theme_link}\n  </head>", 1)

    view.__name__ = f"public_{key}_view"
    return view


def _permanent_redirect(endpoint: str):
    def view():
        return redirect(url_for(endpoint), code=308)

    view.__name__ = f"redirect_to_{endpoint.replace('.', '_')}"
    return view


def _robots_view() -> Response:
    return Response(robots_txt(current_app), mimetype="text/plain")


def _sitemap_view() -> Response:
    return Response(sitemap_xml(current_app), mimetype="application/xml")


def _apply_public_and_indexing_headers(response: Response) -> Response:
    """Apply crawlability and cache policy after the app's default private policy."""
    endpoint = request.endpoint
    authenticated = _legacy.current_user() is not None

    if is_public_indexable_path(request.path, endpoint=endpoint, authenticated=authenticated):
        response.headers["Cache-Control"] = PUBLIC_HTML_CACHE_CONTROL
        response.headers.pop("Pragma", None)
        response.headers.pop("Expires", None)
        response.headers.pop("X-Robots-Tag", None)
    elif is_seo_asset_path(request.path):
        response.headers["Cache-Control"] = SEO_ASSET_CACHE_CONTROL
        response.headers.pop("Pragma", None)
        response.headers.pop("Expires", None)
        response.headers.pop("X-Robots-Tag", None)
    elif should_noindex_path(request.path, endpoint=endpoint, authenticated=authenticated):
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Cache-Control"] = "private, no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    if response.mimetype == "text/html" and not response.is_streamed:
        html = response.get_data(as_text=True)
        if "</head>" in html and _APPLE_TOUCH_ICON not in html:
            response.set_data(html.replace("</head>", f"  {_APPLE_TOUCH_ICON}\n  </head>", 1))

    return response


_original_protect_consumer = _legacy._protect_consumer


def _protect_consumer_with_public_pages():
    """Preserve private guards while allowing canonical marketing routes."""
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint in _PUBLIC_COMPAT_ENDPOINTS:
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

_legacy.consumer_bp.add_url_rule(
    "/mobile-pwa/",
    endpoint="legacy_mobile_pwa",
    view_func=_permanent_redirect("consumer.public_mobile"),
    methods=["GET"],
)
_legacy.consumer_bp.add_url_rule(
    "/broker-connectivity/",
    endpoint="legacy_broker_connectivity",
    view_func=_permanent_redirect("consumer.public_connectivity"),
    methods=["GET"],
)
_legacy.consumer_bp.add_url_rule("/robots.txt", endpoint="public_robots", view_func=_robots_view, methods=["GET"])
_legacy.consumer_bp.add_url_rule("/sitemap.xml", endpoint="public_sitemap", view_func=_sitemap_view, methods=["GET"])


@_legacy.consumer_bp.record_once
def _register_public_response_policy(state) -> None:
    state.app.after_request(_apply_public_and_indexing_headers)


consumer_bp = _legacy.consumer_bp
_sys.modules[__name__] = _legacy
