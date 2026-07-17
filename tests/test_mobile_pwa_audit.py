from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def public_marketing_source() -> str:
    paths = ("templates/marketing/_components.html", "templates/marketing/page.html")
    return "\n".join(source(path) for path in paths)


def test_public_templates_do_not_render_fabricated_operational_data() -> None:
    content = public_marketing_source()
    forbidden = {
        "$128,420.58",
        "Active Strategies",
        "All systems operational",
        "Connected Providers",
        "Heartbeat successful",
        "Interactive Brokers",
        "System Latency",
        "Tradovate",
        '"Binance"',
        '"Bybit"',
        '"Kraken"',
        '"OANDA"',
        "2m ago",
    }
    assert all(value not in content for value in forbidden)


def test_public_templates_keep_server_authoritative_language() -> None:
    content = public_marketing_source()
    required = {
        "No success state before server confirmation",
        "Pending until confirmed",
        "Server-confirmed only",
        "does not provide investment advice",
        "does not publish invented connection counts",
    }
    assert all(value in content for value in required)


def test_mobile_drawer_restores_scroll_focus_and_inert_state() -> None:
    shell = source("static/js/app-shell.js")
    required = {
        'event.key !== "Tab"',
        'event.key === "Escape"',
        'nav.removeAttribute("inert")',
        'nav.setAttribute("inert", "")',
        "returnFocus.focus({ preventScroll: true })",
        "scroll-locked",
        "window.scrollTo(0, scrollY)",
    }
    assert all(value in shell for value in required)


def test_ios_install_help_is_dismissible_and_hidden_standalone() -> None:
    shell = source("static/js/app-shell.js")
    theme = source("static/css/algvault-theme.css")
    assert "window.navigator.standalone === true" in shell
    assert "av-ios-install-help-dismissed" in shell
    assert "data-ios-install-dismiss" in shell
    assert "@media (display-mode: standalone)" in theme


def test_mobile_inputs_and_safe_areas_are_explicit() -> None:
    theme = source("static/css/algvault-theme.css")
    required = {
        "body.scroll-locked",
        "env(safe-area-inset-bottom)",
        "env(safe-area-inset-top)",
        "font-size: 16px !important",
        "min-height: 100dvh",
    }
    assert all(value in theme for value in required)
    assert "overflow-x: hidden" not in theme


def test_manifest_and_service_worker_update_contract() -> None:
    manifest = json.loads(source("static/manifest.json"))
    assert manifest["display"] == "standalone"
    assert manifest["orientation"] == "any"
    assert manifest["scope"] == "/"
    assert manifest["start_url"] == "/"
    assert any("maskable" in icon.get("purpose", "") for icon in manifest["icons"])

    worker = source("static/js/sw.js")
    install_block = worker.split('self.addEventListener("install"', 1)[1].split('self.addEventListener("activate"', 1)[0]
    assert "self.skipWaiting()" not in install_block
    assert 'const CACHE_VERSION = "algvault-v23-ios-audit"' in worker
    assert 'fetch(request, { cache: "no-store", credentials: "same-origin" })' in worker
    assert 'url.pathname.startsWith("/api/")' in worker
    assert 'cache: "reload"' in worker
    assert "MAX_STATIC_ENTRIES = 80" in worker
    assert "No cached response is treated as execution success" in worker


def test_public_sections_use_real_links_and_landmarks() -> None:
    page = source("templates/marketing/page.html")
    components = source("templates/marketing/_components.html")
    assert '<nav class="public-section-nav"' in page
    assert 'href="#capabilities"' in page
    assert 'id="supported-connections"' in page
    assert 'id="mobile-pwa"' in components
    assert 'id="trust"' in components


def test_auth_routes_receive_shared_public_navigation() -> None:
    auth = source("app/routes/auth.py")
    assert "@auth_bp.app_context_processor" in auth
    assert '"public_seo_pages": public_navigation()' in auth
