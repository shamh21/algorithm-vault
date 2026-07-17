from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_templates_do_not_render_fabricated_operational_data() -> None:
    content = "\n".join(
        [
            source("templates/marketing/_components.html"),
            source("templates/marketing/page.html"),
        ]
    )
    forbidden = (
        "$128,420.58",
        "Active Strategies",
        "Connected Providers",
        "System Latency",
        "All systems operational",
        "Interactive Brokers",
        "Tradovate",
        '"Binance"',
        '"Kraken"',
        '"OANDA"',
        '"Bybit"',
        "Heartbeat successful",
        "2m ago",
    )
    for value in forbidden:
        assert value not in content


def test_public_templates_keep_server_authoritative_language() -> None:
    components = source("templates/marketing/_components.html")
    page = source("templates/marketing/page.html")
    assert "Server-confirmed only" in components
    assert "Pending until confirmed" in components
    assert "No success state before server confirmation" in page
    assert "does not provide investment advice" in page
    assert "does not publish invented connection counts" in page


def test_mobile_drawer_restores_scroll_focus_and_inert_state() -> None:
    shell = source("static/js/app-shell.js")
    assert 'nav.removeAttribute("inert")' in shell
    assert 'nav.setAttribute("inert", "")' in shell
    assert "window.scrollTo(0, scrollY)" in shell
    assert "returnFocus.focus({ preventScroll: true })" in shell
    assert 'event.key !== "Tab"' in shell
    assert 'event.key === "Escape"' in shell
    assert "scroll-locked" in shell


def test_ios_install_help_is_dismissible_and_hidden_standalone() -> None:
    shell = source("static/js/app-shell.js")
    theme = source("static/css/algvault-theme.css")
    assert "window.navigator.standalone === true" in shell
    assert "av-ios-install-help-dismissed" in shell
    assert "data-ios-install-dismiss" in shell
    assert "@media (display-mode: standalone)" in theme


def test_mobile_inputs_and_safe_areas_are_explicit() -> None:
    theme = source("static/css/algvault-theme.css")
    assert "font-size: 16px !important" in theme
    assert "env(safe-area-inset-top)" in theme
    assert "env(safe-area-inset-bottom)" in theme
    assert "min-height: 100dvh" in theme
    assert "body.scroll-locked" in theme
    assert "overflow-x: hidden" not in theme


def test_manifest_and_service_worker_update_contract() -> None:
    manifest = json.loads(source("static/manifest.json"))
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    assert manifest["scope"] == "/"
    assert manifest["orientation"] == "any"
    assert any("maskable" in icon.get("purpose", "") for icon in manifest["icons"])

    worker = source("static/js/sw.js")
    assert 'const CACHE_VERSION = "algvault-v23-ios-audit"' in worker
    assert "MAX_STATIC_ENTRIES = 80" in worker
    assert 'cache: "reload"' in worker
    install_block = worker.split('self.addEventListener("install"', 1)[1].split(
        'self.addEventListener("activate"', 1
    )[0]
    assert "self.skipWaiting()" not in install_block
    assert 'url.pathname.startsWith("/api/")' in worker
    assert 'fetch(request, { cache: "no-store", credentials: "same-origin" })' in worker
    assert "No cached response is treated as execution success" in worker


def test_public_sections_use_real_links_and_landmarks() -> None:
    page = source("templates/marketing/page.html")
    assert '<nav class="public-section-nav"' in page
    assert 'href="#capabilities"' in page
    assert 'id="mobile-pwa"' in source("templates/marketing/_components.html")
    assert 'id="trust"' in source("templates/marketing/_components.html")
