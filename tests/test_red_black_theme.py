from __future__ import annotations

from pathlib import Path


PUBLIC_PATHS = (
    "/overview/",
    "/features/",
    "/pricing/",
    "/mobile/",
    "/connectivity/",
    "/security/",
)


def test_canonical_algvault_theme_contract() -> None:
    theme = Path("static/css/algvault-theme.css").read_text(encoding="utf-8")

    assert "--page-bg: #050507" in theme
    assert "--surface: #0c0c10" in theme
    assert "--brand-red: #ff3148" in theme
    assert "--brand-purple: #934cff" in theme
    assert "--state-positive: #32d583" in theme
    assert "--state-warning: #f7b955" in theme
    assert "--state-error: #ff4d64" in theme
    assert "--focus-ring:" in theme
    assert "Green remains reserved" in theme
    assert "#f0b90b" not in theme
    assert "rgba(240, 185, 11" not in theme


def test_private_app_shell_loads_canonical_theme() -> None:
    intro = Path("templates/components/algvault_intro.html").read_text(encoding="utf-8")

    assert "css/algvault-theme.css" in intro
    assert "Opening your vault" in intro


def test_public_pages_render_without_auth_and_load_red_black_theme(app) -> None:
    client = app.test_client()

    for path in PUBLIC_PATHS:
        response = client.get(path)
        html = response.get_data(as_text=True)

        assert response.status_code == 200, path
        assert "css/public.css" in html, path
        assert "css/algvault-theme.css" in html, path
        assert "Sign in to continue." not in html, path
        assert '<meta name="robots" content="index, follow, max-image-preview:large">' in html, path


def test_private_consumer_routes_remain_guarded(app) -> None:
    client = app.test_client()

    wallet = client.get("/wallet/")
    vault = client.get("/vault/")

    assert wallet.status_code == 302
    assert vault.status_code == 302
    assert "/login" in wallet.location
    assert "/login" in vault.location


def test_admin_pwa_uses_algvault_red_brand_tokens() -> None:
    css = Path("admin-pwa/src/app/globals.css").read_text(encoding="utf-8")

    assert "--algvault-red: #ff1f36" in css
    assert "--algvault-bg: #030304" in css
    assert "--algvault-purple: #9b4dff" in css
    assert "rgba(34, 197, 94, 0.1)" not in css
    assert "rgba(245, 158, 11, 0.12)" not in css
    assert '[class*="bg-amber-"]' in css


def test_service_worker_precaches_and_renders_audited_shell() -> None:
    worker = Path("static/js/sw.js").read_text(encoding="utf-8")

    assert '"/static/css/algvault-theme.css"' in worker
    assert '"/static/css/public.css"' in worker
    assert '<meta name="theme-color" content="#050507">' in worker
    assert "rgba(255,49,72,.17)" in worker
    assert "rgba(147,76,255,.18)" in worker
    assert "Protected actions remain unavailable while offline" in worker
    assert "#7dd3fc" not in worker
    assert "#101826" not in worker
