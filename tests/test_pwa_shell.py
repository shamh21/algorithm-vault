from __future__ import annotations

import json
import struct
from pathlib import Path


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def test_manifest_contains_dark_standalone_pwa_contract(app) -> None:
    response = app.test_client().get("/manifest.json")

    assert response.status_code == 200
    assert "must-revalidate" in response.headers["Cache-Control"]

    manifest = response.get_json()
    assert manifest["name"] == "AlgVault"
    assert manifest["short_name"] == "AlgVault"
    assert "vault" in manifest["description"].lower()
    assert manifest["start_url"] == "/login"
    assert manifest["scope"] == "/"
    assert manifest["display"] == "standalone"
    assert manifest["background_color"] == "#050607"
    assert manifest["theme_color"] == "#050607"
    assert "finance" in manifest["categories"]

    icons = {icon["src"]: icon for icon in manifest["icons"]}
    assert icons["/icons/algvault-mascot-192.png"]["sizes"] == "192x192"
    assert icons["/icons/algvault-mascot-192.png"]["purpose"] == "any"
    assert icons["/icons/algvault-maskable-192.png"]["sizes"] == "192x192"
    assert icons["/icons/algvault-maskable-192.png"]["purpose"] == "maskable"
    assert icons["/icons/algvault-mascot-512.png"]["sizes"] == "512x512"
    assert icons["/icons/algvault-maskable-512.png"]["sizes"] == "512x512"
    assert icons["/icons/algvault-mascot-180.png"]["sizes"] == "180x180"
    assert "/admin/panic/" not in {shortcut["url"] for shortcut in manifest["shortcuts"]}
    assert "/login" in {shortcut["url"] for shortcut in manifest["shortcuts"]}
    assert "/convert/" in {shortcut["url"] for shortcut in manifest["shortcuts"]}


def test_manifest_files_stay_in_sync() -> None:
    root = Path("static")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    webmanifest = json.loads((root / "manifest.webmanifest").read_text(encoding="utf-8"))

    assert manifest == webmanifest


def test_webmanifest_route_matches_manifest_json(app) -> None:
    client = app.test_client()
    manifest = client.get("/manifest.json")
    webmanifest = client.get("/manifest.webmanifest")

    assert webmanifest.status_code == 200
    assert "must-revalidate" in webmanifest.headers["Cache-Control"]
    assert webmanifest.get_json() == manifest.get_json()


def test_base_template_has_dark_ios_pwa_metadata(app) -> None:
    response = app.test_client().get("/login")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' in html
    assert '<meta name="theme-color" content="#050607">' in html
    assert '<meta name="color-scheme" content="dark">' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in html
    assert '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">' in html
    assert '<link rel="icon" href="/icons/favicon.ico" sizes="any">' in html
    assert '<link rel="mask-icon" href="/icons/algvault-mask-icon.svg" color="#ff1f36">' in html
    assert '<link rel="apple-touch-icon" sizes="180x180" href="/icons/algvault-mascot-180.png">' in html
    assert "data-theme-toggle" not in html
    assert "av-color-theme" in html
    assert 'data-component="AlgVaultLaunchAnimation"' in html
    settings_template = Path("templates/settings.html").read_text(encoding="utf-8")
    assert "data-theme-toggle" in settings_template


def test_login_shell_shows_redacted_operations_snapshot(app) -> None:
    response = app.test_client().get("/login?next=/wallet/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Continue to Wallet" in html
    assert "Production Status" in html
    assert "Database" in html
    assert "Withdrawals" in html
    assert "Treasury" in html
    assert 'autocapitalize="none"' in html
    assert 'maxlength="6"' in html
    assert "WALLET_MPC_SIGNER_TOKEN" not in html
    assert "TREASURY_ENCRYPTION_KEY" not in html


def test_base_template_uses_command_center_bottom_nav() -> None:
    html = Path("templates/base.html").read_text(encoding="utf-8")
    for label in ("Dashboard", "Wallet", "Convert", "Vault", "Settings"):
        assert f'<span class="nav-item-label">{label}</span>' in html
    assert "bottom_dashboard_href = url_for('dashboard.index') if admin_authenticated" in html
    assert "request.endpoint.startswith('dashboard')" in html
    assert 'data-bottom-nav-section="wallet"' in html
    assert 'data-bottom-nav-section="convert"' in html
    assert 'data-bottom-nav-section="activity"' not in html
    assert 'data-bottom-nav-section="settings"' in html


def test_dark_pwa_icon_and_crypto_symbol_sources_exist() -> None:
    icon_source = Path("static/icons/algvault-icon.svg").read_text(encoding="utf-8")
    crypto_icon = Path("templates/components/crypto_icon.html").read_text(encoding="utf-8")

    assert "happy red algvault mascot" in icon_source.lower()
    assert "holding a dark vault" in icon_source.lower()
    for symbol in ("BTC", "ETH", "ALGO", "USDT", "USDC", "SOL", "XRP"):
        assert symbol in crypto_icon


def test_mascot_pwa_icon_exports_have_expected_dimensions() -> None:
    icon_dir = Path("static/icons")
    for size in (16, 32, 48, 64, 72, 96, 128, 144, 152, 167, 180, 192, 256, 384, 512):
        assert _png_size(icon_dir / f"algvault-mascot-{size}.png") == (size, size)
    assert _png_size(icon_dir / "algvault-maskable-192.png") == (192, 192)
    assert _png_size(icon_dir / "algvault-maskable-512.png") == (512, 512)
    assert _png_size(icon_dir / "apple-touch-icon.png") == (180, 180)
    assert _png_size(icon_dir / "algvault-social.png") == (1200, 630)
    assert (icon_dir / "favicon.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_favicon_route_serves_mascot_icon(app) -> None:
    response = app.test_client().get("/favicon.ico")

    assert response.status_code == 308
    assert response.location == "/icons/favicon.ico"


def test_service_worker_precaches_only_safe_shell_assets() -> None:
    source = Path("static/js/sw.js").read_text(encoding="utf-8")
    app_shell = source.split("];", 1)[0]

    assert '"/static/js/app-shell.js"' in app_shell
    assert '"/manifest.json"' in app_shell
    assert '"/icons/favicon.ico"' in app_shell
    assert '"/icons/algvault-icon.svg"' in app_shell
    assert '"/icons/algvault-mask-icon.svg"' in app_shell
    assert '"/icons/algvault-mascot-180.png"' in app_shell
    assert '"/icons/algvault-mascot-192.png"' in app_shell
    assert '"/static/css/app.css"' not in app_shell
    assert '"/static/css/public.css"' not in app_shell
    assert '"/static/js/responsive-tables.js"' not in app_shell
    assert '"/icons/algvault-mascot-512.png"' not in app_shell
    assert '"/admin/dashboard"' not in app_shell
    assert '"/wallet"' not in app_shell
    assert '"/vault"' not in app_shell
    assert "mini-charts.js" not in app_shell
    assert "dashboard.js" not in app_shell
    assert "backtests.js" not in app_shell
    assert "vendor/" not in app_shell
    assert "isApiRequest" in source
    assert "isAuthPath" in source
    assert 'url.pathname === "/sw.js"' in source
    assert 'cache: "no-store"' in source


def test_command_center_dark_theme_layer_is_final() -> None:
    source = Path("static/css/app.css").read_text(encoding="utf-8")
    final_layer = source.split("/* AlgVault command-center redesign. Final layer wins over legacy theme passes. */", 1)[1]

    assert "--bg: #050607" in final_layer
    assert "--panel: rgba(12, 16, 22, 0.96)" in final_layer
    assert "--accent: #6ee7ff" in final_layer
    assert 'html[data-theme="light"]' in final_layer
    assert ".av-command-center" in final_layer
    assert ".av-home-minimal" in source
    assert ".av-strategy-card" in final_layer
    assert ".settings-theme-toggle" in source


def test_pwa_static_headers_keep_sw_fresh_and_assets_cacheable(app) -> None:
    client = app.test_client()

    sw = client.get("/static/js/sw.js")
    root_sw = client.get("/sw.js")
    css = client.get("/static/css/app.css")
    icon = client.get("/icons/algvault-mascot-192.png")

    assert "must-revalidate" in sw.headers["Cache-Control"]
    assert sw.headers["Service-Worker-Allowed"] == "/"
    assert "must-revalidate" in root_sw.headers["Cache-Control"]
    assert root_sw.headers["Service-Worker-Allowed"] == "/"
    assert "immutable" in css.headers["Cache-Control"]
    assert "immutable" in icon.headers["Cache-Control"]
