from __future__ import annotations

import json
import struct
from pathlib import Path


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header.startswith(b"\x89PNG\r\n\x1a\n")
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

    icons = {icon["sizes"]: icon for icon in manifest["icons"]}
    assert icons["192x192"]["src"] == "/icons/algvault-ios-192.png"
    assert icons["192x192"]["purpose"] == "any maskable"
    assert icons["512x512"]["src"] == "/icons/algvault-ios-512.png"
    assert icons["512x512"]["purpose"] == "any maskable"
    assert icons["180x180"]["src"] == "/icons/algvault-ios-180.png"
    assert "/admin/panic/" not in {shortcut["url"] for shortcut in manifest["shortcuts"]}
    assert "/login" in {shortcut["url"] for shortcut in manifest["shortcuts"]}
    assert "/convert/" in {shortcut["url"] for shortcut in manifest["shortcuts"]}


def test_manifest_files_stay_in_sync() -> None:
    root = Path("static")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    webmanifest = json.loads((root / "manifest.webmanifest").read_text(encoding="utf-8"))

    assert manifest == webmanifest


def test_base_template_has_dark_ios_pwa_metadata(app) -> None:
    response = app.test_client().get("/login")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' in html
    assert '<meta name="theme-color" content="#050607">' in html
    assert '<meta name="color-scheme" content="dark">' in html
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in html
    assert '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">' in html
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
    assert "css/app.css" in html
    assert "login-redblack-auth-8" in html
    assert "auth-pwa-polish-5" in html
    assert "avguard-mascot-1" in html
    assert "avguard-mascot-polish-1" in html
    assert 'class="auth-mascot-lockup"' in html
    assert "/icons/algvault-avguard-mark-192.webp" in html
    assert "/icons/algvault-avguard-mark-192.png" in html
    assert 'class="auth-shell auth-login-shell"' in html
    assert 'class="vault-card auth-card auth-login-card"' in html
    assert "Continue to Wallet" in html
    assert "Production Status" in html
    assert "Database" in html
    assert "Mode" in html
    assert "Withdrawals" in html
    assert "Treasury" in html
    assert 'method="post" action="/login?next=/wallet/"' in html
    assert 'class="form-grid auth-login-form"' in html
    assert 'name="csrf_token"' in html
    assert 'name="username"' in html
    assert 'name="password"' in html
    assert 'name="totp_code"' in html
    assert 'autocapitalize="none"' in html
    assert 'autocorrect="off"' in html
    assert 'autocomplete="current-password"' in html
    assert 'enterkeyhint="next"' in html
    assert 'enterkeyhint="done"' in html
    assert 'maxlength="6"' in html
    assert 'class="primary auth-login-submit"' in html
    assert 'href="/register"' in html
    assert "Continue with Google" not in html
    assert "Forgot password" not in html
    assert "Remember this device" not in html
    assert "WALLET_MPC_SIGNER_TOKEN" not in html
    assert "TREASURY_ENCRYPTION_KEY" not in html


def test_login_shell_renders_when_admin_lookup_is_unavailable(app, monkeypatch) -> None:
    import app as app_module

    def unavailable_admin_lookup() -> bool:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(app_module, "admin_configured", unavailable_admin_lookup)

    response = app.test_client().get("/login")

    assert response.status_code == 200
    assert "Sign In" in response.get_data(as_text=True)


def test_register_shell_preserves_invite_signup_flow(app) -> None:
    app.config["SIGNUP_INVITE_CODE"] = "join-code"
    response = app.test_client().get("/register")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "css/app.css" in html
    assert "register-redblack-auth-2" in html
    assert "auth-pwa-polish-5" in html
    assert "avguard-mascot-1" in html
    assert "avguard-mascot-polish-1" in html
    assert 'class="auth-mascot-lockup"' in html
    assert "/icons/algvault-avguard-mark-192.webp" in html
    assert "/icons/algvault-avguard-mark-192.png" in html
    assert 'class="auth-shell auth-register-shell"' in html
    assert 'class="vault-card auth-card auth-register-card"' in html
    assert 'class="card-kicker auth-register-badge">Invite Required</span>' in html
    assert "Create Account" in html
    assert "Registration requires an invite code" in html
    assert 'method="post" action="/register"' in html
    assert 'class="form-grid auth-register-form"' in html
    assert 'name="csrf_token"' in html
    assert 'name="username"' in html
    assert 'name="password"' in html
    assert 'name="confirm_password"' in html
    assert 'name="invite_code"' in html
    assert 'autocomplete="username"' in html
    assert 'autocomplete="new-password"' in html
    assert 'autocapitalize="none"' in html
    assert 'autocorrect="off"' in html
    assert 'spellcheck="false"' in html
    assert 'enterkeyhint="next"' in html
    assert 'enterkeyhint="done"' in html
    assert 'minlength="8"' in html
    assert "positive Vault Cycle profit" not in html
    assert "deposits, principal, or losses" not in html
    assert "invite-profit-share-note" not in html
    assert 'class="primary auth-register-submit"' in html
    assert 'href="/login"' in html
    assert "Continue with Google" not in html
    assert "Forgot password" not in html
    assert "Remember this device" not in html


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
    mask_source = Path("static/icons/algvault-mask-icon.svg").read_text(encoding="utf-8")
    crypto_icon = Path("templates/components/crypto_icon.html").read_text(encoding="utf-8")

    assert "av guard robot" in icon_source.lower()
    assert "black beanie" in icon_source.lower()
    assert "exchange-grade red and black" in icon_source.lower()
    assert "av guard robot mascot silhouette" in mask_source.lower()
    for symbol in ("BTC", "ETH", "ALGO", "USDT", "USDC", "SOL", "XRP"):
        assert symbol in crypto_icon


def test_avguard_mascot_assets_have_expected_dimensions() -> None:
    icons = Path("static/icons")

    for size in (16, 32, 48, 64, 72, 96, 128, 144, 152, 167, 180, 192, 256, 384, 512):
        assert _png_size(icons / f"algvault-mascot-{size}.png") == (size, size)

    for size in (180, 192, 512):
        assert _png_size(icons / f"algvault-ios-{size}.png") == (size, size)

    for size in (192, 512):
        assert _png_size(icons / f"algvault-maskable-{size}.png") == (size, size)

    assert _png_size(icons / "icon-192.png") == (192, 192)
    assert _png_size(icons / "icon-512.png") == (512, 512)
    assert _png_size(icons / "apple-touch-icon.png") == (180, 180)
    assert _png_size(icons / "algvault-avguard-full.png") == (768, 1132)
    assert _png_size(icons / "algvault-avguard-bust.png") == (1024, 1024)
    assert _png_size(icons / "algvault-avguard-sprite.png") == (2048, 512)
    assert _png_size(icons / "algvault-avguard-mark-96.png") == (96, 96)
    assert _png_size(icons / "algvault-avguard-mark-192.png") == (192, 192)
    assert (icons / "favicon.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"

    for webp in (
        "algvault-avguard-full.webp",
        "algvault-avguard-bust.webp",
        "algvault-avguard-sprite.webp",
        "algvault-avguard-mark-96.webp",
        "algvault-avguard-mark-192.webp",
    ):
        payload = (icons / webp).read_bytes()
        assert payload[:4] == b"RIFF"
        assert payload[8:12] == b"WEBP"


def test_service_worker_precaches_only_safe_shell_assets() -> None:
    source = Path("static/js/sw.js").read_text(encoding="utf-8")
    app_shell = source.split("];", 1)[0]

    assert '"/static/css/app.css"' in app_shell
    assert '"/static/js/app-shell.js"' in app_shell
    assert '"/static/js/responsive-tables.js"' in app_shell
    assert '"/manifest.json"' in app_shell
    assert '"/icons/favicon.ico"' in app_shell
    assert '"/icons/algvault-icon.svg"' in app_shell
    assert '"/icons/algvault-mask-icon.svg"' in app_shell
    assert '"/icons/algvault-avguard-mark-96.png"' in app_shell
    assert '"/icons/algvault-avguard-mark-96.webp"' in app_shell
    assert '"/icons/algvault-avguard-mark-192.png"' in app_shell
    assert '"/icons/algvault-avguard-mark-192.webp"' in app_shell
    assert '"/icons/algvault-mascot-192.png"' in app_shell
    assert '"/icons/algvault-ios-180.png"' in app_shell
    assert '"/icons/algvault-ios-192.png"' in app_shell
    assert '"/icons/algvault-ios-512.png"' in app_shell
    assert '"/icons/icon-192.png"' in app_shell
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
    icon = client.get("/icons/algvault-ios-192.png")

    assert "must-revalidate" in sw.headers["Cache-Control"]
    assert sw.headers["Service-Worker-Allowed"] == "/"
    assert "must-revalidate" in root_sw.headers["Cache-Control"]
    assert root_sw.headers["Service-Worker-Allowed"] == "/"
    assert "immutable" in css.headers["Cache-Control"]
    assert "immutable" in icon.headers["Cache-Control"]
