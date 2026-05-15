from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from app import create_app
from app.auth import password_hash, password_matches
from app.extensions import db
from app.models import User


def test_health_and_readiness_are_public_and_database_backed(app) -> None:
    client = app.test_client()

    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.get_json()["ok"] is True
    assert ready.status_code == 200
    payload = ready.get_json()
    assert payload["ok"] is True
    assert payload["checks"]["database"] is True
    assert payload["checks"]["services"] is True


def test_static_cache_headers_distinguish_assets_from_pwa_control_files(app) -> None:
    client = app.test_client()

    css = client.get("/static/css/app.css")
    manifest = client.get("/manifest.json")
    worker = client.get("/static/js/sw.js")
    root_worker = client.get("/sw.js")
    icon = client.get("/icons/icon-192.png")

    assert css.status_code == 200
    assert "immutable" in css.headers["Cache-Control"]
    assert manifest.status_code == 200
    assert "must-revalidate" in manifest.headers["Cache-Control"]
    assert worker.status_code == 200
    assert "must-revalidate" in worker.headers["Cache-Control"]
    assert worker.headers["Service-Worker-Allowed"] == "/"
    assert root_worker.status_code == 200
    assert "must-revalidate" in root_worker.headers["Cache-Control"]
    assert root_worker.headers["Service-Worker-Allowed"] == "/"
    assert icon.status_code == 200
    assert "immutable" in icon.headers["Cache-Control"]


def test_vercel_static_assets_are_allowlisted_from_committed_static_tree() -> None:
    config = json.loads(Path("vercel.json").read_text(encoding="utf-8"))
    builds = config["builds"]
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}

    assert {"src": "static/**/*", "use": "@vercel/static"} in builds
    assert not any(str(item["src"]).startswith("public/") for item in builds)
    assert rewrites["/sw.js"] == "/static/js/sw.js"
    assert rewrites["/icons/(.*)"] == "/static/icons/$1"
    assert rewrites["/manifest.json"] == "/static/manifest.json"
    assert all(not destination.startswith("/public/") for destination in rewrites.values())


def test_dynamic_auth_responses_are_not_edge_cached(app) -> None:
    client = app.test_client()

    login = client.get("/login")
    protected_redirect = client.get("/", follow_redirects=False)

    for response in (login, protected_redirect):
        assert "no-store" in response.headers["Cache-Control"]
        assert "private" in response.headers["Cache-Control"]
        assert response.headers["Pragma"] == "no-cache"
        assert response.headers["Expires"] == "0"


def test_service_worker_registers_with_root_scope(app) -> None:
    client = app.test_client()

    shell = client.get("/login").get_data(as_text=True)
    app_shell = client.get("/static/js/app-shell.js").get_data(as_text=True)

    assert 'window.AV_SW_SCOPE = "/"' in shell
    assert "window.AlgVaultConfig" in shell
    assert "PUBLIC_API_ORIGIN" not in shell
    assert 'window.AV_SW_URL = "/sw.js?' in shell
    assert "isPrivateIP" in app_shell
    assert (
        "AlgVault is running from a private IP HTTPS origin. iOS Safari may show a certificate warning. Use a stable trusted HTTPS hostname instead."
        in app_shell
    )
    assert 'navigator.serviceWorker.register(window.AV_SW_URL, { scope: window.AV_SW_SCOPE || "/" })' in app_shell


def test_intro_loader_replaces_any_in_app_connection_warning_copy(app) -> None:
    shell = app.test_client().get("/login").get_data(as_text=True)

    assert 'class="app-body app-starting' in shell
    assert 'data-component="AlgVaultLaunchAnimation"' in shell
    assert 'data-intro-loader data-component="AlgVaultLaunchAnimation" aria-hidden="true"' in shell
    assert "Opening your vault" in shell
    assert "This connection is not private" not in shell
    assert "connection is not private" not in shell.lower()
    assert "not private" not in shell.lower()


def test_secure_headers_are_applied_to_public_responses(app) -> None:
    response = app.test_client().get("/healthz")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "same-origin"


def test_production_config_defaults_to_https_proxy_and_secure_cookies(monkeypatch) -> None:
    import app.config as config_module

    for name in (
        "DEPLOYMENT_TARGET",
        "PREFERRED_URL_SCHEME",
        "SESSION_COOKIE_SECURE",
        "SESSION_COOKIE_HTTPONLY",
        "SESSION_COOKIE_SAMESITE",
        "PROXY_FIX_ENABLED",
        "SECURE_HEADERS_HSTS_ENABLED",
        "PUBLIC_APP_ORIGIN",
        "PUBLIC_API_ORIGIN",
    ):
        monkeypatch.delenv(name, raising=False)

    reloaded = importlib.reload(config_module)

    assert reloaded.ProductionConfig.DEPLOYMENT_TARGET == "vps"
    assert reloaded.ProductionConfig.PREFERRED_URL_SCHEME == "https"
    assert reloaded.ProductionConfig.SESSION_COOKIE_SECURE is True
    assert reloaded.ProductionConfig.SESSION_COOKIE_HTTPONLY is True
    assert reloaded.ProductionConfig.SESSION_COOKIE_SAMESITE == "Lax"
    assert reloaded.ProductionConfig.PROXY_FIX_ENABLED is True
    assert reloaded.ProductionConfig.SECURE_HEADERS_HSTS_ENABLED is True
    assert reloaded.ProductionConfig.PUBLIC_APP_ORIGIN == "https://app.algvault.com"
    assert reloaded.ProductionConfig.PUBLIC_API_ORIGIN == "https://app.algvault.com"


def test_vercel_target_selects_production_config(monkeypatch) -> None:
    import app.config as config_module

    with monkeypatch.context() as scoped:
        scoped.delenv("APP_ENV", raising=False)
        scoped.delenv("FLASK_ENV", raising=False)
        scoped.delenv("FLASK_CONFIG", raising=False)
        scoped.setenv("DEPLOYMENT_TARGET", "vercel")
        scoped.setenv("WORKER_PROCESS_CONFIGURED", "true")

        reloaded = importlib.reload(config_module)

        assert reloaded.selected_config_class() is reloaded.ProductionConfig
        assert reloaded.ProductionConfig.DEPLOYMENT_TARGET == "vercel"
        assert reloaded.ProductionConfig.SESSION_COOKIE_SECURE is True
        assert reloaded.ProductionConfig.PROXY_FIX_ENABLED is True
        assert reloaded.ProductionConfig.WORKER_PROCESS_CONFIGURED is True
    importlib.reload(config_module)


@pytest.mark.parametrize(
    ("raw_url", "normalized"),
    [
        ("postgres://bot:secret@db.example.invalid/tradingbot", "postgresql+psycopg://bot:secret@db.example.invalid/tradingbot"),
        (
            "postgresql://bot:secret@db.example.invalid/tradingbot",
            "postgresql+psycopg://bot:secret@db.example.invalid/tradingbot",
        ),
        (
            "postgresql+psycopg://bot:secret@db.example.invalid/tradingbot",
            "postgresql+psycopg://bot:secret@db.example.invalid/tradingbot",
        ),
    ],
)
def test_database_url_uses_psycopg_driver_for_hosted_postgres(monkeypatch, raw_url, normalized) -> None:
    import app.config as config_module

    with monkeypatch.context() as scoped:
        scoped.setenv("DATABASE_URL", raw_url)
        reloaded = importlib.reload(config_module)

        assert normalized == reloaded.BaseConfig.SQLALCHEMY_DATABASE_URI
    importlib.reload(config_module)


def test_vercel_server_entrypoint_exposes_flask_app(monkeypatch) -> None:
    import app.config as config_module

    with monkeypatch.context() as scoped:
        scoped.setenv("APP_ENV", "production")
        scoped.setenv("DEPLOYMENT_TARGET", "vercel")
        scoped.setenv("DATABASE_URL", "postgresql+psycopg://bot:secret@db.example.invalid/tradingbot")
        scoped.setenv("PUBLIC_APP_ORIGIN", "https://app.algvault.com")
        scoped.setenv("PUBLIC_API_ORIGIN", "https://app.algvault.com")
        scoped.setenv("FLASK_SECRET_KEY", "server-entrypoint-secret-key-1234567890")
        scoped.setenv("TOTP_ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
        scoped.setenv("ENABLE_LIVE_TRADING", "false")
        scoped.setenv("APP_MODE", "paper")
        scoped.setenv("ENABLE_IN_PROCESS_WORKERS", "false")
        scoped.setenv("SCHEMA_BOOTSTRAP_ENABLED", "false")
        scoped.setenv("WALLET_WITHDRAWALS_ENABLED", "false")
        scoped.setenv("SKIP_SCHEMA_BOOTSTRAP", "1")

        importlib.reload(config_module)
        sys.modules.pop("server", None)
        module = importlib.import_module("server")

        assert module.app.name == "app"
        assert module.app.config["DEPLOYMENT_TARGET"] == "vercel"
    sys.modules.pop("server", None)
    importlib.reload(config_module)


@pytest.mark.parametrize(
    ("app_origin", "api_origin"),
    [
        ("https://172.20.10.6", "https://app.algvault.com"),
        ("http://app.algvault.com", "https://app.algvault.com"),
        ("https://app.algvault.com", "https://192.168.1.20"),
        ("https://localhost:5000", "https://app.algvault.com"),
    ],
)
def test_production_public_origins_reject_untrusted_hosts(app_origin, api_origin) -> None:
    with pytest.raises(RuntimeError, match="Invalid production public origin configuration"):
        create_app(
            {
                "TESTING": True,
                "DEPLOYMENT_TARGET": "production",
                "SQLALCHEMY_DATABASE_URI": "sqlite://",
                "PUBLIC_APP_ORIGIN": app_origin,
                "PUBLIC_API_ORIGIN": api_origin,
            }
        )


def test_rate_limit_can_protect_login_when_enabled(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'rate-limit.db'}",
            "SECRET_KEY": "test-secret-key-12345678901234567890",
            "WTF_CSRF_ENABLED": False,
            "RATELIMIT_FORCE_ENABLED": True,
            "RATELIMIT_LOGIN_PER_WINDOW": 1,
            "RATELIMIT_WINDOW_SECONDS": 60,
        }
    )
    with app.app_context():
        client = app.test_client()

        first = client.post("/login", data={"username": "missing", "password": "bad"})
        second = client.post("/login", data={"username": "missing", "password": "bad"})

        assert first.status_code == 302
        assert second.status_code == 429
        assert second.headers["Retry-After"]
        db.session.remove()
        db.drop_all()


def test_login_post_recovers_from_missing_anonymous_csrf_session(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'csrf-login.db'}",
            "SECRET_KEY": "test-secret-key-12345678901234567890",
            "WTF_CSRF_ENABLED": True,
        }
    )
    with app.app_context():
        client = app.test_client()

        response = client.post("/login", data={"username": "missing", "password": "bad"}, follow_redirects=False)

        assert response.status_code == 302
        assert response.location == "/login"
        db.session.remove()
        db.drop_all()


def test_admin_api_sign_in_recovers_from_missing_anonymous_csrf_session(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'csrf-admin-login.db'}",
            "SECRET_KEY": "test-secret-key-12345678901234567890",
            "WTF_CSRF_ENABLED": True,
        }
    )
    with app.app_context():
        client = app.test_client()

        response = client.post("/admin/api/sign-in", json={"username": "missing", "password": "bad", "totpCode": "000000"})

        assert response.status_code == 401
        assert response.get_json()["code"] == "admin_sign_in_failed"
        db.session.remove()
        db.drop_all()


def test_malformed_password_hash_fails_closed_without_application_error(app) -> None:
    user = User(username="broken-hash", password_hash=password_hash("password123"))
    db.session.add(user)
    db.session.commit()
    user.password_hash = None  # type: ignore[assignment]

    assert password_matches(user, "password123") is False


def test_rate_limit_can_protect_admin_pwa_sign_in_when_enabled(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'admin-rate-limit.db'}",
            "SECRET_KEY": "test-secret-key-12345678901234567890",
            "WTF_CSRF_ENABLED": False,
            "RATELIMIT_FORCE_ENABLED": True,
            "RATELIMIT_LOGIN_PER_WINDOW": 1,
            "RATELIMIT_WINDOW_SECONDS": 60,
        }
    )
    with app.app_context():
        client = app.test_client()

        first = client.post("/admin/api/sign-in", json={"username": "missing", "password": "bad", "totpCode": "000000"})
        second = client.post("/admin/api/sign-in", json={"username": "missing", "password": "bad", "totpCode": "000000"})

        assert first.status_code == 401
        assert second.status_code == 429
        assert second.headers["Retry-After"]
        assert second.get_json()["error"] == "rate_limited"
        db.session.remove()
        db.drop_all()


def test_auth_protected_route_categories_fail_closed_without_session(app) -> None:
    client = app.test_client()

    protected_paths = [
        "/wallet/",
        "/vault/",
        "/admin/dashboard",
        "/admin/api/dashboard-data",
        "/api/vault/routing-preview",
    ]

    for path in protected_paths:
        response = client.get(path)
        assert response.status_code in {302, 401, 403}, path


def test_pwa_manifest_has_ios_install_shape(app) -> None:
    response = app.test_client().get("/manifest.json")
    payload = json.loads(response.get_data(as_text=True))

    assert payload["name"] == "AlgVault"
    assert payload["short_name"] == "AlgVault"
    assert payload["display"] == "standalone"
    assert payload["id"] == "/"
    assert payload["start_url"] == "/"
    assert payload["scope"] == "/"
    assert payload["background_color"] == "#050607"
    assert payload["theme_color"] == "#050607"
    assert {icon["src"] for icon in payload["icons"]} >= {
        "/icons/algvault-ios-192.png",
        "/icons/algvault-ios-512.png",
        "/icons/algvault-ios-180.png",
    }
    assert any("maskable" in icon["purpose"] and icon["sizes"] == "192x192" for icon in payload["icons"])
    assert any("maskable" in icon["purpose"] and icon["sizes"] == "512x512" for icon in payload["icons"])
    assert {shortcut["short_name"] for shortcut in payload["shortcuts"]} >= {"Wallet", "Vault"}


def test_ios_pwa_head_tags_target_algvault(app) -> None:
    shell = app.test_client().get("/login").get_data(as_text=True)

    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in shell
    assert '<meta name="apple-mobile-web-app-title" content="AlgVault">' in shell
    assert '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">' in shell
    assert '<meta name="theme-color" content="#050607">' in shell
    assert '<meta name="color-scheme" content="dark">' in shell
    assert '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' in shell
    assert '<link rel="apple-touch-icon" href="/icons/algvault-ios-180.png">' in shell
    assert "data-theme-toggle" in shell
    assert '<link rel="manifest" href="/manifest.json">' in shell


def test_service_worker_clears_old_algvault_and_tradingbot_caches(app) -> None:
    worker = app.test_client().get("/static/js/sw.js").get_data(as_text=True)

    assert 'const CACHE_VERSION = "algvault-v9-command-center-dark"' in worker
    assert 'name.startsWith("algvault-") || name.startsWith("tradingbot-")' in worker
    assert "self.clients.claim()" in worker
    assert "/manifest.json" in worker
    assert "/icons/algvault-ios-192.png" in worker


def test_service_worker_offline_fallback_does_not_cache_trading_state(app) -> None:
    worker = app.test_client().get("/static/js/sw.js").get_data(as_text=True)

    assert 'fetch(request, { cache: "no-store", credentials: "same-origin" })' in worker
    assert "Static app assets remain cached safely." in worker
    assert "AlgVault is offline" in worker
    assert "isApiRequest(url) || isAuthPath(url) || isServiceWorkerAsset(url)" in worker


def test_readme_has_iphone_pwa_https_setup_instructions() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## iPhone PWA HTTPS setup" in readme
    assert "Do not install AlgVault from `https://172.20.10.6`" in readme
    assert "`https://app.algvault.com`" in readme
    assert "Delete the old AlgVault PWA icon" in readme
    assert 'Confirm Safari does not show "This connection is not private."' in readme
    assert "Confirm no network requests go to `172.20.10.6`." in readme
    assert "Confirm service worker scope is `/`." in readme
    assert "Confirm manifest `start_url` is `/`." in readme
