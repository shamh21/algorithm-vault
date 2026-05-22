from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

PUBLIC_PATHS = {
    "/overview/": "Automated trading control without the noise",
    "/features/": "All the features you need to trade with confidence",
    "/pricing/": "Simple access tiers for monitored automation",
    "/mobile/": "An iPhone-ready trading command center",
    "/connectivity/": "Secure connections. Operational clarity.",
    "/security/": "Trust built through visible controls",
}


def _meta_content(html: str, name: str) -> str:
    match = re.search(rf'<meta name="{re.escape(name)}" content="([^"]+)">', html)
    assert match, f"missing meta {name}"
    return match.group(1)


def _title(html: str) -> str:
    match = re.search(r"<title>([^<]+)</title>", html)
    assert match, "missing title"
    return match.group(1)


def test_public_home_is_crawlable_and_has_rich_metadata(app) -> None:
    response = app.test_client().get("/overview/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "public, max-age=300" in response.headers["Cache-Control"]
    assert "X-Robots-Tag" not in response.headers
    assert '<meta name="robots" content="index, follow, max-image-preview:large">' in html
    assert '<link rel="canonical" href="https://algvault.app/overview/">' in html
    assert '<meta property="og:title" content="AlgVault | Automated Crypto Trading PWA and Execution Controls">' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html
    assert '"@type":"Organization"' in html or '"@type": "Organization"' in html
    assert '"@type":"SoftwareApplication"' in html or '"@type": "SoftwareApplication"' in html
    assert 'data-component="AlgVaultLaunchAnimation"' not in html
    assert "css/public.css" in html
    assert "css/app.css" not in html
    assert "ops-bridge.js" not in html
    assert "responsive-tables.js" not in html
    assert "crypto-rail" not in html
    assert "<h1" in html
    assert html.count("<h1") == 1
    assert PUBLIC_PATHS["/overview/"] in html
    assert 'aria-label="Open navigation menu"' in html
    assert "overview-hero" in html
    assert "LIVE SYSTEM" in html
    assert "Hyperliquid Connected" in html
    assert "overview-phone" in html
    assert "Active Strategies" in html
    assert "Connected Providers" in html
    assert "System Latency" in html
    assert "Risk Engine Status" in html
    assert "Strategy Monitor" in html
    assert "Broker/API" in html
    assert "Wallet Controls" in html
    assert "Mobile-first by design" in html
    assert "Add to Home Screen" in html
    assert "Security architecture" in html
    assert "Connected ecosystem" in html
    assert "Ready to take control?" in html
    assert "public-link-band" not in html
    assert "guaranteed profits" not in html.lower()
    assert "risk-free trading" not in html.lower()


def test_public_pages_have_unique_metadata_canonicals_and_internal_links(app) -> None:
    client = app.test_client()
    titles: set[str] = set()
    descriptions: set[str] = set()

    for path, heading in PUBLIC_PATHS.items():
        response = client.get(path)
        html = response.get_data(as_text=True)

        assert response.status_code == 200, path
        assert "public, max-age=300" in response.headers["Cache-Control"]
        assert html.count("<h1") == 1, path
        assert heading in html
        assert f'<link rel="canonical" href="https://algvault.app{path}">' in html
        assert '<meta property="og:image" content="https://algvault.app/icons/algvault-mascot-512.png">' in html
        assert '<meta name="twitter:image" content="https://algvault.app/icons/algvault-mascot-512.png">' in html
        assert "/features/" in html
        assert "/security/" in html

        titles.add(_title(html))
        descriptions.add(_meta_content(html, "description"))

    assert len(titles) == len(PUBLIC_PATHS)
    assert len(descriptions) == len(PUBLIC_PATHS)


def test_public_features_page_uses_exchange_landing_system(app) -> None:
    response = app.test_client().get("/features/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert html.count("<h1") == 1
    assert "features-landing" in html
    assert "features-hero" in html
    assert "features-hero-visual" in html
    assert "All features" in html
    assert "Risk &amp; Control" in html
    assert "Platform Features" in html
    assert "Automation &amp; Smart Features" in html
    assert "Insights &amp; Oversight" in html
    assert "Why it matters" in html
    assert "Experience AlgVault the smart way" in html
    assert "Features readiness" in html
    assert "Stay ahead with a platform built for what&#39;s next." in html
    assert "Contextual" in html
    assert "Cycle-based execution" in html
    assert "Portfolio visibility" in html
    assert "Guarded execution workflows" in html
    assert "Strategy posture at a glance" in html
    assert "Signals with control" in html
    assert "Wallet and vault clarity" in html
    assert "Provider-aware workflows" in html
    assert "Gates before action" in html
    assert "Runtime posture in view" in html
    assert "Built for repeated checks, not long reads" in html
    assert "Buttons do not bypass validation" in html
    assert 'href="/register"' in html
    assert 'href="/pricing/"' in html
    assert 'href="/mobile/"' in html
    assert "css/public.css" in html
    assert "css/app.css" not in html
    assert "ops-bridge.js" not in html
    assert "guaranteed profits" not in html.lower()
    assert "risk-free trading" not in html.lower()


def test_public_pricing_page_uses_exchange_plan_comparison(app) -> None:
    response = app.test_client().get("/pricing/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert html.count("<h1") == 1
    assert "pricing-landing" in html
    assert "pricing-hero" in html
    assert "Simple access tiers for monitored automation" in html
    assert "Choose the AlgVault access level that matches your workflow" in html
    assert "Transparent tiers" in html
    assert "No hidden urgency" in html
    assert "No performance claims" in html
    assert "Starter" in html
    assert "Operator" in html
    assert "Recommended" in html
    assert "Custom" in html
    assert "Phone/PWA access" in html
    assert "Automated execution controls" in html
    assert "Operational audit visibility" in html
    assert "Compare access tiers" in html
    assert "Mobile PWA" in html
    assert "Execution controls" in html
    assert "Provider visibility" in html
    assert "Custom workflows" in html
    assert "Built on safety and clarity" in html
    assert "Server-led validation" in html
    assert "No guaranteed returns" in html
    assert "Secure account setup" in html
    assert "Plan boundaries stay clear" in html
    assert "Start with secure setup" in html
    assert "Final plan details are confirmed during account setup." in html
    assert 'href="/register"' in html
    assert 'href="/features/"' in html
    assert 'href="/security/"' in html
    assert "css/public.css" in html
    assert "css/app.css" not in html
    assert "ops-bridge.js" not in html
    assert "public-cta-block" not in html
    assert html.count("Start with secure setup") == 1
    assert "guaranteed profits" not in html.lower()
    assert "risk-free trading" not in html.lower()


def test_public_connectivity_page_uses_operations_dashboard_layout(app) -> None:
    response = app.test_client().get("/connectivity/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert html.count("<h1") == 1
    assert "connectivity-landing" in html
    assert "connection-overview-card" in html
    assert "Secure connections. Operational clarity." in html
    assert "Every connection is validated, encrypted, and continuously supervised." in html
    assert "Encrypted" in html
    assert "Monitored" in html
    assert "Resilient" in html
    assert "Transparent" in html
    assert "Connected" in html
    assert "Degraded" in html
    assert "Disconnected" in html
    assert "Interactive Brokers" in html
    assert "Tradovate" in html
    assert "Binance" in html
    assert "Coinbase Exchange" in html
    assert "Bybit" in html
    assert "Kraken" in html
    assert "OANDA" in html
    assert "dxFeed" in html
    assert "TradingView" in html
    assert "Polygon.io" in html
    assert "Alpaca" in html
    assert "More providers" in html
    assert "Broker Gateway" in html
    assert "Market Data" in html
    assert "Order Routing" in html
    assert "Account Sync" in html
    assert "Notifications" in html
    assert html.count("Operational") >= 5
    assert "Connection established" in html
    assert "Heartbeat successful" in html
    assert "Reconnected" in html
    assert "Authentication refreshed" in html
    assert "Temporary latency detected" in html
    assert "Read-only by default" in html
    assert "Encrypted transport" in html
    assert "Least-privilege access" in html
    assert "Session monitoring" in html
    assert "Connect with confidence" in html
    assert "Review connectivity" in html
    assert "Security model" in html
    assert "API key" not in html
    assert "webhook URL" not in html
    assert "token" not in html.lower()


def test_public_pages_render_premium_product_system_without_sensitive_material(app) -> None:
    client = app.test_client()
    forbidden = (
        "WALLET_MPC_SIGNER_TOKEN",
        "TREASURY_ENCRYPTION_KEY",
        "WEBHOOK_SECRET",
        "KUCOIN_API_SECRET",
        "HYPERLIQUID_PRIVATE_KEY",
        "guaranteed profits",
        "risk-free trading",
    )

    for path in PUBLIC_PATHS:
        html = client.get(path).get_data(as_text=True)

        if path == "/overview/":
            assert "overview-hero" in html, path
            assert "overview-final-cta" in html, path
            assert "public-link-band" not in html, path
        elif path == "/features/":
            assert "features-landing" in html, path
            assert "features-final-cta" in html, path
            assert "features-filter" in html, path
        elif path == "/pricing/":
            assert "pricing-landing" in html, path
            assert "pricing-plan-grid" in html, path
            assert "pricing-final-cta" in html, path
        elif path == "/connectivity/":
            assert "connectivity-landing" in html, path
            assert "connection-overview-card" in html, path
            assert "provider-connection-grid" in html, path
            assert "connectivity-final-cta" in html, path
        else:
            assert "public-device-frame" in html, path
            assert "public-trust-badge" in html, path
            assert "public-cta-block" in html, path
            assert "public-card-system" in html or "public-status-system" in html, path
        assert any(term in html for term in ("server", "Server", "Protected", "Blocked", "Readiness")), path
        for term in forbidden:
            assert term.lower() not in html.lower(), path


def test_legacy_public_marketing_paths_redirect_to_canonical(app) -> None:
    client = app.test_client()

    mobile_alias = client.get("/mobile-pwa/")
    connectivity_alias = client.get("/broker-connectivity/")

    assert mobile_alias.status_code == 308
    assert mobile_alias.location == "/mobile/"
    assert connectivity_alias.status_code == 308
    assert connectivity_alias.location == "/connectivity/"


def test_robots_txt_blocks_private_surfaces_and_links_sitemap(app) -> None:
    response = app.test_client().get("/robots.txt")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert "public, max-age=3600" in response.headers["Cache-Control"]
    assert "Allow: /overview/" in body
    assert "Allow: /features/" in body
    assert "Allow: /connectivity/" in body
    assert "Disallow: /admin/" in body
    assert "Disallow: /api/" in body
    assert "Disallow: /wallet/" in body
    assert "Disallow: /vault/" in body
    assert "Disallow: /convert/" in body
    assert "Disallow: /settings/" in body
    assert "Disallow: /_internal/" in body
    assert "Sitemap: https://algvault.app/sitemap.xml" in body


def test_sitemap_contains_only_public_canonical_urls(app) -> None:
    response = app.test_client().get("/sitemap.xml")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.mimetype == "application/xml"
    assert "public, max-age=3600" in response.headers["Cache-Control"]

    root = ET.fromstring(body)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {node.text for node in root.findall(".//sm:loc", namespace)}

    assert urls == {f"https://algvault.app{path}" for path in PUBLIC_PATHS}
    assert all("/admin" not in url for url in urls if url)
    assert all("/wallet" not in url for url in urls if url)
    assert all("/api/" not in url for url in urls if url)


def test_auth_and_private_routes_are_noindexed(app) -> None:
    client = app.test_client()

    root = client.get("/")
    assert root.status_code == 302
    assert root.location == "/overview/"
    assert root.headers["X-Robots-Tag"] == "noindex, nofollow"

    login = client.get("/login")
    login_html = login.get_data(as_text=True)
    assert login.status_code == 200
    assert login.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert '<meta name="robots" content="noindex, nofollow">' in login_html

    wallet = client.get("/wallet/")
    assert wallet.status_code == 302
    assert wallet.headers["X-Robots-Tag"] == "noindex, nofollow"

    admin = client.get("/admin/dashboard")
    assert admin.status_code == 302
    assert admin.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_admin_pwa_metadata_is_noindex() -> None:
    source = Path("admin-pwa/src/app/layout.tsx").read_text(encoding="utf-8")

    assert "robots:" in source
    assert "index: false" in source
    assert "follow: false" in source
