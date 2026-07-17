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


def test_public_home_is_crawlable_and_uses_compliant_shared_system(app) -> None:
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
    assert html.count("<h1") == 1
    assert PUBLIC_PATHS["/overview/"] in html
    assert 'aria-label="Open navigation menu"' in html
    assert "overview-hero" in html
    assert "ILLUSTRATIVE UI" in html
    assert "Server-confirmed only" in html
    assert "Execution authority" in html
    assert "Provider state" in html
    assert "Trading outcomes" in html
    assert "Not promised" in html
    assert "Designed for repeated iPhone checks" in html
    assert "Security architecture before automation" in html
    assert "Move from public context to protected setup" in html
    for forbidden in ("$128,420.58", "LIVE SYSTEM", "Hyperliquid Connected", "Active Strategies", "42ms"):
        assert forbidden not in html


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
        assert '<meta property="og:image" content="https://algvault.app/icons/algvault-ios-512.png">' in html
        assert '<meta name="twitter:image" content="https://algvault.app/icons/algvault-ios-512.png">' in html
        assert "/features/" in html
        assert "/security/" in html
        assert 'class="public-audit-shell' in html
        assert 'class="public-section-nav"' in html
        assert 'id="capabilities"' in html
        assert 'id="mobile-pwa"' in html
        assert 'id="trust"' in html

        titles.add(_title(html))
        descriptions.add(_meta_content(html, "description"))

    assert len(titles) == len(PUBLIC_PATHS)
    assert len(descriptions) == len(PUBLIC_PATHS)


def test_public_feature_pricing_connectivity_and_security_states_are_explicit(app) -> None:
    client = app.test_client()

    features = client.get("/features/").get_data(as_text=True)
    assert "Operational capabilities" in features
    assert "Guarded execution workflows" in features
    assert "Buttons do not bypass validation" in features
    assert 'href="/pricing/"' in features

    pricing = client.get("/pricing/").get_data(as_text=True)
    assert "Starter" in pricing
    assert "Operator" in pricing
    assert "Custom" in pricing
    assert "No performance claims" in pricing
    assert "Final plan details are confirmed during account setup." in pricing

    connectivity = client.get("/connectivity/").get_data(as_text=True)
    assert 'id="supported-connections"' in connectivity
    assert "Implemented providers remain eligibility-gated" in connectivity
    assert "Hyperliquid" in connectivity
    assert "KuCoin" in connectivity
    assert "does not publish invented connection counts" in connectivity
    for forbidden in ("Interactive Brokers", "Tradovate", "Heartbeat successful", "2m ago"):
        assert forbidden not in connectivity

    security = client.get("/security/").get_data(as_text=True)
    assert "No success state before server confirmation" in security
    assert "Pending until the server returns a final state" in security
    assert "Completed, rejected, or failed remain distinct" in security
    assert "Server-authoritative" in security


def test_public_pages_exclude_sensitive_material_and_false_outcomes(app) -> None:
    client = app.test_client()
    forbidden = (
        "WALLET_MPC_SIGNER_TOKEN",
        "TREASURY_ENCRYPTION_KEY",
        "WEBHOOK_SECRET",
        "KUCOIN_API_SECRET",
        "HYPERLIQUID_PRIVATE_KEY",
        "guaranteed profits",
        "risk-free trading",
        "$128,420.58",
        "All systems operational",
        "Heartbeat successful",
    )

    for path in PUBLIC_PATHS:
        html = client.get(path).get_data(as_text=True)
        assert "public-device-frame" in html, path
        assert "public-trust-badge" in html, path
        assert "public-cta-block" in html, path
        assert "public-card-system" in html, path
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
    for path in ("/admin/", "/api/", "/wallet/", "/vault/", "/convert/", "/settings/", "/_internal/"):
        assert f"Disallow: {path}" in body
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
    assert login.status_code == 200
    assert login.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert '<meta name="robots" content="noindex, nofollow">' in login.get_data(as_text=True)

    for path in ("/wallet/", "/admin/dashboard"):
        response = client.get(path)
        assert response.status_code == 302
        assert response.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_admin_pwa_metadata_is_noindex() -> None:
    source = Path("admin-pwa/src/app/layout.tsx").read_text(encoding="utf-8")

    assert "robots:" in source
    assert "index: false" in source
    assert "follow: false" in source
