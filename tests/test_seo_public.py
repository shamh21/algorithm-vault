from __future__ import annotations

import re
import xml.etree.ElementTree as ET

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


def test_root_remains_login_first_for_unauthenticated_visitors(app) -> None:
    response = app.test_client().get("/")

    assert response.status_code == 302
    assert response.location.startswith("/login")
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_restored_public_pages_are_crawlable_mobile_public_surfaces(app) -> None:
    client = app.test_client()
    titles: set[str] = set()
    descriptions: set[str] = set()

    for path, heading in PUBLIC_PATHS.items():
        response = client.get(path)
        html = response.get_data(as_text=True)

        assert response.status_code == 200, path
        assert "public, max-age=300" in response.headers["Cache-Control"]
        assert "X-Robots-Tag" not in response.headers
        assert html.count("<h1") == 1, path
        assert heading in html
        assert '<meta name="robots" content="index, follow, max-image-preview:large">' in html
        assert f'<link rel="canonical" href="https://algvault.app{path}">' in html
        assert '<meta property="og:image" content="https://algvault.app/icons/algvault-social.png">' in html
        assert '<meta name="twitter:image" content="https://algvault.app/icons/algvault-social.png">' in html
        assert "css/public.css" in html
        assert "css/app.css" not in html
        assert "/overview/" in html
        assert "/features/" in html
        assert "/pricing/" in html
        assert "/mobile/" in html
        assert "/connectivity/" in html
        assert "/security/" in html
        assert "/icons/algvault-mascot-64.png" in html
        assert 'aria-label="Open navigation menu"' in html
        assert 'data-component="AlgVaultLaunchAnimation"' not in html

        title = re.search(r"<title>([^<]+)</title>", html)
        assert title
        titles.add(title.group(1))
        descriptions.add(_meta_content(html, "description"))

    assert len(titles) == len(PUBLIC_PATHS)
    assert len(descriptions) == len(PUBLIC_PATHS)


def test_public_pages_avoid_banned_financial_claims_and_sensitive_terms(app) -> None:
    client = app.test_client()
    forbidden = (
        "guaranteed profit",
        "guaranteed profits",
        "guaranteed return",
        "guaranteed returns",
        "risk-free",
        "market-beating",
        "investment advice",
        "WALLET_MPC_SIGNER_TOKEN",
        "TREASURY_ENCRYPTION_KEY",
        "HYPERLIQUID_PRIVATE_KEY",
        "KUCOIN_API_SECRET",
        "bearer token",
        "private key",
    )

    for path in PUBLIC_PATHS:
        html = client.get(path).get_data(as_text=True)
        lower = html.lower()
        for term in forbidden:
            assert term.lower() not in lower, f"{term} leaked on {path}"


def test_public_page_aliases_redirect_to_canonical_routes(app) -> None:
    client = app.test_client()

    mobile_alias = client.get("/mobile-pwa/")
    connectivity_alias = client.get("/broker-connectivity/")

    assert mobile_alias.status_code == 308
    assert mobile_alias.location == "/mobile/"
    assert connectivity_alias.status_code == 308
    assert connectivity_alias.location == "/connectivity/"


def test_robots_and_sitemap_include_only_public_pages(app) -> None:
    client = app.test_client()

    robots = client.get("/robots.txt")
    robots_body = robots.get_data(as_text=True)
    assert robots.status_code == 200
    assert robots.mimetype == "text/plain"
    assert "public, max-age=3600" in robots.headers["Cache-Control"]
    for path in PUBLIC_PATHS:
        assert f"Allow: {path}" in robots_body
    for private_path in ("/admin/", "/api/", "/wallet/", "/vault/", "/convert/", "/settings/"):
        assert f"Disallow: {private_path}" in robots_body
    assert "Sitemap: https://algvault.app/sitemap.xml" in robots_body

    sitemap = client.get("/sitemap.xml")
    sitemap_body = sitemap.get_data(as_text=True)
    assert sitemap.status_code == 200
    assert sitemap.mimetype == "application/xml"
    assert "public, max-age=3600" in sitemap.headers["Cache-Control"]

    root = ET.fromstring(sitemap_body)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {node.text for node in root.findall(".//sm:loc", namespace)}
    assert urls == {f"https://algvault.app{path}" for path in PUBLIC_PATHS}


def test_auth_and_private_routes_are_noindexed(app) -> None:
    client = app.test_client()

    for path in ("/login", "/register", "/wallet/", "/admin/dashboard", "/admin/api/dashboard-data"):
        response = client.get(path)
        assert response.status_code in {200, 302, 401, 403}, path
        assert response.headers["X-Robots-Tag"] == "noindex, nofollow"
