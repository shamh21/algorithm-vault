"""Central SEO metadata and public indexability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

from flask import Flask

CANONICAL_ORIGIN = "https://algvault.app"
BRAND_NAME = "AlgVault"
SOCIAL_IMAGE_PATH = "/icons/algvault-mascot-512.png"
PUBLIC_HTML_CACHE_CONTROL = "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400"
SEO_ASSET_CACHE_CONTROL = "public, max-age=3600, s-maxage=86400, stale-while-revalidate=86400"


@dataclass(frozen=True)
class PublicSeoPage:
    key: str
    endpoint: str
    path: str
    nav_label: str
    title: str
    description: str
    eyebrow: str
    heading: str
    lead: str
    primary_label: str
    primary_href: str
    secondary_label: str
    secondary_href: str
    highlights: tuple[dict[str, Any], ...]
    sections: tuple[dict[str, Any], ...]
    badges: tuple[dict[str, Any], ...] = ()
    hero_rows: tuple[dict[str, Any], ...] = ()
    cards: tuple[dict[str, Any], ...] = ()
    status_rows: tuple[dict[str, Any], ...] = ()
    cta: dict[str, Any] | None = None
    disclaimer: str = ""


PUBLIC_PAGES: dict[str, PublicSeoPage] = {
    "home": PublicSeoPage(
        key="home",
        endpoint="consumer.public_overview",
        path="/overview/",
        nav_label="Overview",
        title="AlgVault | Automated Crypto Trading PWA and Execution Controls",
        description=(
            "AlgVault is a mobile-first crypto trading PWA for automated strategy monitoring, "
            "server-side execution controls, wallet visibility, and broker/API connectivity."
        ),
        eyebrow="Automated trading infrastructure",
        heading="Automated trading control without the noise",
        lead=(
            "AlgVault gives operators a compact way to review strategy, wallet, provider, and risk state from a premium iPhone-ready PWA."
        ),
        primary_label="Create secure access",
        primary_href="/register",
        secondary_label="Explore features",
        secondary_href="/features/",
        highlights=(
            {"label": "Active Strategies", "value": "12", "detail": "Online strategy monitor."},
            {"label": "Connected Providers", "value": "4", "detail": "Healthy broker/API posture."},
            {"label": "System Latency", "value": "42ms", "detail": "Fast public operating shell."},
            {"label": "Risk Engine Status", "value": "Online", "detail": "Server-side controls visible."},
        ),
        sections=(),
        badges=(
            {"label": "Server-side controls", "tone": "secure"},
            {"label": "iPhone PWA", "tone": "mobile"},
            {"label": "Risk states visible", "tone": "alert"},
        ),
        hero_rows=(
            {"label": "Strategy monitor", "value": "Online", "state": "Signals visible"},
            {"label": "Broker/API", "value": "Checked", "state": "Readiness first"},
            {"label": "Wallet controls", "value": "Protected", "state": "No browser secrets"},
        ),
        cta={
            "kicker": "Secure Access",
            "title": "Move from public overview to protected setup",
            "body": "Create access to review wallet, vault, provider, and strategy controls behind authenticated server-side gates.",
            "primary_label": "Create secure access",
            "primary_href": "/register",
            "secondary_label": "Review security",
            "secondary_href": "/security/",
        },
    ),
    "features": PublicSeoPage(
        key="features",
        endpoint="consumer.public_features",
        path="/features/",
        nav_label="Features",
        title="AlgVault Features | Automated Trading Platform Controls",
        description=(
            "Explore AlgVault features for automated trading, execution state, strategy posture, analytics, "
            "provider readiness, risk controls, and account visibility in a premium crypto trading PWA."
        ),
        eyebrow="Platform features",
        heading="All the features you need to trade with confidence",
        lead=(
            "AlgVault brings automated trading, execution state, strategy posture, analytics, provider readiness, "
            "and account visibility into a compact operating workspace built for fast checks and protected actions."
        ),
        primary_label="View pricing",
        primary_href="/pricing/",
        secondary_label="Mobile PWA",
        secondary_href="/mobile/",
        highlights=(
            {"label": "Automation", "value": "Controlled", "detail": "Execution flows remain server-authoritative."},
            {"label": "Analytics", "value": "Explainable", "detail": "Confidence, blockers, and degraded states are visible."},
            {"label": "Operations", "value": "Real time", "detail": "Provider and runtime posture stay close to actions."},
        ),
        sections=(
            {
                "kicker": "Operational Rhythm",
                "title": "Built for repeated checks, not long reads",
                "body": "Short panels and direct labels make the site scan like a trading console on mobile.",
            },
            {
                "kicker": "Server Authority",
                "title": "Buttons do not bypass validation",
                "body": "Client readiness is advisory; account-sensitive work stays behind backend validation and risk gates.",
            },
        ),
        badges=(
            {"label": "Compact cards", "tone": "mobile"},
            {"label": "Transparent blockers", "tone": "alert"},
            {"label": "Exchange-style UI", "tone": "secure"},
        ),
        hero_rows=(
            {"label": "Signals", "value": "Contextual", "state": "Confidence shown"},
            {"label": "Vault cycles", "value": "Guarded", "state": "Readiness gated"},
            {"label": "Portfolio", "value": "Visible", "state": "Wallet-aware"},
        ),
        cards=(
            {
                "icon": "icon-vault",
                "kicker": "Platform",
                "title": "Features readiness",
                "body": "Stay ahead with a platform built for what's next.",
                "meta": "Readiness",
                "group": "platform",
            },
            {
                "icon": "icon-dashboard",
                "kicker": "Platform",
                "title": "Contextual",
                "body": "Real-time context that powers smarter decisions.",
                "meta": "Context",
                "group": "platform",
            },
            {
                "icon": "icon-activity",
                "kicker": "Platform",
                "title": "Cycle-based execution",
                "body": "Automate with precision at every market cycle.",
                "meta": "Cycles",
                "group": "platform",
            },
            {
                "icon": "icon-wallet",
                "kicker": "Platform",
                "title": "Portfolio visibility",
                "body": "See your entire portfolio in one clear view.",
                "meta": "Portfolio",
                "group": "platform",
            },
            {
                "icon": "icon-convert",
                "kicker": "Automation",
                "title": "Guarded execution workflows",
                "body": "Server-side routing and readiness checks before trading-sensitive actions.",
                "meta": "Automation",
                "group": "automation",
            },
            {
                "icon": "icon-shield",
                "kicker": "Automation",
                "title": "Strategy posture at a glance",
                "body": "Cycles, conditions, readiness, and no-trade zones stay visible.",
                "meta": "Automation",
                "group": "automation",
            },
            {
                "icon": "icon-activity",
                "kicker": "Automation",
                "title": "Signals with control",
                "body": "Forecasts are treated with confidence, data quality, and risk context.",
                "meta": "Automation",
                "group": "automation",
            },
            {
                "icon": "icon-wallet",
                "kicker": "Automation",
                "title": "Wallet and vault clarity",
                "body": "Balances, allocations, conversions, and cycle states in one operating system.",
                "meta": "Automation",
                "group": "automation",
            },
            {
                "icon": "icon-markets",
                "kicker": "Automation",
                "title": "Provider-aware workflows",
                "body": "Connectivity and sync status are visible before protected actions.",
                "meta": "Automation",
                "group": "automation",
            },
            {
                "icon": "icon-shield",
                "kicker": "Execution",
                "title": "Gates before action",
                "body": "Readiness, policy, size, slippage, and future state required.",
                "meta": "Execution",
                "group": "automation",
            },
            {
                "icon": "icon-dashboard",
                "kicker": "System visibility",
                "title": "Runtime posture in view",
                "body": "Operational status surfaced without exposing private infrastructure.",
                "meta": "System visibility",
                "group": "automation",
            },
            {
                "icon": "icon-activity",
                "kicker": "Operational safety",
                "title": "Built for repeated checks, not long reads",
                "body": "Short panels and dense tables make the site scan fast.",
                "meta": "Operational safety",
                "group": "automation",
            },
            {
                "icon": "icon-alert",
                "kicker": "System safety",
                "title": "Buttons do not bypass validation",
                "body": "Every action has backend validation and signed audit.",
                "meta": "System safety",
                "group": "automation",
            },
        ),
        cta={
            "kicker": "Ready to get started?",
            "title": "Experience AlgVault the smart way",
            "body": "Join traders who automate smarter, trade safer, and move faster with server-side controls.",
            "primary_label": "Create access",
            "primary_href": "/register",
            "secondary_label": "Explore pricing",
            "secondary_href": "/pricing/",
        },
        disclaimer=(
            "AlgVault provides automation, analytics, and operational tooling. It does not provide investment advice or guaranteed trading outcomes."
        ),
    ),
    "pricing": PublicSeoPage(
        key="pricing",
        endpoint="consumer.public_pricing",
        path="/pricing/",
        nav_label="Pricing",
        title="AlgVault Pricing | Automated Trading Platform Access",
        description=(
            "Review AlgVault access tiers for mobile strategy monitoring, automated trading controls, "
            "broker/API connectivity, portfolio visibility, and operational analytics."
        ),
        eyebrow="Pricing",
        heading="Simple access tiers for monitored automation",
        lead=("Choose the level of AlgVault access that matches how much operational visibility, connectivity, and control you need."),
        primary_label="Create account",
        primary_href="/register",
        secondary_label="Compare features",
        secondary_href="/features/",
        highlights=(
            {"label": "Setup", "value": "2FA-aware", "detail": "Secure account creation before private tools."},
            {"label": "Billing", "value": "No hype", "detail": "No guaranteed return or investment advice claims."},
            {"label": "Controls", "value": "Server-led", "detail": "Sensitive workflows stay behind validation gates."},
        ),
        sections=(),
        badges=(
            {"label": "Transparent tiers", "tone": "secure"},
            {"label": "No fake urgency", "tone": "mobile"},
            {"label": "No performance claims", "tone": "alert"},
        ),
        hero_rows=(
            {"label": "Starter", "value": "Access", "state": "PWA + monitoring"},
            {"label": "Operator", "value": "Controls", "state": "Vault workflows"},
            {"label": "Desk", "value": "Custom", "state": "Ops support"},
        ),
        cards=(
            {
                "kicker": "Starter",
                "title": "Monitor",
                "meta": "Access",
                "body": "For evaluating AlgVault and tracking core operating posture.",
                "features": ("iPhone PWA access", "Strategy and wallet visibility", "Security-first account setup"),
            },
            {
                "kicker": "Operator",
                "title": "Control",
                "meta": "Core",
                "body": "For users who need monitored automation with provider and vault controls.",
                "features": ("Automated execution controls", "Broker/API connectivity", "Risk and readiness states"),
            },
            {
                "kicker": "Desk",
                "title": "Scale",
                "meta": "Custom",
                "body": "For advanced workflows that require operational review and custom setup.",
                "features": ("Expanded monitoring surfaces", "Operational audit visibility", "Custom onboarding path"),
            },
        ),
        status_rows=(
            {"state": "Included", "label": "Mobile PWA", "detail": "Installable app shell, iOS metadata, and safe-area support."},
            {"state": "Protected", "label": "Account tools", "detail": "Wallet, provider, and execution controls require sign-in."},
            {"state": "Explicit", "label": "Risk language", "detail": "Plans describe tools and controls, not trading outcomes."},
        ),
        cta={
            "kicker": "Access",
            "title": "Start with secure setup",
            "body": "Final plan details are confirmed during account setup. AlgVault does not promise profits or remove trading risk.",
            "primary_label": "Create account",
            "primary_href": "/register",
            "secondary_label": "Review security",
            "secondary_href": "/security/",
        },
        disclaimer=(
            "Pricing content describes platform access and operational tooling only. AlgVault does not provide investment advice, "
            "guaranteed returns, or removal of trading risk."
        ),
    ),
    "mobile": PublicSeoPage(
        key="mobile",
        endpoint="consumer.public_mobile",
        path="/mobile/",
        nav_label="Mobile PWA",
        title="Mobile Trading PWA | AlgVault",
        description=(
            "AlgVault is an iPhone-focused mobile trading PWA with installability, safe-area support, "
            "touch-optimized controls, offline-ready assets, and app-style navigation."
        ),
        eyebrow="Mobile PWA",
        heading="An iPhone-ready trading command center",
        lead=(
            "AlgVault is designed to feel fast and app-like across iPhone Safari and installed PWA mode, "
            "with clear operational states and responsive controls."
        ),
        primary_label="Review connectivity",
        primary_href="/connectivity/",
        secondary_label="Security model",
        secondary_href="/security/",
        highlights=(
            {"label": "Install", "value": "Standalone", "detail": "Manifest, icons, theme color, and Apple metadata."},
            {"label": "Touch", "value": "44px+", "detail": "Thumb-friendly controls and compact navigation."},
            {"label": "Offline", "value": "Safe", "detail": "Cached shell assets without false action success."},
            {"label": "Safe areas", "value": "Ready", "detail": "Viewport-fit spacing for iPhone browser chrome."},
        ),
        sections=(
            {
                "kicker": "iOS Safari",
                "title": "Built around the mobile browser",
                "body": "Safe-area spacing, sticky navigation, responsive cards, and stable dimensions reduce jumpiness on small screens.",
            },
            {
                "kicker": "Installed Mode",
                "title": "Feels like a focused app surface",
                "body": "Dark status bars, concise panels, and app-like transitions support quick repeated checks.",
            },
            {
                "kicker": "Offline Safety",
                "title": "Offline behavior stays honest",
                "body": "Static assets can be cached while account actions continue to require fresh server validation.",
            },
        ),
        badges=(
            {"label": "iPhone Safari optimized", "tone": "mobile"},
            {"label": "Standalone install-ready", "tone": "secure"},
            {"label": "Explicit offline states", "tone": "alert"},
            {"label": "Reduced-motion aware", "tone": "secure"},
        ),
        hero_rows=(
            {"label": "Display mode", "value": "Standalone", "state": "Install ready"},
            {"label": "Tap targets", "value": "44px+", "state": "Thumb safe"},
            {"label": "Network loss", "value": "Explicit", "state": "No false success"},
        ),
        cards=(),
        status_rows=(
            {"state": "Fast", "label": "Perceived load", "detail": "Server-rendered public HTML and deferred shell scripts."},
            {"state": "Stable", "label": "No overflow", "detail": "Narrow-screen grids collapse without horizontal scrolling."},
            {"state": "Reduced", "label": "Motion", "detail": "Hover and transition effects respect reduced-motion preferences."},
        ),
        cta={
            "kicker": "Mobile Operations",
            "title": "Use AlgVault from a focused iPhone shell",
            "body": "The public PWA experience mirrors the same operational language used across the authenticated platform.",
            "primary_label": "Review connectivity",
            "primary_href": "/connectivity/",
            "secondary_label": "Security model",
            "secondary_href": "/security/",
        },
    ),
    "connectivity": PublicSeoPage(
        key="connectivity",
        endpoint="consumer.public_connectivity",
        path="/connectivity/",
        nav_label="Connectivity",
        title="Connectivity | AlgVault Broker and API Integrations",
        description=(
            "AlgVault presents broker/API connectivity, secure authentication, exchange connections, "
            "webhook and event systems, synchronization, and monitoring in a concise operating view."
        ),
        eyebrow="Connectivity",
        heading="Provider connections with readiness before execution",
        lead=(
            "AlgVault treats connectivity as operational infrastructure: authentication, synchronization, "
            "provider health, and live blockers stay visible before automation proceeds."
        ),
        primary_label="Review security",
        primary_href="/security/",
        secondary_label="View features",
        secondary_href="/features/",
        highlights=(
            {"label": "Broker/API", "value": "Checked", "detail": "Provider status appears before sensitive workflows."},
            {"label": "Auth", "value": "Secure", "detail": "Credentials stay behind protected server flows."},
            {"label": "Sync", "value": "Observable", "detail": "Account and event state are monitored for drift."},
        ),
        sections=(
            {
                "kicker": "Credential Boundary",
                "title": "No privileged provider secrets in public code",
                "body": "Browser screens can show state, but credentials, tokens, webhooks, and signing logic stay server-side.",
            },
            {
                "kicker": "Monitoring",
                "title": "Connectivity is part of risk posture",
                "body": "Disconnected, unauthorized, stale, failed sync, and recovery states are labeled before protected action.",
            },
        ),
        badges=(
            {"label": "Provider readiness", "tone": "secure"},
            {"label": "Webhook-aware", "tone": "mobile"},
            {"label": "Disconnected states visible", "tone": "alert"},
        ),
        hero_rows=(
            {"label": "Broker/API", "value": "Checked", "state": "Health visible"},
            {"label": "Account sync", "value": "Gated", "state": "Server-side"},
            {"label": "Execution route", "value": "Blocked if stale", "state": "Risk aware"},
        ),
        cards=(
            {
                "icon": "icon-markets",
                "kicker": "Broker/API connectivity",
                "title": "Provider state near every workflow",
                "body": "Connection health and account readiness stay visible before users reach action paths.",
                "meta": "Providers",
            },
            {
                "icon": "icon-shield",
                "kicker": "Secure authentication",
                "title": "Protected credential flows",
                "body": "Provider credentials remain behind authenticated server-side handling.",
                "meta": "Auth",
            },
            {
                "icon": "icon-convert",
                "kicker": "Exchange connections",
                "title": "Routing with current context",
                "body": "Automation checks provider, balance, market, and risk state before execution.",
                "meta": "Exchange",
            },
            {
                "icon": "icon-activity",
                "kicker": "Webhook/events",
                "title": "Events without leaking secrets",
                "body": "Operational events can be monitored without exposing raw hooks or tokens.",
                "meta": "Events",
            },
            {
                "icon": "icon-dashboard",
                "kicker": "Synchronization",
                "title": "Account data drift is visible",
                "body": "Failed syncs and stale snapshots are treated as first-class states.",
                "meta": "Sync",
            },
            {
                "icon": "icon-alert",
                "kicker": "Monitoring",
                "title": "Clear blockers before action",
                "body": "The UI labels degraded connectivity instead of implying automation is ready.",
                "meta": "Monitor",
            },
        ),
        status_rows=(
            {"state": "Connect", "label": "Broker/API", "detail": "Provider credentials are established through protected flows."},
            {"state": "Validate", "label": "Readiness", "detail": "Server checks account, balance, market, and risk context."},
            {"state": "Monitor", "label": "Sync/events", "detail": "State changes and failures are surfaced before action."},
        ),
        cta={
            "kicker": "Connectivity",
            "title": "Keep provider state visible and server-authoritative",
            "body": "AlgVault treats connectivity as operating context, not an invisible background dependency.",
            "primary_label": "Review security",
            "primary_href": "/security/",
            "secondary_label": "View features",
            "secondary_href": "/features/",
        },
    ),
    "security": PublicSeoPage(
        key="security",
        endpoint="consumer.public_security",
        path="/security/",
        nav_label="Security",
        title="Security & Risk Controls | AlgVault",
        description=(
            "AlgVault emphasizes encrypted connections, credential handling, authentication flows, access controls, "
            "infrastructure reliability, monitoring, auditing, and user visibility without fake compliance claims."
        ),
        eyebrow="Security",
        heading="Trust built through visible controls",
        lead=(
            "AlgVault keeps the public story calm and specific: protected access, server-side secrets, "
            "explicit risk states, and user visibility before sensitive trading or wallet actions."
        ),
        primary_label="Create secure access",
        primary_href="/register",
        secondary_label="Connectivity",
        secondary_href="/connectivity/",
        highlights=(
            {"label": "Auth", "value": "2FA-aware", "detail": "Protected account setup before private tools."},
            {"label": "Secrets", "value": "Server-side", "detail": "Credentials and signing material are not public assets."},
            {"label": "Audit", "value": "Visible", "detail": "Runtime and action states are designed to be inspectable."},
        ),
        sections=(
            {
                "kicker": "No Fake Compliance",
                "title": "Precise security language",
                "body": "The page describes implemented product posture without implying certifications that are not present.",
            },
            {
                "kicker": "User Control",
                "title": "Visibility before automation",
                "body": "Users should see disconnected, blocked, stale, and recovery states before sensitive action proceeds.",
            },
        ),
        badges=(
            {"label": "2FA-aware access", "tone": "secure"},
            {"label": "No browser secrets", "tone": "mobile"},
            {"label": "No fake certifications", "tone": "alert"},
        ),
        hero_rows=(
            {"label": "Connections", "value": "Encrypted", "state": "HTTPS first"},
            {"label": "Credentials", "value": "Server-side", "state": "Never public"},
            {"label": "Risk gates", "value": "Visible", "state": "Before action"},
        ),
        cards=(
            {
                "icon": "icon-shield",
                "kicker": "Encrypted connections",
                "title": "HTTPS public surface",
                "body": "Production pages and app metadata are served through the secure custom domain.",
                "meta": "Transport",
            },
            {
                "icon": "icon-settings",
                "kicker": "Credential handling",
                "title": "Secrets stay backend-owned",
                "body": "Broker keys, wallet material, signer tokens, and webhook secrets do not render in public templates.",
                "meta": "Secrets",
            },
            {
                "icon": "icon-login",
                "kicker": "Authentication flows",
                "title": "Protected setup paths",
                "body": "Private wallet, vault, settings, and admin surfaces require authenticated access.",
                "meta": "Auth",
            },
            {
                "icon": "icon-user-plus",
                "kicker": "Access controls",
                "title": "Private routes stay private",
                "body": "Sensitive app and API routes are noindexed and guarded server-side.",
                "meta": "Access",
            },
            {
                "icon": "icon-dashboard",
                "kicker": "Infrastructure reliability",
                "title": "Runtime readiness is explicit",
                "body": "Health and readiness are distinct so a served app does not imply trading readiness.",
                "meta": "Runtime",
            },
            {
                "icon": "icon-activity",
                "kicker": "Monitoring/auditing",
                "title": "Operational states are visible",
                "body": "Failed syncs, stale data, blocked actions, and recovery states are product states.",
                "meta": "Audit",
            },
            {
                "icon": "icon-alert",
                "kicker": "User visibility",
                "title": "Controls before action",
                "body": "The UI focuses on monitoring, readiness, and control instead of outcome promises.",
                "meta": "Control",
            },
        ),
        status_rows=(
            {"state": "Blocked", "label": "Unauthorized", "detail": "Protected paths redirect and remain noindexed."},
            {"state": "Redacted", "label": "Secrets", "detail": "Public pages do not expose tokens, keys, credentials, or signer details."},
            {"state": "Paused", "label": "Risk gate", "detail": "Trading actions remain blocked when runtime readiness is degraded."},
        ),
        cta={
            "kicker": "Security Model",
            "title": "Keep automation observable and gated",
            "body": "AlgVault presents controls plainly, without compliance claims or guaranteed-performance language.",
            "primary_label": "Create secure access",
            "primary_href": "/register",
            "secondary_label": "Connectivity",
            "secondary_href": "/connectivity/",
        },
        disclaimer=(
            "Security content describes product posture only. It does not imply SOC, ISO, PCI, or other certifications unless AlgVault "
            "publishes those certifications separately."
        ),
    ),
}

PUBLIC_PAGE_KEYS = tuple(PUBLIC_PAGES)
PUBLIC_ENDPOINTS = frozenset(page.endpoint for page in PUBLIC_PAGES.values())
PUBLIC_PATHS = frozenset(page.path for page in PUBLIC_PAGES.values())

PRIVATE_PATH_PREFIXES = (
    "/admin",
    "/admin/api",
    "/api",
    "/_internal",
    "/wallet",
    "/vault",
    "/convert",
    "/settings",
    "/dashboard",
    "/backtests",
    "/panic",
)
PRIVATE_EXACT_PATHS = {"/setup-2fa", "/setup-2fa/", "/logout", "/logout/"}
AUTH_PATHS = {"/login", "/login/", "/register", "/register/"}


def canonical_origin(app: Flask) -> str:
    configured = str(app.config.get("SEO_CANONICAL_ORIGIN") or CANONICAL_ORIGIN).strip().rstrip("/")
    return configured or CANONICAL_ORIGIN


def canonical_url(app: Flask, path: str = "/") -> str:
    normalized = _normalize_path(path)
    return canonical_origin(app) + normalized


def public_navigation() -> tuple[dict[str, str], ...]:
    return tuple({"label": page.nav_label, "href": page.path, "key": page.key} for page in PUBLIC_PAGES.values())


def public_page(key: str) -> PublicSeoPage:
    return PUBLIC_PAGES[key]


def public_sitemap_pages() -> tuple[PublicSeoPage, ...]:
    return tuple(PUBLIC_PAGES[key] for key in PUBLIC_PAGE_KEYS)


def seo_context(app: Flask, *, endpoint: str | None, path: str, authenticated: bool = False) -> dict[str, Any]:
    normalized_path = _normalize_path(path)
    page = _public_page_for_endpoint(endpoint, normalized_path)
    if page is not None and not (authenticated and normalized_path == "/"):
        return _public_seo_context(app, page)

    noindex = should_noindex_path(normalized_path, endpoint=endpoint, authenticated=authenticated)
    title = _fallback_title(endpoint, normalized_path)
    description = _fallback_description(normalized_path, noindex=noindex)
    robots = "noindex, nofollow" if noindex else "index, follow"
    return {
        "title": title,
        "description": description,
        "canonical_url": canonical_url(app, normalized_path),
        "robots": robots,
        "og_title": title,
        "og_description": description,
        "og_type": "website",
        "og_url": canonical_url(app, normalized_path),
        "og_image": canonical_url(app, SOCIAL_IMAGE_PATH),
        "twitter_card": "summary_large_image",
        "twitter_title": title,
        "twitter_description": description,
        "twitter_image": canonical_url(app, SOCIAL_IMAGE_PATH),
        "json_ld": [],
        "is_public": False,
        "is_noindex": noindex,
    }


def should_noindex_path(path: str, *, endpoint: str | None = None, authenticated: bool = False) -> bool:
    normalized = _normalize_path(path)
    if endpoint == "static" or normalized.startswith("/static/") or normalized.startswith("/icons/"):
        return False
    if normalized in {"/healthz", "/readyz", "/ops/status", "/manifest.json", "/manifest.webmanifest", "/sw.js", "/favicon.ico"}:
        return False
    if normalized in AUTH_PATHS or normalized in PRIVATE_EXACT_PATHS:
        return True
    if endpoint == "consumer.home" and normalized == "/" and not authenticated:
        return True
    if authenticated and normalized == "/":
        return True
    return any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in PRIVATE_PATH_PREFIXES)


def is_public_indexable_path(path: str, *, endpoint: str | None = None, authenticated: bool = False) -> bool:
    normalized = _normalize_path(path)
    return normalized in PUBLIC_PATHS and endpoint in PUBLIC_ENDPOINTS and not (authenticated and normalized == "/")


def is_seo_asset_path(path: str) -> bool:
    normalized = _normalize_path(path)
    return normalized in {"/robots.txt", "/sitemap.xml"}


def robots_txt(app: Flask) -> str:
    disallow = (
        "/admin/",
        "/admin/api/",
        "/api/",
        "/_internal/",
        "/wallet/",
        "/vault/",
        "/convert/",
        "/settings/",
        "/setup-2fa",
        "/logout",
        "/dashboard",
        "/backtests/",
        "/panic/",
    )
    lines = [
        "User-agent: *",
        "Allow: /overview/",
        "Allow: /features/",
        "Allow: /pricing/",
        "Allow: /mobile/",
        "Allow: /connectivity/",
        "Allow: /security/",
    ]
    lines.extend(f"Disallow: {path}" for path in disallow)
    lines.extend(("", f"Sitemap: {canonical_url(app, '/sitemap.xml')}"))
    return "\n".join(lines) + "\n"


def sitemap_xml(app: Flask) -> str:
    urlset = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for page in public_sitemap_pages():
        priority = "1.0" if page.key == "home" else "0.8"
        changefreq = "weekly" if page.key == "home" else "monthly"
        urlset.extend(
            [
                "  <url>",
                f"    <loc>{escape(canonical_url(app, page.path))}</loc>",
                f"    <changefreq>{changefreq}</changefreq>",
                f"    <priority>{priority}</priority>",
                "  </url>",
            ]
        )
    urlset.append("</urlset>")
    return "\n".join(urlset) + "\n"


def _public_page_for_endpoint(endpoint: str | None, path: str) -> PublicSeoPage | None:
    for page in PUBLIC_PAGES.values():
        if endpoint == page.endpoint or path == page.path:
            return page
    return None


def _public_seo_context(app: Flask, page: PublicSeoPage) -> dict[str, Any]:
    page_url = canonical_url(app, page.path)
    social_image = canonical_url(app, SOCIAL_IMAGE_PATH)
    schemas = [
        _organization_schema(app),
        _website_schema(app),
        _software_schema(app),
        _service_schema(app),
        _breadcrumb_schema(app, page),
    ]
    return {
        "title": page.title,
        "description": page.description,
        "canonical_url": page_url,
        "robots": "index, follow, max-image-preview:large",
        "og_title": page.title,
        "og_description": page.description,
        "og_type": "website",
        "og_url": page_url,
        "og_image": social_image,
        "twitter_card": "summary_large_image",
        "twitter_title": page.title,
        "twitter_description": page.description,
        "twitter_image": social_image,
        "json_ld": schemas,
        "is_public": True,
        "is_noindex": False,
    }


def _organization_schema(app: Flask) -> dict[str, Any]:
    origin = canonical_origin(app)
    return {
        "@context": "https://schema.org",
        "@type": "Organization",
        "@id": f"{origin}/#organization",
        "name": BRAND_NAME,
        "url": origin + "/",
        "logo": canonical_url(app, SOCIAL_IMAGE_PATH),
        "sameAs": [],
    }


def _website_schema(app: Flask) -> dict[str, Any]:
    origin = canonical_origin(app)
    return {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "@id": f"{origin}/#website",
        "name": BRAND_NAME,
        "url": origin + "/",
        "publisher": {"@id": f"{origin}/#organization"},
    }


def _software_schema(app: Flask) -> dict[str, Any]:
    origin = canonical_origin(app)
    return {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "@id": f"{origin}/#software",
        "name": BRAND_NAME,
        "applicationCategory": "FinanceApplication",
        "operatingSystem": "Web, iOS, Android, macOS, Windows",
        "url": origin + "/",
        "description": PUBLIC_PAGES["home"].description,
        "publisher": {"@id": f"{origin}/#organization"},
        "offers": {"@type": "Offer", "category": "SaaS", "priceCurrency": "USD"},
    }


def _service_schema(app: Flask) -> dict[str, Any]:
    origin = canonical_origin(app)
    return {
        "@context": "https://schema.org",
        "@type": "Service",
        "@id": f"{origin}/#automated-trading-platform",
        "name": "AlgVault automated trading platform",
        "provider": {"@id": f"{origin}/#organization"},
        "serviceType": "Automated trading monitoring and execution controls",
        "url": origin + "/",
        "description": "Mobile-first trading automation infrastructure for monitoring, analytics, broker connectivity, and risk-aware controls.",
        "areaServed": "US",
    }


def _breadcrumb_schema(app: Flask, page: PublicSeoPage) -> dict[str, Any]:
    items: list[dict[str, Any]] = [
        {"@type": "ListItem", "position": 1, "name": "AlgVault", "item": canonical_url(app, "/")},
    ]
    if page.path != "/":
        items.append({"@type": "ListItem", "position": 2, "name": page.nav_label, "item": canonical_url(app, page.path)})
    return {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }


def _normalize_path(path: str) -> str:
    value = str(path or "/").split("?", 1)[0].split("#", 1)[0].strip() or "/"
    if not value.startswith("/"):
        value = f"/{value}"
    if value != "/" and not value.endswith("/") and "." not in value.rsplit("/", 1)[-1]:
        value += "/"
    return value


def _fallback_title(endpoint: str | None, path: str) -> str:
    endpoint_title = {
        "auth.login": "Sign In | AlgVault",
        "auth.register": "Create Account | AlgVault",
        "auth.setup_2fa": "Set Up 2FA | AlgVault",
    }.get(str(endpoint or ""))
    if endpoint_title:
        return endpoint_title
    clean = path.strip("/").replace("-", " ").replace("_", " ")
    if not clean:
        return "AlgVault"
    return f"{clean.title()} | AlgVault"


def _fallback_description(path: str, *, noindex: bool) -> str:
    if noindex:
        return "Protected AlgVault account surface for authenticated trading, wallet, risk, and operational workflows."
    return PUBLIC_PAGES["home"].description
