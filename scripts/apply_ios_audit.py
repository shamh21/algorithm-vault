from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:120]!r}")
    write(path, text.replace(old, new, count))


def append_once(path: str, marker: str, block: str) -> None:
    text = read(path)
    if marker in text:
        return
    write(path, text.rstrip() + "\n\n" + block.strip() + "\n")


# Keep canonical metadata aligned with the hostname Vercel actually serves.
replace("app/services/seo.py", 'CANONICAL_ORIGIN = "https://algvault.app"', 'CANONICAL_ORIGIN = "https://www.algvault.app"')

replace(
    "app/services/seo.py",
    '''        highlights=(
            {"label": "Active Strategies", "value": "12", "detail": "Online strategy monitor."},
            {"label": "Connected Providers", "value": "4", "detail": "Healthy broker/API posture."},
            {"label": "System Latency", "value": "42ms", "detail": "Fast public operating shell."},
            {"label": "Risk Engine Status", "value": "Online", "detail": "Server-side controls visible."},
        ),''',
    '''        highlights=(
            {
                "label": "Execution authority",
                "value": "Server-side",
                "detail": "Protected actions require backend validation.",
            },
            {
                "label": "Mobile surface",
                "value": "iPhone-ready",
                "detail": "Safe-area-aware standalone PWA behavior.",
            },
            {
                "label": "Provider state",
                "value": "Explicit",
                "detail": "Ready, stale, disconnected, restricted, and failed remain distinct.",
            },
            {
                "label": "Trading outcomes",
                "value": "Not promised",
                "detail": "No guaranteed-return or investment-advice claims.",
            },
        ),''',
)
replace(
    "app/services/seo.py",
    '''        hero_rows=(
            {"label": "Strategy monitor", "value": "Online", "state": "Signals visible"},
            {"label": "Broker/API", "value": "Checked", "state": "Readiness first"},
            {"label": "Wallet controls", "value": "Protected", "state": "No browser secrets"},
        ),''',
    '''        hero_rows=(
            {"label": "Strategy monitor", "value": "View only", "state": "Signals require context"},
            {"label": "Broker/API", "value": "Server-validated", "state": "Readiness before action"},
            {"label": "Wallet controls", "value": "Authenticated", "state": "No browser secrets"},
        ),''',
)
replace(
    "app/services/seo.py",
    '"body": "Join traders who automate smarter, trade safer, and move faster with server-side controls.",',
    '"body": "Use a compact operating surface with server-side controls, explicit blockers, and no outcome promises.",',
)

# Durable application shell: skip link, iOS install help, production detection, cache busting.
replace(
    "templates/base.html",
    '''  <body class="app-body app-starting mode-{{ nav_mode|default('live') }}" data-safe-area>
    <svg class="app-icon-sprite" aria-hidden="true" focusable="false">''',
    '''  <body class="app-body app-starting mode-{{ nav_mode|default('live') }}" data-safe-area>
    <a class="skip-link" href="#main-content">Skip to main content</a>
    <svg class="app-icon-sprite" aria-hidden="true" focusable="false">''',
)
replace(
    "templates/base.html",
    '''    <div class="nav-backdrop" id="nav-backdrop" data-nav-backdrop aria-hidden="true"></div>

    {% if current_user and request.endpoint not in ['consumer.home', 'auth.login', 'auth.register', 'auth.setup_2fa'] %}''',
    '''    <div class="nav-backdrop" id="nav-backdrop" data-nav-backdrop aria-hidden="true"></div>

    {% if not current_user %}
      <aside class="ios-install-help" data-ios-install-help hidden aria-label="Install AlgVault on iPhone">
        <div>
          <strong>Install AlgVault on iPhone</strong>
          <span>In Safari, tap Share, then Add to Home Screen for the standalone PWA.</span>
        </div>
        <button type="button" class="secondary" data-ios-install-dismiss aria-label="Dismiss install help">Dismiss</button>
      </aside>
    {% endif %}

    {% if current_user and request.endpoint not in ['consumer.home', 'auth.login', 'auth.register', 'auth.setup_2fa'] %}''',
)
replace(
    "templates/base.html",
    "current_app.config.get('DEPLOYMENT_TARGET') in ['vps', 'production', 'prod', 'postgres']",
    "current_app.config.get('DEPLOYMENT_TARGET') in ['vercel', 'vps', 'production', 'prod', 'postgres']",
)
replace(
    "templates/base.html",
    "current_app.config.get('ASSET_VERSION', '1') ~ '-algvault-red-purple-theme-1'",
    "current_app.config.get('ASSET_VERSION', '1') ~ '-algvault-ios-audit-2'",
)

# Consolidate the working production drawer behavior into main and add keyboard focus containment.
replace(
    "static/js/app-shell.js",
    '''    const closeNav = () => {
      if (!toggle || !nav) return;
      toggle.setAttribute("aria-expanded", "false");
      nav.setAttribute("aria-hidden", "true");
      backdrop?.setAttribute("aria-hidden", "true");
      nav.classList.remove("is-open");
      document.body.classList.remove("nav-open");
    };

    const openNav = () => {
      if (!toggle || !nav) return;
      toggle.setAttribute("aria-expanded", "true");
      nav.setAttribute("aria-hidden", "false");
      backdrop?.setAttribute("aria-hidden", "false");
      nav.classList.add("is-open");
      document.body.classList.add("nav-open");
    };

    if (toggle && nav) {
      nav.setAttribute("aria-hidden", "true");

      toggle.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const isOpen = toggle.getAttribute("aria-expanded") === "true";
        if (isOpen) {
          closeNav();
        } else {
          openNav();
        }
      });

      nav.addEventListener("click", (event) => {
        if (event.target.closest("a")) {
          closeNav();
        }
      });

      backdrop?.addEventListener("click", closeNav);

      document.addEventListener("click", (event) => {
        if (toggle.getAttribute("aria-expanded") !== "true") return;
        const target = event.target;
        if (!(target instanceof Element)) return;
        if (nav.contains(target) || toggle.contains(target)) return;
        closeNav();
      });

      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          closeNav();
        }
      });

      window.matchMedia("(min-width: 761px)").addEventListener?.("change", closeNav);
    }''',
    '''    let lockedScrollY = 0;
    let navReturnFocus = null;

    const setNavInert = (isInert) => {
      if (!nav) return;
      if (isInert) nav.setAttribute("inert", "");
      else nav.removeAttribute("inert");
    };

    const focusableNavItems = () => Array.from(
      nav?.querySelectorAll("a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])") || []
    ).filter((item) => item instanceof HTMLElement && !item.hidden);

    const lockDocumentScroll = () => {
      lockedScrollY = window.scrollY || document.documentElement.scrollTop || 0;
      document.body.style.setProperty("--locked-scroll-y", `-${lockedScrollY}px`);
      document.body.classList.add("nav-open", "scroll-locked");
    };

    const unlockDocumentScroll = ({ restorePosition = true } = {}) => {
      const wasLocked = document.body.classList.contains("scroll-locked");
      document.body.classList.remove("nav-open", "scroll-locked");
      document.body.style.removeProperty("--locked-scroll-y");
      if (wasLocked && restorePosition) window.scrollTo(0, lockedScrollY);
    };

    const closeNav = ({ restoreFocus = false, restorePosition = true } = {}) => {
      if (!toggle || !nav) return;
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", "Open navigation menu");
      nav.setAttribute("aria-hidden", "true");
      backdrop?.setAttribute("aria-hidden", "true");
      nav.classList.remove("is-open");
      setNavInert(true);
      unlockDocumentScroll({ restorePosition });
      if (restoreFocus && navReturnFocus instanceof HTMLElement) {
        navReturnFocus.focus({ preventScroll: true });
      }
      navReturnFocus = null;
    };

    const openNav = () => {
      if (!toggle || !nav) return;
      navReturnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : toggle;
      toggle.setAttribute("aria-expanded", "true");
      toggle.setAttribute("aria-label", "Close navigation menu");
      nav.setAttribute("aria-hidden", "false");
      backdrop?.setAttribute("aria-hidden", "false");
      setNavInert(false);
      nav.classList.add("is-open");
      lockDocumentScroll();
      window.requestAnimationFrame(() => focusableNavItems()[0]?.focus({ preventScroll: true }));
    };

    if (toggle && nav) {
      nav.setAttribute("aria-hidden", "true");
      setNavInert(true);

      toggle.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const isOpen = toggle.getAttribute("aria-expanded") === "true";
        if (isOpen) closeNav();
        else openNav();
      });

      nav.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) return;
        if (target.closest("a, button[type='submit']")) closeNav();
      });

      backdrop?.addEventListener("click", () => closeNav({ restoreFocus: true }));

      document.addEventListener("click", (event) => {
        if (toggle.getAttribute("aria-expanded") !== "true") return;
        const target = event.target;
        if (!(target instanceof Element)) return;
        if (nav.contains(target) || toggle.contains(target)) return;
        closeNav({ restoreFocus: true });
      });

      document.addEventListener("keydown", (event) => {
        if (toggle.getAttribute("aria-expanded") !== "true") return;
        if (event.key === "Escape") {
          event.preventDefault();
          closeNav({ restoreFocus: true });
          return;
        }
        if (event.key !== "Tab") return;
        const items = focusableNavItems();
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      });

      window.matchMedia("(min-width: 761px)").addEventListener?.("change", () => closeNav());
      window.addEventListener("pagehide", () => closeNav({ restorePosition: false }));
    }''',
)
replace(
    "static/js/app-shell.js",
    '''    const registerServiceWorker = () => {''',
    '''    const initIosInstallHelp = () => {
      const help = document.querySelector("[data-ios-install-help]");
      if (!help) return;
      const dismissedKey = "av-ios-install-help-dismissed";
      const ua = window.navigator.userAgent || "";
      const isIos = /iphone|ipad|ipod/i.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
      const isSafari = /safari/i.test(ua) && !/crios|fxios|edgios/i.test(ua);
      const isStandalone = window.navigator.standalone === true || window.matchMedia?.("(display-mode: standalone)")?.matches;
      let dismissed = false;
      try { dismissed = window.localStorage.getItem(dismissedKey) === "true"; } catch {}
      if (isIos && isSafari && !isStandalone && !dismissed) help.hidden = false;
      help.querySelector("[data-ios-install-dismiss]")?.addEventListener("click", () => {
        help.hidden = true;
        try { window.localStorage.setItem(dismissedKey, "true"); } catch {}
      });
    };

    initIosInstallHelp();

    const registerServiceWorker = () => {''',
)

append_once(
    "static/css/algvault-theme.css",
    "/* iOS shell audit 2026-07-16 */",
    '''/* iOS shell audit 2026-07-16 */
html {
  min-height: 100%;
  scroll-padding-top: calc(5.5rem + env(safe-area-inset-top));
}

body,
.app-body {
  min-height: 100svh;
  min-height: 100dvh;
}

main [id],
section[id] {
  scroll-margin-top: calc(5.5rem + env(safe-area-inset-top));
}

.skip-link {
  position: fixed;
  z-index: 10000;
  inset-block-start: max(0.5rem, env(safe-area-inset-top));
  inset-inline-start: max(0.75rem, env(safe-area-inset-left));
  min-height: 44px;
  display: inline-flex;
  align-items: center;
  padding: 0.65rem 0.9rem;
  border: 1px solid var(--border-strong);
  border-radius: 0.75rem;
  background: #111116;
  color: #fff;
  transform: translateY(-180%);
  transition: transform 160ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

body.scroll-locked {
  position: fixed;
  inset-inline: 0;
  top: var(--locked-scroll-y, 0);
  width: 100%;
  overflow: hidden;
  overscroll-behavior: none;
}

.ios-install-help {
  position: fixed;
  z-index: 180;
  inset-inline: max(0.75rem, env(safe-area-inset-left)) max(0.75rem, env(safe-area-inset-right));
  bottom: max(0.75rem, env(safe-area-inset-bottom));
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.9rem;
  max-width: 42rem;
  margin-inline: auto;
  padding: 0.85rem;
  border: 1px solid var(--violet-border);
  border-radius: 1rem;
  background: linear-gradient(135deg, rgba(255, 31, 54, 0.12), rgba(155, 77, 255, 0.13)), rgba(8, 8, 10, 0.97);
  box-shadow: 0 18px 52px rgba(0, 0, 0, 0.48);
}

.ios-install-help[hidden] {
  display: none !important;
}

.ios-install-help div {
  min-width: 0;
}

.ios-install-help strong,
.ios-install-help span {
  display: block;
}

.ios-install-help span {
  margin-top: 0.2rem;
  color: var(--muted-strong);
  font-size: 0.86rem;
  line-height: 1.35;
}

.ios-install-help button,
.site-header a,
.site-header button,
.app-nav a,
.app-nav button,
.public-footer a {
  min-height: 44px;
}

.public-footer {
  padding-bottom: max(1.25rem, env(safe-area-inset-bottom));
}

@media (max-width: 760px) {
  input,
  select,
  textarea,
  .Input {
    min-height: 44px;
    font-size: 16px !important;
  }

  textarea {
    min-height: 7rem;
  }

  .ios-install-help {
    align-items: stretch;
    flex-direction: column;
  }

  .ios-install-help button {
    width: 100%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .skip-link {
    transition: none;
  }
}''',
)

# Remove fabricated balances, provider states, execution history, and do-nothing filter controls from public pages.
replace("templates/marketing/_components.html", "<strong>LIVE SYSTEM</strong>", "<strong>ILLUSTRATIVE UI</strong>")
replace("templates/marketing/_components.html", "<small>Portfolio value</small>\n            <strong>$128,420.58</strong>\n            <span>Protected operating view</span>", "<small>Authenticated balance</small>\n            <strong>Server-confirmed only</strong>\n            <span>No public balance is displayed</span>")
replace("templates/marketing/_components.html", "<article><span></span><div><small>Provider</small><strong>Hyperliquid Connected</strong></div></article>", "<article><span></span><div><small>Provider</small><strong>Readiness is server-validated</strong></div></article>")
replace("templates/marketing/_components.html", "<article><span></span><div><small>Risk engine</small><strong>Server controls active</strong></div></article>", "<article><span></span><div><small>Risk engine</small><strong>Required before action</strong></div></article>")
replace("templates/marketing/_components.html", "<article><span></span><div><small>Vault cycle</small><strong>Readiness checked</strong></div></article>", "<article><span></span><div><small>Vault cycle</small><strong>Pending until confirmed</strong></div></article>")

replace(
    "templates/marketing/page.html",
    '''    <div class="features-filter" role="toolbar" aria-label="Feature categories">
      <button class="is-active" type="button" aria-pressed="true">All features</button>
      <button type="button" aria-pressed="false">Platform</button>
      <button type="button" aria-pressed="false">Automation</button>
      <button type="button" aria-pressed="false">Risk &amp; Control</button>
      <button type="button" aria-pressed="false">Accessibility</button>
      <button type="button" aria-pressed="false">Security</button>
      <button type="button" aria-pressed="false">Insights</button>
    </div>''',
    '''    <nav class="features-filter" aria-label="Feature sections">
      <a class="is-active" href="#features-page-title" aria-current="location">Overview</a>
      <a href="#platform-features">Platform</a>
      <a href="#automation-features">Automation</a>
      <a href="#features-insight-title">Insights</a>
      <a href="#features-matters-title">Why it matters</a>
    </nav>''',
)
replace("templates/marketing/page.html", "<strong>Readiness online</strong>", "<strong>Illustrative state</strong>")
replace("templates/marketing/page.html", "<span><b>Provider</b><em>Synced</em></span>", "<span><b>Provider</b><em>State visible</em></span>")
replace("templates/marketing/page.html", "<span><b>Risk gate</b><em>Active</em></span>", "<span><b>Risk gate</b><em>Server-required</em></span>")
replace("templates/marketing/page.html", "<span><b>Cycle state</b><em>Ready</em></span>", "<span><b>Cycle state</b><em>Pending confirmation</em></span>")
replace("templates/marketing/page.html", "<article class=\"is-green\"><span aria-hidden=\"true\"></span><strong>Sync healthy</strong><p>Data updated just now</p></article>", "<article class=\"is-green\"><span aria-hidden=\"true\"></span><strong>Fresh-data state</strong><p>Shown only after a confirmed refresh</p></article>")

replace("templates/marketing/page.html", "<strong>Gateway live</strong>", "<strong>State model</strong>")
replace("templates/marketing/page.html", '<div class="connection-health-ring" aria-label="12 monitored connections">\n              <span>12</span>\n              <strong>Connections</strong>\n            </div>', '<div class="connection-health-ring" aria-label="Illustrative provider state model">\n              <span>AV</span>\n              <strong>State model</strong>\n            </div>')
replace("templates/marketing/page.html", "<dd>10</dd>", "<dd>Visible</dd>")
replace("templates/marketing/page.html", "<dd>1</dd>", "<dd>Explicit</dd>", count=2)
replace(
    "templates/marketing/page.html",
    '''      {% set connection_providers = [
        {"name": "Interactive Brokers", "type": "Broker", "status": "Connected", "mark": "IB", "key": "interactive-brokers"},
        {"name": "Tradovate", "type": "Broker", "status": "Available", "mark": "TV", "key": "tradovate"},
        {"name": "Binance", "type": "Exchange", "status": "Connected", "mark": "BN", "key": "binance"},
        {"name": "Coinbase Exchange", "type": "Exchange", "status": "Connected", "mark": "CB", "key": "coinbase"},
        {"name": "Bybit", "type": "Exchange", "status": "Available", "mark": "BY", "key": "bybit"},
        {"name": "Kraken", "type": "Exchange", "status": "Degraded", "mark": "KR", "key": "kraken"},
        {"name": "OANDA", "type": "Broker", "status": "Connected", "mark": "OA", "key": "oanda"},
        {"name": "dxFeed", "type": "Data", "status": "Connected", "mark": "DX", "key": "dxfeed"},
        {"name": "TradingView", "type": "Data", "status": "Connected", "mark": "TV", "key": "tradingview"},
        {"name": "Polygon.io", "type": "Data", "status": "Connected", "mark": "PG", "key": "polygon"},
        {"name": "Alpaca", "type": "Broker", "status": "Available", "mark": "AP", "key": "alpaca"},
        {"name": "More providers", "type": "Browse all integrations", "status": "Available", "mark": "+", "key": "more"}
      ] %}''',
    '''      {% set connection_providers = [
        {"name": "Hyperliquid", "type": "Exchange provider", "status": "Gated", "mark": "HL", "key": "hyperliquid"},
        {"name": "KuCoin", "type": "Eligibility-dependent provider", "status": "Restricted", "mark": "KC", "key": "kucoin"},
        {"name": "Additional providers", "type": "Shown only when configured", "status": "Configured", "mark": "+", "key": "more"}
      ] %}''',
)
replace("templates/marketing/page.html", "<h2 id=\"supported-connections-title\">Supported connections</h2>", "<h2 id=\"supported-connections-title\">Provider readiness model</h2>")
replace("templates/marketing/page.html", "<p>Connect to leading brokers, exchanges, and data providers.</p>", "<p>Providers appear only when implemented and remain gated by account, region, credential, and runtime eligibility.</p>")
replace("templates/marketing/page.html", "<h2 id=\"live-connectivity-status-title\">Live connectivity status</h2>", "<h2 id=\"live-connectivity-status-title\">Connectivity states</h2>")
replace("templates/marketing/page.html", "<p>Public operational posture without account-sensitive details.</p>", "<p>Illustrative state labels only; authenticated server responses determine actual readiness.</p>")
replace(
    "templates/marketing/page.html",
    '''        {% set operational_rows = [
          {"label": "Broker Gateway", "detail": "All systems operational", "icon": "icon-vault"},
          {"label": "Market Data", "detail": "Low latency feed active", "icon": "icon-markets"},
          {"label": "Order Routing", "detail": "Execution path verified", "icon": "icon-convert"},
          {"label": "Account Sync", "detail": "Positions and balances updated", "icon": "icon-activity"},
          {"label": "Notifications", "detail": "Delivery channels active", "icon": "icon-alert"}
        ] %}''',
    '''        {% set operational_rows = [
          {"label": "Provider gateway", "detail": "Connected, stale, restricted, and failed are separate states.", "icon": "icon-vault"},
          {"label": "Market data", "detail": "Freshness is checked before data is treated as current.", "icon": "icon-markets"},
          {"label": "Order routing", "detail": "Execution remains unavailable until server validation passes.", "icon": "icon-convert"},
          {"label": "Account sync", "detail": "Balances and positions require authenticated refresh.", "icon": "icon-activity"},
          {"label": "Notifications", "detail": "Delivery state is reported without implying action success.", "icon": "icon-alert"}
        ] %}''',
)
replace("templates/marketing/page.html", '<span class="connectivity-operational-badge">Operational</span>', '<span class="connectivity-operational-badge">State shown</span>')
replace("templates/marketing/page.html", "<h2 id=\"recent-connectivity-activity-title\">Recent activity</h2>", "<h2 id=\"recent-connectivity-activity-title\">Example state transitions</h2>")
replace("templates/marketing/page.html", "<p>Sample connection events only.</p>", "<p>Interface examples, not real account history.</p>")
replace(
    "templates/marketing/page.html",
    '''        {% set activity_rows = [
          {"provider": "Interactive Brokers", "event": "Connection established", "time": "2m ago", "tone": "connected", "mark": "IB"},
          {"provider": "Binance", "event": "Heartbeat successful", "time": "3m ago", "tone": "connected", "mark": "BN"},
          {"provider": "Kraken", "event": "Reconnected", "time": "7m ago", "tone": "connected", "mark": "KR"},
          {"provider": "OANDA", "event": "Authentication refreshed", "time": "11m ago", "tone": "connected", "mark": "OA"},
          {"provider": "Bybit", "event": "Temporary latency detected", "time": "18m ago", "tone": "degraded", "mark": "BY"}
        ] %}''',
    '''        {% set activity_rows = [
          {"provider": "Provider", "event": "Connection confirmed by server", "time": "Example", "tone": "connected", "mark": "OK"},
          {"provider": "Market data", "event": "Freshness window checked", "time": "Example", "tone": "connected", "mark": "MD"},
          {"provider": "Provider", "event": "Reconnect required", "time": "Example", "tone": "degraded", "mark": "RC"},
          {"provider": "Credentials", "event": "Authentication refresh required", "time": "Example", "tone": "degraded", "mark": "AU"},
          {"provider": "Network", "event": "Latency threshold exceeded", "time": "Example", "tone": "degraded", "mark": "NW"}
        ] %}''',
)
replace("templates/marketing/page.html", "<strong>Policy online</strong>", "<strong>Server policy required</strong>")
replace("templates/marketing/page.html", "<strong>Secure</strong>\n              <em>All systems operational</em>", "<strong>Validation required</strong>\n              <em>Illustrative control state</em>")

# PWA update safety: wait for explicit reload and cap cache growth.
replace("static/js/sw.js", 'const CACHE_VERSION = "algvault-v22-red-purple-theme-1";', 'const CACHE_VERSION = "algvault-v23-ios-audit";\nconst MAX_STATIC_ENTRIES = 80;')
replace(
    "static/js/sw.js",
    '''const cacheFirstStatic = async (request) => {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (isCacheableStaticResponse(response)) {
    cache.put(request, response.clone());
  }
  return response;
};''',
    '''const trimCache = async (cache, maxEntries) => {
  const keys = await cache.keys();
  const excess = keys.length - maxEntries;
  if (excess <= 0) return;
  await Promise.all(keys.slice(0, excess).map((key) => cache.delete(key)));
};

const cacheFirstStatic = async (request) => {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (isCacheableStaticResponse(response)) {
    await cache.put(request, response.clone());
    await trimCache(cache, MAX_STATIC_ENTRIES);
  }
  return response;
};''',
)
replace("static/js/sw.js", "    await cache.addAll(APP_SHELL);\n    self.skipWaiting();", "    await cache.addAll(APP_SHELL);")

manifest_path = ROOT / "static/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["orientation"] = "any"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

# CI now targets the active production hostname.
replace(".github/workflows/ci.yml", "PUBLIC_APP_ORIGIN: https://app.algvault.com", "PUBLIC_APP_ORIGIN: https://www.algvault.app")
replace(".github/workflows/ci.yml", "PUBLIC_API_ORIGIN: https://app.algvault.com", "PUBLIC_API_ORIGIN: https://www.algvault.app")

# Pure contract tests catch regressions without requiring live secrets.
test_path = ROOT / "tests/test_mobile_pwa_audit.py"
test_path.write_text(
    '''from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_marketing_has_no_fabricated_operational_data() -> None:
    content = "\n".join(
        [
            source("app/services/seo.py"),
            source("templates/marketing/_components.html"),
            source("templates/marketing/page.html"),
        ]
    )
    forbidden = (
        "$128,420.58",
        '"Active Strategies", "value": "12"',
        '"Connected Providers", "value": "4"',
        '"System Latency", "value": "42ms"',
        "All systems operational",
        "Interactive Brokers",
        '"Binance"',
        '"Kraken"',
        '"OANDA"',
        '"Bybit"',
    )
    for value in forbidden:
        assert value not in content


def test_mobile_drawer_restores_scroll_focus_and_inert_state() -> None:
    shell = source("static/js/app-shell.js")
    assert 'nav.removeAttribute("inert")' in shell
    assert 'nav.setAttribute("inert", "")' in shell
    assert 'window.scrollTo(0, lockedScrollY)' in shell
    assert 'navReturnFocus.focus({ preventScroll: true })' in shell
    assert 'event.key !== "Tab"' in shell


def test_ios_install_help_is_dismissible_and_not_shown_standalone() -> None:
    base = source("templates/base.html")
    shell = source("static/js/app-shell.js")
    assert "data-ios-install-help" in base
    assert "data-ios-install-dismiss" in base
    assert 'window.navigator.standalone === true' in shell
    assert "av-ios-install-help-dismissed" in shell


def test_vercel_is_recognized_as_production() -> None:
    base = source("templates/base.html")
    assert "['vercel', 'vps', 'production', 'prod', 'postgres']" in base


def test_manifest_and_service_worker_update_contract() -> None:
    manifest = json.loads(source("static/manifest.json"))
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    assert manifest["scope"] == "/"
    assert manifest["orientation"] == "any"
    assert any("maskable" in icon.get("purpose", "") for icon in manifest["icons"])

    worker = source("static/js/sw.js")
    assert 'const CACHE_VERSION = "algvault-v23-ios-audit"' in worker
    assert "MAX_STATIC_ENTRIES" in worker
    install_block = worker.split('self.addEventListener("install"', 1)[1].split('self.addEventListener("activate"', 1)[0]
    assert "self.skipWaiting()" not in install_block
    assert 'url.pathname.startsWith("/api/")' in worker
    assert 'fetch(request, { cache: "no-store", credentials: "same-origin" })' in worker


def test_feature_navigation_uses_real_links_not_inert_buttons() -> None:
    page = source("templates/marketing/page.html")
    section = page.split('<nav class="features-filter"', 1)[1].split("</nav>", 1)[0]
    assert "<a " in section
    assert "<button" not in section
''',
    encoding="utf-8",
)

report_path = ROOT / "docs/audits/ios-pwa-2026-07-16.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    '''# AlgVault iOS PWA audit — 2026-07-16

## Architecture

- Flask/Jinja application served on Vercel through `server.py`.
- Public and authenticated routes are Flask blueprints; private execution remains server-authoritative.
- Shared shell: `templates/base.html`, `static/css/app.css`, `static/css/public.css`, `static/css/algvault-theme.css`, and `static/js/app-shell.js`.
- PWA: `static/manifest.json` and `static/js/sw.js`, exposed as `/manifest.json` and `/sw.js`.
- Separate Next.js admin PWA under `admin-pwa`, validated by the existing CI workflow.

## Material findings addressed

- Production had working mobile-drawer fixes on an ephemeral Vercel branch that were absent from `main`.
- The public site displayed fabricated balances, provider counts, latency, readiness, provider integrations, and activity history.
- The feature-category controls looked interactive but did nothing.
- Vercel was not recognized as a production deployment in the client configuration.
- The mobile drawer did not remove `inert` on `main`, did not preserve scroll position, and did not contain keyboard focus.
- The service worker skipped waiting immediately while the UI expected an explicit update/reload flow, and its static cache had no entry cap.
- CI still used the retired `app.algvault.com` origin.

## Implementation

- Consolidated accessible drawer focus, inert, tap-outside, Escape, scroll-lock, and scroll-restoration behavior.
- Added a dismissible iPhone Safari install-help surface that stays hidden in standalone mode.
- Added skip navigation, mobile input sizing, safe-area spacing, modern viewport units, and 44px touch targets.
- Replaced fabricated public metrics and provider history with capability/state-model language.
- Converted inert feature filters to working anchor navigation.
- Aligned canonical metadata and CI with the active `www.algvault.app` host.
- Changed manifest orientation to `any` and improved service-worker update/cache lifecycle.
- Added regression contract tests for public claims, PWA metadata, service-worker behavior, and mobile navigation.
''',
    encoding="utf-8",
)

# Remove the one-shot bootstrap files from the resulting branch.
for temporary in (ROOT / "scripts/apply_ios_audit.py", ROOT / ".github/workflows/apply-ios-audit.yml"):
    if temporary.exists():
        temporary.unlink()
