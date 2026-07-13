const CACHE_VERSION = "algvault-v23-ios-pwa-1";
const SHELL_CACHE = `${CACHE_VERSION}-shell`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;

const APP_SHELL = [
  "/static/css/app.css",
  "/static/css/public.css",
  "/static/css/algvault-theme.css",
  "/static/js/app-shell.js",
  "/static/js/mini-charts.js",
  "/static/js/responsive-tables.js",
  "/manifest.json",
  "/icons/algvault-ios-180.png",
  "/icons/algvault-ios-192.png",
  "/icons/algvault-ios-512.png",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/apple-touch-icon.png",
];

const OFFLINE_HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#030304">
  <meta name="color-scheme" content="dark">
  <title>AlgVault — Offline</title>
  <style>
    *{box-sizing:border-box}
    html,body{margin:0;min-height:100%;background:#030304;color:#f5f5f5;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",Inter,system-ui,sans-serif;-webkit-text-size-adjust:100%}
    body{background:radial-gradient(circle at 22% 8%,rgba(255,31,54,.18),transparent 22rem),radial-gradient(circle at 82% 10%,rgba(155,77,255,.18),transparent 20rem),linear-gradient(180deg,#030304 0%,#09090b 55%,#030304 100%);overscroll-behavior-y:none}
    main{min-height:100svh;display:grid;place-items:center;padding:max(2rem,env(safe-area-inset-top)) max(1.25rem,env(safe-area-inset-right)) max(2rem,env(safe-area-inset-bottom)) max(1.25rem,env(safe-area-inset-left))}
    section{width:min(100%,26rem);border:1px solid rgba(192,92,255,.28);border-radius:1rem;padding:1.5rem;background:linear-gradient(135deg,rgba(255,31,54,.08),rgba(155,77,255,.1)),linear-gradient(180deg,rgba(16,16,20,.98),rgba(6,6,9,.98));box-shadow:0 24px 56px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.05)}
    .chip{display:inline-flex;align-items:center;gap:.3rem;padding:.2rem .55rem;border-radius:999px;background:rgba(95,95,114,.14);border:1px solid rgba(95,95,114,.28);font-size:.68rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#9090a8;margin-bottom:.75rem}
    .dot{width:6px;height:6px;background:#5f5f72;border-radius:50%;flex-shrink:0}
    h1{margin:.25rem 0 .75rem;font-size:1.4rem;letter-spacing:-.03em;font-weight:700}
    p{margin:0 0 .6rem;color:#a0a0b2;font-size:.9rem;line-height:1.55}
    .note{font-size:.78rem;color:#6a6a7e;border-top:1px solid rgba(255,255,255,.06);padding-top:.75rem;margin-top:.5rem}
    a{display:inline-block;margin-top:1rem;padding:.65rem 1.25rem;border-radius:999px;background:linear-gradient(180deg,#e82035,#a8152a);color:#fff;font-size:.9rem;font-weight:700;text-decoration:none;border:none}
  </style>
</head>
<body>
<main>
  <section>
    <span class="chip"><span class="dot"></span>Offline</span>
    <h1>AlgVault is offline</h1>
    <p>You are in read-only mode. Wallet balances, vault state, and market data cannot be refreshed without a connection.</p>
    <p>Protected actions — execution, conversion, and account changes — require a fresh server response and are unavailable offline.</p>
    <p class="note">Static app assets are cached locally. No account action has been queued or confirmed while offline.</p>
    <a href="/">Try reconnecting</a>
  </section>
</main>
</body>
</html>`;

const isHtmlRequest = (request) => request.mode === "navigate" || Boolean(request.headers.get("accept")?.includes("text/html"));
const isStaticAsset = (url) => url.pathname.startsWith("/static/") || url.pathname.startsWith("/icons/");
const isApiRequest = (url) => url.pathname.startsWith("/api/") || url.pathname.startsWith("/admin/api/") || url.pathname.includes("/stream");
const isAuthPath = (url) => ["/login", "/logout", "/register"].some((path) => url.pathname === path || url.pathname.startsWith(`${path}/`));
const isDashboardHtml = (url) => url.pathname === "/admin/dashboard";
const isServiceWorkerAsset = (url) => url.pathname === "/sw.js" || url.pathname === "/static/js/sw.js" || url.pathname === "/manifest.json" || url.pathname === "/static/manifest.webmanifest";
const isCacheableStaticResponse = (response) => response && response.ok && response.type !== "opaqueredirect";

const cacheFirstStatic = async (request) => {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (isCacheableStaticResponse(response)) {
    cache.put(request, response.clone());
  }
  return response;
};

const networkOnly = async (request) => {
  return fetch(request, { cache: "no-store", credentials: "same-origin" });
};

const networkFirstNavigation = async (request) => {
  try {
    const response = await fetch(request, { cache: "no-store", credentials: "same-origin" });
    if (response && response.ok && response.type !== "opaqueredirect") {
      return response;
    }
    return response;
  } catch (error) {
    return new Response(OFFLINE_HTML, {
      status: 503,
      statusText: "Offline",
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  }
};

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    await cache.addAll(APP_SHELL);
    self.skipWaiting();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    const keep = new Set([SHELL_CACHE, STATIC_CACHE]);
    await Promise.all(
      names
        .filter((name) => {
          if (keep.has(name)) return false;
          return name.startsWith("algvault-") || name.startsWith("tradingbot-");
        })
        .map((name) => caches.delete(name))
    );
    await self.clients.claim();
  })());
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
  if (event.data && event.data.type === "CLEAR_CACHES") {
    event.waitUntil((async () => {
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((name) => name.startsWith("algvault-") || name.startsWith("tradingbot-"))
          .map((name) => caches.delete(name))
      );
    })());
  }
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (isApiRequest(url) || isAuthPath(url) || isServiceWorkerAsset(url)) {
    event.respondWith(networkOnly(request));
    return;
  }

  if (isStaticAsset(url)) {
    event.respondWith(cacheFirstStatic(request));
    return;
  }

  if (isHtmlRequest(request) || isDashboardHtml(url)) {
    event.respondWith(networkFirstNavigation(request));
  }
});
