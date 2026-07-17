const CACHE_VERSION = "algvault-v23-ios-audit";
const SHELL_CACHE = `${CACHE_VERSION}-shell`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const MAX_STATIC_ENTRIES = 80;

const APP_SHELL = [
  "/static/css/app.css",
  "/static/css/public.css",
  "/static/css/algvault-theme.css",
  "/static/js/app-shell.js",
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
  <meta name="theme-color" content="#050507">
  <title>AlgVault Offline</title>
  <style>
    :root{color-scheme:dark}*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:#050507;color:#f8f8fb;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text",Inter,system-ui,sans-serif}body{background:radial-gradient(circle at 20% 0,rgba(255,49,72,.17),transparent 24rem),radial-gradient(circle at 88% 8%,rgba(147,76,255,.18),transparent 22rem),linear-gradient(180deg,#050507,#09090d 60%,#040405)}main{min-height:100vh;min-height:100svh;min-height:100dvh;display:grid;place-items:center;padding:calc(2rem + env(safe-area-inset-top)) max(1rem,env(safe-area-inset-right)) calc(2rem + env(safe-area-inset-bottom)) max(1rem,env(safe-area-inset-left))}section{width:min(100%,30rem);border:1px solid rgba(180,98,255,.34);border-radius:18px;padding:1.25rem;background:linear-gradient(135deg,rgba(255,49,72,.09),rgba(147,76,255,.11)),rgba(12,12,16,.98);box-shadow:0 24px 64px rgba(0,0,0,.56)}span{color:#c69aff;font-size:.74rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em}h1{margin:.35rem 0;font-size:1.4rem}p{margin:.4rem 0 0;color:#c7c7d0;line-height:1.5}strong{display:block;margin-top:1rem;color:#ff788a;font-size:.88rem}
  </style>
</head>
<body><main><section><span>Offline</span><h1>AlgVault cannot refresh current state</h1><p>Reconnect to load wallet, vault, provider, market, and readiness data. Protected actions remain unavailable while offline.</p><strong>No cached response is treated as execution success.</strong></section></main></body>
</html>`;

const isHtmlRequest = (request) => request.mode === "navigate" || Boolean(request.headers.get("accept")?.includes("text/html"));
const isStaticAsset = (url) => url.pathname.startsWith("/static/") || url.pathname.startsWith("/icons/");
const isApiRequest = (url) => url.pathname.startsWith("/api/") || url.pathname.startsWith("/admin/api/") || url.pathname.includes("/stream");
const isAuthPath = (url) => ["/login", "/logout", "/register"].some((path) => url.pathname === path || url.pathname.startsWith(`${path}/`));
const isServiceWorkerAsset = (url) => url.pathname === "/sw.js" || url.pathname === "/static/js/sw.js" || url.pathname === "/manifest.json" || url.pathname === "/static/manifest.webmanifest";
const isCriticalShellAsset = (url) => [
  "/static/css/app.css",
  "/static/css/public.css",
  "/static/css/algvault-theme.css",
  "/static/js/app-shell.js",
  "/static/js/responsive-tables.js",
].includes(url.pathname);
const isCacheableStaticResponse = (response) => response && response.ok && response.type !== "opaqueredirect";

const trimCache = async (cache, maxEntries) => {
  const keys = await cache.keys();
  const excess = keys.length - maxEntries;
  if (excess <= 0) return;
  await Promise.all(keys.slice(0, excess).map((key) => cache.delete(key)));
};

const networkOnly = (request) => fetch(request, { cache: "no-store", credentials: "same-origin" });

const refreshCriticalAsset = async (request) => {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const response = await fetch(request, { cache: "reload", credentials: "same-origin" });
    if (isCacheableStaticResponse(response)) await cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;
    throw error;
  }
};

const cacheFirstStatic = async (request) => {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request, { credentials: "same-origin" });
  if (isCacheableStaticResponse(response)) {
    await cache.put(request, response.clone());
    await trimCache(cache, MAX_STATIC_ENTRIES);
  }
  return response;
};

const networkFirstNavigation = async (request) => {
  try {
    return await fetch(request, { cache: "no-store", credentials: "same-origin" });
  } catch {
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
    await Promise.allSettled(APP_SHELL.map(async (path) => {
      const request = new Request(path, { cache: "reload", credentials: "same-origin" });
      const response = await fetch(request);
      if (isCacheableStaticResponse(response)) await cache.put(request, response);
    }));
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    const keep = new Set([SHELL_CACHE, STATIC_CACHE]);
    await Promise.all(
      names
        .filter((name) => !keep.has(name) && (name.startsWith("algvault-") || name.startsWith("tradingbot-")))
        .map((name) => caches.delete(name))
    );
    await self.clients.claim();
  })());
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
  if (event.data?.type === "CLEAR_CACHES") {
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

  if (isCriticalShellAsset(url)) {
    event.respondWith(refreshCriticalAsset(request));
    return;
  }

  if (isStaticAsset(url)) {
    event.respondWith(cacheFirstStatic(request));
    return;
  }

  if (isHtmlRequest(request)) event.respondWith(networkFirstNavigation(request));
});
