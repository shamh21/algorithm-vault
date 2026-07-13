const CACHE_VERSION = "algvault-v22-redblack-shell-1";
const SHELL_CACHE = `${CACHE_VERSION}-shell`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;

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
  <meta name="theme-color" content="#030304">
  <title>AlgVault Offline</title>
  <style>
    html,body{margin:0;min-height:100%;background:#030304;color:#fafafa;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",Inter,system-ui,sans-serif}
    body{background:radial-gradient(circle at 50% 20%,rgba(255,31,54,.18),transparent 25rem),linear-gradient(180deg,#030304,#09090b 58%,#030304)}
    main{min-height:100svh;display:grid;place-items:center;padding:calc(2rem + env(safe-area-inset-top)) max(1.25rem,env(safe-area-inset-right)) calc(2rem + env(safe-area-inset-bottom)) max(1.25rem,env(safe-area-inset-left))}
    section{max-width:28rem;border:1px solid rgba(255,31,54,.34);border-radius:16px;padding:1.25rem;background:linear-gradient(180deg,rgba(18,18,22,.98),rgba(7,7,9,.98));box-shadow:0 22px 56px rgba(0,0,0,.55),0 0 38px rgba(255,31,54,.08)}
    span{color:#ff6877;font-size:.74rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
    h1{margin:.35rem 0;font-size:1.35rem} p{margin:0;color:#b8b8c0}
  </style>
</head>
<body><main><section><span>Offline</span><h1>AlgVault is offline</h1><p>Reconnect to refresh wallet, vault, and market data. The red/black application shell remains available.</p></section></main></body>
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
