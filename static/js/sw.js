const CACHE_VERSION = "v1";
const SHELL_CACHE = `tradingbot-shell-${CACHE_VERSION}`;
const RUNTIME_CACHE = `tradingbot-runtime-${CACHE_VERSION}`;

const APP_SHELL = [
  "/",
  "/wallet",
  "/vault",
  "/activity",
  "/settings/",
  "/static/css/app.css",
  "/static/js/vault.js",
  "/static/js/wallet.js",
  "/static/js/responsive-tables.js",
  "/static/js/sw.js",
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/apple-touch-icon.png",
];

const isHtmlRequest = (request) => {
  if (request.mode === "navigate") {
    return true;
  }
  const acceptsHtml = request.headers.get("accept")?.includes("text/html");
  return Boolean(acceptsHtml);
};

const isStaticAsset = (url) => {
  return url.pathname.startsWith("/static/");
};

const cacheFirst = async (request) => {
  const cache = await caches.open(RUNTIME_CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response && response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    return cached;
  }
};

const networkFirst = async (request) => {
  const cache = await caches.open(RUNTIME_CACHE);
  try {
    const response = await fetch(request);
    if (response && response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }

    if (isHtmlRequest(request)) {
      const fallback = await caches.match("/");
      if (fallback) {
        return fallback;
      }
    }

    return new Response("Offline", { status: 503, statusText: "Offline" });
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
    const cacheNames = await caches.keys();
    const outdated = cacheNames.filter((name) => {
      return ![SHELL_CACHE, RUNTIME_CACHE].includes(name) && (name.startsWith("tradingbot-shell-") || name.startsWith("tradingbot-runtime-"));
    });
    await Promise.all(outdated.map((name) => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;

  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (isStaticAsset(request)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  if (isHtmlRequest(request)) {
    event.respondWith(networkFirst(request));
    return;
  }

  event.respondWith(cacheFirst(request));
});
