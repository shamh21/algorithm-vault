(() => {
  let shellInitialized = false;
  let startupFailed = false;
  let startupComplete = false;
  const startupStartedAt = window.performance?.now?.() || Date.now();
  const STARTUP_MIN_VISIBLE_MS = 520;
  const STARTUP_TIMEOUT_MS = 9000;
  const BOTTOM_NAV_SWITCH_KEY = "av-bottom-nav-switch";

  const startupNodes = () => {
    const loader = document.querySelector("[data-intro-loader]");
    return {
      loader,
      title: loader?.querySelector("[data-intro-loader-title]"),
      detail: loader?.querySelector("[data-intro-loader-detail]"),
      retry: loader?.querySelector("[data-intro-loader-retry]"),
    };
  };

  const finishStartupLoader = (skipDelay = false) => {
    if (startupComplete) return;
    startupComplete = true;
    window.clearTimeout(startupTimeout);

    const elapsed = (window.performance?.now?.() || Date.now()) - startupStartedAt;
    const delay = skipDelay ? 0 : Math.max(0, STARTUP_MIN_VISIBLE_MS - elapsed);
    window.setTimeout(() => {
      const { loader } = startupNodes();
      document.body.classList.remove("app-starting", "app-startup-failed");
      document.body.classList.add("app-ready");
      if (loader) {
        loader.setAttribute("aria-busy", "false");
        loader.setAttribute("aria-hidden", "true");
      }
    }, delay);
  };

  const showStartupFailure = (error) => {
    if (startupComplete || startupFailed) return;
    startupFailed = true;
    if (error) {
      console.error("Application startup failed", error);
    }
    const { loader, title, detail, retry } = startupNodes();
    document.body.classList.add("app-startup-failed");
    loader?.setAttribute("aria-busy", "false");
    if (title) title.textContent = "Startup needs attention";
    if (detail) detail.textContent = "The app shell did not finish initializing. Retry when your connection is available.";
    if (retry) retry.hidden = false;
  };

  const startupTimeout = window.setTimeout(() => {
    console.warn("AlgVault startup timed out; releasing intro screen.");
    finishStartupLoader(true);
  }, STARTUP_TIMEOUT_MS);

  const finishStartupAfterRouteReady = (skipDelay = false) => {
    if (document.readyState === "complete") {
      finishStartupLoader(skipDelay);
      return;
    }
    window.addEventListener("load", () => finishStartupLoader(skipDelay), { once: true });
  };

  const initShell = () => {
    if (shellInitialized) return;
    shellInitialized = true;

    const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const prefersReducedMotion = reducedMotionQuery.matches;
    const consumeBottomNavSwitch = () => {
      try {
        const switchedFromBottomNav = window.sessionStorage.getItem(BOTTOM_NAV_SWITCH_KEY) === "true";
        window.sessionStorage.removeItem(BOTTOM_NAV_SWITCH_KEY);
        return switchedFromBottomNav;
      } catch {
        return false;
      }
    };
    const switchedFromBottomNav = consumeBottomNavSwitch();
    const toggle = document.querySelector("[data-nav-toggle]");
    const nav = document.querySelector("[data-primary-nav]");
    const backdrop = document.querySelector("[data-nav-backdrop]");
    const topbar = document.querySelector("[data-app-topbar]");

    document.body.classList.toggle("reduced-motion", prefersReducedMotion);

    const closeNav = () => {
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
    }

    if (topbar) {
      const updateTopbar = () => topbar.classList.toggle("is-scrolled", window.scrollY > 8);
      updateTopbar();
      window.addEventListener("scroll", updateTopbar, { passive: true });
    }

    document.querySelectorAll(".bottom-nav .bottom-nav-item").forEach((link) => {
      link.addEventListener("click", (event) => {
        if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        if (link.target && link.target !== "_self") return;
        try {
          const url = new URL(link.getAttribute("href") || "", window.location.href);
          if (url.origin === window.location.origin) {
            window.sessionStorage.setItem(BOTTOM_NAV_SWITCH_KEY, "true");
          }
        } catch {}
      });
    });

    document.querySelectorAll("[data-row-cap]").forEach((container) => {
      const cap = Math.max(1, Number.parseInt(container.getAttribute("data-row-cap") || "150", 10));
      const rows = Array.from(container.querySelectorAll("tbody tr"));
      rows.slice(cap).forEach((row) => row.remove());
    });

    let resizeFrame = 0;
    const handleResize = () => {
      if (resizeFrame) return;
      resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        document.documentElement.style.setProperty("--app-vh", `${window.innerHeight * 0.01}px`);
      });
    };
    handleResize();
    window.addEventListener("resize", handleResize, { passive: true });

    document.querySelectorAll("[data-disable-on-submit]").forEach((form) => {
      form.addEventListener("submit", () => {
        const button = form.querySelector("button[type='submit']");
        if (!button) return;
        button.disabled = true;
        button.dataset.originalLabel = button.textContent;
        button.textContent = button.getAttribute("data-loading-label") || "Processing...";
      });
    });

    if (!prefersReducedMotion && !switchedFromBottomNav) {
      const revealTargets = Array.from(
        document.querySelectorAll(".card, .wallet-card, .vault-card, .banner, [data-flash-message]")
      ).slice(0, 18);

      if (window.IntersectionObserver) {
        const observer = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const element = entry.target;
            const index = revealTargets.indexOf(element);
            element.style.animationDelay = `${Math.min(Math.max(index, 0) * 35, 180)}ms`;
            element.classList.add("ui-rise-in");
            observer.unobserve(element);
          });
        }, { rootMargin: "24px 0px" });

        revealTargets.forEach((element) => observer.observe(element));
      } else {
        revealTargets.forEach((element, index) => {
          element.style.animationDelay = `${Math.min(index * 35, 180)}ms`;
          element.classList.add("ui-rise-in");
        });
      }
    }

    const registerServiceWorker = () => {
      const supportsServiceWorker = "serviceWorker" in navigator;
      const hostname = window.location.hostname;
      const isLocalhost = ["localhost", "127.0.0.1", "::1"].includes(hostname);
      const isPrivateIP =
        /^10\./.test(hostname) ||
        /^192\.168\./.test(hostname) ||
        /^172\.(1[6-9]|2[0-9]|3[0-1])\./.test(hostname);
      const isTrustedHttps = window.location.protocol === "https:" && !isPrivateIP;
      const isPrivateHttps = window.location.protocol === "https:" && isPrivateIP;
      if (isPrivateHttps && !window.AlgVaultConfig?.isProduction) {
        console.warn("AlgVault is running from a private IP HTTPS origin. iOS Safari may show a certificate warning. Use a stable trusted HTTPS hostname instead.");
      }
      if (!supportsServiceWorker || !(isTrustedHttps || isLocalhost) || !window.AV_SW_URL) {
        return;
      }

      let reloadOnControllerChange = false;
      const showUpdateBanner = (worker) => {
        if (!worker || document.querySelector("[data-sw-update-banner]")) return;
        const banner = document.createElement("div");
        banner.className = "sw-update-banner";
        banner.dataset.swUpdateBanner = "true";
        const copy = document.createElement("span");
        copy.textContent = "Update ready";
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Reload";
        button.addEventListener("click", () => {
          reloadOnControllerChange = true;
          worker.postMessage({ type: "SKIP_WAITING" });
        });
        banner.append(copy, button);
        document.body.append(banner);
      };

      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (!reloadOnControllerChange) return;
        reloadOnControllerChange = false;
        window.location.reload();
      });

      window.addEventListener("load", () => {
        navigator.serviceWorker.register(window.AV_SW_URL, { scope: window.AV_SW_SCOPE || "/" }).then((registration) => {
          if (registration.waiting) {
            showUpdateBanner(registration.waiting);
          }
          registration.addEventListener("updatefound", () => {
            const worker = registration.installing;
            if (!worker) return;
            worker.addEventListener("statechange", () => {
              if (worker.state === "installed" && navigator.serviceWorker.controller) {
                showUpdateBanner(worker);
              }
            });
          });
        }).catch(() => {});
      }, { once: true });
    };

    registerServiceWorker();
    finishStartupAfterRouteReady(switchedFromBottomNav);
  };

  const runInitShell = () => {
    try {
      initShell();
    } catch (error) {
      showStartupFailure(error);
    }
  };

  const { retry } = startupNodes();
  retry?.addEventListener("click", () => {
    window.location.reload();
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", runInitShell, { once: true });
  } else {
    runInitShell();
  }
})();
