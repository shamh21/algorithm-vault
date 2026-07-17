(() => {
  if (window.__AlgVaultAppShellBootstrapped) return;
  window.__AlgVaultAppShellBootstrapped = true;

  const BOTTOM_NAV_SWITCH_KEY = "av-bottom-nav-switch";
  const THEME_STORAGE_KEY = "av-color-theme";
  const INSTALL_DISMISSED_KEY = "av-ios-install-help-dismissed";
  const MOBILE_BREAKPOINT = "(max-width: 760px)";
  const DESKTOP_BREAKPOINT = "(min-width: 761px)";
  const STARTUP_TIMEOUT_MS = 7000;
  let initialized = false;
  let startupFinished = false;

  const startupTimeout = window.setTimeout(() => {
    if (startupFinished) return;
    document.body?.classList.add("app-startup-failed");
    const loader = document.querySelector("[data-intro-loader]");
    loader?.setAttribute("aria-hidden", "false");
    loader?.setAttribute("aria-busy", "false");
    const title = loader?.querySelector("[data-intro-loader-title]");
    const detail = loader?.querySelector("[data-intro-loader-detail]");
    const retry = loader?.querySelector("[data-intro-loader-retry]");
    if (title) title.textContent = "Startup paused";
    if (detail) detail.textContent = "The application shell did not finish loading. Check your connection and retry.";
    if (retry) retry.hidden = false;
  }, STARTUP_TIMEOUT_MS);

  const finishStartup = () => {
    if (startupFinished) return;
    startupFinished = true;
    window.clearTimeout(startupTimeout);
    document.body.classList.remove("app-starting", "app-startup-failed");
    document.body.classList.add("app-ready");
    const loader = document.querySelector("[data-intro-loader]");
    loader?.setAttribute("aria-busy", "false");
    loader?.setAttribute("aria-hidden", "true");
  };

  const safeStorage = {
    get(key, fallback = null) {
      try {
        return window.localStorage.getItem(key) ?? fallback;
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch {}
    },
  };

  const ensureSkipLink = () => {
    if (document.querySelector(".skip-link")) return;
    const main = document.querySelector("main");
    if (!main) return;
    if (!main.id) main.id = "main-content";
    const link = document.createElement("a");
    link.className = "skip-link";
    link.href = `#${main.id}`;
    link.textContent = "Skip to main content";
    document.body.prepend(link);
  };

  const initTheme = () => {
    const toggles = Array.from(document.querySelectorAll("[data-theme-toggle]"));
    const labels = Array.from(document.querySelectorAll("[data-theme-current-label]"));
    const themeMeta = document.querySelector('meta[name="theme-color"]:not([media])');
    const colorSchemeMeta = document.querySelector('meta[name="color-scheme"]');

    const apply = (theme) => {
      const nextTheme = theme === "light" ? "light" : "dark";
      document.documentElement.dataset.theme = nextTheme;
      const label = nextTheme === "dark" ? "Dark mode" : "Light mode";
      toggles.forEach((toggle) => {
        toggle.setAttribute("aria-pressed", String(nextTheme === "dark"));
        toggle.setAttribute("title", label);
        toggle.querySelectorAll("[data-theme-toggle-label]").forEach((node) => {
          node.textContent = label;
        });
      });
      labels.forEach((node) => {
        node.textContent = label;
      });
      themeMeta?.setAttribute("content", nextTheme === "dark" ? "#050507" : "#f7f7fa");
      colorSchemeMeta?.setAttribute("content", nextTheme === "dark" ? "dark" : "light");
    };

    apply(document.documentElement.dataset.theme || safeStorage.get(THEME_STORAGE_KEY, "dark"));
    toggles.forEach((toggle) => {
      toggle.addEventListener("click", () => {
        const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        apply(next);
        safeStorage.set(THEME_STORAGE_KEY, next);
      });
    });
  };

  const initReducedMotion = () => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => document.body.classList.toggle("reduced-motion", query.matches);
    sync();
    query.addEventListener?.("change", sync);
  };

  const initNavigationDrawer = () => {
    const toggle = document.querySelector("[data-nav-toggle]");
    const nav = document.querySelector("[data-primary-nav]");
    const backdrop = document.querySelector("[data-nav-backdrop]");
    if (!(toggle instanceof HTMLElement) || !(nav instanceof HTMLElement)) return;

    let scrollY = 0;
    let returnFocus = toggle;

    const focusable = () => Array.from(
      nav.querySelectorAll(
        "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])"
      )
    ).filter((item) => item instanceof HTMLElement && !item.hidden && item.getAttribute("aria-hidden") !== "true");

    const setInert = (value) => {
      if (value) nav.setAttribute("inert", "");
      else nav.removeAttribute("inert");
    };

    const lockScroll = () => {
      scrollY = window.scrollY || document.documentElement.scrollTop || 0;
      document.body.style.setProperty("--locked-scroll-y", `-${scrollY}px`);
      document.body.classList.add("nav-open", "scroll-locked");
    };

    const unlockScroll = (restore = true) => {
      const wasLocked = document.body.classList.contains("scroll-locked");
      document.body.classList.remove("nav-open", "scroll-locked");
      document.body.style.removeProperty("--locked-scroll-y");
      if (wasLocked && restore) window.scrollTo(0, scrollY);
    };

    const close = ({ restoreFocus = false, restoreScroll = true } = {}) => {
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", "Open navigation menu");
      nav.setAttribute("aria-hidden", "true");
      backdrop?.setAttribute("aria-hidden", "true");
      nav.classList.remove("is-open");
      setInert(true);
      unlockScroll(restoreScroll);
      if (restoreFocus && returnFocus instanceof HTMLElement) returnFocus.focus({ preventScroll: true });
    };

    const open = () => {
      returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : toggle;
      toggle.setAttribute("aria-expanded", "true");
      toggle.setAttribute("aria-label", "Close navigation menu");
      nav.setAttribute("aria-hidden", "false");
      backdrop?.setAttribute("aria-hidden", "false");
      setInert(false);
      nav.classList.add("is-open");
      lockScroll();
      window.requestAnimationFrame(() => focusable()[0]?.focus({ preventScroll: true }));
    };

    close({ restoreScroll: false });

    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (toggle.getAttribute("aria-expanded") === "true") close();
      else open();
    });

    backdrop?.addEventListener("click", () => close({ restoreFocus: true }));

    nav.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest("a[href], button[type='submit']")) close();
    });

    document.addEventListener("pointerdown", (event) => {
      if (toggle.getAttribute("aria-expanded") !== "true") return;
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (nav.contains(target) || toggle.contains(target)) return;
      close({ restoreFocus: true });
    });

    document.addEventListener("keydown", (event) => {
      if (toggle.getAttribute("aria-expanded") !== "true") return;
      if (event.key === "Escape") {
        event.preventDefault();
        close({ restoreFocus: true });
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusable();
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

    window.matchMedia(DESKTOP_BREAKPOINT).addEventListener?.("change", () => close());
    window.addEventListener("pagehide", () => close({ restoreScroll: false }));
  };

  const initActiveNavigation = () => {
    const pathname = window.location.pathname.replace(/\/+$/, "") || "/";
    document.querySelectorAll(".public-topnav a[href], .app-nav a[href]").forEach((link) => {
      if (!(link instanceof HTMLAnchorElement)) return;
      const target = new URL(link.href, window.location.href).pathname.replace(/\/+$/, "") || "/";
      const active = target === pathname || (target !== "/" && pathname.startsWith(`${target}/`));
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
  };

  const initBottomNavigation = () => {
    const links = Array.from(document.querySelectorAll(".bottom-nav .bottom-nav-item"));
    if (!links.length) return;

    const sync = () => {
      const pathname = window.location.pathname.replace(/\/+$/, "") || "/";
      const hash = window.location.hash || "#dashboard";
      links.forEach((link) => {
        if (!(link instanceof HTMLAnchorElement)) return;
        const url = new URL(link.href, window.location.href);
        const targetPath = url.pathname.replace(/\/+$/, "") || "/";
        const active = targetPath === pathname && (!url.hash || url.hash === hash);
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
      });
    };

    links.forEach((link) => {
      link.addEventListener("click", () => {
        try {
          window.sessionStorage.setItem(BOTTOM_NAV_SWITCH_KEY, "true");
        } catch {}
      });
    });
    sync();
    window.addEventListener("hashchange", sync, { passive: true });
  };

  const initViewportSizing = () => {
    let frame = 0;
    const update = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        const height = window.visualViewport?.height || window.innerHeight;
        document.documentElement.style.setProperty("--app-vh", `${height * 0.01}px`);
        document.documentElement.style.setProperty("--visual-viewport-height", `${height}px`);
      });
    };
    update();
    window.addEventListener("resize", update, { passive: true });
    window.visualViewport?.addEventListener("resize", update, { passive: true });
    window.visualViewport?.addEventListener("scroll", update, { passive: true });
    window.addEventListener("orientationchange", update, { passive: true });
  };

  const initForms = () => {
    const forms = Array.from(document.querySelectorAll("[data-disable-on-submit]"));
    forms.forEach((form) => {
      form.addEventListener("submit", () => {
        const button = form.querySelector("button[type='submit']");
        if (!(button instanceof HTMLButtonElement) || button.disabled) return;
        button.dataset.originalLabel = button.textContent || "Submit";
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
        button.textContent = button.getAttribute("data-loading-label") || "Processing…";
      });
    });

    window.addEventListener("pageshow", (event) => {
      if (!event.persisted) return;
      forms.forEach((form) => {
        const button = form.querySelector("button[type='submit']");
        if (!(button instanceof HTMLButtonElement) || !button.dataset.originalLabel) return;
        button.disabled = false;
        button.removeAttribute("aria-busy");
        button.textContent = button.dataset.originalLabel;
      });
    });
  };

  const initTopbar = () => {
    const topbar = document.querySelector("[data-app-topbar]");
    if (!topbar) return;
    const sync = () => topbar.classList.toggle("is-scrolled", window.scrollY > 8);
    sync();
    window.addEventListener("scroll", sync, { passive: true });
  };

  const initRowCaps = () => {
    document.querySelectorAll("[data-row-cap]").forEach((container) => {
      const cap = Math.max(1, Number.parseInt(container.getAttribute("data-row-cap") || "150", 10));
      Array.from(container.querySelectorAll("tbody tr")).slice(cap).forEach((row) => row.remove());
    });
  };

  const initIosInstallHelp = () => {
    const ua = window.navigator.userAgent || "";
    const isIos = /iphone|ipad|ipod/i.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const isSafari = /safari/i.test(ua) && !/crios|fxios|edgios|opios/i.test(ua);
    const isStandalone = window.navigator.standalone === true || window.matchMedia?.("(display-mode: standalone)")?.matches;
    if (!isIos || !isSafari || isStandalone || safeStorage.get(INSTALL_DISMISSED_KEY) === "true") return;

    let help = document.querySelector("[data-ios-install-help]");
    if (!help) {
      help = document.createElement("aside");
      help.className = "ios-install-help";
      help.dataset.iosInstallHelp = "true";
      help.setAttribute("aria-label", "Install AlgVault on iPhone");
      help.innerHTML = `
        <div><strong>Install AlgVault on iPhone</strong><span>In Safari, tap Share, then Add to Home Screen.</span></div>
        <button type="button" class="secondary" data-ios-install-dismiss>Dismiss</button>`;
      document.body.append(help);
    }
    help.hidden = false;
    help.querySelector("[data-ios-install-dismiss]")?.addEventListener("click", () => {
      help.hidden = true;
      safeStorage.set(INSTALL_DISMISSED_KEY, "true");
    });
  };

  const initServiceWorker = () => {
    if (!("serviceWorker" in navigator) || !window.AV_SW_URL) return;
    const trustedOrigin = window.location.protocol === "https:" || ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
    if (!trustedOrigin) return;

    let reloadOnControllerChange = false;
    const showUpdate = (worker) => {
      if (!worker || document.querySelector("[data-sw-update-banner]")) return;
      const banner = document.createElement("div");
      banner.className = "sw-update-banner";
      banner.dataset.swUpdateBanner = "true";
      banner.setAttribute("role", "status");
      banner.innerHTML = "<span>AlgVault update ready</span>";
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Reload";
      button.addEventListener("click", () => {
        reloadOnControllerChange = true;
        button.disabled = true;
        worker.postMessage({ type: "SKIP_WAITING" });
      });
      banner.append(button);
      document.body.append(banner);
    };

    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!reloadOnControllerChange) return;
      reloadOnControllerChange = false;
      window.location.reload();
    });

    window.addEventListener("load", () => {
      navigator.serviceWorker.register(window.AV_SW_URL, {
        scope: window.AV_SW_SCOPE || "/",
        updateViaCache: "none",
      }).then((registration) => {
        registration.update().catch(() => {});
        if (registration.waiting) showUpdate(registration.waiting);
        registration.addEventListener("updatefound", () => {
          const worker = registration.installing;
          worker?.addEventListener("statechange", () => {
            if (worker.state === "installed" && navigator.serviceWorker.controller) showUpdate(worker);
          });
        });
      }).catch(() => {});
    }, { once: true });
  };

  const init = () => {
    if (initialized) return;
    initialized = true;
    try {
      ensureSkipLink();
      initReducedMotion();
      initTheme();
      initNavigationDrawer();
      initActiveNavigation();
      initBottomNavigation();
      initViewportSizing();
      initForms();
      initTopbar();
      initRowCaps();
      initIosInstallHelp();
      initServiceWorker();
      window.requestAnimationFrame(() => window.requestAnimationFrame(finishStartup));
    } catch (error) {
      console.error("AlgVault shell initialization failed", error);
      finishStartup();
    }
  };

  document.querySelector("[data-intro-loader-retry]")?.addEventListener("click", () => window.location.reload());
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
