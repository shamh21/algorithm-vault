(() => {
  const page = document.querySelector("[data-risk-page]");
  if (!page) return;

  const initialNode = document.getElementById("initial-risk-state");
  const initialState = initialNode ? JSON.parse(initialNode.textContent || "{}") : {};
  const csrf = page.querySelector("[data-risk-csrf]")?.value || "";
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;

  let controls = { ...(initialState.controls || {}) };
  let savedControls = { ...controls };
  let nextAuditPage = page.dataset.nextAuditPage ? Number(page.dataset.nextAuditPage) : null;
  let loadingAudit = false;
  let unlimitedConfirmed = Boolean(controls.daily_loss_unlimited);

  const refs = {
    saveBar: page.querySelector("[data-save-bar]"),
    saveStatus: page.querySelector("[data-save-status]"),
    lossCard: page.querySelector("[data-loss-card]"),
    lossUnlimited: page.querySelector("[data-loss-unlimited]"),
    lossSlider: page.querySelector("[data-loss-slider]"),
    lossInput: page.querySelector("[data-loss-input]"),
    lossValue: page.querySelector("[data-loss-value]"),
    lossCaption: page.querySelector("[data-loss-caption]"),
    lossWarning: page.querySelector("[data-loss-warning]"),
    leverageSlider: page.querySelector("[data-leverage-slider]"),
    leverageValue: page.querySelector("[data-leverage-value]"),
    leverageCap: page.querySelector("[data-leverage-cap]"),
    profileCurrent: page.querySelector("[data-profile-current]"),
    healthRing: page.querySelector("[data-health-ring]"),
    healthScore: page.querySelector("[data-health-score]"),
    safetyStatus: page.querySelector("[data-safety-status]"),
    volatilityState: page.querySelector("[data-volatility-state]"),
    exchangeMax: page.querySelector("[data-exchange-max]"),
    latencyAverage: page.querySelector("[data-latency-average]"),
    slippageEstimate: page.querySelector("[data-slippage-estimate]"),
    slippageConfidence: page.querySelector("[data-slippage-confidence]"),
    marketQuality: page.querySelector("[data-market-quality]"),
    executionHealth: page.querySelector("[data-execution-health]"),
    slippageChart: page.querySelector("[data-slippage-chart]"),
    exchangeList: page.querySelector("[data-exchange-list]"),
    latencyList: page.querySelector("[data-latency-list]"),
    auditFeed: page.querySelector("[data-audit-feed]"),
    auditLoader: page.querySelector("[data-audit-loader]"),
    loadMore: page.querySelector("[data-load-more]"),
    modal: page.querySelector("[data-risk-modal]"),
  };

  const numberValue = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const formatPct = (value) => `${Math.round(numberValue(value, 0))}%`;
  const formatMs = (value) => `${Math.round(numberValue(value, 0))}ms`;
  const formatLeverage = (value) => `${Math.round(numberValue(value, 0))}x`;
  const profileLabel = (value) => String(value || "balanced").replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

  const markDirty = () => {
    const dirty = JSON.stringify(controls) !== JSON.stringify(savedControls);
    if (refs.saveBar) refs.saveBar.hidden = !dirty;
    if (dirty && refs.saveStatus) refs.saveStatus.textContent = "Review before applying.";
  };

  const syncLossControls = () => {
    const unlimited = Boolean(controls.daily_loss_unlimited);
    refs.lossCard?.classList.toggle("is-unlimited", unlimited);
    if (refs.lossUnlimited) refs.lossUnlimited.checked = unlimited;
    if (refs.lossSlider) {
      refs.lossSlider.value = controls.daily_loss_limit_pct;
      refs.lossSlider.disabled = unlimited;
    }
    if (refs.lossInput) {
      refs.lossInput.value = Math.round(numberValue(controls.daily_loss_limit_pct, 0));
      refs.lossInput.disabled = unlimited;
    }
    if (refs.lossValue) refs.lossValue.textContent = Math.round(numberValue(controls.daily_loss_limit_pct, 0));
    if (refs.lossCaption) refs.lossCaption.textContent = unlimited ? "Unlimited mode armed" : "of detected capital";
    if (refs.lossWarning) refs.lossWarning.hidden = !unlimited;
  };

  const syncLeverageControls = () => {
    const max = numberValue(initialState.exchange_limits?.max_exchange_leverage, numberValue(controls.max_leverage, 1));
    controls.max_leverage = Math.max(0, Math.min(numberValue(controls.max_leverage, 0), max || numberValue(controls.max_leverage, 0)));
    if (refs.leverageSlider) {
      refs.leverageSlider.max = max;
      refs.leverageSlider.value = controls.max_leverage;
    }
    if (refs.leverageValue) refs.leverageValue.textContent = Math.round(numberValue(controls.max_leverage, 0));
    if (refs.leverageCap) refs.leverageCap.textContent = `${formatLeverage(max)} max`;
  };

  const syncProfileControls = () => {
    page.querySelectorAll("[data-risk-profile]").forEach((button) => {
      const active = button.dataset.riskProfile === controls.profile;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-checked", active ? "true" : "false");
    });
    if (refs.profileCurrent) refs.profileCurrent.textContent = profileLabel(controls.profile);
  };

  const renderControls = () => {
    syncLossControls();
    syncLeverageControls();
    syncProfileControls();
    markDirty();
  };

  const updateStateMetrics = (state) => {
    if (!state || typeof state !== "object") return;
    initialState.exchange_limits = state.exchange_limits || initialState.exchange_limits;
    if (refs.healthRing) refs.healthRing.style.setProperty("--risk-score", numberValue(state.health_score, 0));
    if (refs.healthScore) refs.healthScore.textContent = Math.round(numberValue(state.health_score, 0));
    if (refs.safetyStatus) refs.safetyStatus.textContent = state.safety_engine_status?.label || "Active";
    if (refs.volatilityState) refs.volatilityState.textContent = state.volatility_state || "Calm";
    if (refs.exchangeMax) refs.exchangeMax.textContent = formatLeverage(state.exchange_limits?.max_exchange_leverage);
    if (refs.latencyAverage) refs.latencyAverage.textContent = formatMs(state.latency?.average_ms);
    if (refs.slippageEstimate) refs.slippageEstimate.textContent = `${numberValue(state.adaptive_slippage?.estimate_bps, 0).toFixed(1)} bps`;
    if (refs.slippageConfidence) refs.slippageConfidence.textContent = formatPct(state.adaptive_slippage?.confidence);
    if (refs.marketQuality) refs.marketQuality.textContent = formatPct(state.adaptive_slippage?.market_quality);
    if (refs.executionHealth) refs.executionHealth.textContent = formatPct(state.adaptive_slippage?.execution_health);
    if (refs.slippageChart && Array.isArray(state.adaptive_slippage?.micro_chart)) {
      refs.slippageChart.innerHTML = state.adaptive_slippage.micro_chart
        .map((value) => `<span style="height: ${Math.max(numberValue(value, 0), 8)}%"></span>`)
        .join("");
    }
    renderExchangeList(state.exchange_limits?.providers || []);
    renderLatencyList(state.latency?.providers || []);
    renderControls();
  };

  const renderExchangeList = (providers) => {
    if (!refs.exchangeList) return;
    refs.exchangeList.innerHTML = providers.length
      ? providers.map((provider) => `<span>${provider.label} · ${formatLeverage(provider.max_leverage)} · ${provider.market_count || 0} markets</span>`).join("")
      : "<span>No active exchange market metadata yet.</span>";
  };

  const renderLatencyList = (providers) => {
    if (!refs.latencyList) return;
    refs.latencyList.innerHTML = providers.length
      ? providers.map((provider) => `
          <div>
            <span>${provider.label}</span>
            <strong>${formatMs(provider.latency_ms)}</strong>
            <em>${provider.quality || "Unknown"}</em>
          </div>
        `).join("")
      : "<div><span>No verified exchange</span><strong>--</strong><em>Connect first</em></div>";
  };

  refs.lossSlider?.addEventListener("input", () => {
    controls.daily_loss_limit_pct = numberValue(refs.lossSlider.value, 0);
    syncLossControls();
    markDirty();
  }, { passive: true });

  refs.lossInput?.addEventListener("input", () => {
    controls.daily_loss_limit_pct = Math.max(0, Math.min(numberValue(refs.lossInput.value, 0), 100));
    syncLossControls();
    markDirty();
  });

  refs.lossUnlimited?.addEventListener("change", () => {
    if (refs.lossUnlimited.checked && !unlimitedConfirmed) {
      refs.lossUnlimited.checked = false;
      if (refs.modal) refs.modal.hidden = false;
      return;
    }
    controls.daily_loss_unlimited = refs.lossUnlimited.checked;
    syncLossControls();
    markDirty();
  });

  page.querySelector("[data-confirm-unlimited]")?.addEventListener("click", () => {
    unlimitedConfirmed = true;
    controls.daily_loss_unlimited = true;
    if (refs.modal) refs.modal.hidden = true;
    renderControls();
  });

  page.querySelector("[data-cancel-unlimited]")?.addEventListener("click", () => {
    unlimitedConfirmed = false;
    controls.daily_loss_unlimited = false;
    if (refs.modal) refs.modal.hidden = true;
    renderControls();
  });

  refs.leverageSlider?.addEventListener("input", () => {
    controls.max_leverage = numberValue(refs.leverageSlider.value, 0);
    syncLeverageControls();
    markDirty();
  }, { passive: true });

  page.querySelectorAll("[data-leverage-step]").forEach((button) => {
    button.addEventListener("click", () => {
      controls.max_leverage = numberValue(controls.max_leverage, 0) + numberValue(button.dataset.leverageStep, 0);
      syncLeverageControls();
      markDirty();
    });
  });

  page.querySelectorAll("[data-leverage-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      const max = numberValue(initialState.exchange_limits?.max_exchange_leverage, 1);
      controls.max_leverage = button.dataset.leveragePreset === "max" ? max : numberValue(button.dataset.leveragePreset, 1);
      syncLeverageControls();
      markDirty();
    });
  });

  page.querySelectorAll("[data-risk-profile]").forEach((button) => {
    button.addEventListener("click", () => {
      controls.profile = button.dataset.riskProfile || "balanced";
      syncProfileControls();
      markDirty();
    });
  });

  page.querySelector("[data-discard-risk]")?.addEventListener("click", () => {
    controls = { ...savedControls };
    unlimitedConfirmed = Boolean(controls.daily_loss_unlimited);
    renderControls();
  });

  page.querySelector("[data-save-risk]")?.addEventListener("click", async () => {
    if (refs.saveStatus) refs.saveStatus.textContent = "Saving...";
    try {
      const response = await fetch(apiUrl(page.dataset.saveUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({ ...controls, confirm_unlimited_loss: unlimitedConfirmed }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Risk controls were not saved.");
      savedControls = { ...(payload.controls || controls) };
      controls = { ...savedControls };
      if (refs.saveStatus) refs.saveStatus.textContent = "Saved.";
      updateStateMetrics(payload.state);
      markDirty();
    } catch (error) {
      if (refs.saveStatus) refs.saveStatus.textContent = error.message || "Save failed.";
      refs.saveBar.hidden = false;
    }
  });

  page.querySelector("[data-refresh-exchange]")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    try {
      const response = await fetch(apiUrl(page.dataset.refreshUrl), { method: "POST", headers: { "X-CSRF-Token": csrf } });
      const payload = await response.json();
      if (payload.state) updateStateMetrics(payload.state);
    } finally {
      button.disabled = false;
    }
  });

  const loadAuditPage = async () => {
    if (!nextAuditPage || loadingAudit) return;
    loadingAudit = true;
    if (refs.auditLoader) refs.auditLoader.hidden = false;
    try {
      const auditUrl = new URL(apiUrl(page.dataset.auditUrl));
      auditUrl.searchParams.set("page", nextAuditPage);
      const response = await fetch(auditUrl);
      const payload = await response.json();
      if (!response.ok || !payload.ok) return;
      refs.auditFeed?.insertAdjacentHTML("beforeend", payload.html || "");
      nextAuditPage = payload.has_next ? Number(payload.next_page) : null;
      if (refs.loadMore) refs.loadMore.hidden = !nextAuditPage;
    } finally {
      loadingAudit = false;
      if (refs.auditLoader) refs.auditLoader.hidden = true;
    }
  };

  refs.loadMore?.addEventListener("click", (event) => {
    event.preventDefault();
    loadAuditPage();
  });

  if ("IntersectionObserver" in window && refs.loadMore) {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) loadAuditPage();
    }, { rootMargin: "220px 0px" });
    observer.observe(refs.loadMore);
  }

  refs.auditFeed?.addEventListener("click", async (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const toggle = target?.closest("[data-toggle-diagnostics]");
    if (toggle) {
      const card = toggle.closest("[data-audit-card]");
      const drawer = card?.querySelector("[data-diagnostics]");
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", expanded ? "false" : "true");
      if (drawer) drawer.hidden = expanded;
      toggle.querySelector("span").textContent = expanded ? "+" : "-";
      return;
    }

    const copy = target?.closest("[data-copy-diagnostics]");
    if (copy) {
      try {
        await navigator.clipboard.writeText(copy.dataset.copyText || "");
        copy.classList.add("is-copied");
        window.setTimeout(() => copy.classList.remove("is-copied"), 800);
      } catch (error) {
        copy.classList.remove("is-copied");
      }
    }
  });

  const refreshState = async () => {
    try {
      const response = await fetch(apiUrl(page.dataset.stateUrl));
      if (!response.ok) return;
      updateStateMetrics(await response.json());
    } catch (error) {
      return;
    }
  };

  if (!prefersReducedMotion) {
    page.addEventListener("touchstart", () => {}, { passive: true });
  }

  renderControls();
  window.setInterval(refreshState, 30000);
})();
