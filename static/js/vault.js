(() => {
  const countdowns = Array.from(document.querySelectorAll("[data-countdown]"));
  const activeCycleCards = Array.from(document.querySelectorAll("[data-vault-cycle-card]"));
  const settlementManuallyChanged = new WeakSet();
  const previewControllers = new WeakMap();
  const previewTimers = new WeakMap();
  const lastPreview = new WeakMap();
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;
  let timerId = null;
  let cycleStatusTimerId = null;

  function randomKey() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function formatRemaining(ms) {
    if (ms <= 0) return "Ready for settlement";
    const totalSeconds = Math.floor(ms / 1000);
    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor((totalSeconds % 86400) / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    return `${minutes}m ${seconds}s`;
  }

  function updateCountdowns() {
    const now = Date.now();
    countdowns.forEach((element) => {
      const unlocksAt = Date.parse(element.dataset.unlocksAt || "");
      if (Number.isNaN(unlocksAt)) return;
      element.textContent = formatRemaining(unlocksAt - now);
    });
  }

  function startTimers() {
    if (document.hidden) return;
    if (countdowns.length && !timerId) {
      updateCountdowns();
      timerId = window.setInterval(updateCountdowns, 1000);
    }
    if (activeCycleCards.length && !cycleStatusTimerId) {
      activeCycleCards.forEach((card) => refreshCycleCard(card));
      cycleStatusTimerId = window.setInterval(() => {
        activeCycleCards.forEach((card) => refreshCycleCard(card));
      }, 20000);
    }
  }

  function stopTimers() {
    if (timerId) {
      window.clearInterval(timerId);
      timerId = null;
    }
    if (cycleStatusTimerId) {
      window.clearInterval(cycleStatusTimerId);
      cycleStatusTimerId = null;
    }
  }

  function numberValue(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatMoney(value) {
    const amount = numberValue(value, 0);
    return `$${amount.toLocaleString(undefined, {
      minimumFractionDigits: amount >= 100 ? 0 : 2,
      maximumFractionDigits: amount >= 100 ? 0 : 2,
    })}`;
  }

  function formatPercent(value) {
    const raw = numberValue(value, 0);
    const percent = raw > 1 ? raw : raw * 100;
    return `${Math.round(percent)}%`;
  }

  function titleCase(value) {
    return String(value || "")
      .replace(/([a-z])([A-Z])/g, "$1 $2")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function readinessLabel(value) {
    const raw = String(value || "caution");
    if (raw === "notReady") return "Not Ready";
    return titleCase(raw);
  }

  function directionLabel(value) {
    return `${titleCase(value || "neutral")} horizon signal`;
  }

  function updateText(root, selector, value) {
    const element = root.querySelector(selector);
    if (element) element.textContent = value;
  }

  function renderCycleRiskNotes(card, notes) {
    const container = card.querySelector("[data-cycle-risk-notes]");
    if (!container) return;
    clearElement(container);
    const rows = Array.isArray(notes) && notes.length ? notes.slice(0, 2) : ["Strategy coherence active"];
    rows.forEach((note) => {
      const item = document.createElement("span");
      item.textContent = String(note || "");
      container.appendChild(item);
    });
  }

  function updateCycleCard(card, payload) {
    const status = payload?.cycle_status || {};
    const coherence = payload?.coherence_summary || {};
    const readiness = coherence.automationReadiness || "caution";
    updateText(card, "[data-cycle-phase]", status.phaseLabel || titleCase(status.phase || "collectingData"));
    updateText(card, "[data-cycle-readiness]", `Automation readiness: ${readinessLabel(readiness)}`);
    updateText(card, "[data-cycle-direction]", directionLabel(coherence.overallDirection));
    updateText(card, "[data-cycle-confidence]", `${Math.round(numberValue(coherence.overallConfidence, 0))}%`);
    updateText(card, "[data-cycle-coherence]", `${Math.round(numberValue(coherence.coherenceScore, 0))}%`);
    updateText(card, "[data-cycle-summary]", coherence.summary || status.statusMessage || "Market forecast updated.");
    updateText(card, "[data-cycle-next]", `Next evaluation ${status.nextScheduled1h10Cycle || "pending"}`);
    renderCycleRiskNotes(card, coherence.riskNotes || []);
    card.classList.toggle("is-stale", Boolean(status.stale));
    card.classList.toggle("is-error", Boolean(status.error));
  }

  async function refreshCycleCard(card) {
    const url = card.dataset.cycleStatusUrl;
    if (!url || document.hidden) return;
    try {
      const response = await window.fetch(apiUrl(url), { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`Cycle status failed: ${response.status}`);
      updateCycleCard(card, await response.json());
    } catch (error) {
      updateText(card, "[data-cycle-phase]", "Status unavailable");
      renderCycleRiskNotes(card, ["Cycle status update failed"]);
      card.classList.add("is-error");
    }
  }

  function blockerText(blocker) {
    if (!blocker) return "";
    if (typeof blocker === "string") return blocker.replaceAll("_", " ");
    return blocker.title || blocker.description || blocker.code || "";
  }

  function blockerDetail(blocker) {
    if (!blocker) return "";
    if (typeof blocker === "string") return blocker.replaceAll("_", " ");
    return blocker.description || blocker.fix_hint || blocker.title || blocker.code || "";
  }

  function statusLabel(provider) {
    const status = provider.status || (provider.ready ? "ready" : "blocked");
    if (status === "ready") return "Ready";
    if (status === "disabled") return "Disabled";
    if (status === "not_connected") return "Connect";
    if (status === "blocked") return "Blocked";
    return "Checking";
  }

  function selectedAllocationOption(form) {
    return form?.querySelector("input[name='deposit_asset']:checked")?.closest("[data-asset-option]");
  }

  function selectedAllocationAsset(form) {
    return form?.querySelector("input[name='deposit_asset']:checked")?.value || "";
  }

  function selectedProviders(form) {
    return Array.from(form.querySelectorAll("[data-provider-toggle]:checked")).map((input) => input.value);
  }

  function settlementHasAsset(settlementSelect, asset) {
    if (!settlementSelect || !asset) return false;
    return Array.from(settlementSelect.options).some((option) => option.value === asset);
  }

  function syncSettlementToAllocation(form, manualOnly) {
    if (!form) return;
    const settlementSelect = form.querySelector("[data-settlement-asset]");
    const linkedSettlement = form.querySelector("[data-linked-settlement]");
    const asset = selectedAllocationAsset(form);
    if (settlementSelect && !settlementManuallyChanged.has(form) && settlementHasAsset(settlementSelect, asset)) {
      settlementSelect.value = asset;
    }
    if (linkedSettlement && asset && !manualOnly) {
      linkedSettlement.value = asset;
    }
  }

  function applySelectedOption(option) {
    const group = option.closest(".asset-selector");
    if (!group) return;
    group.querySelectorAll("[data-asset-option]").forEach((item) => item.classList.remove("is-selected"));
    option.classList.add("is-selected");
    const input = option.querySelector("input[type='radio']");
    if (input) {
      input.checked = true;
      const form = input.closest(".vault-form");
      if (input.name === "deposit_asset") {
        syncSettlementToAllocation(form, false);
      }
      scheduleRoutingPreview(form);
    }
  }

  function parseInitialPreview() {
    const script = document.getElementById("initial-vault-routing-preview");
    if (!script) return null;
    try {
      return JSON.parse(script.textContent || "null");
    } catch (error) {
      return null;
    }
  }

  function formPreviewState(form) {
    return {
      amount: numberValue(form.querySelector("[data-vault-amount]")?.value, 0),
      depositAsset: selectedAllocationAsset(form) || "USDC",
      settlementAsset: form.querySelector("[data-settlement-asset]")?.value || selectedAllocationAsset(form) || "USDC",
      providers: selectedProviders(form),
      acknowledged: form.querySelector("input[name='one_h10_live_ack']")?.checked || false,
    };
  }

  function providersFromPayload(payload) {
    if (Array.isArray(payload?.providers)) return payload.providers;
    const status = payload?.exchange_status || {};
    return Object.entries(status).map(([provider, row]) => ({ provider, ...(row || {}) }));
  }

  function routesFromPayload(payload, providers) {
    const routes = payload?.routing_preview?.routes;
    if (Array.isArray(routes) && routes.length) return routes;
    return providers
      .filter((provider) => provider.enabled && numberValue(provider.allocation_weight, 0) > 0)
      .map((provider) => ({
        exchange: provider.provider,
        label: provider.label,
        allocation_weight: provider.allocation_weight,
        allocation_pct: provider.allocation_pct,
        notional_usd: provider.notional_usd || provider.target_amount,
        target_amount: provider.target_amount,
      }));
  }

  function clearElement(element) {
    if (!element) return;
    while (element.firstChild) element.removeChild(element.firstChild);
  }

  function updateProviderCard(form, provider) {
    const key = provider.provider || provider.exchange;
    const card = form.querySelector(`[data-provider-card][data-provider="${key}"]`);
    if (!card) return;
    const input = card.querySelector("[data-provider-toggle]");
    const enabled = input?.checked ?? provider.enabled;
    const ready = Boolean(provider.ready) || provider.status === "ready";
    const status = enabled ? (ready ? "ready" : provider.status || "blocked") : "disabled";
    const topBlocker = Array.isArray(provider.blockers) ? provider.blockers.find((item) => item?.severity !== "info") || provider.blockers[0] : null;
    const blockerEl = card.querySelector("[data-provider-blocker]");

    card.classList.toggle("is-enabled", Boolean(enabled));
    card.classList.toggle("is-ready", status === "ready");
    card.classList.toggle("is-blocked", status === "blocked" || status === "not_connected");
    card.classList.toggle("is-disabled", !enabled);
    card.querySelector("[data-provider-status]").textContent = statusLabel({ ...provider, status });
    card.querySelector("[data-provider-score]").textContent =
      status === "ready" ? `${Math.round(numberValue(provider.score, numberValue(provider.routing_score, 0) * 100))} score` : statusLabel({ ...provider, status });
    card.querySelector("[data-provider-allocation]").textContent =
      numberValue(provider.allocation_pct, 0) > 0 ? formatPercent(provider.allocation_pct) : formatPercent(provider.allocation_weight);
    if (blockerEl) {
      const text = blockerText(topBlocker);
      blockerEl.textContent = text;
      blockerEl.hidden = !text || status === "ready" || !enabled;
    }
  }

  function renderBlockers(form, payload) {
    const active = Array.isArray(payload?.active_blockers) ? payload.active_blockers : Array.isArray(payload?.blockers) ? payload.blockers : [];
    const exchange = Array.isArray(payload?.exchange_blockers) ? payload.exchange_blockers : [];
    const blockers = active.length ? active : exchange;
    const top = form.querySelector("[data-vault-top-blockers]");
    const all = form.querySelector("[data-vault-all-blockers]");
    const count = form.querySelector("[data-routing-blocker-count]");
    const disabledReason = form.querySelector("[data-start-disabled-reason]");
    clearElement(top);
    clearElement(all);

    const countText = blockers.length ? `${blockers.length} active blocker${blockers.length === 1 ? "" : "s"}` : "Guardrails clear";
    if (count) count.textContent = countText;
    blockers.slice(0, 3).forEach((blocker) => {
      const item = document.createElement("div");
      item.className = `vault-blocker-chip is-${blocker.severity || "blocker"}`;
      item.textContent = blockerText(blocker);
      top?.appendChild(item);
    });
    (active.concat(exchange)).forEach((blocker) => {
      const item = document.createElement("div");
      item.className = `vault-diagnostic-item is-${blocker.severity || "blocker"}`;
      const title = document.createElement("strong");
      title.textContent = blockerText(blocker);
      const detail = document.createElement("span");
      detail.textContent = blockerDetail(blocker);
      item.append(title, detail);
      all?.appendChild(item);
    });
    if (disabledReason) {
      const first = blockers[0];
      disabledReason.textContent = payload?.ready ? "Ready to start." : blockerText(first) || "Readiness check is pending.";
    }
  }

  function updateStartState(form, payload) {
    const button = form.querySelector(".vault-start-button");
    if (!button) return;
    const ready = Boolean(payload?.ready);
    button.disabled = !ready || form.dataset.submitting === "1";
    button.classList.toggle("is-disabled", !ready);
  }

  function renderRoutingPreview(form, payload) {
    if (!form || !payload) return;
    lastPreview.set(form, payload);
    const providers = providersFromPayload(payload);
    providers.forEach((provider) => updateProviderCard(form, provider));

    const summary = payload.summary || {};
    const routingPreview = payload.routing_preview || {};
    const state = formPreviewState(form);
    const total = numberValue(routingPreview.notional_usd, numberValue(summary.allocated_total, 0));
    const readyCount = numberValue(payload.ready_exchange_count, numberValue(summary.ready_provider_count, 0));
    const selectedCount = numberValue(payload.total_exchange_count, numberValue(summary.selected_provider_count, state.providers.length));
    const engine = form.querySelector("[data-routing-engine]");
    const totalElement = form.querySelector("[data-routing-total]");
    const readiness = form.querySelector("[data-routing-readiness]");
    const stateLabel = form.querySelector("[data-vault-state-label]");
    const routingSummary = form.querySelector("[data-routing-summary]");
    const ml = form.querySelector("[data-routing-ml]");
    const list = form.querySelector("[data-routing-list]");
    const bars = form.querySelector("[data-routing-bars]");

    if (engine) engine.textContent = summary.allocation_engine || "1H10 Smart Router";
    if (totalElement) totalElement.textContent = state.amount > 0 ? formatMoney(total) : "Amount required";
    if (readiness) readiness.textContent = `${readyCount}/${selectedCount} exchanges ready`;
    if (stateLabel) stateLabel.textContent = payload.state_label || (payload.ready ? "Live Ready" : "Blocked");
    if (routingSummary) routingSummary.textContent = routingPreview.summary || summary.routing_summary || "Enter amount to generate route.";
    if (ml) {
      const mlReadiness = payload.ml_readiness || summary.ml_readiness || {};
      ml.textContent = mlReadiness.display_status || (mlReadiness.ready ? "ML ready" : "ML check");
    }

    renderBlockers(form, payload);
    updateStartState(form, payload);
    clearElement(list);
    clearElement(bars);

    const routes = routesFromPayload(payload, providers);
    if (!routes.length) {
      const empty = document.createElement("div");
      empty.className = "vault-routing-empty";
      empty.textContent = state.amount > 0 ? "No enabled exchange is ready for this amount." : "Enter amount to generate route.";
      list?.appendChild(empty);
      return;
    }

    routes.forEach((route) => {
      const row = document.createElement("div");
      row.className = "vault-routing-row";
      const label = document.createElement("span");
      label.textContent = route.label || route.exchange;
      const value = document.createElement("strong");
      value.textContent = `${formatMoney(route.notional_usd || route.target_amount)} · ${formatPercent(route.allocation_pct || route.allocation_weight)}`;
      row.append(label, value);
      list?.appendChild(row);

      if (bars) {
        const segment = document.createElement("span");
        const width = numberValue(route.allocation_pct, 0) || numberValue(route.allocation_weight, 0) * 100;
        segment.style.width = `${Math.max(3, Math.round(width))}%`;
        segment.dataset.provider = route.exchange || "";
        segment.title = `${route.label || route.exchange} ${formatPercent(route.allocation_pct || route.allocation_weight)}`;
        bars.appendChild(segment);
      }
    });
  }

  function failedPreview(form, message) {
    const state = formPreviewState(form);
    return {
      ok: false,
      ready: false,
      mode: "blocked",
      state_label: "Blocked",
      ready_exchange_count: 0,
      total_exchange_count: state.providers.length,
      summary: {
        amount: state.amount,
        deposit_asset: state.depositAsset,
        settlement_asset: state.settlementAsset,
        allocation_engine: "1H10 Smart Router",
        selected_provider_count: state.providers.length,
        ready_provider_count: 0,
        allocated_total: 0,
        ml_readiness: { display_status: "Readiness unavailable" },
      },
      providers: state.providers.map((provider) => ({
        provider,
        enabled: true,
        ready: false,
        status: "blocked",
        allocation_weight: 0,
        allocation_pct: 0,
        target_amount: 0,
        score: 0,
        blockers: [
          {
            code: "readiness_fetch_failed",
            title: "Readiness unavailable",
            description: message,
            severity: "blocker",
            fix_hint: "Retry the preview after the server responds.",
          },
        ],
      })),
      active_blockers: [
        {
          code: "readiness_fetch_failed",
          title: "Readiness unavailable",
          description: message,
          severity: "blocker",
          fix_hint: "Retry the preview after the server responds.",
        },
      ],
      routing_preview: { notional_usd: 0, routes: [], summary: "Readiness unavailable." },
    };
  }

  function noProviderPreview(form) {
    const state = formPreviewState(form);
    return {
      ok: false,
      ready: false,
      mode: "blocked",
      state_label: "Exchange Setup Required",
      ready_exchange_count: 0,
      total_exchange_count: 0,
      providers: Array.from(form.querySelectorAll("[data-provider-card]")).map((card) => ({
        provider: card.dataset.provider || "",
        label: card.querySelector(".vault-provider-copy strong")?.textContent || card.dataset.provider || "",
        enabled: false,
        ready: false,
        allocation_weight: 0,
        allocation_pct: 0,
        target_amount: 0,
        score: 0,
        status: "disabled",
        blockers: [],
      })),
      summary: {
        amount: state.amount,
        deposit_asset: state.depositAsset,
        settlement_asset: state.settlementAsset,
        allocation_engine: "1H10 Smart Router",
        selected_provider_count: 0,
        ready_provider_count: 0,
        allocated_total: 0,
        ml_readiness: { display_status: "Provider required" },
      },
      active_blockers: [
        {
          code: "select_at_least_one_exchange",
          title: "Exchange required",
          description: "Enable at least one exchange before starting 1H10.",
          severity: "blocker",
          fix_hint: "Turn on KuCoin or Hyperliquid for this route.",
        },
      ],
      routing_preview: { notional_usd: 0, routes: [], summary: "Enable an exchange to generate route." },
    };
  }

  function scheduleRoutingPreview(form) {
    if (!form || !form.matches("[data-vault-routing-form]")) return;
    const previousTimer = previewTimers.get(form);
    if (previousTimer) window.clearTimeout(previousTimer);
    const timer = window.setTimeout(() => refreshRoutingPreview(form), 180);
    previewTimers.set(form, timer);
  }

  async function refreshRoutingPreview(form) {
    if (!form || !form.matches("[data-vault-routing-form]")) return null;
    const previewUrl = form.dataset.routingPreviewUrl || form.dataset.readinessUrl;
    if (!previewUrl) {
      const payload = failedPreview(form, "Readiness endpoint is not configured.");
      renderRoutingPreview(form, payload);
      return payload;
    }

    const previous = previewControllers.get(form);
    if (previous) previous.abort();
    const controller = new AbortController();
    previewControllers.set(form, controller);

    const state = formPreviewState(form);
    if (!state.providers.length) {
      const payload = noProviderPreview(form);
      renderRoutingPreview(form, payload);
      return payload;
    }
    const url = new URL(apiUrl(previewUrl), window.location.origin);
    url.searchParams.set("cycle_type", "one_h10");
    url.searchParams.set("amount", String(state.amount));
    url.searchParams.set("deposit_asset", state.depositAsset);
    url.searchParams.set("settlement_asset", state.settlementAsset);
    url.searchParams.set("one_h10_live_ack", state.acknowledged ? "1" : "0");
    state.providers.forEach((provider) => url.searchParams.append("providers", provider));
    form.classList.add("is-routing-loading");
    try {
      const response = await window.fetch(url, {
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Readiness check failed: ${response.status}`);
      const payload = await response.json();
      renderRoutingPreview(form, payload);
      return payload;
    } catch (error) {
      if (error.name === "AbortError") return null;
      const payload = failedPreview(form, error.message || "Readiness check failed.");
      renderRoutingPreview(form, payload);
      return payload;
    } finally {
      if (previewControllers.get(form) === controller) {
        previewControllers.delete(form);
        form.classList.remove("is-routing-loading");
      }
    }
  }

  function showResult(form, title, message, ok) {
    const sheet = form.closest(".vault-start-card")?.querySelector("[data-vault-result-sheet]");
    if (!sheet) return;
    sheet.hidden = false;
    sheet.classList.toggle("is-success", Boolean(ok));
    sheet.classList.toggle("is-error", !ok);
    sheet.querySelector("[data-vault-result-title]").textContent = title;
    sheet.querySelector("[data-vault-result-message]").textContent = message;
    window.setTimeout(() => {
      sheet.classList.add("is-visible");
    }, 20);
  }

  function setSubmitting(form, submitting) {
    form.dataset.submitting = submitting ? "1" : "0";
    const button = form.querySelector(".vault-start-button");
    if (!button) return;
    button.disabled = submitting || !Boolean(lastPreview.get(form)?.ready);
    button.classList.toggle("is-loading", submitting);
    const span = button.querySelector("span");
    if (span) span.textContent = submitting ? button.getAttribute("data-loading-label") || "Starting 1H10..." : "Start 1H10 Cycle";
  }

  async function submitVaultCycle(event) {
    const form = event.currentTarget;
    if (!form.matches("[data-vault-routing-form]")) return;
    event.preventDefault();
    if (form.dataset.submitting === "1") return;
    const idempotencyInput = form.querySelector("[data-vault-idempotency-key]");
    if (idempotencyInput && !idempotencyInput.value) {
      idempotencyInput.value = randomKey();
    }
    const preview = lastPreview.get(form) || (await refreshRoutingPreview(form));
    if (!preview?.ready) {
      renderRoutingPreview(form, preview || failedPreview(form, "Readiness check is still pending."));
      showResult(form, "Vault Cycle blocked", blockerText((preview?.active_blockers || [])[0]) || "Resolve readiness blockers before starting.", false);
      return;
    }

    setSubmitting(form, true);
    try {
      const response = await window.fetch(apiUrl(form.dataset.startCycleUrl || form.action), {
        method: "POST",
        headers: {
          Accept: "application/json",
          "X-Requested-With": "fetch",
        },
        body: new FormData(form),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.ok === false) {
        const readiness = payload.readiness || payload;
        renderRoutingPreview(form, readiness);
        showResult(form, "Vault Cycle blocked", payload.message || blockerText((payload.blockers || [])[0]) || "Readiness checks failed.", false);
        setSubmitting(form, false);
        return;
      }
      showResult(form, "Vault Cycle started", payload.duplicate ? "Existing cycle submission detected." : "1H10 cycle is starting with the verified route.", true);
    } catch (error) {
      showResult(form, "Start failed", error.message || "Start request failed.", false);
      setSubmitting(form, false);
    }
  }

  function setupProviderCards(form) {
    form.querySelectorAll("[data-provider-toggle]").forEach((input) => {
      const sync = () => {
        const card = input.closest("[data-provider-card]");
        card?.classList.toggle("is-enabled", input.checked);
        card?.classList.toggle("is-disabled", !input.checked);
        scheduleRoutingPreview(form);
      };
      input.addEventListener("change", sync);
      sync();
    });
  }

  function setupForm(form) {
    const idempotencyInput = form.querySelector("[data-vault-idempotency-key]");
    const amountInput = form.querySelector("[data-vault-amount]");
    const maxButton = form.querySelector("[data-vault-max]");
    const settlementSelect = form.querySelector("[data-settlement-asset]");
    const ackInput = form.querySelector("input[name='one_h10_live_ack']");

    if (idempotencyInput && !idempotencyInput.value) {
      idempotencyInput.value = randomKey();
    }

    form.querySelectorAll("[data-asset-option]").forEach((option) => {
      option.addEventListener("click", () => applySelectedOption(option));
      option.querySelector("input[type='radio']")?.addEventListener("change", () => applySelectedOption(option));
    });

    if (settlementSelect) {
      settlementSelect.addEventListener("change", () => {
        settlementManuallyChanged.add(form);
        scheduleRoutingPreview(form);
      });
    }
    syncSettlementToAllocation(form, false);

    if (amountInput) {
      amountInput.addEventListener("input", () => scheduleRoutingPreview(form));
    }

    if (ackInput) {
      ackInput.addEventListener("change", () => scheduleRoutingPreview(form));
    }

    if (maxButton && amountInput) {
      maxButton.addEventListener("click", () => {
        const option = selectedAllocationOption(form);
        const available = option?.dataset.availableBalance || "0";
        amountInput.value = available;
        amountInput.dispatchEvent(new Event("input", { bubbles: true }));
        amountInput.focus();
      });
    }

    form.addEventListener("submit", submitVaultCycle);
    setupProviderCards(form);
  }

  const initialPreview = parseInitialPreview();
  document.querySelectorAll(".vault-form").forEach((form) => {
    setupForm(form);
    if (initialPreview && form.matches("[data-vault-routing-form]")) {
      renderRoutingPreview(form, initialPreview);
      scheduleRoutingPreview(form);
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopTimers();
    else startTimers();
  });

  startTimers();
})();
