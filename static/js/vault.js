(() => {
  if (window.__AlgVaultVaultInitialized) return;
  window.__AlgVaultVaultInitialized = true;

  const countdowns = Array.from(document.querySelectorAll("[data-countdown]"));
  const activeCycleCards = Array.from(document.querySelectorAll("[data-vault-cycle-card]"));
  const settlementManuallyChanged = new WeakSet();
  const previewControllers = new WeakMap();
  const previewTimers = new WeakMap();
  const lastPreview = new WeakMap();
  const cycleRefreshInFlight = new WeakSet();
  const cycleRefreshAt = new WeakMap();
  const startStatusTimers = new Map();
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;
  const vaultApiUrl = (path) => window.AlgVaultConfig?.vaultApiUrl?.(path) || apiUrl(path);
  const CYCLE_STATUS_INTERVAL_MS = 20000;
  const CYCLE_STATUS_MIN_AGE_MS = 12000;
  const START_STATUS_INTERVAL_MS = 3000;
  const START_STATUS_MAX_POLLS = 30;
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

  function refreshActiveCycleCards(options = {}) {
    if (document.hidden) return;
    activeCycleCards.forEach((card) => refreshCycleCard(card, options));
  }

  function startTimers(forceCycleRefresh = false) {
    if (document.hidden) return;
    if (countdowns.length && !timerId) {
      updateCountdowns();
      timerId = window.setInterval(updateCountdowns, 1000);
    }
    if (activeCycleCards.length && !cycleStatusTimerId) {
      refreshActiveCycleCards({ force: forceCycleRefresh });
      cycleStatusTimerId = window.setInterval(() => {
        refreshActiveCycleCards();
      }, CYCLE_STATUS_INTERVAL_MS);
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

  function renderCycleTradeDecision(card, decision) {
    const container = card.querySelector("[data-cycle-trade-decision]");
    if (!container) return;
    clearElement(container);
    const label = decision?.label || "Trade decision";
    const detail = decision?.reason || decision?.message || "Awaiting server-side order decision";
    [label, detail].forEach((text) => {
      const item = document.createElement("span");
      item.textContent = String(text || "");
      container.appendChild(item);
    });
    container.dataset.tradeDecisionStage = decision?.stage || "pending";
  }

  function updateCycleCard(card, payload) {
    const status = payload?.cycle_status || {};
    const coherence = payload?.coherence_summary || {};
    const tradeDecision = payload?.trade_decision || {};
    const readiness = coherence.automationReadiness || "caution";
    const hasError = Boolean(status.error || payload?.ok === false);
    updateText(card, "[data-cycle-phase]", status.phaseLabel || titleCase(status.phase || "collectingData"));
    updateText(card, "[data-cycle-readiness]", `Automation readiness: ${readinessLabel(readiness)}`);
    updateText(card, "[data-cycle-direction]", directionLabel(coherence.overallDirection));
    updateText(card, "[data-cycle-confidence]", `${Math.round(numberValue(coherence.overallConfidence, 0))}%`);
    updateText(card, "[data-cycle-coherence]", `${Math.round(numberValue(coherence.coherenceScore, 0))}%`);
    updateText(card, "[data-cycle-summary]", coherence.summary || status.statusMessage || "Market forecast updated.");
    updateText(card, "[data-cycle-next]", `Next evaluation ${status.nextScheduled1h10Cycle || "pending"}`);
    renderCycleTradeDecision(card, tradeDecision);
    renderCycleRiskNotes(card, coherence.riskNotes || []);
    card.classList.toggle("is-stale", Boolean(status.stale));
    card.classList.toggle("is-error", hasError);
    card.classList.remove("is-loading");
    card.dataset.cycleRefreshState = hasError ? "error" : status.stale ? "stale" : "ready";
    card.dataset.lastCycleRefresh = new Date().toISOString();
    card.setAttribute("aria-busy", "false");
  }

  async function refreshCycleCard(card, options = {}) {
    const url = card.dataset.cycleStatusUrl;
    if (!url || document.hidden) return;
    if (cycleRefreshInFlight.has(card)) return;
    const now = Date.now();
    const lastRefresh = cycleRefreshAt.get(card) || 0;
    if (!options.force && now - lastRefresh < CYCLE_STATUS_MIN_AGE_MS) return;
    cycleRefreshInFlight.add(card);
    cycleRefreshAt.set(card, now);
    card.classList.add("is-loading");
    card.setAttribute("aria-busy", "true");
    try {
      const response = await window.fetch(vaultApiUrl(url), {
        credentials: "include",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`Cycle status failed: ${response.status}`);
      updateCycleCard(card, await response.json());
    } catch (error) {
      updateText(card, "[data-cycle-phase]", "Status unavailable");
      updateText(card, "[data-cycle-summary]", "Cycle status refresh failed. Showing the last available state.");
      renderCycleRiskNotes(card, ["Cycle status update failed"]);
      card.classList.add("is-error", "is-stale");
      card.classList.remove("is-loading");
      card.dataset.cycleRefreshState = "error";
      card.setAttribute("aria-busy", "false");
      console.warn("Vault cycle status refresh failed", error);
    } finally {
      cycleRefreshInFlight.delete(card);
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
    const status = provider.status || provider.readiness_state || (provider.ready ? "ready" : "blocked");
    if (status === "ready" || status === "ready_auto_funded") return "Ready";
    if (status === "geo_restricted") return "Provider restricted";
    if (status === "needs_wallet") return "Wallet needed";
    if (status === "needs_api_credentials") return "Credentials needed";
    if (status === "needs_verification") return "Verify";
    if (status === "provider_unavailable") return "Provider unavailable";
    if (status === "credential_error") return "Credential error";
    if (status === "transfer_failed") return "Transfer failed";
    if (status === "disabled") return "Disabled";
    if (status === "not_connected") return "Connect";
    if (status === "blocked") return "Action needed";
    return "Checking";
  }

  function metricLabel(provider, status) {
    if (provider.conversion_required && provider.conversion_status === "planned") return "Auto-converts";
    if (provider.fixed_egress_status === "pending" || provider.fixed_egress_status === "missing") return "Fixed egress";
    if (status === "ready_auto_funded" || provider.funding_status === "auto_funded") return "Auto-funded";
    if (status === "geo_restricted") return "Recheck provider";
    if (status === "needs_verification") return "Verify";
    if (status === "needs_wallet" || status === "needs_api_credentials" || status === "credential_error") return "Action needed";
    if (status === "ready") return `${Math.round(numberValue(provider.score, numberValue(provider.routing_score, 0) * 100))} score`;
    return statusLabel({ ...provider, status });
  }

  function titleStatus(value) {
    return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
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

  function normalizeStateKey(value) {
    return String(value || "blocked")
      .trim()
      .replace(/([a-z])([A-Z])/g, "$1-$2")
      .replace(/[\s_]+/g, "-")
      .toLowerCase();
  }

  function setRoutingBusy(form, busy) {
    const preview = form?.querySelector("[data-routing-preview]");
    form?.classList.toggle("is-routing-loading", Boolean(busy));
    if (preview) {
      preview.setAttribute("aria-busy", String(Boolean(busy)));
      if (busy) preview.dataset.previewState = "loading";
    }
  }

  function applyRoutingState(form, payload) {
    if (!form) return;
    const stateKey = normalizeStateKey(payload?.ready ? "success" : payload?.mode || payload?.state_label || "blocked");
    const preview = form.querySelector("[data-routing-preview]");
    form.dataset.vaultReadinessState = stateKey;
    form.classList.toggle("is-routing-ready", stateKey === "success");
    form.classList.toggle("is-routing-blocked", !payload?.ready && stateKey !== "loading");
    form.classList.toggle("is-routing-error", stateKey === "unavailable" || stateKey === "error");
    form.classList.toggle("is-routing-empty", stateKey === "empty" || stateKey === "exchange-setup-required");
    if (preview) {
      preview.dataset.previewState = stateKey;
      preview.setAttribute("aria-busy", "false");
    }
  }

  function updateProviderCard(form, provider) {
    const key = provider.provider || provider.exchange;
    const card = form.querySelector(`[data-provider-card][data-provider="${key}"]`);
    if (!card) return;
    const input = card.querySelector("[data-provider-toggle]");
    const enabled = input?.checked ?? provider.enabled;
    const ready = Boolean(provider.ready) || provider.status === "ready" || provider.status === "ready_auto_funded";
    const status = enabled ? (ready ? (provider.status || "ready") : provider.status || provider.readiness_state || "blocked") : "disabled";
    const topBlocker = Array.isArray(provider.blockers) ? provider.blockers.find((item) => item?.severity !== "info") || provider.blockers[0] : null;
    const blockerEl = card.querySelector("[data-provider-blocker]");

    card.classList.toggle("is-enabled", Boolean(enabled));
    card.classList.toggle("is-ready", status === "ready" || status === "ready_auto_funded");
    card.classList.toggle("is-auto-funded", status === "ready_auto_funded" || provider.funding_status === "auto_funded");
    card.classList.toggle("is-restricted", status === "geo_restricted");
    card.classList.toggle("is-unavailable", status === "provider_unavailable");
    card.classList.toggle("is-disconnected", status === "not_connected" || status === "needs_api_credentials" || status === "needs_wallet");
    card.classList.toggle("is-blocked", status === "blocked" || status === "not_connected" || status === "provider_unavailable" || status === "credential_error" || status === "transfer_failed");
    card.classList.toggle("is-disabled", !enabled);
    card.dataset.providerState = status;
    const safeStatus = statusLabel({ ...provider, status });
    card.querySelector("[data-provider-status]").textContent = safeStatus;
    card.querySelector("[data-provider-score]").textContent = metricLabel(provider, status);
    card.querySelector("[data-provider-allocation]").textContent =
      numberValue(provider.allocation_pct, 0) > 0 ? formatPercent(provider.allocation_pct) : formatPercent(provider.allocation_weight);
    if (blockerEl) {
      const conversionText =
        provider.conversion_required && provider.conversion_status === "planned"
          ? `Auto converts allocation to ${provider.conversion_to || "collateral"}`
          : "";
      const text = blockerText(topBlocker) || conversionText || provider.funding_detail || provider.funding_label || "";
      blockerEl.textContent = text;
      blockerEl.hidden = !text || (status === "ready" && !conversionText) || !enabled;
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
    applyRoutingState(form, payload);
    const providers = providersFromPayload(payload);
    providers.forEach((provider) => updateProviderCard(form, provider));
    updateKucoinDiagnostics(form, payload.kucoin_diagnostics || providers.find((provider) => provider.provider === "kucoin")?.kucoin_diagnostics);

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

  function updateKucoinDiagnostics(form, diagnostics) {
    const panel = form.querySelector("[data-kucoin-diagnostics]");
    if (!panel || !diagnostics) return;
    const ip = diagnostics.ip_restriction || {};
    const permissions = diagnostics.permissions || {};
    const spot = permissions.spot?.status || "not_checked";
    const futures = permissions.futures?.status || "not_checked";
    const unified = permissions.unified?.status || "not_checked";
    const operatorIp = panel.querySelector("[data-kucoin-operator-ip]");
    const serverEgress = panel.querySelector("[data-kucoin-server-egress]");
    const trustedStatus = panel.querySelector("[data-kucoin-trusted-status]");
    const permissionText = panel.querySelector("[data-kucoin-permissions]");
    const trustedMessage = panel.querySelector("[data-kucoin-trusted-message]");
    if (operatorIp) operatorIp.textContent = ip.operator_ip || "Not detected";
    if (serverEgress) serverEgress.textContent = Array.isArray(ip.server_egress_ips) && ip.server_egress_ips.length ? ip.server_egress_ips.join(", ") : "Not configured";
    if (trustedStatus) trustedStatus.textContent = titleStatus(ip.trusted_ip_status);
    if (permissionText) permissionText.textContent = `${titleStatus(spot)} / ${titleStatus(futures)} / ${titleStatus(unified)}`;
    if (trustedMessage) trustedMessage.textContent = ip.trusted_ip_message || "Server-side KuCoin diagnostics pending.";
  }

  function failedPreview(form, message) {
    const state = formPreviewState(form);
    return {
      ok: false,
      ready: false,
      mode: "unavailable",
      state_label: "Unavailable",
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
      routing_preview: { notional_usd: 0, routes: [], summary: "Readiness unavailable. Retry when the server responds." },
    };
  }

  function noProviderPreview(form) {
    const state = formPreviewState(form);
    return {
      ok: false,
      ready: false,
      mode: "empty",
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

    const state = formPreviewState(form);
    if (!state.providers.length) {
      const payload = noProviderPreview(form);
      renderRoutingPreview(form, payload);
      return payload;
    }
    const previous = previewControllers.get(form);
    if (previous) previous.abort();
    const controller = new AbortController();
    previewControllers.set(form, controller);

    const url = new URL(vaultApiUrl(previewUrl), window.location.origin);
    url.searchParams.set("cycle_type", "one_h10");
    url.searchParams.set("amount", String(state.amount));
    url.searchParams.set("deposit_asset", state.depositAsset);
    url.searchParams.set("settlement_asset", state.settlementAsset);
    url.searchParams.set("one_h10_live_ack", state.acknowledged ? "1" : "0");
    state.providers.forEach((provider) => url.searchParams.append("providers", provider));
    setRoutingBusy(form, true);
    try {
      const response = await window.fetch(url, {
        credentials: "include",
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
        setRoutingBusy(form, false);
      }
    }
  }

  function showResult(form, title, message, ok) {
    const sheet = form.closest(".vault-start-card")?.querySelector("[data-vault-result-sheet]");
    if (!sheet) return;
    sheet.hidden = false;
    sheet.setAttribute("aria-live", ok ? "polite" : "assertive");
    sheet.classList.toggle("is-success", Boolean(ok));
    sheet.classList.toggle("is-error", !ok);
    sheet.querySelector("[data-vault-result-title]").textContent = title;
    sheet.querySelector("[data-vault-result-message]").textContent = message;
    window.setTimeout(() => {
      sheet.classList.add("is-visible");
    }, 20);
  }

  function cycleStartStatusUrl(payload) {
    if (payload?.start_status_url) return payload.start_status_url;
    if (payload?.job_id) return `/vault/start-status/${encodeURIComponent(payload.job_id)}`;
    return "";
  }

  function cycleStatusUrl(payload) {
    return payload?.next_status_url || payload?.cycle_status_url || (payload?.cycle_id ? `/api/vault/cycles/${encodeURIComponent(payload.cycle_id)}` : "");
  }

  function cycleStartMessage(payload) {
    const queue = payload?.strategy_run_queue || "";
    const workerMode = payload?.worker_mode || "web";
    const runCount = Array.isArray(payload?.run_ids) ? payload.run_ids.length : 0;
    if (payload?.status === "failed") return payload.error || "Vault Cycle worker startup failed.";
    if (payload?.status === "complete") return "Worker startup completed. Watching the server-side trade decision path.";
    if (queue === "dedicated_worker") {
      return `${runCount || "Strategy"} run${runCount === 1 ? "" : "s"} queued for the dedicated worker (${workerMode}). No order can place until the worker starts the run.`;
    }
    return `${runCount || "Strategy"} run${runCount === 1 ? "" : "s"} started through the server-side Vault Cycle path.`;
  }

  async function fetchCycleJson(url) {
    const response = await window.fetch(vaultApiUrl(url), {
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error(`Status check failed: ${response.status}`);
    return response.json();
  }

  async function refreshStartedCycleMessage(form, payload) {
    const url = cycleStatusUrl(payload);
    if (!url) return;
    const status = await fetchCycleJson(url);
    const decision = status?.trade_decision || {};
    const message = decision.reason || decision.message || status?.cycle_status?.statusMessage || cycleStartMessage(payload);
    showResult(form, decision.label || "Vault Cycle status", message, status?.ok !== false);
    refreshActiveCycleCards({ force: true });
  }

  function trackCycleStart(form, payload) {
    const key = payload?.job_id || payload?.cycle_id || randomKey();
    if (startStatusTimers.has(key)) window.clearInterval(startStatusTimers.get(key));
    showResult(form, payload?.status === "queued" ? "Vault Cycle queued" : "Vault Cycle started", cycleStartMessage(payload), true);
    refreshActiveCycleCards({ force: true });

    const statusUrl = cycleStartStatusUrl(payload);
    if (!statusUrl) {
      refreshStartedCycleMessage(form, payload).catch((error) => console.warn("Vault cycle status check failed", error));
      return;
    }

    let polls = 0;
    const poll = async () => {
      polls += 1;
      try {
        const status = await fetchCycleJson(statusUrl);
        showResult(form, status.status === "failed" ? "Vault Cycle start failed" : "Vault Cycle status", cycleStartMessage(status), status.status !== "failed");
        if (status.status === "complete" || status.status === "failed" || polls >= START_STATUS_MAX_POLLS) {
          window.clearInterval(startStatusTimers.get(key));
          startStatusTimers.delete(key);
          if (status.status !== "failed") {
            await refreshStartedCycleMessage(form, status);
          }
        }
      } catch (error) {
        if (polls >= START_STATUS_MAX_POLLS) {
          window.clearInterval(startStatusTimers.get(key));
          startStatusTimers.delete(key);
        }
        console.warn("Vault cycle start status check failed", error);
      }
    };
    startStatusTimers.set(key, window.setInterval(poll, START_STATUS_INTERVAL_MS));
    poll();
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
      const response = await window.fetch(vaultApiUrl(form.dataset.startCycleUrl || form.action), {
        method: "POST",
        credentials: "include",
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
      trackCycleStart(form, payload);
      setSubmitting(form, false);
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
    if (form.dataset.vaultFormReady === "true") return;
    form.dataset.vaultFormReady = "true";
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
    else startTimers(true);
  });

  startTimers(true);
})();
