(() => {
  const root = document.querySelector("[data-backtest-root]");
  const form = document.querySelector("[data-backtest-form]");
  if (!root || !form) return;

  const refs = {
    payload: document.getElementById("backtest-initial-payload"),
    initialSymbols: document.getElementById("backtest-initial-symbols"),
    dashboard: document.querySelector("[data-backtest-dashboard]"),
    empty: document.querySelector("[data-backtest-empty]"),
    status: document.querySelector("[data-backtest-status]"),
    submit: document.querySelector("[data-backtest-submit]"),
    symbolList: document.querySelector("[data-symbol-list]"),
    universeVenues: document.querySelector("[data-universe-venues]"),
    universeReady: document.querySelector("[data-universe-ready]"),
    universeCount: document.querySelector("[data-universe-count]"),
    universeCollateral: document.querySelector("[data-universe-collateral]"),
    assetList: document.querySelector("[data-allocation-asset-list]"),
    selectedChips: document.querySelector("[data-selected-asset-chips]"),
    useVaultAllocation: document.querySelector("[data-use-vault-allocation]"),
    allocation: document.querySelector("[data-backtest-allocation]"),
    slider: document.querySelector("[data-allocation-slider]"),
    maxButton: document.querySelector("[data-backtest-max]"),
    allocationPreview: document.querySelector("[data-allocation-preview]"),
    chartHost: document.querySelector("[data-backtest-chart]"),
    chartOverlay: document.querySelector("[data-backtest-overlay]"),
    chartTitle: document.querySelector("[data-chart-title]"),
    autopilotConfidence: document.querySelector("[data-autopilot-confidence]"),
    executionScore: document.querySelector("[data-execution-score]"),
    activeStrategies: document.querySelector("[data-active-strategies]"),
    autopilotList: document.querySelector("[data-autopilot-list]"),
    executionList: document.querySelector("[data-execution-list]"),
    portfolioDiagnostics: document.querySelector("[data-portfolio-diagnostics]"),
    allocationPolicy: document.querySelector("[data-allocation-policy]"),
    assetBreakdown: document.querySelector("[data-asset-breakdown]"),
    strategyWeights: document.querySelector("[data-strategy-weights]"),
    strategyWeightCount: document.querySelector("[data-strategy-weight-count]"),
  };

  const urls = { symbols: root.dataset.symbolsUrl || "", chartLib: root.dataset.chartLibSrc || "" };
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;
  const metricNodes = Object.fromEntries(Array.from(document.querySelectorAll("[data-metric]")).map((node) => [node.dataset.metric, node]));
  const systemNodes = Object.fromEntries(Array.from(document.querySelectorAll("[data-system-metric]")).map((node) => [node.dataset.systemMetric, node]));
  const money = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
  const compactMoney = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 });
  const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
  const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
  const state = { symbols: [], allocationAssets: [], selectedAssets: new Set(), payload: null, chartMode: "equity", chart: null, symbolAbort: null };

  const readJson = (node) => {
    if (!node?.textContent?.trim()) return null;
    try { return JSON.parse(node.textContent); } catch (error) { return null; }
  };
  const number = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const setStatus = (message, type = "") => {
    if (!refs.status) return;
    refs.status.textContent = message || "";
    refs.status.dataset.state = type;
  };
  const setLoading = (loading) => {
    if (!refs.submit) return;
    refs.submit.disabled = loading || !canRun();
    refs.submit.textContent = loading
      ? refs.submit.dataset.loadingLabel || "Running backtest..."
      : refs.submit.dataset.defaultLabel || "Run Backtest";
    refs.dashboard?.classList.toggle("is-loading", loading);
  };
  const selectedAssetRows = () => state.allocationAssets.filter((row) => state.selectedAssets.has(String(row.asset || "").toUpperCase()));
  const selectedAssetCap = () => Math.min(number(form.dataset.paperBalance, 10000), selectedAssetRows().reduce((total, row) => total + number(row.available_usd || row.cap_usd), 0));
  const canRun = () => state.selectedAssets.size > 0 && selectedAssetCap() > 0 && number(refs.allocation?.value) > 0 && number(refs.allocation?.value) <= selectedAssetCap() && window.navigator?.onLine !== false;
  const updateSubmitState = () => {
    if (!refs.submit) return;
    const allocation = number(refs.allocation?.value);
    const cap = selectedAssetCap();
    let message = "";
    if (window.navigator?.onLine === false) message = "Offline. Reconnect to run a backtest.";
    else if (!state.allocationAssets.length) message = "Vault allocation assets are unavailable.";
    else if (!state.selectedAssets.size) message = "Select at least one Vault allocation asset.";
    else if (cap <= 0) message = "Selected Vault allocation assets have no available allocation balance.";
    else if (allocation <= 0 || allocation > cap) message = `Enter an allocation between $0 and ${money.format(cap)}.`;
    refs.submit.disabled = Boolean(message);
    setStatus(message, message ? "error" : "");
  };

  const normalizeSymbolPayload = (payload) => Array.isArray(payload?.symbols) ? payload.symbols : [];
  const normalizeAssetPayload = (payload) => Array.isArray(payload?.allocation_assets) ? payload.allocation_assets : [];
  const parseJsonResponse = async (response) => {
    const contentType = response.headers.get("content-type") || "";
    let payload = {};
    if (contentType.includes("application/json")) {
      try { payload = await response.json(); } catch (error) { payload = {}; }
    } else {
      const text = await response.text().catch(() => "");
      payload = { ok: false, error: text.trim().slice(0, 240) };
    }
    const path = response.redirected ? new URL(response.url, window.location.origin).pathname : "";
    if (response.status === 401 || response.status === 403 || path.startsWith("/login")) {
      return { ...payload, ok: false, error: "Admin session required. Sign in as an authorized admin and retry.", error_code: "admin_session_required" };
    }
    return payload;
  };
  const applySymbols = (payload) => {
    state.symbols = normalizeSymbolPayload(payload);
    applyAllocationAssets(payload || {});
    renderUniverse(payload || {});
    renderSymbols();
    updateSubmitState();
  };
  const loadSymbols = async () => {
    if (!urls.symbols) return;
    state.symbolAbort?.abort();
    state.symbolAbort = new AbortController();
    const url = new URL(apiUrl(urls.symbols), window.location.origin);
    url.searchParams.set("limit", "80");
    refs.symbolList?.classList.add("is-loading");
    try {
      const response = await fetch(url, { signal: state.symbolAbort.signal, headers: { Accept: "application/json" } });
      const payload = await parseJsonResponse(response);
      if (!response.ok || !payload?.ok) throw new Error(payload?.error || "Unable to load Vault allocation data.");
      if (payload?.ok) applySymbols(payload);
    } catch (error) {
      if (error.name !== "AbortError") {
        renderUniverse({ symbols: [] });
        setStatus("Unable to load Vault allocation data. Retry when the server responds.", "error");
      }
    } finally {
      refs.symbolList?.classList.remove("is-loading");
    }
  };
  const renderUniverse = (payload) => {
    const rows = normalizeSymbolPayload(payload);
    const venues = [...new Set(rows.map((row) => row.provider_label || row.provider).filter(Boolean))];
    const collateral = [...new Set(rows.map((row) => row.settlement_asset || row.quote_asset).filter(Boolean))];
    const selectedCollateral = Array.from(state.selectedAssets);
    if (refs.universeVenues) refs.universeVenues.textContent = venues.length ? `${venues.join(" + ")} enabled venues` : "No enabled leveraged venues detected";
    if (refs.universeReady) {
      refs.universeReady.textContent = rows.length ? "Ready" : "Needs exchange";
      refs.universeReady.dataset.state = rows.length ? "ready" : "blocked";
    }
    if (refs.universeCount) refs.universeCount.textContent = String(number(payload.total, rows.length));
    if (refs.universeCollateral) refs.universeCollateral.textContent = collateral.length ? collateral.join(" + ") : selectedCollateral.length ? selectedCollateral.join(" + ") : "USDC";
  };
  const applyAllocationAssets = (payload) => {
    const rows = normalizeAssetPayload(payload);
    if (rows.length) {
      state.allocationAssets = rows.map((row) => ({
        ...row,
        asset: String(row.asset || "").toUpperCase(),
        available_usd: number(row.available_usd || row.cap_usd),
        available_balance: number(row.available_balance),
      })).filter((row) => row.asset);
    } else if (!state.allocationAssets.length) {
      state.allocationAssets = Array.from(refs.assetList?.querySelectorAll("[data-allocation-asset]") || []).map((node) => ({
        asset: String(node.dataset.asset || node.querySelector("input")?.value || "").toUpperCase(),
        available_usd: number(node.dataset.availableUsd),
        available_balance: 0,
      })).filter((row) => row.asset);
    }
    const defaultAsset = String(payload.default_allocation_asset || root.dataset.defaultAllocationAsset || form.dataset.defaultAllocationAsset || state.allocationAssets[0]?.asset || "").toUpperCase();
    const currentSelected = selectedAssetsFromDom();
    const nextSelected = currentSelected.length ? currentSelected : (payload.selected_allocation_assets || [defaultAsset]);
    state.selectedAssets = new Set(nextSelected.map((asset) => String(asset || "").toUpperCase()).filter(Boolean));
    if (!state.selectedAssets.size && defaultAsset) state.selectedAssets.add(defaultAsset);
    renderAllocationAssets();
    syncAllocation(refs.allocation || refs.slider || { value: 0 });
  };
  const selectedAssetsFromDom = () => Array.from(refs.assetList?.querySelectorAll("input[name='allocation_assets']:checked") || []).map((input) => String(input.value || "").toUpperCase()).filter(Boolean);
  const renderAllocationAssets = () => {
    if (!refs.assetList) return;
    if (!state.allocationAssets.length) {
      refs.assetList.innerHTML = '<div class="vault-routing-empty" data-allocation-empty>No Vault allocation assets are available.</div>';
      renderSelectedChips();
      return;
    }
    refs.assetList.innerHTML = state.allocationAssets.map((row) => {
      const asset = escapeHtml(row.asset);
      const checked = state.selectedAssets.has(row.asset) ? " checked" : "";
      const selectedClass = state.selectedAssets.has(row.asset) ? " is-selected" : "";
      const stateLabel = row.available_usd > 0 ? money.format(row.available_usd) : row.state === "price_unavailable" ? "Price unavailable" : "No balance";
      return `
        <label class="asset-option vault-asset-option backtest-allocation-option${selectedClass}" data-allocation-asset data-asset="${asset}" data-available-usd="${number(row.available_usd).toFixed(6)}">
          <input type="checkbox" name="allocation_assets" value="${asset}"${checked}>
          <span class="backtest-token-icon">${escapeHtml(asset.slice(0, 1) || "?")}</span>
          <span>
            <strong>${asset}</strong>
            <small>${number(row.available_balance).toFixed(6)} · ${escapeHtml(stateLabel)}</small>
          </span>
        </label>`;
    }).join("");
    renderSelectedChips();
  };
  const renderSelectedChips = () => {
    if (!refs.selectedChips) return;
    const rows = selectedAssetRows();
    if (!rows.length) {
      refs.selectedChips.innerHTML = '<span>Select Vault assets</span>';
      return;
    }
    refs.selectedChips.innerHTML = rows.map((row) => `<span>${escapeHtml(row.asset)} <strong>${money.format(number(row.available_usd))}</strong></span>`).join("");
  };
  const renderSymbols = () => {
    if (!refs.symbolList) return;
    if (!state.symbols.length) {
      refs.symbolList.innerHTML = '<div class="backtest-symbol-empty">No eligible leveraged pairs found.</div>';
      return;
    }
    refs.symbolList.innerHTML = state.symbols.slice(0, 80).map((row) => `
      <div class="backtest-symbol-row" role="listitem">
        <span class="backtest-token-icon">${escapeHtml(row.token_icon || String(row.symbol || "?").slice(0, 1))}</span>
        <span class="backtest-symbol-copy">
          <strong>${escapeHtml(row.symbol || "--")}</strong>
          <small>${escapeHtml(row.provider_label || row.provider || "Enabled")} · ${escapeHtml(row.venue_symbol || row.symbol || "")} · ${escapeHtml(row.settlement_asset || "USDC")}</small>
        </span>
        <span class="backtest-symbol-badges"><em>${escapeHtml(row.max_leverage ? `${number(row.max_leverage).toFixed(0)}x` : "Leveraged")}</em></span>
      </div>`).join("");
  };

  const renderPayload = (payload) => {
    if (!payload?.ok) {
      setStatus(payload?.error || "No backtest results are available yet.", payload?.error ? "error" : "");
      return;
    }
    state.payload = payload;
    refs.dashboard.hidden = false;
    if (refs.empty) refs.empty.hidden = true;
    renderSummary(payload.summary || {});
    renderMetrics(payload.metrics || {});
    renderAutopilot(payload.autopilot || {});
    renderExecution(payload.execution_quality || {}, payload.trade_decision || {}, payload.simulation_scope || {});
    renderPortfolioDiagnostics(payload.portfolio_diagnostics || payload.result?.portfolio_diagnostics || {});
    renderAssetBreakdown(payload.asset_breakdown || payload.result?.asset_breakdown || []);
    renderStrategyWeights(payload.strategy_weights || payload.result?.strategy_weights || []);
    renderSystemMetrics(payload.system_metrics || {});
    renderChart();
  };
  const renderSummary = (summary) => {
    const title = document.querySelector("[data-backtest-title]");
    const chip = document.querySelector("[data-backtest-summary]");
    if (title) title.textContent = summary.title || "Portfolio Vault Cycle";
    if (chip) chip.textContent = summary.subtitle || `${summary.provider_label || "All enabled leveraged pairs"} / ${summary.duration || "1H10"}`;
  };
  const renderMetrics = (metrics) => {
    const roi = number(metrics.roi); const pnl = number(metrics.pnl); const drawdown = number(metrics.max_drawdown);
    setMetric("roi", percent.format(roi), roi); setMetric("pnl", money.format(pnl), pnl);
    setMetric("win_rate", percent.format(number(metrics.win_rate))); setMetric("max_drawdown", percent.format(drawdown), drawdown * -1);
    setMetric("trades", String(Math.round(number(metrics.trades)))); setMetric("fees", money.format(number(metrics.fees)));
    setMetric("average_trade", money.format(number(metrics.average_trade)));
    setMetric("profit_factor", number(metrics.profit_factor).toFixed(2));
    setMetric("closed_trades", String(Math.round(number(metrics.closed_trades ?? metrics.trades))));
    setMetric("open_trades", String(Math.round(number(metrics.open_trades))));
    setMetric("target_progress", percent.format(number(metrics.target_progress)), number(metrics.target_progress));
    setMetric("objective_gap", percent.format(number(metrics.objective_gap_pct) / 100), number(metrics.objective_gap_pct) * -1);
  };
  const setMetric = (key, text, polarity = null) => {
    const node = metricNodes[key]; if (!node) return;
    node.textContent = text;
    if (polarity !== null) { node.classList.toggle("positive", polarity >= 0); node.classList.toggle("negative", polarity < 0); }
  };
  const renderAutopilot = (autopilot) => {
    if (refs.autopilotConfidence) refs.autopilotConfidence.textContent = percent.format(number(autopilot.confidence));
    renderKeyValues(refs.autopilotList, [["Status", autopilot.status || "portfolio-ready"], ["Regime", autopilot.market_regime || "Aggregated"], ["Objective", autopilot.objective || "portfolio vault cycle"], ["Model Stack", Array.isArray(autopilot.model_stack) ? autopilot.model_stack.length : 0]]);
  };
  const renderExecution = (execution, tradeDecision = {}, scope = {}) => {
    if (refs.executionScore) refs.executionScore.textContent = percent.format(number(execution.fill_quality));
    renderKeyValues(refs.executionList, [["Mode", tradeDecision.mode || "backtest"], ["Decision", tradeDecision.label || "Backtest simulation"], ["Worker", scope.queues_worker ? "Queued" : "Not started"], ["Broker Order", tradeDecision.broker_order_submitted || scope.submits_broker_order ? "Submitted" : "No"], ["Venues", execution.venue_count || "--"], ["Pairs", execution.eligible_pair_count || "--"], ["Fees", `${number(execution.fee_bps).toFixed(2)} bps`], ["Slippage", `${number(execution.slippage_bps).toFixed(2)} bps`], ["Liquidity", compactMoney.format(number(execution.liquidity_usd))], ["Max Exposure", compactMoney.format(number(execution.max_exposure_usd))]]);
  };
  const renderPortfolioDiagnostics = (diagnostics) => {
    if (refs.allocationPolicy) refs.allocationPolicy.textContent = String(diagnostics.allocation_policy || "after-cost").replace(/_/g, " ");
    const skipped = diagnostics.skipped_reasons || {};
    const skippedText = Object.entries(skipped).map(([reason, count]) => `${reason.replace(/_/g, " ")} (${count})`).join(", ") || "None";
    renderKeyValues(refs.portfolioDiagnostics, [["Allocated", diagnostics.allocated_candidate_count || 0], ["Skipped", diagnostics.skipped_candidate_count || 0], ["Score", number(diagnostics.total_after_cost_score).toFixed(3)], ["Skipped Reasons", skippedText], ["Live Gates", String(diagnostics.live_authority || "server risk gates").replace(/_/g, " ")]]);
  };
  const renderAssetBreakdown = (rows) => {
    if (!refs.assetBreakdown) return;
    if (refs.activeStrategies) refs.activeStrategies.textContent = `${rows.length} assets`;
    if (!rows.length) { refs.assetBreakdown.innerHTML = '<div class="backtest-symbol-empty">No asset-level results yet.</div>'; return; }
    refs.assetBreakdown.innerHTML = rows.slice().sort((a, b) => Math.abs(number(b.pnl)) - Math.abs(number(a.pnl))).slice(0, 12).map((row) => `
      <div class="backtest-asset-row">
        <div><strong>${escapeHtml(row.asset || row.symbol || "--")}</strong><small>${escapeHtml(row.error || row.exchange || row.provider_label || "Enabled venue")}</small></div>
        <div><span>PnL</span><strong class="${number(row.pnl) >= 0 ? "positive" : "negative"}">${money.format(number(row.pnl))}</strong></div>
        <div><span>ROI</span><strong>${percent.format(number(row.roi))}</strong></div>
        <div><span>Weight</span><strong>${percent.format(number(row.allocation_weight))}</strong></div>
        <div><span>Net Edge</span><strong>${number(row.net_expected_return_bps).toFixed(1)} bps</strong></div>
        <div><span>Cost</span><strong>${number(row.cost_drag_bps).toFixed(1)} bps</strong></div>
        <div><span>Trades</span><strong>${Math.round(number(row.trades))}</strong></div>
        <div><span>Closed/Open</span><strong>${Math.round(number(row.closed_trades ?? row.trades))}/${Math.round(number(row.open_trades))}</strong></div>
        <div><span>Fees</span><strong>${money.format(number(row.fees))}</strong></div>
        <div><span>Max exposure</span><strong>${compactMoney.format(number(row.max_exposure))}</strong></div>
      </div>`).join("");
  };
  const renderStrategyWeights = (rows) => {
    if (!refs.strategyWeights) return;
    const sorted = rows.slice().sort((a, b) => Number(Boolean(b.enabled)) - Number(Boolean(a.enabled)) || number(b.weight) - number(a.weight)).slice(0, 16);
    if (refs.strategyWeightCount) refs.strategyWeightCount.textContent = `${sorted.filter((row) => row.enabled).length} active`;
    if (!sorted.length) { refs.strategyWeights.innerHTML = '<div class="backtest-symbol-empty">No strategy weights yet.</div>'; return; }
    refs.strategyWeights.innerHTML = sorted.map((row) => {
      const enabled = Boolean(row.enabled);
      const reason = enabled ? `${percent.format(number(row.weight))} weight · ${escapeHtml(row.asset || "Portfolio")}` : escapeHtml(row.disabled_reason || "disabled");
      return `
        <div class="backtest-strategy-row${enabled ? "" : " is-disabled"}" style="--weight:${Math.max(0, Math.min(number(row.weight), 1)).toFixed(4)}">
          <div><strong>${escapeHtml(row.label || row.strategy_name || "Strategy")}</strong><small>${reason}</small></div>
          <span>${number(row.net_return_after_costs ?? row.total_return).toFixed(4)}</span>
          <i aria-hidden="true"></i>
        </div>`;
    }).join("");
  };
  const renderSystemMetrics = (metrics) => Object.entries(systemNodes).forEach(([key, node]) => { node.textContent = metrics[key] || node.textContent || "Auto"; });
  const renderKeyValues = (host, rows) => { if (host) host.innerHTML = rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`).join(""); };

  const renderChart = async () => {
    if (refs.chartTitle) refs.chartTitle.textContent = chartLabel(state.chartMode);
    if (!state.chart) state.chart = new BacktestChart(refs.chartHost, refs.chartOverlay, urls.chartLib);
    await state.chart.render(state.payload || {}, state.chartMode);
  };
  const submitForm = async (event) => {
    event.preventDefault();
    updateSubmitState();
    if (refs.submit?.disabled) return;
    setLoading(true); setStatus("Running backtest across eligible leveraged pairs.");
    try {
      const response = await fetch(apiUrl(form.action), { method: "POST", body: new FormData(form), headers: { Accept: "application/json", "X-Requested-With": "fetch" } });
      const payload = await parseJsonResponse(response);
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Unable to complete the backtest. Check connection status and try again.");
      renderPayload(payload); setStatus("Backtest completed.", "success");
    } catch (error) { setStatus(error.message || "Unable to complete the backtest. Check connection status and try again.", "error"); }
    finally { setLoading(false); updateSubmitState(); }
  };
  const syncAllocation = (source) => {
    const cap = selectedAssetCap();
    form.dataset.allocationCap = String(cap);
    let value = Math.max(0, Math.min(number(source.value, 0), cap));
    if (source !== refs.allocation && refs.allocation) refs.allocation.value = value ? String(value) : "";
    if (refs.allocation) refs.allocation.max = String(cap);
    if (refs.slider) {
      refs.slider.max = String(Math.max(cap, 0));
      if (source !== refs.slider) refs.slider.value = String(value);
    }
    const assets = Array.from(state.selectedAssets).join(" + ") || "Vault";
    if (refs.allocationPreview) refs.allocationPreview.innerHTML = `<span>Simulation allocation</span><strong>${money.format(value)} ${escapeHtml(assets)}</strong>`;
    renderSelectedChips();
    updateSubmitState();
  };

  refs.maxButton?.addEventListener("click", () => { const cap = String(selectedAssetCap()); if (refs.allocation) refs.allocation.value = cap; if (refs.slider) refs.slider.value = cap; syncAllocation(refs.allocation || refs.slider); });
  refs.allocation?.addEventListener("input", () => syncAllocation(refs.allocation));
  refs.slider?.addEventListener("input", () => syncAllocation(refs.slider));
  refs.assetList?.addEventListener("change", (event) => {
    const input = event.target;
    if (!(input instanceof HTMLInputElement) || input.name !== "allocation_assets") return;
    state.selectedAssets = new Set(selectedAssetsFromDom());
    refs.assetList.querySelectorAll("[data-allocation-asset]").forEach((node) => node.classList.toggle("is-selected", state.selectedAssets.has(String(node.dataset.asset || "").toUpperCase())));
    syncAllocation(refs.allocation || refs.slider || { value: 0 });
  });
  refs.useVaultAllocation?.addEventListener("click", () => {
    const funded = state.allocationAssets.filter((row) => number(row.available_usd) > 0).map((row) => row.asset);
    state.selectedAssets = new Set(funded.length ? funded : [String(form.dataset.defaultAllocationAsset || state.allocationAssets[0]?.asset || "").toUpperCase()].filter(Boolean));
    renderAllocationAssets();
    syncAllocation(refs.allocation || refs.slider || { value: 0 });
  });
  document.querySelectorAll("[data-chart-mode]").forEach((button) => button.addEventListener("click", () => { state.chartMode = button.dataset.chartMode || "equity"; document.querySelectorAll("[data-chart-mode]").forEach((item) => item.classList.toggle("is-active", item === button)); renderChart(); }));
  form.addEventListener("submit", submitForm);
  window.addEventListener("resize", () => state.chart?.drawOverlay?.());
  window.addEventListener("online", updateSubmitState); window.addEventListener("offline", updateSubmitState);

  const initialSymbols = readJson(refs.initialSymbols); const initialPayload = readJson(refs.payload);
  if (initialSymbols?.symbols) applySymbols(initialSymbols); else loadSymbols();
  if (initialPayload?.ok) renderPayload(initialPayload);
  syncAllocation(refs.allocation || { value: 0 });

  function chartLabel(mode) { return { equity: "Portfolio Equity Curve", pnl: "Portfolio PnL", contribution: "PnL by Asset", timeline: "Trade Timeline" }[mode] || "Portfolio Simulation"; }
  function escapeHtml(value) { return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  class BacktestChart {
    constructor(host, overlay, librarySrc) { this.host = host; this.overlay = overlay; this.librarySrc = librarySrc; this.chart = null; this.lineSeries = null; this.payload = null; this.mode = "equity"; this.libraryPromise = null; }
    async render(payload, mode) { this.payload = payload || {}; this.mode = mode || "equity"; try { await this.ensureChart(); this.applyData(); } catch (error) {} this.drawOverlay(); }
    async ensureChart() {
      if (!this.host || ["contribution", "timeline"].includes(this.mode)) return false;
      const library = await this.loadLibrary(); if (!library?.createChart) return false;
      if (!this.chart) { this.chart = library.createChart(this.host, { autoSize: true, layout: { background: { color: "#05070b" }, textColor: "#8f9bae", fontFamily: "Inter, system-ui, sans-serif" }, grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.055)" } }, rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" }, timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false }, handleScroll: { horzTouchDrag: true, vertTouchDrag: false }, handleScale: { pinch: true, mouseWheel: false } }); this.lineSeries = this.chart.addLineSeries({ color: "#0ecb81", lineWidth: 2, priceLineVisible: false }); }
      return true;
    }
    loadLibrary() { if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts); if (this.libraryPromise) return this.libraryPromise; this.libraryPromise = new Promise((resolve, reject) => { const script = document.createElement("script"); script.src = this.librarySrc; script.async = true; script.onload = () => resolve(window.LightweightCharts); script.onerror = reject; document.head.append(script); }); return this.libraryPromise; }
    applyData() { if (!this.chart || !this.lineSeries) return; const series = Array.isArray(this.payload?.charts?.[this.mode]) ? this.payload.charts[this.mode] : []; this.lineSeries.applyOptions({ color: this.mode === "pnl" ? "#f0b90b" : "#0ecb81" }); this.lineSeries.setData(series.map((row, index) => ({ time: this.timeValue(row.x, index), value: number(row.y) }))); this.chart.timeScale?.().fitContent?.(); }
    drawOverlay() { const canvas = this.overlay; if (!canvas) return; this.resizeCanvas(canvas); const ctx = canvas.getContext("2d"); if (!ctx) return; const dpr = window.devicePixelRatio || 1; const width = canvas.width / dpr; const height = canvas.height / dpr; ctx.clearRect(0, 0, width, height); if (this.mode === "contribution") return this.bars(ctx, width, height); if (this.mode === "timeline") return this.timeline(ctx, width, height); if (!this.chart) this.line(ctx, width, height); }
    line(ctx, width, height) { const series = Array.isArray(this.payload?.charts?.[this.mode]) ? this.payload.charts[this.mode] : []; if (!series.length) return this.empty(ctx, width, height); const values = series.map((row) => number(row.y)); const min = Math.min(...values); const max = Math.max(...values); ctx.save(); ctx.strokeStyle = this.mode === "pnl" ? "rgba(240,185,11,0.92)" : "rgba(14,203,129,0.9)"; ctx.lineWidth = 2; ctx.beginPath(); series.forEach((row, index) => { const x = (index / Math.max(series.length - 1, 1)) * width; const y = height - ((number(row.y) - min) / Math.max(max - min, 1e-9)) * (height - 24) - 12; if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke(); ctx.restore(); }
    bars(ctx, width, height) { const rows = (this.payload?.asset_breakdown || this.payload?.result?.asset_breakdown || []).slice(0, 8); if (!rows.length) return this.empty(ctx, width, height); const max = Math.max(...rows.map((row) => Math.abs(number(row.pnl))), 1); const barH = Math.max(18, (height - 24) / rows.length - 7); ctx.save(); rows.forEach((row, i) => { const value = number(row.pnl); const y = 12 + i * (barH + 7); const w = Math.max(2, Math.abs(value) / max * (width - 120)); ctx.fillStyle = value >= 0 ? "rgba(14,203,129,0.78)" : "rgba(246,70,93,0.78)"; ctx.fillRect(92, y, w, barH); ctx.fillStyle = "rgba(244,247,251,0.88)"; ctx.font = "700 11px Inter, system-ui"; ctx.fillText(String(row.asset || row.symbol || "--").slice(0, 10), 10, y + barH - 4); }); ctx.restore(); }
    timeline(ctx, width, height) { const rows = (this.payload?.charts?.trade_timeline || []).slice(0, 40); if (!rows.length) return this.empty(ctx, width, height); ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.16)"; ctx.beginPath(); ctx.moveTo(16, height / 2); ctx.lineTo(width - 16, height / 2); ctx.stroke(); rows.forEach((row, index) => { const x = 16 + (index / Math.max(rows.length - 1, 1)) * (width - 32); ctx.fillStyle = number(row.pnl) >= 0 ? "#0ecb81" : "#f6465d"; ctx.beginPath(); ctx.arc(x, height / 2, 4, 0, Math.PI * 2); ctx.fill(); }); ctx.restore(); }
    resizeCanvas(canvas) { const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; const width = Math.max(1, Math.floor(rect.width * dpr)); const height = Math.max(1, Math.floor(rect.height * dpr)); if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0); } }
    empty(ctx, width, height) { ctx.save(); ctx.fillStyle = "rgba(255,255,255,0.58)"; ctx.font = "700 12px Inter, system-ui"; ctx.fillText("Awaiting portfolio data", 16, height / 2); ctx.restore(); }
    timeValue(value, index) { const parsed = number(value, 0); if (parsed > 10000000000) return Math.floor(parsed / 1000); if (parsed > 0) return Math.floor(parsed); return Math.floor(Date.now() / 1000) - (80 - index) * 60; }
  }
})();
