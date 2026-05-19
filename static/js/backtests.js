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
    submitButtons: Array.from(document.querySelectorAll("[data-backtest-submit], [data-backtest-top-submit]")),
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
    chartStats: document.querySelector("[data-chart-stats]"),
    resultInsights: document.querySelector("[data-result-insights]"),
    dataQualityPill: document.querySelector("[data-data-quality-pill]"),
    runtimePill: document.querySelector("[data-runtime-pill]"),
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
  const state = { symbols: [], allocationAssets: [], selectedAssets: new Set(), payload: null, chartMode: "equity", chart: null, symbolAbort: null, loading: false };

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
    state.loading = Boolean(loading);
    root.dataset.backtestState = loading ? "loading" : state.payload ? "ready" : "empty";
    form.setAttribute("aria-busy", loading ? "true" : "false");
    refs.submitButtons.forEach((button) => {
      button.disabled = loading || !canRun();
      button.textContent = loading
        ? button.dataset.loadingLabel || "Running..."
        : button.dataset.defaultLabel || "Run Backtest";
      button.setAttribute("aria-disabled", button.disabled ? "true" : "false");
    });
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
    refs.submitButtons.forEach((button) => { button.disabled = Boolean(message) || state.loading; });
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
    refs.symbolList?.setAttribute("aria-busy", "false");
    refs.assetList?.setAttribute("aria-busy", "false");
    updateSubmitState();
  };
  const loadSymbols = async () => {
    if (!urls.symbols) return;
    state.symbolAbort?.abort();
    state.symbolAbort = new AbortController();
    const url = new URL(apiUrl(urls.symbols), window.location.origin);
    url.searchParams.set("limit", "80");
    refs.symbolList?.classList.add("is-loading");
    refs.symbolList?.setAttribute("aria-busy", "true");
    refs.assetList?.setAttribute("aria-busy", "true");
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
      refs.symbolList?.setAttribute("aria-busy", "false");
      refs.assetList?.setAttribute("aria-busy", "false");
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
        price_status: String(row.price_status || row.state || "").toLowerCase(),
        price_source: String(row.price_source || ""),
        price_label: String(row.price_label || ""),
      })).filter((row) => row.asset);
    } else if (!state.allocationAssets.length) {
      state.allocationAssets = Array.from(refs.assetList?.querySelectorAll("[data-allocation-asset]") || []).map((node) => ({
        asset: String(node.dataset.asset || node.querySelector("input")?.value || "").toUpperCase(),
        available_usd: number(node.dataset.availableUsd),
        available_balance: 0,
        price_status: String(node.dataset.priceStatus || ""),
        price_source: String(node.dataset.priceSource || ""),
        price_label: "",
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
      const priceUnavailable = row.price_status === "unavailable" || row.state === "price_unavailable";
      const stateLabel = row.available_usd > 0 ? money.format(row.available_usd) : priceUnavailable ? "Price unavailable" : "No balance";
      const priceNote = priceUnavailable ? "excluded from MAX" : row.price_label || row.price_source || "priced";
      return `
        <label class="asset-option vault-asset-option backtest-allocation-option${selectedClass}" data-allocation-asset data-asset="${asset}" data-available-usd="${number(row.available_usd).toFixed(6)}" data-price-status="${escapeHtml(row.price_status)}" data-price-source="${escapeHtml(row.price_source)}">
          <input type="checkbox" name="allocation_assets" value="${asset}"${checked}>
          <span class="backtest-token-icon">${escapeHtml(asset.slice(0, 1) || "?")}</span>
          <span>
            <strong>${asset}</strong>
            <small>${number(row.available_balance).toFixed(6)} · ${escapeHtml(stateLabel)} · ${escapeHtml(priceNote)}</small>
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
    renderQualityRuntime(payload.data_quality_summary || payload.result?.data_quality_summary || {}, payload.runtime_diagnostics || payload.result?.runtime_diagnostics || {});
    renderResultInsights(payload);
    renderAutopilot(payload.autopilot || {});
    renderExecution(payload.execution_quality || {}, payload.trade_decision || {}, payload.simulation_scope || {});
    renderPortfolioDiagnostics(payload.portfolio_diagnostics || payload.result?.portfolio_diagnostics || {});
    renderAssetBreakdown(payload.asset_diagnostics || payload.result?.asset_diagnostics || payload.asset_breakdown || payload.result?.asset_breakdown || []);
    renderStrategyWeights(payload.strategy_weight_groups || payload.result?.strategy_weight_groups || [], payload.strategy_weights || payload.result?.strategy_weights || []);
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
  const renderQualityRuntime = (quality, runtime) => {
    if (refs.dataQualityPill) {
      const status = String(quality.status || "unknown").replace(/_/g, " ");
      refs.dataQualityPill.textContent = `Data ${status} · ${percent.format(number(quality.score))}`;
      refs.dataQualityPill.dataset.state = quality.status || "";
    }
    if (refs.runtimePill) {
      const elapsed = number(runtime.elapsed_ms);
      const workers = Math.round(number(runtime.max_workers, 1));
      refs.runtimePill.textContent = `${elapsed >= 1000 ? `${(elapsed / 1000).toFixed(1)}s` : `${Math.round(elapsed)}ms`} · ${workers}w`;
    }
  };
  const setMetric = (key, text, polarity = null) => {
    const node = metricNodes[key]; if (!node) return;
    node.textContent = text;
    if (polarity !== null) { node.classList.toggle("positive", polarity >= 0); node.classList.toggle("negative", polarity < 0); }
  };
  const renderAutopilot = (autopilot) => {
    if (refs.autopilotConfidence) refs.autopilotConfidence.textContent = percent.format(number(autopilot.confidence));
    renderKeyValues(refs.autopilotList, [["Status", formatState(autopilot.status || "ready")], ["Regime", formatState(autopilot.market_regime || "aggregate")], ["Models", Array.isArray(autopilot.model_stack) ? autopilot.model_stack.length : 0], ["Active", autopilot.active_strategy_count ?? "--"]]);
  };
  const renderExecution = (execution, tradeDecision = {}, scope = {}) => {
    if (refs.executionScore) refs.executionScore.textContent = percent.format(number(execution.fill_quality));
    renderKeyValues(refs.executionList, [["Mode", tradeDecision.mode || "backtest"], ["Worker", scope.queues_worker ? "queued" : "none"], ["Broker", tradeDecision.broker_order_submitted || scope.submits_broker_order ? "submitted" : "no"], ["Pairs", execution.eligible_pair_count || "--"], ["Fees", `${number(execution.fee_bps).toFixed(2)} bps`], ["Slippage", `${number(execution.slippage_bps).toFixed(2)} bps`], ["Liquidity", compactMoney.format(number(execution.liquidity_usd))], ["Exposure", compactMoney.format(number(execution.max_exposure_usd))]]);
  };
  const renderPortfolioDiagnostics = (diagnostics) => {
    if (refs.allocationPolicy) refs.allocationPolicy.textContent = String(diagnostics.allocation_policy || "after-cost").replace(/_/g, " ");
    const skipped = diagnostics.skipped_reasons || {};
    const skippedText = Object.entries(skipped).map(([reason, count]) => `${formatReason(reason)} (${count})`).join(", ") || "none";
    renderKeyValues(refs.portfolioDiagnostics, [["Allocated", diagnostics.allocated_candidate_count || 0], ["Skipped", diagnostics.skipped_candidate_count || 0], ["Score", number(diagnostics.total_after_cost_score).toFixed(3)], ["Reasons", skippedText], ["Gates", String(diagnostics.live_authority || "server risk gates").replace(/_/g, " ")]]);
  };
  const renderResultInsights = (payload) => {
    if (!refs.resultInsights) return;
    const metrics = payload.metrics || {};
    const diagnostics = payload.portfolio_diagnostics || payload.result?.portfolio_diagnostics || {};
    const quality = payload.data_quality_summary || payload.result?.data_quality_summary || {};
    const runtime = payload.runtime_diagnostics || payload.result?.runtime_diagnostics || {};
    const assets = assetContributionRows(payload);
    const bestAsset = assets.find((row) => number(row.pnl) > 0) || assets[0] || null;
    const elapsed = number(runtime.elapsed_ms);
    const cacheHits = Object.values(runtime.cache_hits || {}).reduce((total, value) => total + number(value), 0);
    const skipped = number(diagnostics.skipped_candidate_count);
    const insightRows = [
      ["Best asset", bestAsset ? `${bestAsset.asset || bestAsset.symbol || "--"} ${money.format(number(bestAsset.pnl))}` : "--", bestAsset && number(bestAsset.pnl) >= 0 ? "positive" : ""],
      ["Capital", money.format(number(metrics.initial_balance ?? payload.summary?.allocation)), ""],
      ["Allocated", `${number(diagnostics.allocated_candidate_count, assets.length)} assets`, ""],
      ["Skipped", `${skipped} assets`, skipped ? "warning" : ""],
      ["Data", percent.format(number(quality.score)), quality.status === "degraded" ? "warning" : "positive"],
      ["Runtime", elapsed >= 1000 ? `${(elapsed / 1000).toFixed(1)}s` : `${Math.round(elapsed)}ms`, ""],
      ["Cache", `${compact.format(cacheHits)} hits`, cacheHits ? "positive" : ""],
    ];
    refs.resultInsights.innerHTML = insightRows.map(([label, value, tone]) => `
      <div class="backtest-insight-chip${tone ? ` is-${tone}` : ""}">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>`).join("");
  };
  const renderAssetBreakdown = (rows) => {
    if (!refs.assetBreakdown) return;
    if (refs.activeStrategies) refs.activeStrategies.textContent = `${rows.length} assets`;
    if (!rows.length) { refs.assetBreakdown.innerHTML = '<div class="backtest-symbol-empty">No asset-level results yet.</div>'; return; }
    const sortedRows = rows.slice().sort((a, b) => Math.abs(number(b.pnl)) - Math.abs(number(a.pnl))).slice(0, 12);
    const cards = sortedRows.map((row) => {
      const status = String(row.status || (row.error ? "failed" : "simulated")).toLowerCase();
      const validation = row.market_history_validation || {};
      const detail = assetDetail(row, validation);
      const historyText = compactHistoryLabel(validation, row.fallback_timeframe);
      const qualityText = qualityLabel(validation);
      const pnl = number(row.pnl);
      const roi = number(row.roi);
      const weight = number(row.allocation_weight);
      return `
      <article class="backtest-asset-card is-${escapeHtml(status)}">
        <div class="backtest-asset-card-head">
          <div>
            <strong>${escapeHtml(row.asset || row.symbol || "--")}</strong>
            <small>${escapeHtml(detail)}</small>
          </div>
          <em class="backtest-status-chip is-${escapeHtml(status)}">${escapeHtml(statusLabel(row, status))}</em>
        </div>
        <div class="backtest-asset-card-metrics">
          <span><small>PnL</small><strong class="${pnl >= 0 ? "positive" : "negative"}">${money.format(pnl)}</strong></span>
          <span><small>ROI</small><strong class="${roi >= 0 ? "positive" : "negative"}">${percent.format(roi)}</strong></span>
          <span><small>Weight</small><strong>${percent.format(weight)}</strong></span>
          <span><small>Trades</small><strong>${Math.round(number(row.trades ?? row.trade_count))}</strong></span>
        </div>
        <div class="backtest-asset-card-foot">
          <span>${escapeHtml(historyText || "History --")}</span>
          <span>${escapeHtml(qualityText)}</span>
        </div>
      </article>`;
    }).join("");
    const tableRows = sortedRows.map((row) => {
      const status = String(row.status || (row.error ? "failed" : "simulated")).toLowerCase();
      const validation = row.market_history_validation || {};
      const historyText = compactHistoryLabel(validation, row.fallback_timeframe);
      const detail = assetDetail(row, validation);
      const qualityText = qualityLabel(validation);
      return `
      <tr class="backtest-asset-row is-${escapeHtml(status)}">
        <th scope="row">
          <strong>${escapeHtml(row.asset || row.symbol || "--")}</strong>
          <small>${escapeHtml(detail)}</small>
          <span class="backtest-history-note">${escapeHtml(historyText)}</span>
        </th>
        <td><em class="backtest-status-chip is-${escapeHtml(status)}">${escapeHtml(statusLabel(row, status))}</em></td>
        <td><strong class="${number(row.pnl) >= 0 ? "positive" : "negative"}">${money.format(number(row.pnl))}</strong></td>
        <td>${percent.format(number(row.roi))}</td>
        <td>${percent.format(number(row.allocation_weight))}</td>
        <td>${number(row.net_expected_return_bps).toFixed(1)} bps</td>
        <td>${number(row.cost_drag_bps).toFixed(1)} bps</td>
        <td>${Math.round(number(row.trades ?? row.trade_count))}</td>
        <td>${escapeHtml(qualityText)}</td>
      </tr>`;
    }).join("");
    refs.assetBreakdown.innerHTML = `
      <div class="backtest-asset-cards" data-mobile-asset-cards>${cards}</div>
      <table class="backtest-asset-table">
        <thead><tr><th>Asset</th><th>Status</th><th>PnL</th><th>ROI</th><th>Weight</th><th>Edge</th><th>Cost</th><th>Trades</th><th>Quality</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>`;
  };
  const renderStrategyWeights = (groups, rows) => {
    if (!refs.strategyWeights) return;
    const groupedRows = Array.isArray(groups) && groups.length ? groups : groupStrategyRows(rows);
    if (groupedRows.length) {
      if (refs.strategyWeightCount) refs.strategyWeightCount.textContent = `${groupedRows.reduce((total, group) => total + number(group.active_count), 0)} active`;
      refs.strategyWeights.innerHTML = groupedRows.map((group, index) => {
        const reasons = group.disabled_reasons || {};
        const reasonText = Object.entries(reasons).map(([reason, count]) => `${formatReason(reason)} (${count})`).join(", ") || "All active";
        const childRows = Array.isArray(group.rows) ? group.rows : [];
        return `
          <details class="backtest-strategy-group" ${number(group.active_count) > 0 ? "open" : ""}>
            <summary>
              <span class="backtest-strategy-group-copy">
                <strong>${escapeHtml(group.label || "Strategy")}</strong>
                <small>
                  <span class="backtest-strategy-pill is-active">${number(group.active_count)} active</span>
                  <span class="backtest-strategy-pill">${number(group.disabled_count)} off</span>
                  <span class="backtest-strategy-reason">${escapeHtml(reasonText)}</span>
                </small>
              </span>
              <em>${percent.format(number(group.total_weight))}</em>
            </summary>
            <div>
              ${childRows.slice(0, 8).map((row) => strategyRow(row)).join("")}
            </div>
          </details>`;
      }).join("");
      return;
    }
    const sorted = rows.slice().sort((a, b) => Number(Boolean(b.enabled)) - Number(Boolean(a.enabled)) || number(b.weight) - number(a.weight)).slice(0, 16);
    if (refs.strategyWeightCount) refs.strategyWeightCount.textContent = `${sorted.filter((row) => row.enabled).length} active`;
    if (!sorted.length) { refs.strategyWeights.innerHTML = '<div class="backtest-symbol-empty">No strategy weights yet.</div>'; return; }
    refs.strategyWeights.innerHTML = sorted.map((row) => strategyRow(row)).join("");
  };
  const renderSystemMetrics = (metrics) => Object.entries(systemNodes).forEach(([key, node]) => { node.textContent = metrics[key] || node.textContent || "Auto"; });
  const renderKeyValues = (host, rows) => { if (host) host.innerHTML = rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`).join(""); };
  const renderChartStats = (payload, mode) => {
    if (!refs.chartStats) return;
    const metrics = payload.metrics || {};
    const charts = payload.charts || {};
    const series = Array.isArray(charts[mode]) ? charts[mode] : [];
    const values = series.map((row) => number(row.y)).filter(Number.isFinite);
    const assets = assetContributionRows(payload);
    const trades = Array.isArray(charts.trade_timeline) ? charts.trade_timeline : [];
    const qualities = qualityRowsFromPayload(payload);
    const dataQuality = payload.data_quality_summary || payload.result?.data_quality_summary || {};
    let rows = [];
    if (mode === "assets") {
      const best = assets.find((row) => number(row.pnl) > 0) || assets[0] || {};
      const skipped = number(payload.portfolio_diagnostics?.skipped_candidate_count || payload.result?.portfolio_diagnostics?.skipped_candidate_count);
      rows = [["Best", best.asset || best.symbol || "--", "positive"], ["Asset PnL", money.format(number(best.pnl)), number(best.pnl) >= 0 ? "positive" : "negative"], ["Positive", String(assets.filter((row) => number(row.pnl) > 0).length), ""], ["Skipped", String(skipped), skipped ? "warning" : ""]];
    } else if (mode === "trades") {
      rows = [["Trades", String(Math.round(number(metrics.trades))), ""], ["Winners", String(trades.filter((row) => number(row.pnl) >= 0).length), "positive"], ["Losers", String(trades.filter((row) => number(row.pnl) < 0).length), "negative"], ["Avg", money.format(number(metrics.average_trade)), ""]];
    } else if (mode === "quality") {
      const average = qualities.length ? qualities.reduce((total, row) => total + number(row.score), 0) / qualities.length : number(dataQuality.score);
      const issues = number(dataQuality.gap_count) + number(dataQuality.malformed_candle_count) + number(dataQuality.duplicate_timestamp_count) + number(dataQuality.outlier_candle_count);
      rows = [["Score", percent.format(average), average >= 0.9 ? "positive" : "warning"], ["Assets", String(number(dataQuality.asset_count, qualities.length)), ""], ["Valid", compact.format(number(dataQuality.valid_candle_count)), ""], ["Issues", compact.format(issues), issues ? "warning" : "positive"]];
    } else {
      const current = values.length ? values[values.length - 1] : 0;
      const high = values.length ? Math.max(...values) : 0;
      const low = values.length ? Math.min(...values) : 0;
      const formatter = mode === "drawdown" ? percent : money;
      rows = [["Current", formatter.format(current), mode === "drawdown" || current < 0 ? "negative" : "positive"], ["High", formatter.format(high), "positive"], ["Low", formatter.format(low), low < 0 ? "negative" : ""], ["Points", String(series.length), ""]];
    }
    refs.chartStats.innerHTML = rows.map(([label, value, tone]) => `
      <div class="backtest-chart-stat${tone ? ` is-${tone}` : ""}">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>`).join("");
  };

  const renderChart = async () => {
    if (refs.chartTitle) refs.chartTitle.textContent = chartLabel(state.chartMode);
    renderChartStats(state.payload || {}, state.chartMode);
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
  document.querySelectorAll("[data-chart-mode]").forEach((button) => {
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", button.classList.contains("is-active") ? "true" : "false");
    button.addEventListener("click", () => {
      state.chartMode = button.dataset.chartMode || "equity";
      document.querySelectorAll("[data-chart-mode]").forEach((item) => {
        const active = item === button;
        item.classList.toggle("is-active", active);
        item.setAttribute("aria-selected", active ? "true" : "false");
      });
      renderChart();
    });
  });
  form.addEventListener("submit", submitForm);
  window.addEventListener("resize", () => state.chart?.drawOverlay?.());
  window.addEventListener("online", updateSubmitState); window.addEventListener("offline", updateSubmitState);

  const initialSymbols = readJson(refs.initialSymbols); const initialPayload = readJson(refs.payload);
  if (initialSymbols?.symbols) applySymbols(initialSymbols); else loadSymbols();
  if (initialPayload?.ok) renderPayload(initialPayload);
  syncAllocation(refs.allocation || { value: 0 });

  function chartLabel(mode) {
    return {
      equity: "Portfolio Equity Curve",
      pnl: "Portfolio PnL",
      drawdown: "Drawdown Curve",
      assets: "Asset Contribution",
      trades: "Trade Timeline",
      quality: "Data Quality",
    }[mode] || "Portfolio Simulation";
  }
  function formatReason(value) {
    const raw = String(value ?? "").trim();
    if (!raw || raw.toLowerCase() === "none") return "";
    const normalized = raw.toLowerCase().replace(/[\s-]+/g, "_");
    const labels = {
      disabled: "Off",
      failed: "Failed",
      insufficient_history: "Insufficient history",
      negative_after_cost_return: "Negative after-cost return",
      no_after_cost_trades: "No after-cost trades",
      no_positive_after_cost_trades: "No after-cost trades",
      no_trades: "No trades",
      price_unavailable: "Price unavailable",
      stale_market_data: "Stale market data",
      zero_allocation: "No allocation",
    };
    if (labels[normalized]) return labels[normalized];
    return raw.replace(/[_-]+/g, " ").replace(/\s+/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }
  function formatState(value) {
    const label = formatReason(value);
    return label || String(value || "--");
  }
  function assetContributionRows(payload) {
    const rows = payload?.asset_contribution || payload?.charts?.asset_contribution || payload?.asset_diagnostics || payload?.asset_breakdown || payload?.result?.asset_breakdown || [];
    return Array.isArray(rows) ? rows.slice().sort((a, b) => Math.abs(number(b.pnl)) - Math.abs(number(a.pnl))) : [];
  }
  function qualityRowsFromPayload(payload) {
    const rows = payload?.charts?.data_quality || payload?.data_quality_summary?.assets || payload?.asset_diagnostics || [];
    return Array.isArray(rows) ? rows.map((row) => ({ ...row, score: row.score ?? row.data_quality_score ?? row.market_history_validation?.data_quality_score })) : [];
  }
  function assetDetail(row, validation) {
    const reason = formatReason(row.error || row.skip_reason || row.disabled_reason || "");
    if (reason) return reason;
    return fundingLabel(row) || compactHistoryLabel(validation, row.fallback_timeframe) || row.exchange || row.provider_label || "Enabled venue";
  }
  function statusLabel(row, fallback) {
    return formatReason(row.status_label || fallback || row.status || "simulated") || "Simulated";
  }
  function qualityLabel(validation) {
    const quality = number(validation?.data_quality_score, NaN);
    if (Number.isFinite(quality)) return percent.format(quality);
    const count = historyCount(validation);
    return count === "--" ? "Quality --" : count;
  }
  function strategyRow(row) {
    const enabled = Boolean(row.enabled);
    const objective = `ROI eff ${percent.format(number(row.roi_efficiency_score))} · 10x ${percent.format(number(row.ten_x_target_probability))}`;
    const activeMeta = `${row.asset || "Portfolio"} · ${percent.format(number(row.weight))} weight · ${objective}`;
    const disabledReason = formatReason(row.disabled_reason || "disabled");
    const meta = enabled ? escapeHtml(activeMeta) : `<em class="backtest-reason-chip">Off</em>${escapeHtml(disabledReason)}`;
    return `
      <div class="backtest-strategy-row${enabled ? "" : " is-disabled"}" style="--weight:${Math.max(0, Math.min(number(row.weight), 1)).toFixed(4)}">
        <div><strong>${escapeHtml(row.label || row.strategy_name || "Strategy")}</strong><small>${meta}</small></div>
        <span>${number(row.net_return_after_costs ?? row.total_return).toFixed(4)}</span>
        <i aria-hidden="true"></i>
      </div>`;
  }
  function groupStrategyRows(rows) {
    const groups = new Map();
    rows.forEach((row) => {
      const label = String(row.label || row.strategy_name || "Strategy");
      if (!groups.has(label)) groups.set(label, { label, active_count: 0, disabled_count: 0, total_weight: 0, rows: [], disabled_reasons: {} });
      const group = groups.get(label);
      group.rows.push(row);
      if (row.enabled) {
        group.active_count += 1;
        group.total_weight += number(row.weight);
      } else {
        group.disabled_count += 1;
        const reason = String(row.disabled_reason || "disabled");
        group.disabled_reasons[reason] = (group.disabled_reasons[reason] || 0) + 1;
      }
    });
    return Array.from(groups.values()).sort((a, b) => Number(b.active_count > 0) - Number(a.active_count > 0) || b.total_weight - a.total_weight);
  }
  function historyCount(validation) {
    if (!validation || typeof validation !== "object") return "--";
    const valid = number(validation.valid_candle_count, NaN);
    const required = number(validation.required_candle_count, NaN);
    if (!Number.isFinite(valid) || !Number.isFinite(required)) return "--";
    return `${Math.round(valid)}/${Math.round(required)}`;
  }
  function compactHistoryLabel(validation, fallback) {
    if (!validation || typeof validation !== "object") return fallback ? `Fallback ${fallback}` : "";
    const source = validation.source_timeframe || validation.requested_timeframe || "";
    const count = historyCount(validation);
    const issues = number(validation.gap_count) + number(validation.malformed_candle_count) + number(validation.duplicate_timestamp_count) + number(validation.outlier_candle_count);
    const parts = [];
    if (count !== "--") parts.push(`${count} valid`);
    if (source) parts.push(source);
    if (fallback || validation.fallback_timeframe) parts.push(`fallback ${fallback || validation.fallback_timeframe}`);
    if (issues) parts.push(`${issues} issue${issues === 1 ? "" : "s"}`);
    return parts.join(" · ");
  }
  function fundingLabel(row) {
    const fundingAsset = String(row.funding_asset || row.vault_allocation_asset || "").toUpperCase();
    const collateralAsset = String(row.collateral_asset || row.quote_asset || "").toUpperCase();
    if (row.conversion_required && fundingAsset && collateralAsset) return `${fundingAsset} -> ${collateralAsset} paper`;
    if (fundingAsset && collateralAsset && fundingAsset !== collateralAsset) return `${fundingAsset} funding · ${collateralAsset} collateral`;
    return fundingAsset ? `${fundingAsset} funded` : "";
  }
  function escapeHtml(value) { return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  class BacktestChart {
    constructor(host, overlay, librarySrc) { this.host = host; this.overlay = overlay; this.librarySrc = librarySrc; this.chart = null; this.lineSeries = null; this.payload = null; this.mode = "equity"; this.libraryPromise = null; }
    async render(payload, mode) { this.payload = payload || {}; this.mode = mode || "equity"; try { await this.ensureChart(); this.applyData(); } catch (error) {} this.drawOverlay(); }
    async ensureChart() {
      if (!this.host) return false;
      const canvasOnly = ["assets", "trades", "quality"].includes(this.mode);
      this.host.hidden = canvasOnly;
      this.host.setAttribute("aria-hidden", canvasOnly ? "true" : "false");
      if (canvasOnly) return false;
      const library = await this.loadLibrary(); if (!library?.createChart) return false;
      if (!this.chart) { this.chart = library.createChart(this.host, { autoSize: true, layout: { background: { color: "#05070b" }, textColor: "#8f9bae", fontFamily: "Inter, system-ui, sans-serif" }, grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.055)" } }, rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" }, timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false }, handleScroll: { horzTouchDrag: true, vertTouchDrag: false }, handleScale: { pinch: true, mouseWheel: false } }); this.lineSeries = this.chart.addLineSeries({ color: "#0ecb81", lineWidth: 2, priceLineVisible: false }); }
      return true;
    }
    loadLibrary() { if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts); if (this.libraryPromise) return this.libraryPromise; this.libraryPromise = new Promise((resolve, reject) => { const script = document.createElement("script"); script.src = this.librarySrc; script.async = true; script.onload = () => resolve(window.LightweightCharts); script.onerror = reject; document.head.append(script); }); return this.libraryPromise; }
    applyData() {
      if (!this.chart || !this.lineSeries || ["assets", "trades", "quality"].includes(this.mode)) return;
      const series = Array.isArray(this.payload?.charts?.[this.mode]) ? this.payload.charts[this.mode] : [];
      const color = this.mode === "pnl" ? "#f0b90b" : this.mode === "drawdown" ? "#f6465d" : "#0ecb81";
      this.lineSeries.applyOptions({ color });
      this.lineSeries.setData(series.map((row, index) => ({ time: this.timeValue(row.x, index), value: number(row.y) })));
      this.chart.timeScale?.().fitContent?.();
    }
    drawOverlay() {
      const canvas = this.overlay; if (!canvas) return;
      this.resizeCanvas(canvas);
      const ctx = canvas.getContext("2d"); if (!ctx) return;
      const dpr = window.devicePixelRatio || 1; const width = canvas.width / dpr; const height = canvas.height / dpr;
      ctx.clearRect(0, 0, width, height);
      if (["assets", "trades", "quality"].includes(this.mode)) this.grid(ctx, width, height);
      if (this.mode === "assets") return this.bars(ctx, width, height);
      if (this.mode === "trades") return this.timeline(ctx, width, height);
      if (this.mode === "quality") return this.quality(ctx, width, height);
      if (!this.chart || !window.LightweightCharts) this.line(ctx, width, height);
    }
    grid(ctx, width, height) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.055)";
      ctx.lineWidth = 1;
      for (let index = 1; index <= 3; index += 1) {
        const y = (height / 4) * index;
        ctx.beginPath(); ctx.moveTo(8, y); ctx.lineTo(width - 8, y); ctx.stroke();
      }
      ctx.restore();
    }
    line(ctx, width, height) {
      const series = Array.isArray(this.payload?.charts?.[this.mode]) ? this.payload.charts[this.mode] : [];
      if (!series.length) return this.empty(ctx, width, height);
      const values = series.map((row) => number(row.y)); const min = Math.min(...values); const max = Math.max(...values);
      const color = this.mode === "pnl" ? "rgba(240,185,11,0.92)" : this.mode === "drawdown" ? "rgba(246,70,93,0.9)" : "rgba(14,203,129,0.9)";
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      series.forEach((row, index) => {
        const x = (index / Math.max(series.length - 1, 1)) * width;
        const y = height - ((number(row.y) - min) / Math.max(max - min, 1e-9)) * (height - 24) - 12;
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke(); ctx.restore();
    }
    bars(ctx, width, height) {
      const rows = this.assetRows().slice(0, 10);
      if (!rows.length) return this.empty(ctx, width, height);
      const max = Math.max(...rows.map((row) => Math.abs(number(row.pnl))), 1);
      const barH = Math.max(16, (height - 24) / rows.length - 7);
      ctx.save();
      rows.forEach((row, i) => {
        const value = number(row.pnl); const y = 12 + i * (barH + 7); const labelWidth = Math.min(92, Math.max(58, width * 0.26));
        const w = Math.max(2, Math.abs(value) / max * Math.max(width - labelWidth - 20, 24));
        ctx.fillStyle = value >= 0 ? "rgba(14,203,129,0.78)" : "rgba(246,70,93,0.78)";
        ctx.fillRect(labelWidth, y, w, barH);
        ctx.fillStyle = "rgba(244,247,251,0.9)";
        ctx.font = "700 11px Inter, system-ui";
        ctx.textAlign = "left";
        ctx.fillText(String(row.asset || row.symbol || "--").slice(0, 10), 10, y + barH - 4);
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(169,181,200,0.92)";
        ctx.fillText(money.format(value), width - 10, y + barH - 4);
      });
      ctx.restore();
    }
    timeline(ctx, width, height) {
      const rows = (this.payload?.charts?.trade_timeline || []).slice(0, 48);
      if (!rows.length) return this.empty(ctx, width, height);
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.beginPath(); ctx.moveTo(16, height / 2); ctx.lineTo(width - 16, height / 2); ctx.stroke();
      ctx.fillStyle = "rgba(169,181,200,0.92)";
      ctx.font = "800 11px Inter, system-ui";
      ctx.fillText(`${rows.length} fills`, 16, 18);
      rows.forEach((row, index) => {
        const x = 16 + (index / Math.max(rows.length - 1, 1)) * (width - 32);
        const radius = Math.max(3, Math.min(6, Math.sqrt(Math.abs(number(row.pnl))) + 3));
        ctx.fillStyle = number(row.pnl) >= 0 ? "#0ecb81" : "#f6465d";
        ctx.beginPath(); ctx.arc(x, height / 2, radius, 0, Math.PI * 2); ctx.fill();
      });
      ctx.restore();
    }
    quality(ctx, width, height) {
      const rows = this.qualityRows().slice(0, 10);
      if (!rows.length) return this.empty(ctx, width, height);
      const barH = Math.max(16, (height - 24) / rows.length - 7);
      ctx.save();
      rows.forEach((row, i) => {
        const score = Math.max(0, Math.min(number(row.score ?? row.data_quality_score), 1));
        const y = 12 + i * (barH + 7); const labelWidth = Math.min(92, Math.max(58, width * 0.26));
        const w = Math.max(2, score * Math.max(width - labelWidth - 20, 24));
        ctx.fillStyle = score >= 0.9 ? "rgba(14,203,129,0.78)" : score >= 0.72 ? "rgba(240,185,11,0.78)" : "rgba(246,70,93,0.78)";
        ctx.fillRect(labelWidth, y, w, barH);
        ctx.fillStyle = "rgba(244,247,251,0.9)";
        ctx.font = "700 11px Inter, system-ui";
        ctx.textAlign = "left";
        ctx.fillText(String(row.asset || row.symbol || "--").slice(0, 10), 10, y + barH - 4);
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(169,181,200,0.92)";
        ctx.fillText(percent.format(score), width - 10, y + barH - 4);
      });
      ctx.restore();
    }
    assetRows() {
      const rows = this.payload?.asset_contribution || this.payload?.charts?.asset_contribution || this.payload?.asset_diagnostics || this.payload?.asset_breakdown || this.payload?.result?.asset_breakdown || [];
      return Array.isArray(rows) ? rows.slice().sort((a, b) => Math.abs(number(b.pnl)) - Math.abs(number(a.pnl))) : [];
    }
    qualityRows() {
      const rows = this.payload?.charts?.data_quality || this.payload?.data_quality_summary?.assets || this.payload?.asset_diagnostics || [];
      return Array.isArray(rows) ? rows.map((row) => ({ ...row, score: row.score ?? row.data_quality_score ?? row.market_history_validation?.data_quality_score })) : [];
    }
    resizeCanvas(canvas) { const rect = canvas.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1; const width = Math.max(1, Math.floor(rect.width * dpr)); const height = Math.max(1, Math.floor(rect.height * dpr)); if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0); } }
    empty(ctx, width, height) { ctx.save(); ctx.fillStyle = "rgba(255,255,255,0.58)"; ctx.font = "700 12px Inter, system-ui"; ctx.fillText("Awaiting portfolio data", 16, height / 2); ctx.restore(); }
    timeValue(value, index) { const parsed = number(value, 0); if (parsed > 10000000000) return Math.floor(parsed / 1000); if (parsed > 0) return Math.floor(parsed); return Math.floor(Date.now() / 1000) - (80 - index) * 60; }
  }
})();
