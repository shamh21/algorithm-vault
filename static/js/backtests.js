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
    search: document.querySelector("[data-symbol-search]"),
    symbolList: document.querySelector("[data-symbol-list]"),
    selectedAsset: document.querySelector("[data-selected-asset]"),
    providerInput: document.querySelector("[data-backtest-provider]"),
    symbolInput: document.querySelector("[data-backtest-symbol]"),
    venueInput: document.querySelector("[data-backtest-venue-symbol]"),
    timeframeInput: document.querySelector("[data-backtest-timeframe]"),
    allocation: document.querySelector("[data-backtest-allocation]"),
    slider: document.querySelector("[data-allocation-slider]"),
    maxButton: document.querySelector("[data-backtest-max]"),
    conversion: document.querySelector("[data-conversion-preview]"),
    chartHost: document.querySelector("[data-backtest-chart]"),
    chartOverlay: document.querySelector("[data-backtest-overlay]"),
    chartTitle: document.querySelector("[data-chart-title]"),
    autopilotConfidence: document.querySelector("[data-autopilot-confidence]"),
    executionScore: document.querySelector("[data-execution-score]"),
    activeStrategies: document.querySelector("[data-active-strategies]"),
    autopilotList: document.querySelector("[data-autopilot-list]"),
    executionList: document.querySelector("[data-execution-list]"),
    strategyList: document.querySelector("[data-strategy-list]"),
  };

  const urls = {
    symbols: root.dataset.symbolsUrl || "",
    quote: root.dataset.quoteUrl || "",
    chartLib: root.dataset.chartLibSrc || "",
  };
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;

  const metricNodes = Object.fromEntries(
    Array.from(document.querySelectorAll("[data-metric]")).map((node) => [node.dataset.metric, node])
  );
  const systemNodes = Object.fromEntries(
    Array.from(document.querySelectorAll("[data-system-metric]")).map((node) => [node.dataset.systemMetric, node])
  );

  const state = {
    symbols: [],
    nextCursor: null,
    hasMore: false,
    selected: null,
    payload: null,
    chartMode: "market",
    timeframe: "live",
    searchTimer: null,
    symbolAbort: null,
    quoteAbort: null,
    chart: null,
  };

  const cacheKey = "algorithmVault.backtest.symbols.v1";
  const cacheTtl = 5 * 60 * 1000;
  const money = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
  const compactMoney = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 });
  const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
  const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });

  const readJson = (node) => {
    if (!node?.textContent?.trim()) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (error) {
      return null;
    }
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
    refs.submit.disabled = loading;
    if (loading) {
      refs.submit.dataset.originalLabel = refs.submit.textContent;
      refs.submit.textContent = refs.submit.dataset.loadingLabel || "Simulating...";
      refs.dashboard?.classList.add("is-loading");
    } else {
      refs.submit.textContent = refs.submit.dataset.originalLabel || "Run Vault Simulation";
      refs.dashboard?.classList.remove("is-loading");
    }
  };

  const cacheSymbols = (payload) => {
    try {
      localStorage.setItem(cacheKey, JSON.stringify({ at: Date.now(), payload }));
    } catch (error) {
      // Cache is optional.
    }
  };

  const cachedSymbols = () => {
    try {
      const cached = JSON.parse(localStorage.getItem(cacheKey) || "{}");
      if (!cached.at || Date.now() - cached.at > cacheTtl) return null;
      return cached.payload || null;
    } catch (error) {
      return null;
    }
  };

  const normalizeSymbolPayload = (payload) => {
    const rows = Array.isArray(payload?.symbols) ? payload.symbols : [];
    return {
      symbols: rows,
      nextCursor: payload?.next_cursor || null,
      hasMore: Boolean(payload?.has_more),
    };
  };

  const applySymbols = (payload, { append = false, cache = false } = {}) => {
    const normalized = normalizeSymbolPayload(payload);
    state.symbols = append ? mergeSymbols(state.symbols, normalized.symbols) : normalized.symbols;
    state.nextCursor = normalized.nextCursor;
    state.hasMore = normalized.hasMore;
    if (cache && !append) cacheSymbols(payload);
    if (!state.selected && state.symbols.length) selectSymbol(state.symbols[0], { quote: false });
    renderSymbols();
    refreshQuote();
  };

  const mergeSymbols = (current, incoming) => {
    const seen = new Set(current.map((row) => `${row.provider}:${row.venue_symbol}:${row.symbol}`));
    const merged = current.slice();
    incoming.forEach((row) => {
      const key = `${row.provider}:${row.venue_symbol}:${row.symbol}`;
      if (seen.has(key)) return;
      seen.add(key);
      merged.push(row);
    });
    return merged;
  };

  const loadSymbols = async ({ append = false, refresh = false } = {}) => {
    if (!urls.symbols) return;
    state.symbolAbort?.abort();
    state.symbolAbort = new AbortController();
    const url = new URL(apiUrl(urls.symbols), window.location.origin);
    const query = refs.search?.value?.trim() || "";
    if (query) url.searchParams.set("q", query);
    if (append && state.nextCursor) url.searchParams.set("cursor", state.nextCursor);
    if (refresh) url.searchParams.set("refresh", "1");
    url.searchParams.set("limit", "40");
    refs.symbolList?.classList.add("is-loading");
    try {
      const response = await fetch(url, { signal: state.symbolAbort.signal, headers: { Accept: "application/json" } });
      const payload = await response.json();
      if (payload?.ok) applySymbols(payload, { append, cache: !query && !append });
    } catch (error) {
      if (error.name !== "AbortError") renderSymbols();
    } finally {
      refs.symbolList?.classList.remove("is-loading");
    }
  };

  const renderSymbols = () => {
    if (!refs.symbolList) return;
    if (!state.symbols.length) {
      refs.symbolList.innerHTML = '<div class="backtest-symbol-empty">No supported assets found.</div>';
      return;
    }
    const rows = state.symbols.slice(0, 80);
    refs.symbolList.innerHTML = rows.map((row) => symbolButton(row)).join("") + (state.hasMore ? '<button type="button" class="backtest-load-more" data-load-more-symbols>Load more</button>' : "");
    refs.symbolList.querySelectorAll("[data-symbol-row]").forEach((button) => {
      button.addEventListener("click", () => {
        const row = state.symbols.find((item) => button.dataset.key === symbolKey(item));
        if (row) selectSymbol(row);
      });
    });
    refs.symbolList.querySelector("[data-load-more-symbols]")?.addEventListener("click", () => loadSymbols({ append: true }));
  };

  const symbolButton = (row) => {
    const selected = state.selected && symbolKey(state.selected) === symbolKey(row);
    const badges = Array.isArray(row.compatibility_badges) ? row.compatibility_badges.slice(0, 2) : [];
    return `
      <button type="button" class="backtest-symbol-row ${selected ? "is-active" : ""}" role="option" aria-selected="${selected ? "true" : "false"}" data-symbol-row data-key="${escapeHtml(symbolKey(row))}">
        ${cryptoIconMarkup(row.symbol)}
        <span class="backtest-symbol-copy">
          <strong>${escapeHtml(row.symbol || "--")}</strong>
          <small>${escapeHtml(row.provider_label || row.provider || "Configured")} · ${escapeHtml(row.venue_symbol || row.symbol || "")}</small>
        </span>
        <span class="backtest-symbol-badges">
          ${badges.map((badge) => `<em>${escapeHtml(badge)}</em>`).join("")}
        </span>
      </button>
    `;
  };

  const selectSymbol = (row, { quote = true } = {}) => {
    state.selected = row;
    if (refs.providerInput) refs.providerInput.value = row.provider || "global";
    if (refs.symbolInput) refs.symbolInput.value = row.symbol || "";
    if (refs.venueInput) refs.venueInput.value = row.venue_symbol || row.symbol || "";
    if (refs.selectedAsset) {
      refs.selectedAsset.innerHTML = `
        ${cryptoIconMarkup(row.symbol)}
        <span>
          <strong>${escapeHtml(row.symbol || "--")}</strong>
          <small>${escapeHtml(row.provider_label || row.provider || "Configured")} · ${escapeHtml(row.settlement_asset || "USDT")}</small>
        </span>
        <em>${row.max_leverage ? `${number(row.max_leverage).toFixed(0)}x max` : "Ready"}</em>
      `;
    }
    renderSymbols();
    if (quote) refreshQuote();
  };

  const refreshQuote = () => {
    if (!state.selected || !urls.quote) return;
    state.quoteAbort?.abort();
    state.quoteAbort = new AbortController();
    const url = new URL(apiUrl(urls.quote), window.location.origin);
    url.searchParams.set("provider", state.selected.provider || "global");
    url.searchParams.set("symbol", state.selected.symbol || "");
    url.searchParams.set("venue_symbol", state.selected.venue_symbol || state.selected.symbol || "");
    url.searchParams.set("allocation_usd", refs.allocation?.value || "0");
    if (refs.conversion) refs.conversion.classList.add("is-loading");
    fetch(url, { signal: state.quoteAbort.signal, headers: { Accept: "application/json" } })
      .then((response) => response.json())
      .then((payload) => {
        if (!payload?.ok || !refs.conversion) return;
        refs.conversion.innerHTML = `
          <span>Live conversion @ ${money.format(number(payload.mid))}</span>
          <strong>${escapeHtml(payload.asset_amount_formatted || "--")} ${escapeHtml(payload.symbol || "")}</strong>
        `;
      })
      .catch((error) => {
        if (error.name !== "AbortError" && refs.conversion) {
          refs.conversion.innerHTML = "<span>Live conversion</span><strong>Unavailable</strong>";
        }
      })
      .finally(() => refs.conversion?.classList.remove("is-loading"));
  };

  const renderPayload = (payload) => {
    if (!payload?.ok) return;
    state.payload = payload;
    refs.dashboard.hidden = false;
    if (refs.empty) refs.empty.hidden = true;
    renderSummary(payload.summary || {});
    renderMetrics(payload.metrics || {});
    renderAutopilot(payload.autopilot || {});
    renderExecution(payload.execution_quality || {});
    renderStrategies(payload.strategy_weights || []);
    renderSystemMetrics(payload.system_metrics || {});
    renderChart();
  };

  const renderSummary = (summary) => {
    const symbol = summary.symbol || "--";
    const timeframe = summary.timeframe || "LIVE";
    const provider = summary.provider_label || summary.provider || "Paper";
    const duration = summary.duration || "1H10";
    const title = document.querySelector("[data-backtest-title]");
    const chip = document.querySelector("[data-backtest-summary]");
    if (title) title.textContent = `${symbol} ${timeframe}`;
    if (chip) chip.textContent = `${provider} / ${duration}`;
  };

  const renderMetrics = (metrics) => {
    const roi = number(metrics.roi);
    const pnl = number(metrics.pnl);
    const drawdown = number(metrics.max_drawdown);
    setMetric("roi", percent.format(roi), roi);
    setMetric("pnl", money.format(pnl), pnl);
    setMetric("win_rate", percent.format(number(metrics.win_rate)));
    setMetric("max_drawdown", percent.format(drawdown), drawdown * -1);
    setMetric("trades", String(Math.round(number(metrics.trades))));
    setMetric("fees", money.format(number(metrics.fees)));
    setMetric("ending_balance", money.format(number(metrics.ending_balance)));
  };

  const setMetric = (key, text, polarity = null) => {
    const node = metricNodes[key];
    if (!node) return;
    node.textContent = text;
    if (polarity !== null) {
      node.classList.toggle("positive", polarity >= 0);
      node.classList.toggle("negative", polarity < 0);
    }
  };

  const renderAutopilot = (autopilot) => {
    if (refs.autopilotConfidence) refs.autopilotConfidence.textContent = percent.format(number(autopilot.confidence));
    if (refs.activeStrategies) refs.activeStrategies.textContent = `${autopilot.active_strategy_count || 0}/${autopilot.strategy_count || 0} active`;
    renderKeyValues(refs.autopilotList, [
      ["Status", autopilot.status || "optimized"],
      ["Regime", autopilot.market_regime || "--"],
      ["Objective", autopilot.objective || "--"],
      ["Model Stack", Array.isArray(autopilot.model_stack) ? autopilot.model_stack.length : 0],
    ]);
  };

  const renderExecution = (execution) => {
    if (refs.executionScore) refs.executionScore.textContent = percent.format(number(execution.fill_quality));
    renderKeyValues(refs.executionList, [
      ["Leverage", `${number(execution.auto_leverage, 1).toFixed(2)}x`],
      ["Max Venue", `${number(execution.max_exchange_leverage, 1).toFixed(0)}x`],
      ["Fees", `${number(execution.fee_bps).toFixed(2)} bps`],
      ["Slippage", `${number(execution.slippage_bps).toFixed(2)} bps`],
      ["Spread", `${number(execution.spread_bps).toFixed(2)} bps`],
      ["Liquidity", compactMoney.format(number(execution.liquidity_usd))],
    ]);
  };

  const renderStrategies = (rows) => {
    if (!refs.strategyList) return;
    if (!rows.length) {
      refs.strategyList.innerHTML = '<div class="backtest-symbol-empty">No strategy weights yet.</div>';
      return;
    }
    refs.strategyList.innerHTML = rows.map((row) => {
      const weight = number(row.weight);
      return `
        <div class="backtest-strategy-row ${row.enabled ? "is-enabled" : "is-disabled"}">
          <div>
            <strong>${escapeHtml(row.label || row.strategy_name || "--")}</strong>
            <small>${row.enabled ? `${percent.format(weight)} allocation weight` : escapeHtml(row.disabled_reason || "disabled")}</small>
          </div>
          <span>${percent.format(number(row.total_return))}</span>
          <i style="--weight:${Math.max(weight, 0.02)}"></i>
        </div>
      `;
    }).join("");
  };

  const renderSystemMetrics = (metrics) => {
    Object.entries(systemNodes).forEach(([key, node]) => {
      node.textContent = metrics[key] || node.textContent || "Auto";
    });
  };

  const renderKeyValues = (host, rows) => {
    if (!host) return;
    host.innerHTML = rows.map(([label, value]) => `
      <div>
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(String(value))}</strong>
      </div>
    `).join("");
  };

  const renderChart = async () => {
    const payload = state.payload || {};
    if (refs.chartTitle) {
      refs.chartTitle.textContent = state.chartMode === "market" ? "Market + AI Overlay" : chartLabel(state.chartMode);
    }
    if (!state.chart) {
      state.chart = new BacktestChart(refs.chartHost, refs.chartOverlay, urls.chartLib);
    }
    await state.chart.render(payload, state.chartMode);
  };

  const submitForm = async (event) => {
    event.preventDefault();
    if (!form) return;
    setLoading(true);
    setStatus("");
    try {
      const response = await fetch(apiUrl(form.action), {
        method: "POST",
        body: new FormData(form),
        headers: {
          Accept: "application/json",
          "X-Requested-With": "fetch",
        },
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) throw new Error(payload.error || "Vault simulation failed.");
      renderPayload(payload);
      setStatus("Vault simulation completed.", "success");
    } catch (error) {
      setStatus(error.message || "Vault simulation failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const syncAllocation = (source) => {
    const cap = number(form.dataset.allocationCap || form.dataset.paperBalance, 10000);
    let value = number(source.value, 0);
    value = Math.max(0, Math.min(value, cap));
    if (source !== refs.allocation && refs.allocation) refs.allocation.value = value ? String(value) : "";
    if (source !== refs.slider && refs.slider) refs.slider.value = String(value);
    window.clearTimeout(state.quoteTimer);
    state.quoteTimer = window.setTimeout(refreshQuote, 160);
  };

  refs.search?.addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(() => loadSymbols(), 180);
  });

  refs.maxButton?.addEventListener("click", () => {
    const cap = form.dataset.allocationCap || form.dataset.paperBalance || "10000";
    if (refs.allocation) refs.allocation.value = cap;
    if (refs.slider) refs.slider.value = cap;
    refreshQuote();
  });

  refs.allocation?.addEventListener("input", () => syncAllocation(refs.allocation));
  refs.slider?.addEventListener("input", () => syncAllocation(refs.slider));

  document.querySelectorAll("[data-timeframe-option]").forEach((button) => {
    button.addEventListener("click", () => {
      state.timeframe = button.dataset.timeframeOption || "live";
      if (refs.timeframeInput) refs.timeframeInput.value = state.timeframe;
      document.querySelectorAll("[data-timeframe-option]").forEach((item) => item.classList.toggle("is-active", item === button));
      button.animate?.([{ transform: "scale(0.98)" }, { transform: "scale(1)" }], { duration: 150, easing: "ease-out" });
    });
  });

  document.querySelectorAll("[data-chart-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      state.chartMode = button.dataset.chartMode || "market";
      document.querySelectorAll("[data-chart-mode]").forEach((item) => item.classList.toggle("is-active", item === button));
      renderChart();
    });
  });

  form.addEventListener("submit", submitForm);
  window.addEventListener("resize", () => state.chart?.drawOverlay?.());

  const initialSymbols = readJson(refs.initialSymbols);
  const initialPayload = readJson(refs.payload);
  const cached = cachedSymbols();
  if (cached?.symbols?.length) applySymbols(cached);
  else if (initialSymbols?.symbols?.length) applySymbols(initialSymbols, { cache: true });
  else loadSymbols();
  if (initialPayload?.ok) renderPayload(initialPayload);

  function symbolKey(row) {
    return `${row.provider || "global"}:${row.venue_symbol || row.symbol || ""}:${row.symbol || ""}`;
  }

  function chartLabel(mode) {
    return {
      equity: "Equity Curve",
      pnl: "Profit Curve",
      drawdown: "Drawdown Risk",
      market: "Market + AI Overlay",
    }[mode] || "Simulation";
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function cryptoIconMarkup(symbol) {
    const symbolKey = String(symbol || "GEN").toUpperCase();
    const classKey = symbolKey.toLowerCase().replace(/[^a-z0-9-]/g, "") || "gen";
    const fallback = escapeHtml(symbolKey.slice(0, 2));
    const paths = {
      BTC: '<path d="M10 5.2v13.6M14 5.2v13.6"></path><path d="M8.4 8h5.3c1.7 0 2.8.8 2.8 2s-1.1 2-2.8 2H8.4"></path><path d="M8.4 12h5.9c1.8 0 3 .9 3 2.2s-1.2 2.2-3 2.2H8.4"></path>',
      ETH: '<path d="m12 3.8-5 8.3 5 2.9 5-2.9-5-8.3Z"></path><path d="m7 13.3 5 6.9 5-6.9"></path><path d="m7 12.1 5-2.1 5 2.1"></path>',
      ALGO: '<path d="m7 17 5.4-10.5 4.6 10.5"></path><path d="M9.5 12.3h5.4"></path><path d="m14 7.6 3.2-1.2"></path>',
      USDT: '<circle cx="12" cy="12" r="7.4"></circle><path d="M8.4 9.2h7.2"></path><path d="M12 9.2v6.9"></path><path d="M8.9 11.4c1.9.8 4.3.8 6.2 0"></path>',
      USDC: '<circle cx="12" cy="12" r="7.4"></circle><path d="M8.4 9.2h7.2"></path><path d="M12 9.2v6.9"></path><path d="M15 7.2a5 5 0 1 0 0 9.6"></path>',
      SOL: '<path d="M7.2 7.2h9.6l-1.8 2H5.4l1.8-2Z"></path><path d="M8.9 11h9.4l-1.8 2H7.1l1.8-2Z"></path><path d="M7.5 14.8h9.6l-1.8 2H5.7l1.8-2Z"></path>',
      XRP: '<path d="M6.2 7.2c1.8 2 3.7 3 5.8 3s4-1 5.8-3"></path><path d="M6.2 16.8c1.8-2 3.7-3 5.8-3s4 1 5.8 3"></path>',
    };
    const mark = paths[symbolKey] || `<circle cx="12" cy="12" r="7"></circle><text x="12" y="14.7" text-anchor="middle">${fallback}</text>`;
    return `<span class="coin-icon backtest-token-icon coin-icon-${classKey}" aria-hidden="true" data-asset-symbol="${escapeHtml(symbolKey)}"><svg viewBox="0 0 24 24" focusable="false">${mark}</svg></span>`;
  }

  class BacktestChart {
    constructor(host, overlay, librarySrc) {
      this.host = host;
      this.overlay = overlay;
      this.librarySrc = librarySrc;
      this.chart = null;
      this.candleSeries = null;
      this.lineSeries = null;
      this.payload = null;
      this.mode = "market";
      this.libraryPromise = null;
    }

    async render(payload, mode) {
      this.payload = payload || {};
      this.mode = mode || "market";
      try {
        await this.ensureChart();
        this.applyData();
      } catch (error) {
        // Canvas overlay remains as the fallback rendering path.
      }
      this.drawOverlay();
    }

    async ensureChart() {
      if (!this.host) return false;
      const library = await this.loadLibrary();
      if (!library?.createChart) return false;
      if (this.chart) return true;
      this.chart = library.createChart(this.host, {
        autoSize: true,
        layout: { background: { color: "#05070b" }, textColor: "#8f9bae", fontFamily: "Inter, system-ui, sans-serif" },
        grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.055)" } },
        rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
        timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false },
        crosshair: { mode: 0 },
        kineticScroll: { touch: true, mouse: true },
        handleScale: { pinch: true, mouseWheel: true, axisPressedMouseMove: { time: true, price: true } },
        handleScroll: { horzTouchDrag: true, vertTouchDrag: false, mouseWheel: true, pressedMouseMove: true },
      });
      this.candleSeries = this.chart.addCandlestickSeries({
        upColor: "#0ecb81",
        downColor: "#f6465d",
        borderVisible: false,
        wickUpColor: "#0ecb81",
        wickDownColor: "#f6465d",
      });
      this.lineSeries = this.chart.addLineSeries({
        color: "#f0b90b",
        lineWidth: 2,
        priceLineVisible: false,
      });
      return true;
    }

    loadLibrary() {
      if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts);
      if (this.libraryPromise) return this.libraryPromise;
      this.libraryPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = this.librarySrc;
        script.async = true;
        script.onload = () => resolve(window.LightweightCharts);
        script.onerror = reject;
        document.head.append(script);
      });
      return this.libraryPromise;
    }

    applyData() {
      if (!this.chart || !this.candleSeries || !this.lineSeries) return;
      if (this.mode === "market") {
        const candles = Array.isArray(this.payload?.charts?.candles) ? this.payload.charts.candles.slice(-150) : [];
        this.candleSeries.setData(candles.map((row, index) => ({
          time: this.timeValue(row.time, index),
          open: number(row.open),
          high: number(row.high),
          low: number(row.low),
          close: number(row.close),
        })));
        this.lineSeries.setData([]);
      } else {
        const series = Array.isArray(this.payload?.charts?.[this.mode]) ? this.payload.charts[this.mode] : [];
        this.candleSeries.setData([]);
        this.lineSeries.applyOptions({ color: this.mode === "drawdown" ? "#f6465d" : this.mode === "pnl" ? "#f0b90b" : "#0ecb81" });
        this.lineSeries.setData(series.map((row, index) => ({ time: this.timeValue(row.x, index), value: number(row.y) })));
      }
      this.chart.timeScale?.().fitContent?.();
    }

    drawOverlay() {
      const canvas = this.overlay;
      if (!canvas) return;
      this.resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;
      ctx.clearRect(0, 0, width, height);
      if (this.mode !== "market") {
        this.fallbackLine(ctx, width, height);
        return;
      }
      const candles = Array.isArray(this.payload?.charts?.candles) ? this.payload.charts.candles.slice(-150) : [];
      const overlays = this.payload?.overlays || {};
      if (!candles.length) {
        this.empty(ctx, width, height);
        return;
      }
      if (!this.chart) this.fallbackCandles(ctx, candles, width, height);
      this.confidence(ctx, overlays, candles, width, height);
      this.path(ctx, overlays, candles, width, height);
      this.zones(ctx, overlays, candles, width, height);
      this.fibonacci(ctx, overlays, width, height);
    }

    fallbackLine(ctx, width, height) {
      const series = Array.isArray(this.payload?.charts?.[this.mode]) ? this.payload.charts[this.mode] : [];
      if (!series.length || this.chart) return;
      const values = series.map((row) => number(row.y));
      const min = Math.min(...values);
      const max = Math.max(...values);
      ctx.save();
      ctx.strokeStyle = this.mode === "drawdown" ? "rgba(246,70,93,0.9)" : "rgba(240,185,11,0.92)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.forEach((row, index) => {
        const x = (index / Math.max(series.length - 1, 1)) * width;
        const y = height - ((number(row.y) - min) / Math.max(max - min, 1e-9)) * height;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    fallbackCandles(ctx, candles, width, height) {
      ctx.save();
      ctx.strokeStyle = "rgba(14,203,129,0.78)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      candles.forEach((row, index) => {
        const x = (index / Math.max(candles.length - 1, 1)) * width;
        const y = this.priceToY(number(row.close), candles, height);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    zones(ctx, overlays, candles, width, height) {
      const zones = overlays.zones || {};
      [
        ["entry", zones.entry?.price, "rgba(240,185,11,0.8)"],
        ["exit", zones.exit?.price, "rgba(14,203,129,0.76)"],
        ["stop_loss", zones.stop_loss?.price, "rgba(246,70,93,0.76)"],
      ].forEach(([name, price, color]) => {
        const value = number(price);
        if (value <= 0) return;
        const y = this.priceToY(value, candles, height);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.setLineDash(name === "entry" ? [] : [5, 5]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
        ctx.restore();
      });
    }

    path(ctx, overlays, candles, width, height) {
      const path = Array.isArray(overlays.path) ? overlays.path : [];
      if (!path.length) return;
      ctx.save();
      ctx.strokeStyle = "rgba(240,185,11,0.96)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      path.forEach((point, index) => {
        const x = width * (0.62 + (index / Math.max(path.length - 1, 1)) * 0.34);
        const y = this.priceToY(number(point.value), candles, height, path);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    confidence(ctx, overlays, candles, width, height) {
      const upper = overlays.confidence_band?.upper;
      const lower = overlays.confidence_band?.lower;
      if (!Array.isArray(upper) || !Array.isArray(lower) || !upper.length || upper.length !== lower.length) return;
      ctx.save();
      ctx.fillStyle = "rgba(240,185,11,0.08)";
      ctx.beginPath();
      upper.forEach((point, index) => {
        const x = width * (0.62 + (index / Math.max(upper.length - 1, 1)) * 0.34);
        const y = this.priceToY(number(point.value), candles, height, upper.concat(lower));
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      [...lower].reverse().forEach((point, index) => {
        const x = width * (0.96 - (index / Math.max(lower.length - 1, 1)) * 0.34);
        ctx.lineTo(x, this.priceToY(number(point.value), candles, height, upper.concat(lower)));
      });
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    fibonacci(ctx, overlays, width, height) {
      const zones = Array.isArray(overlays.fibonacci_time_zones) ? overlays.fibonacci_time_zones : [];
      if (!zones.length) return;
      ctx.save();
      ctx.strokeStyle = "rgba(124,167,255,0.2)";
      ctx.lineWidth = 1;
      zones.slice(0, 6).forEach((zone, index) => {
        const x = width * (0.62 + (index / Math.max(zones.length - 1, 1)) * 0.34);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      });
      ctx.restore();
    }

    priceToY(value, candles, height, extras = []) {
      if (this.chart && this.candleSeries) {
        const coord = this.candleSeries.priceToCoordinate(value);
        if (Number.isFinite(coord)) return coord;
      }
      const values = candles.flatMap((row) => [number(row.high), number(row.low), number(row.close)]);
      extras.forEach((row) => values.push(number(row.value, number(row.upper, number(row.lower)))));
      const filtered = values.filter((item) => item > 0);
      const min = Math.min(...filtered);
      const max = Math.max(...filtered);
      return height - ((value - min) / Math.max(max - min, 1e-9)) * height;
    }

    resizeCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }

    empty(ctx, width, height) {
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.52)";
      ctx.font = "700 12px Inter, system-ui, sans-serif";
      ctx.fillText("Awaiting market data", 16, height / 2);
      ctx.restore();
    }

    timeValue(value, index) {
      const parsed = number(value, 0);
      if (parsed > 10000000000) return Math.floor(parsed / 1000);
      if (parsed > 0) return Math.floor(parsed);
      return Math.floor(Date.now() / 1000) - (150 - index) * 60;
    }
  }
})();
