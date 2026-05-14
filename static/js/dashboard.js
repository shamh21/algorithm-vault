(() => {
  const MAX_ROWS = 150;
  const PAGE_SIZE = 30;
  const ROW_HEIGHT = 62;
  const ACTIVITY_ROW_HEIGHT = 58;
  const POLL_FOREGROUND_MS = 15000;
  const POLL_BACKGROUND_MS = 60000;

  const root = document.querySelector("[data-dashboard]");
  if (!root) return;

  const refs = {
    list: root.querySelector("[data-opportunity-list]"),
    status: root.querySelector("[data-stream-status]"),
    chartPanel: root.querySelector("[data-chart-panel]"),
    chartHost: root.querySelector("[data-lightweight-chart]"),
    overlay: root.querySelector("[data-forecast-overlay]"),
    chartTitle: root.querySelector("[data-chart-title]"),
    chartProvider: root.querySelector("[data-chart-provider]"),
    chartLoading: root.querySelector("[data-chart-loading]"),
    timeframeTabs: Array.from(root.querySelectorAll("[data-timeframe]")),
    refreshButtons: Array.from(root.querySelectorAll("[data-refresh-opportunities]")),
    filterButtons: Array.from(root.querySelectorAll("[data-opportunity-filter]")),
    quickActions: Array.from(root.querySelectorAll("[data-quick-action]")),
    activity: root.querySelector("[data-activity-feed]"),
    neutral: root.querySelector("[data-opportunity-neutral]"),
    setupHelper: root.querySelector("[data-setup-helper]"),
    vaultPulse: root.querySelector("[data-vault-pulse]"),
    vaultWalletValue: root.querySelector("[data-vault-wallet-value]"),
    vaultWalletStatus: root.querySelector("[data-vault-wallet-status]"),
    vaultPerformanceValue: root.querySelector("[data-vault-performance-value]"),
    vaultPerformanceSparkline: root.querySelector("[data-vault-performance-sparkline]"),
    vaultPerformancePath: root.querySelector("[data-vault-performance-path]"),
    vaultPerformanceEmpty: root.querySelector("[data-vault-performance-empty]"),
    vaultPulseError: root.querySelector("[data-vault-pulse-error]"),
    sheet: root.querySelector("[data-preview-sheet]"),
    sheetTitle: root.querySelector("[data-sheet-title]"),
    sheetBody: root.querySelector("[data-sheet-body]"),
    sheetAction: root.querySelector("[data-sheet-action]"),
    sheetClose: root.querySelector("[data-close-preview]"),
    sheetBackdrop: root.querySelector("[data-preview-backdrop]"),
    quickSymbol: root.querySelector("[data-quick-symbol]"),
    forecastDirection: root.querySelector("[data-forecast-direction]"),
    forecastRoi: root.querySelector("[data-forecast-roi]"),
    forecastConfidence: root.querySelector("[data-forecast-confidence]"),
    forecastRisk: root.querySelector("[data-forecast-risk]"),
    intel: {
      pair: root.querySelector("[data-intel-pair]"),
      provider: root.querySelector("[data-intel-provider]"),
      entry: root.querySelector("[data-intel-entry]"),
      exit: root.querySelector("[data-intel-exit]"),
      stop: root.querySelector("[data-intel-stop]"),
      liquidity: root.querySelector("[data-intel-liquidity]"),
      slippage: root.querySelector("[data-intel-slippage]"),
      ml: root.querySelector("[data-intel-ml]"),
      fib: root.querySelector("[data-intel-fib]"),
    },
  };

  const urls = {
    opportunities: root.dataset.opportunitiesUrl,
    chart: root.dataset.chartUrl,
    stream: root.dataset.streamUrl,
    activity: root.dataset.activityUrl,
    dashboardData: root.dataset.dashboardDataUrl,
    chartModule: root.dataset.chartModuleSrc,
    chartLib: root.dataset.chartLibSrc,
  };
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;
  const defaultSheetHref = refs.sheetAction?.href || "";
  const initialPayload = (() => {
    const node = document.getElementById("dashboard-initial-payload");
    if (!node?.textContent) return {};
    try {
      return JSON.parse(node.textContent) || {};
    } catch (error) {
      return {};
    }
  })();

  const state = {
    opportunities: [],
    activity: [],
    active: null,
    filter: "all",
    timeframe: "live",
    chart: null,
    chartPayload: null,
    chartInView: false,
    chartModulePromise: null,
    eventSource: null,
    pollTimer: null,
    reconnectTimer: null,
    reconnectAttempt: 0,
    destroyed: false,
    longPressTimer: 0,
    raf: {},
    requests: {
      opportunities: { id: 0, controller: null },
      chart: { id: 0, controller: null },
      activity: { id: 0, controller: null },
      dashboard: { id: 0, controller: null },
    },
  };

  const number = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const hasNumber = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const dash = "—";
  const percent = (value, digits = 0) => (hasNumber(value) ? `${(Number(value) * 100).toFixed(digits)}%` : dash);
  const roi = (value) => (hasNumber(value) ? `${Number(value).toFixed(2)}%` : dash);
  const currency = (value) => number(value).toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
  const price = (value) => {
    if (!hasNumber(value)) return dash;
    const parsed = Number(value);
    if (!parsed) return dash;
    if (Math.abs(parsed) >= 1000) return parsed.toLocaleString(undefined, { maximumFractionDigits: 2 });
    if (Math.abs(parsed) >= 1) return parsed.toFixed(4);
    return parsed.toPrecision(5);
  };
  const label = (value) => String(value || dash).replace(/_/g, " ");
  const isHold = (row) => String(row?.direction || row?.action || "").toLowerCase() === "hold";
  const hasExecutableSetup = (row) => {
    if (!row || isHold(row)) return false;
    const prices = [row.entry, row.exit, row.stop_loss];
    if (!prices.every(hasNumber)) return false;
    if (prices.every((value) => Number(value) === 1)) return false;
    return true;
  };
  const fieldValue = (row, key, formatter) => (hasExecutableSetup(row) || !["entry", "exit", "stop_loss"].includes(key) ? formatter(row?.[key]) : dash);
  const formatTime = (value) => {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };
  const summarizeDetail = (detail = "", title = "") => {
    const text = String(detail || "").trim();
    const titleText = String(title || "").trim();
    if (!text) return "";
    const provider = /kucoin/i.test(text + titleText) ? "KuCoin" : /hyperliquid/i.test(text + titleText) ? "Hyperliquid" : "Provider";
    if (/connection|timeout|unavailable|request failed|invalid request ip|network/i.test(text)) return `${provider} unavailable`;
    if (text.length > 120) return `${text.slice(0, 117)}…`;
    return text;
  };

  const schedule = (key, fn) => {
    if (state.raf[key]) return;
    state.raf[key] = requestAnimationFrame(() => {
      state.raf[key] = 0;
      fn();
    });
  };

  const setStatus = (text, tone = "") => {
    if (!refs.status) return;
    refs.status.classList.toggle("is-live", tone === "live");
    refs.status.classList.toggle("is-stale", tone === "stale");
    const copy = refs.status.querySelector("span:last-child");
    if (copy) copy.textContent = text;
  };

  const requestJson = async (key, url) => {
    const slot = state.requests[key];
    slot.controller?.abort();
    slot.controller = "AbortController" in window ? new AbortController() : null;
    const id = ++slot.id;
    const response = await fetch(apiUrl(url), {
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: slot.controller?.signal,
    });
    if (id !== slot.id) return null;
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  };

  const setOpportunities = (rows) => {
    state.opportunities = (Array.isArray(rows) ? rows.filter(Boolean) : []).slice(0, MAX_ROWS);
    if (!state.active && state.opportunities.length) {
      state.active = state.opportunities[0];
      updateSelectionText(state.active);
      if (state.chartInView) fetchChart(state.active);
    }
    refs.neutral?.toggleAttribute("hidden", !state.opportunities.length || state.opportunities.some((row) => hasExecutableSetup(row)));
    schedule("opportunities", renderOpportunities);
  };

  const setActivity = (rows) => {
    state.activity = (Array.isArray(rows) ? rows.filter(Boolean) : []).slice(0, MAX_ROWS);
    schedule("activity", renderActivity);
  };

  const renderVirtual = (container, rows, rowHeight, renderRow, emptyText) => {
    if (!container) return;
    if (!rows.length) {
      container.replaceChildren(Object.assign(document.createElement("div"), { className: "activity-empty", textContent: emptyText }));
      return;
    }
    const viewport = container.clientHeight || 360;
    const start = Math.max(0, Math.floor(container.scrollTop / rowHeight) - 4);
    const count = Math.ceil(viewport / rowHeight) + 10;
    const end = Math.min(rows.length, start + count);
    const fragment = document.createDocumentFragment();
    const top = document.createElement("div");
    top.style.height = `${start * rowHeight}px`;
    fragment.append(top);
    rows.slice(start, end).forEach((row, localIndex) => fragment.append(renderRow(row, start + localIndex)));
    const bottom = document.createElement("div");
    bottom.style.height = `${Math.max(0, rows.length - end) * rowHeight}px`;
    fragment.append(bottom);
    container.replaceChildren(fragment);
  };

  const filteredOpportunities = () => {
    if (state.filter === "all") return state.opportunities;
    return state.opportunities.filter((row) => String(row.direction || row.action || "").toLowerCase() === state.filter);
  };

  const renderOpportunities = () => {
    const rows = filteredOpportunities();
    renderVirtual(refs.list, rows, ROW_HEIGHT, (row, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "opportunity-card";
      if (state.active && row.provider === state.active.provider && row.symbol === state.active.symbol) {
        button.classList.add("is-active");
      }
      button.dataset.provider = row.provider || "";
      button.dataset.symbol = row.symbol || "";
      button.dataset.venueSymbol = row.venue_symbol || row.symbol || "";
      button.setAttribute("role", "listitem");

      const rank = document.createElement("span");
      rank.className = "opportunity-rank";
      rank.textContent = `#${index + 1}`;

      const main = document.createElement("span");
      main.className = "opportunity-main";
      const title = document.createElement("strong");
      title.textContent = row.symbol || "--";
      const sub = document.createElement("small");
      sub.textContent = `${label(row.provider).toUpperCase()} · ${label(row.direction).toUpperCase()}`;
      const conf = document.createElement("small");
      conf.textContent = `Confidence ${percent(row.confidence, 0)}`;
      main.append(title, sub, conf);

      const meta = document.createElement("span");
      meta.className = "opportunity-score";
      meta.innerHTML = `<small>Score</small>${hasNumber(row.score) ? number(row.score).toFixed(2) : dash}`;

      const roiNode = document.createElement("span");
      roiNode.className = number(row.predicted_roi) >= 0 ? "opportunity-roi positive" : "opportunity-roi negative";
      roiNode.innerHTML = `<small>ROI</small>${roi(row.predicted_roi)}`;

      button.append(rank, main, meta, roiNode);
      button.addEventListener("click", () => selectOpportunity(row));
      button.addEventListener("touchstart", () => startLongPress(row), { passive: true });
      button.addEventListener("touchend", cancelLongPress, { passive: true });
      button.addEventListener("touchmove", cancelLongPress, { passive: true });
      return button;
    }, state.filter === "all" ? "Waiting for ranked markets." : `No ${state.filter.toUpperCase()} setups in the current scanner window.`);
  };


  const getBalanceAmount = (balances) => {
    const rows = Array.isArray(balances) ? balances : [];
    return rows.reduce((total, row) => {
      const estimated = Number(row?.estimated_usd_value);
      if (Number.isFinite(estimated) && estimated > 0) return total + estimated;
      const asset = String(row?.asset || "").toUpperCase();
      const value = Number(row?.value ?? row?.total_balance ?? row?.available_balance);
      if (!Number.isFinite(value)) return total;
      return total + (["USD", "USDC", "USDT"].includes(asset) ? value : 0);
    }, 0);
  };

  const performancePoints = (curve) => (Array.isArray(curve) ? curve : [])
    .map((point, index) => ({
      x: number(point?.timestamp ?? point?.time ?? point?.x ?? index, index),
      y: number(point?.equity ?? point?.balance ?? point?.value ?? point?.y, NaN),
    }))
    .filter((point) => Number.isFinite(point.y));

  const formatSyncStatus = (syncedAt, hasBalance, loading = false) => {
    if (loading) return hasBalance ? "Refreshing account data…" : "Syncing account data…";
    if (!hasBalance) return "Connect or sync an account to view balance.";
    if (!syncedAt) return "Updated just now";
    const date = new Date(syncedAt);
    if (Number.isNaN(date.getTime())) return "Updated just now";
    const deltaSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
    if (deltaSeconds < 60) return "Updated just now";
    return `Last synced ${date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}`;
  };

  const renderSparkline = (points, changePercent) => {
    if (!refs.vaultPerformanceSparkline || !refs.vaultPerformancePath) return;
    if (points.length < 2) {
      refs.vaultPerformancePath.setAttribute("d", "");
      refs.vaultPerformanceSparkline.setAttribute("aria-label", "Past account performance trend unavailable");
      return;
    }
    const width = 120;
    const height = 38;
    const pad = 3;
    const values = points.map((point) => point.y);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const step = (width - pad * 2) / Math.max(points.length - 1, 1);
    const path = points.map((point, index) => {
      const x = pad + step * index;
      const y = height - pad - ((point.y - min) / span) * (height - pad * 2);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    }).join(" ");
    refs.vaultPerformancePath.setAttribute("d", path);
    const direction = changePercent > 0 ? "up" : changePercent < 0 ? "down" : "flat";
    refs.vaultPerformanceSparkline.setAttribute(
      "aria-label",
      `Past account performance trend ${direction} ${Math.abs(changePercent).toFixed(2)} percent over available history`,
    );
  };

  const renderVaultPulse = (payload = {}, { loading = false, error = false } = {}) => {
    if (!refs.vaultPulse) return;
    const balances = Array.isArray(payload.balances) ? payload.balances : [];
    const balance = getBalanceAmount(balances);
    const hasBalance = balances.length > 0 && balance > 0;
    refs.vaultPulse.setAttribute("aria-busy", loading ? "true" : "false");
    refs.vaultPulse.classList.toggle("is-loading", loading);
    refs.vaultPulse.classList.toggle("has-error", error);
    if (refs.vaultPulseError) refs.vaultPulseError.hidden = !error;

    if (refs.vaultWalletValue) {
      refs.vaultWalletValue.classList.toggle("dashboard-skeleton", loading && !hasBalance);
      refs.vaultWalletValue.textContent = hasBalance ? currency(balance) : loading ? "Loading balance" : "Wallet balance unavailable";
    }
    if (refs.vaultWalletStatus) {
      refs.vaultWalletStatus.textContent = formatSyncStatus(payload.account_synced_at, hasBalance, loading);
    }

    const points = performancePoints(payload.paper_equity_curve);
    const hasPerformance = points.length >= 2;
    const first = hasPerformance ? points[0].y : 0;
    const last = hasPerformance ? points[points.length - 1].y : 0;
    const change = last - first;
    const changePercent = first ? (change / Math.abs(first)) * 100 : 0;

    if (refs.vaultPerformanceValue) {
      refs.vaultPerformanceValue.classList.toggle("dashboard-skeleton", loading && !hasPerformance);
      refs.vaultPerformanceValue.classList.toggle("positive", hasPerformance && change > 0);
      refs.vaultPerformanceValue.classList.toggle("negative", hasPerformance && change < 0);
      refs.vaultPerformanceValue.textContent = hasPerformance
        ? `${change >= 0 ? "+" : "−"}${Math.abs(changePercent).toFixed(2)}% (${change >= 0 ? "+" : "−"}${currency(Math.abs(change))}) historical`
        : loading ? "Loading history" : "No history yet";
      refs.vaultPerformanceValue.setAttribute(
        "aria-label",
        hasPerformance
          ? `Past account performance ${change >= 0 ? "up" : "down"} ${Math.abs(changePercent).toFixed(2)} percent, ${currency(Math.abs(change))} historical change`
          : "Performance history will appear once account data is available.",
      );
    }
    if (refs.vaultPerformanceEmpty) refs.vaultPerformanceEmpty.hidden = hasPerformance || loading;
    renderSparkline(hasPerformance ? points : [], changePercent);
  };

  const fetchVaultPulse = async () => {
    renderVaultPulse(initialPayload, { loading: true });
    if (!urls.dashboardData) {
      renderVaultPulse(initialPayload, { loading: false });
      return;
    }
    try {
      const payload = await requestJson("dashboard", urls.dashboardData);
      if (payload) renderVaultPulse(payload, { loading: false });
    } catch (error) {
      if (error.name !== "AbortError") renderVaultPulse(initialPayload, { loading: false, error: true });
    }
  };

  const renderActivity = () => {
    renderVirtual(refs.activity, state.activity, ACTIVITY_ROW_HEIGHT, (row) => {
      const item = document.createElement("article");
      item.className = `activity-row activity-${label(row.severity).toLowerCase()}`;
      item.setAttribute("role", "listitem");
      const marker = document.createElement("span");
      marker.className = "activity-marker";
      const copy = document.createElement("span");
      copy.className = "activity-copy";
      const title = document.createElement("strong");
      title.textContent = row.title || label(row.kind);
      const meta = document.createElement("small");
      const severity = label(row.severity || "info");
      const badge = document.createElement("span");
      badge.className = "activity-badge";
      badge.textContent = severity;
      const timestamp = document.createElement("time");
      timestamp.textContent = formatTime(row.created_at);
      meta.append(badge, timestamp);
      const detail = document.createElement("small");
      const rawDetail = row.detail || "";
      detail.textContent = summarizeDetail(rawDetail, row.title);
      copy.append(title, meta, detail);
      if (String(rawDetail).length > 120 || /exception|traceback|request failed|invalid request ip/i.test(String(rawDetail))) {
        const details = document.createElement("details");
        details.className = "activity-technical-details";
        const summary = document.createElement("summary");
        summary.textContent = "View technical details";
        const pre = document.createElement("pre");
        pre.textContent = String(rawDetail);
        details.append(summary, pre);
        copy.append(details);
      }
      item.append(marker, copy);
      return item;
    }, "Waiting for live activity.");
  };

  const fetchOpportunities = async (refresh = false) => {
    if (!urls.opportunities) return;
    const url = new URL(urls.opportunities, window.location.origin);
    url.searchParams.set("limit", String(PAGE_SIZE));
    if (refresh) url.searchParams.set("refresh", "1");
    try {
      const payload = await requestJson("opportunities", url);
      if (!payload) return;
      setOpportunities(payload.opportunities || []);
      setStatus(payload.diagnostics?.stale ? "Cached" : "Live", payload.diagnostics?.stale ? "stale" : "live");
    } catch (error) {
      if (error.name !== "AbortError") setStatus("Stale", "stale");
    }
  };

  const fetchActivity = async () => {
    if (!urls.activity) return;
    const url = new URL(urls.activity, window.location.origin);
    url.searchParams.set("limit", String(PAGE_SIZE));
    try {
      const payload = await requestJson("activity", url);
      if (payload) setActivity(payload.items || []);
    } catch (error) {
      if (error.name !== "AbortError") setActivity(state.activity);
    }
  };

  const fetchChart = async (row) => {
    if (!row || !urls.chart || !state.chartInView) return;
    refs.chartLoading?.removeAttribute("hidden");
    const url = new URL(urls.chart, window.location.origin);
    url.searchParams.set("provider", row.provider || "");
    url.searchParams.set("symbol", row.symbol || "");
    url.searchParams.set("venue_symbol", row.venue_symbol || row.symbol || "");
    url.searchParams.set("timeframe", state.timeframe);
    try {
      const payload = await requestJson("chart", url);
      if (!payload) return;
      state.chartPayload = payload;
      await renderChart(payload);
    } catch (error) {
      if (error.name !== "AbortError") await renderChart(state.chartPayload || {});
    } finally {
      refs.chartLoading?.setAttribute("hidden", "hidden");
    }
  };

  const loadChartModule = () => {
    if (window.AlgorithmVaultDashboardChart) return Promise.resolve(window.AlgorithmVaultDashboardChart);
    if (state.chartModulePromise) return state.chartModulePromise;
    state.chartModulePromise = new Promise((resolve, reject) => {
      if (!urls.chartModule) {
        reject(new Error("chart module url missing"));
        return;
      }
      const script = document.createElement("script");
      script.src = urls.chartModule;
      script.async = true;
      script.onload = () => resolve(window.AlgorithmVaultDashboardChart);
      script.onerror = reject;
      document.head.append(script);
    });
    return state.chartModulePromise;
  };

  const renderChart = async (payload) => {
    const Chart = await loadChartModule();
    if (!state.chart) {
      state.chart = new Chart({ host: refs.chartHost, overlay: refs.overlay, librarySrc: urls.chartLib });
    }
    await state.chart.render(payload || {});
  };

  const selectOpportunity = (row) => {
    state.active = row;
    updateSelectionText(row);
    navigator.vibrate?.(8);
    schedule("opportunities", renderOpportunities);
    fetchChart(row);
  };

  const updateSelectionText = (row) => {
    if (!row) return;
    if (refs.chartTitle) refs.chartTitle.textContent = `${row.symbol || "--"} Projection`;
    if (refs.chartProvider) refs.chartProvider.textContent = label(row.provider).toUpperCase();
    if (refs.quickSymbol) refs.quickSymbol.textContent = row.symbol || "Best Setup";
    updateIntelligence(row);
  };

  const updateIntelligence = (row) => {
    if (!row) return;
    const executable = hasExecutableSetup(row);
    const mapping = {
      pair: row.symbol || dash,
      provider: label(row.provider).toUpperCase(),
      entry: fieldValue(row, "entry", price),
      exit: fieldValue(row, "exit", price),
      stop: fieldValue(row, "stop_loss", price),
      liquidity: percent(row.liquidity_score, 0),
      slippage: hasNumber(row.slippage_bps) ? `${number(row.slippage_bps).toFixed(2)} bps` : dash,
      ml: percent(row.ml_model_agreement, 0),
      fib: percent(row.fibonacci_alignment, 0),
    };
    Object.entries(mapping).forEach(([key, value]) => {
      if (refs.intel[key]) refs.intel[key].textContent = value;
    });
    if (refs.forecastDirection) refs.forecastDirection.textContent = executable ? label(row.direction).toUpperCase() : "No active setup";
    if (refs.forecastRoi) refs.forecastRoi.textContent = executable ? roi(row.predicted_roi) : dash;
    if (refs.forecastConfidence) refs.forecastConfidence.textContent = executable ? percent(row.confidence, 0) : dash;
    if (refs.forecastRisk) refs.forecastRisk.textContent = executable && hasNumber(row.risk_reward) ? number(row.risk_reward).toFixed(2) : dash;
    refs.setupHelper?.toggleAttribute("hidden", executable);
  };

  const startLongPress = (row) => {
    cancelLongPress();
    state.longPressTimer = window.setTimeout(() => {
      state.active = row;
      updateSelectionText(row);
      openPreview("setup");
    }, 420);
  };

  const cancelLongPress = () => {
    if (state.longPressTimer) window.clearTimeout(state.longPressTimer);
    state.longPressTimer = 0;
  };

  const previewCopy = (action) => {
    const row = state.active || state.opportunities[0] || {};
    const setup = [
      ["Pair", row.symbol || "--"],
      ["Exchange", label(row.provider).toUpperCase()],
      ["Predicted ROI", hasExecutableSetup(row) ? roi(row.predicted_roi) : dash],
      ["Confidence", hasExecutableSetup(row) ? percent(row.confidence, 0) : dash],
      ["Entry", fieldValue(row, "entry", price)],
      ["Exit", fieldValue(row, "exit", price)],
      ["Stop", fieldValue(row, "stop_loss", price)],
      ["Risk/Reward", hasExecutableSetup(row) && hasNumber(row.risk_reward) ? number(row.risk_reward).toFixed(2) : dash],
    ];
    const actions = {
      allocate: { title: "Allocate Capital", action: "Open Vault Flow", rows: setup },
      cycle: { title: "Start Cycle Preview", action: "Open Vault Flow", rows: setup },
      panic: { title: "Emergency Stop Preview", action: "Open Panic Controls", rows: [["Status", "Preview only"], ["Effect", "Opens guarded panic controls"], ["Direct Submit", "No"]] },
      risk: { title: "Risk Mode Preview", action: "Open Risk Controls", rows: [["Status", "Preview only"], ["Direct Submit", "No"]] },
      setup: { title: `${row.symbol || "--"} · ${label(row.direction).toUpperCase()}`, action: "Open Vault Flow", rows: setup },
    };
    return actions[action] || actions.setup;
  };

  const openPreview = (action = "setup", href = "") => {
    if (!refs.sheet || !refs.sheetBody) return;
    const copy = previewCopy(action);
    refs.sheetTitle.textContent = copy.title;
    refs.sheetBody.replaceChildren();
    copy.rows.forEach(([term, value]) => {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = value;
      wrapper.append(dt, dd);
      refs.sheetBody.append(wrapper);
    });
    if (refs.sheetAction) {
      refs.sheetAction.textContent = copy.action;
      refs.sheetAction.href = href || defaultSheetHref;
    }
    refs.sheet.setAttribute("aria-hidden", "false");
    refs.sheet.classList.add("is-open");
    refs.sheetBackdrop?.setAttribute("aria-hidden", "false");
    refs.sheetBackdrop?.classList.add("is-open");
  };

  const closePreview = () => {
    refs.sheet?.setAttribute("aria-hidden", "true");
    refs.sheet?.classList.remove("is-open");
    refs.sheetBackdrop?.setAttribute("aria-hidden", "true");
    refs.sheetBackdrop?.classList.remove("is-open");
  };

  const toggleFullscreenChart = () => {
    root.classList.toggle("chart-fullscreen");
    document.body.classList.toggle("dashboard-chart-fullscreen", root.classList.contains("chart-fullscreen"));
    schedule("chart-resize", () => state.chart?.resize?.());
  };

  const stopPolling = () => {
    if (state.pollTimer) window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  };

  const startPolling = () => {
    stopPolling();
    const tick = () => {
      fetchOpportunities(false);
      fetchActivity();
    };
    state.pollTimer = window.setInterval(tick, document.hidden ? POLL_BACKGROUND_MS : POLL_FOREGROUND_MS);
  };

  const closeStream = () => {
    state.eventSource?.close();
    state.eventSource = null;
  };

  const scheduleReconnect = () => {
    if (state.destroyed || document.hidden) return;
    closeStream();
    startPolling();
    window.clearTimeout(state.reconnectTimer);
    const delay = Math.min(30000, 1000 * (2 ** Math.min(state.reconnectAttempt, 5))) + Math.floor(Math.random() * 650);
    state.reconnectAttempt += 1;
    state.reconnectTimer = window.setTimeout(connectStream, delay);
  };

  const connectStream = () => {
    if (state.destroyed || document.hidden) return;
    if (!window.EventSource || !urls.stream) {
      startPolling();
      return;
    }
    closeStream();
    try {
      const url = apiUrl(urls.stream);
      state.eventSource = new EventSource(url);
      state.eventSource.onopen = () => {
        state.reconnectAttempt = 0;
        stopPolling();
        setStatus("Live", "live");
      };
      state.eventSource.onerror = () => {
        setStatus("Polling", "stale");
        scheduleReconnect();
      };
      state.eventSource.addEventListener("opportunities", (event) => {
        const payload = JSON.parse(event.data || "{}");
        setOpportunities(payload.opportunities || []);
      });
      state.eventSource.addEventListener("activity", (event) => {
        const payload = JSON.parse(event.data || "{}");
        setActivity(payload.items || []);
      });
      state.eventSource.addEventListener("chart_delta", (event) => {
        const payload = JSON.parse(event.data || "{}");
        if (!state.chartPayload && payload.chart) {
          state.chartPayload = payload.chart;
          if (state.chartInView) renderChart(payload.chart);
        }
      });
      state.eventSource.addEventListener("heartbeat", () => setStatus("Live", "live"));
    } catch (error) {
      scheduleReconnect();
    }
  };

  const initLazyChart = () => {
    if (!refs.chartPanel) return;
    const activate = () => {
      state.chartInView = true;
      if (state.active) fetchChart(state.active);
    };
    if (!("IntersectionObserver" in window)) {
      activate();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      observer.disconnect();
      activate();
    }, { rootMargin: "180px 0px" });
    observer.observe(refs.chartPanel);
  };

  const initEvents = () => {
    refs.refreshButtons.forEach((button) => button.addEventListener("click", () => fetchOpportunities(true)));
    refs.filterButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.filter = button.dataset.opportunityFilter || "all";
        refs.filterButtons.forEach((item) => item.classList.toggle("is-active", item === button));
        if (refs.list) refs.list.scrollTop = 0;
        schedule("opportunities", renderOpportunities);
      });
    });
    refs.quickActions.forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.quickAction || "setup";
        if (action === "refresh") {
          fetchOpportunities(true);
          return;
        }
        if (action === "fullscreen") {
          toggleFullscreenChart();
          return;
        }
        if (["panic", "allocate"].includes(action)) {
          const copy = action === "panic"
            ? "Open emergency stop controls? This will not submit an emergency stop from the dashboard."
            : "Open allocation preview? Review the vault flow before any live-impacting action.";
          if (!window.confirm(copy)) return;
        }
        openPreview(action, button.dataset.actionHref || "");
      });
    });
    refs.sheetClose?.addEventListener("click", closePreview);
    refs.sheetBackdrop?.addEventListener("click", closePreview);
    refs.list?.addEventListener("scroll", () => schedule("opportunities", renderOpportunities), { passive: true });
    refs.activity?.addEventListener("scroll", () => schedule("activity", renderActivity), { passive: true });
    refs.timeframeTabs.forEach((button) => {
      button.addEventListener("click", () => {
        state.timeframe = button.dataset.timeframe || "live";
        refs.timeframeTabs.forEach((tab) => tab.classList.toggle("is-active", tab === button));
        if (state.active) fetchChart(state.active);
      });
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        closeStream();
        startPolling();
      } else {
        fetchOpportunities(false);
        fetchActivity();
        connectStream();
      }
    });
    window.addEventListener("resize", () => schedule("chart-resize", () => state.chart?.resize?.()), { passive: true });
    window.addEventListener("pagehide", cleanup, { once: true });
  };

  function cleanup() {
    state.destroyed = true;
    closeStream();
    stopPolling();
    window.clearTimeout(state.reconnectTimer);
    Object.values(state.requests).forEach((slot) => slot.controller?.abort());
    state.chart?.destroy?.();
  }

  initEvents();
  fetchVaultPulse();
  fetchOpportunities(false);
  fetchActivity();
  connectStream();
  initLazyChart();
})();
