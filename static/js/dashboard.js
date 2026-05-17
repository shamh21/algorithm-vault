(() => {
  const MAX_ROWS = 150;
  const PAGE_SIZE = 30;
  const ROW_HEIGHT = 62;
  const ACTIVITY_ROW_HEIGHT = 58;
  const POLL_FOREGROUND_MS = 15000;
  const POLL_BACKGROUND_MS = 60000;
  const RESTORE_CACHE_KEY = "av-admin-dashboard-restore-v1";
  const RESTORE_CACHE_TTL_MS = 5 * 60 * 1000;

  const root = document.querySelector("[data-dashboard]");
  if (!root) return;

  const refs = {
    list: root.querySelector("[data-opportunity-list]"),
    status: root.querySelector("[data-stream-status]"),
    connectionBanner: root.querySelector("[data-dashboard-connection-banner]"),
    connectionTitle: root.querySelector("[data-dashboard-connection-title]"),
    connectionDetail: root.querySelector("[data-dashboard-connection-detail]"),
    connectionRetry: root.querySelector("[data-dashboard-retry]"),
    cacheState: root.querySelector("[data-cache-state]"),
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
    equityValue: root.querySelector("[data-equity-value]"),
    equityStatus: root.querySelector("[data-equity-status]"),
    dailyPnl: root.querySelector("[data-daily-pnl]"),
    brokerValue: root.querySelector("[data-broker-value]"),
    brokerStatus: root.querySelector("[data-broker-status]"),
    syncValue: root.querySelector("[data-sync-value]"),
    syncStatus: root.querySelector("[data-sync-status]"),
    cacheNote: root.querySelector("[data-cache-note]"),
    providerHealthProvider: root.querySelector("[data-provider-health-provider]"),
    providerHealthStatus: root.querySelector("[data-provider-health-status]"),
    providerHealthLastCheck: root.querySelector("[data-provider-health-last-check]"),
    providerHealthImpact: root.querySelector("[data-provider-health-impact]"),
    marketTape: root.querySelector("[data-market-tape]"),
    strategyRankingsTable: root.querySelector("[data-strategy-rankings-table]"),
    openOrdersTable: root.querySelector("[data-open-orders-table]"),
    positionsTable: root.querySelector("[data-positions-table]"),
    recentTradesTable: root.querySelector("[data-recent-trades-table]"),
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
    signalQuality: root.querySelector("[data-signal-quality]"),
    marketRegime: root.querySelector("[data-market-regime]"),
    providerQuality: root.querySelector("[data-provider-quality]"),
    forecastExpiry: root.querySelector("[data-forecast-expiry]"),
    confidenceMeter: root.querySelector("[data-confidence-meter]"),
    confidenceFill: root.querySelector("[data-confidence-fill]"),
    rationale: root.querySelector("[data-signal-rationale]"),
    rationaleBody: root.querySelector("[data-signal-rationale-body]"),
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
    opportunityDiagnostics: {},
    filter: "all",
    timeframe: "live",
    chart: null,
    chartPayload: null,
    chartInView: false,
    chartModulePromise: null,
    restoreCache: null,
    connectionState: "connecting",
    eventSource: null,
    pollTimer: null,
    reconnectTimer: null,
    staleStreamTimer: null,
    expiryTimer: null,
    reconnectAttempt: 0,
    lastHeartbeatAt: 0,
    latestAppliedAt: {
      opportunities: 0,
      chart: 0,
      activity: 0,
      dashboard: 0,
    },
    forecastExpiresAt: 0,
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
  const plainCurrency = (value) => (hasNumber(value) ? Number(value).toFixed(2) : dash);
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
  const isPlainObject = (value) => value && typeof value === "object" && !Array.isArray(value);
  const payloadTime = (payload) => {
    const raw = payload?.updated_at || payload?.generated_at || payload?.at || payload?.savedAt || 0;
    if (!raw) return Date.now();
    const parsed = typeof raw === "number" ? raw : Date.parse(raw);
    if (!Number.isFinite(parsed)) return Date.now();
    return parsed < 10_000_000_000 ? parsed * 1000 : parsed;
  };
  const shouldApplyPayload = (key, payload) => {
    const stamp = payloadTime(payload);
    if (stamp + 250 < (state.latestAppliedAt[key] || 0)) return false;
    state.latestAppliedAt[key] = Math.max(stamp, Date.now());
    return true;
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

  const readRestoreCache = () => {
    try {
      const raw = window.sessionStorage.getItem(RESTORE_CACHE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      const savedAt = Number(parsed?.savedAt || 0);
      if (!savedAt || Date.now() - savedAt > RESTORE_CACHE_TTL_MS) {
        window.sessionStorage.removeItem(RESTORE_CACHE_KEY);
        return null;
      }
      if (!isPlainObject(parsed)) {
        window.sessionStorage.removeItem(RESTORE_CACHE_KEY);
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  };

  const writeRestoreCache = (patch = {}) => {
    try {
      const next = { ...(state.restoreCache || {}), ...patch, savedAt: Date.now() };
      window.sessionStorage.setItem(RESTORE_CACHE_KEY, JSON.stringify(next));
      state.restoreCache = next;
    } catch {}
  };

  const cacheAgeLabel = (savedAt) => {
    const seconds = Math.max(1, Math.round((Date.now() - Number(savedAt || 0)) / 1000));
    if (seconds < 60) return "just now";
    return `${Math.round(seconds / 60)} min ago`;
  };

  const showCacheState = (text = "") => {
    if (!refs.cacheState) return;
    refs.cacheState.hidden = !text;
    refs.cacheState.textContent = text;
  };

  const setStatus = (text, tone = "") => {
    if (!refs.status) return;
    refs.status.classList.toggle("is-live", tone === "live");
    refs.status.classList.toggle("is-stale", tone === "stale");
    const copy = refs.status.querySelector("span:last-child");
    if (copy) copy.textContent = text;
  };

  const setConnectionState = (name, detail = "") => {
    const normalized = navigator.onLine === false ? "offline" : name || "connecting";
    state.connectionState = normalized;
    root.dataset.connectionState = normalized;
    const copy = {
      connecting: ["Connecting", "Connecting", "Live dashboard data is refreshing.", ""],
      live: ["Live", "Live", "Stream connected.", "live"],
      polling: ["Polling", "Reconnecting", "Live stream is unavailable, so the dashboard is polling for updates.", "stale"],
      reconnecting: ["Polling", "Reconnecting", detail || "Reconnecting to the live dashboard stream.", "stale"],
      cached: ["Cached", "Restored", detail || "Showing the latest local dashboard snapshot while live data refreshes.", "stale"],
      stale: ["Stale", "Stale data", detail || "Live data is delayed. Existing values remain visible while the dashboard retries.", "stale"],
      offline: ["Offline", "Offline", "Reconnect to refresh account, market, and activity data.", "stale"],
      error: ["Stale", "Data issue", detail || "Unable to refresh one or more dashboard panels.", "stale"],
    }[normalized] || ["Connecting", "Connecting", detail || "Live dashboard data is refreshing.", ""];
    setStatus(copy[0], copy[3]);
    const showBanner = !["connecting", "live"].includes(normalized);
    if (refs.connectionBanner) {
      refs.connectionBanner.hidden = !showBanner;
      refs.connectionBanner.dataset.connectionTone = normalized;
    }
    if (refs.connectionTitle) refs.connectionTitle.textContent = copy[1];
    if (refs.connectionDetail) refs.connectionDetail.textContent = copy[2];
  };

  const requestJson = async (key, url) => {
    if (navigator.onLine === false) {
      setConnectionState("offline");
      throw new Error("offline");
    }
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
    const payload = await response.json();
    if (id !== slot.id) return null;
    return payload;
  };

  const setOpportunities = (rows) => {
    state.opportunities = (Array.isArray(rows) ? rows.filter(Boolean) : []).slice(0, MAX_ROWS);
    if (!state.opportunities.length) {
      state.active = null;
      updateEmptyForecast();
    } else if (!state.active || !state.opportunities.some((row) => row.provider === state.active.provider && row.symbol === state.active.symbol)) {
      state.active = state.opportunities[0];
      updateSelectionText(state.active);
      if (state.chartInView) fetchChart(state.active);
    } else {
      const refreshed = state.opportunities.find((row) => row.provider === state.active.provider && row.symbol === state.active.symbol);
      if (refreshed) {
        state.active = refreshed;
        updateSelectionText(state.active);
      }
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
    const target = state.filter === "long" ? "buy" : state.filter === "short" ? "sell" : state.filter;
    return state.opportunities.filter((row) => String(row.direction || row.action || "").toLowerCase() === target);
  };

  const renderOpportunities = () => {
    const rows = filteredOpportunities();
    const emptyText = state.opportunityDiagnostics?.stale
      ? "Cached scanner data has no ranked markets."
      : state.opportunityDiagnostics?.error
        ? "Unable to load ranked markets. Check provider health and refresh again."
      : "No ranked markets loaded. Check provider health or run a strategy cycle to populate rankings.";
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
      const confidenceScore = hasNumber(row.confidence_score) ? Number(row.confidence_score) : number(row.confidence) * 100;
      const quality = row.signal_quality?.grade || (confidenceScore >= 70 ? "High" : confidenceScore >= 45 ? "Moderate" : "Low");
      conf.textContent = `Confidence ${Math.round(confidenceScore)} · ${quality}`;
      const regime = document.createElement("small");
      regime.className = "opportunity-quality-line";
      regime.textContent = `${label(row.market_regime?.state || "regime pending")} · ${label(row.data_quality?.state || "data pending")}`;
      main.append(title, sub, conf, regime);

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
    }, state.filter === "all" ? emptyText : `No ${state.filter.toUpperCase()} setups in the current scanner window.`);
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

  const formatSyncValue = (value) => {
    if (!value) return dash;
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  };

  const accountStatusCopy = (status) => {
    const normalized = String(status || "").toLowerCase();
    if (normalized === "live") return "Live snapshot";
    if (normalized === "cached") return "Cached snapshot";
    if (normalized === "degraded") return "Provider unavailable";
    if (normalized === "error") return "Provider error";
    return "Not loaded";
  };

  const renderRows = (tbody, rows, columns, emptyText) => {
    if (!tbody) return;
    const data = Array.isArray(rows) ? rows.slice(0, MAX_ROWS) : [];
    if (!data.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = columns.length;
      td.textContent = emptyText;
      tr.append(td);
      tbody.replaceChildren(tr);
      return;
    }
    const fragment = document.createDocumentFragment();
    data.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((column) => {
        const td = document.createElement("td");
        const value = column.value(row);
        if (column.className) td.className = column.className(row);
        td.textContent = value;
        tr.append(td);
      });
      fragment.append(tr);
    });
    tbody.replaceChildren(fragment);
  };

  const renderMarketSummary = (rows) => {
    if (!refs.marketTape) return;
    const data = (Array.isArray(rows) ? rows : [])
      .filter((row) => row && row.symbol && String(row.symbol).toUpperCase() !== "N/A")
      .slice(0, 8);
    if (!data.length) {
      refs.marketTape.replaceChildren(Object.assign(document.createElement("div"), {
        className: "trade-empty-state",
        textContent: "Market monitor is waiting for live market data.",
      }));
      return;
    }
    const fragment = document.createDocumentFragment();
    data.forEach((row) => {
      const item = document.createElement("div");
      item.className = "market-tape-row";
      const symbol = document.createElement("strong");
      symbol.textContent = String(row.symbol || "").toUpperCase();
      const mid = document.createElement("span");
      mid.textContent = price(row.mid);
      const change = document.createElement("em");
      const changePct = hasNumber(row.change_pct) ? Number(row.change_pct) : null;
      change.textContent = changePct !== null ? `${changePct.toFixed(2)}%` : dash;
      change.classList.toggle("positive", changePct !== null && changePct >= 0);
      change.classList.toggle("negative", changePct !== null && changePct < 0);
      item.append(symbol, mid, change);
      fragment.append(item);
    });
    refs.marketTape.replaceChildren(fragment);
  };

  const renderAccountSummary = (payload = {}) => {
    const snapshot = payload.account_snapshot || {};
    const providerHealth = payload.provider_health || {};
    const balances = Array.isArray(payload.balances) ? payload.balances : [];
    const balance = getBalanceAmount(balances);
    const equity = hasNumber(snapshot.equity_usd) ? Number(snapshot.equity_usd) : balance > 0 ? balance : null;
    const status = String(snapshot.status || "unavailable").toLowerCase();
    const provider = snapshot.provider || providerHealth.provider || "";
    const syncedAt = snapshot.synced_at || providerHealth.last_checked_at || payload.account_synced_at || "";
    const dailyPnl = payload.risk_status?.daily_realized_pnl;

    if (refs.equityValue) refs.equityValue.textContent = equity !== null ? plainCurrency(equity) : dash;
    if (refs.equityStatus) refs.equityStatus.textContent = accountStatusCopy(status);
    if (refs.dailyPnl) {
      refs.dailyPnl.textContent = hasNumber(dailyPnl) ? Number(dailyPnl).toFixed(4) : dash;
      refs.dailyPnl.classList.toggle("positive", hasNumber(dailyPnl) && Number(dailyPnl) >= 0);
      refs.dailyPnl.classList.toggle("negative", hasNumber(dailyPnl) && Number(dailyPnl) < 0);
    }
    if (refs.brokerValue) refs.brokerValue.textContent = provider ? String(provider).toUpperCase() : dash;
    if (refs.brokerStatus) refs.brokerStatus.textContent = providerHealth.status_label || (provider ? "Connected" : "Disconnected or not loaded");
    if (refs.syncValue) refs.syncValue.textContent = formatSyncValue(syncedAt);
    if (refs.syncStatus) refs.syncStatus.textContent = syncedAt ? `${accountStatusCopy(status)} · ${formatSyncValue(syncedAt)}` : "Loading or unavailable";
    if (refs.cacheNote) {
      refs.cacheNote.textContent = providerHealth.impact
        || (status === "cached" && syncedAt ? `Cached snapshot · Last updated ${formatSyncValue(syncedAt)}` : "Dashboard opens with one read-only live exchange refresh; cached data remains visible during provider backoff.");
    }
    if (refs.providerHealthProvider) refs.providerHealthProvider.textContent = provider ? String(provider).toUpperCase() : "Activity feed";
    if (refs.providerHealthStatus) refs.providerHealthStatus.textContent = providerHealth.status_label || (provider ? "Connected" : "Provider status appears in Activity");
    if (refs.providerHealthLastCheck) refs.providerHealthLastCheck.textContent = formatSyncValue(syncedAt);
    if (refs.providerHealthImpact) refs.providerHealthImpact.textContent = providerHealth.impact || "No structured health data loaded";

    renderMarketSummary(payload.market_summary);

    renderRows(refs.strategyRankingsTable, payload.strategy_rankings, [
      { value: (row) => `${row.strategy_name || dash}${row.symbol ? ` · ${row.symbol}/${row.timeframe || "live"}` : ""}` },
      { value: (row) => hasNumber(row.score) ? Number(row.score).toFixed(3) : dash },
      { value: (row) => hasNumber(row.recent_performance_score) ? `${(Number(row.recent_performance_score) * 100).toFixed(2)}%` : dash },
      { value: (row) => hasNumber(row.max_drawdown) ? `${(Number(row.max_drawdown) * 100).toFixed(2)}%` : dash },
      { value: (row) => row.rejected ? "rejected" : "live-readiness candidate" },
    ], "No optimizer rankings yet. Run a strategy cycle to populate rankings.");
    renderRows(refs.openOrdersTable, payload.open_orders, [
      { value: (row) => row.symbol || dash },
      { value: (row) => label(row.side).toUpperCase() },
      { value: (row) => price(row.price ?? row.limit_price) },
      { value: (row) => price(row.size ?? row.quantity) },
      { value: (row) => row.reduce_only ? "yes" : "no" },
    ], "No open orders. Live execution has no active resting orders.");
    renderRows(refs.positionsTable, payload.positions, [
      { value: (row) => row.symbol || dash },
      { value: (row) => price(row.quantity ?? row.size) },
      { value: (row) => price(row.entry_price ?? row.entry) },
      { value: (row) => price(row.mark_price ?? row.mark_value) },
      {
        value: (row) => hasNumber(row.unrealized_pnl) ? Number(row.unrealized_pnl).toFixed(4) : dash,
        className: (row) => hasNumber(row.unrealized_pnl) ? Number(row.unrealized_pnl) >= 0 ? "positive" : "negative" : "",
      },
    ], "No positions. The connected account currently has no open exposure.");
    renderRows(refs.recentTradesTable, payload.recent_trades, [
      { value: (row) => row.symbol || dash },
      { value: (row) => label(row.side).toUpperCase() },
      { value: (row) => price(row.price) },
      { value: (row) => price(row.size ?? row.quantity) },
      { value: (row) => hasNumber(row.closed_pnl) ? Number(row.closed_pnl).toFixed(4) : dash },
    ], "No recent trades. Executed trades will appear here after the next filled order.");
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

  const updateConfidenceMeter = (score) => {
    const value = Math.max(0, Math.min(Math.round(number(score)), 100));
    if (refs.confidenceMeter) {
      refs.confidenceMeter.setAttribute("aria-valuenow", String(value));
      refs.confidenceMeter.setAttribute("aria-valuetext", `${value} percent confidence`);
      refs.confidenceMeter.dataset.confidenceTone = value >= 75 ? "high" : value >= 50 ? "medium" : "low";
    }
    if (refs.confidenceFill) refs.confidenceFill.style.width = `${value}%`;
  };

  const factorList = (title, rows) => {
    const section = document.createElement("section");
    const heading = document.createElement("h3");
    heading.textContent = title;
    const list = document.createElement("ul");
    (Array.isArray(rows) && rows.length ? rows : ["No material factors reported."]).slice(0, 5).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = String(item);
      list.append(li);
    });
    section.append(heading, list);
    return section;
  };

  const renderRationale = (row) => {
    if (!refs.rationaleBody) return;
    refs.rationaleBody.replaceChildren();
    if (!row) {
      refs.rationaleBody.append(Object.assign(document.createElement("p"), { textContent: "Signal rationale appears once a forecast is selected." }));
      return;
    }
    const explanation = isPlainObject(row.explanation) ? row.explanation : {};
    const summary = document.createElement("p");
    summary.textContent = explanation.summary || "Confidence reflects weighted market, liquidity, model, and data-quality inputs.";
    const meta = document.createElement("dl");
    meta.className = "signal-rationale-meta";
    [
      ["Data freshness", explanation.data_freshness || row.data_quality?.signal_freshness || "unknown"],
      ["Provider reliability", explanation.provider_reliability || row.data_quality?.provider_reliability || "unknown"],
      ["Volatility", explanation.volatility_condition || row.market_regime?.label || "unknown"],
      ["Trend probability", hasNumber(row.trend_continuation_probability) ? `${Math.round(Number(row.trend_continuation_probability))}%` : dash],
    ].forEach(([term, value]) => {
      const wrap = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = String(value);
      wrap.append(dt, dd);
      meta.append(wrap);
    });
    refs.rationaleBody.append(
      summary,
      meta,
      factorList("Bullish", row.bullish_factors || explanation.bullish_factors),
      factorList("Bearish", row.bearish_factors || explanation.bearish_factors),
      factorList("Neutralizing", row.neutralizing_factors || explanation.neutralizing_factors),
      factorList("Risk penalties", row.risk_penalties || explanation.risk_penalties),
    );
  };

  const updateForecastExpiry = () => {
    if (!refs.forecastExpiry) return;
    if (!state.forecastExpiresAt) {
      refs.forecastExpiry.textContent = "Expiry pending";
      return;
    }
    const remaining = Math.max(0, Math.round((state.forecastExpiresAt - Date.now()) / 1000));
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    refs.forecastExpiry.textContent = remaining > 0 ? `Expires ${minutes}:${String(seconds).padStart(2, "0")}` : "Forecast expired";
    refs.forecastExpiry.classList.toggle("is-stale", remaining <= 0);
  };

  const startForecastExpiry = (seconds) => {
    window.clearInterval(state.expiryTimer);
    const resolvedSeconds = Math.max(0, Math.round(number(seconds)));
    state.forecastExpiresAt = resolvedSeconds > 0 ? Date.now() + resolvedSeconds * 1000 : 0;
    updateForecastExpiry();
    if (state.forecastExpiresAt) state.expiryTimer = window.setInterval(updateForecastExpiry, 1000);
  };

  const updateTrustState = (row) => {
    const confidenceScore = hasNumber(row?.confidence_score) ? Number(row.confidence_score) : number(row?.confidence) * 100;
    const quality = row?.signal_quality?.grade || (confidenceScore >= 75 ? "High" : confidenceScore >= 50 ? "Moderate" : "Low");
    const dataQuality = row?.data_quality?.state || "unknown";
    const regime = row?.market_regime?.label || label(row?.market_regime?.state || "unknown");
    if (refs.signalQuality) refs.signalQuality.textContent = `Signal ${quality}`;
    if (refs.marketRegime) refs.marketRegime.textContent = `Regime ${regime}`;
    if (refs.providerQuality) refs.providerQuality.textContent = `Data ${label(dataQuality)}`;
    updateConfidenceMeter(confidenceScore);
    startForecastExpiry(row?.forecast_expiry_seconds || row?.forecast?.horizon_seconds || 0);
    renderRationale(row);
    const degraded = ["poor", "stale", "insufficient"].includes(String(dataQuality));
    if (degraded) {
      setConnectionState("stale", "Forecast confidence is degraded because market data quality is limited.");
    }
  };

  const updateEmptyForecast = () => {
    refs.chartLoading?.setAttribute("hidden", "hidden");
    if (refs.chartTitle) refs.chartTitle.textContent = "Market Projection";
    if (refs.chartProvider) refs.chartProvider.textContent = "Forecast";
    if (refs.forecastDirection) refs.forecastDirection.textContent = "No active setup";
    if (refs.forecastRoi) refs.forecastRoi.textContent = dash;
    if (refs.forecastConfidence) refs.forecastConfidence.textContent = dash;
    if (refs.forecastRisk) refs.forecastRisk.textContent = dash;
    if (refs.signalQuality) refs.signalQuality.textContent = "Quality pending";
    if (refs.marketRegime) refs.marketRegime.textContent = "Regime pending";
    if (refs.providerQuality) refs.providerQuality.textContent = "Provider pending";
    if (refs.forecastExpiry) refs.forecastExpiry.textContent = "Expiry pending";
    updateConfidenceMeter(0);
    renderRationale(null);
    Object.values(refs.intel).forEach((node) => {
      if (node) node.textContent = dash;
    });
    refs.setupHelper?.removeAttribute("hidden");
  };

  const fetchVaultPulse = async () => {
    const fallback = state.restoreCache?.payload || initialPayload;
    renderVaultPulse(fallback, { loading: !state.restoreCache?.payload });
    if (!urls.dashboardData) {
      renderAccountSummary(fallback);
      renderVaultPulse(fallback, { loading: false });
      return;
    }
    try {
      const payload = await requestJson("dashboard", urls.dashboardData);
      if (payload && shouldApplyPayload("dashboard", payload)) {
        renderAccountSummary(payload);
        renderVaultPulse(payload, { loading: false });
        writeRestoreCache({ payload });
        showCacheState("");
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        renderAccountSummary(fallback);
        renderVaultPulse(fallback, { loading: false, error: true });
        setConnectionState(navigator.onLine === false ? "offline" : "stale", "Account data could not refresh. The last visible snapshot remains on screen.");
      }
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
      if (!shouldApplyPayload("opportunities", payload)) return;
      state.opportunityDiagnostics = payload.diagnostics || {};
      setOpportunities(payload.opportunities || []);
      writeRestoreCache({ opportunities: state.opportunities, opportunityDiagnostics: state.opportunityDiagnostics });
      setConnectionState(payload.diagnostics?.stale ? "stale" : "live", payload.diagnostics?.stale ? "Scanner data is cached while providers refresh." : "");
    } catch (error) {
      if (error.name !== "AbortError") {
        state.opportunityDiagnostics = { error: String(error?.message || error || "Unable to load ranked markets.") };
        if (!state.opportunities.length) setOpportunities([]);
        else schedule("opportunities", renderOpportunities);
        setConnectionState(navigator.onLine === false ? "offline" : "error", "Unable to load ranked markets. Check provider health and refresh again.");
      }
    }
  };

  const fetchActivity = async () => {
    if (!urls.activity) return;
    const url = new URL(urls.activity, window.location.origin);
    url.searchParams.set("limit", String(PAGE_SIZE));
    try {
      const payload = await requestJson("activity", url);
      if (payload && shouldApplyPayload("activity", payload)) {
        setActivity(payload.items || []);
        writeRestoreCache({ activity: state.activity });
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        setActivity(state.activity);
        setConnectionState(navigator.onLine === false ? "offline" : "stale", "Activity feed could not refresh. Existing entries remain visible.");
      }
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
      if (!shouldApplyPayload("chart", payload)) return;
      state.chartPayload = payload;
      if (state.active && isPlainObject(payload.forecast)) {
        state.active = { ...state.active, ...payload.forecast, data_quality: payload.data_quality || payload.forecast.data_quality || state.active.data_quality };
        updateSelectionText(state.active);
      }
      writeRestoreCache({ chartPayload: payload });
      await renderChart(payload);
    } catch (error) {
      if (error.name !== "AbortError") {
        await renderChart(state.chartPayload || state.restoreCache?.chartPayload || {});
        setConnectionState(navigator.onLine === false ? "offline" : "stale", "Chart data could not refresh. The last projection remains visible.");
      }
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
    if (refs.forecastConfidence) {
      const score = hasNumber(row.confidence_score) ? `${Math.round(Number(row.confidence_score))}%` : percent(row.confidence, 0);
      refs.forecastConfidence.textContent = executable ? score : dash;
    }
    if (refs.forecastRisk) refs.forecastRisk.textContent = executable && hasNumber(row.risk_reward) ? number(row.risk_reward).toFixed(2) : dash;
    updateTrustState(row);
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
    window.clearTimeout(state.staleStreamTimer);
    state.staleStreamTimer = null;
  };

  const armStaleStreamTimer = () => {
    window.clearTimeout(state.staleStreamTimer);
    state.staleStreamTimer = window.setTimeout(() => {
      if (state.destroyed || document.hidden) return;
      const staleAfter = Math.max(POLL_FOREGROUND_MS * 2, 32000);
      if (Date.now() - state.lastHeartbeatAt > staleAfter) {
        setConnectionState("stale", "Live stream heartbeat is delayed. Polling remains active while the stream recovers.");
        startPolling();
      }
    }, Math.max(POLL_FOREGROUND_MS * 2, 32000));
  };

  const parseEvent = (event) => {
    try {
      return JSON.parse(event.data || "{}");
    } catch {
      return {};
    }
  };

  const scheduleReconnect = () => {
    if (state.destroyed || document.hidden) return;
    closeStream();
    startPolling();
    window.clearTimeout(state.reconnectTimer);
    const delay = Math.min(30000, 1000 * (2 ** Math.min(state.reconnectAttempt, 5))) + Math.floor(Math.random() * 650);
    state.reconnectAttempt += 1;
    setConnectionState("reconnecting", `Reconnecting to the live stream in ${Math.ceil(delay / 1000)} seconds. Polling remains active.`);
    state.reconnectTimer = window.setTimeout(connectStream, delay);
  };

  const connectStream = () => {
    if (state.destroyed || document.hidden) return;
    if (!window.EventSource || !urls.stream) {
      startPolling();
      setConnectionState("polling");
      return;
    }
    closeStream();
    try {
      const url = apiUrl(urls.stream);
      state.eventSource = new EventSource(url);
      state.eventSource.onopen = () => {
        state.reconnectAttempt = 0;
        state.lastHeartbeatAt = Date.now();
        armStaleStreamTimer();
        stopPolling();
        setConnectionState("live");
      };
      state.eventSource.onerror = () => {
        setConnectionState("polling");
        scheduleReconnect();
      };
      state.eventSource.addEventListener("opportunities", (event) => {
        const payload = parseEvent(event);
        if (!shouldApplyPayload("opportunities", payload)) return;
        setOpportunities(payload.opportunities || []);
      });
      state.eventSource.addEventListener("activity", (event) => {
        const payload = parseEvent(event);
        if (!shouldApplyPayload("activity", payload)) return;
        setActivity(payload.items || []);
      });
      state.eventSource.addEventListener("chart_delta", (event) => {
        const payload = parseEvent(event);
        if (!shouldApplyPayload("chart", payload)) return;
        if (!state.chartPayload && payload.chart) {
          state.chartPayload = payload.chart;
          if (state.chartInView) renderChart(payload.chart);
        }
      });
      state.eventSource.addEventListener("heartbeat", (event) => {
        state.lastHeartbeatAt = Date.now();
        const payload = parseEvent(event);
        shouldApplyPayload("dashboard", payload);
        armStaleStreamTimer();
        setConnectionState("live");
      });
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
    const refreshDashboard = () => {
      setConnectionState(navigator.onLine === false ? "offline" : "connecting");
      fetchVaultPulse();
      fetchOpportunities(true);
      fetchActivity();
      connectStream();
    };
    const queueChartResize = () => schedule("chart-resize", () => state.chart?.resize?.());

    refs.connectionRetry?.addEventListener("click", refreshDashboard);
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
        if (action === "allocate") {
          if (!window.confirm("Open allocation preview? Review the vault flow before any live-impacting action.")) return;
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
        setConnectionState(navigator.onLine === false ? "offline" : "connecting");
        fetchOpportunities(false);
        fetchActivity();
        connectStream();
      }
    });
    window.addEventListener("online", refreshDashboard, { passive: true });
    window.addEventListener("offline", () => {
      closeStream();
      startPolling();
      setConnectionState("offline");
    }, { passive: true });
    window.addEventListener("resize", queueChartResize, { passive: true });
    window.addEventListener("orientationchange", queueChartResize, { passive: true });
    window.visualViewport?.addEventListener("resize", queueChartResize, { passive: true });
    window.visualViewport?.addEventListener("scroll", queueChartResize, { passive: true });
    window.addEventListener("pagehide", cleanup, { once: true });
  };

  function cleanup() {
    state.destroyed = true;
    closeStream();
    stopPolling();
    window.clearTimeout(state.reconnectTimer);
    window.clearTimeout(state.staleStreamTimer);
    window.clearInterval(state.expiryTimer);
    Object.values(state.requests).forEach((slot) => slot.controller?.abort());
    state.chart?.destroy?.();
  }

  const restoreDisplayCache = () => {
    state.restoreCache = readRestoreCache();
    if (!state.restoreCache) return false;
    if (isPlainObject(state.restoreCache.payload)) {
      renderAccountSummary(state.restoreCache.payload);
      renderVaultPulse(state.restoreCache.payload, { loading: false });
    }
    if (Array.isArray(state.restoreCache.opportunities)) {
      state.opportunityDiagnostics = state.restoreCache.opportunityDiagnostics || { stale: true };
      setOpportunities(state.restoreCache.opportunities);
    }
    if (Array.isArray(state.restoreCache.activity)) {
      setActivity(state.restoreCache.activity);
    }
    if (isPlainObject(state.restoreCache.chartPayload)) {
      state.chartPayload = state.restoreCache.chartPayload;
    }
    const restored = `Restored ${cacheAgeLabel(state.restoreCache.savedAt)} while live data refreshes.`;
    showCacheState(restored);
    setConnectionState("cached", restored);
    return true;
  };

  initEvents();
  renderAccountSummary(initialPayload);
  if (!restoreDisplayCache()) setConnectionState("connecting");
  if (!state.active) updateEmptyForecast();
  fetchVaultPulse();
  fetchOpportunities(false);
  fetchActivity();
  connectStream();
  initLazyChart();
})();
