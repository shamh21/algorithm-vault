(() => {
  const root = document.querySelector("[data-treasury-dashboard]");
  if (!root) return;

  const refs = {
    health: root.querySelector("[data-treasury-health]"),
    balance: root.querySelector("[data-treasury-balance]"),
    liability: root.querySelector("[data-treasury-liability]"),
    ratio: root.querySelector("[data-treasury-ratio]"),
    runway: root.querySelector("[data-treasury-runway]"),
    queue: root.querySelector("[data-treasury-queue]"),
    deficit: root.querySelector("[data-treasury-deficit]"),
    chart: root.querySelector("[data-treasury-chart]"),
    streamStatus: root.querySelector("[data-treasury-stream-status]"),
    alertList: root.querySelector("[data-treasury-alert-list]"),
    sheet: root.querySelector("[data-treasury-alert-sheet]"),
    sheetList: root.querySelector("[data-treasury-sheet-list]"),
    alertToggle: root.querySelector("[data-treasury-alert-toggle]"),
    alertClose: root.querySelector("[data-treasury-alert-close]"),
  };

  const urls = {
    solvency: root.dataset.solvencyUrl,
    stream: root.dataset.streamUrl,
  };
  const apiUrl = (path) => window.AlgVaultConfig?.apiUrl?.(path) || path;

  const state = {
    chart: null,
    frame: 0,
    latest: null,
    eventSource: null,
  };

  const formatEth = (value) => `${Number(value || 0).toFixed(8)} ETH`;
  const formatRatio = (value) => `${Number(value || 0).toFixed(2)}x`;
  const formatRunway = (value) => (Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}h` : "n/a");
  const title = (value) => String(value || "").replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());

  const scheduleRender = (payload) => {
    state.latest = payload;
    if (state.frame) return;
    state.frame = window.requestAnimationFrame(() => {
      state.frame = 0;
      render(state.latest || {});
    });
  };

  const render = (payload) => {
    const reserve = payload.state || {};
    if (refs.health) refs.health.textContent = title(reserve.health_status || "unknown");
    if (refs.balance) refs.balance.textContent = Number(reserve.total_eth_balance || 0).toFixed(8);
    if (refs.liability) refs.liability.textContent = formatEth(reserve.total_estimated_liability);
    if (refs.ratio) refs.ratio.textContent = formatRatio(reserve.reserve_ratio);
    if (refs.runway) refs.runway.textContent = formatRunway(reserve.projected_runway);
    if (refs.queue) refs.queue.textContent = String(reserve.queued_withdrawal_count || 0);
    if (refs.deficit) refs.deficit.textContent = formatEth(reserve.deficit_eth);
    renderAlerts(payload.alerts || []);
    renderChart(payload.forecasts || []);
  };

  const renderAlerts = (alerts) => {
    const rows = alerts.slice(0, 12).map((alert) => (
      `<div class="treasury-alert-item treasury-alert-${escapeHtml(alert.severity || "info")}">
        <strong>${escapeHtml(title(alert.event_type))}</strong>
        <span>${escapeHtml(alert.message || "")}</span>
      </div>`
    )).join("");
    const html = rows || '<p class="muted">No treasury solvency alerts.</p>';
    if (refs.alertList) refs.alertList.innerHTML = rows || html;
    if (refs.sheetList) refs.sheetList.innerHTML = html;
  };

  const renderChart = (forecasts) => {
    if (!refs.chart || !window.Chart) return;
    const ordered = [...forecasts].reverse().slice(-8);
    const labels = ordered.map((row) => row.forecast_window || "");
    const reserve = ordered.map((row) => Number(row.projected_reserve || 0));
    const liability = ordered.map((row) => Number(row.projected_liability || 0));
    if (!state.chart) {
      state.chart = new window.Chart(refs.chart, {
        type: "line",
        data: {
          labels,
          datasets: [
            { label: "Reserve", data: reserve, borderColor: "#00c076", backgroundColor: "rgba(0,192,118,.12)", tension: 0.32, pointRadius: 2 },
            { label: "Liability", data: liability, borderColor: "#f0b90b", backgroundColor: "rgba(240,185,11,.10)", tension: 0.32, pointRadius: 2 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 180 },
          plugins: { legend: { labels: { color: "#d8dee9", boxWidth: 10 } } },
          scales: {
            x: { ticks: { color: "#8f8f8f", maxRotation: 0 }, grid: { color: "rgba(255,255,255,.06)" } },
            y: { ticks: { color: "#8f8f8f" }, grid: { color: "rgba(255,255,255,.06)" } },
          },
        },
      });
      return;
    }
    state.chart.data.labels = labels;
    state.chart.data.datasets[0].data = reserve;
    state.chart.data.datasets[1].data = liability;
    state.chart.update("none");
  };

  const fetchSnapshot = async () => {
    if (!urls.solvency) return;
    const url = new URL(apiUrl(urls.solvency), window.location.origin);
    url.searchParams.set("recalculate", "1");
    const response = await fetch(url, { credentials: "same-origin", cache: "no-store" });
    if (!response.ok) throw new Error(`Treasury snapshot failed: ${response.status}`);
    scheduleRender(await response.json());
  };

  const connectStream = () => {
    if (!urls.stream || !window.EventSource) return;
    state.eventSource = new EventSource(apiUrl(urls.stream));
    state.eventSource.addEventListener("solvency", (event) => {
      if (refs.streamStatus) refs.streamStatus.textContent = "Live";
      scheduleRender(JSON.parse(event.data || "{}"));
    });
    state.eventSource.onerror = () => {
      if (refs.streamStatus) refs.streamStatus.textContent = "Polling";
      state.eventSource?.close();
      state.eventSource = null;
      window.setTimeout(() => fetchSnapshot().catch(() => {}), 4000);
    };
  };

  const escapeHtml = (value) => String(value || "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;",
  }[char]));

  refs.alertToggle?.addEventListener("click", () => {
    refs.sheet?.classList.add("is-open");
    refs.sheet?.setAttribute("aria-hidden", "false");
  });
  refs.alertClose?.addEventListener("click", () => {
    refs.sheet?.classList.remove("is-open");
    refs.sheet?.setAttribute("aria-hidden", "true");
  });
  refs.sheet?.addEventListener("click", (event) => {
    if (event.target === refs.sheet) refs.alertClose?.click();
  });

  fetchSnapshot().catch(() => {});
  connectStream();

  window.addEventListener("pagehide", () => state.eventSource?.close(), { once: true });
})();
