(() => {
  const root = document.querySelector("[data-command-center]");
  if (!root) return;

  const payloadNode = document.getElementById("command-center-payload");
  const payload = (() => {
    try {
      return JSON.parse(payloadNode?.textContent || "{}");
    } catch {
      return {};
    }
  })();

  const number = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const money = (value) => {
    const parsed = number(value);
    return `${parsed >= 0 ? "+" : ""}$${parsed.toFixed(2)}`;
  };

  const renderPerformanceChart = () => {
    const canvas = root.querySelector("[data-vault-performance-chart]");
    const empty = root.querySelector("[data-vault-performance-empty]");
    const points = Array.isArray(payload.performance) ? payload.performance : [];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = width / dpr;
    const h = height / dpr;
    ctx.clearRect(0, 0, w, h);

    if (points.length < 2) {
      empty?.removeAttribute("hidden");
      return;
    }
    empty?.setAttribute("hidden", "hidden");

    const values = points.map((point) => number(point.value));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = 18;
    const xFor = (index) => pad + (index / Math.max(points.length - 1, 1)) * (w - pad * 2);
    const yFor = (value) => h - pad - ((value - min) / Math.max(max - min, 1e-9)) * (h - pad * 2);

    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.055)";
    ctx.lineWidth = 1;
    for (let index = 1; index <= 3; index += 1) {
      const y = (h / 4) * index;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, "rgba(125,211,252,0.26)");
    gradient.addColorStop(1, "rgba(125,211,252,0)");
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(number(point.value));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(xFor(points.length - 1), h - pad);
    ctx.lineTo(xFor(0), h - pad);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(number(point.value));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#7dd3fc";
    ctx.lineWidth = 2;
    ctx.stroke();

    const last = points[points.length - 1];
    ctx.fillStyle = number(last.pnl) >= 0 ? "#86efac" : "#fb7185";
    ctx.beginPath();
    ctx.arc(xFor(points.length - 1), yFor(number(last.value)), 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  const sheet = root.parentElement?.querySelector("[data-strategy-sheet]") || document.querySelector("[data-strategy-sheet]");
  const backdrop = document.querySelector("[data-strategy-backdrop]");
  const sheetTitle = sheet?.querySelector("[data-strategy-sheet-title]");
  const sheetBody = sheet?.querySelector("[data-strategy-sheet-body]");

  const closeSheet = () => {
    sheet?.setAttribute("aria-hidden", "true");
    backdrop?.setAttribute("aria-hidden", "true");
    document.body.classList.remove("strategy-sheet-open");
  };

  const openSheet = (bot) => {
    if (!sheet || !sheetBody || !bot) return;
    if (sheetTitle) sheetTitle.textContent = bot.name || "Strategy";
    const rows = [
      ["Status", bot.status || "Monitoring"],
      ["Market", `${bot.provider || "Global"} · ${bot.pair || "--"} · ${bot.timeframe || "--"}`],
      ["Recent P&L", `${number(bot.pnl_pct).toFixed(2)}%`],
      ["Win Rate", `${number(bot.win_rate).toFixed(0)}%`],
      ["Drawdown", `${number(bot.drawdown).toFixed(2)}%`],
      ["Score", number(bot.score).toFixed(2)],
      ["Risk Mode", bot.risk_mode || "Standard"],
      ["Last Action", bot.last_action || "Monitoring"],
      ["Note", bot.detail || "Read-only automation candidate."],
    ];
    sheetBody.replaceChildren(...rows.map(([label, value]) => {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = value;
      row.append(dt, dd);
      return row;
    }));
    sheet.setAttribute("aria-hidden", "false");
    backdrop?.setAttribute("aria-hidden", "false");
    document.body.classList.add("strategy-sheet-open");
  };

  root.querySelectorAll("[data-open-strategy-sheet]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number.parseInt(button.getAttribute("data-strategy-index") || "0", 10);
      openSheet((payload.bots || [])[index]);
    });
  });

  document.querySelector("[data-close-strategy-sheet]")?.addEventListener("click", closeSheet);
  backdrop?.addEventListener("click", closeSheet);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSheet();
  });

  let resizeFrame = 0;
  const scheduleChart = () => {
    if (resizeFrame) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = 0;
      renderPerformanceChart();
    });
  };
  scheduleChart();
  window.addEventListener("resize", scheduleChart, { passive: true });

  if (window.location.hash) {
    const target = root.querySelector(window.location.hash);
    target?.scrollIntoView({ block: "start" });
  }
})();
