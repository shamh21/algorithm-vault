(() => {
  const root = document.querySelector("[data-command-center]");
  if (!root) return;

  const parseJson = (id) => {
    const node = document.getElementById(id);
    if (!node) return {};
    try {
      return JSON.parse(node.textContent || "{}");
    } catch {
      return {};
    }
  };

  const payload = parseJson("account-pnl-data");
  const fallbackPayload = parseJson("command-center-payload").pnl_history || {};
  const pnl = Object.keys(payload).length ? payload : fallbackPayload;

  const number = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const moneyCompact = (value) => {
    const parsed = number(value);
    const sign = parsed > 0 ? "+" : parsed < 0 ? "-" : "";
    const amount = Math.abs(parsed);
    if (amount >= 1000000) return `${sign}$${(amount / 1000000).toFixed(1)}M`;
    if (amount >= 1000) return `${sign}$${(amount / 1000).toFixed(1)}K`;
    return `${sign}$${amount.toFixed(2)}`;
  };

  const cssColor = (name, fallback) => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  };

  const chartCanvas = root.querySelector("[data-account-pnl-chart]");
  const chartEmpty = root.querySelector("[data-account-pnl-empty]");
  const chartLoading = root.querySelector("[data-account-pnl-loading]");
  const chartShell = root.querySelector("[data-account-pnl-shell]");
  const points = Array.isArray(pnl.points) ? pnl.points : [];
  const scheduleFrame = window.requestAnimationFrame
    ? window.requestAnimationFrame.bind(window)
    : (callback) => window.setTimeout(callback, 0);

  const resizeCanvas = (canvas) => {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: width / dpr, height: height / dpr };
  };

  const showEmpty = () => {
    chartLoading?.setAttribute("hidden", "hidden");
    chartEmpty?.removeAttribute("hidden");
    chartShell?.setAttribute("data-state", pnl.state || "empty");
  };

  const renderChart = () => {
    if (!chartCanvas) return;
    chartLoading?.removeAttribute("hidden");
    const { ctx, width, height } = resizeCanvas(chartCanvas);
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    if ((pnl.state || "") === "error" || points.length < 2) {
      showEmpty();
      return;
    }

    chartEmpty?.setAttribute("hidden", "hidden");
    chartShell?.setAttribute("data-state", "ready");

    const values = points.map((point) => number(point.value));
    const min = Math.min(0, ...values);
    const max = Math.max(0, ...values);
    const spread = Math.max(max - min, 1);
    const pad = {
      top: 18,
      right: 16,
      bottom: 32,
      left: width < 440 ? 36 : 48,
    };
    const plotWidth = Math.max(1, width - pad.left - pad.right);
    const plotHeight = Math.max(1, height - pad.top - pad.bottom);
    const xFor = (index) => pad.left + (index / Math.max(points.length - 1, 1)) * plotWidth;
    const yFor = (value) => pad.top + (1 - (value - min) / spread) * plotHeight;
    const finalValue = values[values.length - 1] || 0;
    const lineColor = finalValue >= 0 ? cssColor("--success", "#00c076") : cssColor("--danger", "#ff4d4f");
    const gridColor = "rgba(148, 163, 184, 0.14)";
    const textColor = cssColor("--muted", "#8f8f8f");
    const zeroY = yFor(0);

    ctx.save();
    ctx.font = "700 11px Inter, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.fillStyle = textColor;
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    for (let index = 0; index <= 3; index += 1) {
      const tick = min + (spread / 3) * index;
      const y = yFor(tick);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      ctx.fillText(moneyCompact(tick), 4, y);
    }

    ctx.strokeStyle = "rgba(255, 255, 255, 0.24)";
    ctx.beginPath();
    ctx.moveTo(pad.left, zeroY);
    ctx.lineTo(width - pad.right, zeroY);
    ctx.stroke();

    const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
    gradient.addColorStop(0, finalValue >= 0 ? "rgba(0, 192, 118, 0.22)" : "rgba(255, 77, 79, 0.2)");
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = xFor(index);
      const y = yFor(number(point.value));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(xFor(points.length - 1), zeroY);
    ctx.lineTo(xFor(0), zeroY);
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
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.stroke();

    const last = points[points.length - 1];
    const lastX = xFor(points.length - 1);
    const lastY = yFor(number(last.value));
    ctx.fillStyle = lineColor;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = textColor;
    ctx.font = "800 10px Inter, system-ui, sans-serif";
    ctx.fillText(String(points[0]?.label || ""), pad.left, height - 10);
    const endLabel = String(last?.label || "");
    const labelWidth = ctx.measureText(endLabel).width;
    ctx.fillText(endLabel, Math.max(pad.left, width - pad.right - labelWidth), height - 10);
    ctx.restore();

    chartLoading?.setAttribute("hidden", "hidden");
  };

  let frame = 0;
  const scheduleRender = () => {
    if (frame) return;
    frame = scheduleFrame(() => {
      frame = 0;
      renderChart();
    });
  };

  scheduleRender();
  window.addEventListener("resize", scheduleRender, { passive: true });
  if (window.ResizeObserver && chartShell) {
    const observer = new ResizeObserver(scheduleRender);
    observer.observe(chartShell);
  }
})();
