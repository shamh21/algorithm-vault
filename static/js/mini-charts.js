(() => {
  const SELECTOR = "[data-mini-chart]";
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

  const parsePayload = (canvas) => {
    const sourceId = canvas.getAttribute("data-chart-source");
    if (!sourceId) return null;
    const node = document.getElementById(sourceId);
    if (!node) return null;
    try {
      const payload = JSON.parse(node.textContent || "{}");
      return payload && typeof payload === "object" ? payload : null;
    } catch {
      return null;
    }
  };

  const valuesFromPayload = (payload) => {
    const points = Array.isArray(payload?.points) ? payload.points : [];
    return points
      .map((point) => Number(point?.value ?? point?.y ?? point))
      .filter((value) => Number.isFinite(value));
  };

  const resizeCanvas = (canvas) => {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(Math.round(rect.width * ratio), 2);
    const height = Math.max(Math.round(rect.height * ratio), 2);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height, ratio };
  };

  const drawLine = (canvas, payload) => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const values = valuesFromPayload(payload);
    const { width, height, ratio } = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    if (values.length < 2) {
      canvas.closest("[data-chart-card]")?.classList.add("is-empty");
      return;
    }
    canvas.closest("[data-chart-card]")?.classList.remove("is-empty");

    const pad = 10 * ratio;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, Math.abs(max) * 0.02, 1);
    const xStep = (width - pad * 2) / Math.max(values.length - 1, 1);
    const yFor = (value) => height - pad - ((value - min) / span) * (height - pad * 2);

    const gradient = ctx.createLinearGradient(0, pad, 0, height - pad);
    gradient.addColorStop(0, "rgba(125, 211, 252, 0.22)");
    gradient.addColorStop(1, "rgba(125, 211, 252, 0)");

    ctx.beginPath();
    values.forEach((value, index) => {
      const x = pad + index * xStep;
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(width - pad, height - pad);
    ctx.lineTo(pad, height - pad);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    values.forEach((value, index) => {
      const x = pad + index * xStep;
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineWidth = 2.25 * ratio;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = canvas.getAttribute("data-chart-color") || "#7dd3fc";
    ctx.stroke();

    if (!reduceMotion) {
      const lastX = pad + (values.length - 1) * xStep;
      const lastY = yFor(values[values.length - 1]);
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3.5 * ratio, 0, Math.PI * 2);
      ctx.fillStyle = canvas.getAttribute("data-chart-point-color") || "#86efac";
      ctx.fill();
    }
  };

  const drawAll = () => {
    document.querySelectorAll(SELECTOR).forEach((canvas) => {
      if (!(canvas instanceof HTMLCanvasElement)) return;
      drawLine(canvas, parsePayload(canvas));
    });
  };

  let resizeFrame = 0;
  const scheduleDraw = () => {
    if (resizeFrame) return;
    resizeFrame = window.requestAnimationFrame(() => {
      resizeFrame = 0;
      drawAll();
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", drawAll, { once: true });
  } else {
    drawAll();
  }
  window.addEventListener("resize", scheduleDraw, { passive: true });
})();
