(() => {
  const SELECTOR = "[data-mini-chart]";
  const SVG_SELECTOR = "[data-svg-sparkline]";
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
    ctx.strokeStyle = canvas.getAttribute("data-chart-color") || "#9b4dff";
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

  // ── Inline SVG sparkline renderer ──
  // Targets <svg data-svg-sparkline data-chart-source="json-element-id">
  const buildSvgPath = (values, width, height, pad) => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, Math.abs(max) * 0.02, 1);
    const xStep = (width - pad * 2) / Math.max(values.length - 1, 1);
    const yFor = (v) => height - pad - ((v - min) / span) * (height - pad * 2);
    const pts = values.map((v, i) => `${pad + i * xStep},${yFor(v)}`);
    return `M ${pts.join(" L ")}`;
  };

  const buildAreaPath = (values, width, height, pad) => {
    const linePts = buildSvgPath(values, width, height, pad);
    return `${linePts} L ${pad + (values.length - 1) * (width - pad * 2) / Math.max(values.length - 1, 1)},${height - pad} L ${pad},${height - pad} Z`;
  };

  const renderSvgSparklines = () => {
    document.querySelectorAll(SVG_SELECTOR).forEach((svg) => {
      const sourceId = svg.getAttribute("data-chart-source");
      const node = sourceId ? document.getElementById(sourceId) : null;
      let values = [];
      if (node) {
        try {
          const payload = JSON.parse(node.textContent || "{}");
          values = (Array.isArray(payload?.points) ? payload.points : [])
            .map((p) => Number(p?.value ?? p?.y ?? p))
            .filter(Number.isFinite);
        } catch {}
      }

      const vb = (svg.getAttribute("viewBox") || "0 0 120 38").split(" ").map(Number);
      const width = vb[2] || 120;
      const height = vb[3] || 38;
      const pad = 3;
      const stroke = svg.getAttribute("data-stroke") || "#9b4dff";
      const fill   = svg.getAttribute("data-fill")   || "rgba(155,77,255,0.14)";

      // Clear previous children except <defs>
      Array.from(svg.children).forEach((child) => {
        if (child.tagName.toLowerCase() !== "defs") child.remove();
      });

      if (values.length < 2) {
        svg.closest("[data-chart-card]")?.classList.add("is-empty");
        return;
      }
      svg.closest("[data-chart-card]")?.classList.remove("is-empty");

      const ns = "http://www.w3.org/2000/svg";

      // Area fill
      const area = document.createElementNS(ns, "path");
      area.setAttribute("d", buildAreaPath(values, width, height, pad));
      area.setAttribute("fill", fill);
      area.setAttribute("class", "av-sparkline-area");
      svg.appendChild(area);

      // Line
      const line = document.createElementNS(ns, "path");
      const pathD = buildSvgPath(values, width, height, pad);
      line.setAttribute("d", pathD);
      line.setAttribute("stroke", stroke);
      line.setAttribute("class", "av-sparkline-line av-sparkline-path");

      if (!reduceMotion) {
        const totalLength = 200; // approximate; CSS handles animation
        line.setAttribute("pathLength", "200");
        line.style.strokeDasharray = "200";
        line.style.strokeDashoffset = "200";
        line.style.animation = "av-sparkline-draw 0.8s cubic-bezier(0.22,1,0.36,1) 0.1s both";
      }
      svg.appendChild(line);

      // Endpoint dot
      if (!reduceMotion && values.length > 0) {
        const xStep = (width - pad * 2) / Math.max(values.length - 1, 1);
        const lastX = pad + (values.length - 1) * xStep;
        const minV = Math.min(...values);
        const maxV = Math.max(...values);
        const spanV = Math.max(maxV - minV, Math.abs(maxV) * 0.02, 1);
        const lastY = height - pad - ((values[values.length - 1] - minV) / spanV) * (height - pad * 2);
        const dot = document.createElementNS(ns, "circle");
        dot.setAttribute("cx", lastX);
        dot.setAttribute("cy", lastY);
        dot.setAttribute("r", "2.5");
        dot.setAttribute("fill", svg.getAttribute("data-dot-color") || "#c58bff");
        dot.style.animation = "av-fade-in 0.3s ease 0.85s both";
        svg.appendChild(dot);
      }
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderSvgSparklines, { once: true });
  } else {
    renderSvgSparklines();
  }
})();
