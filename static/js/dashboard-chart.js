(() => {
  class DashboardChart {
    constructor({ host, overlay, librarySrc }) {
      this.host = host;
      this.overlay = overlay;
      this.librarySrc = librarySrc;
      this.chart = null;
      this.candleSeries = null;
      this.payload = null;
      this.libraryPromise = null;
      this.resizeFrame = 0;
      this.resizeObserver = null;
      this.boundQueueResize = () => this.queueResize();
      this.setupAutoResize();
    }

    async render(payload) {
      this.payload = payload || {};
      this.payload.candles = this.cleanCandles(Array.isArray(this.payload.candles) ? this.payload.candles : []);
      try {
        await this.ensureChart();
        const candles = Array.isArray(this.payload.candles) ? this.payload.candles.slice(-150) : [];
        this.candleSeries?.setData(candles.map((row) => ({
          time: this.number(row.time),
          open: this.number(row.open),
          high: this.number(row.high),
          low: this.number(row.low),
          close: this.number(row.close),
        })));
        this.chart?.timeScale?.().fitContent?.();
      } catch (error) {
        // The overlay canvas is a complete fallback when the chart library is unavailable.
      }
      this.draw();
      this.queueResize();
    }

    resize() {
      const rect = this.host?.getBoundingClientRect?.();
      if (rect && rect.width > 0 && rect.height > 0 && this.chart?.resize) {
        this.chart.resize(Math.floor(rect.width), Math.floor(rect.height));
      } else {
        this.chart?.applyOptions?.({ autoSize: true });
      }
      this.draw();
    }

    destroy() {
      if (this.resizeFrame) {
        cancelAnimationFrame(this.resizeFrame);
        this.resizeFrame = 0;
      }
      this.resizeObserver?.disconnect?.();
      window.removeEventListener("resize", this.boundQueueResize);
      window.removeEventListener("orientationchange", this.boundQueueResize);
      window.visualViewport?.removeEventListener("resize", this.boundQueueResize);
      window.visualViewport?.removeEventListener("scroll", this.boundQueueResize);
      this.chart?.remove?.();
      this.chart = null;
      this.candleSeries = null;
    }

    setupAutoResize() {
      if (this.host && "ResizeObserver" in window) {
        this.resizeObserver = new ResizeObserver(this.boundQueueResize);
        this.resizeObserver.observe(this.host);
      }
      window.addEventListener("resize", this.boundQueueResize, { passive: true });
      window.addEventListener("orientationchange", this.boundQueueResize, { passive: true });
      window.visualViewport?.addEventListener("resize", this.boundQueueResize, { passive: true });
      window.visualViewport?.addEventListener("scroll", this.boundQueueResize, { passive: true });
    }

    queueResize() {
      if (this.resizeFrame) return;
      this.resizeFrame = requestAnimationFrame(() => {
        this.resizeFrame = 0;
        this.resize();
      });
    }

    async ensureChart() {
      if (!this.host) return false;
      const library = await this.loadLibrary();
      if (!library?.createChart) return false;
      if (this.chart) return true;
      this.chart = library.createChart(this.host, {
        autoSize: true,
        layout: { background: { color: "#050505" }, textColor: "#a8a8a8", fontFamily: "Inter, system-ui, sans-serif" },
        grid: { vertLines: { color: "rgba(255,255,255,0.045)" }, horzLines: { color: "rgba(255,255,255,0.055)" } },
        rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
        timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: true, secondsVisible: false },
        crosshair: { mode: 0 },
        kineticScroll: { touch: true, mouse: true },
        trackingMode: { exitMode: 1 },
      });
      this.candleSeries = this.chart.addCandlestickSeries({
        upColor: "#00c076",
        downColor: "#ff4d4f",
        borderVisible: false,
        wickUpColor: "#00c076",
        wickDownColor: "#ff4d4f",
      });
      return true;
    }

    loadLibrary() {
      if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts);
      if (this.libraryPromise) return this.libraryPromise;
      this.libraryPromise = new Promise((resolve, reject) => {
        if (!this.librarySrc) {
          reject(new Error("chart library url missing"));
          return;
        }
        const script = document.createElement("script");
        script.src = this.librarySrc;
        script.async = true;
        script.onload = () => resolve(window.LightweightCharts);
        script.onerror = reject;
        document.head.append(script);
      });
      return this.libraryPromise;
    }

    draw() {
      const canvas = this.overlay;
      if (!canvas) return;
      const candles = Array.isArray(this.payload?.candles) ? this.payload.candles.slice(-150) : [];
      const overlays = this.payload?.overlays || {};
      this.resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;
      ctx.clearRect(0, 0, width, height);
      if (!candles.length) {
        this.empty(ctx, width, height);
        return;
      }
      if (!this.chart || !this.candleSeries) {
        this.fallbackLine(ctx, candles, width, height);
      }
      this.uncertainty(ctx, overlays, width, height, candles);
      this.riskBands(ctx, overlays, width, height, candles);
      this.zones(ctx, overlays, width, height, candles);
      this.path(ctx, overlays, width, height, candles);
      this.confidence(ctx, overlays, width, height, candles);
      this.forecastLabel(ctx, overlays, width, height);
    }

    fallbackLine(ctx, candles, width, height) {
      ctx.save();
      ctx.strokeStyle = "rgba(240,185,11,0.85)";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      candles.forEach((row, index) => {
        const x = (index / Math.max(candles.length - 1, 1)) * width;
        const y = this.yForPrice(this.number(row.close), candles, height);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    zones(ctx, overlays, width, height, candles) {
      const zones = overlays.zones || {};
      [
        ["entry", zones.entry?.price, "rgba(240,185,11,0.8)"],
        ["exit", zones.exit?.price, "rgba(0,192,118,0.75)"],
        ["stop", zones.stop_loss?.price, "rgba(255,77,79,0.75)"],
      ].forEach(([name, raw, color]) => {
        const value = this.number(raw);
        if (!value) return;
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

    path(ctx, overlays, width, height, candles) {
      const path = Array.isArray(overlays.path) ? overlays.path : [];
      if (!path.length) return;
      const allPrices = [...candles.map((row) => this.number(row.close)), ...path.map((row) => this.number(row.value))];
      const min = Math.min(...allPrices);
      const max = Math.max(...allPrices);
      const y = (value) => height - ((value - min) / Math.max(max - min, 1e-9)) * height;
      ctx.save();
      ctx.strokeStyle = "rgba(240,185,11,0.96)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      path.forEach((point, index) => {
        const x = width * (0.62 + (index / Math.max(path.length - 1, 1)) * 0.34);
        const py = y(this.number(point.value));
        if (index === 0) ctx.moveTo(x, py);
        else ctx.lineTo(x, py);
      });
      ctx.stroke();
      ctx.restore();
    }

    confidence(ctx, overlays, width, height, candles) {
      const upper = overlays.projected_range?.upper || overlays.confidence_band?.upper;
      const lower = overlays.projected_range?.lower || overlays.confidence_band?.lower;
      if (!Array.isArray(upper) || !Array.isArray(lower) || !upper.length || upper.length !== lower.length) return;
      const allPrices = [
        ...candles.map((row) => this.number(row.close)),
        ...upper.map((row) => this.number(row.value)),
        ...lower.map((row) => this.number(row.value)),
      ];
      const min = Math.min(...allPrices);
      const max = Math.max(...allPrices);
      const y = (value) => height - ((value - min) / Math.max(max - min, 1e-9)) * height;
      ctx.save();
      ctx.fillStyle = "rgba(240,185,11,0.08)";
      ctx.beginPath();
      upper.forEach((point, index) => {
        const x = width * (0.62 + (index / Math.max(upper.length - 1, 1)) * 0.34);
        const py = y(this.number(point.value));
        if (index === 0) ctx.moveTo(x, py);
        else ctx.lineTo(x, py);
      });
      [...lower].reverse().forEach((point, index) => {
        const x = width * (0.96 - (index / Math.max(lower.length - 1, 1)) * 0.34);
        ctx.lineTo(x, y(this.number(point.value)));
      });
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    uncertainty(ctx, overlays, width, height) {
      const intensity = Math.max(0, Math.min(this.number(overlays.uncertainty_shading?.intensity, 0), 1));
      if (!intensity) return;
      ctx.save();
      const gradient = ctx.createLinearGradient(width * 0.58, 0, width, 0);
      gradient.addColorStop(0, `rgba(240,185,11,${0.02 + intensity * 0.03})`);
      gradient.addColorStop(1, `rgba(240,185,11,${0.08 + intensity * 0.06})`);
      ctx.fillStyle = gradient;
      ctx.fillRect(width * 0.58, 0, width * 0.42, height);
      ctx.restore();
    }

    riskBands(ctx, overlays, width, height, candles) {
      const band = overlays.stop_loss_band || {};
      const upper = this.number(band.upper);
      const lower = this.number(band.lower);
      if (upper > 0 && lower > 0) {
        const y1 = this.priceToY(upper, candles, height);
        const y2 = this.priceToY(lower, candles, height);
        ctx.save();
        ctx.fillStyle = "rgba(255,77,79,0.08)";
        ctx.fillRect(0, Math.min(y1, y2), width, Math.max(Math.abs(y2 - y1), 1));
        ctx.restore();
      }
      const invalidation = this.number(overlays.invalidation_zone?.price);
      if (invalidation > 0) {
        const y = this.priceToY(invalidation, candles, height);
        ctx.save();
        ctx.strokeStyle = "rgba(255,77,79,0.55)";
        ctx.setLineDash([2, 7]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(width * 0.58, y);
        ctx.lineTo(width, y);
        ctx.stroke();
        ctx.restore();
      }
    }

    forecastLabel(ctx, overlays, width) {
      const probability = this.number(overlays.trend_continuation_probability, NaN);
      const regime = overlays.volatility_expansion?.state;
      if (!Number.isFinite(probability) && !regime) return;
      ctx.save();
      ctx.fillStyle = "rgba(5,5,5,0.72)";
      ctx.strokeStyle = "rgba(255,255,255,0.10)";
      ctx.lineWidth = 1;
      const text = [
        Number.isFinite(probability) ? `Trend ${Math.round(probability)}%` : "",
        regime ? String(regime).replace(/_/g, " ") : "",
      ].filter(Boolean).join(" · ");
      ctx.font = "700 11px Inter, system-ui, sans-serif";
      const labelWidth = Math.min(width - 24, ctx.measureText(text).width + 22);
      const x = Math.max(12, width - labelWidth - 12);
      ctx.beginPath();
      ctx.roundRect?.(x, 12, labelWidth, 26, 6);
      if (!ctx.roundRect) ctx.rect(x, 12, labelWidth, 26);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.82)";
      ctx.fillText(text, x + 11, 29);
      ctx.restore();
    }

    cleanCandles(rows) {
      const seen = new Set();
      const cleaned = [];
      let previousClose = 0;
      rows.forEach((row, index) => {
        const time = this.number(row.time || row.timestamp || index);
        const close = this.number(row.close || row.c || row.price);
        if (!time || close <= 0 || seen.has(time)) return;
        if (previousClose > 0 && Math.abs(close / previousClose - 1) > 0.45) return;
        const open = this.number(row.open || row.o, close);
        const high = Math.max(this.number(row.high || row.h, close), open, close);
        const low = Math.min(this.number(row.low || row.l, close), open, close);
        if (low <= 0 || high <= 0) return;
        previousClose = close;
        seen.add(time);
        cleaned.push({ time, open, high, low, close, volume: this.number(row.volume || row.v) });
      });
      return cleaned.sort((a, b) => a.time - b.time).slice(-150);
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

    priceToY(value, candles, height) {
      if (this.chart && this.candleSeries) {
        const coord = this.candleSeries.priceToCoordinate(value);
        if (Number.isFinite(coord)) return coord;
      }
      return this.yForPrice(value, candles, height);
    }

    yForPrice(value, candles, height) {
      const values = candles.flatMap((row) => [this.number(row.high), this.number(row.low), this.number(row.close)]).filter((item) => item > 0);
      const min = Math.min(...values);
      const max = Math.max(...values);
      return height - ((value - min) / Math.max(max - min, 1e-9)) * height;
    }

    empty(ctx, width, height) {
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.42)";
      ctx.font = "600 12px Inter, system-ui, sans-serif";
      ctx.fillText("Awaiting chart data", 16, height / 2);
      ctx.restore();
    }

    number(value, fallback = 0) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    }
  }

  window.AlgorithmVaultDashboardChart = DashboardChart;
})();
