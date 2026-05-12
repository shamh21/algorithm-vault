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
    }

    async render(payload) {
      this.payload = payload || {};
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
    }

    resize() {
      this.chart?.applyOptions?.({ autoSize: true });
      this.draw();
    }

    destroy() {
      this.chart?.remove?.();
      this.chart = null;
      this.candleSeries = null;
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
      this.zones(ctx, overlays, width, height, candles);
      this.path(ctx, overlays, width, height, candles);
      this.confidence(ctx, overlays, width, height, candles);
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
      const upper = overlays.confidence_band?.upper;
      const lower = overlays.confidence_band?.lower;
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
