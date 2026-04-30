(() => {
  const countdowns = Array.from(document.querySelectorAll("[data-countdown]"));
  const durationInputs = Array.from(document.querySelectorAll("input[name='lock_duration']"));
  const submitButton = document.querySelector(".vault-form button[type='submit']");
  const riskNotice = document.querySelector(".vault-form .risk-note span");
  const customDurationInput = document.querySelector("input[name='custom_duration_hours']");
  const customDurationUnit = document.querySelector("select[name='custom_duration_unit']");
  let timerId = null;
  const durationCopy = {
    "1": "1h cycle uses short-horizon allocation with strict stops, smaller position caps, and liquidity checks. Estimated performance can change during the cycle.",
    "24": "24h cycle balances momentum and mean-reversion signals with moderate caps, spread controls, and backend risk checks.",
    "48": "48h cycle optimizes a balanced multi-factor scope with drawdown, volatility, liquidity, and turnover penalties.",
    "168": "7d cycle uses slower trend and volatility-breakout scopes with stronger drawdown controls and risk-gated execution.",
    custom: "Custom cycle uses the closest matching horizon model and remains subject to liquidity, spread, drawdown, and backend risk checks.",
  };

  function formatRemaining(ms) {
    if (ms <= 0) return "Ready for settlement";
    const totalSeconds = Math.floor(ms / 1000);
    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor((totalSeconds % 86400) / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    return `${minutes}m ${seconds}s`;
  }

  function updateCountdowns() {
    const now = Date.now();
    countdowns.forEach((element) => {
      const unlocksAt = Date.parse(element.dataset.unlocksAt || "");
      if (Number.isNaN(unlocksAt)) return;
      element.textContent = formatRemaining(unlocksAt - now);
    });
  }

  function startTimers() {
    if (!countdowns.length || timerId || document.hidden) return;
    updateCountdowns();
    timerId = window.setInterval(updateCountdowns, 1000);
  }

  function stopTimers() {
    if (!timerId) return;
    window.clearInterval(timerId);
    timerId = null;
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopTimers();
    else startTimers();
  });

  document.querySelectorAll("[data-asset-option]").forEach((option) => {
    option.addEventListener("click", () => {
      const group = option.closest(".asset-selector");
      if (!group) return;
      group.querySelectorAll("[data-asset-option]").forEach((item) => item.classList.remove("is-selected"));
      option.classList.add("is-selected");
      const input = option.querySelector("input[type='radio']");
      if (input) input.checked = true;
      updateSubmitLabel();
    });
  });

  function selectedDurationLabel(selected) {
    if (!selected) return "1h";
    if (selected.value === "168") return "7d";
    if (selected.value === "custom") {
      const amount = Math.max(1, Number.parseInt(customDurationInput?.value || "24", 10) || 24);
      return customDurationUnit?.value === "minutes" ? `${amount}m` : `${amount}h`;
    }
    return `${selected.value}h`;
  }

  function updateSubmitLabel() {
    if (!submitButton || !durationInputs.length) return;
    const selected = durationInputs.find((input) => input.checked);
    if (!selected) return;
    const label = selectedDurationLabel(selected);
    submitButton.textContent = `Start ${label} Cycle`;
    if (riskNotice) {
      riskNotice.textContent = durationCopy[selected.value] || durationCopy.custom;
    }
  }

  durationInputs.forEach((input) => input.addEventListener("change", updateSubmitLabel));
  if (customDurationInput) customDurationInput.addEventListener("input", updateSubmitLabel);
  if (customDurationUnit) customDurationUnit.addEventListener("change", updateSubmitLabel);
  updateSubmitLabel();
  startTimers();
})();
