(() => {
  const form = document.querySelector(".vault-form");
  if (!form) return;

  const countdowns = Array.from(document.querySelectorAll("[data-countdown]"));
  const durationInputs = Array.from(document.querySelectorAll("input[name='lock_duration']"));
  const submitButton = form.querySelector("button[type='submit']");
  const riskNotice = form.querySelector(".risk-note span");
  const idempotencyInput = form.querySelector("[data-vault-idempotency-key]");
  const oneH10Ack = document.querySelector("[data-one-h10-live-ack]");
  const oneH10AckInput = oneH10Ack?.querySelector("input[type='checkbox']");
  const oneH10AckStatus = document.querySelector("[data-one-h10-ack-status]");
  const customDurationInput = document.querySelector("input[name='custom_duration_hours']");
  const customDurationUnit = document.querySelector("select[name='custom_duration_unit']");
  let timerId = null;
  const durationCopy = {
    "1": "1H10 aims to 10x the user's input amount in 1 hour. This is a strategy objective, not a guaranteed return, and execution remains risk-gated.",
    "24": "24h cycle balances momentum and mean-reversion signals with moderate caps, spread controls, and backend risk checks.",
    "48": "48h cycle optimizes a balanced multi-factor scope with drawdown, volatility, liquidity, and turnover penalties.",
    "168": "7d cycle uses slower trend and volatility-breakout scopes with stronger drawdown controls and risk-gated execution.",
    custom: "Custom cycle uses the closest matching horizon model and remains subject to liquidity, spread, drawdown, and backend risk checks.",
  };

  function randomKey() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  if (idempotencyInput && !idempotencyInput.value) {
    idempotencyInput.value = randomKey();
  }

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
    if (!selected) return "1H10";
    if (selected.value === "1") return "1H10";
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
    if (oneH10Ack) {
      const isOneH10 = selected.value === "1";
      oneH10Ack.hidden = !isOneH10;
      if (!isOneH10 && oneH10AckInput) oneH10AckInput.checked = false;
      updateOneH10AckStatus();
    }
  }

  function updateOneH10AckStatus() {
    if (!oneH10AckStatus || !oneH10AckInput) return;
    oneH10AckStatus.textContent = oneH10AckInput.checked ? "Acknowledged" : "Not acknowledged";
  }

  durationInputs.forEach((input) => input.addEventListener("change", updateSubmitLabel));
  if (oneH10AckInput) oneH10AckInput.addEventListener("change", updateOneH10AckStatus);
  if (customDurationInput) customDurationInput.addEventListener("input", updateSubmitLabel);
  if (customDurationUnit) customDurationUnit.addEventListener("change", updateSubmitLabel);
  updateSubmitLabel();
  startTimers();
})();
