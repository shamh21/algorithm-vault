(() => {
  const setCopyState = (button, feedback, label) => {
    button.textContent = label;
    if (feedback) feedback.textContent = label === "Copied" ? "Address copied" : "";
  };

  document.querySelectorAll("[data-copy-button]").forEach((button) => {
    const defaultLabel = button.getAttribute("data-copy-default") || button.textContent || "Copy";
    button.addEventListener("click", async () => {
      const scope = button.closest(".wallet-flow-card, .qr-panel, .address-copy-panel") || document;
      const input = scope.querySelector("[data-copy-source]");
      const feedback = scope.querySelector("[data-copy-feedback]");
      if (!input) return;
      try {
        await navigator.clipboard.writeText(input.value);
        setCopyState(button, feedback, "Copied");
        window.setTimeout(() => setCopyState(button, feedback, defaultLabel), 1400);
      } catch (error) {
        input.select();
        document.execCommand("copy");
        setCopyState(button, feedback, "Copied");
        window.setTimeout(() => setCopyState(button, feedback, defaultLabel), 1400);
      }
    });
  });

  document.querySelectorAll("form[data-disable-on-submit]").forEach((form) => {
    form.addEventListener("submit", () => {
      if (!form.checkValidity()) return;
      form.querySelectorAll("button[type='submit']").forEach((button) => {
        const loadingLabel = button.getAttribute("data-loading-label");
        if (loadingLabel) button.textContent = loadingLabel;
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
      });
    });
  });

  const offlineNotice = document.querySelector("[data-wallet-offline]");
  if (offlineNotice) {
    const updateOfflineState = (event) => {
      offlineNotice.hidden = event?.type === "offline" ? false : window.navigator.onLine;
    };
    updateOfflineState();
    window.addEventListener("online", updateOfflineState, { passive: true });
    window.addEventListener("offline", updateOfflineState, { passive: true });
  }
})();
