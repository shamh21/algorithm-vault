(() => {
  document.querySelectorAll("[data-copy-button]").forEach((button) => {
    button.addEventListener("click", async () => {
      const scope = button.closest(".wallet-flow-card, .qr-panel, .address-copy-panel") || document;
      const input = scope.querySelector("[data-copy-source]");
      if (!input) return;
      try {
        await navigator.clipboard.writeText(input.value);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = button.classList.contains("copy-button") ? "Copy" : "Copy Secret";
        }, 1400);
      } catch (error) {
        input.select();
        document.execCommand("copy");
      }
    });
  });
})();
