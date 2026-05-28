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

  const parseJson = (id, fallback) => {
    const node = document.getElementById(id);
    if (!node) return fallback;
    try {
      return JSON.parse(node.textContent || "");
    } catch (error) {
      return fallback;
    }
  };

  const sheet = document.querySelector("[data-wallet-onramp-sheet]");
  const form = document.querySelector("[data-wallet-onramp-form]");
  if (!sheet || !form) return;

  const cardConfig = parseJson("wallet-card-buy-data", {});
  const applePayConfig = parseJson("wallet-apple-pay-data", {});
  const legacyOnrampConfig = parseJson("wallet-onramp-data", {});
  const panel = document.querySelector("[data-wallet-card-buy-panel]");
  const csrfToken = panel?.getAttribute("data-csrf-token") || "";
  const dialog = sheet.querySelector(".wallet-onramp-dialog");
  const dialogTitle = sheet.querySelector("#wallet-onramp-title");
  const dialogCopy = sheet.querySelector("[data-wallet-onramp-copy]");
  const stateNode = sheet.querySelector("[data-wallet-onramp-state]");
  const submitButton = sheet.querySelector("[data-wallet-onramp-submit]");
  const assetSelect = sheet.querySelector("[data-onramp-asset]");
  const networkSelect = sheet.querySelector("[data-onramp-network]");
  const amountInput = sheet.querySelector("[data-onramp-amount]");
  const destinationAsset = sheet.querySelector("[data-onramp-destination-asset]");
  const destinationAddress = sheet.querySelector("[data-onramp-destination-address]");
  const quoteWrap = sheet.querySelector("[data-wallet-onramp-quote]");
  const quoteSkeleton = sheet.querySelector("[data-wallet-quote-skeleton]");
  const gatewayPanel = sheet.querySelector("[data-card-gateway-panel]");
  const gatewayFrame = sheet.querySelector("[data-card-gateway-frame]");
  const gatewayState = sheet.querySelector("[data-card-gateway-state]");
  const tokenFallback = sheet.querySelector("[data-card-token-fallback]");
  const tokenInput = sheet.querySelector("[data-card-token-input]");
  const methodButtons = Array.from(sheet.querySelectorAll("[data-onramp-method]"));
  const amountPresetButtons = Array.from(sheet.querySelectorAll("[data-onramp-amount-preset]"));
  const feeNodes = {
    purchase: sheet.querySelector("[data-buy-purchase-amount]"),
    fee: sheet.querySelector("[data-apple-pay-treasury-fee]"),
    execution: sheet.querySelector("[data-apple-pay-execution-fee]"),
    total: sheet.querySelector("[data-buy-total-charged]"),
    receive: sheet.querySelector("[data-apple-pay-net]"),
  };

  const state = {
    method: "card",
    phase: "idle",
    order: null,
    applePayRequest: null,
    applePaySession: null,
    gatewayToken: "",
    pollTimer: null,
    lastFocus: null,
  };

  const money = (value, currency = "USD") =>
    new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: 2 }).format(Number(value || 0));
  const amount = (value, asset) => `${Number(value || 0).toLocaleString("en-US", { maximumFractionDigits: 8 })} ${asset || ""}`.trim();

  const activeConfig = () => (state.method === "apple_pay" ? applePayConfig : cardConfig);
  const assets = () => {
    const configured = Array.isArray(activeConfig().assets) ? activeConfig().assets : [];
    if (configured.length > 0) return configured;
    return Array.isArray(legacyOnrampConfig.assets) ? legacyOnrampConfig.assets : [];
  };

  const setMessage = (message, tone = "muted") => {
    if (!stateNode) return;
    stateNode.textContent = message || "";
    stateNode.dataset.tone = tone;
    stateNode.hidden = !message;
  };

  const setBusy = (busy, label) => {
    if (!submitButton) return;
    submitButton.disabled = Boolean(busy);
    submitButton.setAttribute("aria-busy", busy ? "true" : "false");
    if (label) submitButton.textContent = label;
  };

  const clearPolling = () => {
    if (state.pollTimer) {
      window.clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  };

  const resetQuote = () => {
    clearPolling();
    if (state.applePaySession) {
      try {
        state.applePaySession.abort();
      } catch (error) {
        // The session may already be complete or canceled.
      }
    }
    state.phase = "idle";
    state.order = null;
    state.applePayRequest = null;
    state.applePaySession = null;
    state.gatewayToken = "";
    if (tokenInput) tokenInput.value = "";
    if (quoteWrap) quoteWrap.hidden = true;
    if (quoteSkeleton) quoteSkeleton.hidden = true;
    if (gatewayPanel) gatewayPanel.hidden = true;
    if (gatewayFrame) {
      gatewayFrame.hidden = true;
      gatewayFrame.removeAttribute("src");
    }
    if (tokenFallback) tokenFallback.hidden = true;
    Object.values(feeNodes).forEach((node) => {
      if (node) node.textContent = "--";
    });
    setBusy(false, "Review Quote");
  };

  const setMethod = (method) => {
    state.method = method === "apple_pay" ? "apple_pay" : "card";
    const config = activeConfig();
    if (dialogTitle) dialogTitle.textContent = state.method === "apple_pay" ? "Buy with Apple Pay" : "Buy with Card";
    if (dialogCopy) {
      dialogCopy.textContent =
        state.method === "apple_pay"
          ? "Review the server quote, then authorize with Apple Pay on this device."
          : "Review the server quote, then authorize with hosted tokenized card fields.";
    }
    methodButtons.forEach((button) => {
      const selected = button.getAttribute("data-onramp-method") === state.method;
      button.setAttribute("aria-selected", selected ? "true" : "false");
    });
    if (amountInput) {
      amountInput.min = String(config.min_fiat_usd || 10);
      amountInput.max = String(config.max_fiat_usd || 5000);
      amountInput.value = String(config.default_amount || amountInput.value || config.min_fiat_usd || 10);
    }
    resetQuote();
    populateAssets();
    if (!config.ready) {
      setMessage(config.unavailable_copy || "This payment method is not available yet.", "warning");
    } else {
      setMessage("Ready to prepare a server-validated quote.", "muted");
    }
  };

  const populateAssets = () => {
    const options = assets();
    if (!assetSelect || !networkSelect) return;
    assetSelect.replaceChildren();
    options.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.asset;
      option.textContent = item.label || item.asset;
      assetSelect.append(option);
    });
    const defaultAsset = activeConfig().default_asset || options[0]?.asset || "";
    assetSelect.value = defaultAsset;
    populateNetworks();
  };

  const selectedAsset = () => assets().find((item) => item.asset === assetSelect?.value) || assets()[0] || null;

  const populateNetworks = () => {
    const option = selectedAsset();
    if (!networkSelect) return;
    networkSelect.replaceChildren();
    (option?.networks || []).forEach((network) => {
      const node = document.createElement("option");
      node.value = network;
      node.textContent = network;
      networkSelect.append(node);
    });
    networkSelect.value = option?.default_network || option?.networks?.[0] || "";
    updateDestination();
  };

  const updateDestination = () => {
    const option = selectedAsset();
    const network = networkSelect?.value || option?.default_network || "";
    const address = option?.addresses?.[network] || option?.address || "";
    if (destinationAsset) destinationAsset.textContent = option?.asset || activeConfig().default_asset || "Asset";
    if (destinationAddress) destinationAddress.textContent = address || "Deposit address pending";
  };

  const payload = () => ({
    asset: assetSelect?.value || activeConfig().default_asset || "",
    network: networkSelect?.value || activeConfig().default_network || "",
    fiat_currency: activeConfig().fiat_currency || "USD",
    fiat_amount: Number(amountInput?.value || 0),
    payment_method: state.method,
  });

  const requestJson = async (url, body, idempotencyKey) => {
    const response = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken,
        "Idempotency-Key": idempotencyKey || `wallet-buy-${Date.now()}`,
      },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) {
      throw new Error(data.message || "The payment provider is unavailable. Try again.");
    }
    return data;
  };

  const renderQuote = (order) => {
    const currency = order.fiat_currency || activeConfig().fiat_currency || "USD";
    if (feeNodes.purchase) feeNodes.purchase.textContent = money(order.purchase_amount, currency);
    if (feeNodes.fee) feeNodes.fee.textContent = money(order.algvault_fee, currency);
    if (feeNodes.execution) feeNodes.execution.textContent = money(order.provider_network_estimate, currency);
    if (feeNodes.total) feeNodes.total.textContent = money(order.total_charged, currency);
    if (feeNodes.receive) feeNodes.receive.textContent = amount(order.estimated_receive_amount, order.asset);
    if (quoteWrap) quoteWrap.hidden = false;
  };

  const showGateway = (gateway) => {
    if (!gatewayPanel) return;
    gatewayPanel.hidden = false;
    if (gatewayState) gatewayState.textContent = "Hosted fields ready";
    const tokenizationUrl = gateway?.tokenization_url || activeConfig().gateway?.tokenization_url || "";
    if (gatewayFrame && tokenizationUrl) {
      const url = new URL(tokenizationUrl, window.location.origin);
      if (state.order?.order_id) url.searchParams.set("order_id", state.order.order_id);
      gatewayFrame.src = url.toString();
      gatewayFrame.hidden = false;
      if (tokenFallback) tokenFallback.hidden = true;
    } else if (tokenFallback) {
      tokenFallback.hidden = false;
      if (gatewayState) gatewayState.textContent = "Gateway token required";
    }
  };

  const quote = async () => {
    const config = activeConfig();
    if (!window.navigator.onLine) {
      setMessage("Offline. Reconnect before requesting a quote.", "warning");
      return;
    }
    if (!config.ready) {
      setMessage(config.unavailable_copy || "This payment method is unavailable.", "warning");
      return;
    }
    if (!form.reportValidity()) return;
    state.phase = "quoting";
    setMessage("Preparing a server-validated quote...", "muted");
    setBusy(true, "Loading quote");
    if (quoteWrap) quoteWrap.hidden = false;
    if (quoteSkeleton) quoteSkeleton.hidden = false;
    try {
      const idempotencyKey = `wallet-${state.method}-${Date.now()}`;
      const data = await requestJson(config.quote_url, payload(), idempotencyKey);
      state.order = data.order;
      state.applePayRequest = data.apple_pay_request || {};
      state.phase = "quoted";
      if (quoteSkeleton) quoteSkeleton.hidden = true;
      renderQuote(data.order);
      if (state.method === "card") {
        showGateway(data.gateway || {});
        setBusy(false, "Authorize Card");
        setMessage("Quote locked. Complete the hosted card fields, then authorize.", "success");
      } else {
        setBusy(false, "Authorize Apple Pay");
        setMessage("Quote locked. Continue with Apple Pay authorization.", "success");
      }
      pollStatus();
    } catch (error) {
      resetQuote();
      setMessage(error.message || "Quote failed. Try again.", "danger");
    }
  };

  const applePayAvailable = () => {
    const ApplePay = window.ApplePaySession;
    return Boolean(ApplePay && typeof ApplePay.canMakePayments === "function" && ApplePay.canMakePayments());
  };

  const applePayStatus = (success) => {
    const ApplePay = window.ApplePaySession;
    return success ? ApplePay.STATUS_SUCCESS : ApplePay.STATUS_FAILURE;
  };

  const applePayPaymentRequest = () => {
    const config = activeConfig();
    const request = state.applePayRequest || {};
    const totalCharged = Number(state.order?.total_charged || state.order?.fiat_gross_amount || amountInput?.value || 0);
    return {
      countryCode: request.countryCode || config.country_code || "CA",
      currencyCode: request.currencyCode || state.order?.fiat_currency || config.fiat_currency || "USD",
      merchantCapabilities: request.merchantCapabilities || config.merchant_capabilities || ["supports3DS"],
      supportedNetworks: request.supportedNetworks || config.supported_networks || [],
      lineItems: request.lineItems || state.order?.line_items || [],
      total: request.total || {
        label: config.display_name || "AlgVault",
        amount: totalCharged.toFixed(2),
      },
    };
  };

  const authorizeApplePay = () => {
    const config = activeConfig();
    if (!state.order?.order_id) {
      quote();
      return;
    }
    if (!config.ready) {
      setMessage(config.unavailable_copy || "Apple Pay is unavailable.", "warning");
      return;
    }
    if (!applePayAvailable()) {
      setMessage("Apple Pay is not available in this browser or on this device.", "warning");
      return;
    }
    const paymentRequest = applePayPaymentRequest();
    if (!paymentRequest.supportedNetworks.length || !paymentRequest.total?.amount) {
      setMessage("Apple Pay request details are incomplete. Refresh the quote and try again.", "danger");
      return;
    }

    let session;
    let suppressCancelMessage = false;
    try {
      session = new window.ApplePaySession(3, paymentRequest);
    } catch (error) {
      setMessage(error.message || "Apple Pay could not be started on this device.", "danger");
      return;
    }

    state.applePaySession = session;
    state.phase = "authorizing";
    setBusy(true, "Authorizing");
    setMessage("Opening Apple Pay...", "muted");

    session.onvalidatemerchant = async (event) => {
      try {
        const data = await requestJson(
          config.merchant_session_url,
          { validation_url: event.validationURL, initiative_context: window.location.hostname },
          `wallet-apple-merchant-${state.order.order_id}`,
        );
        session.completeMerchantValidation(data.merchant_session);
      } catch (error) {
        suppressCancelMessage = true;
        setBusy(false, "Authorize Apple Pay");
        setMessage(error.message || "Apple Pay merchant validation failed.", "danger");
        session.abort();
      }
    };

    session.onpaymentauthorized = async (event) => {
      try {
        const data = await requestJson(
          config.authorize_url,
          { order_id: state.order.order_id, payment_token: event.payment?.token || {} },
          `wallet-apple-auth-${state.order.order_id}`,
        );
        state.order = data.order;
        renderQuote(data.order);
        session.completePayment(applePayStatus(true));
        setMessage("Payment captured. Fulfillment is pending treasury execution.", "success");
        setBusy(false, "Processing");
        pollStatus(true);
      } catch (error) {
        suppressCancelMessage = true;
        session.completePayment(applePayStatus(false));
        setBusy(false, "Authorize Apple Pay");
        setMessage(error.message || "Apple Pay authorization failed.", "danger");
      }
    };

    session.oncancel = () => {
      state.phase = "quoted";
      state.applePaySession = null;
      setBusy(false, "Authorize Apple Pay");
      if (!suppressCancelMessage) setMessage("Apple Pay was canceled before authorization.", "warning");
    };

    session.begin();
  };

  const authorize = async () => {
    if (!state.order?.order_id) {
      await quote();
      return;
    }
    const config = activeConfig();
    if (state.method !== "card") {
      authorizeApplePay();
      return;
    }
    const token = state.gatewayToken || tokenInput?.value || "";
    if (!token) {
      setMessage("Complete the hosted card fields before authorization.", "warning");
      return;
    }
    setBusy(true, "Authorizing");
    setMessage("Card authorization pending...", "muted");
    try {
      const data = await requestJson(
        config.authorize_url,
        { order_id: state.order.order_id, gateway_payment_token: { token } },
        `wallet-card-auth-${state.order.order_id}`,
      );
      state.order = data.order;
      renderQuote(data.order);
      setMessage("Payment captured. Fulfillment is pending treasury execution.", "success");
      setBusy(false, "Processing");
      pollStatus(true);
    } catch (error) {
      setBusy(false, "Authorize Card");
      setMessage(error.message || "Card authorization failed.", "danger");
    }
  };

  const pollStatus = (immediate = false) => {
    clearPolling();
    if (!state.order?.order_id) return;
    const template = activeConfig().status_url_template || "";
    if (!template) return;
    const terminal = new Set(["complete", "failed", "expired"]);
    const load = async () => {
      try {
        const response = await fetch(template.replace("__ORDER_ID__", encodeURIComponent(state.order.order_id)), {
          credentials: "same-origin",
          cache: "no-store",
        });
        const data = await response.json();
        if (!response.ok || data.ok === false) throw new Error(data.message || "Status check failed.");
        state.order = data.order;
        renderQuote(data.order);
        const status = String(data.order.status || "");
        if (status === "complete") {
          setMessage("Purchase complete. Wallet delivery is confirmed.", "success");
          setBusy(false, "Done");
          return;
        }
        if (status === "expired") {
          setMessage("Quote expired. Request a fresh quote before authorizing.", "warning");
          setBusy(false, "Review Quote");
          state.phase = "idle";
          return;
        }
        if (status === "failed") {
          setMessage(data.order.failure_reason || "Purchase failed. No wallet credit was applied.", "danger");
          setBusy(false, "Review Quote");
          return;
        }
        if (!terminal.has(status)) {
          state.pollTimer = window.setTimeout(load, 3000);
        }
      } catch (error) {
        setMessage("Network issue while checking status. The order remains server-tracked.", "warning");
        state.pollTimer = window.setTimeout(load, 5000);
      }
    };
    state.pollTimer = window.setTimeout(load, immediate ? 0 : 3000);
  };

  const openSheet = (method = "card") => {
    state.lastFocus = document.activeElement;
    sheet.hidden = false;
    document.documentElement.classList.add("wallet-onramp-open");
    setMethod(method);
    window.setTimeout(() => dialog?.focus(), 0);
  };

  const closeSheet = () => {
    clearPolling();
    sheet.hidden = true;
    document.documentElement.classList.remove("wallet-onramp-open");
    resetQuote();
    if (state.lastFocus && typeof state.lastFocus.focus === "function") state.lastFocus.focus();
  };

  document.querySelectorAll("[data-wallet-buy-open]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.hasAttribute("disabled")) return;
      openSheet(button.getAttribute("data-onramp-method") || "card");
    });
  });
  sheet.querySelectorAll("[data-wallet-onramp-close]").forEach((button) => button.addEventListener("click", closeSheet));
  methodButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (!button.disabled) setMethod(button.getAttribute("data-onramp-method") || "card");
    });
  });
  amountPresetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (amountInput) amountInput.value = button.getAttribute("data-onramp-amount-preset") || amountInput.value;
      resetQuote();
      setMessage("Amount updated. Request a fresh quote.", "muted");
    });
  });
  assetSelect?.addEventListener("change", () => {
    resetQuote();
    populateNetworks();
  });
  networkSelect?.addEventListener("change", () => {
    resetQuote();
    updateDestination();
  });
  amountInput?.addEventListener("input", () => {
    if (state.order) {
      resetQuote();
      setMessage("Amount changed. Request a fresh quote.", "muted");
    }
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.phase === "quoted") {
      authorize();
    } else {
      quote();
    }
  });
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (!data || typeof data !== "object") return;
    if (data.type === "algvault.card.tokenized" && data.token) {
      state.gatewayToken = String(data.token);
      if (gatewayState) gatewayState.textContent = "Card token received";
      setMessage("Card token received. You can authorize now.", "success");
    }
    if (data.type === "algvault.card.failed") {
      setMessage(String(data.message || "Card tokenization failed."), "danger");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (sheet.hidden) return;
    if (event.key === "Escape") closeSheet();
    if (event.key !== "Tab" || !dialog) return;
    const focusable = Array.from(dialog.querySelectorAll("button:not([disabled]), input:not([disabled]), select:not([disabled]), iframe, [tabindex]:not([tabindex='-1'])"));
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
})();
