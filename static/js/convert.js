(() => {
  const form = document.querySelector("[data-convert-form]");
  if (!form) return;

  const widget = document.querySelector("[data-convert-widget]");
  const offlineBanner = document.querySelector("[data-convert-offline]");
  const submitButton = form.querySelector("[data-convert-submit]");
  const maxButton = form.querySelector("[data-convert-max]");
  const swapButton = form.querySelector("[data-convert-swap]");
  const amountInput = form.querySelector("#convert-amount");
  const sourceSelect = form.querySelector("#convert-from-asset");
  const destinationSelect = form.querySelector("#convert-to-asset");
  const sourceMeta = form.querySelector("[data-convert-from-meta]");
  const destinationMeta = form.querySelector("[data-convert-to-meta]");
  const amountMeta = form.querySelector("[data-convert-amount-meta]");
  const preview = form.querySelector("[data-convert-preview]");
  const previewState = form.querySelector("[data-convert-preview-state]");
  const previewTitle = form.querySelector("[data-convert-preview-title]");
  const previewDetail = form.querySelector("[data-convert-preview-detail]");
  const previewDetails = form.querySelector("[data-convert-preview-details]");
  const previewValue = form.querySelector("[data-convert-preview-value]");
  const previewRate = form.querySelector("[data-convert-preview-rate]");
  const errors = {
    from_asset: form.querySelector('[data-convert-error="from_asset"]'),
    to_asset: form.querySelector('[data-convert-error="to_asset"]'),
    amount: form.querySelector('[data-convert-error="amount"]'),
  };
  const canConvert = form.dataset.canConvert === "true";
  let submitting = false;

  const selectedSourceOption = () => sourceSelect?.selectedOptions?.[0] || null;
  const selectedDestinationOption = () => destinationSelect?.selectedOptions?.[0] || null;
  const parseAmount = (value) => Number.parseFloat(String(value || "").replace(/,/g, "").trim());
  const finiteNumber = (value) => (Number.isFinite(value) ? value : 0);
  const formatAssetAmount = (value) => finiteNumber(value).toFixed(8);
  const trimAssetAmount = (value) => formatAssetAmount(value).replace(/0+$/, "").replace(/\.$/, "") || "0";
  const formatUsd = (value, places = 2) => `$${finiteNumber(value).toFixed(places)}`;

  const setFieldError = (name, message = "") => {
    const fieldError = errors[name];
    const control = name === "amount" ? amountInput : name === "from_asset" ? sourceSelect : destinationSelect;
    if (fieldError) {
      fieldError.textContent = message;
      fieldError.toggleAttribute("aria-hidden", !message);
    }
    control?.setAttribute("aria-invalid", message ? "true" : "false");
  };

  const clearFieldErrors = () => {
    Object.keys(errors).forEach((name) => setFieldError(name));
  };

  const setButtonState = (disabled) => {
    if (!submitButton) return;
    submitButton.disabled = disabled;
    submitButton.setAttribute("aria-disabled", disabled ? "true" : "false");
  };

  const renderState = (kind, title, detail, options = {}) => {
    if (!previewState || !previewTitle || !previewDetail) return;
    previewState.className = `convert-state-panel is-${kind}`;
    previewState.dataset.convertPreviewState = "";
    if (kind === "quote") {
      previewTitle.textContent = "You receive";
      previewDetail.textContent = detail;
    } else {
      previewTitle.textContent = title;
      previewDetail.textContent = detail;
    }
    if (previewDetails) previewDetails.hidden = kind !== "quote";
    if (previewValue && options.valueText) previewValue.textContent = options.valueText;
    if (previewRate && options.rateText) previewRate.textContent = options.rateText;
    if (preview) preview.setAttribute("aria-busy", kind === "loading" ? "true" : "false");
  };

  const updateAssetMeta = () => {
    const sourceOption = selectedSourceOption();
    const destinationOption = selectedDestinationOption();
    if (sourceOption) {
      const available = parseAmount(sourceOption.dataset.available);
      const price = parseAmount(sourceOption.dataset.price);
      if (sourceMeta) sourceMeta.textContent = `Available ${formatAssetAmount(available)} ${sourceOption.value}`;
      if (amountMeta) {
        amountMeta.textContent = sourceOption.dataset.priceAvailable === "true"
          ? `Source price ${formatUsd(price, 4)}`
          : "Source price unavailable";
      }
    }
    if (destinationOption && destinationMeta) {
      const price = parseAmount(destinationOption.dataset.price);
      destinationMeta.textContent = destinationOption.dataset.priceAvailable === "true"
        ? `Reference price ${formatUsd(price, 4)}`
        : "Reference price unavailable";
    }
  };

  const updateMaxButton = () => {
    const option = selectedSourceOption();
    if (!option || !maxButton) return;
    const available = parseAmount(option.dataset.available);
    const disabled = !Number.isFinite(available) || available <= 0;
    maxButton.disabled = disabled;
    maxButton.setAttribute("aria-disabled", disabled ? "true" : "false");
    maxButton.title = disabled ? "No available balance" : `Use ${formatAssetAmount(available)} ${option.value}`;
  };

  const validateAndRender = ({ forceError = false } = {}) => {
    const offline = navigator.onLine === false;
    const sourceOption = selectedSourceOption();
    const destinationOption = selectedDestinationOption();
    const amountText = amountInput?.value?.trim() || "";
    const amount = parseAmount(amountText);
    const sourceAsset = sourceOption?.value || "";
    const destinationAsset = destinationOption?.value || "";
    const sourcePrice = parseAmount(sourceOption?.dataset.price);
    const destinationPrice = parseAmount(destinationOption?.dataset.price);
    const available = parseAmount(sourceOption?.dataset.available);
    const sourcePriceReady = sourceOption?.dataset.priceAvailable === "true";
    const destinationPriceReady = destinationOption?.dataset.priceAvailable === "true";

    clearFieldErrors();
    updateAssetMeta();
    updateMaxButton();
    offlineBanner.hidden = !offline;
    widget?.classList.toggle("is-offline", offline);

    if (submitting) {
      setButtonState(true);
      renderState("loading", "Submitting conversion", "Final balance and pricing checks are running.");
      return false;
    }

    if (offline) {
      setButtonState(true);
      renderState("stale", "Network unavailable", "Reconnect before submitting a conversion.");
      return false;
    }

    if (!canConvert) {
      setButtonState(true);
      renderState("empty", "No convertible balance", "Deposit or free an available wallet balance before converting.");
      return false;
    }

    if (!sourceOption || !destinationOption) {
      setButtonState(true);
      renderState("error", "Asset unavailable", "Choose a supported source and destination asset.");
      if (!sourceOption) setFieldError("from_asset", "Choose a supported source asset.");
      if (!destinationOption) setFieldError("to_asset", "Choose a supported destination asset.");
      return false;
    }

    if (sourceAsset === destinationAsset) {
      setButtonState(true);
      setFieldError("to_asset", "Choose two different assets.");
      renderState("error", "Review required", "Choose two different assets before converting.");
      return false;
    }

    if (!sourcePriceReady || sourcePrice <= 0) {
      setButtonState(true);
      setFieldError("from_asset", `${sourceAsset} price is unavailable.`);
      renderState("stale", "Price unavailable", "Choose assets with current reference pricing.");
      return false;
    }

    if (!destinationPriceReady || destinationPrice <= 0) {
      setButtonState(true);
      setFieldError("to_asset", `${destinationAsset} price is unavailable.`);
      renderState("stale", "Price unavailable", "Choose assets with current reference pricing.");
      return false;
    }

    if (!amountText) {
      setButtonState(true);
      if (forceError) setFieldError("amount", "Enter an amount greater than zero.");
      renderState("default", "Enter an amount", "Choose assets and enter an amount to preview the conversion.");
      return false;
    }

    if (!Number.isFinite(amount) || amount <= 0) {
      setButtonState(true);
      setFieldError("amount", "Enter an amount greater than zero.");
      renderState("error", "Review required", "Enter an amount greater than zero.");
      return false;
    }

    if (!Number.isFinite(available) || amount > available + 0.000000000001) {
      setButtonState(true);
      setFieldError("amount", `Amount exceeds available ${sourceAsset} balance.`);
      renderState("error", "Review required", `Available balance is ${formatAssetAmount(available)} ${sourceAsset}.`);
      return false;
    }

    const usdValue = amount * sourcePrice;
    const convertedAmount = usdValue / destinationPrice;
    const rate = sourcePrice / destinationPrice;
    if (!Number.isFinite(convertedAmount) || convertedAmount <= 0) {
      setButtonState(true);
      setFieldError("amount", "Conversion amount is too small.");
      renderState("error", "Review required", "Enter a larger amount to preview this conversion.");
      return false;
    }

    setButtonState(false);
    renderState("quote", "You receive", `${formatAssetAmount(convertedAmount)} ${destinationAsset}`, {
      valueText: formatUsd(usdValue),
      rateText: `1 ${sourceAsset} = ${formatAssetAmount(rate)} ${destinationAsset}`,
    });
    return true;
  };

  maxButton?.addEventListener("click", () => {
    const option = selectedSourceOption();
    const available = parseAmount(option?.dataset.available);
    if (!amountInput || !Number.isFinite(available) || available <= 0) return;
    amountInput.value = trimAssetAmount(available);
    amountInput.focus({ preventScroll: true });
    validateAndRender();
  });

  swapButton?.addEventListener("click", () => {
    if (!sourceSelect || !destinationSelect) return;
    const sourceValue = sourceSelect.value;
    sourceSelect.value = destinationSelect.value;
    destinationSelect.value = sourceValue;
    validateAndRender();
    sourceSelect.focus({ preventScroll: true });
  });

  sourceSelect?.addEventListener("change", validateAndRender);
  destinationSelect?.addEventListener("change", validateAndRender);
  amountInput?.addEventListener("input", validateAndRender);
  amountInput?.addEventListener("blur", () => validateAndRender({ forceError: true }));

  form.addEventListener(
    "submit",
    (event) => {
      if (!validateAndRender({ forceError: true })) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (navigator.onLine === false) {
          offlineBanner?.focus?.();
        } else {
          const invalidControl = form.querySelector('[aria-invalid="true"]');
          if (invalidControl instanceof HTMLElement) {
            invalidControl.focus({ preventScroll: true });
          } else {
            amountInput?.focus({ preventScroll: true });
          }
        }
        return;
      }
      submitting = true;
      widget?.classList.add("is-submitting");
      validateAndRender();
    },
    true
  );

  window.addEventListener("online", () => validateAndRender(), { passive: true });
  window.addEventListener("offline", () => validateAndRender(), { passive: true });
  validateAndRender();
})();
