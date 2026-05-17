(() => {
  const TABLE_BREAKPOINT = 760;
  const STACK_CLASS = "is-stacked";
  const WIDE_CLASS = "is-wide-table";
  const tableCache = new WeakMap();
  const tableSelectors = Array.from(document.querySelectorAll("table")).filter((table) => {
    return !table.classList.contains("no-responsive-table");
  });

  if (!tableSelectors.length) return;

  const numericRegex = /^[\s\$€]?[+-]?(?:(?:\d{1,3}(?:,\d{3})*)|\d*)(?:\.\d+)?%?$/;

  const configuredHeaders = (table) => {
    return String(table.dataset.responsiveHeaders || table.dataset.tableHeaders || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  };

  const headerCellsForTable = (table) => {
    const theadHeaders = Array.from(table.tHead?.querySelectorAll("th") ?? []);
    if (theadHeaders.length) return { headers: theadHeaders, hasBodyHeader: false };
    const firstBodyRow = table.tBodies?.[0]?.rows?.[0];
    const bodyHeaders = Array.from(firstBodyRow?.children ?? []).filter((cell) => cell.tagName === "TH");
    return { headers: bodyHeaders, hasBodyHeader: bodyHeaders.length > 0 };
  };

  const labelFromCell = (headerLabels, index) => {
    if (index < headerLabels.length) {
      return headerLabels[index] || `Column ${index + 1}`;
    }
    return `Column ${index + 1}`;
  };

  const markRows = (table) => {
    const { headers, hasBodyHeader } = headerCellsForTable(table);
    const headerLabels = headers.length
      ? headers.map((cell, index) => cell.textContent.trim() || `Column ${index + 1}`)
      : configuredHeaders(table);
    const rows = Array.from(table.tBodies ?? []).flatMap((tbody) => Array.from(tbody.rows));
    const dataRows = hasBodyHeader ? rows.slice(1) : rows;
    const columnCount = Math.max(
      headerLabels.length,
      ...dataRows.map((row) => row.children.length),
      0
    );
    const signature = `${headerLabels.join("|")}::${dataRows.length}::${columnCount}`;
    const forcedMode = table.dataset.tableMode || "";
    const isWideTable =
      forcedMode === "scroll" ||
      table.classList.contains(WIDE_CLASS) ||
      (forcedMode !== "stack" && columnCount > 5);

    table.classList.toggle(WIDE_CLASS, isWideTable);
    table.classList.toggle("has-responsive-labels", headerLabels.length > 0);
    table.classList.toggle("has-body-header", hasBodyHeader);

    if (tableCache.get(table) === signature) return isWideTable;
    tableCache.set(table, signature);

    dataRows.forEach((row) => {
      Array.from(row.children).forEach((cell, index) => {
        if (!(cell instanceof HTMLTableCellElement)) return;
        const label = labelFromCell(headerLabels, index);
        cell.dataset.label = label;

        const value = String(cell.textContent ?? "").trim();

        if (value && numericRegex.test(value.replace(/\s/g, ""))) {
          cell.classList.add("is-numeric");
        } else {
          cell.classList.remove("is-numeric");
        }

        if (label && !cell.hasAttribute("aria-label")) {
          cell.setAttribute("aria-label", value ? `${label}: ${value}` : label);
        }
      });
    });

    return isWideTable;
  };

  const applyMode = () => {
    const shouldStack = window.matchMedia(`(max-width: ${TABLE_BREAKPOINT}px)`).matches;

    tableSelectors.forEach((table) => {
      const isWideTable = markRows(table);
      table.classList.toggle(STACK_CLASS, shouldStack && !isWideTable);
    });
  };

  const mediaQuery = window.matchMedia(`(max-width: ${TABLE_BREAKPOINT}px)`);

  applyMode();

  if (mediaQuery.addEventListener) {
    mediaQuery.addEventListener("change", applyMode);
  } else {
    mediaQuery.addListener(applyMode);
  }
})();
