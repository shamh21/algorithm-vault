(() => {
  const TABLE_BREAKPOINT = 760;
  const STACK_CLASS = "is-stacked";
  const tableSelectors = Array.from(document.querySelectorAll("table")).filter((table) => {
    return !table.classList.contains("no-responsive-table");
  });

  if (!tableSelectors.length) return;

  const numericRegex = /^[\s\$€]?[+-]?(?:(?:\d{1,3}(?:,\d{3})*)|\d*)(?:\.\d+)?%?$/;

  const labelFromCell = (thCells, index) => {
    if (index < thCells.length) {
      return thCells[index].textContent.trim() || `Column ${index + 1}`;
    }
    return `Column ${index + 1}`;
  };

  const markRows = (table) => {
    const headerCells = Array.from(table.tHead?.querySelectorAll("th") ?? []);
    const rows = Array.from(table.tBodies?.[0]?.querySelectorAll("tr") ?? []);

    rows.forEach((row) => {
      Array.from(row.children).forEach((cell, index) => {
        if (!(cell instanceof HTMLTableCellElement)) return;
        cell.dataset.label = labelFromCell(headerCells, index);

        const value = String(cell.textContent ?? "").trim();

        if (numericRegex.test(value.replace(/\s/g, ""))) {
          cell.classList.add("is-numeric");
        }
      });
    });
  };

  const applyMode = () => {
    const shouldStack = window.matchMedia(`(max-width: ${TABLE_BREAKPOINT}px)`).matches;

    tableSelectors.forEach((table) => {
      table.classList.toggle(STACK_CLASS, shouldStack);
      markRows(table);
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
