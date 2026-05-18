"""KuCoin compliance helpers for live routing gates."""

from typing import Any

_RESTRICTED_REGION_ALIASES = {
    "BC": "British Columbia",
    "BRITISH COLUMBIA": "British Columbia",
    "BRITISH COLUMBIA CANADA": "British Columbia",
    "BRITISH COLUMBIA OF CANADA": "British Columbia",
    "ON": "Ontario",
    "ONTARIO": "Ontario",
    "ONTARIO CANADA": "Ontario",
    "ONTARIO OF CANADA": "Ontario",
    "US": "United States",
    "USA": "United States",
    "U S": "United States",
    "U S A": "United States",
    "UNITED STATES": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "PUERTO RICO": "Puerto Rico",
    "GUAM": "Guam",
    "NORTHERN MARIANA ISLANDS": "Northern Mariana Islands",
    "AMERICAN SAMOA": "American Samoa",
    "SINGAPORE": "Singapore",
    "MAINLAND CHINA": "mainland China",
    "CHINA": "mainland China",
    "HONG KONG": "Hong Kong",
    "MALAYSIA": "Malaysia",
    "FRANCE": "France",
    "NETHERLANDS": "Netherlands",
    "CRIMEA": "Crimea",
    "DONETSK": "Donetsk",
    "LUHANSK": "Luhansk",
    "ZAPORIZHZHIA": "Zaporizhzhia",
    "KHERSON": "Kherson",
}

_RESTRICTED_REGION_MARKERS = (
    ("BRITISH COLUMBIA", "British Columbia"),
    ("ONTARIO", "Ontario"),
    ("UNITED STATES", "United States"),
    ("PUERTO RICO", "Puerto Rico"),
    ("GUAM", "Guam"),
    ("NORTHERN MARIANA", "Northern Mariana Islands"),
    ("AMERICAN SAMOA", "American Samoa"),
    ("SINGAPORE", "Singapore"),
    ("MAINLAND CHINA", "mainland China"),
    ("HONG KONG", "Hong Kong"),
    ("MALAYSIA", "Malaysia"),
    ("FRANCE", "France"),
    ("NETHERLANDS", "Netherlands"),
    ("CRIMEA", "Crimea"),
    ("DONETSK", "Donetsk"),
    ("LUHANSK", "Luhansk"),
    ("ZAPORIZHZHIA", "Zaporizhzhia"),
    ("KHERSON", "Kherson"),
)


def kucoin_operator_region_status(config: dict[str, Any]) -> dict[str, Any]:
    """Return the configured KuCoin operator/account region and restriction status."""

    raw = str(config.get("KUCOIN_OPERATOR_REGION") or "").strip()
    if not raw:
        return {"configured": False, "region": "", "label": "", "restricted": False}
    normalized = _normalize_region(raw)
    label = _RESTRICTED_REGION_ALIASES.get(normalized)
    if label is None:
        label = next((restricted_label for marker, restricted_label in _RESTRICTED_REGION_MARKERS if marker in normalized), "")
    return {
        "configured": True,
        "region": raw,
        "label": label or raw,
        "restricted": bool(label),
    }


def _normalize_region(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() else " " for ch in value.upper())
    return " ".join(normalized.split())
