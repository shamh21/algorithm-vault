"""Audit event pagination, retention, and UI classification helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from ..extensions import db
from ..models import AuditLog

AUDIT_EVENT_PAGE_SIZE = 5
AUDIT_EVENT_RETENTION_LIMIT = 50
AUDIT_EVENT_UI_LIMIT = 50

_LISTENER_REGISTERED = False


@dataclass(frozen=True)
class AuditEventPage:
    events: list[dict[str, Any]]
    records: list[AuditLog]
    page: int
    page_size: int
    total: int
    total_pages: int
    has_prev: bool
    has_next: bool
    prev_page: int | None
    next_page: int | None
    page_numbers: list[int]


def register_audit_retention_listener() -> None:
    """Mark audit retention initialized without pruning operational history."""

    global _LISTENER_REGISTERED
    if _LISTENER_REGISTERED:
        return
    _LISTENER_REGISTERED = True


def get_audit_events_page(page: Any, *, page_size: int = AUDIT_EVENT_PAGE_SIZE) -> AuditEventPage:
    """Return one bounded page of newest-first audit events."""

    safe_page_size = max(1, int(page_size or AUDIT_EVENT_PAGE_SIZE))
    total = int(db.session.scalar(select(func.count(AuditLog.id))) or 0)
    visible_total = min(total, AUDIT_EVENT_UI_LIMIT)
    total_pages = max(1, math.ceil(visible_total / safe_page_size))
    current_page = min(max(_safe_page(page), 1), total_pages)
    offset = (current_page - 1) * safe_page_size

    records = AuditLog.query.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(safe_page_size).offset(offset).all()

    max_visible_pages = max(1, math.ceil(AUDIT_EVENT_RETENTION_LIMIT / safe_page_size))
    return AuditEventPage(
        events=[audit_event_view(record) for record in records],
        records=records,
        page=current_page,
        page_size=safe_page_size,
        total=visible_total,
        total_pages=total_pages,
        has_prev=current_page > 1,
        has_next=current_page < total_pages,
        prev_page=current_page - 1 if current_page > 1 else None,
        next_page=current_page + 1 if current_page < total_pages else None,
        page_numbers=_page_numbers(current_page, total_pages, max_visible_pages),
    )


def prune_audit_events(
    *,
    retention_limit: int = AUDIT_EVENT_RETENTION_LIMIT,
    connection: Connection | None = None,
) -> int:
    """Delete audit records outside the newest retention window."""

    table = AuditLog.__table__
    safe_limit = max(0, int(retention_limit or 0))
    executor = connection if connection is not None else db.session
    total = int(executor.scalar(select(func.count()).select_from(table)) or 0)
    if total <= safe_limit:
        return 0
    if safe_limit <= 0:
        result = executor.execute(table.delete())
        return max(0, int(result.rowcount or 0))

    keep_ids = select(table.c.id).order_by(table.c.created_at.desc(), table.c.id.desc()).limit(safe_limit).subquery()
    delete_old = table.delete().where(~table.c.id.in_(select(keep_ids.c.id)))
    result = executor.execute(delete_old)
    return max(0, int(result.rowcount or 0))


def classify_audit_event(audit: AuditLog) -> str:
    """Return a compact visual tone for an audit event."""

    category = str(audit.category or "").lower()
    action = str(audit.action or "").lower()
    message = str(audit.message or "").lower()
    details = audit.details
    detail_text = " ".join(str(value).lower() for value in details.values())
    text = f"{category} {action} {message} {detail_text}"

    if action == "provider_runtime_backoff" or "runtime_backoff" in text or "backoff" in text:
        return "runtime-backoff"
    if action == "no_trade" or "no_trade" in text:
        return "no-trade"
    if "provider" in text and any(token in text for token in ("error", "failed", "failure", "429", "rate limit", "unavailable")):
        return "provider-error"
    if any(token in text for token in ("error", "failed", "failure", "exception", "blocked", "panic")):
        return "error"
    if any(token in text for token in ("warning", "warn", "rejected", "disabled", "limited", "reset")):
        return "warning"
    if any(token in text for token in ("success", "passed", "verified", "activated", "enabled", "saved", "completed", "cleared")):
        return "success"
    return "info"


def audit_event_view(audit: AuditLog) -> dict[str, Any]:
    details = audit.details
    tone = classify_audit_event(audit)
    diagnostics_json = json.dumps(details, sort_keys=True, default=str, indent=2)
    return {
        "id": audit.id,
        "tone": tone,
        "severity": _severity(tone),
        "icon_key": _icon_key(tone, audit.category, audit.action),
        "tone_label": _tone_label(tone),
        "category_label": _label(audit.category),
        "action_label": _label(audit.action),
        "title": _label(audit.action),
        "message": _compact_value(audit.message, 180),
        "created_iso": _created_iso(audit.created_at),
        "created_label": audit.created_at.strftime("%m-%d %H:%M") if audit.created_at else "",
        "time_label": audit.created_at.strftime("%H:%M") if audit.created_at else "",
        "date_label": audit.created_at.strftime("%b %d, %Y") if audit.created_at else "Unknown Date",
        "provider_badge": _provider_badge(audit, details),
        "chips": _diagnostic_chips(audit, details),
        "details_items": _details_items(details),
        "diagnostics_json": diagnostics_json,
        "copy_text": f"{audit.category}:{audit.action} {audit.message}\n{diagnostics_json}",
    }


def _safe_page(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _page_numbers(current_page: int, total_pages: int, max_visible_pages: int) -> list[int]:
    if total_pages <= max_visible_pages:
        return list(range(1, total_pages + 1))
    half_window = max_visible_pages // 2
    start = max(1, current_page - half_window)
    end = start + max_visible_pages - 1
    if end > total_pages:
        end = total_pages
        start = max(1, end - max_visible_pages + 1)
    return list(range(start, end + 1))


def _label(value: Any) -> str:
    text = str(value or "unknown").replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "Unknown"


def _tone_label(tone: str) -> str:
    return {
        "runtime-backoff": "Backoff",
        "no-trade": "No Trade",
        "provider-error": "Provider",
        "error": "Error",
        "warning": "Warning",
        "success": "Success",
        "info": "Info",
    }.get(tone, "Info")


def _severity(tone: str) -> str:
    if tone in {"error", "provider-error"}:
        return "error"
    if tone in {"warning", "runtime-backoff", "no-trade"}:
        return "warning"
    if tone == "success":
        return "success"
    return "info"


def _icon_key(tone: str, category: Any, action: Any) -> str:
    text = f"{tone} {category} {action}".lower()
    if "panic" in text or tone in {"error", "provider-error"}:
        return "alert"
    if "provider" in text or "connection" in text:
        return "plug"
    if "wallet" in text:
        return "wallet"
    if "risk" in text:
        return "shield"
    if tone == "success":
        return "check"
    return "pulse"


def _provider_badge(audit: AuditLog, details: dict[str, Any]) -> str:
    provider = details.get("provider") or details.get("execution_venue")
    if provider:
        return _label(provider)
    if audit.trading_connection_id:
        return f"Connection {audit.trading_connection_id}"
    return _label(audit.category)


def _created_iso(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _diagnostic_chips(audit: AuditLog, details: dict[str, Any]) -> list[str]:
    chips: list[str] = []
    for key in ("provider", "symbol", "timeframe", "blocker_category", "no_trade_reason", "run_id"):
        value = details.get(key)
        if value not in (None, ""):
            chips.append(f"{_label(key)}: {_compact_value(value, 42)}")
    if audit.trading_connection_id:
        chips.append(f"Connection: {audit.trading_connection_id}")
    if audit.user_id:
        chips.append(f"User: {audit.user_id}")
    return chips[:5]


def _details_items(details: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for key in sorted(details):
        value = details.get(key)
        if value in (None, ""):
            continue
        items.append({"label": _label(key), "value": _compact_value(value, 140)})
        if len(items) >= 8:
            break
    return items


def _compact_value(value: Any, limit: int) -> str:
    text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list, tuple)) else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}..."
