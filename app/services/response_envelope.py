"""Small response envelopes for action and readiness endpoints."""

from __future__ import annotations

from typing import Any

from .failures import AlgVaultError


def action_envelope(
    *,
    ok: bool,
    code: str,
    message: str,
    blockers: list[Any] | None = None,
    details: dict[str, Any] | None = None,
    next_actions: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return the standard JSON envelope without dropping legacy fields."""

    payload: dict[str, Any] = {
        "ok": bool(ok),
        "code": str(code or ("ok" if ok else "error")),
        "message": str(message or ""),
        "blockers": list(blockers or []),
        "details": dict(details or {}),
        "next_actions": list(next_actions or []),
    }
    payload.update(extra)
    return payload


def readiness_envelope(
    payload: dict[str, Any],
    *,
    ready_key: str = "ready",
    code: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Overlay the standard envelope shape onto an existing readiness payload."""

    ready = bool(payload.get(ready_key, False))
    blockers = list(payload.get("active_blockers") or payload.get("blockers") or [])
    envelope = action_envelope(
        ok=ready,
        code=code or ("ready" if ready else "not_ready"),
        message=message or str(payload.get("message") or ("Ready." if ready else "Readiness checks did not pass.")),
        blockers=blockers,
        details={"readiness": dict(payload)},
    )
    return {**payload, **envelope}


def exception_envelope(
    exc: Exception,
    *,
    default_code: str,
    default_message: str | None = None,
    blockers: list[Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return the standard action envelope for typed or legacy exceptions."""

    if isinstance(exc, AlgVaultError):
        code = exc.code or default_code
        message = exc.message or default_message or str(exc)
        details = dict(exc.context or {})
    else:
        code = default_code
        message = default_message or str(exc)
        details = {"error_type": exc.__class__.__name__}
    return action_envelope(
        ok=False,
        code=code,
        message=message,
        blockers=blockers or [{"code": code, "description": message}],
        details=details,
        error=message,
        **extra,
    )
