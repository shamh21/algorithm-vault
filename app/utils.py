"""Small presentation helpers shared by routes and templates."""

from __future__ import annotations

from typing import Any


def format_duration_seconds(value: Any) -> str:
    """Format seconds as compact human-readable duration text."""

    try:
        total_seconds = int(float(value or 0))
    except (TypeError, ValueError):
        total_seconds = 0

    total_seconds = max(0, total_seconds)
    if total_seconds == 0:
        return "0m"

    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds and not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts[:3])
