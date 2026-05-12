"""Compatibility module alias for consumer routes."""

from __future__ import annotations

import sys as _sys

from .consumer_parts import legacy as _legacy

consumer_bp = _legacy.consumer_bp
_sys.modules[__name__] = _legacy
