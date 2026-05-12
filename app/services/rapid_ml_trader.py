"""Compatibility module alias for rapid ML trader service."""

from __future__ import annotations

import sys as _sys

from .rapid_ml_trader_parts import legacy as _legacy

RapidMLTraderService = _legacy.RapidMLTraderService
_sys.modules[__name__] = _legacy
