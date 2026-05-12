"""Compatibility module alias for strategy runner service."""

from __future__ import annotations

import sys as _sys

from .strategy_runner_parts import legacy as _legacy

StrategyManager = _legacy.StrategyManager
_sys.modules[__name__] = _legacy
