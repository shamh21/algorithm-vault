"""Compatibility module alias for vault strategy selector service."""

from __future__ import annotations

import sys as _sys

from .vault_selector_parts import legacy as _legacy

VaultStrategySelector = _legacy.VaultStrategySelector
_sys.modules[__name__] = _legacy
