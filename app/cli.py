"""Compatibility module alias for AlgVault Flask CLI commands."""

from __future__ import annotations

import sys as _sys

from .cli_commands import legacy as _legacy

register_cli = _legacy.register_cli
_sys.modules[__name__] = _legacy
