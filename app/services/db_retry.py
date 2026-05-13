"""Small retry helpers for transient SQLite write locks."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.exc import OperationalError

from ..extensions import db


T = TypeVar("T")


def is_database_locked(exc: BaseException) -> bool:
    return isinstance(exc, OperationalError) and "database is locked" in str(exc).lower()


def commit_with_retry(
    *,
    attempts: int = 8,
    sleep_seconds: float = 0.12,
) -> None:
    """Commit the current session, retrying transient SQLite lock failures."""

    last_error: OperationalError | None = None
    for attempt in range(max(1, attempts)):
        try:
            db.session.commit()
            return
        except OperationalError as exc:
            if not is_database_locked(exc) or attempt >= attempts - 1:
                raise
            last_error = exc
            db.session.rollback()
            time.sleep(sleep_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error


def write_with_retry(
    operation: Callable[[], T],
    *,
    attempts: int = 8,
    sleep_seconds: float = 0.12,
) -> T:
    """Run a DB mutation function and commit it, reapplying on lock retry."""

    last_error: OperationalError | None = None
    for attempt in range(max(1, attempts)):
        try:
            result = operation()
            db.session.commit()
            return result
        except OperationalError as exc:
            if not is_database_locked(exc) or attempt >= attempts - 1:
                raise
            last_error = exc
            db.session.rollback()
            time.sleep(sleep_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("database write retry failed without an exception")
