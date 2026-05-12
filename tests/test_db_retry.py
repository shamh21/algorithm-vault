from __future__ import annotations

from sqlalchemy.exc import OperationalError

from app.extensions import db
from app.models import Setting
from app.services.db_retry import write_with_retry


def test_write_with_retry_reapplies_operation_after_sqlite_lock(app, monkeypatch) -> None:
    calls = {"operation": 0, "commit": 0}
    original_commit = db.session.commit

    def flaky_commit() -> None:
        calls["commit"] += 1
        if calls["commit"] == 1:
            raise OperationalError("UPDATE setting SET value = ?", {}, Exception("database is locked"))
        original_commit()

    def operation() -> None:
        calls["operation"] += 1
        Setting.set_json("retry_marker", {"attempt": calls["operation"]})

    monkeypatch.setattr(db.session, "commit", flaky_commit)

    write_with_retry(operation, attempts=2, sleep_seconds=0.0)

    assert calls == {"operation": 2, "commit": 2}
    assert Setting.get_json("retry_marker") == {"attempt": 2}
