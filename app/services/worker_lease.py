"""Database-backed worker leases and idempotency helpers."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import AuditLog, WorkerJobRun, WorkerLease


def default_owner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"


def in_process_workers_enabled(config: dict[str, Any]) -> bool:
    mode = str(config.get("WORKER_MODE", "web") or "web").strip().lower()
    return mode in {"worker", "local", "dev", "test"} or bool(config.get("ENABLE_IN_PROCESS_WORKERS", False))


class WorkerLeaseService:
    """Small DB-backed lease coordinator for singletons and recurring jobs."""

    def __init__(self, config: dict[str, Any], owner_id: str | None = None) -> None:
        self.config = config
        self.owner_id = owner_id or default_owner_id()

    @staticmethod
    def strategy_run_lease_name(run_id: int) -> str:
        return f"strategy_run:{int(run_id)}"

    def acquire_strategy_run(
        self,
        run_id: int,
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerLease | None:
        ttl = int(ttl_seconds or self.config.get("STRATEGY_RUN_LEASE_TTL_SECONDS", 0) or self._default_strategy_ttl())
        return self.acquire(
            self.strategy_run_lease_name(run_id),
            ttl_seconds=ttl,
            metadata={"run_id": int(run_id), "lease_type": "strategy_run", **dict(metadata or {})},
        )

    def heartbeat_strategy_run(self, run_id: int, *, ttl_seconds: int | None = None) -> bool:
        ttl = int(ttl_seconds or self.config.get("STRATEGY_RUN_LEASE_TTL_SECONDS", 0) or self._default_strategy_ttl())
        return self.heartbeat(self.strategy_run_lease_name(run_id), ttl_seconds=ttl)

    def release_strategy_run(self, run_id: int, *, status: str = "released") -> None:
        self.release(self.strategy_run_lease_name(run_id), status=status)

    def _default_strategy_ttl(self) -> int:
        poll_seconds = float(self.config.get("STRATEGY_POLL_SECONDS", 20) or 20)
        return max(30, int(poll_seconds * 6))

    def acquire(
        self,
        lease_name: str,
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerLease | None:
        now = datetime.utcnow()
        ttl = max(5, int(ttl_seconds or self.config.get("WORKER_LEASE_TTL_SECONDS", 120) or 120))
        lease = WorkerLease.query.filter_by(lease_name=lease_name).one_or_none()
        if (
            lease is not None
            and lease.status == "active"
            and lease.expires_at
            and lease.expires_at > now
            and lease.owner_id != self.owner_id
        ):
            self._audit(
                "worker_lease_acquire_failed", lease_name, "blocked", {"owner_id": self.owner_id, "active_owner_id": lease.owner_id}
            )
            db.session.commit()
            return None
        if lease is None:
            lease = WorkerLease(lease_name=lease_name)
            db.session.add(lease)
        lease.owner_id = self.owner_id
        lease.acquired_at = now
        lease.heartbeat_at = now
        lease.expires_at = now + timedelta(seconds=ttl)
        lease.status = "active"
        lease.details = metadata or {}
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            self._audit("worker_lease_acquire_failed", lease_name, "race_lost", {"owner_id": self.owner_id})
            db.session.commit()
            return None
        self._audit("worker_lease_acquired", lease_name, "active", {"owner_id": self.owner_id, "ttl_seconds": ttl})
        db.session.commit()
        return lease

    def heartbeat(self, lease_name: str, *, ttl_seconds: int | None = None) -> bool:
        lease = WorkerLease.query.filter_by(lease_name=lease_name, owner_id=self.owner_id, status="active").one_or_none()
        if lease is None:
            return False
        now = datetime.utcnow()
        ttl = max(5, int(ttl_seconds or self.config.get("WORKER_LEASE_TTL_SECONDS", 120) or 120))
        lease.heartbeat_at = now
        lease.expires_at = now + timedelta(seconds=ttl)
        db.session.commit()
        return True

    def release(self, lease_name: str, *, status: str = "released") -> None:
        lease = WorkerLease.query.filter_by(lease_name=lease_name, owner_id=self.owner_id).one_or_none()
        if lease is None:
            return
        lease.status = status
        lease.expires_at = datetime.utcnow()
        self._audit("worker_lease_released", lease_name, status, {"owner_id": self.owner_id})
        db.session.commit()

    def start_job(
        self,
        job_name: str,
        idempotency_key: str,
        *,
        lease_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, WorkerJobRun]:
        existing = WorkerJobRun.query.filter_by(idempotency_key=idempotency_key).one_or_none()
        if existing is not None and existing.status in {"running", "complete"}:
            return False, existing
        if existing is not None:
            existing.status = "running"
            existing.failure_reason = None
            existing.started_at = datetime.utcnow()
            existing.completed_at = None
            existing.owner_id = self.owner_id
            existing.details = metadata or existing.details
            db.session.commit()
            return True, existing
        run = WorkerJobRun(
            job_name=job_name,
            idempotency_key=idempotency_key,
            lease_name=lease_name,
            owner_id=self.owner_id,
        )
        run.details = metadata or {}
        db.session.add(run)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            existing = WorkerJobRun.query.filter_by(idempotency_key=idempotency_key).one()
            return False, existing
        return True, run

    def complete_job(self, run: WorkerJobRun, metadata: dict[str, Any] | None = None) -> None:
        run.status = "complete"
        run.completed_at = datetime.utcnow()
        if metadata is not None:
            run.details = metadata
        self._audit("worker_job_complete", run.job_name, "complete", {"idempotency_key": run.idempotency_key})
        db.session.commit()

    def fail_job(self, run: WorkerJobRun, exc: Exception, metadata: dict[str, Any] | None = None) -> None:
        run.status = "failed"
        run.completed_at = datetime.utcnow()
        run.failure_reason = str(exc)
        if metadata is not None:
            run.details = metadata
        self._audit(
            "worker_job_failed",
            run.job_name,
            "failed",
            {"idempotency_key": run.idempotency_key, "error_type": exc.__class__.__name__, "error": str(exc)},
        )
        db.session.commit()

    def run_leased_job(
        self,
        job_name: str,
        fn: Callable[[], dict[str, Any] | list[Any] | None],
        *,
        lease_name: str | None = None,
        idempotency_key: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        lease_key = lease_name or job_name
        lease = self.acquire(lease_key, ttl_seconds=ttl_seconds, metadata={"job_name": job_name})
        if lease is None:
            return {"ok": False, "status": "lease_blocked", "job_name": job_name, "lease_name": lease_key}
        should_run, run = self.start_job(job_name, idempotency_key, lease_name=lease_key)
        if not should_run:
            self.release(lease_key)
            return {"ok": True, "status": "duplicate", "job_name": job_name, "idempotency_key": idempotency_key}
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            self.fail_job(run, exc)
            self.release(lease_key, status="failed")
            raise
        payload = {"result": result}
        self.complete_job(run, payload)
        self.release(lease_key)
        return {"ok": True, "status": "complete", "job_name": job_name, "idempotency_key": idempotency_key, "result": result}

    def _audit(self, action: str, lease_name: str, status: str, details: dict[str, Any]) -> None:
        audit = AuditLog(category="worker", action=action, message=f"Worker {action.replace('_', ' ')} for {lease_name}.")
        audit.details = {"lease_name": lease_name, "status": status, **details}
        db.session.add(audit)
