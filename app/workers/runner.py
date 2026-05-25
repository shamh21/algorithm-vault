"""Dedicated worker runner for recurring AlgVault jobs."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from datetime import datetime
from typing import Any

from flask import current_app

from app import create_app
from app.extensions import db
from app.models import StrategyRun, WalletAddress, WalletTransaction
from app.runtime import get_service
from app.services.db_retry import commit_with_retry
from app.services.wallet_activity import DEFAULT_ACTIVITY_RETENTION_LIMIT
from app.services.worker_lease import WorkerLeaseService, default_owner_id

logger = logging.getLogger(__name__)


def run_worker(
    *, once: bool = False, interval_seconds: int | None = None, owner_id: str | None = None, job_filter: set[str] | None = None
) -> list[dict[str, Any]]:
    app = create_app({"WORKER_MODE": "worker", "WORKER_PROCESS_CONFIGURED": True})
    owner = owner_id or default_owner_id()
    poll = max(1, int(interval_seconds or app.config.get("WORKER_POLL_SECONDS", 15) or 15))
    stop = {"requested": False}

    def _stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    results: list[dict[str, Any]] = []
    with app.app_context():
        lease_service = WorkerLeaseService(app.config, owner_id=owner)
        while not stop["requested"]:
            cycle_results = _run_due_jobs(lease_service, job_filter=job_filter)
            results.extend(cycle_results)
            db.session.remove()
            if once:
                break
            time.sleep(poll)
    return results


def _run_due_jobs(lease_service: WorkerLeaseService, *, job_filter: set[str] | None = None) -> list[dict[str, Any]]:
    jobs = {
        "strategy_starter": _strategy_starter_job,
        "vault_cycle_enforcement": _vault_cycle_enforcement_job,
        "treasury_solvency": _treasury_solvency_job,
        "wallet_activity_retention": _wallet_activity_retention_job,
        "wallet_custody_sync": _wallet_custody_sync_job,
    }
    config = lease_service.config
    enabled = {
        "strategy_starter": bool(config.get("WORKER_STRATEGY_STARTER_ENABLED", True)),
        "vault_cycle_enforcement": bool(config.get("WORKER_VAULT_ENFORCEMENT_ENABLED", True)),
        "treasury_solvency": bool(config.get("WORKER_TREASURY_SOLVENCY_ENABLED", True)),
        "wallet_activity_retention": bool(config.get("WORKER_WALLET_ACTIVITY_RETENTION_ENABLED", True)),
        "wallet_custody_sync": bool(config.get("WORKER_WALLET_CUSTODY_SYNC_ENABLED", True)),
    }
    results: list[dict[str, Any]] = []
    for name, fn in jobs.items():
        if job_filter and name not in job_filter:
            continue
        if not enabled.get(name, False):
            continue
        bucket = int(time.time() // _job_interval_seconds(config, name))
        try:
            results.append(
                lease_service.run_leased_job(
                    name,
                    fn,
                    lease_name=f"{name}:singleton",
                    idempotency_key=f"{name}:{bucket}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker job failed: %s", name)
            results.append({"ok": False, "status": "failed", "job_name": name, "error": str(exc)})
    return results


def _job_interval_seconds(config: dict[str, Any], name: str) -> int:
    specific_keys = {
        "wallet_activity_retention": "WORKER_WALLET_ACTIVITY_RETENTION_INTERVAL_SECONDS",
        "wallet_custody_sync": "WORKER_WALLET_CUSTODY_SYNC_INTERVAL_SECONDS",
    }
    key = specific_keys.get(name, "WORKER_POLL_SECONDS")
    return max(1, int(config.get(key, config.get("WORKER_POLL_SECONDS", 15)) or 15))


def _strategy_starter_job() -> dict[str, Any]:
    manager = get_service("strategy_manager")
    runs = (
        StrategyRun.query.filter(StrategyRun.manual_enabled.is_(True))
        .filter(StrategyRun.status.in_(["queued", "starting", "running"]))
        .order_by(StrategyRun.updated_at.asc(), StrategyRun.id.asc())
        .limit(100)
        .all()
    )
    started: list[int] = []
    lease_blocked: list[int] = []
    for run in runs:
        started_result = manager.start(int(run.id))
        if started_result is False:
            lease_blocked.append(int(run.id))
            continue
        started.append(int(run.id))
    return {"started_run_ids": started, "started_count": len(started), "lease_blocked_run_ids": lease_blocked}


def _vault_cycle_enforcement_job() -> dict[str, Any]:
    enforcement_results = get_service("vault_cycle_trading_enforcer").enforce_active_cycles(None)
    settlement_results = get_service("vault_cycle_orchestrator").resume_due_cycles(None)
    if settlement_results:
        commit_with_retry()
    return {
        "cycle_count": len(enforcement_results),
        "cycles": enforcement_results,
        "settlement_count": len(settlement_results),
        "settlements": settlement_results,
    }


def _treasury_solvency_job() -> dict[str, Any]:
    result = get_service("platform_treasury").process_solvency_cycle()
    return dict(result or {})


def _wallet_activity_retention_job() -> dict[str, Any]:
    service = get_service("wallet_activity")
    limit = max(1, int(current_app.config.get("WORKER_WALLET_ACTIVITY_RETENTION_LIMIT", DEFAULT_ACTIVITY_RETENTION_LIMIT) or 1))
    user_ids = [
        int(row[0])
        for row in db.session.query(WalletTransaction.user_id).filter(WalletTransaction.user_id.isnot(None)).distinct().all()
        if row[0] is not None
    ]
    pruned_by_user: dict[int, int] = {}
    total_pruned = 0
    for user_id in user_ids:
        pruned = int(service.prune_user_activity(user_id, limit=limit) or 0)
        if pruned:
            pruned_by_user[user_id] = pruned
            total_pruned += pruned
    if total_pruned:
        commit_with_retry()
    return {"user_count": len(user_ids), "pruned_count": total_pruned, "pruned_by_user": pruned_by_user}


def _wallet_custody_sync_job() -> dict[str, Any]:
    custody = get_service("wallet_custody")
    if not getattr(custody, "enabled", False):
        return {"enabled": False, "synced_user_count": 0, "failed_user_count": 0}
    limit = max(1, int(current_app.config.get("WORKER_WALLET_CUSTODY_SYNC_USER_LIMIT", 25) or 25))
    user_ids = [
        int(row[0])
        for row in db.session.query(WalletAddress.user_id)
        .filter(WalletAddress.status == "active", WalletAddress.user_id.isnot(None))
        .distinct()
        .order_by(WalletAddress.user_id.asc())
        .limit(limit)
        .all()
        if row[0] is not None
    ]
    synced: list[int] = []
    failed: dict[int, str] = {}
    for user_id in user_ids:
        try:
            custody.sync_user(user_id)
            commit_with_retry()
            synced.append(user_id)
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            failed[user_id] = str(exc)
            logger.warning("Worker wallet custody sync failed for user %s: %s", user_id, exc)
    return {
        "enabled": True,
        "candidate_user_count": len(user_ids),
        "synced_user_count": len(synced),
        "failed_user_count": len(failed),
        "failed_users": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlgVault dedicated worker jobs.")
    parser.add_argument("--once", action="store_true", help="Run due jobs once and exit.")
    parser.add_argument("--interval", type=int, default=None, help="Worker polling interval in seconds.")
    parser.add_argument("--owner-id", default="", help="Stable owner id for lease diagnostics.")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        choices=[
            "strategy_starter",
            "vault_cycle_enforcement",
            "treasury_solvency",
            "wallet_activity_retention",
            "wallet_custody_sync",
        ],
    )
    args = parser.parse_args()
    results = run_worker(
        once=bool(args.once),
        interval_seconds=args.interval,
        owner_id=args.owner_id or None,
        job_filter=set(args.job) if args.job else None,
    )
    print(json.dumps({"ok": True, "ran_at": datetime.utcnow().isoformat(), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
