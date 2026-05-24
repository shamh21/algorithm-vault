"""Dedicated worker runner for recurring AlgVault jobs."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from datetime import datetime
from typing import Any

from app import create_app
from app.extensions import db
from app.models import StrategyRun
from app.runtime import get_service
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
        "apple_pay_fulfillment": _apple_pay_fulfillment_job,
    }
    config = lease_service.config
    enabled = {
        "strategy_starter": bool(config.get("WORKER_STRATEGY_STARTER_ENABLED", True)),
        "vault_cycle_enforcement": bool(config.get("WORKER_VAULT_ENFORCEMENT_ENABLED", True)),
        "treasury_solvency": bool(config.get("WORKER_TREASURY_SOLVENCY_ENABLED", True)),
        "apple_pay_fulfillment": bool(config.get("WORKER_APPLE_PAY_FULFILLMENT_ENABLED", True)),
    }
    results: list[dict[str, Any]] = []
    bucket = int(time.time() // max(1, int(config.get("WORKER_POLL_SECONDS", 15) or 15)))
    for name, fn in jobs.items():
        if job_filter and name not in job_filter:
            continue
        if not enabled.get(name, False):
            continue
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
    results = get_service("vault_cycle_trading_enforcer").enforce_active_cycles(None)
    return {"cycle_count": len(results), "cycles": results}


def _treasury_solvency_job() -> dict[str, Any]:
    result = get_service("platform_treasury").process_solvency_cycle()
    return dict(result or {})


def _apple_pay_fulfillment_job() -> dict[str, Any]:
    result = get_service("wallet_apple_pay_purchase").process_pending_orders()
    return dict(result or {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlgVault dedicated worker jobs.")
    parser.add_argument("--once", action="store_true", help="Run due jobs once and exit.")
    parser.add_argument("--interval", type=int, default=None, help="Worker polling interval in seconds.")
    parser.add_argument("--owner-id", default="", help="Stable owner id for lease diagnostics.")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        choices=["strategy_starter", "vault_cycle_enforcement", "treasury_solvency", "apple_pay_fulfillment"],
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
