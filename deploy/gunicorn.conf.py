"""Gunicorn configuration for Algorithm Vault production deployments."""

from __future__ import annotations

import multiprocessing
import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except ValueError:
        return default


bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")

requested_workers = max(1, _int_env("WEB_CONCURRENCY", _int_env("GUNICORN_WORKERS", 1)))
workers = requested_workers
threads = max(1, _int_env("GUNICORN_THREADS", 4))
worker_class = "gthread"

timeout = max(30, _int_env("GUNICORN_TIMEOUT", 120))
graceful_timeout = max(30, _int_env("GUNICORN_GRACEFUL_TIMEOUT", 45))
keepalive = max(2, _int_env("GUNICORN_KEEPALIVE", 5))
worker_tmp_dir = os.getenv("GUNICORN_WORKER_TMP_DIR") or ("/dev/shm" if os.path.isdir("/dev/shm") else None)
max_requests = max(0, _int_env("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = max(0, _int_env("GUNICORN_MAX_REQUESTS_JITTER", 100))
preload_app = os.getenv("GUNICORN_PRELOAD", "false").lower() in {"1", "true", "yes"}

accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
capture_output = True

proc_name = os.getenv("GUNICORN_PROC_NAME", "algorithm-vault")
worker_connections = max(100, _int_env("GUNICORN_WORKER_CONNECTIONS", multiprocessing.cpu_count() * 100))
