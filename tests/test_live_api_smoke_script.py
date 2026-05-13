from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Callable

FRONTEND_ORIGIN = "https://app.algvault.test"
DISALLOWED_ORIGIN = "https://evil.example"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_api_smoke.py"


class _SmokeHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._handle()

    def do_GET(self) -> None:
        self._handle()

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
        return None

    def _handle(self) -> None:
        status, headers, body = self.server.response_for(self.command, self.path, self.headers)  # type: ignore[attr-defined]
        raw_body = body.encode("utf-8")
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(raw_body)))
        self.end_headers()
        self.wfile.write(raw_body)


@contextmanager
def _server(response_for: Callable):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SmokeHandler)
    server.response_for = response_for  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _safe_headers(origin: str) -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Accept, Content-Type, X-CSRF-Token, X-Requested-With, Idempotency-Key",
        "Vary": "Origin",
    }


def _safe_response_for(auth_status: int = 401):
    def response_for(method: str, path: str, headers):  # noqa: ANN001
        origin = str(headers.get("Origin") or "")
        if method == "OPTIONS":
            return (204, _safe_headers(FRONTEND_ORIGIN), "") if origin == FRONTEND_ORIGIN else (204, {}, "")
        return auth_status, _safe_headers(FRONTEND_ORIGIN), '{"ok": false, "error": "auth_required"}'

    return response_for


def _run_script(base_url: str | None = None, *, include_required_env: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in ("LIVE_API_BASE_URL", "FRONTEND_ORIGIN", "DISALLOWED_ORIGIN", "LIVE_API_SMOKE_TIMEOUT_SECONDS"):
        env.pop(key, None)
    if include_required_env and base_url:
        env.update(
            {
                "LIVE_API_BASE_URL": base_url,
                "FRONTEND_ORIGIN": FRONTEND_ORIGIN,
                "DISALLOWED_ORIGIN": DISALLOWED_ORIGIN,
                "LIVE_API_SMOKE_TIMEOUT_SECONDS": "2",
            }
        )
    return subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, env=env, timeout=10)


def test_live_api_smoke_allowed_origin_succeeds() -> None:
    with _server(_safe_response_for()) as base_url:
        result = _run_script(base_url)

    assert result.returncode == 0, result.stderr
    assert "Live API smoke passed" in result.stdout


def test_live_api_smoke_wildcard_plus_credentials_fails() -> None:
    def response_for(method: str, path: str, headers):  # noqa: ANN001
        del method, path, headers
        return 204, {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Credentials": "true", "Vary": "Origin"}, ""

    with _server(response_for) as base_url:
        result = _run_script(base_url)

    assert result.returncode == 1
    assert "Access-Control-Allow-Origin is '*'" in result.stderr


def test_live_api_smoke_disallowed_origin_with_credentialed_cors_fails() -> None:
    def response_for(method: str, path: str, headers):  # noqa: ANN001
        origin = str(headers.get("Origin") or "")
        if method == "OPTIONS":
            return 204, _safe_headers(origin), ""
        return 401, _safe_headers(FRONTEND_ORIGIN), '{"ok": false, "error": "auth_required"}'

    with _server(response_for) as base_url:
        result = _run_script(base_url)

    assert result.returncode == 1
    assert "disallowed origin received credentialed CORS headers" in result.stderr


def test_live_api_smoke_missing_required_env_fails_clearly() -> None:
    result = _run_script(include_required_env=False)

    assert result.returncode == 2
    assert "LIVE_API_BASE_URL" in result.stderr
    assert "FRONTEND_ORIGIN" in result.stderr


def test_live_api_smoke_detects_provider_ip_and_stack_trace_leaks() -> None:
    def response_for(method: str, path: str, headers):  # noqa: ANN001
        origin = str(headers.get("Origin") or "")
        if method == "OPTIONS":
            return (204, _safe_headers(FRONTEND_ORIGIN), "") if origin == FRONTEND_ORIGIN else (204, {}, "")
        body = (
            'KuCoin unavailable: {"code": "400302", "msg": "Our services are currently unavailable in the U.S. '
            'current ip: 3.86.110.165 and current area: US"}\nTraceback (most recent call last)'
        )
        return 401, _safe_headers(FRONTEND_ORIGIN), body

    with _server(response_for) as base_url:
        result = _run_script(base_url)

    assert result.returncode == 1
    assert "unsafe" in result.stderr
    assert "body was not printed" in result.stderr
    assert "3.86.110.165" not in result.stderr


def test_live_api_smoke_accepts_expected_auth_failure_status() -> None:
    with _server(_safe_response_for(auth_status=403)) as base_url:
        result = _run_script(base_url)

    assert result.returncode == 0, result.stderr
