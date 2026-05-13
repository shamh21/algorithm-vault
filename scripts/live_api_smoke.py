#!/usr/bin/env python3
"""No-secrets deployment smoke test for the dedicated Vault live API."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys
from typing import Mapping
from urllib import error, parse, request

DEFAULT_ENDPOINT = "/api/vault/routing-preview"
DEFAULT_DISALLOWED_ORIGIN = "https://evil.example"
DEFAULT_TIMEOUT_SECONDS = 10.0
AUTH_FAILURE_STATUSES = {302, 303, 307, 308, 401, 403}
PREFLIGHT_REQUEST_HEADERS = "Accept, Content-Type, X-CSRF-Token, X-Requested-With, Idempotency-Key"

SENSITIVE_BODY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("backend IP text", re.compile(r"\bcurrent\s+ip\b|\bcurrent\s+area\s*:\s*us\b", re.IGNORECASE)),
    (
        "raw KuCoin region restriction text",
        re.compile(
            r"our services are currently unavailable in the u\.s|restricted country/region using a supported ip",
            re.IGNORECASE,
        ),
    ),
    (
        "raw KuCoin provider JSON",
        re.compile(r"\{\s*\"code\"\s*:\s*\"?400302\"?|kucoin unavailable:\s*\{", re.IGNORECASE),
    ),
    (
        "stack trace",
        re.compile(r"traceback \(most recent call last\)|werkzeug\.debug|<div class=\"traceback\"|file \"[^\"]+\", line \d+", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class SmokeConfig:
    live_api_base_url: str
    frontend_origin: str
    disallowed_origin: str
    endpoint: str
    timeout_seconds: float


@dataclass(frozen=True)
class SmokeResponse:
    status: int
    headers: dict[str, str]
    body: str


class SmokeConfigError(ValueError):
    pass


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D401
        return None


def _clean_origin(value: str, *, name: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = parse.urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SmokeConfigError(f"{name} must be an absolute http(s) origin.")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _clean_endpoint(value: str) -> str:
    raw = str(value or DEFAULT_ENDPOINT).strip() or DEFAULT_ENDPOINT
    if not raw.startswith("/"):
        raw = f"/{raw}"
    return raw


def config_from_env(environ: Mapping[str, str] | None = None) -> SmokeConfig:
    env = environ or os.environ
    missing = [name for name in ("LIVE_API_BASE_URL", "FRONTEND_ORIGIN") if not str(env.get(name, "")).strip()]
    if missing:
        raise SmokeConfigError(f"Missing required environment variable(s): {', '.join(missing)}.")
    timeout_raw = str(env.get("LIVE_API_SMOKE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as exc:
        raise SmokeConfigError("LIVE_API_SMOKE_TIMEOUT_SECONDS must be a number.") from exc
    if timeout_seconds <= 0:
        raise SmokeConfigError("LIVE_API_SMOKE_TIMEOUT_SECONDS must be greater than 0.")
    return SmokeConfig(
        live_api_base_url=_clean_origin(str(env["LIVE_API_BASE_URL"]), name="LIVE_API_BASE_URL"),
        frontend_origin=_clean_origin(str(env["FRONTEND_ORIGIN"]), name="FRONTEND_ORIGIN"),
        disallowed_origin=_clean_origin(str(env.get("DISALLOWED_ORIGIN", DEFAULT_DISALLOWED_ORIGIN)), name="DISALLOWED_ORIGIN"),
        endpoint=_clean_endpoint(str(env.get("LIVE_API_SMOKE_ENDPOINT", DEFAULT_ENDPOINT))),
        timeout_seconds=timeout_seconds,
    )


def _target_url(config: SmokeConfig) -> str:
    return f"{config.live_api_base_url}{config.endpoint}"


def _header(response: SmokeResponse, name: str) -> str:
    return response.headers.get(name.lower(), "").strip()


def _headers_dict(headers) -> dict[str, str]:  # noqa: ANN001
    result: dict[str, str] = {}
    for key in headers.keys():
        values = headers.get_all(key) or []
        result[key.lower()] = ", ".join(str(value) for value in values)
    return result


def _read_body(handle) -> str:  # noqa: ANN001
    raw = handle.read(32768)
    return raw.decode("utf-8", errors="replace")


def send_request(config: SmokeConfig, *, method: str, origin: str) -> SmokeResponse:
    headers = {
        "Accept": "application/json",
        "Origin": origin,
        "User-Agent": "AlgorithmVaultLiveApiSmoke/1.0",
    }
    if method.upper() == "OPTIONS":
        headers["Access-Control-Request-Method"] = "GET"
        headers["Access-Control-Request-Headers"] = PREFLIGHT_REQUEST_HEADERS
    req = request.Request(_target_url(config), headers=headers, method=method.upper())
    opener = request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=config.timeout_seconds) as handle:
            return SmokeResponse(status=int(handle.status), headers=_headers_dict(handle.headers), body=_read_body(handle))
    except error.HTTPError as exc:
        return SmokeResponse(status=int(exc.code), headers=_headers_dict(exc.headers), body=_read_body(exc))
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"{method.upper()} request failed before receiving a response: {reason}") from exc


def _vary_includes_origin(response: SmokeResponse) -> bool:
    return "origin" in {item.strip().lower() for item in _header(response, "vary").split(",") if item.strip()}


def _credentials_allowed(response: SmokeResponse) -> bool:
    return _header(response, "access-control-allow-credentials").lower() == "true"


def _check_no_wildcard_credentials(label: str, response: SmokeResponse, errors: list[str]) -> None:
    allow_origin = _header(response, "access-control-allow-origin")
    if allow_origin == "*" and _credentials_allowed(response):
        errors.append(f"{label}: Access-Control-Allow-Origin is '*' while credentials are allowed.")


def _check_no_sensitive_body(label: str, response: SmokeResponse, errors: list[str]) -> None:
    if not response.body:
        return
    for name, pattern in SENSITIVE_BODY_PATTERNS:
        if pattern.search(response.body):
            errors.append(f"{label}: response body contains unsafe {name}; body was not printed.")


def run_smoke(config: SmokeConfig) -> list[str]:
    errors: list[str] = []
    allowed = send_request(config, method="OPTIONS", origin=config.frontend_origin)
    if allowed.status >= 400:
        errors.append(f"allowed preflight: expected a non-error status, got HTTP {allowed.status}.")
    allowed_origin = _header(allowed, "access-control-allow-origin")
    if allowed_origin != config.frontend_origin:
        errors.append(
            "allowed preflight: Access-Control-Allow-Origin "
            f"must equal FRONTEND_ORIGIN ({config.frontend_origin}), got {allowed_origin or '<missing>'}."
        )
    if not _credentials_allowed(allowed):
        errors.append("allowed preflight: Access-Control-Allow-Credentials must be true.")
    if not _vary_includes_origin(allowed):
        errors.append("allowed preflight: Vary must include Origin.")
    _check_no_wildcard_credentials("allowed preflight", allowed, errors)
    _check_no_sensitive_body("allowed preflight", allowed, errors)

    disallowed = send_request(config, method="OPTIONS", origin=config.disallowed_origin)
    disallowed_origin = _header(disallowed, "access-control-allow-origin")
    if disallowed_origin in {"*", config.disallowed_origin}:
        errors.append("disallowed preflight: disallowed origin received a permissive Access-Control-Allow-Origin value.")
    if disallowed_origin and _credentials_allowed(disallowed):
        errors.append("disallowed preflight: disallowed origin received credentialed CORS headers.")
    _check_no_wildcard_credentials("disallowed preflight", disallowed, errors)
    _check_no_sensitive_body("disallowed preflight", disallowed, errors)

    unauthenticated = send_request(config, method="GET", origin=config.frontend_origin)
    if unauthenticated.status not in AUTH_FAILURE_STATUSES:
        errors.append(
            "unauthenticated GET: expected an auth failure/redirect status "
            f"{sorted(AUTH_FAILURE_STATUSES)}, got HTTP {unauthenticated.status}."
        )
    _check_no_wildcard_credentials("unauthenticated GET", unauthenticated, errors)
    _check_no_sensitive_body("unauthenticated GET", unauthenticated, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        config = config_from_env()
        errors = run_smoke(config)
    except SmokeConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Live API smoke failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print("Live API smoke failed:", file=sys.stderr)
        for item in errors:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Live API smoke passed for {_target_url(config)} with frontend origin {config.frontend_origin}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
