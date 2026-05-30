"""Health checking for the LiteLLM proxy.

Combines a process-liveness check (is the PID alive?) with an HTTP probe
against LiteLLM's unauthenticated liveliness endpoint so callers can tell
the difference between "process gone", "process up but not serving", and
"healthy".
"""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import TypedDict

from trifle_proxy.config import DEFAULT_HOST, DEFAULT_PORT
from trifle_proxy.logging_config import get_logger
from trifle_proxy.proxy import is_running

log = get_logger("trifle_proxy.health")

# LiteLLM exposes an unauthenticated liveliness probe at this path.
HEALTH_PATH = "/health/liveliness"
DEFAULT_TIMEOUT = 5.0

STATUS_HEALTHY = "healthy"
STATUS_UNHEALTHY = "unhealthy"
STATUS_STOPPED = "stopped"


class HealthResult(TypedDict):
    """Structured result of a health check."""

    status: str
    process_running: bool
    http_reachable: bool
    endpoint: str
    detail: str


def _http_probe(url: str, timeout: float) -> tuple[bool, str]:
    """GET ``url`` and report whether it returned a 2xx response."""
    # Guard against file:/custom schemes — we only ever probe the local proxy.
    if not url.startswith(("http://", "https://")):
        return False, f"unsupported URL scheme: {url}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            code = getattr(resp, "status", None) or resp.getcode()
            if 200 <= code < 300:
                return True, f"HTTP {code}"
            return False, f"HTTP {code}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, str(e)


def check(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> HealthResult:
    """Check proxy health.

    Returns a :class:`HealthResult`. ``status`` is one of ``stopped`` (the
    process is not running), ``unhealthy`` (process is up but the HTTP probe
    failed), or ``healthy`` (process up and HTTP probe succeeded).
    """
    url = f"http://{host}:{port}{HEALTH_PATH}"

    process_running = is_running()
    if not process_running:
        log.info("health_check", status=STATUS_STOPPED, endpoint=url)
        return HealthResult(
            status=STATUS_STOPPED,
            process_running=False,
            http_reachable=False,
            endpoint=url,
            detail="proxy process is not running",
        )

    reachable, detail = _http_probe(url, timeout)
    status = STATUS_HEALTHY if reachable else STATUS_UNHEALTHY
    log.info(
        "health_check",
        status=status,
        endpoint=url,
        http_reachable=reachable,
        detail=detail,
    )
    return HealthResult(
        status=status,
        process_running=True,
        http_reachable=reachable,
        endpoint=url,
        detail=detail,
    )


def is_healthy(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Convenience wrapper returning True only when fully healthy."""
    return check(host=host, port=port, timeout=timeout)["status"] == STATUS_HEALTHY
