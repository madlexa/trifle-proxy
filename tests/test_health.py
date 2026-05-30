"""Tests for health.py — proxy health checking."""

from __future__ import annotations

import urllib.error

import pytest

from trifle_proxy import health


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# --- _http_probe --------------------------------------------------------


def test_http_probe_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health.urllib.request, "urlopen", lambda url, timeout: FakeResponse(200))
    ok, detail = health._http_probe("http://x/health", 1.0)
    assert ok is True
    assert "200" in detail


def test_http_probe_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health.urllib.request, "urlopen", lambda url, timeout: FakeResponse(503))
    ok, detail = health._http_probe("http://x/health", 1.0)
    assert ok is False
    assert "503" in detail


def test_http_probe_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, timeout):
        raise urllib.error.HTTPError(url, 500, "err", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(health.urllib.request, "urlopen", boom)
    ok, detail = health._http_probe("http://x/health", 1.0)
    assert ok is False
    assert "500" in detail


def test_http_probe_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(health.urllib.request, "urlopen", boom)
    ok, detail = health._http_probe("http://x/health", 1.0)
    assert ok is False
    assert "refused" in detail


def test_http_probe_rejects_non_http_scheme() -> None:
    ok, detail = health._http_probe("file:///etc/passwd", 1.0)
    assert ok is False
    assert "unsupported URL scheme" in detail


# --- check --------------------------------------------------------------


def test_check_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "is_running", lambda: False)
    result = health.check()
    assert result["status"] == health.STATUS_STOPPED
    assert result["process_running"] is False
    assert result["http_reachable"] is False


def test_check_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "is_running", lambda: True)
    monkeypatch.setattr(health, "_http_probe", lambda url, timeout: (True, "HTTP 200"))
    result = health.check(host="127.0.0.1", port=4000)
    assert result["status"] == health.STATUS_HEALTHY
    assert result["process_running"] is True
    assert result["http_reachable"] is True
    assert result["endpoint"].endswith(health.HEALTH_PATH)


def test_check_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "is_running", lambda: True)
    monkeypatch.setattr(health, "_http_probe", lambda url, timeout: (False, "refused"))
    result = health.check()
    assert result["status"] == health.STATUS_UNHEALTHY
    assert result["process_running"] is True
    assert result["http_reachable"] is False


# --- is_healthy ---------------------------------------------------------


def test_is_healthy_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "check", lambda **kw: {"status": health.STATUS_HEALTHY})
    assert health.is_healthy() is True


def test_is_healthy_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "check", lambda **kw: {"status": health.STATUS_STOPPED})
    assert health.is_healthy() is False
