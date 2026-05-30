"""Tests for metrics.py — Prometheus-compatible metrics and HTTP exposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from trifle_proxy import cli, metrics, proxy
from trifle_proxy.metrics import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    MetricError,
    MetricsServer,
)


def ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# --- Counter ------------------------------------------------------------


def test_counter_inc_and_value() -> None:
    c = Counter("test_total", "doc")
    assert c.value() == 0
    c.inc()
    c.inc(4)
    assert c.value() == 5


def test_counter_rejects_negative() -> None:
    c = Counter("test_total", "doc")
    with pytest.raises(MetricError, match="only increase"):
        c.inc(-1)


def test_counter_with_labels() -> None:
    c = Counter("reqs_total", "doc", labelnames=("provider", "status"))
    c.inc(provider="moonshot", status="success")
    c.inc(2, provider="moonshot", status="success")
    c.inc(provider="deepseek", status="error")
    assert c.value(provider="moonshot", status="success") == 3
    assert c.value(provider="deepseek", status="error") == 1


def test_counter_label_mismatch_raises() -> None:
    c = Counter("reqs_total", "doc", labelnames=("provider",))
    with pytest.raises(MetricError, match="expected labels"):
        c.inc(provider="x", status="bad")
    with pytest.raises(MetricError):
        c.inc()


def test_counter_collect_zero_when_unlabelled() -> None:
    c = Counter("test_total", "doc")
    assert c.collect() == ["test_total 0"]


# --- Gauge --------------------------------------------------------------


def test_gauge_set_inc_dec() -> None:
    g = Gauge("temp", "doc")
    g.set(10)
    assert g.value() == 10
    g.inc(5)
    assert g.value() == 15
    g.dec(20)
    assert g.value() == -5


def test_gauge_with_labels() -> None:
    g = Gauge("up", "doc", labelnames=("name",))
    g.set(1, name="a")
    g.set(0, name="b")
    assert g.value(name="a") == 1
    assert g.value(name="b") == 0


# --- Histogram ----------------------------------------------------------


def test_histogram_observe_buckets_cumulative() -> None:
    h = Histogram("lat", "doc", buckets=(0.1, 0.5, 1.0))
    h.observe(0.05)
    h.observe(0.3)
    h.observe(2.0)
    lines = h.collect()
    rendered = "\n".join(lines)
    # 0.05 -> le=0.1; 0.3 -> le=0.5; 2.0 -> only +Inf
    assert 'lat_bucket{le="0.1"} 1' in rendered
    assert 'lat_bucket{le="0.5"} 2' in rendered
    assert 'lat_bucket{le="1"} 2' in rendered
    assert 'lat_bucket{le="+Inf"} 3' in rendered
    assert "lat_count 3" in rendered
    assert "lat_sum 2.35" in rendered


def test_histogram_count_and_sorted_buckets() -> None:
    h = Histogram("lat", "doc", buckets=(1.0, 0.1))  # unsorted on purpose
    assert h.buckets == (0.1, 1.0)
    h.observe(0.05)
    assert h.count() == 1


def test_histogram_empty_buckets_raises() -> None:
    with pytest.raises(MetricError, match="at least one bucket"):
        Histogram("lat", "doc", buckets=())


def test_histogram_time_context_uses_injected_clock() -> None:
    h = Histogram("lat", "doc", buckets=(1.0,), labelnames=("provider", "model"))
    ticks = iter([100.0, 100.4])
    with h.time(lambda: next(ticks), provider="p", model="m"):
        pass
    line = "\n".join(h.collect())
    assert 'lat_sum{provider="p",model="m"} 0.4' in line
    assert h.count(provider="p", model="m") == 1


# --- helpers ------------------------------------------------------------


def test_format_float() -> None:
    assert metrics._format_float(3.0) == "3"
    assert metrics._format_float(2.5) == "2.5"


def test_format_float_non_finite() -> None:
    assert metrics._format_float(float("inf")) == "+Inf"
    assert metrics._format_float(float("-inf")) == "-Inf"
    assert metrics._format_float(float("nan")) == "NaN"


def test_histogram_observe_inf_renders() -> None:
    h = metrics.Histogram("lat", "latency", buckets=(0.1, 1.0))
    h.observe(float("inf"))
    # render() must not raise even with a non-finite sum.
    rendered = "\n".join(h.collect())
    assert "lat_sum" in rendered
    assert "+Inf" in rendered


def test_label_value_escaping() -> None:
    c = Counter("test_total", "doc", labelnames=("k",))
    c.inc(k='a"b\\c\nd')
    rendered = "\n".join(c.collect())
    assert 'k="a\\"b\\\\c\\nd"' in rendered


def test_invalid_metric_name_raises() -> None:
    with pytest.raises(MetricError):
        Counter("bad name!", "doc")
    with pytest.raises(MetricError):
        Counter("1leading", "doc")


# --- CollectorRegistry --------------------------------------------------


def test_registry_render_includes_help_and_type() -> None:
    reg = CollectorRegistry()
    c = Counter("reqs_total", "Total requests")
    c.inc(3)
    reg.register(c)
    out = reg.render()
    assert "# HELP reqs_total Total requests" in out
    assert "# TYPE reqs_total counter" in out
    assert "reqs_total 3" in out
    assert out.endswith("\n")


def test_registry_render_empty() -> None:
    assert CollectorRegistry().render() == ""


def test_registry_duplicate_registration_raises() -> None:
    reg = CollectorRegistry()
    reg.register(Counter("dup_total", "doc"))
    with pytest.raises(MetricError, match="already registered"):
        reg.register(Counter("dup_total", "doc"))


def test_registry_unregister() -> None:
    reg = CollectorRegistry()
    c = Counter("x_total", "doc")
    reg.register(c)
    reg.unregister("x_total")
    assert reg.render() == ""


# --- module-level recording helpers -------------------------------------


@pytest.fixture
def reset_default_metrics() -> None:
    """Clear accumulated state in the shared default metrics."""
    metrics.PROXY_STARTS._values.clear()
    metrics.PROXY_STOPS._values.clear()
    metrics.PROXY_UP._values.clear()
    metrics.REQUESTS._values.clear()
    metrics.ERRORS._values.clear()
    metrics.REQUEST_LATENCY._children.clear()


def test_record_start_stop(reset_default_metrics: None) -> None:
    metrics.record_start()
    assert metrics.PROXY_STARTS.value() == 1
    assert metrics.PROXY_UP.value() == 1
    metrics.record_stop()
    assert metrics.PROXY_STOPS.value() == 1
    assert metrics.PROXY_UP.value() == 0


def test_set_proxy_up(reset_default_metrics: None) -> None:
    metrics.set_proxy_up(True)
    assert metrics.PROXY_UP.value() == 1
    metrics.set_proxy_up(False)
    assert metrics.PROXY_UP.value() == 0


def test_record_request_and_error(reset_default_metrics: None) -> None:
    metrics.record_request("moonshot", "kimi", status="success", latency_seconds=0.2)
    assert metrics.REQUESTS.value(provider="moonshot", model="kimi", status="success") == 1
    assert metrics.REQUEST_LATENCY.count(provider="moonshot", model="kimi") == 1

    metrics.record_error("deepseek", "chat", error_type="timeout")
    assert metrics.ERRORS.value(provider="deepseek", model="chat", type="timeout") == 1
    assert metrics.REQUESTS.value(provider="deepseek", model="chat", status="error") == 1


def test_default_registry_has_expected_metrics() -> None:
    out = metrics.render()
    for name in (
        "trifle_proxy_proxy_starts_total",
        "trifle_proxy_proxy_stops_total",
        "trifle_proxy_proxy_up",
        "trifle_proxy_requests_total",
        "trifle_proxy_request_errors_total",
        "trifle_proxy_request_latency_seconds",
    ):
        assert name in out


# --- HTTP server + scrape -----------------------------------------------


def test_metrics_server_serves_metrics() -> None:
    reg = CollectorRegistry()
    c = Counter("served_total", "doc")
    c.inc(7)
    reg.register(c)

    server = MetricsServer(port=0, registry=reg)
    port = server.start()
    try:
        text = metrics.scrape(f"http://127.0.0.1:{port}/metrics")
        assert "served_total 7" in text
    finally:
        server.stop()


def test_metrics_server_context_manager_and_404() -> None:
    reg = CollectorRegistry()
    reg.register(Counter("ctx_total", "doc"))
    with MetricsServer(port=0, registry=reg) as server:
        with pytest.raises(MetricError):
            metrics.scrape(f"http://127.0.0.1:{server.port}/nope")
        # The metrics path still works.
        assert "ctx_total" in metrics.scrape(f"http://127.0.0.1:{server.port}/metrics")


def test_metrics_server_double_start_raises() -> None:
    server = MetricsServer(port=0)
    server.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            server.start()
    finally:
        server.stop()


def test_metrics_server_stop_is_idempotent() -> None:
    server = MetricsServer(port=0)
    server.stop()  # never started — no-op


def test_scrape_rejects_bad_scheme() -> None:
    with pytest.raises(MetricError, match="unsupported URL scheme"):
        metrics.scrape("file:///etc/passwd")


def test_scrape_network_failure() -> None:
    # Nothing listens on this port → connection refused → MetricError.
    with pytest.raises(MetricError, match="failed to scrape"):
        metrics.scrape("http://127.0.0.1:1/metrics", timeout=0.5)


def test_metrics_url_builder() -> None:
    assert metrics.metrics_url("host", 1234) == "http://host:1234/metrics"


# --- CLI metrics command ------------------------------------------------


def test_cmd_metrics_local(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.cmd_metrics(ns(local=True, url=None, host="127.0.0.1", port=9090))
    assert rc == 0
    out = capsys.readouterr().out
    assert "trifle_proxy_proxy_up" in out


def test_cmd_metrics_scrapes_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(metrics, "scrape", lambda url, timeout=5.0: "scraped_metric 1\n")
    rc = cli.cmd_metrics(ns(local=False, url="http://x/metrics", host="h", port=1))
    assert rc == 0
    assert "scraped_metric 1" in capsys.readouterr().out


def test_cmd_metrics_falls_back_to_local_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(url, timeout=5.0):
        raise MetricError("connection refused")

    monkeypatch.setattr(metrics, "scrape", boom)
    rc = cli.cmd_metrics(ns(local=False, url=None, host="127.0.0.1", port=9090))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Could not scrape" in out
    assert "Falling back to local metrics" in out
    assert "trifle_proxy_proxy_up" in out


# --- proxy instrumentation ----------------------------------------------


def test_proxy_start_records_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "PID_FILE", tmp_path / "logs" / "litellm.pid")
    monkeypatch.setattr(proxy, "LOG_FILE", tmp_path / "logs" / "litellm.log")
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "_has_uvx", lambda: False)
    config = tmp_path / "litellm.yaml"
    config.write_text("model_list: []")

    class DummyPopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 4321

    monkeypatch.setattr(proxy.subprocess, "Popen", DummyPopen)

    recorded = {"start": False}
    monkeypatch.setattr(proxy.metrics, "record_start", lambda: recorded.__setitem__("start", True))

    proxy.start(config=config)
    assert recorded["start"] is True


def test_proxy_stop_records_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "PID_FILE", tmp_path / "logs" / "litellm.pid")
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid

        def is_running(self):
            return True

        def status(self):
            return "running"

        def cmdline(self):
            return ["python", "-m", "litellm"]

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid))

    recorded = {"stop": False}
    monkeypatch.setattr(proxy.metrics, "record_stop", lambda: recorded.__setitem__("stop", True))

    assert proxy.stop() is True
    assert recorded["stop"] is True
