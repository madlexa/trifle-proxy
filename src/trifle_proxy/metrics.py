"""Prometheus-compatible metrics for trifle-proxy.

Like :mod:`trifle_proxy.security` and :mod:`trifle_proxy.resilience`, this
module is dependency-free (beyond the logger) and thread-safe so it can be
updated from signal handlers and background threads. It implements just enough
of the Prometheus text exposition format (version 0.0.4) to be scraped by a
Prometheus server or rendered on the CLI — counters, gauges, and histograms
with labels — without taking on ``prometheus_client`` as a runtime dependency.

The proxy itself is a child LiteLLM process; trifle-proxy instruments the
points it controls (process lifecycle, per-provider request outcomes and
latency) and can additionally scrape LiteLLM's native ``/metrics`` endpoint.
"""

from __future__ import annotations

import http.server
import math
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from trifle_proxy.logging_config import get_logger

if TYPE_CHECKING:
    from types import TracebackType

log = get_logger("trifle_proxy.metrics")

METRICS_PATH = "/metrics"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_METRICS_PORT = 9090
DEFAULT_SCRAPE_TIMEOUT = 5.0
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Prometheus client default histogram buckets (seconds), plus +Inf.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


class MetricError(Exception):
    """Raised for invalid metric definitions or label usage."""


def _validate_name(name: str) -> str:
    """Reject names that would corrupt the exposition format."""
    if not name or not all(c.isalnum() or c in "_:" for c in name):
        raise MetricError(f"invalid metric/label name: {name!r}")
    if name[0].isdigit():
        raise MetricError(f"metric/label name may not start with a digit: {name!r}")
    return name


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus exposition spec."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_labels(label_names: Sequence[str], values: tuple[str, ...]) -> str:
    """Render ``{a="1",b="2"}`` (or empty string when there are no labels)."""
    if not label_names:
        return ""
    pairs = ",".join(
        f'{name}="{_escape_label_value(val)}"'
        for name, val in zip(label_names, values, strict=True)
    )
    return "{" + pairs + "}"


class _Metric:
    """Base for label-aware metrics. Subclasses define ``metric_type``."""

    metric_type = "untyped"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        self.name = _validate_name(name)
        self.documentation = documentation
        self.labelnames = tuple(_validate_name(n) for n in labelnames)
        self._lock = threading.Lock()

    def _key(self, labels: Mapping[str, str] | None) -> tuple[str, ...]:
        labels = labels or {}
        if set(labels) != set(self.labelnames):
            raise MetricError(
                f"{self.name}: expected labels {self.labelnames}, got {tuple(labels)}"
            )
        return tuple(str(labels[name]) for name in self.labelnames)

    def collect(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError


class Counter(_Metric):
    """A monotonically increasing counter."""

    metric_type = "counter"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise MetricError("counters may only increase")
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        if not items and not self.labelnames:
            items = [((), 0.0)]
        return [
            f"{self.name}{_render_labels(self.labelnames, key)} {_format_float(val)}"
            for key, val in items
        ]


class Gauge(_Metric):
    """A value that can go up and down."""

    metric_type = "gauge"

    def __init__(self, name: str, documentation: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def set(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, **labels: str) -> None:
        self.inc(-amount, **labels)

    def value(self, **labels: str) -> float:
        key = self._key(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[str]:
        with self._lock:
            items = sorted(self._values.items())
        if not items and not self.labelnames:
            items = [((), 0.0)]
        return [
            f"{self.name}{_render_labels(self.labelnames, key)} {_format_float(val)}"
            for key, val in items
        ]


class _HistogramChild:
    __slots__ = ("buckets", "counts", "sum", "total")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0 for _ in buckets]  # per-bucket counts (value <= le)
        self.sum = 0.0
        self.total = 0  # all observations, including those above every bucket


class Histogram(_Metric):
    """A cumulative histogram of observed values (e.g. latency seconds)."""

    metric_type = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] = (),
        buckets: Sequence[float] = DEFAULT_BUCKETS,
    ) -> None:
        super().__init__(name, documentation, labelnames)
        ordered = tuple(sorted(float(b) for b in buckets))
        if not ordered:
            raise MetricError("histogram needs at least one bucket")
        self.buckets = ordered
        self._children: dict[tuple[str, ...], _HistogramChild] = {}

    def _child(self, key: tuple[str, ...]) -> _HistogramChild:
        child = self._children.get(key)
        if child is None:
            child = _HistogramChild(self.buckets)
            self._children[key] = child
        return child

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        with self._lock:
            child = self._child(key)
            child.sum += value
            child.total += 1
            for i, upper in enumerate(self.buckets):
                if value <= upper:
                    child.counts[i] += 1

    @contextmanager
    def time(self, time_func: Callable[[], float], **labels: str) -> Iterator[None]:
        """Observe the wall-clock duration of the wrapped block.

        ``time_func`` is injected (no implicit clock) so timing stays
        deterministic under test.
        """
        start = time_func()
        try:
            yield
        finally:
            self.observe(time_func() - start, **labels)

    def count(self, **labels: str) -> int:
        key = self._key(labels)
        with self._lock:
            child = self._children.get(key)
            return child.total if child else 0

    def collect(self) -> list[str]:
        with self._lock:
            items = sorted(self._children.items())
            snapshot = [(key, list(child.counts), child.sum, child.total) for key, child in items]
        lines: list[str] = []
        for key, counts, total_sum, total_count in snapshot:
            for upper, count in zip(self.buckets, counts, strict=True):
                bucket_labels = self._bucket_labels(key, _format_float(upper))
                lines.append(f"{self.name}_bucket{bucket_labels} {count}")
            inf_labels = self._bucket_labels(key, "+Inf")
            lines.append(f"{self.name}_bucket{inf_labels} {total_count}")
            base = _render_labels(self.labelnames, key)
            lines.append(f"{self.name}_sum{base} {_format_float(total_sum)}")
            lines.append(f"{self.name}_count{base} {total_count}")
        return lines

    def _bucket_labels(self, key: tuple[str, ...], le: str) -> str:
        names = (*self.labelnames, "le")
        values = (*key, le)
        return _render_labels(names, values)


def _format_float(value: float) -> str:
    """Render a float without a trailing ``.0`` for whole numbers."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(value)


class CollectorRegistry:
    """A set of named metrics rendered together into exposition text."""

    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def register(self, metric: _Metric) -> None:
        with self._lock:
            if metric.name in self._metrics:
                raise MetricError(f"metric already registered: {metric.name}")
            self._metrics[metric.name] = metric

    def unregister(self, name: str) -> None:
        with self._lock:
            self._metrics.pop(name, None)

    def render(self) -> str:
        """Render all registered metrics as Prometheus exposition text."""
        with self._lock:
            metrics = list(self._metrics.values())
        blocks: list[str] = []
        for metric in metrics:
            lines = metric.collect()
            block = [
                f"# HELP {metric.name} {metric.documentation}",
                f"# TYPE {metric.name} {metric.metric_type}",
                *lines,
            ]
            blocks.append("\n".join(block))
        return "\n".join(blocks) + "\n" if blocks else ""


# --- Default registry and the metrics trifle-proxy itself records ----------

REGISTRY = CollectorRegistry()

PROXY_STARTS = Counter("trifle_proxy_proxy_starts_total", "Number of proxy start operations")
PROXY_STOPS = Counter("trifle_proxy_proxy_stops_total", "Number of proxy stop operations")
PROXY_UP = Gauge("trifle_proxy_proxy_up", "Whether the proxy is currently running (1) or not (0)")
REQUESTS = Counter(
    "trifle_proxy_requests_total",
    "Total proxied requests by provider/model/status",
    labelnames=("provider", "model", "status"),
)
ERRORS = Counter(
    "trifle_proxy_request_errors_total",
    "Total proxied request errors by provider/model/type",
    labelnames=("provider", "model", "type"),
)
REQUEST_LATENCY = Histogram(
    "trifle_proxy_request_latency_seconds",
    "Request latency in seconds by provider/model",
    labelnames=("provider", "model"),
)

for _metric in (PROXY_STARTS, PROXY_STOPS, PROXY_UP, REQUESTS, ERRORS, REQUEST_LATENCY):
    REGISTRY.register(_metric)


def record_start() -> None:
    """Record a successful proxy start."""
    PROXY_STARTS.inc()
    PROXY_UP.set(1)


def record_stop() -> None:
    """Record a proxy stop."""
    PROXY_STOPS.inc()
    PROXY_UP.set(0)


def set_proxy_up(running: bool) -> None:
    """Reflect current proxy liveness in the gauge."""
    PROXY_UP.set(1 if running else 0)


def record_request(
    provider: str,
    model: str,
    *,
    status: str = "success",
    latency_seconds: float | None = None,
) -> None:
    """Record a single proxied request outcome and (optionally) its latency."""
    REQUESTS.inc(provider=provider, model=model, status=status)
    if latency_seconds is not None:
        REQUEST_LATENCY.observe(latency_seconds, provider=provider, model=model)


def record_error(provider: str, model: str, *, error_type: str = "unknown") -> None:
    """Record a proxied request error, also bumping the failed request counter."""
    ERRORS.inc(provider=provider, model=model, type=error_type)
    REQUESTS.inc(provider=provider, model=model, status="error")


def render(registry: CollectorRegistry | None = None) -> str:
    """Render the default (or a supplied) registry to exposition text."""
    return (registry or REGISTRY).render()


# --- HTTP exposition --------------------------------------------------------


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    """Serves ``GET /metrics`` from the server's registry; 404 otherwise."""

    # Set by MetricsServer before the server starts.
    registry: CollectorRegistry = REGISTRY

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.rstrip("/") not in ("", METRICS_PATH):
            self.send_error(404, "not found")
            return
        body = self.registry.render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Route through structlog rather than stderr.
        log.debug("metrics_http", client=self.address_string(), message=format % args)


class MetricsServer:
    """A daemon-thread HTTP server exposing the registry at ``/metrics``."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_METRICS_PORT,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.registry = registry or REGISTRY
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> int:
        """Start serving in a background thread; return the bound port."""
        if self._server is not None:
            raise RuntimeError("metrics server already started")

        handler = type(
            "_BoundMetricsHandler",
            (_MetricsHandler,),
            {"registry": self.registry},
        )
        self._server = http.server.ThreadingHTTPServer((self.host, self.port), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="trifle-proxy-metrics",
            daemon=True,
        )
        self._thread.start()
        log.info("metrics_server_started", host=self.host, port=self.port)
        return self.port

    def stop(self) -> None:
        """Stop the server and join its thread."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        log.info("metrics_server_stopped", host=self.host, port=self.port)
        self._server = None
        self._thread = None

    def __enter__(self) -> MetricsServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def scrape(url: str, timeout: float = DEFAULT_SCRAPE_TIMEOUT) -> str:
    """Fetch exposition text from a metrics endpoint.

    Raises :class:`MetricError` for an unsupported scheme or any network/HTTP
    failure so callers can fall back to local rendering.
    """
    if not url.startswith(("http://", "https://")):
        raise MetricError(f"unsupported URL scheme: {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310  # nosec B310
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise MetricError(f"failed to scrape {url}: {e}") from e


def metrics_url(host: str = DEFAULT_HOST, port: int = DEFAULT_METRICS_PORT) -> str:
    """Build the metrics endpoint URL for ``host``/``port``."""
    return f"http://{host}:{port}{METRICS_PATH}"
