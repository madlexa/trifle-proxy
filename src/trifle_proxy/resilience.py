"""Resilience primitives: retry with exponential backoff, circuit breaker, fallback.

Like :mod:`trifle_proxy.security`, this module is dependency-free (beyond the
logger) and clock-injectable so behaviour is deterministic in tests. It is used
both at the process-management layer (retrying a flaky proxy start, tripping a
breaker when health probes keep failing) and as config the LiteLLM router is
seeded with.
"""

from __future__ import annotations

import contextlib
import functools
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from trifle_proxy.logging_config import get_logger

log = get_logger("trifle_proxy.resilience")

T = TypeVar("T")
R = TypeVar("R")


class ResilienceError(Exception):
    """Base class for resilience failures."""


class RetryError(ResilienceError):
    """Raised when every retry attempt has been exhausted."""

    def __init__(self, attempts: int, last_exception: BaseException | None) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"all {attempts} attempt(s) failed; last error: {last_exception!r}")


class CircuitOpenError(ResilienceError):
    """Raised when a call is attempted while the circuit breaker is open."""


class FallbackExhaustedError(ResilienceError):
    """Raised when every fallback target has failed."""

    def __init__(self, last_exception: BaseException | None) -> None:
        self.last_exception = last_exception
        super().__init__(f"all fallback targets failed; last error: {last_exception!r}")


# --- Retry with exponential backoff ---------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry policy.

    Delay before the *n*-th retry is ``base_delay * multiplier**(n-1)`` capped
    at ``max_delay``. Jitter, when set, adds a deterministic fraction of the
    delay derived from the attempt number (no RNG, so tests stay stable).
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < 0:
            raise ValueError("max_delay must be >= 0")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be in [0, 1]")

    def delay_for(self, attempt: int) -> float:
        """Backoff delay (seconds) after a failed ``attempt`` (1-based)."""
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        raw = self.base_delay * (self.multiplier ** (attempt - 1))
        delay = min(raw, self.max_delay)
        if self.jitter:
            # Deterministic pseudo-jitter: spread successive attempts apart
            # without an RNG so retry timing is reproducible under test.
            frac = ((attempt * 2654435761) % 1000) / 1000.0
            delay += delay * self.jitter * frac
        return delay


def retry_call(
    func: Callable[[], R],
    *,
    policy: RetryPolicy | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> R:
    """Call ``func`` with retries, backing off between attempts.

    Re-raises immediately for exceptions not in ``retry_on``. When every
    attempt fails with a retryable error, raises :class:`RetryError` whose
    ``last_exception`` is the final failure.
    """
    policy = policy or RetryPolicy()
    last_exc: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return func()
        except retry_on as exc:
            last_exc = exc
            if attempt >= policy.max_attempts:
                break
            delay = policy.delay_for(attempt)
            log.warning(
                "retry",
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay=round(delay, 3),
                error=str(exc),
            )
            if on_retry is not None:
                on_retry(attempt, exc)
            if delay > 0:
                sleep(delay)

    log.error("retry_exhausted", attempts=policy.max_attempts)
    raise RetryError(policy.max_attempts, last_exc)


# --- Circuit breaker -------------------------------------------------------

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Tuning for :class:`CircuitBreaker`."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 1

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Closed -> Open after ``failure_threshold`` consecutive failures. After
    ``recovery_timeout`` seconds the breaker moves to Half-Open and admits a
    trial call; ``success_threshold`` consecutive successes close it again,
    while any failure re-opens it. A monotonic clock is injectable via
    ``time_func`` for deterministic tests.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        name: str = "default",
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self._time = time_func
        self._lock = threading.Lock()
        self._state = STATE_CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0

    def _maybe_recover(self) -> None:
        """Promote Open -> Half-Open once the recovery window elapses."""
        if (
            self._state == STATE_OPEN
            and self._time() - self._opened_at >= self.config.recovery_timeout
        ):
            self._state = STATE_HALF_OPEN
            self._successes = 0
            log.info("circuit_half_open", name=self.name)

    @property
    def state(self) -> str:
        """Current state, lazily promoting Open -> Half-Open if it is due."""
        with self._lock:
            self._maybe_recover()
            return self._state

    def allow(self) -> bool:
        """Return whether a call may proceed right now."""
        with self._lock:
            self._maybe_recover()
            return self._state != STATE_OPEN

    def record_success(self) -> None:
        with self._lock:
            if self._state == STATE_HALF_OPEN:
                self._successes += 1
                if self._successes >= self.config.success_threshold:
                    self._close_locked()
            else:
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == STATE_HALF_OPEN:
                self._open_locked()
                return
            self._failures += 1
            if self._failures >= self.config.failure_threshold:
                self._open_locked()

    def _open_locked(self) -> None:
        self._state = STATE_OPEN
        self._opened_at = self._time()
        self._successes = 0
        log.warning("circuit_open", name=self.name, failures=self._failures)

    def _close_locked(self) -> None:
        self._state = STATE_CLOSED
        self._failures = 0
        self._successes = 0
        log.info("circuit_closed", name=self.name)

    def call(self, func: Callable[[], R]) -> R:
        """Run ``func`` through the breaker, recording its outcome.

        Raises :class:`CircuitOpenError` without calling ``func`` if the
        breaker is open.
        """
        if not self.allow():
            log.warning("circuit_rejected", name=self.name)
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            result = func()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


# --- Fallback over multiple targets ----------------------------------------


def call_with_fallback(
    targets: Sequence[T],
    call: Callable[[T], R],
    *,
    breakers: dict[T, CircuitBreaker] | None = None,
    fallback_on: tuple[type[BaseException], ...] = (Exception,),
) -> R:
    """Try ``call(target)`` for each target in order, returning the first success.

    Targets whose circuit breaker is open are skipped. A target that raises an
    exception in ``fallback_on`` is recorded as a failure and the next target is
    tried. Raises :class:`FallbackExhaustedError` if no target succeeds.
    """
    if not targets:
        raise FallbackExhaustedError(None)

    last_exc: BaseException | None = None
    for target in targets:
        breaker = breakers.get(target) if breakers else None
        if breaker is not None and not breaker.allow():
            log.info("fallback_skip_open", target=str(target))
            continue
        try:
            if breaker is not None:
                return breaker.call(functools.partial(call, target))
            return call(target)
        except CircuitOpenError as exc:
            last_exc = exc
            continue
        except fallback_on as exc:
            last_exc = exc
            log.warning("fallback_failed", target=str(target), error=str(exc))
            continue

    log.error("fallback_exhausted", targets=len(targets))
    raise FallbackExhaustedError(last_exc)


# --- Cleanup / graceful recovery -------------------------------------------


@contextlib.contextmanager
def cleanup_on_error(
    cleanup: Callable[[], None],
    *,
    reraise: bool = True,
) -> Iterator[None]:
    """Run ``cleanup`` if the wrapped block raises, then optionally re-raise.

    Cleanup failures are logged and swallowed so the original error is never
    masked. Catches :class:`BaseException` so that ``KeyboardInterrupt`` /
    ``SystemExit`` also trigger rollback.
    """
    try:
        yield
    except BaseException as exc:
        log.error("cleanup_triggered", error=str(exc))
        try:
            cleanup()
        except Exception as cleanup_exc:  # pragma: no cover - defensive
            log.error("cleanup_failed", error=str(cleanup_exc))
        if reraise:
            raise


# --- Config -----------------------------------------------------------------


@dataclass(frozen=True)
class ResilienceConfig:
    """Resilience settings parsed from litellm.yaml's ``resilience`` section."""

    retry: RetryPolicy = field(default_factory=RetryPolicy)
    circuit: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    fallback_models: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ResilienceConfig:
        """Build from a (possibly partial) mapping, falling back to defaults.

        Unknown keys are ignored and malformed values fall back to defaults so
        a typo in the YAML never crashes the proxy.
        """
        data = data or {}

        retry_data = data.get("retry") or {}
        retry = RetryPolicy(
            max_attempts=int(retry_data.get("max_attempts", RetryPolicy.max_attempts)),
            base_delay=float(retry_data.get("base_delay", RetryPolicy.base_delay)),
            max_delay=float(retry_data.get("max_delay", RetryPolicy.max_delay)),
            multiplier=float(retry_data.get("multiplier", RetryPolicy.multiplier)),
            jitter=float(retry_data.get("jitter", RetryPolicy.jitter)),
        )

        cb_data = data.get("circuit_breaker") or {}
        circuit = CircuitBreakerConfig(
            failure_threshold=int(
                cb_data.get("failure_threshold", CircuitBreakerConfig.failure_threshold)
            ),
            recovery_timeout=float(
                cb_data.get("recovery_timeout", CircuitBreakerConfig.recovery_timeout)
            ),
            success_threshold=int(
                cb_data.get("success_threshold", CircuitBreakerConfig.success_threshold)
            ),
        )

        raw_fallbacks = data.get("fallback_models") or []
        fallback_models = tuple(str(m) for m in raw_fallbacks)

        return cls(retry=retry, circuit=circuit, fallback_models=fallback_models)
