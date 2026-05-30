"""Tests for retry, circuit breaker, fallback, and graceful recovery."""

from __future__ import annotations

import pytest

from trifle_proxy.resilience import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    FallbackExhaustedError,
    ResilienceConfig,
    RetryError,
    RetryPolicy,
    call_with_fallback,
    cleanup_on_error,
    retry_call,
)


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --- RetryPolicy -----------------------------------------------------------


def test_retry_policy_delay_grows_exponentially() -> None:
    policy = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=100.0)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0


def test_retry_policy_caps_at_max_delay() -> None:
    policy = RetryPolicy(base_delay=1.0, multiplier=10.0, max_delay=5.0)
    assert policy.delay_for(5) == 5.0


def test_retry_policy_jitter_is_deterministic_and_bounded() -> None:
    policy = RetryPolicy(base_delay=1.0, multiplier=1.0, jitter=0.5)
    d1 = policy.delay_for(2)
    d2 = policy.delay_for(2)
    assert d1 == d2  # reproducible
    assert 1.0 <= d1 <= 1.5  # within jitter band


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"base_delay": -1},
        {"max_delay": -1},
        {"multiplier": 0.5},
        {"jitter": 2.0},
    ],
)
def test_retry_policy_rejects_bad_params(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_retry_policy_delay_for_rejects_bad_attempt() -> None:
    with pytest.raises(ValueError):
        RetryPolicy().delay_for(0)


# --- retry_call ------------------------------------------------------------


def test_retry_call_returns_on_first_success() -> None:
    calls = []

    def fn() -> str:
        calls.append(1)
        return "ok"

    assert retry_call(fn, sleep=lambda _: None) == "ok"
    assert len(calls) == 1


def test_retry_call_retries_then_succeeds() -> None:
    attempts = []

    def fn() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("boom")
        return "ok"

    slept: list[float] = []
    policy = RetryPolicy(max_attempts=5, base_delay=1.0)
    result = retry_call(fn, policy=policy, sleep=slept.append)
    assert result == "ok"
    assert len(attempts) == 3
    assert slept == [1.0, 2.0]  # backoff between the two retries


def test_retry_call_exhausts_and_raises() -> None:
    def fn() -> None:
        raise ValueError("always")

    policy = RetryPolicy(max_attempts=3)
    with pytest.raises(RetryError) as ei:
        retry_call(fn, policy=policy, sleep=lambda _: None)
    assert ei.value.attempts == 3
    assert isinstance(ei.value.last_exception, ValueError)


def test_retry_call_does_not_retry_unlisted_exception() -> None:
    calls = []

    def fn() -> None:
        calls.append(1)
        raise KeyError("nope")

    with pytest.raises(KeyError):
        retry_call(fn, retry_on=(ValueError,), sleep=lambda _: None)
    assert len(calls) == 1


def test_retry_call_invokes_on_retry_callback() -> None:
    seen: list[int] = []

    def fn() -> None:
        raise ValueError("x")

    with pytest.raises(RetryError):
        retry_call(
            fn,
            policy=RetryPolicy(max_attempts=3),
            sleep=lambda _: None,
            on_retry=lambda attempt, _exc: seen.append(attempt),
        )
    assert seen == [1, 2]


# --- CircuitBreaker --------------------------------------------------------


def test_circuit_opens_after_threshold() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    assert cb.state == STATE_CLOSED
    for _ in range(3):
        cb.record_failure()
    assert cb.state == STATE_OPEN
    assert not cb.allow()


def test_circuit_success_resets_failure_count() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    cb.record_failure()
    cb.record_failure()
    cb.record_success()  # resets
    cb.record_failure()
    cb.record_failure()
    assert cb.state == STATE_CLOSED  # never reached 3 in a row


def test_circuit_half_opens_after_recovery_timeout() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0),
        time_func=clock,
    )
    cb.record_failure()
    assert cb.state == STATE_OPEN
    clock.advance(9.0)
    assert cb.state == STATE_OPEN
    clock.advance(1.0)
    assert cb.state == STATE_HALF_OPEN
    assert cb.allow()


def test_circuit_closes_after_successes_in_half_open() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1.0, success_threshold=2),
        time_func=clock,
    )
    cb.record_failure()
    clock.advance(1.0)
    assert cb.state == STATE_HALF_OPEN
    cb.record_success()
    assert cb.state == STATE_HALF_OPEN  # need 2
    cb.record_success()
    assert cb.state == STATE_CLOSED


def test_circuit_reopens_on_failure_in_half_open() -> None:
    clock = _FakeClock()
    cb = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, recovery_timeout=1.0),
        time_func=clock,
    )
    cb.record_failure()
    clock.advance(1.0)
    assert cb.state == STATE_HALF_OPEN
    cb.record_failure()
    assert cb.state == STATE_OPEN


def test_circuit_call_passes_through_and_records() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

    assert cb.call(lambda: 42) == 42

    def boom() -> None:
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        cb.call(boom)
    with pytest.raises(RuntimeError):
        cb.call(boom)
    assert cb.state == STATE_OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: 1)


def test_circuit_breaker_config_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        CircuitBreakerConfig(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreakerConfig(recovery_timeout=-1)
    with pytest.raises(ValueError):
        CircuitBreakerConfig(success_threshold=0)


# --- call_with_fallback ----------------------------------------------------


def test_fallback_returns_first_success() -> None:
    def call(target: str) -> str:
        if target == "a":
            raise ValueError("a down")
        return f"served by {target}"

    assert call_with_fallback(["a", "b", "c"], call) == "served by b"


def test_fallback_exhausted_raises() -> None:
    def call(_target: str) -> str:
        raise ValueError("all down")

    with pytest.raises(FallbackExhaustedError) as ei:
        call_with_fallback(["a", "b"], call)
    assert isinstance(ei.value.last_exception, ValueError)


def test_fallback_empty_targets_raises() -> None:
    with pytest.raises(FallbackExhaustedError):
        call_with_fallback([], lambda t: t)


def test_fallback_skips_open_breaker() -> None:
    open_breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    open_breaker.record_failure()  # now open
    breakers = {"a": open_breaker, "b": CircuitBreaker()}

    served: list[str] = []

    def call(target: str) -> str:
        served.append(target)
        return target

    result = call_with_fallback(["a", "b"], call, breakers=breakers)
    assert result == "b"
    assert served == ["b"]  # "a" was skipped, never called


def test_fallback_records_failure_on_breaker() -> None:
    breaker_a = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    breakers = {"a": breaker_a, "b": CircuitBreaker()}

    def call(target: str) -> str:
        if target == "a":
            raise ValueError("a down")
        return target

    assert call_with_fallback(["a", "b"], call, breakers=breakers) == "b"
    assert breaker_a.state == STATE_OPEN  # failure recorded through breaker


# --- cleanup_on_error ------------------------------------------------------


def test_cleanup_on_error_runs_on_exception() -> None:
    cleaned: list[int] = []
    with pytest.raises(ValueError), cleanup_on_error(lambda: cleaned.append(1)):
        raise ValueError("boom")
    assert cleaned == [1]


def test_cleanup_on_error_skips_on_success() -> None:
    cleaned: list[int] = []
    with cleanup_on_error(lambda: cleaned.append(1)):
        pass
    assert cleaned == []


def test_cleanup_on_error_can_suppress() -> None:
    cleaned: list[int] = []
    with cleanup_on_error(lambda: cleaned.append(1), reraise=False):
        raise ValueError("boom")
    assert cleaned == [1]


def test_cleanup_on_error_triggers_on_keyboard_interrupt() -> None:
    cleaned: list[int] = []
    with pytest.raises(KeyboardInterrupt), cleanup_on_error(lambda: cleaned.append(1)):
        raise KeyboardInterrupt
    assert cleaned == [1]


# --- ResilienceConfig ------------------------------------------------------


def test_resilience_config_defaults_from_none() -> None:
    cfg = ResilienceConfig.from_dict(None)
    assert cfg.retry == RetryPolicy()
    assert cfg.circuit == CircuitBreakerConfig()
    assert cfg.fallback_models == ()


def test_resilience_config_parses_full_dict() -> None:
    cfg = ResilienceConfig.from_dict(
        {
            "retry": {"max_attempts": 5, "base_delay": 1.0, "multiplier": 3.0},
            "circuit_breaker": {"failure_threshold": 10, "recovery_timeout": 60.0},
            "fallback_models": ["kimi-k2.5", "deepseek-chat"],
        }
    )
    assert cfg.retry.max_attempts == 5
    assert cfg.retry.base_delay == 1.0
    assert cfg.retry.multiplier == 3.0
    assert cfg.circuit.failure_threshold == 10
    assert cfg.circuit.recovery_timeout == 60.0
    assert cfg.fallback_models == ("kimi-k2.5", "deepseek-chat")


def test_resilience_config_partial_uses_defaults() -> None:
    cfg = ResilienceConfig.from_dict({"retry": {"max_attempts": 7}})
    assert cfg.retry.max_attempts == 7
    assert cfg.retry.base_delay == RetryPolicy.base_delay  # default kept
