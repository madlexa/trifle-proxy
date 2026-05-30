"""Security primitives: input validation, path sanitization, rate limiting.

This module is deliberately dependency-free and side-effect-free so it can be
reused from any layer (config parsing, file IO, request handling) without
pulling in the proxy lifecycle.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from trifle_proxy.logging_config import get_logger

log = get_logger("trifle_proxy.security")


class SecurityError(Exception):
    """Raised when an input fails a security check."""


# --- API key validation ---------------------------------------------------

# Keys we accept look like "sk-...", "sk-or-...", provider tokens, etc. We are
# intentionally permissive on the prefix but strict on shape: printable ASCII,
# no whitespace, within a sane length band. The goal is to reject obviously
# broken or injected values (empty, newlines, shell metacharacters), not to
# authenticate against a provider.
MIN_API_KEY_LENGTH = 8
MAX_API_KEY_LENGTH = 512
_API_KEY_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


def is_valid_api_key(key: object) -> bool:
    """Return True if ``key`` is a structurally plausible API key."""
    if not isinstance(key, str):
        return False
    if not (MIN_API_KEY_LENGTH <= len(key) <= MAX_API_KEY_LENGTH):
        return False
    return bool(_API_KEY_RE.match(key))


def validate_api_key(key: object, *, field: str = "api_key") -> str:
    """Validate an API key, returning it unchanged or raising ``SecurityError``."""
    if not is_valid_api_key(key):
        # Never log the key itself.
        log.warning("invalid_api_key", field=field)
        raise SecurityError(
            f"{field}: invalid API key (expected {MIN_API_KEY_LENGTH}-"
            f"{MAX_API_KEY_LENGTH} chars of [A-Za-z0-9_-.])"
        )
    return key  # type: ignore[return-value]


def redact_secret(value: str, *, keep: int = 4) -> str:
    """Return a log-safe rendering of a secret, keeping only a prefix."""
    if not isinstance(value, str) or not value:
        return ""
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep)


# --- Path sanitization -----------------------------------------------------


def sanitize_path(candidate: Path | str, *, base: Path | str) -> Path:
    """Resolve ``candidate`` and ensure it stays within ``base``.

    Protects against path-traversal (``../``), absolute-path escapes, and
    symlink-based escapes by resolving both paths before comparing. Raises
    ``SecurityError`` if the resolved candidate is not inside ``base``.
    """
    base_resolved = Path(base).resolve()
    raw = Path(candidate)

    # Treat absolute candidates as escapes unless they already live under base.
    target = raw if raw.is_absolute() else base_resolved / raw
    resolved = target.resolve()

    if resolved != base_resolved and base_resolved not in resolved.parents:
        log.warning("path_traversal_blocked", candidate=str(candidate), base=str(base_resolved))
        raise SecurityError(f"path escapes base directory: {candidate!r}")
    return resolved


# --- Backup integrity ------------------------------------------------------


def file_checksum(path: Path | str, *, algorithm: str = "sha256") -> str:
    """Return the hex digest of ``path``'s contents."""
    h = hashlib.new(algorithm)
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_backup_integrity(
    path: Path | str,
    expected_checksum: str,
    *,
    algorithm: str = "sha256",
) -> bool:
    """Return True if ``path`` exists and matches ``expected_checksum``."""
    p = Path(path)
    if not p.exists():
        log.warning("backup_missing", path=str(p))
        return False
    actual = file_checksum(p, algorithm=algorithm)
    ok = actual == expected_checksum
    if not ok:
        log.warning("backup_checksum_mismatch", path=str(p))
    return ok


# --- Rate limiting ---------------------------------------------------------


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    ``capacity`` tokens accumulate at ``refill_rate`` tokens/second up to the
    capacity. Each :meth:`allow` consumes tokens; it returns False (without
    blocking) when not enough tokens are available.

    A monotonic clock is injectable via ``time_func`` to keep tests
    deterministic.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        *,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._time = time_func
        self._tokens = float(capacity)
        self._updated = self._now()
        self._lock = threading.Lock()

    def _now(self) -> float:
        return float(self._time())

    def _refill(self) -> None:
        now = self._now()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._updated = now

    def allow(self, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens if available; return whether it was allowed."""
        if cost <= 0:
            return True
        with self._lock:
            self._refill()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    @property
    def tokens(self) -> float:
        """Current token count (refilled lazily)."""
        with self._lock:
            self._refill()
            return self._tokens
