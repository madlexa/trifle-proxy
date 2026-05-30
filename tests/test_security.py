from pathlib import Path

import pytest

from trifle_proxy.security import (
    MAX_API_KEY_LENGTH,
    SecurityError,
    TokenBucket,
    file_checksum,
    is_valid_api_key,
    redact_secret,
    sanitize_path,
    validate_api_key,
    verify_backup_integrity,
)

# --- API key validation ---


@pytest.mark.parametrize(
    "key",
    [
        "sk-abcdef12",
        "sk-or-v1-0123456789abcdef",
        "ABCdef_-.123",
        "a" * MAX_API_KEY_LENGTH,
    ],
)
def test_is_valid_api_key_accepts(key: str) -> None:
    assert is_valid_api_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "short",  # < 8 chars
        "has space",
        "has\nnewline",
        "semi;colon",
        "a" * (MAX_API_KEY_LENGTH + 1),
        123,
        None,
    ],
)
def test_is_valid_api_key_rejects(key: object) -> None:
    assert not is_valid_api_key(key)


def test_validate_api_key_returns_key() -> None:
    assert validate_api_key("sk-validkey123") == "sk-validkey123"


def test_validate_api_key_raises() -> None:
    with pytest.raises(SecurityError):
        validate_api_key("bad key")


def test_redact_secret() -> None:
    assert redact_secret("sk-1234567890") == "sk-1*********"
    assert redact_secret("abc") == "***"
    assert redact_secret("") == ""


# --- Path sanitization ---


def test_sanitize_path_allows_inside(tmp_path: Path) -> None:
    result = sanitize_path("sub/file.json", base=tmp_path)
    assert result == (tmp_path / "sub" / "file.json").resolve()


def test_sanitize_path_allows_base_itself(tmp_path: Path) -> None:
    assert sanitize_path(".", base=tmp_path) == tmp_path.resolve()


def test_sanitize_path_blocks_traversal(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        sanitize_path("../../etc/passwd", base=tmp_path)


def test_sanitize_path_blocks_absolute_escape(tmp_path: Path) -> None:
    with pytest.raises(SecurityError):
        sanitize_path("/etc/passwd", base=tmp_path)


def test_sanitize_path_allows_absolute_inside(tmp_path: Path) -> None:
    inside = tmp_path / "x.json"
    assert sanitize_path(inside, base=tmp_path) == inside.resolve()


# --- Backup integrity ---


def test_file_checksum_and_verify(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text("hello world")
    digest = file_checksum(f)
    assert len(digest) == 64  # sha256 hex
    assert verify_backup_integrity(f, digest)


def test_verify_backup_integrity_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "data.json"
    f.write_text("original")
    digest = file_checksum(f)
    f.write_text("tampered")
    assert not verify_backup_integrity(f, digest)


def test_verify_backup_integrity_missing(tmp_path: Path) -> None:
    assert not verify_backup_integrity(tmp_path / "nope.json", "deadbeef")


# --- Token bucket ---


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_token_bucket_allows_up_to_capacity() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(capacity=3, refill_rate=1, time_func=clock)
    assert bucket.allow()
    assert bucket.allow()
    assert bucket.allow()
    assert not bucket.allow()  # exhausted


def test_token_bucket_refills_over_time() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(capacity=2, refill_rate=1, time_func=clock)
    assert bucket.allow()
    assert bucket.allow()
    assert not bucket.allow()
    clock.advance(1.0)  # one token back
    assert bucket.allow()
    assert not bucket.allow()


def test_token_bucket_caps_at_capacity() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(capacity=2, refill_rate=10, time_func=clock)
    clock.advance(100)  # would overflow without cap
    assert bucket.tokens == 2


def test_token_bucket_zero_cost_always_allowed() -> None:
    clock = _FakeClock()
    bucket = TokenBucket(capacity=1, refill_rate=1, time_func=clock)
    assert bucket.allow()
    assert not bucket.allow(1)
    assert bucket.allow(0)


def test_token_bucket_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, refill_rate=1)
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, refill_rate=0)
