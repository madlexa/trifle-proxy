"""Tests for proxy.py — LiteLLM process management (mocked psutil/subprocess)."""

from __future__ import annotations

import signal
from pathlib import Path

import psutil
import pytest

from trifle_proxy import proxy


@pytest.fixture
def fake_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect PID/LOG files into a temp dir."""
    monkeypatch.setattr(proxy, "PID_FILE", tmp_path / "logs" / "litellm.pid")
    monkeypatch.setattr(proxy, "LOG_FILE", tmp_path / "logs" / "litellm.log")
    return tmp_path


class FakeProc:
    def __init__(
        self,
        pid: int = 1234,
        running: bool = True,
        status: str = "running",
        create_time: float = 1000.0,
        cmdline: list[str] | None = None,
    ):
        self.pid = pid
        self._running = running
        self._status = status
        self._create_time = create_time
        # Default to a command line that looks like a proxy we launched, so the
        # legacy single-line PID path treats us as ours unless told otherwise.
        self._cmdline = (
            cmdline
            if cmdline is not None
            else [
                "python",
                "-m",
                "litellm",
                "--config",
                "x.yaml",
                "--host",
                "127.0.0.1",
                "--port",
                "4000",
            ]
        )
        self.terminated = False
        self.killed = False
        self.waited = False

    def is_running(self) -> bool:
        return self._running

    def status(self) -> str:
        return self._status

    def create_time(self) -> float:
        return self._create_time

    def cmdline(self) -> list[str]:
        return self._cmdline

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> None:
        self.waited = True
        self.wait_timeout = timeout


# --- is_running ---------------------------------------------------------


def test_is_running_no_pidfile(fake_paths: Path) -> None:
    assert proxy.is_running() is False


def test_is_running_bad_pidfile(fake_paths: Path) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("not-a-number")
    assert proxy.is_running() is False


def test_is_running_true(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid))
    assert proxy.is_running() is True


def test_is_running_zombie(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(
        proxy.psutil,
        "Process",
        lambda pid: FakeProc(pid, status=psutil.STATUS_ZOMBIE),
    )
    assert proxy.is_running() is False


def test_is_running_no_such_process(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")

    def raise_nosuch(pid: int):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(proxy.psutil, "Process", raise_nosuch)
    assert proxy.is_running() is False


def test_is_running_access_denied(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")

    def raise_denied(pid: int):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(proxy.psutil, "Process", raise_denied)
    assert proxy.is_running() is False


def test_is_running_pid_reused_create_time_mismatch(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PID file records create_time 1000.0, but the live process with that PID
    # was created at 2000.0 — i.e. the PID was recycled by a stranger.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234\n1000.0")
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid, create_time=2000.0))
    assert proxy.is_running() is False


def test_is_running_create_time_matches(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234\n1000.0")
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid, create_time=1000.0))
    assert proxy.is_running() is True


def test_stop_does_not_kill_reused_pid(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # is_running() is forced True, but the recorded create_time no longer
    # matches the live process: stop must NOT terminate the stranger.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234\n1000.0")
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    stranger = FakeProc(1234, create_time=2000.0)
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: stranger)

    assert proxy.stop() is True  # treated as already-gone
    assert stranger.terminated is False
    assert stranger.killed is False
    assert not proxy.PID_FILE.exists()


def test_is_running_legacy_pidfile_verifies_cmdline(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Single-line (legacy/fallback) PID file: no create_time to compare, so the
    # command line is what identifies the process as ours.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid))
    assert proxy.is_running() is True


def test_is_running_legacy_pidfile_reused_by_stranger(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Single-line PID file but the live PID belongs to an unrelated process
    # whose command line is nothing like our proxy — must not be claimed.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(
        proxy.psutil,
        "Process",
        lambda pid: FakeProc(pid, cmdline=["sshd", "-D"]),
    )
    assert proxy.is_running() is False


def test_is_running_legacy_pidfile_substring_match_rejected(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Legacy PID file reused by a process that only *mentions* litellm as a
    # filename substring (e.g. an editor or log tail). The old substring match
    # accepted these; the token + proxy-flag check must reject them.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    for impostor in (
        ["vim", "litellm.yaml"],
        ["tail", "-f", "logs/litellm.log"],
        ["grep", "-r", "litellm", "."],
    ):
        monkeypatch.setattr(
            proxy.psutil, "Process", lambda pid, c=impostor: FakeProc(pid, cmdline=c)
        )
        assert proxy.is_running() is False, impostor


def test_is_running_legacy_pidfile_uvx_launch_shape(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The uvx launch form must still be recognized via its standalone litellm
    # token plus proxy flags.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    uvx_cmd = [
        "uvx",
        "--python",
        "3.12",
        "--from",
        "litellm[proxy]",
        "litellm",
        "--config",
        "litellm.yaml",
        "--host",
        "127.0.0.1",
        "--port",
        "4000",
    ]
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid, cmdline=uvx_cmd))
    assert proxy.is_running() is True


def test_stop_does_not_kill_reused_legacy_pid(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Legacy single-line PID file whose PID was recycled by a stranger: stop
    # must not terminate it just because no create_time was recorded.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    stranger = FakeProc(1234, cmdline=["postgres", "-D", "/var/lib/pg"])
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: stranger)

    assert proxy.stop() is True  # treated as already-gone
    assert stranger.terminated is False
    assert stranger.killed is False
    assert not proxy.PID_FILE.exists()


def test_record_pid_writes_create_time(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid, create_time=1234.5))
    proxy._record_pid(4321)
    assert proxy.PID_FILE.read_text() == "4321\n1234.5"


def test_record_pid_falls_back_when_proc_gone(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    def raise_nosuch(pid: int):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(proxy.psutil, "Process", raise_nosuch)
    proxy._record_pid(4321)
    # No create_time available — single-line file, still usable by legacy path.
    assert proxy.PID_FILE.read_text() == "4321"


# --- start --------------------------------------------------------------


def test_start_already_running(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    with pytest.raises(RuntimeError, match="already running"):
        proxy.start()


def test_start_missing_config(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    with pytest.raises(FileNotFoundError):
        proxy.start(config=fake_paths / "missing.yaml")


def test_start_missing_master_key(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    config = fake_paths / "litellm.yaml"
    config.write_text("model_list: []")
    with pytest.raises(RuntimeError, match="LITELLM_MASTER_KEY"):
        proxy.start(config=config)


def test_start_success(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "_has_uvx", lambda: False)
    config = fake_paths / "litellm.yaml"
    config.write_text("model_list: []")

    captured: dict = {}

    class DummyPopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            self.pid = 4321

    monkeypatch.setattr(proxy.subprocess, "Popen", DummyPopen)
    # The just-spawned PID is fictional in tests; record it without a
    # create_time so the assertion below is deterministic.
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid, create_time=42.0))

    pid = proxy.start(host="0.0.0.0", port=5000, config=config)

    assert pid == 4321
    assert proxy.PID_FILE.read_text() == "4321\n42.0"
    assert "--port" in captured["cmd"]
    assert "5000" in captured["cmd"]
    assert "0.0.0.0" in captured["cmd"]
    assert captured["kwargs"]["start_new_session"] is True


def test_start_uses_uvx_when_available(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test")
    monkeypatch.setattr(proxy, "_has_uvx", lambda: True)
    config = fake_paths / "litellm.yaml"
    config.write_text("model_list: []")

    captured: dict = {}

    class DummyPopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.pid = 99

    monkeypatch.setattr(proxy.subprocess, "Popen", DummyPopen)
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: FakeProc(pid, create_time=42.0))
    proxy.start(config=config)
    assert captured["cmd"][0] == "uvx"


# --- start_resilient ----------------------------------------------------


def test_start_resilient_succeeds_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "start", lambda **kw: 4321)
    pid = proxy.start_resilient(sleep=lambda _: None)
    assert pid == 4321


def test_start_resilient_retries_transient_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"n": 0}

    def flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise OSError("address in use")
        return 99

    monkeypatch.setattr(proxy, "start", flaky)
    pid = proxy.start_resilient(sleep=lambda _: None)
    assert pid == 99
    assert attempts["n"] == 2


def test_start_resilient_runs_cleanup_on_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(**kw):
        raise OSError("nope")

    monkeypatch.setattr(proxy, "start", always_fail)
    cleaned = {"done": False}

    from trifle_proxy.resilience import RetryError, RetryPolicy

    with pytest.raises(RetryError):
        proxy.start_resilient(
            policy=RetryPolicy(max_attempts=2),
            on_failure=lambda: cleaned.__setitem__("done", True),
            sleep=lambda _: None,
        )
    assert cleaned["done"] is True


def test_start_resilient_does_not_retry_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"n": 0}

    def already_running(**kw):
        attempts["n"] += 1
        raise RuntimeError("Proxy is already running")

    monkeypatch.setattr(proxy, "start", already_running)
    cleaned = {"done": False}

    with pytest.raises(RuntimeError):
        proxy.start_resilient(
            on_failure=lambda: cleaned.__setitem__("done", True),
            sleep=lambda _: None,
        )
    assert attempts["n"] == 1  # not retried
    assert cleaned["done"] is True  # but cleanup still runs


# --- stop ---------------------------------------------------------------


def test_stop_not_running(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    assert proxy.stop() is False


def test_stop_running_terminates(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    fake = FakeProc(1234)
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: fake)

    assert proxy.stop() is True
    assert fake.terminated is True
    assert not proxy.PID_FILE.exists()


def test_stop_force_kills_on_timeout(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)

    fake = FakeProc(1234)

    def wait_timeout(timeout=None):
        raise psutil.TimeoutExpired(timeout, pid=1234)

    fake.wait = wait_timeout  # type: ignore[method-assign]
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: fake)

    assert proxy.stop() is True
    assert fake.killed is True


def test_stop_process_vanishes_before_lookup(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # is_running() passes, then the process exits before psutil.Process(pid).
    # Must not raise NameError on the force-kill path; should clean up and return True.
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)

    def raise_nosuch(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(proxy.psutil, "Process", raise_nosuch)

    assert proxy.stop() is True
    assert not proxy.PID_FILE.exists()


def test_stop_process_exits_during_terminate(
    fake_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)

    fake = FakeProc(1234)

    def wait_gone(timeout=None):
        raise psutil.NoSuchProcess(1234)

    fake.wait = wait_gone  # type: ignore[method-assign]
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: fake)

    assert proxy.stop() is True
    assert fake.killed is False  # already gone, no force-kill needed
    assert not proxy.PID_FILE.exists()


def test_stop_bad_pidfile(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("garbage")
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    assert proxy.stop() is False
    assert not proxy.PID_FILE.exists()


def test_stop_uses_grace_timeout(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("1234")
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    fake = FakeProc(1234)
    monkeypatch.setattr(proxy.psutil, "Process", lambda pid: fake)

    assert proxy.stop(timeout=3.5) is True
    # The drain wait should use the supplied grace timeout.
    assert fake.wait_timeout == 3.5


# --- install_signal_handlers --------------------------------------------


def test_install_signal_handlers_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: dict[int, object] = {}
    monkeypatch.setattr(
        proxy.signal,
        "signal",
        lambda sig, handler: registered.__setitem__(sig, handler),
    )
    proxy.install_signal_handlers(lambda: None)
    assert signal.SIGTERM in registered
    assert signal.SIGINT in registered


def test_install_signal_handlers_runs_cleanup_then_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[int, object] = {}
    monkeypatch.setattr(
        proxy.signal,
        "signal",
        lambda sig, handler: captured.setdefault(sig, handler),
    )
    killed: dict[str, int] = {}
    monkeypatch.setattr(proxy.os, "kill", lambda pid, sig: killed.__setitem__("sig", sig))
    monkeypatch.setattr(proxy.os, "getpid", lambda: 999)

    cleaned = {"done": False}
    proxy.install_signal_handlers(lambda: cleaned.__setitem__("done", True))

    handler = captured[signal.SIGTERM]
    handler(signal.SIGTERM, None)  # type: ignore[operator]

    assert cleaned["done"] is True
    assert killed["sig"] == signal.SIGTERM


def test_install_signal_handlers_swallows_registration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(sig, handler):
        raise ValueError("not main thread")

    monkeypatch.setattr(proxy.signal, "signal", boom)
    # Should not raise.
    proxy.install_signal_handlers(lambda: None)


# --- get_status ---------------------------------------------------------


def test_get_status_stopped(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "is_running", lambda: False)
    status = proxy.get_status()
    assert status["running"] is False
    assert status["pid"] == ""


def test_get_status_running(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.PID_FILE.write_text("777")
    monkeypatch.setattr(proxy, "is_running", lambda: True)
    status = proxy.get_status()
    assert status["running"] is True
    assert status["pid"] == "777"
    assert status["url"].startswith("http://")


# --- tail_logs ----------------------------------------------------------


def test_tail_logs_missing(fake_paths: Path) -> None:
    with pytest.raises(FileNotFoundError):
        proxy.tail_logs()


def test_tail_logs_invokes_tail(fake_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proxy.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    proxy.LOG_FILE.write_text("log line")
    called: dict = {}

    def fake_run(cmd, check=False):
        called["cmd"] = cmd

    monkeypatch.setattr(proxy.subprocess, "run", fake_run)
    proxy.tail_logs()
    assert called["cmd"][0] == "tail"


# --- _has_uvx -----------------------------------------------------------


def test_has_uvx_true(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 0

    monkeypatch.setattr(proxy.subprocess, "run", lambda *a, **k: Result())
    assert proxy._has_uvx() is True


def test_has_uvx_false(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1

    monkeypatch.setattr(proxy.subprocess, "run", lambda *a, **k: Result())
    assert proxy._has_uvx() is False


def test_has_uvx_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> object:
        raise FileNotFoundError("uvx")

    monkeypatch.setattr(proxy.subprocess, "run", boom)
    # Must degrade to False so start() falls back to python -m litellm.
    assert proxy._has_uvx() is False
