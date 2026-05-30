"""LiteLLM proxy process management."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import psutil
else:
    import psutil

from trifle_proxy import metrics
from trifle_proxy.logging_config import get_logger
from trifle_proxy.resilience import RetryError, RetryPolicy, retry_call

log = get_logger("trifle_proxy.proxy")

PID_FILE = Path("logs/litellm.pid")
LOG_FILE = Path("logs/litellm.log")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4000
DEFAULT_CONFIG = Path("litellm.yaml")

# Seconds to let the proxy drain in-flight requests after SIGTERM before we
# force-kill it.
GRACEFUL_TIMEOUT = 10.0


# Tolerance (seconds) when comparing a process's recorded vs. live creation
# time. psutil reports create_time with sub-second precision; a small epsilon
# absorbs float round-tripping through the PID file.
_CREATE_TIME_EPSILON = 0.5


def _read_pidfile() -> tuple[int, float | None] | None:
    """Parse the PID file into ``(pid, create_time)``.

    The file holds the PID on the first line and, when known, the process
    creation time on the second. ``create_time`` is ``None`` for legacy
    single-line files. Returns ``None`` if the file is missing or unparseable.
    """
    try:
        raw = PID_FILE.read_text()
    except OSError:
        return None
    lines = raw.splitlines()
    if not lines:
        return None
    try:
        pid = int(lines[0].strip())
    except ValueError:
        return None
    create_time: float | None = None
    if len(lines) > 1:
        try:
            create_time = float(lines[1].strip())
        except ValueError:
            create_time = None
    return pid, create_time


def _record_pid(pid: int) -> None:
    """Persist ``pid`` plus its creation time so we can detect PID reuse later."""
    create_time: float | None = None
    try:
        create_time = psutil.Process(pid).create_time()
    except psutil.Error:
        create_time = None
    if create_time is None:
        PID_FILE.write_text(str(pid))
    else:
        PID_FILE.write_text(f"{pid}\n{create_time!r}")


def _looks_like_proxy(proc: psutil.Process) -> bool:
    """Best-effort check that ``proc`` is a LiteLLM proxy we launched.

    Used as a weaker identity signal than creation time, so it must be specific
    enough not to claim an unrelated process that merely mentions ``litellm`` in
    its arguments (``vim litellm.yaml``, ``tail logs/litellm.log``). Both forms
    ``start`` spawns — ``uvx ... litellm`` and ``python -m litellm`` — carry
    ``litellm`` as a *standalone* argv token (not a substring of a filename) and
    always append our ``--config``/``--host``/``--port`` flags. We require both
    signals so a stranger that inherited a legacy (creation-time-less) PID is
    treated as not ours.
    """
    try:
        cmdline = proc.cmdline()
    except psutil.Error:
        return False
    has_litellm_token = any(
        part == "litellm" or part.rsplit("/", 1)[-1] == "litellm" for part in cmdline
    )
    has_proxy_flags = "--config" in cmdline and "--port" in cmdline
    return has_litellm_token and has_proxy_flags


def _live_proc(pid: int, create_time: float | None) -> psutil.Process | None:
    """Return the live ``psutil.Process`` for our proxy, or ``None``.

    Returns ``None`` if the PID is gone, a zombie, inaccessible, or — crucially
    — has been *reused* by an unrelated process. Reuse is detected via a
    creation-time mismatch when we recorded one; for legacy/fallback single-line
    PID files that carry no creation time, we fall back to a command-line check
    so ``stop`` never terminates a stranger that inherited the recorded PID.
    """
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return None
        if create_time is not None:
            try:
                if abs(proc.create_time() - create_time) > _CREATE_TIME_EPSILON:
                    # PID was recycled by a different process — not ours.
                    return None
            except psutil.Error:
                return None
        elif not _looks_like_proxy(proc):
            # No recorded creation time to compare against (legacy/fallback PID
            # file). Without the command-line signal we can't trust the PID, so
            # refuse to claim — and later kill — an unrelated process.
            return None
        return proc
    except psutil.Error:
        # NoSuchProcess, ZombieProcess, AccessDenied (PID reused by another
        # user's process), etc. — treat any as "not our running proxy".
        return None


def is_running() -> bool:
    """Check if our proxy process is alive (and still the one we started)."""
    record = _read_pidfile()
    if record is None:
        return False
    pid, create_time = record
    return _live_proc(pid, create_time) is not None


def start(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    config: Path = DEFAULT_CONFIG,
) -> int:
    """Start LiteLLM proxy and return PID."""
    if is_running():
        log.error("start_failed", reason="already_running")
        raise RuntimeError("Proxy is already running")

    if not config.exists():
        log.error("start_failed", reason="config_not_found", config=str(config))
        raise FileNotFoundError(f"Config file not found: {config}")

    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        log.error("start_failed", reason="master_key_not_set")
        raise RuntimeError("LITELLM_MASTER_KEY is not set")

    log.info("starting", host=host, port=port, config=str(config))

    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Use uvx if available, otherwise fall back to python -m litellm
    uvx_cmd = ["uvx", "--python", "3.12", "--from", "litellm[proxy]", "litellm"]
    python_cmd = [sys.executable, "-m", "litellm"]

    cmd = uvx_cmd if _has_uvx() else python_cmd
    cmd += [
        "--config",
        str(config),
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Open log file and start process
    with LOG_FILE.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    _record_pid(proc.pid)
    metrics.record_start()
    log.info("started", pid=proc.pid, host=host, port=port)
    return proc.pid


def start_resilient(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    config: Path = DEFAULT_CONFIG,
    *,
    policy: RetryPolicy | None = None,
    on_failure: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Start the proxy with retry/backoff, rolling back on terminal failure.

    Transient errors (the spawned process not coming up, transient OS errors)
    are retried per ``policy``. Deterministic misconfigurations
    (already-running, missing config, missing master key) are *not* retried —
    they raise immediately. If every retry is exhausted, ``on_failure`` is run
    (e.g. to unwire Claude Code) before the error propagates.
    """
    policy = policy or RetryPolicy()

    def _attempt() -> int:
        return start(host=host, port=port, config=config)

    try:
        return retry_call(
            _attempt,
            policy=policy,
            retry_on=(OSError,),
            sleep=sleep,
        )
    except (RetryError, RuntimeError, FileNotFoundError):
        if on_failure is not None:
            log.warning("start_resilient_cleanup")
            on_failure()
        raise


def stop(timeout: float = GRACEFUL_TIMEOUT) -> bool:
    """Gracefully stop LiteLLM proxy.

    Sends SIGTERM and waits up to ``timeout`` seconds for the proxy to drain
    in-flight requests and exit on its own. If it does not exit in time, it is
    force-killed (SIGKILL).

    Returns True if a running proxy was stopped, False if it was not running.
    """
    if not is_running():
        PID_FILE.unlink(missing_ok=True)
        log.info("stop_noop", reason="not_running")
        return False

    record = _read_pidfile()
    if record is None:
        PID_FILE.unlink(missing_ok=True)
        log.warning("stop_failed", reason="bad_pidfile")
        return False
    pid, create_time = record

    log.info("stopping", pid=pid, grace_timeout=timeout)
    proc = _live_proc(pid, create_time)
    if proc is None:
        # Raced with the process exiting (or the PID was reused) between
        # is_running() and here. Either way there is nothing of ours to kill.
        PID_FILE.unlink(missing_ok=True)
        metrics.record_stop()
        log.info("stopped", pid=pid, note="already_gone")
        return True

    try:
        proc.terminate()
        proc.wait(timeout=timeout)
        log.info("drained", pid=pid)
    except psutil.NoSuchProcess:
        log.info("stopped", pid=pid, note="exited_during_terminate")
    except psutil.TimeoutExpired:
        log.warning("stop_force_kill", pid=pid)
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.kill()

    PID_FILE.unlink(missing_ok=True)
    metrics.record_stop()
    log.info("stopped", pid=pid)
    return True


def install_signal_handlers(cleanup: Callable[[], None]) -> None:
    """Install SIGTERM/SIGINT handlers that run ``cleanup`` then exit.

    Lets a foreground caller drain the proxy and roll back any wiring when it
    receives a termination signal. Safe to call only from the main thread;
    signal registration failures (e.g. running off-thread) are logged and
    swallowed so they never crash the caller.
    """

    def _handler(signum: int, _frame: object) -> None:
        log.info("signal_received", signal=signal.Signals(signum).name)
        try:
            cleanup()
        finally:
            # Re-raise with the default disposition so exit codes are correct.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as e:  # not main thread / unsupported
            log.warning("signal_install_failed", signal=sig, error=str(e))


def get_status() -> dict[str, str | bool]:
    """Get proxy status info."""
    running = is_running()
    status: dict[str, str | bool] = {
        "running": running,
        "pid": "",
        "url": "",
        "log": str(LOG_FILE),
    }
    if running:
        record = _read_pidfile()
        if record is not None:
            status["pid"] = str(record[0])
            status["url"] = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    return status


def tail_logs() -> None:
    """Tail proxy logs to stdout."""
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file does not exist yet: {LOG_FILE}")
    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(["tail", "-f", str(LOG_FILE)], check=False)


def _has_uvx() -> bool:
    """Check if uvx is available.

    Returns False (rather than raising) when ``uvx`` is not on PATH, so
    ``start`` falls back to ``python -m litellm`` instead of crashing.
    """
    try:
        return (
            subprocess.run(
                ["uvx", "--version"],
                capture_output=True,
            ).returncode
            == 0
        )
    except OSError:
        return False
