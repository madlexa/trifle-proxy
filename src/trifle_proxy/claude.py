"""Manage Claude Code global settings.json."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from trifle_proxy.logging_config import get_logger
from trifle_proxy.security import file_checksum, verify_backup_integrity

log = get_logger("trifle_proxy.claude")

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
BACKUP_DIR = Path.home() / ".claude" / "backups"
BACKUP_KEY = "_trifle_proxy_backup"
BACKUP_CHECKSUM_KEY = "_trifle_proxy_backup_sha256"


def _load_settings() -> dict[str, Any]:
    """Load Claude Code settings.json.

    A missing or corrupt file degrades to an empty dict (logged) so that
    recovery paths — ``rollback`` / ``unwire`` / ``is_wired`` — never crash on
    malformed JSON. ``wire`` always backs up the raw file first, so the original
    bytes are preserved even when they don't parse.
    """
    if not CLAUDE_SETTINGS.exists():
        return {}
    try:
        with CLAUDE_SETTINGS.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.error("settings_load_failed", path=str(CLAUDE_SETTINGS), error=str(e))
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _save_settings(settings: dict[str, Any]) -> None:
    """Save Claude Code settings.json atomically.

    Writes to a temp file in the same directory and renames it into place so an
    interrupted write can never truncate the user's live settings file.
    """
    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=CLAUDE_SETTINGS.parent, prefix=".settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_name, CLAUDE_SETTINGS)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def wire(env: dict[str, str]) -> Path:
    """Add env vars to Claude Code settings.json and return backup path."""
    # Backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    backup_path = BACKUP_DIR / f"settings.backup.{timestamp}.json"
    checksum: str | None = None
    if CLAUDE_SETTINGS.exists():
        shutil.copy2(CLAUDE_SETTINGS, backup_path)
        checksum = file_checksum(backup_path)
        log.info("backup_created", path=str(backup_path))

    settings = _load_settings()

    # Remove apiKeyHelper to avoid auth conflict with ANTHROPIC_AUTH_TOKEN
    settings.pop("apiKeyHelper", None)

    # Store backup path (and integrity checksum) for restore on unwire
    settings[BACKUP_KEY] = str(backup_path)
    if checksum is not None:
        settings[BACKUP_CHECKSUM_KEY] = checksum
    settings["env"] = env

    _save_settings(settings)
    log.info("wired", settings=str(CLAUDE_SETTINGS))
    return backup_path


def unwire() -> None:
    """Restore Claude Code settings.json from backup."""
    settings = _load_settings()
    backup_path_str = settings.pop(BACKUP_KEY, None)
    expected_checksum = settings.pop(BACKUP_CHECKSUM_KEY, None)

    if backup_path_str:
        backup_path = Path(backup_path_str)
        if backup_path.exists():
            # Refuse to restore a backup that was tampered with on disk.
            if expected_checksum and not verify_backup_integrity(backup_path, expected_checksum):
                log.error("restore_backup_corrupt", backup=backup_path_str)
            else:
                shutil.copy2(backup_path, CLAUDE_SETTINGS)
                log.info("restored", backup=str(backup_path))
                return
        else:
            log.warning("restore_backup_missing", backup=backup_path_str)

    # Fallback: just remove env if no (usable) backup
    settings.pop("env", None)
    _save_settings(settings)
    log.info("unwired", fallback=True)


@contextlib.contextmanager
def wired(env: dict[str, str]) -> Iterator[Path]:
    """Wire on enter; automatically roll back if the wrapped block fails.

    Used to make the start sequence transactional: if anything after wiring
    (e.g. confirming the proxy is healthy) raises, Claude Code's settings are
    restored instead of being left pointed at a dead proxy. Catches
    ``BaseException`` so Ctrl-C also triggers rollback.
    """
    backup = wire(env)
    try:
        yield backup
    except BaseException:
        log.error("wired_block_failed", reason="rolling_back")
        try:
            rollback()
        except Exception as exc:  # pragma: no cover - defensive
            log.error("rollback_failed", error=str(exc))
        raise


def is_wired() -> bool:
    """Check if Claude Code is currently wired by trifle-proxy."""
    # Key solely on the trifle-owned marker. Matching a bare ``env`` key would
    # treat a user's own unrelated ``env`` block as trifle-wired and strip it
    # on stop/rollback. ``wire`` always sets BACKUP_KEY, so this is sufficient.
    settings = _load_settings()
    return BACKUP_KEY in settings


def rollback() -> bool:
    """Restore Claude Code settings if they were wired by trifle-proxy.

    Idempotent crash-recovery primitive: if the proxy dies and leaves Claude
    Code pointed at a dead endpoint, this restores the pre-wire settings.
    Returns True if a rollback was performed, False if nothing was wired.
    """
    if not is_wired():
        return False
    log.warning("rollback", reason="restoring_settings")
    unwire()
    return True
