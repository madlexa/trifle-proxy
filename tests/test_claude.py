"""Tests for claude.py — Claude Code settings.json manipulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trifle_proxy import claude


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect claude module paths into a temporary home directory."""
    settings = tmp_path / ".claude" / "settings.json"
    backups = tmp_path / ".claude" / "backups"
    monkeypatch.setattr(claude, "CLAUDE_SETTINGS", settings)
    monkeypatch.setattr(claude, "BACKUP_DIR", backups)
    return tmp_path


def _write_settings(data: dict) -> None:
    claude.CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    claude.CLAUDE_SETTINGS.write_text(json.dumps(data), encoding="utf-8")


def test_load_settings_missing_returns_empty(fake_home: Path) -> None:
    assert claude._load_settings() == {}


def test_load_settings_corrupt_json_returns_empty(fake_home: Path) -> None:
    claude.CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    claude.CLAUDE_SETTINGS.write_text("{not valid json", encoding="utf-8")
    assert claude._load_settings() == {}


def test_load_settings_non_dict_json_returns_empty(fake_home: Path) -> None:
    claude.CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    claude.CLAUDE_SETTINGS.write_text("[1, 2, 3]", encoding="utf-8")
    assert claude._load_settings() == {}


def test_recovery_paths_survive_corrupt_settings(fake_home: Path) -> None:
    claude.CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    claude.CLAUDE_SETTINGS.write_text("}}corrupt{{", encoding="utf-8")
    # None of the crash-recovery primitives should raise on malformed JSON.
    assert claude.is_wired() is False
    assert claude.rollback() is False


def test_save_and_load_roundtrip(fake_home: Path) -> None:
    claude._save_settings({"foo": "bar"})
    assert claude._load_settings() == {"foo": "bar"}
    # trailing newline written
    assert claude.CLAUDE_SETTINGS.read_text(encoding="utf-8").endswith("\n")


def test_is_wired_false_when_no_settings(fake_home: Path) -> None:
    assert claude.is_wired() is False


def test_wire_creates_backup_and_sets_env(fake_home: Path) -> None:
    _write_settings({"existing": "value", "apiKeyHelper": "something"})
    env = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:4000"}

    backup_path = claude.wire(env)

    assert backup_path.exists()
    settings = claude._load_settings()
    assert settings["env"] == env
    # apiKeyHelper removed to avoid auth conflicts
    assert "apiKeyHelper" not in settings
    # backup pointer stored
    assert settings[claude.BACKUP_KEY] == str(backup_path)
    # original settings preserved in backup
    backup_data = json.loads(backup_path.read_text(encoding="utf-8"))
    assert backup_data["existing"] == "value"
    assert backup_data["apiKeyHelper"] == "something"


def test_wire_without_existing_settings(fake_home: Path) -> None:
    env = {"X": "1"}
    backup_path = claude.wire(env)
    # No prior settings.json -> no backup file copied, but pointer still set
    assert not backup_path.exists()
    settings = claude._load_settings()
    assert settings["env"] == env
    assert claude.is_wired() is True


def test_is_wired_true_after_wire(fake_home: Path) -> None:
    claude.wire({"X": "1"})
    assert claude.is_wired() is True


def test_unwire_restores_from_backup(fake_home: Path) -> None:
    _write_settings({"original": True, "model": "opus"})
    claude.wire({"ANTHROPIC_MODEL": "kimi"})

    # Wired state differs from original
    assert "env" in claude._load_settings()

    claude.unwire()

    restored = claude._load_settings()
    assert restored == {"original": True, "model": "opus"}
    assert "env" not in restored
    assert claude.BACKUP_KEY not in restored


def test_unwire_fallback_without_backup(fake_home: Path) -> None:
    # Settings have env but no backup pointer and no backup file
    _write_settings({"env": {"X": "1"}, "keep": "me"})
    claude.unwire()
    settings = claude._load_settings()
    assert "env" not in settings
    assert settings["keep"] == "me"


def test_unwire_backup_pointer_missing_file(fake_home: Path) -> None:
    # Backup pointer references a non-existent file -> fallback path
    _write_settings(
        {claude.BACKUP_KEY: str(fake_home / "gone.json"), "env": {"X": "1"}, "keep": "y"}
    )
    claude.unwire()
    settings = claude._load_settings()
    assert "env" not in settings
    assert settings["keep"] == "y"


def test_wire_unwire_full_cycle(fake_home: Path) -> None:
    _write_settings({"theme": "dark"})
    claude.wire({"ANTHROPIC_BASE_URL": "http://localhost:4000"})
    assert claude.is_wired() is True
    claude.unwire()
    assert claude.is_wired() is False
    assert claude._load_settings() == {"theme": "dark"}


def test_wire_stores_backup_checksum(fake_home: Path) -> None:
    _write_settings({"original": True})
    claude.wire({"X": "1"})
    settings = claude._load_settings()
    assert claude.BACKUP_CHECKSUM_KEY in settings
    assert len(settings[claude.BACKUP_CHECKSUM_KEY]) == 64


def test_unwire_refuses_corrupt_backup(fake_home: Path) -> None:
    _write_settings({"original": True, "model": "opus"})
    backup_path = claude.wire({"ANTHROPIC_MODEL": "kimi"})

    # Tamper with the backup on disk after the checksum was recorded.
    backup_path.write_text(json.dumps({"malicious": "payload"}), encoding="utf-8")

    claude.unwire()

    restored = claude._load_settings()
    # Corrupt backup must NOT be restored; fallback strips env instead.
    assert restored.get("malicious") is None
    assert "env" not in restored


def test_wired_context_keeps_settings_on_success(fake_home: Path) -> None:
    _write_settings({"theme": "dark"})
    with claude.wired({"ANTHROPIC_BASE_URL": "http://localhost:4000"}) as backup:
        assert isinstance(backup, Path)
        assert claude.is_wired() is True
    # Block succeeded -> wiring stays in place.
    assert claude.is_wired() is True


def test_wired_context_rolls_back_on_error(fake_home: Path) -> None:
    _write_settings({"theme": "dark"})
    with (
        pytest.raises(RuntimeError),
        claude.wired({"ANTHROPIC_BASE_URL": "http://localhost:4000"}),
    ):
        assert claude.is_wired() is True
        raise RuntimeError("startup failed")
    # Block raised -> settings restored to pre-wire state.
    assert claude.is_wired() is False
    assert claude._load_settings() == {"theme": "dark"}


def test_rollback_noop_when_not_wired(fake_home: Path) -> None:
    assert claude.rollback() is False


def test_rollback_restores_when_wired(fake_home: Path) -> None:
    _write_settings({"theme": "dark"})
    claude.wire({"ANTHROPIC_BASE_URL": "http://localhost:4000"})
    assert claude.is_wired() is True

    assert claude.rollback() is True
    assert claude.is_wired() is False
    assert claude._load_settings() == {"theme": "dark"}
