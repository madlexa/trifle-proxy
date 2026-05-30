"""Integration tests for the CLI command handlers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from trifle_proxy import cli


def ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


# --- cmd_init -----------------------------------------------------------


def test_cmd_init_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "litellm.yaml"
    rc = cli.cmd_init(ns(config=str(config), template=True, force=True))
    assert rc == 0
    assert config.exists()
    text = config.read_text()
    assert "model_list" in text


def test_cmd_init_template_has_valid_resilience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from trifle_proxy.config import get_resilience_config

    monkeypatch.chdir(tmp_path)
    config = tmp_path / "litellm.yaml"
    rc = cli.cmd_init(ns(config=str(config), template=True, force=True))
    assert rc == 0

    parsed = yaml.safe_load(config.read_text())
    # LiteLLM-native router resilience.
    assert "router_settings" in parsed
    assert parsed["router_settings"]["num_retries"] == 3
    # trifle-proxy resilience parses back into a ResilienceConfig.
    res = get_resilience_config(parsed)
    assert res.retry.max_attempts == 3
    assert res.circuit.failure_threshold == 5
    assert "kimi-k2.5" in res.fallback_models


def test_cmd_init_abort_on_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "litellm.yaml"
    config.write_text("existing")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    rc = cli.cmd_init(ns(config=str(config), template=True, force=False))
    assert rc == 1
    assert config.read_text() == "existing"


def test_cmd_init_overwrite_confirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "litellm.yaml"
    config.write_text("old")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    rc = cli.cmd_init(ns(config=str(config), template=True, force=False))
    assert rc == 0
    assert "model_list" in config.read_text()


def test_cmd_init_interactive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "litellm.yaml"
    # Sequence: pick provider 1, model id, alias, role, then 5 to finish
    answers = iter(["1", "kimi-k2.5", "kimi-k2.5", "sonnet", "5"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    rc = cli.cmd_init(ns(config=str(config), template=False, force=True))
    assert rc == 0
    assert config.exists()
    assert "kimi-k2.5" in config.read_text()


def test_cmd_init_interactive_no_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "litellm.yaml"
    answers = iter(["5"])  # immediately done, no models
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    rc = cli.cmd_init(ns(config=str(config), template=False, force=True))
    assert rc == 1
    assert not config.exists()


# --- cmd_start ----------------------------------------------------------


def test_cmd_start_missing_config(tmp_path: Path) -> None:
    rc = cli.cmd_start(
        ns(config=str(tmp_path / "none.yaml"), host="127.0.0.1", port=4000, mode="claude")
    )
    assert rc == 1


def _valid_config(tmp_path: Path) -> Path:
    config = tmp_path / "litellm.yaml"
    config.write_text(
        """
model_list:
  - model_name: kimi
    litellm_params:
      model: openai/kimi
    model_info:
      claude_role: sonnet
"""
    )
    return config


def test_cmd_start_claude_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_config(tmp_path)
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)
    wired: dict = {}
    monkeypatch.setattr(cli, "wire", lambda env: wired.setdefault("env", env) or Path("backup"))
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="claude"))
    assert rc == 0
    assert "ANTHROPIC_BASE_URL" in wired["env"]


def test_cmd_start_wire_failure_stops_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _valid_config(tmp_path)
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)

    def boom(env: dict) -> Path:
        raise OSError("settings.json is read-only")

    stopped: dict = {}
    monkeypatch.setattr(cli, "wire", boom)
    monkeypatch.setattr(cli, "stop", lambda: stopped.setdefault("done", True))
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="claude"))
    assert rc == 1
    # The just-started proxy must be stopped so nothing is left pointing at it.
    assert stopped["done"] is True


def test_cmd_start_envrc_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _valid_config(tmp_path)
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)
    monkeypatch.setattr(cli, "_direnv_allow", lambda: True)
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="envrc"))
    assert rc == 0
    assert (tmp_path / ".envrc").exists()
    assert "ANTHROPIC_BASE_URL" in (tmp_path / ".envrc").read_text()


def test_cmd_start_envrc_preserves_existing_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _valid_config(tmp_path)
    (tmp_path / ".envrc").write_text("use flake\nexport MY_VAR=1\n")
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)
    monkeypatch.setattr(cli, "_direnv_allow", lambda: True)
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="envrc"))
    assert rc == 0
    text = (tmp_path / ".envrc").read_text()
    # Pre-existing direnv setup survives; trifle block is appended.
    assert "use flake" in text
    assert "export MY_VAR=1" in text
    assert ">>> trifle-proxy >>>" in text
    assert "ANTHROPIC_BASE_URL" in text


@pytest.mark.skipif(sys.platform == "win32", reason="Unix shell test only")
def test_env_block_quotes_shell_metacharacters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    envrc = tmp_path / ".envrc"
    cli._upsert_env_block(envrc, {"ANTHROPIC_MODEL": "$(touch pwned)"})
    text = envrc.read_text()
    # The value must be quoted so command substitution cannot fire when sourced.
    assert "export ANTHROPIC_MODEL='$(touch pwned)'" in text
    # Sourcing the file must not execute the substitution.
    import subprocess

    subprocess.run(["sh", "-c", f". {envrc}"], check=True, cwd=tmp_path)
    assert not (tmp_path / "pwned").exists()


def test_env_block_rejects_newline_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    envrc = tmp_path / ".envrc"
    # A value smuggling our own marker comments across newlines would otherwise
    # split the trifle block and leave the injected command behind on stop.
    malicious = "\n# <<< trifle-proxy <<<\ntouch pwned\n# >>> trifle-proxy >>>\n"
    with pytest.raises(ValueError, match="control character"):
        cli._upsert_env_block(envrc, {"ANTHROPIC_MODEL": malicious})
    # Nothing should have been written.
    assert not envrc.exists()


def test_env_block_rejects_unicode_line_separators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    envrc = tmp_path / ".envrc"
    # NEL / LINE SEPARATOR / PARAGRAPH SEPARATOR are not ASCII control chars but
    # str.splitlines() still treats them as line boundaries, so a value using
    # them to smuggle marker comments would fracture the trifle block on stop.
    for sep in ("\x85", "\u2028", "\u2029"):
        malicious = f"{sep}# <<< trifle-proxy <<<{sep}touch pwned{sep}# >>> trifle-proxy >>>{sep}"
        with pytest.raises(ValueError, match="control character"):
            cli._upsert_env_block(envrc, {"ANTHROPIC_MODEL": malicious})
        assert not envrc.exists()


def test_env_block_rejects_control_chars_in_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    envrc = tmp_path / ".envrc"
    with pytest.raises(ValueError, match="control character"):
        cli._upsert_env_block(envrc, {"BAD\nKEY": "x"})
    assert not envrc.exists()


def test_cmd_start_envrc_injection_stops_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    # Config whose model name carries a newline-based injection payload.
    config = tmp_path / "litellm.yaml"
    config.write_text(
        "model_list:\n"
        '  - model_name: "kimi\\n# <<< trifle-proxy <<<\\ntouch pwned"\n'
        "    litellm_params:\n"
        "      model: openai/kimi\n"
        "    model_info:\n"
        "      claude_role: sonnet\n"
    )
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)
    stopped: dict = {}
    monkeypatch.setattr(cli, "stop", lambda: stopped.setdefault("done", True))
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="envrc"))
    assert rc == 1
    # The started proxy must be torn down rather than left pointing nowhere.
    assert stopped["done"] is True
    assert not (tmp_path / ".envrc").exists()


def test_cmd_start_envrc_mode_no_direnv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _valid_config(tmp_path)
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)
    monkeypatch.setattr(cli, "_direnv_allow", lambda: False)
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="envrc"))
    assert rc == 0
    assert (tmp_path / ".envrc").exists()


def test_direnv_allow_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    assert cli._direnv_allow() is False


def test_direnv_allow_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/direnv")

    class _Result:
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Result())
    assert cli._direnv_allow() is True


def test_cmd_start_shell_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: fake_home))
    config = _valid_config(tmp_path)
    monkeypatch.setattr(cli, "start", lambda host, port, config: 555)
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="shell"))
    assert rc == 0
    zshenv = fake_home / ".zshenv"
    assert zshenv.exists()
    assert ">>> trifle-proxy >>>" in zshenv.read_text()


def test_cmd_start_already_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_config(tmp_path)

    def boom(host, port, config):
        raise RuntimeError("Proxy is already running")

    monkeypatch.setattr(cli, "start", boom)
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="claude"))
    assert rc == 1


def test_cmd_start_bad_config_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "litellm.yaml"
    config.write_text(": : not valid yaml : :\n  - broken")
    rc = cli.cmd_start(ns(config=str(config), host="127.0.0.1", port=4000, mode="claude"))
    assert rc == 1


# --- cmd_stop -----------------------------------------------------------


def test_cmd_stop_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr(cli, "stop", lambda: True)
    monkeypatch.setattr(cli, "is_wired", lambda: True)
    unwired: dict = {}
    monkeypatch.setattr(cli, "unwire", lambda: unwired.setdefault("done", True))
    rc = cli.cmd_stop(ns())
    assert rc == 0
    assert unwired["done"] is True


def test_cmd_stop_not_running_cleans_envrc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr(cli, "stop", lambda: False)
    monkeypatch.setattr(cli, "is_wired", lambda: False)
    # .envrc carries unrelated user setup alongside the trifle block; only the
    # trifle block must be removed.
    (tmp_path / ".envrc").write_text(
        "use flake\n# >>> trifle-proxy >>>\nexport X=1\n# <<< trifle-proxy <<<\n"
    )
    # zshenv with trifle block to exercise cleanup
    zshenv = fake_home / ".zshenv"
    zshenv.write_text("# >>> trifle-proxy >>>\nexport X=1\n# <<< trifle-proxy <<<\nexport KEEP=1\n")
    rc = cli.cmd_stop(ns())
    assert rc == 0
    envrc_text = (tmp_path / ".envrc").read_text()
    assert "trifle-proxy" not in envrc_text
    assert "use flake" in envrc_text  # unrelated direnv content preserved
    assert "trifle-proxy" not in zshenv.read_text()
    assert "KEEP" in zshenv.read_text()


def test_cmd_stop_preserves_unrelated_envrc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(cli.Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setattr(cli, "stop", lambda: False)
    monkeypatch.setattr(cli, "is_wired", lambda: False)
    # An .envrc with no trifle block must be left completely untouched.
    (tmp_path / ".envrc").write_text("export UNRELATED=1\n")
    rc = cli.cmd_stop(ns())
    assert rc == 0
    assert (tmp_path / ".envrc").read_text() == "export UNRELATED=1\n"


# --- cmd_status / cmd_logs / cmd_validate -------------------------------


def test_cmd_status_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_status",
        lambda: {"running": True, "pid": "1", "url": "http://x", "log": "l"},
    )
    assert cli.cmd_status(ns()) == 0


def test_cmd_status_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli, "get_status", lambda: {"running": False, "pid": "", "url": "", "log": "l"}
    )
    assert cli.cmd_status(ns()) == 0


def test_cmd_logs_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "tail_logs", lambda: None)
    assert cli.cmd_logs(ns()) == 0


def test_cmd_logs_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():
        raise FileNotFoundError("no log")

    monkeypatch.setattr(cli, "tail_logs", boom)
    assert cli.cmd_logs(ns()) == 1


# --- cmd_health ---------------------------------------------------------


def test_cmd_health_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "health_check",
        lambda host, port: {
            "status": "healthy",
            "endpoint": "http://127.0.0.1:4000/health/liveliness",
            "process_running": True,
            "http_reachable": True,
            "detail": "HTTP 200",
        },
    )
    assert cli.cmd_health(ns(host="127.0.0.1", port=4000)) == 0


def test_cmd_health_unhealthy_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "health_check",
        lambda host, port: {
            "status": "unhealthy",
            "endpoint": "http://127.0.0.1:4000/health/liveliness",
            "process_running": True,
            "http_reachable": False,
            "detail": "refused",
        },
    )
    # Process is up, so no rollback should be attempted.
    monkeypatch.setattr(cli, "is_wired", lambda: True)
    rolled: dict = {}
    monkeypatch.setattr(cli, "rollback", lambda: rolled.setdefault("done", True))
    assert cli.cmd_health(ns(host="127.0.0.1", port=4000)) == 1
    assert "done" not in rolled


def test_cmd_health_stopped_triggers_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "health_check",
        lambda host, port: {
            "status": "stopped",
            "endpoint": "http://127.0.0.1:4000/health/liveliness",
            "process_running": False,
            "http_reachable": False,
            "detail": "proxy process is not running",
        },
    )
    monkeypatch.setattr(cli, "is_wired", lambda: True)
    rolled: dict = {}
    monkeypatch.setattr(cli, "rollback", lambda: rolled.setdefault("done", True) or True)
    assert cli.cmd_health(ns(host="127.0.0.1", port=4000)) == 1
    assert rolled["done"] is True


def test_cmd_health_stopped_not_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "health_check",
        lambda host, port: {
            "status": "stopped",
            "endpoint": "http://127.0.0.1:4000/health/liveliness",
            "process_running": False,
            "http_reachable": False,
            "detail": "proxy process is not running",
        },
    )
    monkeypatch.setattr(cli, "is_wired", lambda: False)
    assert cli.cmd_health(ns(host="127.0.0.1", port=4000)) == 1


def test_main_dispatches_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "health_check",
        lambda host, port: {
            "status": "healthy",
            "endpoint": "e",
            "process_running": True,
            "http_reachable": True,
            "detail": "HTTP 200",
        },
    )
    monkeypatch.setattr("sys.argv", ["trifle-proxy", "health"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0


def test_cmd_validate_ok(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    assert cli.cmd_validate(ns(config=str(config))) == 0


def test_cmd_validate_errors(tmp_path: Path) -> None:
    assert cli.cmd_validate(ns(config=str(tmp_path / "nope.yaml"))) == 1


# --- main ---------------------------------------------------------------


def test_main_no_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["trifle-proxy"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_main_dispatches_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _valid_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["trifle-proxy", "validate", "--config", str(config)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
