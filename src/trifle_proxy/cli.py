"""CLI entry point for trifle-proxy."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from jinja2 import Template

from trifle_proxy import __version__, metrics
from trifle_proxy.claude import is_wired, rollback, unwire, wire
from trifle_proxy.config import (
    DEFAULT_CONFIG_PATH,
    build_env_vars,
    load_config,
    validate_config,
)
from trifle_proxy.health import check as health_check
from trifle_proxy.logging_config import configure_logging
from trifle_proxy.proxy import get_status, start, stop, tail_logs

# Path for --local mode (current directory)
LOCAL_CONFIG_PATH = Path("litellm.yaml")

# ASCII art banner
BANNER = r"""
  _____    _  __ _
 |_   _| _(_)/ _| |___
   | || '_| |  _| / -_)
  _|_||_| |_|_| |_\___|
 | _ \_ _ _____ ___  _
 |  _/ '_/ _ \ \ / || |
 |_| |_| \___/_\_\\_, |
                  |__/
"""


def _print(message: str) -> None:
    print(message)


# Markers delimiting the trifle-proxy-owned block in shell dotfiles. We only
# ever touch lines between these markers so unrelated user content (direnv
# setup, other exports) is preserved across start/stop.
_MARKER_START = "# >>> trifle-proxy >>>"
_MARKER_END = "# <<< trifle-proxy <<<"


def _strip_trifle_block(lines: list[str]) -> tuple[list[str], bool]:
    """Drop the trifle-owned marker block. Returns (remaining_lines, found)."""
    in_block = False
    found = False
    out: list[str] = []
    for line in lines:
        if line.strip() == _MARKER_START:
            in_block = True
            found = True
            continue
        if line.strip() == _MARKER_END:
            in_block = False
            continue
        if not in_block:
            out.append(line)
    return out, found


# Characters ``str.splitlines()`` treats as line boundaries but which the
# C0/DEL control-character check below does not catch: NEL, LINE SEPARATOR, and
# PARAGRAPH SEPARATOR. ``_strip_trifle_block`` parses dotfiles with
# ``splitlines()``, so a value carrying one of these would be split into extra
# logical lines on read even though it occupies a single physical line on disk.
_UNICODE_LINE_SEPARATORS = ("\x85", "\u2028", "\u2029")


def _assert_single_line_env(env: dict[str, str]) -> None:
    """Reject env keys/values carrying newlines or other control characters.

    ``shlex.quote`` neutralizes command substitution but happily preserves a
    literal newline, turning one ``export`` into several physical lines. A value
    smuggling our own marker comments could then split the trifle block so
    ``_strip_trifle_block`` leaves injected commands behind on ``stop``. Control
    characters have no legitimate place in a model name, host, or token, so we
    refuse to write them rather than try to encode around them.

    The check covers every character ``str.splitlines()`` honors as a line
    boundary — the C0 controls and DEL plus the Unicode NEL/LS/PS separators —
    because ``_strip_trifle_block`` re-parses the dotfile with ``splitlines()``;
    a separator we let through could fracture the marker block on a later read.
    """
    for key, value in env.items():
        for label, text in (("name", key), ("value", value)):
            if any(
                ord(ch) < 0x20 or ord(ch) == 0x7F or ch in _UNICODE_LINE_SEPARATORS for ch in text
            ):
                raise ValueError(
                    f"Refusing to write env {label} for {key!r}: "
                    "contains a newline or control character"
                )


def _upsert_env_block(path: Path, env: dict[str, str]) -> None:
    """Insert or replace the trifle-owned export block in ``path``.

    Existing (non-trifle) content is preserved. Values are quoted with
    ``shlex.quote`` so a hostile model name or host (e.g. ``$(rm -rf ~)``)
    cannot trigger command substitution when the file is sourced, and rejected
    outright if they contain control characters that would break the one
    physical line per ``export`` invariant the marker block relies on.
    """
    _assert_single_line_env(env)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text().splitlines()
    lines, _ = _strip_trifle_block(lines)
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines:
        lines.append("")
    lines.append(_MARKER_START)
    lines.extend(f"export {k}={shlex.quote(v)}" for k, v in env.items())
    lines.append(_MARKER_END)
    lines.append("")
    path.write_text("\n".join(lines))


def _remove_env_block(path: Path) -> bool:
    """Remove the trifle-owned block from ``path``.

    Returns True if a block was removed. Unrelated content is preserved; if
    stripping the block leaves the file empty it is deleted so we don't leave
    an orphaned empty dotfile behind.
    """
    if not path.exists():
        return False
    remaining, found = _strip_trifle_block(path.read_text().splitlines())
    if not found:
        return False
    while remaining and remaining[-1].strip() == "":
        remaining.pop()
    if remaining:
        path.write_text("\n".join(remaining) + "\n")
    else:
        path.unlink()
    return True


def _direnv_allow() -> bool:
    """Run ``direnv allow`` if direnv is installed. Returns True on success."""
    direnv = shutil.which("direnv")
    if direnv is None:
        return False
    result = subprocess.run(
        [direnv, "allow"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def cmd_init(args: argparse.Namespace) -> int:
    """Interactive init to create litellm.yaml."""
    config_path = Path(args.config)
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and not args.force:
        response = input(f"{config_path} already exists. Overwrite? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            _print("Aborted.")
            return 1

    if args.template:
        # Simple template copy without interaction
        template_path = Path(__file__).parent.parent.parent / "templates" / "litellm.yaml.j2"
        if not template_path.exists():
            _print(f"Template not found: {template_path}")
            return 1
        # Render with default example data
        models: list[dict[str, Any]] = [
            {
                "name": "kimi-k2.5",
                "provider": "moonshot",
                "model": "kimi-k2.5",
                "env_key": "MOONSHOT_API_KEY",
                "api_base": "https://api.moonshot.ai/v1",
                "role": "sonnet",
            },
            {
                "name": "deepseek-chat",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "env_key": "DEEPSEEK_API_KEY",
                "api_base": "https://api.deepseek.com",
                "role": "haiku",
            },
        ]
        template = Template(template_path.read_text())
        config_path.write_text(template.render(models=models))
        _print(f"Created {config_path} from template")
        _print("Edit the file and set your API keys as environment variables.")
        return 0

    # Interactive wizard
    _print("\n=== Trifle Proxy Init ===\n")
    _print("Let's configure your LLM proxy.\n")

    providers = {
        "1": ("moonshot", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1"),
        "2": ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
        "3": ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        "4": ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com"),
    }

    _print("Available providers:")
    for key, (name, env_key, _) in providers.items():
        _print(f"  {key}. {name} (env: {env_key})")
    _print("  5. Done adding providers")

    models = []
    while True:
        choice = input("\nSelect provider (1-5): ").strip()
        if choice == "5":
            break
        if choice not in providers:
            _print("Invalid choice.")
            continue

        provider_name, env_key, api_base = providers[choice]
        model_id = input("Model ID (e.g. kimi-k2.5): ").strip()
        alias = input("Alias name (e.g. kimi-k2.5): ").strip() or model_id

        _print("Roles: opus, sonnet, haiku, subagent")
        role = input("Claude role for this model [sonnet]: ").strip() or "sonnet"

        models.append(
            {
                "name": alias,
                "provider": provider_name,
                "model": model_id,
                "env_key": env_key,
                "api_base": api_base,
                "role": role,
            }
        )
        _print(f"Added {alias} ({provider_name}/{model_id}) as '{role}'")

    if not models:
        _print("No providers configured. Aborting.")
        return 1

    template_path = Path(__file__).parent.parent.parent / "templates" / "litellm.yaml.j2"
    if not template_path.exists():
        _print(f"Template not found: {template_path}")
        return 1

    template = Template(template_path.read_text())
    config_path.write_text(template.render(models=models))
    _print(f"\nCreated {config_path}")
    _print("\nNext steps:")
    _print("  1. Set your API keys as environment variables:")
    seen = set()
    for m in models:
        if m["env_key"] not in seen:
            _print(f"     export {m['env_key']}=your_key_here")
            seen.add(m["env_key"])
    _print("  2. export LITELLM_MASTER_KEY=sk-local-claude-code")
    _print("  3. trifle-proxy start")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    """Start proxy and wire agent."""
    config_path = Path(args.config)
    if not config_path.exists():
        _print(f"Config file not found: {config_path}")
        _print("Run 'trifle-proxy init' first.")
        return 1

    try:
        config = load_config(config_path)
        env = build_env_vars(config, host=args.host, port=args.port)
    except Exception as e:
        _print(f"Failed to load config: {e}")
        return 1

    mode = args.mode or "claude"

    try:
        pid = start(host=args.host, port=args.port, config=config_path)
    except RuntimeError as e:
        _print(str(e))
        return 1
    except FileNotFoundError as e:
        _print(str(e))
        return 1

    _print(BANNER)
    _print(f"LiteLLM proxy started (PID: {pid})")
    _print(f"URL: http://{args.host}:{args.port}")

    if mode == "claude":
        try:
            backup = wire(env)
        except OSError as e:
            # Wiring failed after the proxy was started; don't leave a running
            # proxy that nothing points at. Stop it and report cleanly.
            _print(f"Failed to wire Claude Code: {e}")
            stop()
            return 1
        _print(f"Backup saved to {backup}")
        _print(f"Claude Code wired via port {args.port}")
    elif mode == "envrc":
        envrc_path = Path(".envrc")
        try:
            _upsert_env_block(envrc_path, env)
        except (ValueError, OSError) as e:
            # Either the env failed validation (ValueError) or the dotfile is
            # unreadable/unwritable (OSError). Either way the proxy is already
            # running; stop it rather than leaving an orphan nothing points at.
            _print(f"Failed to update {envrc_path}: {e}")
            stop()
            return 1
        _print(f"Updated {envrc_path}")
        if _direnv_allow():
            _print("direnv: allowed .envrc")
        else:
            _print("⚠️  direnv not installed or not configured.")
            _print("   Install: brew install direnv")
            _print('   Then add to ~/.zshrc: eval "$(direnv export zsh)"')
    elif mode == "shell":
        zshenv = Path.home() / ".zshenv"
        try:
            _upsert_env_block(zshenv, env)
        except (ValueError, OSError) as e:
            # Either the env failed validation (ValueError) or the dotfile is
            # unreadable/unwritable (OSError). Either way the proxy is already
            # running; stop it rather than leaving an orphan nothing points at.
            _print(f"Failed to update {zshenv}: {e}")
            stop()
            return 1
        _print(f"Updated {zshenv}")
        _print("Open a new terminal or run: source ~/.zshenv")

    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop proxy and unwire agent."""
    was_running = stop()
    if was_running:
        _print("LiteLLM proxy stopped")
    else:
        _print("LiteLLM proxy is not running")

    # Clean up all modes
    if is_wired():
        unwire()
        _print("Cleared Claude Code settings")

    # Only remove the trifle-owned block from each dotfile; unrelated direnv
    # setup or user exports are left intact.
    if _remove_env_block(Path(".envrc")):
        _print("Removed trifle-proxy env from .envrc")

    if _remove_env_block(Path.home() / ".zshenv"):
        _print("Cleared global env from ~/.zshenv")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show proxy status."""
    status = get_status()
    if status["running"]:
        _print("running")
        _print(f"PID: {status['pid']}")
        _print(f"URL: {status['url']}")
        _print(f"Log: {status['log']}")
    else:
        _print("stopped")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    """Tail proxy logs."""
    try:
        tail_logs()
    except FileNotFoundError as e:
        _print(str(e))
        return 1
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check proxy health and auto-rollback a crashed-but-wired state."""
    result = health_check(host=args.host, port=args.port)

    _print(f"status: {result['status']}")
    _print(f"endpoint: {result['endpoint']}")
    _print(f"process_running: {result['process_running']}")
    _print(f"http_reachable: {result['http_reachable']}")
    _print(f"detail: {result['detail']}")

    if result["status"] == "healthy":
        return 0

    # Proxy is gone but Claude Code is still pointed at it: clean up so the
    # user isn't left with a dead endpoint wired in.
    if not result["process_running"] and is_wired() and rollback():
        _print("Proxy is down — rolled back Claude Code settings.")

    return 1


def cmd_metrics(args: argparse.Namespace) -> int:
    """Print Prometheus metrics.

    By default scrapes the proxy's own ``/metrics`` endpoint; with ``--local``
    (or when scraping fails) it renders trifle-proxy's in-process registry.
    """
    if args.local:
        _print(metrics.render().rstrip("\n"))
        return 0

    url = args.url or metrics.metrics_url(host=args.host, port=args.port)
    try:
        text = metrics.scrape(url)
    except metrics.MetricError as e:
        _print(f"Could not scrape {url}: {e}")
        _print("Falling back to local metrics:")
        _print(metrics.render().rstrip("\n"))
        return 0
    _print(text.rstrip("\n"))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate litellm.yaml config."""
    errors = validate_config(args.config)
    if errors:
        _print("Config validation failed:")
        for err in errors:
            _print(f"  - {err}")
        return 1
    _print("Config is valid.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trifle-proxy",
        description="A lightweight CLI to run LiteLLM proxy and wire Claude Code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trifle-proxy init                  # Interactive config setup
  trifle-proxy start                 # Start proxy + wire Claude Code
  trifle-proxy start --envrc         # Use direnv instead of global wire
  trifle-proxy stop                  # Stop proxy + unwire
  trifle-proxy status                # Check proxy status
  trifle-proxy health                # Probe proxy health endpoint
  trifle-proxy logs                  # Tail proxy logs
  trifle-proxy metrics               # Show Prometheus metrics
  trifle-proxy validate              # Validate litellm.yaml
        """.strip(),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug-level) logging",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set log level (default: info)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Emit logs as JSON lines instead of console format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_parser = subparsers.add_parser("init", help="Create litellm.yaml interactively")
    init_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config file path")
    init_parser.add_argument(
        "--template", action="store_true", help="Use template without interaction"
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")
    init_parser.set_defaults(func=cmd_init)

    # start
    start_parser = subparsers.add_parser("start", help="Start the proxy")
    start_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config file path")
    start_parser.add_argument("--host", default="127.0.0.1", help="Proxy host")
    start_parser.add_argument("--port", type=int, default=4000, help="Proxy port")
    start_parser.add_argument(
        "--mode",
        choices=["claude", "envrc", "shell"],
        default=None,
        help="Wire mode: claude (default), envrc (direnv), or shell (~/.zshenv)",
    )
    start_parser.set_defaults(func=cmd_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the proxy")
    stop_parser.set_defaults(func=cmd_stop)

    # status
    status_parser = subparsers.add_parser("status", help="Show proxy status")
    status_parser.set_defaults(func=cmd_status)

    # logs
    logs_parser = subparsers.add_parser("logs", help="Tail proxy logs")
    logs_parser.set_defaults(func=cmd_logs)

    # health
    health_parser = subparsers.add_parser("health", help="Check proxy health")
    health_parser.add_argument("--host", default="127.0.0.1", help="Proxy host")
    health_parser.add_argument("--port", type=int, default=4000, help="Proxy port")
    health_parser.set_defaults(func=cmd_health)

    # metrics
    metrics_parser = subparsers.add_parser("metrics", help="Show Prometheus metrics")
    metrics_parser.add_argument("--host", default="127.0.0.1", help="Metrics host")
    metrics_parser.add_argument(
        "--port",
        type=int,
        default=metrics.DEFAULT_METRICS_PORT,
        help=f"Metrics port (default: {metrics.DEFAULT_METRICS_PORT})",
    )
    metrics_parser.add_argument(
        "--url", default=None, help="Full metrics URL (overrides --host/--port)"
    )
    metrics_parser.add_argument(
        "--local",
        action="store_true",
        help="Render the in-process registry instead of scraping",
    )
    metrics_parser.set_defaults(func=cmd_metrics)

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate litellm.yaml")
    validate_parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="Config file path"
    )
    validate_parser.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    level = "debug" if args.verbose else args.log_level
    configure_logging(level=level, json_logs=args.log_json)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    sys.exit(args.func(args))
