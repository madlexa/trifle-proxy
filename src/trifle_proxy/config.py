"""Configuration parsing for litellm.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trifle_proxy.resilience import ResilienceConfig
from trifle_proxy.security import is_valid_api_key

DEFAULT_CONFIG_PATH = Path.home() / ".trifle" / "proxy" / "litellm.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4000


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and parse litellm.yaml.

    An empty or comment-only file parses to ``None``; we normalize that to an
    empty dict so callers can use ``.get`` without a ``NoneType`` crash.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_model_roles(config: dict[str, Any]) -> dict[str, str]:
    """Extract claude_role -> model_name mapping from config.

    Tolerates malformed entries (non-list ``model_list``, non-mapping items,
    non-mapping ``model_info``) by skipping them rather than crashing — config
    validation reports those separately. Only string ``model_name`` and
    ``claude_role`` values are kept: the result feeds env vars that must be
    plain strings, and a non-string (e.g. ``model_name: 123``) would otherwise
    propagate into ``build_env_vars`` and crash downstream string handling.
    """
    roles: dict[str, str] = {}
    model_list = config.get("model_list", [])
    if not isinstance(model_list, list):
        return roles
    for item in model_list:
        if not isinstance(item, dict):
            continue
        model_name = item.get("model_name")
        info = item.get("model_info")
        role = info.get("claude_role") if isinstance(info, dict) else None
        if isinstance(model_name, str) and isinstance(role, str) and model_name and role:
            roles[role] = model_name
    return roles


def build_env_vars(
    config: dict[str, Any],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> dict[str, str]:
    """Build environment variables dict from config."""
    roles = get_model_roles(config)

    static_env = {
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ENABLE_TOOL_SEARCH": "false",
        "CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING": "1",
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
        "CLAUDE_CODE_EFFORT_LEVEL": "high",
    }

    env: dict[str, str] = dict(static_env)
    env["ANTHROPIC_BASE_URL"] = f"http://{host}:{port}"
    # Fixed local sentinel token for the loopback proxy, not a real secret.
    env["ANTHROPIC_AUTH_TOKEN"] = "sk-local-claude-code"  # nosec B105
    env["ANTHROPIC_MODEL"] = roles.get(
        "sonnet", next(iter(roles.values()), "kimi-k2.5") if roles else "kimi-k2.5"
    )
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = roles.get("opus", "")
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = roles.get("sonnet", "")
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = roles.get("haiku", "")
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = roles.get("subagent", roles.get("haiku", ""))

    return env


def get_resilience_config(config: dict[str, Any]) -> ResilienceConfig:
    """Extract resilience settings from the ``resilience`` section of config.

    Missing or partial settings fall back to defaults, so configs written for
    older versions keep working unchanged.
    """
    return ResilienceConfig.from_dict(config.get("resilience"))


def validate_config(path: Path | str = DEFAULT_CONFIG_PATH) -> list[str]:
    """Validate config and return list of errors."""
    errors: list[str] = []
    path = Path(path)

    if not path.exists():
        errors.append(f"Config file not found: {path}")
        return errors

    try:
        config = load_config(path)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML: {e}")
        return errors

    if not isinstance(config, dict):
        errors.append("Config root must be a mapping (got a non-dict YAML document)")
        return errors

    model_list = config.get("model_list")
    if not model_list:
        errors.append("Missing 'model_list' in config")
        return errors

    if not isinstance(model_list, list):
        errors.append("'model_list' must be a list of model entries")
        return errors

    for i, item in enumerate(model_list):
        if not isinstance(item, dict):
            errors.append(f"model_list[{i}]: must be a mapping")
            continue
        if "model_name" not in item:
            errors.append(f"model_list[{i}]: missing 'model_name'")
        params = item.get("litellm_params")
        if params is None:
            errors.append(f"model_list[{i}]: missing 'litellm_params'")
        elif isinstance(params, dict):
            api_key = params.get("api_key")
            # Skip env-var indirection (e.g. "os.environ/FOO") which LiteLLM
            # resolves at runtime; only validate inline literal keys.
            if (
                isinstance(api_key, str)
                and not api_key.startswith("os.environ/")
                and not is_valid_api_key(api_key)
            ):
                errors.append(f"model_list[{i}]: 'api_key' has invalid format")

    roles = get_model_roles(config)
    if not roles:
        errors.append("No models have 'model_info.claude_role' defined")

    return errors
