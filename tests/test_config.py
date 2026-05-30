from pathlib import Path

import pytest

from trifle_proxy.config import (
    build_env_vars,
    get_model_roles,
    get_resilience_config,
    load_config,
    validate_config,
)


def test_load_config(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("""
model_list:
  - model_name: test-model
    litellm_params:
      model: openai/gpt-4
    model_info:
      claude_role: sonnet
litellm_settings:
  drop_params: true
""")
    config = load_config(config_path)
    assert len(config["model_list"]) == 1
    assert config["model_list"][0]["model_name"] == "test-model"


def test_load_config_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("# just a comment\n")
    assert load_config(config_path) == {}


def test_validate_config_empty_file_reports_error_not_crash(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    errors = validate_config(config_path)
    assert errors == ["Missing 'model_list' in config"]


def test_validate_config_non_dict_root(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- just\n- a\n- list\n")
    errors = validate_config(config_path)
    assert errors == ["Config root must be a mapping (got a non-dict YAML document)"]


def test_build_env_vars_with_empty_config_uses_fallback() -> None:
    env = build_env_vars({})
    assert env["ANTHROPIC_MODEL"] == "kimi-k2.5"


def test_get_model_roles() -> None:
    config = {
        "model_list": [
            {"model_name": "opus-model", "model_info": {"claude_role": "opus"}},
            {"model_name": "sonnet-model", "model_info": {"claude_role": "sonnet"}},
        ]
    }
    roles = get_model_roles(config)
    assert roles == {"opus": "opus-model", "sonnet": "sonnet-model"}


def test_get_resilience_config_defaults_when_absent() -> None:
    cfg = get_resilience_config({"model_list": []})
    assert cfg.retry.max_attempts == 3
    assert cfg.circuit.failure_threshold == 5
    assert cfg.fallback_models == ()


def test_get_resilience_config_reads_section() -> None:
    config = {
        "resilience": {
            "retry": {"max_attempts": 4},
            "circuit_breaker": {"failure_threshold": 8},
            "fallback_models": ["a", "b"],
        }
    }
    cfg = get_resilience_config(config)
    assert cfg.retry.max_attempts == 4
    assert cfg.circuit.failure_threshold == 8
    assert cfg.fallback_models == ("a", "b")


def test_build_env_vars() -> None:
    config = {
        "model_list": [
            {"model_name": "kimi-k2.6", "model_info": {"claude_role": "opus"}},
            {"model_name": "kimi-k2.5", "model_info": {"claude_role": "sonnet"}},
            {"model_name": "deepseek", "model_info": {"claude_role": "haiku"}},
        ]
    }
    env = build_env_vars(config, host="127.0.0.1", port=4000)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
    assert env["ANTHROPIC_MODEL"] == "kimi-k2.5"
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "kimi-k2.6"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "kimi-k2.5"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "deepseek"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "deepseek"


def test_validate_config_missing_file(tmp_path: Path) -> None:
    errors = validate_config(tmp_path / "nonexistent.yaml")
    assert len(errors) == 1
    assert "not found" in errors[0]


def test_validate_config_valid(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("""
model_list:
  - model_name: test
    litellm_params:
      model: openai/gpt-4
    model_info:
      claude_role: sonnet
""")
    errors = validate_config(config_path)
    assert len(errors) == 0


def test_load_config_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_validate_config_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("model_list: [unclosed\n  - : :")
    errors = validate_config(config_path)
    assert any("Invalid YAML" in e for e in errors)


def test_validate_config_missing_model_list(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("litellm_settings:\n  drop_params: true\n")
    errors = validate_config(config_path)
    assert any("model_list" in e for e in errors)


def test_validate_config_missing_fields_and_roles(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("""
model_list:
  - litellm_params:
      model: openai/gpt-4
""")
    errors = validate_config(config_path)
    # missing model_name + no claude_role defined
    assert any("missing 'model_name'" in e for e in errors)
    assert any("claude_role" in e for e in errors)


def test_validate_config_missing_litellm_params(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("""
model_list:
  - model_name: solo
    model_info:
      claude_role: sonnet
""")
    errors = validate_config(config_path)
    assert any("missing 'litellm_params'" in e for e in errors)


def test_validate_config_invalid_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("""
model_list:
  - model_name: test
    litellm_params:
      model: openai/gpt-4
      api_key: "bad key with spaces"
    model_info:
      claude_role: sonnet
""")
    errors = validate_config(config_path)
    assert any("api_key" in e for e in errors)


def test_validate_config_env_indirection_api_key_ok(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("""
model_list:
  - model_name: test
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      claude_role: sonnet
""")
    errors = validate_config(config_path)
    assert errors == []


def test_build_env_vars_empty_roles() -> None:
    env = build_env_vars({"model_list": []})
    # Falls back to default model name when no roles present
    assert env["ANTHROPIC_MODEL"] == "kimi-k2.5"


def test_validate_config_model_list_not_a_list(tmp_path: Path) -> None:
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("model_list: just-a-string\n")
    errors = validate_config(config_path)
    assert errors == ["'model_list' must be a list of model entries"]


def test_validate_config_model_list_scalar_item_does_not_crash(tmp_path: Path) -> None:
    # Previously raised AttributeError on item.get(...) for non-mapping entries.
    config_path = tmp_path / "litellm.yaml"
    config_path.write_text("model_list:\n  - bad\n  - 123\n")
    errors = validate_config(config_path)
    assert any("model_list[0]: must be a mapping" in e for e in errors)
    assert any("model_list[1]: must be a mapping" in e for e in errors)


def test_get_model_roles_tolerates_malformed_entries() -> None:
    config = {
        "model_list": [
            "bad",
            {"model_name": "ok", "model_info": "not-a-dict"},
            {"model_name": "good", "model_info": {"claude_role": "sonnet"}},
        ]
    }
    assert get_model_roles(config) == {"sonnet": "good"}


def test_get_model_roles_non_list_model_list() -> None:
    assert get_model_roles({"model_list": "nope"}) == {}


def test_get_model_roles_skips_non_string_model_name() -> None:
    # A non-string model_name (e.g. ``model_name: 123``) must not leak into the
    # result: it would become a non-string env value and crash the string-only
    # env writer downstream. The bad entry is skipped, the valid one kept.
    config = {
        "model_list": [
            {"model_name": 123, "model_info": {"claude_role": "opus"}},
            {"model_name": "ok", "model_info": {"claude_role": 99}},
            {"model_name": "good", "model_info": {"claude_role": "sonnet"}},
        ]
    }
    roles = get_model_roles(config)
    assert roles == {"sonnet": "good"}
    assert all(isinstance(v, str) for v in roles.values())
