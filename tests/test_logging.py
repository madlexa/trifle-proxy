"""Tests for logging_config.py — structured logging setup."""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from trifle_proxy import logging_config


@pytest.fixture(autouse=True)
def reset_structlog():
    """Reset structlog global config between tests."""
    logging_config._configured = False
    structlog.reset_defaults()
    yield
    logging_config._configured = False
    structlog.reset_defaults()


def test_resolve_level_known() -> None:
    assert logging_config._resolve_level("debug") == logging.DEBUG
    assert logging_config._resolve_level("ERROR") == logging.ERROR


def test_resolve_level_unknown_defaults_info() -> None:
    assert logging_config._resolve_level("bogus") == logging.INFO


def test_configure_logging_sets_configured_flag() -> None:
    assert logging_config._configured is False
    logging_config.configure_logging(level="debug")
    assert logging_config._configured is True


def test_configure_logging_invalid_level_falls_back() -> None:
    # Should not raise even with an invalid level
    logging_config.configure_logging(level="nonsense")
    assert logging_config._configured is True


def test_get_logger_autoconfigures() -> None:
    assert logging_config._configured is False
    log = logging_config.get_logger("test")
    assert logging_config._configured is True
    assert log is not None


def test_get_logger_does_not_reconfigure() -> None:
    logging_config.configure_logging(level="warning")
    assert logging_config._configured is True
    # Calling get_logger again should not flip/reset anything
    log = logging_config.get_logger("test2")
    assert log is not None
    assert logging_config._configured is True


def test_json_renderer_emits_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    logging_config.configure_logging(level="info", json_logs=True)
    log = structlog.get_logger("json_test")
    log.info("hello", key="value")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["key"] == "value"
    assert payload["level"] == "info"


def test_console_renderer_outputs_event(capsys: pytest.CaptureFixture[str]) -> None:
    logging_config.configure_logging(level="info", json_logs=False)
    log = structlog.get_logger("console_test")
    log.info("console_event", foo="bar")
    captured = capsys.readouterr()
    assert "console_event" in captured.err


def test_level_filters_below_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    logging_config.configure_logging(level="error", json_logs=True)
    log = structlog.get_logger("filter_test")
    log.info("should_be_filtered")
    log.error("should_appear")
    captured = capsys.readouterr()
    assert "should_be_filtered" not in captured.err
    assert "should_appear" in captured.err
