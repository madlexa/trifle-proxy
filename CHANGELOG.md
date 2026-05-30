# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Test coverage** across all modules (`claude`, `proxy`, `cli`, `health`,
  `security`, `resilience`, `metrics`, `logging`); coverage gate raised to 80%.
- **Structured logging** via `structlog` (`logging_config.py`) with console and
  JSON output; `--verbose`, `--log-level`, and `--log-json` CLI flags.
- **Health checks** (`health.py`): process-liveness plus an HTTP liveliness
  probe; new `trifle-proxy health` command that auto-rolls-back a
  crashed-but-wired state.
- **Graceful shutdown**: `SIGTERM`/`SIGINT` handlers drain in-flight requests up
  to a 10s timeout before force-kill; transactional `claude.wired` context
  manager and `rollback` primitive.
- **Security primitives** (`security.py`): API-key shape validation, path
  sanitization against traversal, token-bucket rate limiting, and sha256 backup
  integrity verification.
- **Resilience primitives** (`resilience.py`): retry with exponential backoff,
  circuit breaker, fallback over multiple targets, and `cleanup_on_error`;
  configurable via a `resilience:` section in `litellm.yaml`.
  `proxy.start_resilient` retries transient start failures and unwires on
  terminal failure.
- **Prometheus metrics** (`metrics.py`): dependency-free counters/gauges/
  histograms with labels, a `/metrics` HTTP server (`MetricsServer`), a scrape
  client, and the `trifle-proxy metrics` command.
- **CI/CD quality gates**: `ruff` (lint + format), `mypy`, and
  `bandit`/`pip-audit` jobs; a 3-OS × 4-Python test matrix; a `Makefile`
  (`test`, `lint`, `format`, `typecheck`, `security`, `audit`, `check`).
- **Developer documentation**: `docs/architecture.md`, `docs/CONTRIBUTING.md`,
  `docs/SECURITY.md`, this changelog, and an expanded README.

## [1.0.0] - 2026-05-30

### Added

- Initial release.
- `trifle-proxy` CLI: `init`, `start`, `stop`, `status`, `logs`, `validate`.
- On-demand LiteLLM proxy launch via `uvx` (no global install).
- Three wire modes: `claude` (edits `~/.claude/settings.json`), `envrc`
  (direnv), and `shell` (`~/.zshenv`).
- `litellm.yaml` parsing/validation and `model_info.claude_role` → Claude Code
  model-tier mapping.
- Interactive `init` wizard and a non-interactive `--template` mode.
- Cross-platform support (macOS, Linux, Windows) on Python 3.10+.

[Unreleased]: https://github.com/madlexa/trifle-proxy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/madlexa/trifle-proxy/releases/tag/v1.0.0
