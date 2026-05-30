# trifle-proxy

[![CI](https://github.com/madlexa/trifle-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/madlexa/trifle-proxy/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)](https://github.com/madlexa/trifle-proxy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/trifle-proxy)](https://pypi.org/project/trifle-proxy/)
[![Python](https://img.shields.io/pypi/pyversions/trifle-proxy)](https://pypi.org/project/trifle-proxy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight CLI to run [LiteLLM](https://github.com/BerriAI/litellm) proxy and wire [Claude Code](https://github.com/anthropics/claude-code).

LiteLLM is launched on-demand via `uvx` (no global installation needed).

## Features

- 🚀 **Zero-config start** — run `trifle-proxy start` and Claude Code is instantly wired to your local proxy
- 🔌 **Multi-provider** — OpenAI, Anthropic, Moonshot, DeepSeek, Google Gemini, and more via LiteLLM
- 🏠 **Per-project or global** — wire Claude Code globally, per-project via `direnv`, or via shell env
- ⚙️ **Interactive init** — `trifle-proxy init` walks you through creating `litellm.yaml`
- 🪟 **Cross-platform** — works on macOS, Linux, and Windows

## Quick Start

```bash
# Install trifle-proxy
pipx install trifle-proxy
# or
uv tool install trifle-proxy

# Create config
trifle-proxy init

# Set API keys
export MOONSHOT_API_KEY=your_key
export LITELLM_MASTER_KEY=sk-local-claude-code

# Start proxy + wire Claude Code
trifle-proxy start

# Done! Now run `claude` anywhere and it will use your local proxy

# Stop when done
trifle-proxy stop
```

## Installation

### pipx (recommended)

```bash
pipx install trifle-proxy
```

### uv

```bash
uv tool install trifle-proxy
```

### From source

```bash
pipx install git+https://github.com/madlexa/trifle-proxy.git
```

## Commands

```bash
trifle-proxy --help               # Show help
trifle-proxy --version            # Show version

trifle-proxy init                 # Interactive config wizard
trifle-proxy init --template      # Create from template without questions

trifle-proxy start                # Start proxy + wire Claude Code globally
trifle-proxy start --mode envrc   # Wire via .envrc (direnv) — per-project
trifle-proxy start --mode shell   # Wire via ~/.zshenv — global shell env

trifle-proxy stop                 # Stop proxy (graceful) + unwire everything
trifle-proxy status               # Check proxy status
trifle-proxy health               # Probe health; auto-rollback if crashed
trifle-proxy logs                 # Tail proxy logs
trifle-proxy metrics              # Show Prometheus metrics
trifle-proxy validate             # Validate litellm.yaml
```

### Global flags

These apply to any command:

```bash
-v, --verbose                     # Debug-level logging
--log-level {debug,info,warning,error,critical}
--log-json                        # Emit logs as JSON lines
```

### Health & metrics

```bash
trifle-proxy health                       # status: healthy | unhealthy | stopped
trifle-proxy metrics --local               # render trifle-proxy's in-process registry
trifle-proxy metrics --url URL             # scrape a Prometheus endpoint (e.g. LiteLLM's /metrics)
```

trifle-proxy tracks process-lifecycle counters in an in-process registry
(`--local`). Live per-request metrics come from LiteLLM itself — point
`--url` at LiteLLM's `/metrics` endpoint to scrape them.

`health` combines a process-liveness check with an HTTP liveliness probe. If the
proxy process is gone but Claude Code is still wired to it, `health` restores
your `~/.claude/settings.json` from backup so you are never left pointed at a
dead endpoint.

## Configuration

`trifle-proxy init` creates a `litellm.yaml` in the current directory. Example:

```yaml
model_list:
  - model_name: kimi-k2.5
    litellm_params:
      model: moonshot/kimi-k2.5
      api_key: os.environ/MOONSHOT_API_KEY
      api_base: https://api.moonshot.ai/v1
    model_info:
      claude_role: sonnet

  - model_name: deepseek-chat
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY
      api_base: https://api.deepseek.com
    model_info:
      claude_role: haiku

litellm_settings:
  drop_params: true
  master_key: os.environ/LITELLM_MASTER_KEY
```

### Resilience (optional)

Runtime resilience for upstream providers — retries, provider cooldown
(circuit breaking), and model fallbacks — is handled by **LiteLLM's**
`router_settings` block, which the `init` template emits and LiteLLM consumes.

The separate `resilience:` section below configures trifle-proxy's own library
primitives (`proxy.start_resilient`, `resilience.CircuitBreaker`, etc.). These
are provided and unit-tested but **not yet wired into the default `start`
path** — see [docs/architecture.md](docs/architecture.md). Setting it has no
effect on the current CLI; it is reserved for callers that invoke those
primitives directly. Any missing key falls back to a sane default.

```yaml
resilience:
  retry:
    max_attempts: 3
    base_delay: 0.5
    max_delay: 30.0
    multiplier: 2.0
    jitter: 0.0
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 30.0
    success_threshold: 1
  fallback_models:
    - deepseek-chat
```

### `claude_role` mapping

`model_info.claude_role` tells trifle-proxy which Claude Code model tier to map to:

| `claude_role` | Claude Code env var |
|---------------|---------------------|
| `opus`        | `ANTHROPIC_DEFAULT_OPUS_MODEL` |
| `sonnet`      | `ANTHROPIC_MODEL` + `ANTHROPIC_DEFAULT_SONNET_MODEL` |
| `haiku`       | `ANTHROPIC_DEFAULT_HAIKU_MODEL` |
| `subagent`    | `CLAUDE_CODE_SUBAGENT_MODEL` |

## How it works

### `--claude` mode (default)

trifle-proxy modifies `~/.claude/settings.json`:

1. Backs up your existing settings
2. Injects `env.ANTHROPIC_BASE_URL` and model mappings
3. Claude Code reads these on startup — no shell env needed
4. On `stop`, the backup is restored and `env` is removed

### `--envrc` mode

Creates a `.envrc` file for [direnv](https://direnv.net/). Great for per-project isolation.

```bash
# Install direnv
brew install direnv

# Add to ~/.zshrc
eval "$(direnv export zsh)"

# Now `cd` into your project activates the proxy env automatically
```

### `--shell` mode

Appends `export` statements to `~/.zshenv`. Works in any new terminal window.

## Requirements

- Python 3.10+
- One of: `pipx`, `uv`, or `pip`
- `uv` (recommended) — LiteLLM will be launched via `uvx` without global installation
- API keys for the LLM providers you want to use

## Troubleshooting

### `LITELLM_MASTER_KEY is not set`

```bash
export LITELLM_MASTER_KEY=sk-local-claude-code
```

This can be any string — it's used to authenticate local requests.

### Proxy won't start

```bash
trifle-proxy validate          # Check your litellm.yaml
trifle-proxy logs              # See error messages
```

### Claude Code still hits Anthropic API

Make sure Claude Code is not running when you `trifle-proxy start`. It reads `~/.claude/settings.json` on startup. Restart Claude Code after starting the proxy.

You can confirm the proxy is reachable:

```bash
trifle-proxy health            # should report status: healthy
```

### Proxy started but `health` says `unhealthy`

The process is up but not yet serving. LiteLLM can take a few seconds to load on
first run (especially when fetched via `uvx`). Wait and re-run `trifle-proxy
health`, and check `trifle-proxy logs` for upstream errors (bad API key, wrong
`api_base`).

### Proxy crashed and Claude Code is stuck pointing at it

Run `trifle-proxy health` — when the process is gone but the wiring remains, it
restores your settings from backup automatically. You can also force cleanup
with `trifle-proxy stop`.

### Port 4000 already in use

Another proxy (or a stale process) holds the port. Stop it, or start on a
different port:

```bash
trifle-proxy stop
trifle-proxy start --port 4100
```

### `direnv` not picking up `.envrc`

Install direnv and hook it into your shell:

```bash
brew install direnv
echo 'eval "$(direnv export zsh)"' >> ~/.zshrc   # then restart the shell
```

`trifle-proxy start --mode envrc` runs `direnv allow` for you if direnv is
installed.

### Restoring `~/.claude/settings.json` manually

Backups are written to `~/.claude/backups/settings.backup.<timestamp>.json`
before each wire. If anything goes wrong, the most recent backup is your
pre-wire state.

## Documentation

- [Architecture](docs/architecture.md) — components, data flow, failure handling
- [Contributing](docs/CONTRIBUTING.md) — local setup, quality gates, PR process
- [Security policy](docs/SECURITY.md) — reporting vulnerabilities, security model
- [Changelog](CHANGELOG.md) — release history

## Development

```bash
make install      # editable install with dev dependencies
make check        # lint + typecheck + security + test (all CI gates)
make test         # pytest with coverage
```

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

## License

MIT
