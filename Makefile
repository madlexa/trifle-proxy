.PHONY: help install test lint format typecheck security audit check clean

PY ?= python
PIP ?= $(PY) -m pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"

test: ## Run the test suite with coverage
	pytest tests/ -v

lint: ## Lint with ruff (check only)
	ruff check src tests

format: ## Auto-format and auto-fix with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck: ## Type-check with mypy
	mypy src

security: ## Run bandit security scan (fail on medium+)
	bandit -c pyproject.toml -r src --severity-level medium

audit: ## Check dependencies for known vulnerabilities
	pip-audit --strict

check: lint typecheck security test ## Run all quality gates (lint, typecheck, security, test)

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
