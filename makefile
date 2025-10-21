.PHONY: help install dev format format-check lint lint-unsafe type-check test test-cov clean build check all demo python-version

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

python-version: ## Show Python version in use
	@uv run python --version

install: ## Install runtime dependencies only
	uv sync

dev: ## Install all dependencies (including dev)
	uv sync --all-extras

format: ## Format code with ruff
	uv run ruff format src

format-check: ## Check code formatting without modifying
	uv run ruff format --check src

lint: ## Lint code with ruff (auto-fix safe issues)
	uv run ruff check src --fix

lint-unsafe: ## Lint code with ruff (including unsafe fixes)
	uv run ruff check src --fix --unsafe-fixes

lint-check: ## Check linting without auto-fixing
	uv run ruff check src

type-check: ## Type check with pyright
	uv run pyright src

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=src/peepomap --cov-report=term-missing --cov-report=xml

clean: ## Clean Python cache files and build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
	rm -rf dist build *.egg-info

build: ## Build package distributions
	uv build

demo: ## Generate colormap demo images
	uv run python -m peepomap

check: format-check lint-check type-check ## Run all checks without modifying code

ci: format lint type-check test ## Run CI checks (with auto-fixes)

all: clean ci build ## Run complete pipeline: clean, check, test, build 
