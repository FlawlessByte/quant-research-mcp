# quant_research_mcp — common tasks. Run `make help` for the list.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Absolute project dir so `make register` emits a portable client config.
PROJECT_DIR := $(shell pwd)

# GHCR image name (lowercase, per registry rules).
IMAGE := ghcr.io/flawlessbyte/quant-research-mcp

.PHONY: help install install-cli uninstall-cli dev run inspect smoke test check \
        build docker-build mcpb clean register claude-add claude-remove

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies (uv sync)
	uv sync

install-cli: ## Install the `quant-research-mcp` binary onto your PATH (uv tool)
	# --reinstall --no-cache forces a fresh build so source edits ship even when
	# the project version is unchanged (uv otherwise serves a cached wheel).
	uv tool install --force --reinstall --no-cache .
	@echo
	@echo "Installed. Binary location:"
	@command -v quant-research-mcp || { \
		echo "  not on PATH yet — run: uv tool update-shell  (then restart your shell)"; \
		echo "  or add $$(uv tool dir)/../bin to PATH"; }

uninstall-cli: ## Remove the `quant-research-mcp` binary from your PATH
	uv tool uninstall quant-research-mcp

dev: ## Install plus dev extras (ruff) for linting
	uv sync
	uv add --dev ruff

run: ## Run the MCP server over stdio
	uv run python server.py

inspect: ## Launch the MCP Inspector against the server
	npx @modelcontextprotocol/inspector uv run python server.py

smoke: ## Import the server and list its tools (no network)
	uv run python -c "import asyncio, server; \
		print('tools:', sorted(t.name for t in asyncio.run(server.mcp.list_tools())))"

test: ## Run the pytest suite (no network needed)
	uv run pytest tests/ -q

check: ## Tests + lint + confirm no LLM/subprocess calls
	uv run pytest tests/ -q
	-uv run ruff check . 2>/dev/null || echo "ruff not installed; run 'make dev'"
	@if grep -rniE 'subprocess|claude -p' --include='*.py' \
		--exclude-dir=.venv --exclude-dir=__pycache__ . \
		| grep -v 'no LLM, no network, no subprocess'; then \
		echo "FAIL: LLM/subprocess reference found"; exit 1; \
	else echo "OK: no LLM/subprocess calls"; fi

register: ## Print the stdio MCP client config block for this project
	@echo '{'
	@echo '  "mcpServers": {'
	@echo '    "quant_research": {'
	@echo '      "command": "uv",'
	@echo '      "args": ["run", "python", "server.py"],'
	@echo '      "cwd": "$(PROJECT_DIR)"'
	@echo '    }'
	@echo '  }'
	@echo '}'

claude-add: ## Register this local clone with Claude Code (user scope)
	claude mcp add quant_research --scope user -- uv run --directory $(PROJECT_DIR) python server.py

claude-remove: ## Remove this server from Claude Code
	claude mcp remove quant_research

build: ## Build the sdist + wheel into dist/
	uv build

docker-build: ## Build the Docker image locally
	docker build -t $(IMAGE) .

mcpb: ## Pack a Claude Desktop .mcpb bundle
	npx -y @anthropic-ai/mcpb pack . quant-research-mcp.mcpb

clean: ## Remove caches and build artifacts
	rm -rf .venv __pycache__ */__pycache__ */*/__pycache__ .ruff_cache dist build *.egg-info *.mcpb
