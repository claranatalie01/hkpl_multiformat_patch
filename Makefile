SHELL := /bin/sh
UV ?= uv
PYTHON ?= python

.PHONY: help install check test ci compose-config up down logs corpus-status crawl-smoke

help: ## Show the supported developer commands.
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "%-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the locked Python environment.
	$(UV) sync --frozen

check: ## Run dependency-free repository and Python syntax checks.
	PYTHONPATH=src $(UV) run python scripts/ci/check_repository.py
	PYTHONPATH=src $(UV) run python -m compileall -q src scripts tests main.py

test: ## Run the unit test suite.
	PYTHONPATH=src $(UV) run python -m unittest discover -s tests/unit -v

compose-config: ## Validate the Docker Compose configuration.
	docker compose config --quiet

ci: check test compose-config ## Run the same checks used by CI.

up: ## Start the local PoC stack.
	docker compose up -d --build

down: ## Stop the local PoC stack without deleting volumes.
	docker compose down

logs: ## Follow the application logs.
	docker compose logs -f langgraph-agent

corpus-status: ## Show vector-corpus counts.
	docker compose run --rm langgraph-agent python scripts/rag_benchmark_workflow.py status

crawl-smoke: ## Crawl a small HKPL sample (requires a writable corpus).
	docker compose run --rm -e KNOWLEDGE_CORPUS_READ_ONLY=false langgraph-agent \
		python scripts/crawl_hkpl_site.py --max-pages 5 --max-depth 1

