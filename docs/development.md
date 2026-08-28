# Development and CI/CD

## Local setup

The locked runtime uses Python 3.11 and `uv`:

```bash
cp .env.example .env
uv sync --frozen
make check
make test
```

`PYTHONPATH=src` exposes the `hkpl_agent` package without adding repository
packaging side effects. `main.py` remains a compatibility entry point for
`uvicorn main:app`.

## Repository rules

- Put reusable application logic under `src/hkpl_agent`.
- Keep command-line scripts as argument parsing and orchestration wrappers.
- Keep deployment assets under `infra`.
- Put fast isolated tests under `tests/unit`; add `tests/integration` or
  `tests/e2e` only when those suites have explicit service requirements.
- Give every Python file a module docstring explaining its responsibility.
- Never import from the old `src.*` module names.
- Do not commit `.env`, API keys, Python caches, model artifacts, generated
  evaluation results, or service logs.
- Preserve saved source artifacts in `uploads` when they are required for
  reproducible reindexing; do not treat them as disposable caches.

The dependency-free repository policy is implemented by
`scripts/ci/check_repository.py` and runs before the test suite.

## CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `RAG_ONLY` or
`main`:

1. Install Python 3.11 and `uv`.
2. Recreate the locked environment with `uv sync --frozen`.
3. Check syntax, module documentation, imports, layout, and tracked artifacts.
4. Run unit tests.
5. Validate the Docker Compose model in a separate lightweight job.

Run the same gates locally:

```bash
make ci
```

## Container delivery

`.github/workflows/container.yml` builds the agent image when relevant files
change in a pull request. A reviewed Git tag matching `v*` also publishes the
same image to GitHub Container Registry:

```text
ghcr.io/<owner>/<repository>/agent:<tag>
```

This is a delivery mechanism, not an automatic production deployment. A real
LCSD deployment still needs an approved environment, pinned model artifacts,
secrets management, database migrations, release gates, and rollback policy.

## Pull-request checklist

- [ ] Behavior change has a focused test.
- [ ] `make check` passes.
- [ ] `make test` passes.
- [ ] `docker compose config --quiet` passes.
- [ ] Environment-variable or operator-command changes are documented.
- [ ] Database changes are backward-compatible or have an explicit migration.
- [ ] No secret, log, cache, model, or generated benchmark output is staged.
- [ ] Ingestion changes preserve the last known good corpus on failure.
- [ ] Retrieval or prompt changes are evaluated on the reviewed benchmark.
