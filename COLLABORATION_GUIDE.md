# HKPL Agentic RAG — Collaboration Guide

## Start here

This repository contains a FastAPI/LangGraph RAG service for Hong Kong Public
Libraries. PostgreSQL/pgvector stores searchable chunks, llama.cpp serves the
embedding, reranking, and answer models, and Phoenix records traces and
evaluation results.

Read these documents in order:

1. `COLLABORATION_GUIDE.md` — how the team changes and operates the project.
2. `IMPLEMENTATION_GUIDE.md` — setup, ingestion, API, and evaluation commands.
3. `DATA_DOCUMENTATION.md` — corpus lineage, labels, metrics, and governance.

The local repository is for development. The remote server owns the working
Docker volumes, database, models, crawler state, and Phoenix history. Git is
the source of truth for code.

## Architecture

```text
Documents -> extract -> chunk -> embed -> PostgreSQL/pgvector

Question -> safety -> intent -> rewrite -> retrieve 10 -> rerank 5
         -> answer from context -> citations -> output safety -> response

Retrieval + reranking + generation + evaluation -> Phoenix traces
```

Docker services:

| Service | Host port | Purpose |
|---|---:|---|
| `postgres` | 5433 | Metadata, evaluation rows, and pgvector |
| `embedding` | 8003 | Qwen3 embedding model |
| `reranker` | 8004 | Qwen3 reranker |
| `llm` | 8081 | Qwen3.5 answer and evaluator model |
| `langgraph-agent` | 8001 | FastAPI/LangGraph application |
| `phoenix` | 6006 | Traces and evaluation annotations |

Important code:

| Path | Responsibility |
|---|---|
| `main.py` | API routes, uploads, chat streaming, admin security |
| `src/graph.py` | Workflow order and branches |
| `src/nodes.py` | Safety, intent, RAG, answer, citations, memory |
| `src/retrieval.py` | Vector search, reranking, fallbacks, traces |
| `src/ingestion/` | Readers, chunking, registry, indexing, write guard |
| `scripts/evaluate_rag.py` | Per-question and aggregate RAG evaluation |
| `scripts/rag_benchmark_workflow.py` | Corpus lock and benchmark orchestration |

## Working agreement

Every task needs a goal, one owner, likely files, a proof command, risk, and
rollback point. Keep tasks small enough for the other collaborator to review.
Agree on ownership before both people edit the same file.

Suggested temporary split:

- Collaborator A: API, graph, prompts, retrieval, and reranking.
- Collaborator B: ingestion, datasets, evaluation, and Phoenix analysis.
- Both: configuration, releases, database changes, and backups.

Rotate ownership so neither person becomes the only one who understands a
subsystem.

## Git workflow

Create a branch from the current shared branch:

```bash
git fetch origin
git switch RAG_ONLY
git pull --ff-only
git switch -c feature/short-description
```

Before review:

```bash
git status
git diff --check
git diff
git add <intentional-files>
git commit -m "Describe the completed outcome"
git push -u origin feature/short-description
```

The reviewer checks behavior, simplicity, secrets, generated files, data
effects, tests, and rollback. Merge through GitHub after review. Do not
force-push a shared branch or use `git reset --hard` to resolve collaboration
problems.

Never commit `.env`, logs, Python caches, generated evaluation reports,
downloaded benchmark caches, or crawler runtime state. `.env.example` contains
the variables that the current Compose file actually interpolates. Retrieval,
chunking, model, and path defaults are currently set in `docker-compose.yml`;
putting same-named values in `.env` does not override those literal entries.

## Shared remote server

Only one person should deploy, reindex, change model configuration, mutate the
corpus, or run a full benchmark at a time. These operations share GPUs and
state. The checked-in Compose configuration sets
`KNOWLEDGE_CORPUS_READ_ONLY=true`, so normal API corpus mutations return HTTP
423 and corpus-writing scripts stop unless an operator deliberately opens a
maintenance window. Follow the lock procedure in `IMPLEMENTATION_GUIDE.md`.

Announce each remote operation:

```text
Task: five-question RAG smoke test
Branch/commit: feature/example @ abc1234
Impact: GPU load only; no corpus mutation
Expected duration: <estimate>
```

Before deployment:

```bash
git status --short
git branch --show-current
git rev-parse --short HEAD
docker compose ps
```

Afterward, record the commit, commands, configuration/data changes, output
paths, Phoenix project, result, and rollback point. Never paste secrets into a
run log or chat.

## Minimum verification

Start and inspect the remote stack:

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 langgraph-agent
```

Validate the evaluation data and corpus state using the commands in
`DATA_DOCUMENTATION.md`. Both HotpotQA and Webz News must exist in the shared
vector table, and all active evaluation rows must link to current evidence.
Then run the five-question smoke evaluation:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/evaluate_rag.py --limit 5
```

For retrieval, model, prompt, chunking, or configuration changes, run the full
benchmark workflow described in `DATA_DOCUMENTATION.md` and compare its
summary with the last known-good baseline. Inspect at least one Phoenix trace.

## Safety rules

- Back up PostgreSQL and `.env` before migrations or major rebuilds.
- Do not run `docker compose down -v`; it removes persistent volumes.
- Do not change the embedding model/dimension without rebuilding vectors.
- Do not edit only inside a running container; commit source changes.
- Keep reranker fallbacks: they preserve service availability and expose the
  failure through tracing.
- Treat uploads and HKPL source files as corpus inputs, not repository trash.

## Known limitations

- Coordinate-based nearest-library resolution is still a placeholder.
- Background ingestion uses FastAPI background tasks, not a durable queue; a
  restart can interrupt indexing.
- The evaluator uses the same model family as answer generation, so borderline
  results benefit from manual or independent review.
- GPU model startup can take several minutes; check container health before
  debugging Python code.

## First collaboration session

Together, complete this path:

```text
clone -> copy .env.example to .env -> inspect Docker services
-> start stack -> run five-question evaluation -> open Phoenix
-> inspect one retrieval/reranking trace -> test one chat question
```

Then swap roles: the new collaborator explains the trace and likely failure
stage while the existing collaborator reviews. That confirms shared
understanding of both operation and implementation.
