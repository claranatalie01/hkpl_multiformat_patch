# HKPL Agentic RAG

HKPL Agentic RAG is a bounded LangGraph application for answering questions
from approved Hong Kong Public Libraries webpages and documents. It combines
multi-format ingestion, PostgreSQL/pgvector retrieval, multilingual embedding
and reranking, evidence-grounded generation, safety controls, and Phoenix-based
development evaluation.

This repository is a proof-of-concept implementation. The target production
architecture and governance requirements are defined in `AGENTS.md`.

## Start here

Prerequisites:

- Docker Engine with Docker Compose
- NVIDIA Container Toolkit and two development GPUs for the full PoC profile
- Git and a project `.env` created from `.env.example`

```bash
cp .env.example .env
docker compose up -d --build
docker compose ps
```

The main local endpoints are:

| Service | URL | Purpose |
|---|---|---|
| Agent API | <http://localhost:8001> | Chat and protected administration API |
| Phoenix | <http://localhost:6006> | Development/evaluation traces |
| PostgreSQL | `localhost:5433` | Registry, evaluation rows, and pgvector data |
| Embedding | <http://localhost:8003> | Qwen embedding service |
| Reranker | <http://localhost:8004> | Qwen reranking service |
| Generator | <http://localhost:8081> | Qwen answer and judge service |

Do not commit `.env`, credentials, generated traces, or model artifacts.

## System flow

```text
approved upload or allowlisted HKPL URL
        |
        v
acquire -> extract -> normalize -> chunk -> embed -> pgvector
                                                    |
user -> safety -> intent -> rewrite -> configured retrieve -> rerank
                                                    |
                                                    v
                             context pack -> generate -> validate -> answer
                                                    |
                                                    v
                                  OpenTelemetry / Phoenix evaluation
```

The online graph is finite and inspectable. Retrieved text is treated as
untrusted evidence, not as executable instructions or policy.

## Common commands

Show all supported developer shortcuts:

```bash
make help
```

Run the repository checks and unit tests:

```bash
make check
make test
```

Check corpus counts:

```bash
docker compose run --rm langgraph-agent \
  python scripts/rag_benchmark_workflow.py status
```

Run a small crawl and ingestion smoke test:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/crawl_hkpl_site.py \
  --max-pages 5 \
  --max-depth 1
```

Generate a reviewable 100-question candidate dataset:

```bash
docker compose run --rm langgraph-agent \
  python scripts/rag_benchmark_workflow.py prepare-candidate \
  --output data/evaluation_dataset_100.csv \
  --target-questions 100 \
  --all-chunks
```

Evaluate three rows and send traces to Phoenix:

```bash
docker compose run --rm \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_100 \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py evaluate \
  --limit 3 \
  --phoenix-project hkpl-rag-smoke
```

See the focused runbooks below before running a full crawl, changing the
corpus lock, promoting an evaluation dataset, or switching vector tables.

## Repository layout

```text
.
├── src/hkpl_agent/
│   ├── app.py               # FastAPI application
│   ├── api/                 # HTTP schemas and transport helpers
│   ├── agent/               # LangGraph state, nodes, and graph
│   ├── rag/                 # Retrieval and grounded-answer construction
│   ├── ingestion/           # Readers, chunking, registry, and ingestion service
│   ├── infrastructure/      # PostgreSQL, pgvector, embedding, and model clients
│   ├── safety/              # Compliance and safety policy code
│   ├── memory/              # Conversation persistence
│   ├── observability/       # OpenTelemetry and Phoenix adapters
│   └── evaluation/          # Shared evaluation schemas
├── scripts/                 # Operator-facing ingestion/evaluation commands
├── tests/
│   ├── unit/                # Fast automated tests
│   └── fixtures/            # Stable local test inputs
├── infra/
│   ├── docker/              # Agent image definition
│   └── postgres/            # PostgreSQL bootstrap SQL
├── docs/                    # Focused architecture and operational guides
├── data/                    # Versioned inputs and ignored generated results
├── uploads/                 # Saved source artifacts used for reproducible reindexing
├── storage/                 # Generated parser and crawler state
├── main.py                  # Compatibility ASGI entry point
└── docker-compose.yml       # Local/PoC runtime topology
```

## Documentation

- [Documentation index](docs/README.md)
- [Proof-of-concept report](docs/poc-report.md)
- [Architecture and code map](docs/architecture.md)
- [Ingestion, chunking, embedding, and vector storage](docs/ingestion.md)
- [Evaluation dataset generation and Phoenix](docs/evaluation.md)
- [Operations and troubleshooting](docs/operations.md)
- [Development and CI/CD](docs/development.md)

## Design boundaries

- PostgreSQL with pgvector is the vector database; FAISS is not used.
- The crawler discovers and downloads sources; the ingestion service owns
  extraction, chunking, embedding, and vector insertion.
- The LLM does not build the vector database. It is used for bounded source
  classification, answer generation, and evaluation-question/judge tasks.
- The write guard prevents unintended mutations even when ingestion is invoked
  outside cron. Scheduled crawling and database write protection are separate
  controls.
- Phoenix is an evaluation/development viewer, not the production audit store.
- `data_hkpl_knowledge` is a physical pgvector table name produced from the
  logical `VECTOR_TABLE=hkpl_knowledge` setting.

## Contributing

Keep business logic in `src/hkpl_agent`; scripts should parse arguments and
delegate. New modules need a module docstring and tests in the matching test
area. Before opening a pull request, run:

```bash
make ci
```

CI repeats the repository checks and unit tests on pushes and pull requests.
Version tags matching `v*` build and publish the agent image to GitHub Container
Registry after the same source is reviewed.
