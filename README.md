# HKPL Agentic RAG

This repository implements a locally hosted retrieval-augmented generation
(RAG) service for Hong Kong Public Libraries content. It can crawl or ingest
multi-format sources, extract and chunk their text, create embeddings with a
local Qwen model, store them in PostgreSQL/pgvector, retrieve and rerank relevant
evidence, and generate source-aware answers.

## Documentation

- [Implementation guide](IMPLEMENTATION_GUIDE.md): installation, corpus locks,
  ingestion commands, API usage, chunking, and evaluation procedures.
- [Data documentation](DATA_DOCUMENTATION.md): corpus inventory, evaluation
  fields, metrics, benchmark results, and reproducibility controls.
- [Collaboration guide](COLLABORATION_GUIDE.md): architecture summary,
  contributor workflow, ownership, and operational safeguards.

## High-level workflow

```text
Website or uploaded document
        ↓
Extract structured text
        ↓
Create document-aware chunks
        ↓
Generate 1,024-dimensional embeddings
        ↓
Store chunks, metadata, and vectors in PostgreSQL/pgvector
        ↓
Embed user question and retrieve similar chunks
        ↓
Rerank evidence and generate a cited answer
```

The bulk website ingestion entry point is
[`scripts/crawl_hkpl_site.py`](scripts/crawl_hkpl_site.py). It is a one-shot
command, not a scheduled crawler. New or changed sources are handed to the
shared ingestion service, which completes extraction, chunking, embedding, and
vector insertion before the crawler continues.

## File-by-file guide

### Runtime and configuration

| File | Individual responsibility |
| --- | --- |
| [`docker-compose.yml`](docker-compose.yml) | Starts PostgreSQL, embedding, reranker, LLM, FastAPI, and Phoenix. It does not schedule the crawler. |
| [`Dockerfile.agent`](Dockerfile.agent) | Builds the Python container and installs Tesseract OCR. |
| [`pyproject.toml`](pyproject.toml) | Lists Python dependencies. |
| `uv.lock` | Pins exact dependency versions. |
| [`.env.example`](.env.example) | Documents secrets and Compose substitutions expected in a local `.env`. |
| [`postgres-init/init.sql`](postgres-init/init.sql) | Enables pgvector and creates initial ordinary tables for a fresh database. |

### Infrastructure modules

| File | Individual responsibility |
| --- | --- |
| [`src/infrastructure/db.py`](src/infrastructure/db.py) | Creates the SQLAlchemy connection for registry, memory, and maintenance SQL. |
| [`src/infrastructure/embedding.py`](src/infrastructure/embedding.py) | Adapts the local embedding HTTP endpoint to LlamaIndex. |
| [`src/infrastructure/vector_store.py`](src/infrastructure/vector_store.py) | Configures the `hkpl_knowledge` LlamaIndex PGVectorStore with 1,024 dimensions. |

### Ingestion modules

| File | Individual responsibility |
| --- | --- |
| [`scripts/crawl_hkpl_site.py`](scripts/crawl_hkpl_site.py) | Crawls HKPL, checks `robots.txt`, discovers links, cleans pages, and starts ingestion. |
| [`src/ingestion/service.py`](src/ingestion/service.py) | Coordinates registration, extraction, chunking, embedding, and insertion. |
| [`src/ingestion/registry.py`](src/ingestion/registry.py) | Manages document identity, hashes, versions, status, and chunk counts. |
| [`src/ingestion/readers.py`](src/ingestion/readers.py) | Extracts structured text from HTML, PDF, Office, JSON, XML, and image formats. |
| [`src/ingestion/document_types.py`](src/ingestion/document_types.py) | Classifies content and chooses its chunking strategy. |
| [`src/ingestion/chunking.py`](src/ingestion/chunking.py) | Splits extracted sections, removes duplicates, and assigns stable chunk IDs. |
| [`src/ingestion/write_guard.py`](src/ingestion/write_guard.py) | Stops Python corpus writes when read-only mode is enabled. |
| [`src/ingestion/webpage.py`](src/ingestion/webpage.py) | Downloads and cleans one URL supplied through the admin API. |
| [`scripts/ingest_documents.py`](scripts/ingest_documents.py) | Ingests files already available locally. |
| [`scripts/ingest_pgvector_llamaindex.py`](scripts/ingest_pgvector_llamaindex.py) | Rebuilds FAQ/registered sources, synchronizes evaluation data, and audits chunks. |
| [`scripts/manage_corpus_lock.py`](scripts/manage_corpus_lock.py) | Controls database triggers that protect the frozen corpus from direct SQL changes. |

### API and RAG modules

| File | Individual responsibility |
| --- | --- |
| [`main.py`](main.py) | Hosts chat and admin HTTP endpoints. |
| [`src/graph.py`](src/graph.py) | Defines LangGraph node connections and routing. |
| [`src/state.py`](src/state.py) | Defines information passed between workflow nodes. |
| [`src/nodes.py`](src/nodes.py) | Implements safety, intent, rewriting, retrieval, answering, citations, and saving. |
| [`src/retrieval.py`](src/retrieval.py) | Embeds questions, searches pgvector, and reranks candidates. |
| [`src/llm_client.py`](src/llm_client.py) | Calls the generative answer model and normalizes token usage. |
| [`src/memory.py`](src/memory.py) | Loads and saves recent conversation history. |
| [`src/compliance.py`](src/compliance.py) | Stores and applies prohibited-keyword rules. |
| [`src/corpus.py`](src/corpus.py) | Distinguishes primary HKPL vectors from benchmark distractors. |

### Evaluation and benchmark scripts

| File | Individual responsibility |
| --- | --- |
| [`scripts/generate_evaluation_dataset.py`](scripts/generate_evaluation_dataset.py) | Generates candidate questions from existing vector chunks. |
| [`scripts/validate_evaluation_dataset.py`](scripts/validate_evaluation_dataset.py) | Verifies expected answers, snippets, and chunk references. |
| [`scripts/evaluate_rag.py`](scripts/evaluate_rag.py) | Measures retrieval, reranking, correctness, faithfulness, and latency. |
| [`scripts/rag_benchmark_workflow.py`](scripts/rag_benchmark_workflow.py) | Coordinates candidate generation, validation, promotion, and evaluation. |
| [`scripts/migrate_evaluation_benchmark.py`](scripts/migrate_evaluation_benchmark.py) | Migrates labels to current document and chunk IDs. |
| [`scripts/normalize_evaluation_schema.py`](scripts/normalize_evaluation_schema.py) | Normalizes evaluation CSV structure. |
| [`scripts/hotpotqa_benchmark.py`](scripts/hotpotqa_benchmark.py) | Adds HotpotQA retrieval distractors. |
| [`scripts/webz_news_benchmark.py`](scripts/webz_news_benchmark.py) | Adds external news retrieval distractors. |

### Observability modules

| File | Individual responsibility |
| --- | --- |
| [`src/observability.py`](src/observability.py) | Initializes Phoenix/OpenTelemetry. |
| [`src/tracing_helpers.py`](src/tracing_helpers.py) | Adds prompts, outputs, and documents to trace spans. |
| [`src/phoenix_annotations.py`](src/phoenix_annotations.py) | Adds evidence and answer-quality annotations. |
| [`src/token_counting.py`](src/token_counting.py) | Counts model tokens for limits and evaluation reporting. |

## Core database distinction

The application uses two related storage layers:

| Storage | Purpose |
| --- | --- |
| `knowledge_documents` | Document-level registry containing source identity, hash, version, status, and chunk count. |
| `data_hkpl_knowledge` | LlamaIndex-managed vector table containing searchable chunk text, metadata, IDs, and embeddings. |

`postgres-init/init.sql` enables pgvector and creates the ordinary application
tables. `src/infrastructure/vector_store.py` configures LlamaIndex, which
creates and manages `data_hkpl_knowledge` when the vector store is used.
