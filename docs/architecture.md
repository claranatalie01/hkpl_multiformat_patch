# Architecture and code map

## Runtime boundaries

The repository is a modular monolith with an ingestion workflow. LangGraph is
used to make the online flow finite and observable; it is not an unrestricted
autonomous agent.

```text
FastAPI transport
    -> LangGraph workflow
        -> safety and intent
        -> query rewrite
        -> dense + lexical retrieval
        -> reranking and context packing
        -> grounded answer generation
        -> citations and output safety
    -> PostgreSQL session persistence

Crawler or administrator upload
    -> ingestion service
        -> reader
        -> structure-aware chunker
        -> embedding adapter
        -> PostgreSQL/pgvector
        -> document registry
```

Dependencies point inward: HTTP handlers and command-line scripts call reusable
application modules. Application modules use infrastructure adapters. Core
logic must not call a CLI script.

## Application package

| File | Responsibility |
|---|---|
| `src/hkpl_agent/app.py` | FastAPI lifecycle, chat streaming, and protected admin routes |
| `src/hkpl_agent/api/schemas.py` | Typed HTTP request contracts |
| `src/hkpl_agent/api/uploads.py` | Upload size, extension, signature, and filename validation |
| `src/hkpl_agent/api/streaming.py` | Server-sent event formatting |
| `src/hkpl_agent/agent/factory.py` | Complete initial state for one graph invocation |
| `src/hkpl_agent/agent/state.py` | Typed state shared by graph nodes |
| `src/hkpl_agent/agent/graph.py` | Finite LangGraph topology and conditional routing |
| `src/hkpl_agent/agent/nodes.py` | Safety, intent, rewrite, retrieval, generation, citation, and persistence nodes |
| `src/hkpl_agent/rag/retrieval.py` | Dense/lexical candidate retrieval, rank fusion, reranking, and diagnostics |
| `src/hkpl_agent/rag/answering.py` | Canonical evidence-grounded prompt and context formatting |
| `src/hkpl_agent/rag/corpus.py` | Corpus-role metadata and dataset-scoped vector maintenance |
| `src/hkpl_agent/safety/compliance.py` | Prohibited-keyword policy storage and checks |
| `src/hkpl_agent/memory/repository.py` | Conversation-history reads and writes |
| `src/hkpl_agent/observability/setup.py` | Optional OpenTelemetry/Phoenix setup |
| `src/hkpl_agent/observability/tracing.py` | Safe span input/output and token attributes |
| `src/hkpl_agent/observability/annotations.py` | Evaluation annotations attached to Phoenix spans |

## Ingestion package

| File | Responsibility |
|---|---|
| `ingestion/service.py` | Transactional coordinator for registration, extraction, chunking, embedding, and replacement |
| `ingestion/config.py` | Shared uploads path and OCR language configuration |
| `ingestion/readers.py` | HTML, PDF, DOCX, text, Markdown, CSV, XLSX, JSON, and XML extraction |
| `ingestion/chunking.py` | Stable structure-aware and token-bounded chunks |
| `ingestion/document_types.py` | Document labels and chunking-policy selection |
| `ingestion/formats.py` | Supported, legacy, deterministic, and Docling file extensions |
| `ingestion/classification.py` | Bounded document-type classification with deterministic fallback |
| `ingestion/tokenizer.py` | Pinned embedding tokenizer used for chunk limits |
| `ingestion/registry.py` | Source document identity, versions, hashes, status, and chunk counts |
| `ingestion/webpage.py` | Single approved webpage acquisition and cleanup |
| `ingestion/html_utils.py` | Deterministic HTML text normalization |
| `ingestion/write_guard.py` | Application-level corpus mutation protection |

## Infrastructure and evaluation packages

| File | Responsibility |
|---|---|
| `infrastructure/db.py` | Shared SQLAlchemy PostgreSQL engine |
| `infrastructure/embedding.py` | Local Qwen embedding endpoint adapter |
| `infrastructure/vector_store.py` | LlamaIndex PostgreSQL/pgvector configuration and hybrid-search schema |
| `infrastructure/table_names.py` | Strict validation and translation of configurable SQL identifiers |
| `infrastructure/llm_client.py` | Local OpenAI-compatible generator/judge calls and usage normalization |
| `infrastructure/token_counting.py` | Local tokenizer endpoint calls and conservative fallback counts |
| `evaluation/schema.py` | Canonical evaluation CSV/SQL row parsing and evidence-array rules |

## Operator scripts

Scripts expose repeatable operations; shared logic belongs in the package.

| Script | Responsibility |
|---|---|
| `crawl_hkpl_site.py` | Bounded, robots-aware HKPL discovery and ingestion scheduling |
| `ingest_documents.py` | Ingest a local file or directory through the shared service |
| `ingest_pgvector_llamaindex.py` | Evaluation-table ingestion, corpus audit, and maintenance operations |
| `generate_evaluation_dataset.py` | Generate reviewable questions from existing HKPL vector chunks |
| `validate_evaluation_dataset.py` | Validate schema and evidence references against pgvector |
| `normalize_evaluation_schema.py` | Normalize legacy evaluation CSV columns |
| `migrate_evaluation_benchmark.py` | Controlled migration of an evaluation benchmark table |
| `rag_benchmark_workflow.py` | High-level status, prepare, validate, promote, and evaluate workflow |
| `evaluate_rag.py` | Retrieval/reranking/generation metrics and Phoenix traces |
| `hotpotqa_benchmark.py` | Manage the HotpotQA distractor corpus |
| `webz_news_benchmark.py` | Manage the Webz news distractor corpus |
| `manage_corpus_lock.py` | Enable, disable, or inspect corpus write protection |
| `ci/check_repository.py` | Dependency-free structural and syntax policy checks |

## Why the code is not arranged like a traditional RAG demo

A traditional RAG demo often has one linear ingest-and-query script. This
project retains separate agent workflow, safety, memory, observability,
evaluation, versioned ingestion, and administrative controls because those are
real system responsibilities. The structural lesson adopted here is separation
of concerns, not removal of the agent-specific behavior.
