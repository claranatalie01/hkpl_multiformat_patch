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

The runtime GLiGuard input-safety classifier uses the GPU selected by
`SAFETY_GPU_ID` and is moved to container-local `SAFETY_DEVICE=cuda:0` when it
is first needed. The `langgraph-agent` container therefore requires the NVIDIA
Container Toolkit. A requested CUDA device never silently falls back to CPU;
if CUDA or sufficient GPU memory is unavailable, the safety stage fails closed
and does not continue to RAG generation. GPU 0 is also used by the embedding
and reranker services in the PoC profile, so its combined memory usage must be
checked with `nvidia-smi` before enabling the agent guard there.

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

Evaluation rows support both single-chunk and multi-chunk evidence. The legacy
`expected_context_snippet` and `source_chunk_id` columns identify the primary
evidence. `expected_context_snippets_json` and `source_chunk_ids_json` contain
parallel JSON arrays when a complete answer requires several chunks, such as a
talk with three dates or a roving exhibition with several venues. The evaluator
reports a complete retrieval only when every labeled chunk (or every equivalent
evidence snippet) reaches the relevant stage.

During candidate generation, sibling chunks from the same webpage are shown to
the question generator. A question about one occurrence must state its exact
date/month and venue/branch. A question about the named activity generally must
combine every matching occurrence and label every supporting chunk. Existing
nine-column CSV files remain readable; run the schema normalizer with `--yes`
when you want to write the two new JSON-array columns into an older file.

To create exactly 100 deduplicated candidate questions, run:

```bash
docker compose run --rm langgraph-agent \
  python scripts/rag_benchmark_workflow.py prepare-candidate \
  --output data/evaluation_dataset_100.csv \
  --target-questions 100
```

The generator may inspect more than 100 chunks because rejected or duplicate
questions do not count toward the 100-row target.
The existing `data/evaluation_dataset.csv` is not changed by this command.

### Jina Reranker v3 Q8_0 experiment

The normal Compose stack uses Qwen. To test the exact
`jinaai/jina-reranker-v3-GGUF:Q8_0` artifact without changing that baseline,
add the Jina override file. Build and start its adapter first:

```bash
docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker.yml \
  build reranker-jina

# Avoid GPU contention with the Qwen reranker during the A/B run.
docker compose stop reranker

docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker.yml \
  up -d reranker-jina

curl http://localhost:8005/health
```

Run the 128-question dense-retrieval experiment with identical retrieval and
output counts:

```bash
docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker.yml \
  run --rm --no-deps \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e RETRIEVAL_MODE=dense \
  -e DENSE_TOP_K=10 \
  -e SIMILARITY_TOP_K=10 \
  -e RERANK_TOP_N=5 \
  -e RERANKER_THRESHOLD=0.30 \
  -e RERANKER_TIMEOUT_SECONDS=120 \
  -e MAX_CONTEXT_TOKENS=4000 \
  -e ANSWER_MAX_TOKENS=256 \
  -e EVALUATION_ANSWER_MAX_TOKENS=512 \
  -e RAG_EVALUATION_RESULTS_PATH=/app/data/rag_evaluation/results_jina_v3_q8.csv \
  -e RAG_EVALUATION_SUMMARY_PATH=/app/data/rag_evaluation/summary_jina_v3_q8.json \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py evaluate \
  --phoenix-project hkpl-rag-jina-v3-q8
```

This is the official GGUF plus external-projector execution path. Its current
upstream implementation starts `llama-embedding` for each reranking request,
so it is expected to be slower than a persistent model server. Use its result
to measure this exact deployable Q8_0 path, not as a hardware-equivalent
comparison with persistent PyTorch/H100 benchmarks.

These shared values match the Qwen dense baseline. The Jina override changes
only the reranker URL, model identity, and tokenizer. The same `0.30` threshold
is retained for this controlled reproduction, although Jina cosine scores and
Qwen relevance scores are not inherently calibrated to the same scale. If
Jina is considered for promotion, run threshold calibration as a separate
experiment instead of silently changing this A/B configuration.

To return to Qwen, omit the override file and restart its service:

```bash
docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker.yml \
  stop reranker-jina
docker compose up -d reranker
```

### Jina Reranker v3.5 Q8_0 experiment

Jina v3.5 uses a different pinned llama.cpp fork and its own tokenizer. Stop
the other rerankers, then build and start the isolated v3.5 service:

```bash
docker compose stop reranker
docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker.yml \
  stop reranker-jina

docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker-v3.5.yml \
  build reranker-jina-v35

docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker-v3.5.yml \
  up -d reranker-jina-v35

curl http://localhost:8006/health
```

Run it against the same 128 questions and the same Qwen baseline settings:

```bash
docker compose \
  -f docker-compose.yml \
  -f infra/compose/jina-reranker-v3.5.yml \
  run --rm --no-deps \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e RETRIEVAL_MODE=dense \
  -e DENSE_TOP_K=10 \
  -e SIMILARITY_TOP_K=10 \
  -e RERANK_TOP_N=5 \
  -e RERANKER_THRESHOLD=0.30 \
  -e RERANKER_TIMEOUT_SECONDS=120 \
  -e MAX_CONTEXT_TOKENS=4000 \
  -e ANSWER_MAX_TOKENS=256 \
  -e EVALUATION_ANSWER_MAX_TOKENS=512 \
  -e RAG_EVALUATION_RESULTS_PATH=/app/data/rag_evaluation/results_jina_v35_q8.csv \
  -e RAG_EVALUATION_SUMMARY_PATH=/app/data/rag_evaluation/summary_jina_v35_q8.json \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py evaluate \
  --phoenix-project hkpl-rag-jina-v35-q8
```

The container downloads the pinned Q8_0 GGUF, projector, tokenizer, and
official `rerank.py` into the shared model cache. Manual `hf download` is not
required.

Each accepted anchor consumes every chunk listed in its
`source_chunk_ids_json`. Those sibling evidence chunks are checkpointed and
skipped as future anchors, preventing the same multi-chunk fact from generating
several paraphrased questions. Each chunk ID may appear in at most one accepted
evaluation row. Siblings that were shown as context but were not cited remain
eligible for other facts. The generator also accepts at most one LLM candidate
per anchor even if the model returns extra JSON objects;
deduplication remains as a final safety check for repeated content across
different source documents.

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
