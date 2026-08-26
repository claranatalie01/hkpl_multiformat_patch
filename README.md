# HKPL Agentic RAG

This repository implements a locally hosted retrieval-augmented generation
(RAG) service for Hong Kong Public Libraries content. It crawls or accepts
documents, extracts structured evidence, creates searchable chunks, generates
1,024-dimensional embeddings, stores them in PostgreSQL/pgvector, retrieves and
reranks evidence, generates grounded answers, and records evaluation traces in
Phoenix.

This README is the single human-facing operational guide. `AGENTS.md` contains
repository-level architecture and safety instructions for coding agents and is
kept separately because it controls repository maintenance rather than runtime
operation.

## Quick command map

Run commands from the repository root on the server.

| Goal | Entry point |
| --- | --- |
| Start the stack | `docker compose up -d` |
| Inspect services | `docker compose ps` |
| Crawl, chunk, embed, and ingest HKPL webpages | `scripts/crawl_hkpl_site.py` |
| Ingest a local file or directory | `scripts/ingest_documents.py` |
| Rebuild saved sources with the current chunker | `scripts/ingest_pgvector_llamaindex.py --rebuild-all` |
| Check or change the corpus lock | `scripts/manage_corpus_lock.py` |
| Audit stored chunks | `scripts/ingest_pgvector_llamaindex.py --audit-chunks` |
| Generate evaluation questions | `scripts/rag_benchmark_workflow.py prepare-candidate` |
| Validate evaluation evidence | `scripts/validate_evaluation_dataset.py` |
| Run RAG evaluation and send traces to Phoenix | `scripts/rag_benchmark_workflow.py evaluate` |
| Debug one evaluation question | `scripts/evaluate_rag.py --question-exact ...` |
| Open PostgreSQL | `docker compose exec postgres psql -U postgres -d hkpl_vector_db` |
| Open Phoenix | `http://SERVER_HOST:6006` |

`chunking.py`, `embedding.py`, and `vector_store.py` are library modules. Do
not run them directly. The ingestion service calls them in the correct order.

## System workflow

```text
Approved webpage or document
        ↓
Acquire and save immutable source
        ↓
Register source, hash, version, and status
        ↓
Extract structured evidence
        ↓
Create structure-aware, token-bounded chunks
        ↓
Generate Qwen embeddings
        ↓
Store text + metadata + vector in PostgreSQL/pgvector
        ↓
Embed user query and retrieve vector candidates
        ↓
Rerank candidates and pack final context
        ↓
Generate grounded answer
        ↓
Evaluate and inspect traces in Phoenix
```

The crawler is a one-shot command. Docker Compose starts services but does not
schedule recurring crawls. A user-owned cron job may schedule it separately.

## Docker services

| Service | Host port | Purpose |
| --- | ---: | --- |
| `postgres` | 5433 | Registry, application tables, evaluation rows, and pgvector |
| `embedding` | 8003 | Qwen3-Embedding-0.6B, 1,024-dimensional vectors |
| `reranker` | 8004 | Qwen3-Reranker-0.6B |
| `llm` | 8081 | Qwen3.5-9B answer generation, classification, and evaluation |
| `langgraph-agent` | 8001 | FastAPI and LangGraph application |
| `phoenix` | 6006 | Development/evaluation traces and annotations |

The Compose profile uses GPU 0 for embedding/reranking and GPU 1 for the LLM.
The agent safety model uses the physical GPU selected by `SAFETY_GPU_ID`; that
device appears as `cuda:0` inside the agent container.

## Installation and startup

Create the writable host directories and environment file:

```bash
mkdir -p uploads storage data/rag_evaluation
cp .env.example .env
```

Set non-placeholder values in `.env`, especially:

```text
DB_PASSWORD
ADMIN_API_KEY
```

Never commit `.env`. Start the services:

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 langgraph-agent
```

Do not run `docker compose down -v` unless permanent deletion of PostgreSQL,
Phoenix, and model-cache volumes is explicitly intended.

## Configuration and physical table names

Important defaults are defined in `docker-compose.yml`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `VECTOR_TABLE` | `hkpl_knowledge` | Logical LlamaIndex collection name |
| `EMBED_DIM` | `1024` | Required pgvector dimension |
| `CHUNK_SIZE` | `512` | Maximum embedding tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap only for oversized structural leaves |
| `SIMILARITY_TOP_K` | `10` | Vector candidates |
| `DENSE_TOP_K` | `30` | Dense-vector candidates used by hybrid retrieval |
| `LEXICAL_TOP_K` | `30` | Full-text/trigram candidates used by hybrid retrieval |
| `FUSION_TOP_K` | `20` | Candidates retained after reciprocal-rank fusion |
| `RRF_K` | `60` | Reciprocal-rank-fusion constant |
| `RERANK_TOP_N` | `8` | Final reranked contexts |
| `MAX_CONTEXT_TOKENS` | `12000` | Answer context budget |
| `EVALUATION_ANSWER_MAX_TOKENS` | `512` | Visible-answer allowance |
| `EVALUATION_MAX_TOKENS` | `2048` | Evaluator-judge completion limit |
| `KNOWLEDGE_CORPUS_READ_ONLY` | `true` | Application-level write guard |

LlamaIndex automatically creates/uses a physical table named
`data_<VECTOR_TABLE>`:

```text
VECTOR_TABLE=hkpl_knowledge
→ data_hkpl_knowledge

VECTOR_TABLE=hkpl_knowledge_hybrid
→ data_hkpl_knowledge_hybrid
```

Never pass the `data_` prefix to `VECTOR_TABLE`, or LlamaIndex will target a
table such as `data_data_hkpl_knowledge_hybrid`.

## Corpus write protection

Knowledge writes require both protections to be opened:

1. the PostgreSQL corpus lock must be disabled;
2. the individual write command must set
   `KNOWLEDGE_CORPUS_READ_ONLY=false`.

Check the hybrid-table lock:

```bash
docker compose run --rm --no-deps \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/manage_corpus_lock.py --status
```

Open an approved maintenance window:

```bash
docker compose run --rm --no-deps \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/manage_corpus_lock.py --disable --yes
```

Run only the intended write operation, then immediately restore protection:

```bash
docker compose run --rm --no-deps \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/manage_corpus_lock.py --enable
```

The lock protects `knowledge_documents` and the selected physical vector
table against insert, update, delete, and truncate operations. It is defense
in depth: turning off cron does not prevent manual, API, or accidental writes.

## Website ingestion

### What the crawler actually calls

The important handoff in `scripts/crawl_hkpl_site.py` is:

```python
result = ingest_path_sync(...)
```

That call executes the complete pipeline:

```text
crawl_hkpl_site.py
  fetch() and robots.txt check
  → extract_main_html()
  → save_for_ingestion()
  → classify_batch_items_sync()
  → ingestion.service.ingest_path_sync()
  → register_upload()
  → process_registered_document()
  → readers.load_file()
  → chunking.chunk_documents()
  → VectorStoreIndex(..., embed_model=embed_model)
  → PGVectorStore
```

You do not run a separate embedding or database-insert command.

### Crawl one webpage

After opening the corpus lock, replace the example URL:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/crawl_hkpl_site.py \
  --seed-url "https://www.hkpl.gov.hk/en/example-page.html" \
  --max-pages 1 \
  --max-depth 0 \
  --delay-seconds 0
```

`--max-depth 0` prevents link traversal. A new or changed page should report
`indexed: 1`. An identical registered page reports `unchanged: 1` and is not
re-chunked or re-embedded.

### Run the bounded HKPL crawl

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/crawl_hkpl_site.py \
  --max-pages 300 \
  --max-depth 3 \
  --delay-seconds 0.5
```

The crawler:

- accepts only the configured HKPL host and English paths;
- checks HKPL's `robots.txt` before fetching;
- rejects blocked paths and unsupported file extensions;
- discovers links breadth-first;
- retains useful HTML and JSON-LD while removing common navigation elements;
- includes PDF links unless `--exclude-pdfs` is supplied;
- hashes cleaned content and processes only new or changed sources;
- saves source files under `uploads/` for audit and rebuild.

### Re-chunk unchanged saved webpages

Changing `chunking.py` does not change the webpage content hash. The crawler
will therefore report unchanged. Use the rebuild path to apply a new chunker
to already saved sources.

Preflight without changing the corpus:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/ingest_pgvector_llamaindex.py --check-rebuild
```

After a successful preflight and after opening the corpus lock:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/ingest_pgvector_llamaindex.py --rebuild-all
```

`--rebuild-all` deletes and recreates HKPL primary chunks in the selected
table while preserving rows tagged as HotpotQA or Webz News distractors. It
creates new document versions and chunk IDs, so evaluation labels must be
revalidated and often regenerated afterward.

## Local document ingestion

Supported direct formats:

```text
.pdf .docx .pptx .xlsx .xlsm .csv .md .txt
.html .htm .xml .json .jsonl
.jpg .jpeg .png .tif .tiff
```

Convert legacy `.doc`, `.xls`, and `.ppt` files to modern formats first.

Ingest one mounted file:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/ingest_documents.py \
  /app/data/example.pdf \
  --document-type auto
```

Ingest a directory:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/ingest_documents.py /app/data/documents
```

Files are copied into `uploads/`, registered, extracted, chunked, embedded,
and inserted through the same service used by the crawler and admin API.

## Readers and extraction

`src/ingestion/readers.py` dispatches each saved source:

| Input | Extraction behavior |
| --- | --- |
| HTML FAQ | Deterministic FAQ-pair extraction when recognizable |
| Other HTML | Pinned Docling structured extraction |
| PDF | Docling; OCR is available for scanned content |
| DOCX/PPTX/Markdown/text/images | Docling structured parsing |
| CSV/XLSX/XML/JSON/JSONL | Deterministic row/record parsing |

Docling preserves headings, hierarchy, tables, and locators when supported by
the source. OCR uses `eng+chi_tra+chi_sim` by default. OCR quality still depends
on scan resolution and layout.

## Chunking

`src/ingestion/chunking.py` is deterministic; it does not ask the answer LLM to
invent boundaries. The classifier may first select a content type:

```text
faq     keep each question and answer together
record  keep one self-contained record together
prose   use structured headings and leaves
skip    retain source for discovery/audit but create no vectors
```

Chunk behavior:

- exact evidence text and source locator are preserved;
- source title, structure headings, aliases, and record/table headers form the
  searchable text;
- every node receives stable provenance metadata and a content-derived ID;
- structural units at or below 512 embedding tokens remain atomic;
- only an oversized structural leaf is split;
- oversized splits use a 64-token overlap and repeat useful record context;
- FAQ questions, record titles, and table headers are repeated when necessary;
- metadata is excluded from the embedding input because useful context is
  already explicitly included in `search_text`.

There is no standalone chunking command. Trigger chunking through crawler,
local ingestion, reindex, preview, or `--rebuild-all`.

## Embedding and vector insertion

`src/infrastructure/embedding.py` adapts the local llama.cpp embedding endpoint
to LlamaIndex. Embedding is not answer generation and does not use the Qwen
answer prompt.

For each `TextNode`, this call triggers embedding and insertion:

```python
VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)
```

The embedding service receives `search_text` and returns a 1,024-dimensional
vector. `src/infrastructure/vector_store.py` passes the node to LlamaIndex's
`PGVectorStore`, which stores approximately:

```text
node_id    chunk ID
text       searchable chunk text
metadata_  JSON source and provenance metadata
embedding  vector(1024)
```

Do not change the embedding model or dimension without rebuilding every vector
searched by that model.

## Database tables

Open PostgreSQL:

```bash
docker compose exec postgres psql -U postgres -d hkpl_vector_db
```

List tables:

```sql
\dt
```

Core tables commonly present in the current PoC:

| Table | Purpose |
| --- | --- |
| `knowledge_documents` | Source registry, hashes, versions, status, saved filename, chunk count |
| `data_hkpl_knowledge` | Default LlamaIndex chunk/vector table |
| `data_hkpl_knowledge_hybrid` | Alternate hybrid chunk/vector table when created |
| `evaluation_dataset` | Active evaluation questions |
| `evaluation_dataset_100` | Workflow candidate validation table |
| `conversation_history` | Chat history used by the current application |
| `knowledge_corpus_control` | Shared corpus read-only state |
| `prohibited_keywords` | Deterministic safety terms |
| `prohibited_keyword_audit_log` | Safety-rule changes |
| `ingestion_preview_documents` | Non-published preview document records, when installed |
| `ingestion_preview_chunks` | Non-published preview chunks, when installed |

Count corpora in the hybrid table:

```sql
SELECT
    COALESCE(NULLIF(metadata_->>'dataset', ''), 'hkpl') AS dataset,
    COALESCE(NULLIF(metadata_->>'corpus_role', ''), 'primary') AS corpus_role,
    COUNT(*) AS chunks
FROM data_hkpl_knowledge_hybrid
GROUP BY 1, 2
ORDER BY 1, 2;
```

Inspect chunks for one registered document:

```sql
SELECT
    node_id AS chunk_id,
    metadata_->>'document_version' AS version,
    metadata_->>'evidence_text' AS evidence_text,
    metadata_->'locator' AS locator
FROM data_hkpl_knowledge_hybrid
WHERE metadata_->>'kb_document_id' = 'DOCUMENT_ID_HERE'
ORDER BY node_id;
```

Verify that every selected row has an embedding:

```sql
SELECT
    COUNT(*) AS chunks,
    COUNT(embedding) AS chunks_with_embeddings
FROM data_hkpl_knowledge_hybrid
WHERE metadata_->>'kb_document_id' = 'DOCUMENT_ID_HERE';
```

Exit with `\q`.

## Corpus audit

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/ingest_pgvector_llamaindex.py --audit-chunks
```

The audit checks embeddings, dimensions, empty evidence, provenance metadata,
locator collisions, duplicate lineage, and token limits. Resolve audit failures
before generating or running an official evaluation dataset.

## Distractor corpora

HKPL primary chunks and benchmark distractors can share one vector table. The
metadata fields distinguish them:

```text
dataset=hkpl       corpus_role=primary
dataset=hotpotqa   corpus_role=distractor
dataset=webz_news  corpus_role=distractor
```

Load HotpotQA distractors during an approved write window:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/hotpotqa_benchmark.py prepare --limit 1000
```

List and load Webz News archives only after reviewing their terms:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/webz_news_benchmark.py list --limit 20
```

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  python scripts/webz_news_benchmark.py prepare \
  --archive latest \
  --limit 1000 \
  --accept-terms
```

For reproducible experiments, use an exact archive filename instead of
`latest`.

## Retrieval and answer generation

The current evaluation retrieval path is:

```text
Question
→ Qwen query embedding
→ pgvector cosine-distance search, top 10
→ Qwen cross-encoder reranking, top 5
→ context packing
→ Qwen3.5 answer generation
```

Cosine distance is converted to similarity by the vector-store integration;
higher similarity indicates a smaller angle between query and chunk vectors.
The reranker examines query/document pairs and can change the vector-search
order. `build_context` concatenates selected reranked evidence; it does not use
an LLM to write new evidence.

## Prompt ownership

The terminal commands in this README are not LLM prompts. Runtime prompts are
owned by code and should be versioned, reviewed, and evaluated rather than
manually pasted into containers.

| Stage | Code | Prompt behavior |
| --- | --- | --- |
| Crawler batch classification | `src/ingestion/classification.py` | Sends title, file type, and bounded source text; returns one schema-constrained `faq`, `record`, `prose`, or `skip` decision per source |
| Readers | `src/ingestion/readers.py` | No generative prompt for deterministic formats; Docling/OCR extract source content |
| Chunking | `src/ingestion/chunking.py` | No LLM prompt; deterministic structure and tokenizer rules |
| Embedding | `src/infrastructure/embedding.py` | No generative prompt; sends chunk/search text to the embedding endpoint |
| Evaluation dataset generation | `scripts/generate_evaluation_dataset.py` | Requests one grounded question, complete answer, exact evidence snippets, and parallel chunk IDs from anchor and sibling evidence |
| Live answer | `src/nodes.py` | Requires evidence-only, concise, final answers with no exposed analysis |
| Evaluation answer | `scripts/evaluate_rag.py` | Uses the retrieved context and general constraint/specificity rules; no question-specific expected answer is included |
| Correctness judge | `scripts/evaluate_rag.py` | Compares generated answer with accepted reference answers on a 1–5 scale |
| Faithfulness judge | `scripts/evaluate_rag.py` | Checks whether every answer claim is supported by combined context |
| Relevancy judge | `scripts/evaluate_rag.py` | Checks whether the response directly answers every requested item |

Important distinction:

- an LLM may classify source type and generate evaluation questions;
- the LLM does not decide raw chunk boundaries;
- the embedding model creates vectors but does not generate answers;
- Phoenix records calls and scores but does not retrieve or generate answers.

## Generate an evaluation dataset

Evaluation questions must be generated only after the corpus is finalized and
audited. Candidate rows require human semantic review even when automated
validation passes.

### Select the candidate source

`scripts/generate_evaluation_dataset.py` reads normal embedded candidates from
the physical `data_<VECTOR_TABLE>` table. For example,
`VECTOR_TABLE=hkpl_knowledge_hybrid` selects
`data_hkpl_knowledge_hybrid`. Passing `--preview-run-id` instead selects one
explicit non-embedded ingestion preview and requires a separate output file.

Hybrid retrieval requires a vector table created with its generated full-text
column. Use a fresh `VECTOR_TABLE` and reingest when enabling hybrid retrieval;
an older dense-only table is rejected rather than partially upgraded in place.

### Check corpus status

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py status
```

### Generate exactly 100 candidate questions

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py prepare-candidate \
  --output data/evaluation_dataset_100.csv \
  --target-questions 100 \
  --all-chunks
```

Generation behavior:

- only `dataset=hkpl`, `corpus_role=primary` chunks are eligible;
- chunks shorter than 120 characters are excluded;
- without `--all-chunks`, at most eight chunks per document are anchors;
- only the first 1,800 characters of a chunk are sent to generation;
- sibling chunks from the same document are available for complete evidence;
- accepted evidence chunk IDs are consumed and cannot become later anchors;
- invalid, ambiguous, unsupported, or duplicate candidates do not count;
- progress is checkpointed after processed anchors;
- one accepted question is produced per anchor at most.

Resume the same output and options after interruption:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py prepare-candidate \
  --output data/evaluation_dataset_100.csv \
  --target-questions 100 \
  --all-chunks \
  --resume
```

The workflow loads candidate rows into `evaluation_dataset_100`. Review every
row manually, especially dates, recurring events, branches, venues, lists,
multi-chunk answers, and time-sensitive facts.

### Evaluation row schema

```text
domain
query
expected_answer_text
expected_context_snippet
expected_context_snippets_json
accepted_answers_json
source_title
source_url
source_document_id
source_chunk_id
source_chunk_ids_json
```

`expected_context_snippets_json` and `source_chunk_ids_json` must be parallel
arrays. Item `n` in the snippet array must come from chunk ID `n` in the chunk
ID array. The singular fields identify primary evidence for compatibility.

Validate an existing candidate without regenerating it:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py validate-candidate \
  --candidate data/evaluation_dataset_100.csv
```

Only promote after manual review:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py promote \
  --candidate data/evaluation_dataset_100.csv \
  --active data/evaluation_dataset.csv \
  --yes
```

## Validate a selected evaluation table

Example for the 128-row table linked to the hybrid corpus:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  langgraph-agent \
  python scripts/validate_evaluation_dataset.py
```

An official run requires:

```text
Expected chunk found: all rows
Snippet text found: all rows
Evaluation evidence status: READY
```

`Answer verbatim found` is informational. A complete answer may paraphrase or
combine facts. A missing chunk or missing expected snippet blocks readiness.

## Run Phoenix evaluation

Phoenix must be running:

```bash
docker compose up -d phoenix postgres embedding reranker llm
docker compose ps
```

Open Phoenix at:

```text
http://SERVER_HOST:6006
```

For a diagnostic run before distractor corpora are loaded, evaluate the HKPL
corpus directly:

```bash
docker compose run --rm langgraph-agent \
  python scripts/evaluate_rag.py \
  --allow-missing-distractors
```

Official benchmark runs should use the workflow command below so corpus and
evaluation-reference validation run first.

### Full 128-row hybrid evaluation

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  -e RAG_EVALUATION_RESULTS_PATH=/app/data/rag_evaluation/results_hkpl_128.csv \
  -e RAG_EVALUATION_SUMMARY_PATH=/app/data/rag_evaluation/summary_hkpl_128.json \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py evaluate \
  --phoenix-project hkpl-rag-evaluation-128
```

The workflow first checks that HKPL, HotpotQA, and Webz News exist, then
validates every evaluation reference. It does not create question traces when
the dataset status is `REVIEW REQUIRED`.

### Three-question smoke test

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py evaluate \
  --limit 3 \
  --phoenix-project hkpl-rag-smoke-3
```

### Debug one exact question

The workflow wrapper does not expose `--question-exact`, so call the low-level
runner directly:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  -e RAG_EVALUATION_RESULTS_PATH=/app/data/rag_evaluation/debug_question.csv \
  -e RAG_EVALUATION_SUMMARY_PATH=/app/data/rag_evaluation/debug_question.json \
  langgraph-agent \
  python scripts/evaluate_rag.py \
  --question-exact 'PASTE THE EXACT DATABASE QUESTION HERE' \
  --phoenix-project hkpl-rag-debug-question
```

Filtered runs add tags to the requested result filename; inspect the filename
printed at the end of the run.

### Optional bounded answer reasoning

Normal answer and evaluator reasoning are off. To test answer reasoning only:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_128 \
  langgraph-agent \
  python scripts/evaluate_rag.py \
  --question-exact 'PASTE THE EXACT DATABASE QUESTION HERE' \
  --answer-reasoning \
  --reasoning-budget 500 \
  --phoenix-project hkpl-rag-debug-reasoning
```

llama.cpp counts reasoning and visible answer tokens within the completion
budget. The runner therefore requests the reasoning allowance in addition to
the 512-token answer allowance. `thinking_budget_tokens` is a maximum, not a
requirement.

Do not enable evaluator reasoning when comparing an answer-reasoning run with
a no-reasoning baseline; changing generator and judge behavior together makes
the cause of score changes unclear.

## Evaluation metrics and diagnosis

| Metric | Meaning |
| --- | --- |
| Retriever Hit@10 | Exact labelled chunk or recognized equivalent evidence appeared among vector candidates |
| Reranker Hit@5 | Labelled/equivalent evidence remained among final contexts |
| Distractor Rate@5 | Fraction of final contexts from configured distractor corpora |
| Correctness | Judge comparison against accepted reference answers, 1–5 |
| Faithfulness | Whether every answer claim is supported by supplied context |
| Relevancy | Whether the answer directly addresses every requested item |

Diagnosis logic:

```text
Evidence absent from retrieval + failed answer → retrieval_problem
Evidence retrieved but removed by reranking → reranker_problem
Evidence reranked but absent from packed context → context_building_problem
Evidence reaches LLM but core answer is wrong → llm_generation_problem
Answer includes unsupported claims → ungrounded_answer
Answer fails to address the question → irrelevant_answer
All required checks pass → working_correctly
```

Faithfulness and Retriever Hit measure different things. An answer can be
faithful to retrieved context while the benchmark's expected evidence was not
retrieved.

## Token accounting

Per-row result fields include:

```text
retriever_query_tokens
reranker_input_tokens
context_tokens
prompt_tokens
completion_tokens
reasoning_tokens
llm_total_tokens
pipeline_total_tokens
```

```text
llm_total_tokens = prompt_tokens + completion_tokens

pipeline_total_tokens = retriever query tokens
                      + reranker input tokens
                      + answer-model total tokens
```

Reasoning tokens are part of completion tokens even when they are returned in
a separate hidden field. Evaluator-judge tokens are not currently included in
`pipeline_total_tokens`; inspect the evaluator child spans in Phoenix for the
complete benchmark compute.

## API operations

Test the streaming chat endpoint:

```bash
curl -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input_string": "Where can I read e-books?",
    "session_id": "local-test-session"
  }'
```

Admin upload and reindex endpoints require `X-Admin-Key` and an intentionally
writable maintenance window. Do not expose the development API publicly.

## Automation

Docker Compose does not run the crawler periodically. A user-owned cron entry
can run without sudo when that user has Docker permission. Example four-hour
schedule:

```cron
0 */4 * * * cd /absolute/path/to/hkpl_multiformat_patch && docker compose run --rm -e VECTOR_TABLE=hkpl_knowledge_hybrid -e KNOWLEDGE_CORPUS_READ_ONLY=false langgraph-agent python scripts/crawl_hkpl_site.py >> logs/hkpl_crawler.log 2>&1
```

Do not schedule this while the database corpus lock is enabled. For a frozen
evaluation corpus, remove the cron entry and keep both write protections on.

## Git and generated files

Before changing or pulling code:

```bash
git status
git branch --show-current
git fetch origin
```

Do not commit:

- `.env` or credentials;
- crawler logs;
- Python caches;
- downloaded model artifacts;
- generated evaluation result files unless explicitly approved as release
  artifacts.

Container-created files may be owned by root on Ubuntu. If necessary, repair
only the intended bind-mounted directory through a root container; do not
recursively change ownership of an unknown or broad host path.

## Troubleshooting

### `Knowledge corpus is frozen`

Both the database lock and `KNOWLEDGE_CORPUS_READ_ONLY` guard are working. Open
an approved maintenance window, run one intended write, then restore the lock.

### Crawler says `unchanged`

The source hash has not changed. Use `--rebuild-all` when the chunker changed;
do not expect an unchanged crawl to create new chunks.

### Evaluation says `REVIEW REQUIRED`

At least one expected chunk is missing or its expected snippet does not match
stored evidence. Repair the dataset/CSV source of truth and revalidate. Do not
use `--allow-incomplete-dataset` for official reporting.

### `&` versus `&amp;`

HTML evidence can preserve encoded entities. Keep evaluation snippets aligned
with stored evidence or implement a reviewed normalization rule; do not edit
only the PostgreSQL copy if a CSV synchronization will restore the mismatch.

### `expected_context_snippets_json` array error

`expected_context_snippets_json` and `source_chunk_ids_json` must have equal
lengths and matching positions.

### CUDA out of memory

Run `nvidia-smi`. The embedding, reranker, LLM, and safety guard share the two
PoC GPUs. Stop unnecessary GPU processes or choose a device with sufficient
memory; the safety guard intentionally does not silently fall back to CPU.

### `.env` mount is not a file

Verify that the host path exists and is a regular file:

```bash
ls -ld .env
file .env
```

If `.env` is accidentally a directory, move it aside and recreate it from
`.env.example`.

## Safety, governance, and current limitations

The current repository is a PoC, not an approved production deployment.
Before public release, address at least these controls:

- rotate and purge any credential that ever entered Git history;
- enforce public/active/approved/effective-date authorization in the database
  before retrieval ranking;
- issue server-owned sessions and bound history by expiry and ownership;
- fail closed on every safety dependency error or malformed result;
- separate system policy, patron input, and untrusted retrieved evidence using
  real message roles and strict schemas;
- harden URL ingestion against SSRF, redirect abuse, private addresses,
  oversized bodies, and unapproved content types;
- require configured identity and role-based authorization for admin routes;
- remove raw patron/source content from routine production telemetry;
- move schema ownership to reproducible migrations;
- publish source versions atomically only after validation and approval;
- evaluate the real public workflow across multilingual, safety, authorization,
  stale-content, citation, and no-answer slices;
- run repeatable concurrency and overload tests before claiming the 100-request
  capacity target;
- pin container/model artifacts by immutable revision and keep internal
  services off public host interfaces;
- replace in-process background ingestion with durable worker jobs for
  production.

Evaluation questions generated by an LLM always require human review. Event
dates, hours, fees, policies, and service availability are time-sensitive. A
frozen benchmark evaluates a frozen corpus snapshot, not necessarily the live
website.

## File-by-file reference

### Runtime and infrastructure

| File | Responsibility |
| --- | --- |
| `docker-compose.yml` | Starts PostgreSQL, model servers, FastAPI, and Phoenix; does not schedule crawling |
| `Dockerfile.agent` | Builds the Python/Tesseract agent image |
| `.env.example` | Documents required secrets and Compose substitutions |
| `postgres-init/init.sql` | Enables pgvector and initializes ordinary tables on a clean volume |
| `src/infrastructure/db.py` | SQLAlchemy database connection |
| `src/infrastructure/embedding.py` | LlamaIndex adapter for the embedding endpoint |
| `src/infrastructure/vector_store.py` | PGVectorStore configuration plus full-text and trigram indexes |

### Ingestion

| File | Responsibility |
| --- | --- |
| `scripts/crawl_hkpl_site.py` | Robots-aware crawl, discovery, cleaning, change detection, ingestion handoff |
| `scripts/ingest_documents.py` | Local file/directory ingestion |
| `scripts/ingest_pgvector_llamaindex.py` | FAQ ingestion, rebuild, evaluation sync, and chunk audit |
| `scripts/manage_corpus_lock.py` | Database-level corpus freeze/unfreeze |
| `src/ingestion/service.py` | Registration, extraction, chunking, embedding, insertion, replacement |
| `src/ingestion/registry.py` | Document identity, hash, version, status, and count |
| `src/ingestion/readers.py` | Deterministic and Docling extraction |
| `src/ingestion/classification.py` | Bounded batch document-type classification |
| `src/ingestion/document_types.py` | Record kind and chunk-policy selection |
| `src/ingestion/chunking.py` | Structure-aware, tokenizer-bounded nodes and IDs |
| `src/ingestion/write_guard.py` | Application-level corpus write protection |
| `src/ingestion/webpage.py` | Single admin-URL acquisition path |

### RAG and application

| File | Responsibility |
| --- | --- |
| `main.py` | FastAPI chat/admin routes |
| `src/graph.py` | LangGraph workflow edges and routes |
| `src/state.py` | Typed graph state |
| `src/nodes.py` | Safety, query processing, context, answer, citations, memory |
| `src/retrieval.py` | Dense/lexical retrieval, rank fusion, reranking, and diagnostics |
| `src/llm_client.py` | Local answer-model client and usage normalization |
| `src/corpus.py` | Primary/distractor metadata maintenance |
| `src/memory.py` | Conversation history |
| `src/compliance.py` | Prohibited-keyword rules |

### Evaluation and observability

| File | Responsibility |
| --- | --- |
| `scripts/generate_evaluation_dataset.py` | Candidate question generation from stored/preview chunks |
| `scripts/validate_evaluation_dataset.py` | Schema and evidence-reference validation |
| `scripts/evaluate_rag.py` | Retrieval, reranking, answer, judge metrics, diagnoses, traces |
| `scripts/rag_benchmark_workflow.py` | Guarded status/audit/generate/validate/promote/evaluate workflow |
| `scripts/normalize_evaluation_schema.py` | Evaluation CSV schema normalization |
| `scripts/migrate_evaluation_benchmark.py` | Evidence-label migration after corpus changes |
| `scripts/hotpotqa_benchmark.py` | HotpotQA distractor preparation |
| `scripts/webz_news_benchmark.py` | Webz News distractor preparation |
| `src/observability.py` | Phoenix/OpenTelemetry setup |
| `src/tracing_helpers.py` | Trace I/O and token attributes |
| `src/phoenix_annotations.py` | Evidence and answer annotations |
| `src/token_counting.py` | Token counting through model tokenizer endpoints |
