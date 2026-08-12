# HKPL Agentic RAG Implementation Guide

For the current corpus inventory, evaluation-dataset data dictionary, metric
definitions, benchmark results, and reproducibility controls, see
[`DATA_DOCUMENTATION.md`](DATA_DOCUMENTATION.md).

This guide describes the current repository. Run commands from the actual
project directory on the remote server; do not assume a fixed checkout path.

## Implemented capabilities

1. Renames the general vector collection from `hkpl_faq` to
   `hkpl_knowledge`.
2. Keeps FAQ question-answer pairs as atomic chunks.
3. Adds ingestion for:
   - PDF, including OCR fallback for scanned pages
   - DOCX
   - PPTX
   - XLSX and XLSM
   - CSV
   - Markdown and TXT
   - HTML
   - XML
   - JSON and JSONL crawler output
   - JPG, JPEG, PNG, TIFF images through OCR
4. Adds structure-aware chunking.
5. Adds a PostgreSQL document registry.
6. Adds upload, status, replacement, listing, and deletion APIs.
7. Adds page, section, slide, sheet, and row citations.
8. Replaces unconditional vector-store clearing with document-level updates.
9. Fixes the greeting route so it does not require RAG faithfulness.
10. Makes SSE multiline answers standards-compliant.
11. Makes reranking explicit and maps scores using the returned document index.
12. Preserves complete chunks instead of cutting context in the middle.

## Project structure

```text
src/
├── infrastructure/       # database, embedding client, PGVectorStore
├── ingestion/            # readers, chunking, registry, service, write guard
├── compliance.py
├── corpus.py
├── graph.py
├── llm_client.py
├── memory.py
├── nodes.py
├── observability.py
├── phoenix_annotations.py
├── retrieval.py
├── state.py
├── token_counting.py
└── tracing_helpers.py

scripts/
├── evaluate_rag.py
├── rag_benchmark_workflow.py
├── manage_corpus_lock.py
├── ingest_pgvector_llamaindex.py
├── validate_evaluation_dataset.py
└── other ingestion, migration, and benchmark utilities
```

## Installation

Back up the current remote checkout and data before a major migration:

```bash
cd /path/to/parent
cp -a hkpl_multiformat_patch hkpl_multiformat_patch-backup
```

Create persistent directories:

```bash
cd /path/to/hkpl_multiformat_patch
mkdir -p uploads storage
```

Create or update `.env`:

```bash
cp .env.example .env
```

Set a non-empty admin key in `.env` before exposing the service.
Only `DB_PASSWORD`, `ADMIN_API_KEY`, `PHOENIX_PROJECT_NAME`, and `HF_TOKEN` are
interpolated from `.env` by the current Compose file. Other runtime defaults are
literal entries in `docker-compose.yml`.

Rebuild the agent because system and Python dependencies changed:

```bash
docker compose down
docker compose build --no-cache langgraph-agent
docker compose up -d
docker compose logs -f langgraph-agent
```

The registry table is created automatically when FastAPI starts.
`postgres-init/init.sql` is used for clean database installations.

## Corpus freeze and controlled writes

The checked-in deployment is intentionally read-only:

```text
KNOWLEDGE_CORPUS_READ_ONLY=true
```

The application guard blocks corpus-writing scripts, and document mutation
endpoints return HTTP 423. A separate PostgreSQL trigger lock may also protect
`knowledge_documents` and `data_hkpl_knowledge`.

Check the database lock:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --status
```

For an approved CLI maintenance window, disable the database lock, explicitly
override the application guard for each write command, and immediately
re-enable the database lock afterward:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --disable --yes

docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python THE_WRITE_SCRIPT.py

docker compose run --rm langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --enable
```

Do not leave the database unlocked. To use the upload, replace, reindex,
delete, index-URL, or crawler paths, an operator must also run the agent with
`KNOWLEDGE_CORPUS_READ_ONLY=false`; the checked-in long-running service does
not permit those mutations. All corpus-write commands below assume the
database lock has first been disabled and must be followed by `--enable`.

## Re-ingest the existing FAQ data

The retriever now uses `data_hkpl_knowledge`, so ingest the FAQ data into that
new collection:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python scripts/ingest_pgvector_llamaindex.py
```

`--rebuild-all` rebuilds registered HKPL sources while preserving rows tagged
as distractor corpora.

## HotpotQA benchmark in the shared vector table

HotpotQA and HKPL chunks coexist in `data_hkpl_knowledge`. HotpotQA rows are
identified by `metadata_->>'dataset' = 'hotpotqa'`; HKPL rows retain their
existing document metadata.

Load the first deterministic 1,000 examples from the official
`hotpotqa/hotpot_qa` distractor validation split with Hugging Face
`datasets.load_dataset(..., streaming=True)`, create one vector per unique
paragraph, and replace only previous HotpotQA vectors:

The dataset is public and does not require authentication. Optionally set
`HF_TOKEN` in `.env` to use authenticated Hugging Face rate limits.

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python scripts/hotpotqa_benchmark.py prepare --limit 1000
```

This deterministic subset creates 9,769 unique HotpotQA paragraph vectors.
Embedding them can take several minutes. Re-running the command is safe: it
replaces HotpotQA vectors and leaves HKPL vectors untouched.

## Webz.io news distractor corpus

The news loader reads the public `Webhose/free-news-datasets` repository. Each
JSON article remains a separate document. Its body is split at sentence
boundaries into 512-token chunks with 64-token overlap, and the article title
is repeated in every chunk. Articles are never joined together.

Review the repository terms before ingestion. List available weekly archives:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/webz_news_benchmark.py list --limit 20
```

Ingest the latest archive, up to 1,000 articles:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python scripts/webz_news_benchmark.py prepare \
  --archive latest --limit 1000 --accept-terms
```

For a reproducible experiment, replace `latest` with an exact filename from
the list command. The ZIP is cached under `data/webz_news/`. Re-running the
command replaces only rows tagged `dataset=webz_news`; HKPL and HotpotQA rows
remain untouched.

Repeat `--archive ARCHIVE_NAME.zip` to combine several themes. `--limit` is the
total article limit across all selected archives; avoid ingesting the entire
repository unless that scale is part of the experiment.

### Corpus-first benchmark workflow

Use the workflow command to enforce the required order: finalize and audit the
HKPL corpus, generate a candidate benchmark from HKPL chunks, review it,
promote it, then evaluate against the combined corpus.

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py status

docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py audit-corpus

docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py prepare-candidate
```

The candidate is written to `data/evaluation_dataset.candidate.csv` and loaded
into `evaluation_dataset_candidate`. Review all ambiguous, multi-part, and
time-sensitive labels before promotion.

If generation completed but candidate import or evidence validation failed,
continue from the saved CSV without regenerating questions:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py validate-candidate
```

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py promote --yes

docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py evaluate \
  --phoenix-project hkpl-rag
```

`prepare-candidate` uses up to eight chunks per document by default. Add
`--all-chunks` to create a comprehensive candidate pool from every eligible
HKPL primary chunk; this can take substantially longer and still requires
semantic label review. Candidate generation never uses HotpotQA or Webz News
chunks. Evaluation requires both distractor corpora to be present.

Generation checkpoints the candidate after every completed chunk. Resume an
interrupted run with the same generation options plus `--resume`:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py prepare-candidate --resume
```

### HKPL evaluation with combined distractor noise

`scripts/evaluate_rag.py` is the low-level evaluation runner;
`scripts/rag_benchmark_workflow.py evaluate` is the guarded workflow wrapper.
The evaluator loads HKPL
questions, expected answers, and expected chunks from `evaluation_dataset`,
then searches the combined vector table containing HKPL, HotpotQA paragraphs,
and Webz.io news chunks. The external corpora contribute retrieval noise only;
their questions and answers are not evaluation labels.

Run a short smoke test with five HKPL questions:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/evaluate_rag.py --limit 5
```

Even a limited or single-question run validates the full active benchmark and
requires both configured distractor corpora. The
`--allow-incomplete-dataset` and `--allow-missing-distractors` flags exist only
for targeted diagnostics, not benchmark reporting.

Run the complete evaluation:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/evaluate_rag.py
```

Run one matching question when investigating a regression:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/evaluate_rag.py \
  --question-contains "Library Catalogue"
```

Every question reports retrieval and reranker Hit, Recall, Complete,
evidence-hit, and distractor metrics at 1/3/5/10, plus MRR, LlamaIndex
correctness, faithfulness, relevancy, hallucination derived from faithfulness,
token usage, latency, answer/evidence outcomes, and a stage-specific RAG
diagnosis. The primary robustness gates are equivalent-evidence-aware
Retriever Hit@10 and Reranker Hit@5.

Phoenix displays all runs in the `hkpl-rag` project as `RAG Evaluation Query`
traces with `eval.dataset=hkpl`. The aggregate is exported as the HKPL RAG
evaluation summary.

Results are written to `data/rag_evaluation/results.csv` and
`data/rag_evaluation/summary.json`. Detailed aggregate diagnostics are written
to `data/rag_evaluation/summary.diagnostics.json`. Filtered or reasoning runs
use tagged filenames so they do not overwrite the normal full-run outputs.

Verify all corpora are in the same physical vector table:

```sql
SELECT
    COALESCE(metadata_->>'dataset', 'hkpl') AS corpus,
    COUNT(*) AS chunks
FROM data_hkpl_knowledge
GROUP BY corpus
ORDER BY corpus;
```

## Test the chat endpoint

```bash
curl -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input_string": "Where can I read e-books?",
    "session_id": "multiformat-test-001"
  }'
```

## Upload a document

The checked-in service is read-only, so this endpoint returns HTTP 423 unless
an operator deliberately opens the application and database maintenance locks
described above.

```bash
curl -X POST http://localhost:8001/admin/documents/upload \
  -H "X-Admin-Key: change-this-before-production" \
  -F "file=@/absolute/path/to/borrowing_rules.pdf" \
  -F "source_title=HKPL Borrowing Rules" \
  -F "access_level=public"
```

The response returns a `document_id` and status `uploaded`.

Check processing status:

```bash
curl http://localhost:8001/admin/documents/DOCUMENT_ID \
  -H "X-Admin-Key: change-this-before-production"
```

Expected status progression:

```text
uploaded → extracting → chunking → embedding → completed
```

List documents:

```bash
curl http://localhost:8001/admin/documents \
  -H "X-Admin-Key: change-this-before-production"
```

Replace a document:

```bash
curl -X POST \
  http://localhost:8001/admin/documents/DOCUMENT_ID/replace \
  -H "X-Admin-Key: change-this-before-production" \
  -F "file=@/absolute/path/to/new_borrowing_rules.pdf" \
  -F "source_title=HKPL Borrowing Rules" \
  -F "access_level=public"
```

Delete a document and its vector chunks:

```bash
curl -X DELETE \
  http://localhost:8001/admin/documents/DOCUMENT_ID \
  -H "X-Admin-Key: change-this-before-production"
```

## Command-line ingestion

Files passed through the CLI are copied into `/app/uploads` and registered:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python scripts/ingest_documents.py /app/data/sample.pdf --document-type prose
```

To ingest a mounted directory:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python scripts/ingest_documents.py /app/data/documents
```

## Chunking behaviour

- A librarian labels an individual source as `faq`, `record`, or `prose`.
  An unlabelled individual source uses the bounded 512-token/64-token-overlap
  fallback and does not call the classifier.
- Directory and crawler batches use the existing generation model once per
  mini-batch of at most 20 sources. The schema-constrained result must contain
  exactly one decision for every input ID or the batch fails without fallback.
- The batch-only `skip` label marks listing, index, and navigation pages as
  discovery-only. Their immutable source files are retained for rebuilds and
  audits, but they produce zero searchable chunks. Physical tables bypass the
  model and use deterministic rows.
- `faq` keeps each question with its answer; `record` keeps one notice, event,
  or branch profile together; `prose` uses Docling headings and hierarchy.
- CSV, Excel, JSON, JSONL, and XML use deterministic record parsing with typed
  locators. PDF, DOCX, PPTX, HTML, Markdown, text, and images use Docling.
- The Qwen3 embedding tokenizer enforces a 512-token hard cap including source
  context. Clean structural boundaries have no overlap; only an oversized leaf
  uses 64-token overlap and repeats its FAQ question, record title, or table
  header when applicable.

The classifier prompt uses only title, file type, and the first 1,200 characters
of each source, with temperature zero and reasoning disabled. Its labels are:

```text
faq    actual question-answer pairs
record one self-contained notice, event detail, or branch profile
prose  policies, guidance, articles, and other useful narrative content
skip   listing/index/navigation content whose main value is links
```

Domain categories remain separate metadata; they do not change chunking.

These defaults are set in `docker-compose.yml`. Production requires the pinned
Qwen tokenizer under `models/qwen3-embedding` and pre-fetched Docling artifacts
under `models/docling`; ingestion does not download models at runtime.

## Supported and unsupported formats

Direct support:

```text
.pdf .docx .pptx .xlsx .xlsm .csv .md .txt
.html .htm .xml .json .jsonl
.jpg .jpeg .png .tif .tiff
```

Legacy `.doc`, `.xls`, and `.ppt` files must first be converted to the modern
Office formats. The misspelled extensions `.docs` and `.xlsv` are not real
standard Office formats.

## Important limitations

- Docling preserves reading order, headings, tables, and page provenance, but
  extraction quality still depends on the source document and local artifacts.
- OCR quality depends on scan quality and installed languages.
- FastAPI BackgroundTasks is suitable for this prototype, but a production
  deployment should use a durable worker queue.
- The upload signature checks are a baseline, not malware scanning.
- If `ADMIN_API_KEY` is empty, admin endpoints are open for local development.
- Corpus mutation endpoints are disabled by the checked-in read-only guard and
  may also be blocked by the PostgreSQL corpus lock.
- Access level is stored, but retrieval-time authorization filtering is a
  later phase.
- Coordinate-based nearest-library resolution remains a placeholder because
  no branch-coordinate dataset was provided.

## Evaluation after implementation

Run the existing FAQ evaluation first. Then create a multi-format evaluation
set with at least:

- two native PDFs;
- one scanned PDF;
- one DOCX with headings and a table;
- one XLSX with multiple sheets;
- one HTML page;
- one Traditional Chinese document;
- one image containing text.

For each source, record the expected document ID, page/sheet/row, and answer.
Measure extraction success, Recall@1, Recall@3, MRR, answer correctness,
faithfulness, citation accuracy, and latency.
