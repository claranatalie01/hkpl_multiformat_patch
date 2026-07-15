# HKPL Multi-Format Ingestion Change Pack

This directory contains replacement and new files for the current
`/home/cnatalie/agentic-RAG` project.

## What this changes

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

## New project structure

```text
src/
├── infrastructure/
│   ├── __init__.py
│   ├── db.py
│   ├── embedding.py
│   └── vector_store.py
├── ingestion/
│   ├── __init__.py
│   ├── readers.py
│   ├── chunking.py
│   ├── registry.py
│   └── service.py
├── graph.py
├── memory.py
├── nodes.py
├── retrieval.py
└── state.py
```

## Installation

Back up the current project first:

```bash
cd /home/cnatalie
cp -a agentic-RAG agentic-RAG-backup-before-multiformat
```

Copy the files in this patch over the project, preserving their paths.

Create persistent directories:

```bash
cd /home/cnatalie/agentic-RAG
mkdir -p uploads storage
```

Create or update `.env`:

```bash
cp .env.example .env
```

Set a non-empty admin key in `.env` before exposing the service.

Rebuild the agent because system and Python dependencies changed:

```bash
docker compose down
docker compose build --no-cache langgraph-agent
docker compose up -d
docker compose logs -f langgraph-agent
```

The registry table is created automatically when FastAPI starts. The updated
`postgres-init/init.sql` is also included for clean installations.

## Re-ingest the existing FAQ data

The retriever now uses `data_hkpl_knowledge`, so ingest the FAQ data into that
new collection:

```bash
docker compose run --rm langgraph-agent \
  python scripts/ingest_pgvector_llamaindex.py
```

`--rebuild-all` rebuilds registered HKPL sources while preserving rows tagged
as the HotpotQA benchmark corpus.

## HotpotQA benchmark in the shared vector table

HotpotQA and HKPL chunks coexist in `data_hkpl_knowledge`. HotpotQA rows are
identified by `metadata_->>'dataset' = 'hotpotqa'`; HKPL rows retain their
existing document metadata.

Download a deterministic 1,000-question validation subset, create one vector
chunk per unique paragraph, and replace only previous HotpotQA rows:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/hotpotqa_benchmark.py prepare --limit 1000
```

This deterministic subset creates 9,769 unique HotpotQA paragraph vectors.
Embedding them can take several minutes. Re-running the command is safe: it
replaces HotpotQA vectors and leaves HKPL vectors untouched.

Evaluate retrieval and reranking across the complete combined vector table:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/hotpotqa_benchmark.py evaluate --limit 100
```

Include answer generation and official-style exact-match/token-F1 metrics:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/hotpotqa_benchmark.py evaluate \
  --limit 100 --answers
```

Add the LlamaIndex correctness, faithfulness, and relevancy judges when a
slower, more expensive full evaluation is needed:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/hotpotqa_benchmark.py evaluate \
  --limit 100 --answers --llama-evaluators
```

Phoenix displays these runs as `HotpotQA RAG Query` traces in the existing
`hkpl-rag` project. Results are also written to
`data/hotpotqa/results.csv` and `data/hotpotqa/summary.json`.

### Combined HKPL and HotpotQA report

Both benchmarks use the same Phoenix project and carry an `eval.dataset`
attribute (`hkpl`, `hotpotqa`, or `combined`). This keeps traces together for
one deployed RAG pipeline while allowing dataset-specific filtering.

After running HKPL retrieval evaluation, HKPL answer evaluation, and HotpotQA
answer evaluation, generate the combined report:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/report_rag_evaluation.py
```

The report is written to `data/combined_evaluation_summary.json` and exported
to Phoenix as `Combined RAG Evaluation Summary`. It preserves the complete
per-dataset summaries and reports:

- macro averages, where HKPL and HotpotQA have equal weight;
- question-weighted averages, where each evaluated question has equal weight.

The normalized macro answer-quality score averages HKPL correctness divided by
5 with HotpotQA token F1. Dataset-specific scores remain the primary results
because the two benchmarks use different labels and answer metrics.

Verify both corpora are in the same physical vector table:

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
docker compose run --rm langgraph-agent \
  python scripts/ingest_documents.py /app/data/sample.pdf
```

To ingest a mounted directory:

```bash
docker compose run --rm langgraph-agent \
  python scripts/ingest_documents.py /app/data/documents
```

## Chunking behaviour

- FAQ, CSV, Excel, XML, JSON, and JSONL records use an atomic strategy.
- PDF, DOCX, PPTX, Markdown, TXT, HTML, and OCR text use overlapping prose
  chunks.
- Default prose chunk size: 512 tokens.
- Default overlap: 64 tokens.
- Large atomic records can still split at 2048 tokens.

Tune through `.env`, not source code.

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

- PDF extraction uses PyMuPDF plus page OCR fallback. It is not a complete
  layout understanding system for highly complex multi-column documents,
  charts, or merged tables.
- OCR quality depends on scan quality and installed languages.
- FastAPI BackgroundTasks is suitable for this prototype, but a production
  deployment should use a durable worker queue.
- The upload signature checks are a baseline, not malware scanning.
- If `ADMIN_API_KEY` is empty, admin endpoints are open for local development.
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
