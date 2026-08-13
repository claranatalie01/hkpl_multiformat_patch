# HKPL ingestion runbook

This runbook is written for the current server deployment and repository:

```text
Checkout                 ~/hkpl_multiformat_patch
Feature branch           feat/rag-ingestion-work
Incoming files           ~/hkpl_multiformat_patch/data/incoming
Preview logs and exports ~/hkpl_multiformat_patch/data/ingestion_preview
Docling JSON cache       ~/hkpl_multiformat_patch/storage/docling
Docling models           ~/hkpl-models/docling
Qwen tokenizer           ~/hkpl-models/qwen3-embedding
PostgreSQL container     hkpl_postgres
Database                 hkpl_vector_db
Database user            postgres
```

Large model artifacts stay outside Git. Input files, preview output, and
Docling JSON use existing repository-mounted directories.

There are two distinct operations:

1. **Pre-embedding preview** classifies and chunks already registered sources
   into `ingestion_preview_documents` and `ingestion_preview_chunks`.
2. **Live ingestion** acquires, classifies, chunks, embeds, and writes the live
   corpus. It remains blocked while the corpus read-only lock is enabled.

The preview does not modify `knowledge_documents` or
`data_hkpl_knowledge` and creates no embeddings.

## Pre-embedding preview

### 1. Update the checkout and agent image

```bash
cd "$HOME/hkpl_multiformat_patch"
git fetch origin
git switch feat/rag-ingestion-work
git pull --ff-only origin feat/rag-ingestion-work
docker compose build langgraph-agent
```

- `git fetch` refreshes remote branch information.
- `git switch` selects the ingestion feature branch.
- `git pull --ff-only` updates it without creating a merge commit.
- `docker compose build` places the latest Python code in the agent image.

Confirm the checkout and commit:

```bash
pwd
git branch --show-current
git log -1 --oneline
```

Expected directory and branch:

```text
/home/maxchong/hkpl_multiformat_patch
feat/rag-ingestion-work
```

### 2. Start and check preview dependencies

```bash
cd "$HOME/hkpl_multiformat_patch"
docker compose up -d postgres llm
docker compose ps postgres llm
curl -fsS http://127.0.0.1:8081/health
```

`up -d` starts PostgreSQL and the local Qwen3.5-9B generation service if they
are stopped. It does not run ingestion.

Confirm that the database container is the existing shared deployment:

```bash
docker inspect hkpl_postgres --format \
  '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}'
```

The existing container was created from
`/home/cnatalie/agentic-RAG-v2/hkpl_multiformat_patch`. That is acceptable for
this experiment because the preview command connects to the existing database
but runs code from Max's newly built agent image.

### 3. Prepare offline artifacts once

Create the two user-owned model directories:

```bash
mkdir -p "$HOME/hkpl-models/docling"
mkdir -p "$HOME/hkpl-models/qwen3-embedding"
```

Download Docling's layout and TableFormer artifacts:

```bash
cd "$HOME/hkpl_multiformat_patch"
uv run docling-tools models download \
  layout tableformer \
  --output-dir "$HOME/hkpl-models/docling"
```

Download only the Qwen embedding tokenizer files. This does not download or
run the embedding model weights:

```bash
cd "$HOME/hkpl_multiformat_patch"
TOKENIZER_DIR="$HOME/hkpl-models/qwen3-embedding" uv run python -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-Embedding-0.6B",
    local_dir=os.environ["TOKENIZER_DIR"],
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
)'
```

Verify the downloaded files:

```bash
find "$HOME/hkpl-models/docling" -path '*/accurate/tm_config.json' -print
find "$HOME/hkpl-models/docling" -name model.safetensors -print | head
test -s "$HOME/hkpl-models/qwen3-embedding/tokenizer.json" \
  && echo "tokenizer files OK"
```

All three commands must print output.

Finally, load the tokenizer inside the agent image using the same mount as the
full run:

```bash
cd "$HOME/hkpl_multiformat_patch"
docker compose run --rm --no-deps \
  --volume "$HOME/hkpl-models/qwen3-embedding:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python -c '
from src.ingestion.tokenizer import get_embedding_tokenizer
t = get_embedding_tokenizer()
print("tokenizer load OK; sample tokens:", t.count_tokens("Hong Kong public library"))'
```

Do not start the full preview unless this prints `tokenizer load OK`.

### 4. Create repository working directories

```bash
cd "$HOME/hkpl_multiformat_patch"
mkdir -p data/incoming data/ingestion_preview storage/docling
```

- `data/incoming` holds files intentionally copied into the ingestion mount.
- `data/ingestion_preview` holds run logs and CSV exports.
- `storage/docling` holds reusable extracted Docling JSON.

### 5. Clear smoke-test and failed preview runs

This removes only preview rows:

```bash
docker exec hkpl_postgres psql \
  -U postgres -d hkpl_vector_db -v ON_ERROR_STOP=1 \
  -c 'TRUNCATE TABLE ingestion_preview_chunks, ingestion_preview_documents;'
```

Confirm that preview tables are empty and live counts remain intact:

```bash
docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db -c '
SELECT (SELECT COUNT(*) FROM ingestion_preview_documents) AS preview_documents,
       (SELECT COUNT(*) FROM ingestion_preview_chunks) AS preview_chunks,
       (SELECT COUNT(*) FROM knowledge_documents) AS live_documents,
       (SELECT COUNT(*) FROM data_hkpl_knowledge) AS live_chunks;'
```

Expected preview counts are zero. The known live baseline before this
experiment was 325 documents and 15,699 chunks.

### 6. Run the full pre-embedding preview

```bash
cd "$HOME/hkpl_multiformat_patch"
RUN_ID="full-preview-$(date -u +%Y%m%dT%H%M%SZ)"
echo "$RUN_ID" | tee data/ingestion_preview/latest_run_id.txt

docker compose run --rm --no-deps \
  --volume "$HOME/hkpl-models/docling:/app/models/docling:ro" \
  --volume "$HOME/hkpl-models/qwen3-embedding:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/preview_ingestion.py \
    --all-sources \
    --run-id "$RUN_ID" \
  2>&1 | tee "data/ingestion_preview/$RUN_ID.log"
```

This command:

- selects all registered crawler and uploaded sources;
- extracts their content;
- classifies unlabelled sources with the local Qwen3.5-9B model, with reasoning
  disabled;
- creates 512-token-bounded pre-embedding chunks;
- writes only the two preview tables;
- saves the terminal log under `data/ingestion_preview`.

Remove `--all-sources` to run crawler sources only. Add `--limit 20` for
another bounded smoke test. Do not add `--classify-only` when testing chunks.

Pressing `Ctrl-C` stops the preview container. PostgreSQL, the LLM, and the
live corpus remain running.

Messages such as Tesseract `Too few characters` are OCR warnings unless the
source ultimately has `status = 'failed'`.

### 7. Monitor the current run

Open a second SSH session and load the latest run ID:

```bash
cd "$HOME/hkpl_multiformat_patch"
RUN_ID=$(cat data/ingestion_preview/latest_run_id.txt)
echo "$RUN_ID"
```

Check progress by status and document type:

```bash
docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db -c "
SELECT status, document_type, COUNT(*)
FROM ingestion_preview_documents
WHERE run_id = '$RUN_ID'
GROUP BY status, document_type
ORDER BY status, document_type;"
```

Follow the saved log:

```bash
tail -f "data/ingestion_preview/$RUN_ID.log"
```

Show only likely failures:

```bash
grep -E 'FAILED|Traceback|Error|Exception' \
  "data/ingestion_preview/$RUN_ID.log"
```

### 8. Check completion and failures

```bash
docker exec -it hkpl_postgres psql -U postgres -d hkpl_vector_db
```

Inside `psql`, load the run ID printed by the full-run command:

```sql
\set run_id 'full-preview-20260813T000000Z'

SELECT status, document_type, COUNT(*)
FROM ingestion_preview_documents
WHERE run_id = :'run_id'
GROUP BY status, document_type
ORDER BY status, document_type;

SELECT source_title, source_url, error_message
FROM ingestion_preview_documents
WHERE run_id = :'run_id' AND status = 'failed'
ORDER BY source_title;
```

Replace the example timestamp with the actual run ID. Exit `psql` with `\q`.
Do not evaluate chunk quality until extraction and classification failures are
understood.

### 9. Audit chunk quality before embedding

Open `psql` and set the run ID as in step 8, then run the following checks.

Token distribution and hard-limit violations:

```sql
SELECT COUNT(*) AS chunks,
       MIN(token_count) AS min_tokens,
       ROUND(AVG(token_count), 1) AS avg_tokens,
       MAX(token_count) AS max_tokens,
       COUNT(*) FILTER (WHERE token_count > 512) AS over_limit
FROM ingestion_preview_chunks
WHERE run_id = :'run_id';
```

Chunk counts per document:

```sql
SELECT d.document_type,
       d.source_title,
       d.status,
       COUNT(c.chunk_id) AS chunks,
       MIN(c.token_count) AS min_tokens,
       ROUND(AVG(c.token_count), 1) AS avg_tokens,
       MAX(c.token_count) AS max_tokens
FROM ingestion_preview_documents d
LEFT JOIN ingestion_preview_chunks c
  ON c.run_id = d.run_id AND c.document_id = d.document_id
WHERE d.run_id = :'run_id'
GROUP BY d.document_id, d.document_type, d.source_title, d.status
ORDER BY d.document_type, d.source_title;
```

Missing text or provenance:

```sql
SELECT document_id, ordinal, chunk_id
FROM ingestion_preview_chunks
WHERE run_id = :'run_id'
  AND (BTRIM(evidence_text) = ''
       OR BTRIM(search_text) = ''
       OR locator = '{}'::jsonb);
```

Inspect actual chunk text and provenance:

```sql
\x on
SELECT d.source_title,
       d.source_url,
       d.document_type,
       c.ordinal,
       c.record_kind,
       c.chunk_policy,
       c.token_count,
       c.structure_path,
       c.locator,
       c.evidence_text,
       c.search_text
FROM ingestion_preview_chunks c
JOIN ingestion_preview_documents d
  ON d.run_id = c.run_id AND d.document_id = c.document_id
WHERE c.run_id = :'run_id'
ORDER BY d.source_title, c.ordinal
LIMIT 50;
```

### 10. List and export preview runs

List all runs:

```bash
docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db -c "
SELECT run_id,
       COUNT(*) AS documents,
       COUNT(*) FILTER (WHERE status = 'completed') AS completed,
       COUNT(*) FILTER (WHERE status = 'failed') AS failed,
       COUNT(*) FILTER (WHERE status = 'skipped') AS skipped,
       MAX(created_at) AS last_update
FROM ingestion_preview_documents
GROUP BY run_id
ORDER BY last_update DESC;"
```

Export the latest run to CSV:

```bash
cd "$HOME/hkpl_multiformat_patch"
RUN_ID=$(cat data/ingestion_preview/latest_run_id.txt)

docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db --csv -c \
  "SELECT * FROM ingestion_preview_documents WHERE run_id = '$RUN_ID' ORDER BY source_title" \
  > "data/ingestion_preview/$RUN_ID-documents.csv"

docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db --csv -c \
  "SELECT * FROM ingestion_preview_chunks WHERE run_id = '$RUN_ID' ORDER BY document_id, ordinal" \
  > "data/ingestion_preview/$RUN_ID-chunks.csv"
```

### 11. Remove only the latest run

```bash
cd "$HOME/hkpl_multiformat_patch"
RUN_ID=$(cat data/ingestion_preview/latest_run_id.txt)

docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db \
  -v ON_ERROR_STOP=1 -v run_id="$RUN_ID" -c "
BEGIN;
DELETE FROM ingestion_preview_chunks WHERE run_id = :'run_id';
DELETE FROM ingestion_preview_documents WHERE run_id = :'run_id';
COMMIT;"
```

This retains other preview runs. To remove all preview runs, use the `TRUNCATE`
command in step 5. Neither operation touches the live corpus.

Do not reuse a partially populated run ID. Delete the failed run, correct the
cause, and generate a new timestamped ID.

### 12. Review interactively with DBeaver

On the Windows computer, open an SSH tunnel using the same server account and
server address normally used for SSH. Forward local port `15432` to server
port `5433`:

```text
ssh -N -L 15432:127.0.0.1:5433 maxchong@the-server-address
```

In DBeaver use:

```text
Host      127.0.0.1
Port      15432
Database  hkpl_vector_db
Username  postgres
Password  postgres
```

Replace only `the-server-address` with the address from the normal SSH command.
Keep the SSH window open while DBeaver is connected. Do not expose PostgreSQL
publicly.

## Live ingestion

These commands write embeddings and live ingestion records. They intentionally
fail while `KNOWLEDGE_CORPUS_READ_ONLY=true` or the database corpus lock is
enabled. Keep the lock enabled during the preview experiment.

### Ingest one file

Copy the source into the repository's incoming directory. The example below
assumes the source is named `example.pdf` in the current directory:

```bash
cd "$HOME/hkpl_multiformat_patch"
mkdir -p data/incoming
cp example.pdf data/incoming/example.pdf
```

For a librarian-labelled prose file:

```bash
docker compose run --rm \
  --volume "$HOME/hkpl-models/docling:/app/models/docling:ro" \
  --volume "$HOME/hkpl-models/qwen3-embedding:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/ingest_documents.py \
  /app/data/incoming/example.pdf \
  --document-type prose
```

Change `prose` to `faq` or `record` when the librarian supplies that label.
Use `auto` only when no librarian label exists; the local 9B model then
classifies the extracted content.

### Ingest a directory of files

Place the batch under `data/incoming/batch` and run:

```bash
cd "$HOME/hkpl_multiformat_patch"
mkdir -p data/incoming/batch

docker compose run --rm \
  --volume "$HOME/hkpl-models/docling:/app/models/docling:ro" \
  --volume "$HOME/hkpl-models/qwen3-embedding:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/ingest_documents.py /app/data/incoming/batch
```

Directory ingestion always classifies unlabelled files in efficient 9B batches.

### Crawl HKPL websites

Start with a bounded official HKPL crawl:

```bash
cd "$HOME/hkpl_multiformat_patch"
docker compose run --rm \
  --volume "$HOME/hkpl-models/docling:/app/models/docling:ro" \
  --volume "$HOME/hkpl-models/qwen3-embedding:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/crawl_hkpl_site.py \
  --seed-url 'https://www.hkpl.gov.hk/en/' \
  --max-pages 20 \
  --max-depth 2
```

The crawler follows allowlisted HKPL URLs, stores supported HTML/PDF content,
classifies sources in 9B batches, and embeds non-`skip` sources. Listing and
navigation pages classified as `skip` create no searchable chunks.

Increase `--max-pages` or `--max-depth` only after inspecting the bounded run.

## Current limitation

`preview_ingestion.py` only previews sources already registered in the
database. It does not acquire or register a new file or URL. Keep the live
corpus lock enabled for this experiment.
