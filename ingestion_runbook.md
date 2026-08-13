# HKPL ingestion runbook

This runbook covers two different operations:

1. **Pre-embedding preview**: classify and chunk registered sources into isolated
   preview tables for human inspection. This is the current experiment.
2. **Live ingestion**: acquire, classify, chunk, embed, validate, and publish
   sources. Run this only in an approved writable environment.

The preview does not modify `knowledge_documents` or `data_hkpl_knowledge` and
does not create embeddings.

## Placeholders used below

Set these once in each SSH shell. Change values inside `<...>`; remove the
angle brackets when replacing them.

```bash
export REPO_DIR="<absolute checkout path, e.g. $HOME/hkpl_multiformat_patch>"
export FEATURE_BRANCH="<branch, e.g. feat/rag-ingestion-work>"
export DOCLING_MODELS_DIR="<Docling directory, e.g. $HOME/hkpl-models/docling>"
export TOKENIZER_DIR="<tokenizer directory, e.g. $HOME/hkpl-models/qwen3-embedding>"
export DB_CONTAINER="<PostgreSQL container, e.g. hkpl_postgres>"
export DB_NAME="<database, e.g. hkpl_vector_db>"
export DB_USER="<database user, e.g. postgres>"
```

`export` makes a value available to later commands in the same shell. Check a
value with `printf '%s\n' "$REPO_DIR"`. A new SSH shell must set the variables
again.

## Pre-embedding preview

### 1. Update and build

From the server checkout:

```bash
cd "$REPO_DIR"
git fetch origin
git switch "$FEATURE_BRANCH"
git pull --ff-only origin "$FEATURE_BRANCH"
docker compose build langgraph-agent
```

What these do:

- `cd` selects the intended checkout. This matters when several clones exist.
- `git fetch` refreshes remote branch information without changing files.
- `git switch` selects the feature branch.
- `git pull --ff-only` updates it without creating an accidental merge commit.
- `docker compose build` rebuilds the agent image with the checked-out code.

Confirm that Compose is using this checkout, not another user's project:

```bash
pwd
git branch --show-current
git log -1 --oneline
docker inspect "$DB_CONTAINER" --format \
  '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}'
```

The `docker inspect` command reports which checkout created the existing
database container. It may intentionally differ on a shared server; record the
result and verify the database name and live-table counts before continuing.

The PostgreSQL and 9B generation services must already be running and healthy:

```bash
docker compose ps postgres llm
curl -fsS http://127.0.0.1:8081/health
```

If either service is stopped, start only the dependencies needed by preview:

```bash
docker compose up -d postgres llm
docker compose ps postgres llm
```

`up -d` changes container state but does not run ingestion.

### 2. Prepare offline models once

Docling needs the layout and TableFormer artifacts:

```bash
mkdir -p "$DOCLING_MODELS_DIR"
uv run docling-tools models download \
  layout tableformer \
  --output-dir "$DOCLING_MODELS_DIR"
```

Full chunking also needs the pinned embedding tokenizer. This downloads only
the tokenizer for local token counting; it does not run the embedding model:

```bash
mkdir -p "$TOKENIZER_DIR"
TOKENIZER_DIR="$TOKENIZER_DIR" uv run python -c '
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3-Embedding-0.6B",
    local_dir=os.environ["TOKENIZER_DIR"],
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
)'
```

Verify both directories before starting a long run:

```bash
find "$DOCLING_MODELS_DIR" -path '*/accurate/tm_config.json' -print
find "$DOCLING_MODELS_DIR" -name model.safetensors -print | head
test -s "$TOKENIZER_DIR/tokenizer.json" && echo "tokenizer files OK"
```

All three checks must produce output.

The decisive tokenizer check is to load it inside the same image and mount used
by the full run:

```bash
docker compose run --rm --no-deps \
  --volume "$TOKENIZER_DIR:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python -c '
from src.ingestion.tokenizer import get_embedding_tokenizer
t = get_embedding_tokenizer()
print("tokenizer load OK; sample tokens:", t.count_tokens("Hong Kong 公共圖書館"))'
```

Do not start the full run unless this prints `tokenizer load OK`.

### 3. Clear earlier preview runs

This removes only the two preview tables' rows. It does not touch the live
registry or embedded corpus:

```bash
docker exec "$DB_CONTAINER" psql \
  -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 \
  -c 'TRUNCATE TABLE ingestion_preview_chunks, ingestion_preview_documents;'
```

Confirm that the preview is empty and the live corpus is unchanged:

```bash
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c '
SELECT (SELECT COUNT(*) FROM ingestion_preview_documents) AS preview_documents,
       (SELECT COUNT(*) FROM ingestion_preview_chunks) AS preview_chunks,
       (SELECT COUNT(*) FROM knowledge_documents) AS live_documents,
       (SELECT COUNT(*) FROM data_hkpl_knowledge) AS live_chunks;'
```

### 4. Run classification and chunking

The default scope is crawler sources. Add `--all-sources` to include registered
uploads as well. Do not pass `--classify-only` for the full preview.

```bash
RUN_ID="full-preview-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID
echo "Run ID: $RUN_ID"

docker compose run --rm --no-deps \
  --volume "$DOCLING_MODELS_DIR:/app/models/docling:ro" \
  --volume "$TOKENIZER_DIR:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/preview_ingestion.py \
    --all-sources \
    --run-id "$RUN_ID" \
  2>&1 | tee "$RUN_ID.log"
```

What to change:

- Remove `--all-sources` to preview crawler sources only.
- Add `--limit <NUMBER_OF_DOCUMENTS>` for a bounded trial.
- Replace the generated `RUN_ID` with a meaningful unique value if desired.
- Keep `--no-deps` only when PostgreSQL and the 9B service are already running.

`tee` displays output and saves the same output in `<RUN_ID>.log`. Keep the
foreground command attached. Pressing `Ctrl-C` stops the preview container; it
does not affect PostgreSQL or the live corpus.

OCR messages such as `Too few characters` are warnings unless the document is
ultimately recorded as `failed`. The final table status is authoritative.

The pipeline extracts the source, sends a content sample to the local
Qwen3.5-9B generation service with reasoning disabled, and assigns one of:

- `faq`: question-answer records;
- `record`: a self-contained notice, event, or branch profile;
- `prose`: policies, guidance, articles, and other narrative material;
- `skip`: listing/navigation pages whose value is primarily outbound links.

`faq`, `record`, and `prose` are then chunked. Physical table structure takes
precedence over the label and produces row-aware chunks. `skip` produces no
chunks. The hard limit is 512 embedding-tokenizer tokens; 64-token overlap is
used only when an oversized leaf must be split.

### 5. Monitor the run

Open another SSH session while the command is running:

```bash
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
SELECT status, document_type, COUNT(*)
FROM ingestion_preview_documents
WHERE run_id = '$RUN_ID'
GROUP BY status, document_type
ORDER BY status, document_type;"
```

If `RUN_ID` was set only in the first shell, replace `$RUN_ID` with the printed
run ID.

Watch the saved application log:

```bash
tail -f "$REPO_DIR/$RUN_ID.log"
```

Show only failures and tracebacks:

```bash
grep -E 'FAILED|Traceback|Error|Exception' "$REPO_DIR/$RUN_ID.log"
```

### 6. Check completion and failures

```bash
docker exec -it "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME"
```

Inside `psql`:

```sql
\set run_id 'full-preview-YYYYMMDDTHHMMSSZ'

\d+ ingestion_preview_documents
\d+ ingestion_preview_chunks

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

Do not evaluate chunk quality until extraction failures are understood. A
small number of explicitly unsupported or empty sources may be excluded only
after recording why.

### 7. Audit chunk quality before embedding

Token distribution and hard-limit violations:

```sql
SELECT
    COUNT(*) AS chunks,
    MIN(token_count) AS min_tokens,
    ROUND(AVG(token_count), 1) AS avg_tokens,
    MAX(token_count) AS max_tokens,
    COUNT(*) FILTER (WHERE token_count > 512) AS over_limit
FROM ingestion_preview_chunks
WHERE run_id = :'run_id';
```

Chunk counts per document:

```sql
SELECT
    d.document_type,
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
  AND (
    BTRIM(evidence_text) = '' OR
    BTRIM(search_text) = '' OR
    locator = '{}'::jsonb
  );
```

Read actual chunks with expanded output:

```sql
\x on
SELECT
    d.source_title,
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

Exit with `\q`. For sustained review, connect DBeaver through an SSH tunnel and
filter the same two tables by `run_id`.

### 8. List, export, and remove preview runs

List every run without opening interactive `psql`:

```bash
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -c "
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

Export one run to CSV files on the server. Replace `<RUN_ID>` if the variable
is not set in this shell:

```bash
export RUN_ID="<RUN_ID>"

docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" --csv -c \
  "SELECT * FROM ingestion_preview_documents WHERE run_id = '$RUN_ID' ORDER BY source_title" \
  > "$RUN_ID-documents.csv"

docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" --csv -c \
  "SELECT * FROM ingestion_preview_chunks WHERE run_id = '$RUN_ID' ORDER BY document_id, ordinal" \
  > "$RUN_ID-chunks.csv"
```

Delete one failed or unwanted run while retaining other previews:

```bash
export RUN_ID="<RUN_ID_TO_DELETE>"

docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -v run_id="$RUN_ID" -c "
BEGIN;
DELETE FROM ingestion_preview_chunks WHERE run_id = :'run_id';
DELETE FROM ingestion_preview_documents WHERE run_id = :'run_id';
COMMIT;"
```

Delete every preview run:

```bash
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 \
  -c 'TRUNCATE TABLE ingestion_preview_chunks, ingestion_preview_documents;'
```

Neither deletion command touches the live document registry or embedded table.

### 9. Recover from a failed full preview

Do not reuse a partially populated `RUN_ID`: completed chunks already have
primary keys for that run. Delete the failed run, correct the cause, and start
with a new run ID.

For the tokenizer error `Couldn't instantiate the backend tokenizer`, repeat
the tokenizer download and decisive in-container load check in step 2.

Malformed 9B JSON is recorded as a failed classification batch and later
batches continue. After the run, query failed documents as shown in step 6.
Do not silently relabel them; rerun them after investigating the model output.

Check disk space before another full run:

```bash
df -h "$REPO_DIR" "$DOCLING_MODELS_DIR" "$TOKENIZER_DIR"
docker system df
```

`docker system df` is read-only. Do not run Docker prune commands on this
shared server without administrator approval.

### 10. Interactive database access with DBeaver

On the local computer, create an SSH tunnel. Replace both placeholders:

```bash
ssh -N -L <LOCAL_PORT>:127.0.0.1:5433 <SSH_USER>@<SERVER_HOST>
```

Example local port: `15432`. In DBeaver use:

- host: `127.0.0.1`;
- port: `<LOCAL_PORT>`;
- database: `<DB_NAME>`;
- username: `<DB_USER>`;
- password: `<DATABASE_PASSWORD>`.

Keep the SSH command running while DBeaver is connected. Do not expose port
5433 publicly merely to use a database GUI.

## Adding sources to the live ingestion pipeline

These commands are not preview-only. They classify, chunk, embed, and attempt
to write the live corpus. They will intentionally fail while
`KNOWLEDGE_CORPUS_READ_ONLY=true` or the database corpus lock is enabled.

### Files

Place files under a directory mounted into the agent container, such as
`data/incoming`.

```bash
mkdir -p "$REPO_DIR/data/incoming"
cp "<ABSOLUTE_SOURCE_FILE>" "$REPO_DIR/data/incoming/<FILE_NAME>"
```

`cp` creates a test copy inside the checkout; it does not remove the original.

For one librarian-labelled file:

```bash
docker compose run --rm \
  --volume "$DOCLING_MODELS_DIR:/app/models/docling:ro" \
  --volume "$TOKENIZER_DIR:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/ingest_documents.py \
  /app/data/incoming/<FILE_NAME> \
  --document-type <faq|record|prose|auto>
```

Allowed labels are `faq`, `record`, and `prose`. Use `--document-type auto` if
no librarian label exists; the 9B model will classify by content.

For a directory, files are registered and classified in batches by the 9B
model:

```bash
docker compose run --rm \
  --volume "$DOCLING_MODELS_DIR:/app/models/docling:ro" \
  --volume "$TOKENIZER_DIR:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/ingest_documents.py /app/data/incoming/<DIRECTORY_NAME>
```

### Websites

The crawler follows allowlisted HKPL URLs, stores supported HTML/PDF sources,
classifies them in batches, chunks them, and embeds non-`skip` sources:

```bash
docker compose run --rm \
  --volume "$DOCLING_MODELS_DIR:/app/models/docling:ro" \
  --volume "$TOKENIZER_DIR:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/crawl_hkpl_site.py \
  --seed-url '<ALLOWLISTED_HTTPS_URL>' \
  --max-pages <MAX_PAGES> \
  --max-depth <MAX_DEPTH>
```

Start with a bounded crawl and inspect its logs before increasing the page or
depth limits. Listing/navigation pages classified as `skip` contribute no
searchable chunks.

## Current limitation

`preview_ingestion.py` previews sources already registered in the database; it
does not acquire or register a new file or URL. Keep the live corpus lock on
during this experiment. Add a separate preview-registration path only if new,
unregistered fixtures must be evaluated before live ingestion.
