# HKPL ingestion runbook

This runbook covers two different operations:

1. **Pre-embedding preview**: classify and chunk registered sources into isolated
   preview tables for human inspection. This is the current experiment.
2. **Live ingestion**: acquire, classify, chunk, embed, validate, and publish
   sources. Run this only in an approved writable environment.

The preview does not modify `knowledge_documents` or `data_hkpl_knowledge` and
does not create embeddings.

## Pre-embedding preview

### 1. Update and build

From the server checkout:

```bash
cd "$HOME/hkpl_multiformat_patch"
git fetch origin
git switch feat/rag-ingestion-work
git pull --ff-only origin feat/rag-ingestion-work
docker compose build langgraph-agent
```

The PostgreSQL and 9B generation services must already be running and healthy:

```bash
docker compose ps postgres llm
curl -fsS http://127.0.0.1:8081/health
```

### 2. Prepare offline models once

Docling needs the layout and TableFormer artifacts:

```bash
mkdir -p "$HOME/hkpl-models/docling"
uv run docling-tools models download \
  layout tableformer \
  --output-dir "$HOME/hkpl-models/docling"
```

Full chunking also needs the pinned embedding tokenizer. This downloads only
the tokenizer for local token counting; it does not run the embedding model:

```bash
mkdir -p "$HOME/hkpl-models/qwen3-embedding"
MODEL_DIR="$HOME/hkpl-models/qwen3-embedding" \
uv run python -c 'import os; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=False).save_pretrained(os.environ["MODEL_DIR"])'
```

Verify both directories before starting a long run:

```bash
find "$HOME/hkpl-models/docling" -path '*/accurate/tm_config.json' -print
find "$HOME/hkpl-models/docling" -name model.safetensors -print | head
test -f "$HOME/hkpl-models/qwen3-embedding/tokenizer_config.json" && echo "tokenizer OK"
```

All three checks must produce output.

### 3. Clear earlier preview runs

This removes only the two preview tables' rows. It does not touch the live
registry or embedded corpus:

```bash
docker exec hkpl_postgres psql \
  -U postgres -d hkpl_vector_db -v ON_ERROR_STOP=1 \
  -c 'TRUNCATE TABLE ingestion_preview_chunks, ingestion_preview_documents;'
```

Confirm that the preview is empty and the live corpus is unchanged:

```bash
docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db -c '
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
echo "Run ID: $RUN_ID"

docker compose run --rm --no-deps \
  --volume "$HOME/hkpl-models/docling:/app/models/docling:ro" \
  --volume "$HOME/hkpl-models/qwen3-embedding:/app/models/qwen3-embedding:ro" \
  langgraph-agent \
  uv run python scripts/preview_ingestion.py \
    --all-sources \
    --run-id "$RUN_ID" \
  2>&1 | tee "$RUN_ID.log"
```

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
docker exec hkpl_postgres psql -U postgres -d hkpl_vector_db -c "
SELECT status, document_type, COUNT(*)
FROM ingestion_preview_documents
WHERE run_id = '$RUN_ID'
GROUP BY status, document_type
ORDER BY status, document_type;"
```

If `RUN_ID` was set only in the first shell, replace `$RUN_ID` with the printed
run ID.

### 6. Check completion and failures

```bash
docker exec -it hkpl_postgres psql -U postgres -d hkpl_vector_db
```

Inside `psql`:

```sql
\set run_id 'full-preview-YYYYMMDDTHHMMSSZ'

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

## Adding sources to the live ingestion pipeline

These commands are not preview-only. They classify, chunk, embed, and attempt
to write the live corpus. They will intentionally fail while
`KNOWLEDGE_CORPUS_READ_ONLY=true` or the database corpus lock is enabled.

### Files

Place files under a directory mounted into the agent container, such as
`data/incoming`.

For one librarian-labelled file:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/ingest_documents.py \
  /app/data/incoming/example.pdf \
  --document-type prose
```

Allowed labels are `faq`, `record`, and `prose`. Use `--document-type auto` if
no librarian label exists; the 9B model will classify by content.

For a directory, files are registered and classified in batches by the 9B
model:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/ingest_documents.py /app/data/incoming
```

### Websites

The crawler follows allowlisted HKPL URLs, stores supported HTML/PDF sources,
classifies them in batches, chunks them, and embeds non-`skip` sources:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/crawl_hkpl_site.py \
  --seed-url 'https://www.hkpl.gov.hk/en/' \
  --max-pages 20 \
  --max-depth 2
```

Start with a bounded crawl and inspect its logs before increasing the page or
depth limits. Listing/navigation pages classified as `skip` contribute no
searchable chunks.

## Current limitation

`preview_ingestion.py` previews sources already registered in the database; it
does not acquire or register a new file or URL. Keep the live corpus lock on
during this experiment. Add a separate preview-registration path only if new,
unregistered fixtures must be evaluated before live ingestion.
