# Ingestion, chunking, embedding, and vector storage

## End-to-end flow

```text
scripts/crawl_hkpl_site.py or scripts/ingest_documents.py
        |
        v
hkpl_agent.ingestion.service
        |
        +-- registry.py        document ID, hash, version, status
        +-- readers.py         structured extraction
        +-- chunking.py        stable evidence chunks
        +-- embedding.py       1024-dimensional vectors
        +-- vector_store.py    PostgreSQL/pgvector insertion
        v
knowledge_documents + data_<VECTOR_TABLE>
```

The crawler does not implement a second ingestion pipeline. It discovers an
approved URL, saves an immutable source artifact under `uploads`, and delegates
to the same ingestion service used by administrative uploads.

## What each stage does

1. **Acquire.** The crawler follows bounded HKPL links and `robots.txt`, or an
   operator provides a supported local file.
2. **Register.** `registry.py` records source identity, content hash, version,
   stored file name, and processing status in `knowledge_documents`.
3. **Extract.** `readers.py` chooses a format reader. Docling handles complex
   document formats; deterministic readers preserve tabular rows and webpage
   structure. OCR is available to Docling/Tesseract when a PDF has no usable
   text layer.
4. **Classify.** An explicit document type wins. `auto` uses deterministic
   signals and a bounded classifier fallback to select an appropriate policy.
5. **Chunk.** `chunking.py` preserves atomic records such as events, FAQ items,
   branches, or table rows where possible, then applies token bounds. A source
   document can produce many chunks.
6. **Embed.** `embedding.py` sends each chunk's search text to the local Qwen
   embedding endpoint. The answer LLM is not used to create vectors.
7. **Insert.** LlamaIndex writes text, JSON metadata, stable node ID, and a
   1024-dimensional vector into PostgreSQL/pgvector.
8. **Activate.** Registry state is marked complete only after successful
   insertion. Replacement keeps the last complete version available until the
   new version succeeds.

## Vector-table naming

`VECTOR_TABLE` is the logical name used by the application. The current adapter
adds the physical `data_` prefix used by LlamaIndex:

| Setting | Physical table |
|---|---|
| `VECTOR_TABLE=hkpl_knowledge` | `data_hkpl_knowledge` |
| `VECTOR_TABLE=hkpl_knowledge_hybrid` | `data_hkpl_knowledge_hybrid` |

Use one table per coherent embedding/index configuration. Different corpora can
share a table when metadata such as `dataset` and `corpus_role` cleanly isolates
their purpose. A second table is useful for an experimental chunking or
embedding configuration that must not modify the active baseline.

## Crawl and ingest

Inspect the lock before writing:

```bash
docker compose run --rm --no-deps langgraph-agent \
  python scripts/manage_corpus_lock.py --status
```

Run a bounded crawl into the default vector table:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/crawl_hkpl_site.py \
  --max-pages 300 \
  --max-depth 3 \
  --delay-seconds 0.5
```

Ingest a saved file or directory through the same pipeline:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/ingest_documents.py /app/uploads/example.pdf
```

To build the hybrid experiment instead, change only the logical table setting:

```bash
docker compose run --rm \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  -e VECTOR_TABLE=hkpl_knowledge_hybrid \
  langgraph-agent \
  python scripts/crawl_hkpl_site.py \
  --max-pages 300 \
  --max-depth 3 \
  --delay-seconds 0.5
```

Changing `chunking.py` does not update existing rows. Re-run ingestion or a
reindex operation against the intended table to materialize new chunks and
vectors.

## Inspect chunks with SQL

Open `psql` inside the PostgreSQL service:

```bash
docker compose exec postgres psql -U postgres -d hkpl_vector_db
```

Count HKPL primary chunks:

```sql
SELECT COUNT(*)
FROM data_hkpl_knowledge
WHERE metadata_->>'dataset' = 'hkpl'
  AND metadata_->>'corpus_role' = 'primary';
```

Show chunks for one registered document:

```sql
SELECT
    node_id,
    metadata_->>'chunk_id' AS chunk_id,
    metadata_->>'part_number' AS part_number,
    text
FROM data_hkpl_knowledge
WHERE metadata_->>'kb_document_id' = 'DOCUMENT_UUID'
ORDER BY COALESCE((metadata_->>'part_number')::integer, 0), node_id;
```

Use SQL for exact corpus inspection. Use Phoenix when diagnosing how a query
retrieved, reranked, packed, and used those chunks.
