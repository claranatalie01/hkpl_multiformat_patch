# HKPL ingestion and RAG evaluation

The supported workflow is now one pass:

```text
crawl → classify → extract → chunk → embed → generate evaluation data → evaluate
```

Use crawler limits to keep a trial run small.

## 1. Start services

```bash
cd "$HOME/hkpl_multiformat_patch"
docker compose up -d postgres embedding reranker llm phoenix
```

Check whether the database corpus lock is enabled:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --status
```

If it is enabled, disable it for ingestion:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --disable --yes
```

## 2. Run a bounded ingestion

This example visits at most 50 URLs and follows links two levels from the seed.
Every new or changed source is classified, extracted, chunked, embedded with
Qwen3-Embedding, and stored before the command finishes.

```bash
docker compose run --rm --no-deps \
  -e KNOWLEDGE_CORPUS_READ_ONLY=false \
  langgraph-agent \
  uv run python scripts/crawl_hkpl_site.py \
  --seed-url 'https://www.hkpl.gov.hk/en/' \
  --max-pages 50 \
  --max-depth 2
```

Change `--max-pages` and `--max-depth` for later runs. Unchanged URLs do not
create duplicate vectors. To omit PDFs, add `--exclude-pdfs`.

## 3. Inspect ingestion results

Documents:

```sql
SELECT
    COALESCE(NULLIF(source_title, ''), original_file_name) AS source,
    NULLIF(source_url, '') AS source_url,
    document_type,
    classification_source,
    status,
    chunk_count,
    error_message
FROM knowledge_documents
ORDER BY updated_at DESC, source;
```

Chunks:

```sql
SELECT
    COALESCE(v.metadata_->>'source_title', v.metadata_->>'file_name') AS document,
    v.metadata_->>'source_url' AS source_url,
    v.metadata_->>'chunk_id' AS chunk_id,
    v.metadata_->'locator' AS locator,
    v.metadata_->>'evidence_text' AS content
FROM data_hkpl_knowledge AS v
WHERE COALESCE(NULLIF(v.metadata_->>'dataset', ''), 'hkpl') = 'hkpl'
ORDER BY document, chunk_id;
```

Freeze the corpus again before generating benchmark labels:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --enable
```

## 4. Generate an evaluation candidate

The benchmark workflow audits the embedded corpus, generates candidate
questions, loads them into a candidate table, and validates every evidence
reference:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py prepare-candidate \
  --target-questions 100
```

The output is `data/evaluation_dataset.candidate.csv`. Review it before
promotion; generated questions are labels proposed by an LLM, not trusted gold.

After review:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py promote --yes
```

## 5. Run evaluation

For an initial HKPL-only diagnostic without distractor corpora:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/evaluate_rag.py \
  --allow-missing-distractors
```

For the full benchmark, load HotpotQA and Webz News distractors, then run:

```bash
docker compose run --rm --no-deps langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py evaluate
```
