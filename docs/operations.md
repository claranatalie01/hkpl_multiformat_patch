# Operations and troubleshooting

## Service inspection

```bash
docker compose ps
docker compose logs --tail=100 langgraph-agent
docker compose config --quiet
```

Rebuild the agent after dependency or Dockerfile changes:

```bash
docker compose up -d --build langgraph-agent
```

Source code and scripts are bind-mounted in the PoC profile, so a one-off
`docker compose run --rm` command sees current local Python files. Restart the
long-running API process after changing imported runtime code.

## PostgreSQL access and tables

```bash
docker compose exec postgres psql -U postgres -d hkpl_vector_db
```

Inside `psql`:

```sql
\dt
SELECT COUNT(*) FROM data_hkpl_knowledge;
SELECT COUNT(*) FROM knowledge_documents;
```

Major tables:

| Table | Purpose |
|---|---|
| `data_hkpl_knowledge` | LlamaIndex text, metadata, node IDs, and pgvector embeddings |
| `knowledge_documents` | Source registry, versions, hashes, statuses, and chunk counts |
| `evaluation_dataset*` | Reviewed or candidate benchmark rows |
| `conversation_history` | Short-lived conversation turns in the current PoC |
| `knowledge_corpus_control` | Corpus write-lock state |
| `prohibited_keywords` | Application compliance rules |
| `prohibited_keyword_audit_log` | Changes to those compliance rules |

## Corpus write protection

Stopping cron prevents that scheduler from starting new crawls. The write guard
also blocks accidental manual, API, migration, or alternate-scheduler writes.
They protect against different causes, so a frozen benchmark should use both.

Inspect the current setting:

```bash
docker compose run --rm --no-deps langgraph-agent \
  python scripts/manage_corpus_lock.py --status
```

Enable protection:

```bash
docker compose run --rm --no-deps langgraph-agent \
  python scripts/manage_corpus_lock.py --enable --yes
```

Disable protection only for a controlled ingestion window:

```bash
docker compose run --rm --no-deps \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/manage_corpus_lock.py --disable --yes
```

Re-enable it immediately after the write and verify with `--status`.

## Cron

The application does not create cron jobs. A user-owned crontab can schedule a
crawl without `sudo` when that user already has Docker access. Inspect it with:

```bash
crontab -l
```

An entry such as this runs every four hours:

```cron
0 */4 * * * cd /absolute/project/path && docker compose run --rm langgraph-agent python scripts/crawl_hkpl_site.py >> logs/hkpl_crawler.log 2>&1
```

Edit with `crontab -e`. The `/tmp/crontab.*` file shown by the editor is a
temporary edit buffer; the installed crontab is read with `crontab -l`.

## Common failures

### Transaction is aborted

After one PostgreSQL statement fails inside `BEGIN`, later statements are
ignored until rollback:

```sql
ROLLBACK;
```

Fix the original error, then start a new transaction.

### Knowledge corpus is frozen

The database trigger or application guard is working. Check the lock, confirm
the intended `VECTOR_TABLE`, open a controlled write window, perform the write,
then re-enable protection.

### `.env` is missing or invalid

Compose injects the project `.env` into the agent container. Create it from the
tracked template and keep it out of Git:

```bash
ls -ld .env
cp .env.example .env
```

If `.env` is a directory, move it aside only after confirming it contains no
needed data, then create the regular file. The older direct `/app/.env` bind
mount is no longer used.

### Permission denied on `__pycache__`

A container previously wrote root-owned cache files into a bind mount. Confirm
the exact project path, repair ownership from a root container, and keep Python
caches ignored. Never run a recursive ownership command against a broad or
unverified directory.

### CUDA out of memory

Check `nvidia-smi`. The PoC runs generator, embedding, reranker, and safety
workloads; another process may already hold the GPU. Stop only known in-scope
processes or explicitly select an available GPU. The safety classifier does not
silently fall back to CPU when CUDA was requested.

### `&` versus `&amp;` evidence mismatch

Those strings are semantically the same HTML text but raw substring comparison
can fail. Normalize HTML entities consistently in evaluation evidence matching
rather than changing retrieval relevance.
