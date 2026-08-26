#!/usr/bin/env python3
"""Rebuild registered HKPL sources, synchronize evaluations, or audit chunks.

Registered crawler and upload sources are rebuilt from their durable copies in
``uploads/``. The shared readers, chunker, embedding client, registry, and
vector store are used so a rebuild follows the same pipeline as ingestion.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.schema import (
    EVALUATION_DATASET_COLUMNS,
    has_supported_evaluation_columns,
    parse_json_string_array,
    parse_parallel_evidence,
    serialize_string_array,
)
from src.infrastructure.table_names import configured_table_name
from src.infrastructure.vector_store import (
    EMBED_DIM,
    VECTOR_TABLE_NAME,
    ensure_hybrid_search_schema,
)
from src.infrastructure.db import engine
from src.ingestion.chunking import (
    chunk_documents,
)
from src.ingestion.document_types import normalize_document_type
from src.ingestion.registry import (
    ensure_registry_schema,
    list_documents,
)
from src.ingestion.readers import load_file
from src.ingestion.service import (
    OCR_LANGUAGES,
    UPLOAD_DIR,
    reindex_registered_document,
)
from src.ingestion.write_guard import ensure_corpus_writable
EVALUATION_DATASET_TABLE = configured_table_name(
    "EVALUATION_DATASET_TABLE",
    "evaluation_dataset",
)


def create_evaluation_dataset_table() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS {EVALUATION_DATASET_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    domain TEXT NOT NULL DEFAULT '',
                    query TEXT NOT NULL,
                    expected_answer_text TEXT NOT NULL DEFAULT '',
                    expected_context_snippet TEXT NOT NULL DEFAULT '',
                    expected_context_snippets_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    accepted_answers_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    source_title TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    source_document_id TEXT NOT NULL DEFAULT '',
                    source_chunk_id TEXT NOT NULL DEFAULT '',
                    source_chunk_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        )
        connection.execute(
            text(f"""
                ALTER TABLE {EVALUATION_DATASET_TABLE}
                ADD COLUMN IF NOT EXISTS accepted_answers_json JSONB
                NOT NULL DEFAULT '[]'::jsonb
            """)
        )
        connection.execute(
            text(f"""
                ALTER TABLE {EVALUATION_DATASET_TABLE}
                ADD COLUMN IF NOT EXISTS expected_context_snippets_json JSONB
                NOT NULL DEFAULT '[]'::jsonb
            """)
        )
        connection.execute(
            text(f"""
                ALTER TABLE {EVALUATION_DATASET_TABLE}
                ADD COLUMN IF NOT EXISTS source_chunk_ids_json JSONB
                NOT NULL DEFAULT '[]'::jsonb
            """)
        )
        connection.execute(
            text(f"""
                ALTER TABLE {EVALUATION_DATASET_TABLE}
                DROP COLUMN IF EXISTS source_type
            """)
        )
        connection.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS idx_{EVALUATION_DATASET_TABLE}_source_document
                ON {EVALUATION_DATASET_TABLE} (source_document_id)
            """)
        )
        connection.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS idx_{EVALUATION_DATASET_TABLE}_source_chunk
                ON {EVALUATION_DATASET_TABLE} (source_chunk_id)
            """)
        )
        # Evaluation questions are expected to be unique. Clean up legacy
        # duplicates before enforcing idempotent imports.
        connection.execute(
            text(f"""
                DELETE FROM {EVALUATION_DATASET_TABLE} older
                USING {EVALUATION_DATASET_TABLE} newer
                WHERE older.query = newer.query
                  AND older.id < newer.id
            """)
        )
        connection.execute(
            text(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_{EVALUATION_DATASET_TABLE}_query_unique
                ON {EVALUATION_DATASET_TABLE} (query)
            """)
        )


def ingest_evaluation_dataset(csv_path: str) -> tuple[int, int, int]:
    path = Path(csv_path)
    if not path.exists():
        print(f"Evaluation dataset not found, skipping: {path}")
        return 0, 0, 0

    create_evaluation_dataset_table()

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        actual_columns = list(reader.fieldnames or [])
        if not has_supported_evaluation_columns(actual_columns):
            raise ValueError(
                "Evaluation CSV columns must exactly match, in order: "
                f"{list(EVALUATION_DATASET_COLUMNS)}. Found: {actual_columns}"
            )

        rows = []
        seen_queries: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Malformed evaluation CSV row {line_number}: too many fields"
                )
            missing_values = [
                column
                for column in EVALUATION_DATASET_COLUMNS
                if column not in {
                    "accepted_answers_json",
                    "expected_context_snippets_json",
                    "source_chunk_ids_json",
                    "source_url",
                }
                and not str(row.get(column) or "").strip()
            ]
            if missing_values:
                raise ValueError(
                    f"Evaluation CSV row {line_number} has empty required fields: "
                    + ", ".join(missing_values)
                )

            item = {
                column: str(row.get(column) or "").strip()
                for column in EVALUATION_DATASET_COLUMNS
            }
            query_key = re.sub(r"\s+", " ", item["query"]).casefold()
            if query_key in seen_queries:
                raise ValueError(
                    f"Duplicate evaluation question at row {line_number}: "
                    f"{item['query']!r}"
                )
            seen_queries.add(query_key)

            document_id = item["source_document_id"]
            chunk_id = item["source_chunk_id"]
            if not re.fullmatch(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                document_id,
            ):
                raise ValueError(
                    f"Invalid source_document_id at row {line_number}: "
                    f"{document_id!r}"
                )
            if not chunk_id.startswith(f"{document_id}:"):
                raise ValueError(
                    f"source_chunk_id does not belong to source_document_id "
                    f"at row {line_number}: {chunk_id!r}"
                )
            evidence = parse_parallel_evidence(
                item,
                context=f"row {line_number} ({item['query']!r})",
                deduplicate_pairs=True,
            )
            item["expected_context_snippets_json"] = serialize_string_array(
                evidence.snippets
            )
            item["source_chunk_ids_json"] = serialize_string_array(
                evidence.chunk_ids
            )
            try:
                aliases = parse_json_string_array(
                    item.get("accepted_answers_json"),
                    field_name="accepted_answers_json",
                    strict=True,
                    deduplicate=True,
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid accepted_answers_json for {item['query']!r}: "
                    f"{error}"
                ) from error
            item["accepted_answers_json"] = serialize_string_array(aliases)
            rows.append(item)

    with engine.begin() as connection:
        if rows:
            result = connection.execute(
                text(f"""
                    INSERT INTO {EVALUATION_DATASET_TABLE} (
                        domain,
                        query,
                        expected_answer_text,
                        expected_context_snippet,
                        expected_context_snippets_json,
                        accepted_answers_json,
                        source_title,
                        source_url,
                        source_document_id,
                        source_chunk_id,
                        source_chunk_ids_json
                    )
                    VALUES (
                        :domain,
                        :query,
                        :expected_answer_text,
                        :expected_context_snippet,
                        CAST(:expected_context_snippets_json AS jsonb),
                        CAST(:accepted_answers_json AS jsonb),
                        :source_title,
                        :source_url,
                        :source_document_id,
                        :source_chunk_id,
                        CAST(:source_chunk_ids_json AS jsonb)
                    )
                    ON CONFLICT (query) DO UPDATE SET
                        domain = EXCLUDED.domain,
                        expected_answer_text = EXCLUDED.expected_answer_text,
                        expected_context_snippet = EXCLUDED.expected_context_snippet,
                        expected_context_snippets_json = EXCLUDED.expected_context_snippets_json,
                        accepted_answers_json = EXCLUDED.accepted_answers_json,
                        source_title = EXCLUDED.source_title,
                        source_url = EXCLUDED.source_url,
                        source_document_id = EXCLUDED.source_document_id,
                        source_chunk_id = EXCLUDED.source_chunk_id,
                        source_chunk_ids_json = EXCLUDED.source_chunk_ids_json
                    WHERE (
                        {EVALUATION_DATASET_TABLE}.domain,
                        {EVALUATION_DATASET_TABLE}.expected_answer_text,
                        {EVALUATION_DATASET_TABLE}.expected_context_snippet,
                        {EVALUATION_DATASET_TABLE}.expected_context_snippets_json,
                        {EVALUATION_DATASET_TABLE}.accepted_answers_json,
                        {EVALUATION_DATASET_TABLE}.source_title,
                        {EVALUATION_DATASET_TABLE}.source_url,
                        {EVALUATION_DATASET_TABLE}.source_document_id,
                        {EVALUATION_DATASET_TABLE}.source_chunk_id,
                        {EVALUATION_DATASET_TABLE}.source_chunk_ids_json
                    ) IS DISTINCT FROM (
                        EXCLUDED.domain,
                        EXCLUDED.expected_answer_text,
                        EXCLUDED.expected_context_snippet,
                        EXCLUDED.expected_context_snippets_json,
                        EXCLUDED.accepted_answers_json,
                        EXCLUDED.source_title,
                        EXCLUDED.source_url,
                        EXCLUDED.source_document_id,
                        EXCLUDED.source_chunk_id,
                        EXCLUDED.source_chunk_ids_json
                    )
                """),
                rows,
            )
            delete_result = connection.execute(
                text(f"""
                    DELETE FROM {EVALUATION_DATASET_TABLE}
                    WHERE NOT (
                        query = ANY(CAST(:queries AS text[]))
                    )
                """),
                {"queries": [row["query"] for row in rows]},
            )
            return (
                int(result.rowcount or 0),
                int(delete_result.rowcount or 0),
                len(rows),
            )

    return 0, 0, 0


def clear_hkpl_chunks() -> int:
    """Clear HKPL chunks while preserving benchmark rows in the shared table."""
    table_name = VECTOR_TABLE_NAME
    with engine.begin() as connection:
        result = connection.execute(
            text(f"""
                DELETE FROM {table_name}
                WHERE COALESCE(metadata_->>'corpus_role', '') <> 'distractor'
                  AND COALESCE(metadata_->>'dataset', '')
                      NOT IN ('hotpotqa', 'webz_news')
            """),
        )
    return int(result.rowcount or 0)


def registered_documents_for_rebuild() -> list[dict]:
    """Return rebuildable registry rows, failing before vectors are cleared."""
    ensure_registry_schema()
    documents = list_documents()
    if not documents:
        raise RuntimeError(
            "Full rebuild aborted before clearing vectors because "
            "knowledge_documents contains no registered sources."
        )
    missing_sources = [
        document
        for document in documents
        if not (UPLOAD_DIR / document["stored_file_name"]).is_file()
    ]

    if missing_sources:
        details = "\n".join(
            "- "
            f"{document['document_id']} "
            f"{document['original_file_name']} "
            f"(expected {UPLOAD_DIR / document['stored_file_name']})"
            for document in missing_sources
        )
        raise RuntimeError(
            "Full rebuild aborted before clearing vectors because registered "
            f"source files are missing:\n{details}"
        )

    unreadable_sources: list[str] = []
    for index, document in enumerate(documents, start=1):
        document_id = str(document["document_id"])
        stored_path = UPLOAD_DIR / document["stored_file_name"]
        print(
            f"[{index}/{len(documents)}] Checking extraction for "
            f"{document['original_file_name']} ({document_id})"
        )
        try:
            if normalize_document_type(document.get("document_type")) == "skip":
                print("  Ready: discovery-only source (no chunks)")
                continue
            extracted = load_file(
                stored_path,
                document_id=document_id,
                original_file_name=document["original_file_name"],
                source_title=document.get("source_title") or "",
                source_url=document.get("source_url") or "",
                source_type=document.get("source_type") or "admin_upload",
                access_level=document.get("access_level") or "public",
                document_version=int(document["version"]),
                content_hash=document["content_hash"],
                ocr_languages=OCR_LANGUAGES,
                category=document.get("category"),
                language=document.get("language"),
                effective_date=str(document.get("effective_date") or ""),
                source_kind=(
                    document.get("source_kind")
                    or document.get("source_type")
                    or "upload"
                ),
                document_type=document.get("document_type") or "auto",
                classification_source=document.get("classification_source") or None,
            )
            if not extracted:
                raise ValueError("No readable content was extracted")
            nodes = chunk_documents(extracted)
            if not nodes:
                raise ValueError("No chunks were created")
            print(f"  Ready: {len(extracted)} sections, {len(nodes)} chunks")
        except Exception as error:
            unreadable_sources.append(
                f"{document_id} {document['original_file_name']}: {error}"
            )
            print(f"  NOT READY: {error}")

    if unreadable_sources:
        raise RuntimeError(
            "Full rebuild aborted before clearing vectors because registered "
            "sources could not be extracted and chunked:\n- "
            + "\n- ".join(unreadable_sources)
        )

    return documents


def rebuild_registered_documents(documents: list[dict]) -> tuple[int, list[str]]:
    completed = 0
    failures: list[str] = []

    for index, document in enumerate(documents, start=1):
        document_id = str(document["document_id"])
        print(
            f"[{index}/{len(documents)}] Reindexing "
            f"{document['original_file_name']} ({document_id})"
        )
        try:
            result = reindex_registered_document(document_id)
            completed += 1
            print(
                f"  Created {result['chunks_created']} chunks; "
                "source_kind="
                f"{document.get('source_kind') or document.get('source_type') or 'upload'}"
            )
        except Exception as error:
            failures.append(f"{document_id}: {error}")
            print(f"  FAILED: {error}")

    return completed, failures


def audit_knowledge_chunks() -> bool:
    ensure_hybrid_search_schema()
    table_name = VECTOR_TABLE_NAME
    with engine.connect() as connection:
        summary = connection.execute(
            text(f"""
                SELECT
                    COUNT(*) AS total_chunks,
                    COUNT(DISTINCT metadata_->>'kb_document_id') AS vector_documents,
                    COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_embeddings,
                    COUNT(*) FILTER (
                        WHERE embedding IS NOT NULL
                          AND vector_dims(embedding) <> :embed_dim
                    ) AS wrong_dimensions,
                    COUNT(*) FILTER (
                        WHERE COALESCE(metadata_->>'token_count', '0')::integer > 512
                    ) AS over_limit_chunks,
                    COUNT(*) FILTER (
                        WHERE COALESCE(trim(metadata_->>'evidence_text'), '') = ''
                    ) AS empty_evidence,
                    COUNT(*) FILTER (
                        WHERE COALESCE(metadata_->>'document_id', '') = ''
                           OR COALESCE(metadata_->>'kb_document_id', '') = ''
                           OR COALESCE(metadata_->>'chunk_id', '') = ''
                           OR COALESCE(metadata_->>'record_kind', '') = ''
                           OR COALESCE(metadata_->>'chunk_policy', '') = ''
                           OR COALESCE(metadata_->>'parent_record_id', '') = ''
                           OR COALESCE(metadata_->>'parser_version', '') = ''
                           OR COALESCE(metadata_->>'search_text', '') = ''
                           OR COALESCE(metadata_->>'token_count', '') = ''
                           OR COALESCE(
                               (metadata_->'locator')::jsonb,
                               '{{}}'::jsonb
                           ) = '{{}}'::jsonb
                           OR COALESCE(metadata_->>'document_version', '') = ''
                    ) AS missing_metadata
                FROM {table_name}
            """),
            {"embed_dim": EMBED_DIM},
        ).mappings().one()

        type_rows = connection.execute(
            text(f"""
                SELECT
                    COALESCE(metadata_->>'record_kind', '(missing)') AS record_kind,
                    COALESCE(metadata_->>'chunk_policy', '(missing)') AS policy,
                    COUNT(*) AS chunks,
                    ROUND(AVG(COALESCE(metadata_->>'token_count', '0')::integer), 1)
                        AS average_tokens,
                    ROUND(AVG(length(text)), 1) AS average_characters,
                    MIN(length(text)) AS minimum_characters,
                    MAX(length(text)) AS maximum_characters
                FROM {table_name}
                GROUP BY record_kind, policy
                ORDER BY record_kind, policy
            """)
        ).mappings().all()

        corpus_rows = connection.execute(
            text(f"""
                SELECT
                    COALESCE(NULLIF(metadata_->>'dataset', ''), 'hkpl') AS dataset,
                    COALESCE(NULLIF(metadata_->>'corpus_role', ''), 'primary') AS corpus_role,
                    COUNT(*) AS chunks
                FROM {table_name}
                GROUP BY dataset, corpus_role
                ORDER BY dataset, corpus_role
            """)
        ).mappings().all()

        registry_mismatches = connection.execute(
            text(f"""
                WITH actual AS (
                    SELECT metadata_->>'kb_document_id' AS document_id, COUNT(*) AS chunks
                    FROM {table_name}
                    WHERE COALESCE(metadata_->>'corpus_role', '') <> 'distractor'
                      AND COALESCE(metadata_->>'dataset', '')
                          NOT IN ('hotpotqa', 'webz_news')
                    GROUP BY metadata_->>'kb_document_id'
                )
                SELECT
                    documents.document_id::text AS document_id,
                    documents.source_title,
                    documents.chunk_count AS registered_chunks,
                    COALESCE(actual.chunks, 0) AS actual_chunks
                FROM knowledge_documents documents
                LEFT JOIN actual
                  ON actual.document_id = documents.document_id::text
                WHERE documents.status <> 'deleted'
                  AND documents.chunk_count <> COALESCE(actual.chunks, 0)
                ORDER BY documents.source_title
            """),
        ).mappings().all()

        stale_or_orphaned = connection.execute(
            text(f"""
                SELECT chunks.node_id
                FROM {table_name} chunks
                LEFT JOIN knowledge_documents documents
                  ON chunks.metadata_->>'kb_document_id' = documents.document_id::text
                 AND documents.status <> 'deleted'
                WHERE (
                    documents.document_id IS NULL
                    AND COALESCE(chunks.metadata_->>'corpus_role', '')
                        <> 'distractor'
                    AND COALESCE(chunks.metadata_->>'dataset', '')
                        NOT IN ('hotpotqa', 'webz_news')
                ) OR (
                    documents.document_id IS NOT NULL
                    AND chunks.metadata_->>'document_version'
                        <> documents.version::text
                )
                LIMIT 50
            """),
        ).scalars().all()

        locator_collisions = connection.execute(
            text(f"""
                SELECT
                    metadata_->>'source_version_id' AS source_version_id,
                    (metadata_->'locator')::jsonb AS locator,
                    metadata_->>'part_number' AS part_number,
                    COUNT(*) AS chunks
                FROM {table_name}
                GROUP BY source_version_id, locator, part_number
                HAVING COUNT(*) > 1
                ORDER BY chunks DESC
            """)
        ).mappings().all()

        duplicate_groups = connection.execute(
            text(f"""
                SELECT
                    md5(metadata_->>'evidence_text') AS content_hash,
                    COUNT(*) AS copies,
                    COUNT(DISTINCT metadata_->>'kb_document_id') AS documents,
                    STRING_AGG(
                        DISTINCT COALESCE(NULLIF(metadata_->>'dataset', ''), 'hkpl'),
                        ', '
                    ) AS datasets,
                    LEFT(
                        MIN(regexp_replace(metadata_->>'evidence_text', '\\s+', ' ', 'g')),
                        180
                    ) AS preview
                FROM {table_name}
                GROUP BY md5(metadata_->>'evidence_text')
                HAVING COUNT(*) > 1
                ORDER BY copies DESC
                LIMIT 20
            """)
        ).mappings().all()

        typed_chunks = connection.execute(
            text(f"""
                SELECT node_id, text, metadata_
                FROM {table_name}
                WHERE metadata_->>'record_kind' IN ('faq', 'table')
            """)
        ).mappings().all()

    faq_issues: list[dict] = []
    table_header_issues: list[str] = []

    for row in typed_chunks:
        node_id = row["node_id"]
        metadata = row["metadata_"] or {}
        evidence = metadata.get("evidence_text") or ""
        record_kind = metadata.get("record_kind")

        if record_kind == "faq" and metadata.get("structural_kind") == "faq_pair":
            question = metadata.get("question") or ""
            answer = metadata.get("answer_text") or evidence[len(question):].strip()
            if not question or not answer or not evidence.startswith(question):
                faq_issues.append({
                    "node_id": node_id,
                    "reason": "missing or detached question/answer evidence",
                    "title": metadata.get("source_title", ""),
                    "url": metadata.get("source_url") or metadata.get("url", ""),
                    "preview": re.sub(r"\s+", " ", evidence).strip()[:240],
                })
        elif (
            record_kind == "table"
            and metadata.get("chunk_policy") == "oversized_leaf"
            and not (metadata.get("table_header") or metadata.get("repeat_context"))
        ):
            table_header_issues.append(node_id)

    checks = {
        "missing embeddings": int(summary["missing_embeddings"]),
        "wrong embedding dimensions": int(summary["wrong_dimensions"]),
        "chunks over 512 tokens": int(summary["over_limit_chunks"]),
        "chunks with empty evidence": int(summary["empty_evidence"]),
        "chunks missing required metadata": int(summary["missing_metadata"]),
        "registry chunk-count mismatches": len(registry_mismatches),
        "stale or orphaned chunks": len(stale_or_orphaned),
        "source/version/locator/part collisions": len(locator_collisions),
        "FAQ pairing issues": len(faq_issues),
        "headerless split tables": len(table_header_issues),
    }

    print("=" * 80)
    print("Shared Knowledge Chunk Audit")
    print("=" * 80)
    print(f"Total chunks       : {summary['total_chunks']}")
    print(f"Vector documents   : {summary['vector_documents']}")
    print(f"Embedding dimension: {EMBED_DIM}")
    print("\nCorpus distribution:")
    for row in corpus_rows:
        print(
            f"- {row['dataset']}/{row['corpus_role']}: "
            f"{row['chunks']} chunks"
        )
    print("\nRecord kind and chunk policy distribution:")
    for row in type_rows:
        print(
            f"- {row['record_kind']}/{row['policy']}: "
            f"{row['chunks']} chunks, tokens avg={row['average_tokens']} "
            f"chars avg={row['average_characters']} "
            f"min={row['minimum_characters']} max={row['maximum_characters']}"
        )

    print("\nInvariant checks:")
    for label, count in checks.items():
        status = "PASS" if count == 0 else "FAIL"
        print(f"- {status}: {label} ({count})")

    if locator_collisions:
        print("\nLocator collisions:")
        for row in locator_collisions[:20]:
            print(
                f"- source_version={row['source_version_id']} locator={row['locator']} "
                f"part={row['part_number']} chunks={row['chunks']}"
            )
    if faq_issues:
        print("\nFAQ pairing issues:")
        for issue in faq_issues[:20]:
            print(
                f"- {issue['node_id']} reason={issue['reason']} "
                f"title={issue['title']} url={issue['url']}"
            )
            print(f"  preview={issue['preview']}")
    if table_header_issues:
        print("\nHeaderless split table chunks:")
        for node_id in table_header_issues[:20]:
            print(f"- {node_id}")
    if duplicate_groups:
        print("\nReview warning: exact duplicate chunk text exists:")
        for row in duplicate_groups:
            print(
                f"- hash={row['content_hash']} copies={row['copies']} "
                f"documents={row['documents']} datasets={row['datasets']}"
            )
            print(f"  preview={row['preview']}")

    passed = all(count == 0 for count in checks.values())
    print("\nResult:", "PASSED" if passed else "FAILED")
    print(
        "Exact duplicates are warnings because identical official text can "
        "legitimately appear in different sources."
    )
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild registered knowledge, synchronize evaluations, or audit chunks."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Synchronize evaluation_dataset.csv without changing knowledge chunks.",
    )
    mode.add_argument(
        "--rebuild-all",
        action="store_true",
        help=(
            "Clear HKPL primary chunks and rebuild every non-deleted document "
            "registered in knowledge_documents, including saved crawler HTML."
        ),
    )
    mode.add_argument(
        "--check-rebuild",
        action="store_true",
        help="Verify all rebuild source files without changing the database.",
    )
    mode.add_argument(
        "--audit-chunks",
        action="store_true",
        help="Audit every stored knowledge chunk without changing the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_dataset_path = os.getenv(
        "EVALUATION_DATASET_PATH",
        "/app/data/evaluation_dataset.csv",
    )

    if args.audit_chunks:
        if not audit_knowledge_chunks():
            raise SystemExit(1)
        return

    if args.check_rebuild:
        registered_documents = registered_documents_for_rebuild()
        source_counts: dict[str, int] = {}
        for document in registered_documents:
            source_kind = (
                document.get("source_kind")
                or document.get("source_type")
                or "upload"
            )
            source_counts[source_kind] = source_counts.get(source_kind, 0) + 1

        print(f"Registered documents ready: {len(registered_documents)}")
        for source_kind, count in sorted(source_counts.items()):
            print(f"- {source_kind}: {count}")
        print("Preflight passed. No vectors or registry rows were changed.")
        return

    rebuild_all_from_env = (
        os.getenv(
            "REBUILD_ALL",
            "false",
        ).lower()
        == "true"
    )
    rebuild_all = args.rebuild_all or (
        rebuild_all_from_env
        and not args.evaluation_only
    )

    registered_documents: list[dict] = []
    if rebuild_all:
        ensure_corpus_writable("rebuild registered knowledge chunks")
        registered_documents = registered_documents_for_rebuild()
        print(
            "Full rebuild preflight passed: "
            f"{len(registered_documents)} registered documents are available."
        )
        removed_chunks = clear_hkpl_chunks()
        print(
            f"Removed {removed_chunks} HKPL chunks from {VECTOR_TABLE_NAME}; "
            "distractor corpus chunks were preserved."
        )

    if rebuild_all:
        completed, failures = rebuild_registered_documents(
            registered_documents
        )
        print(
            f"Rebuilt {completed}/{len(registered_documents)} registered documents."
        )
        if failures:
            raise RuntimeError(
                "Full rebuild finished with failures:\n- "
                + "\n- ".join(failures)
            )
        print(
            "Evaluation rows were not synchronized because rebuilt chunk IDs "
            "include new document versions. Regenerate the evaluation dataset "
            "from the rebuilt knowledge base before evaluating."
        )

    if not rebuild_all:
        changed_rows, deleted_rows, csv_rows = ingest_evaluation_dataset(
            evaluation_dataset_path
        )
        print(
            f"Synchronized {csv_rows} CSV rows to {EVALUATION_DATASET_TABLE}: "
            f"inserted or updated {changed_rows}, deleted {deleted_rows}; "
            "unchanged rows were skipped"
        )


if __name__ == "__main__":
    main()
