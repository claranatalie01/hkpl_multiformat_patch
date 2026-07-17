#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from sqlalchemy import text

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.embedding import (
    embed_model,
)
from src.infrastructure.vector_store import (
    EMBED_DIM,
    VECTOR_TABLE,
    vector_store,
)
from src.infrastructure.db import engine
from src.ingestion.chunking import (
    chunk_documents,
)
from src.ingestion.registry import (
    ensure_registry_schema,
    list_documents,
)
from src.ingestion.readers import file_content_hash, load_file
from src.ingestion.service import (
    OCR_LANGUAGES,
    UPLOAD_DIR,
    reindex_registered_document,
)


EVALUATION_DATASET_TABLE = os.getenv(
    "EVALUATION_DATASET_TABLE",
    "evaluation_dataset",
)
EVALUATION_DATASET_COLUMNS = [
    "domain",
    "query",
    "expected_answer_text",
    "expected_context_snippet",
    "source_title",
    "source_url",
    "source_type",
    "source_document_id",
    "source_chunk_id",
]


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
                    source_title TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT '',
                    source_document_id TEXT NOT NULL DEFAULT '',
                    source_chunk_id TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
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


def ingest_evaluation_dataset(csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        print(f"Evaluation dataset not found, skipping: {path}")
        return 0

    create_evaluation_dataset_table()

    with path.open("r", newline="", encoding="utf-8") as file:
        rows = [
            {
                column: (row.get(column) or "")
                for column in EVALUATION_DATASET_COLUMNS
            }
            for row in csv.DictReader(file)
            if row.get("query")
        ]

    with engine.begin() as connection:
        if rows:
            result = connection.execute(
                text(f"""
                    INSERT INTO {EVALUATION_DATASET_TABLE} (
                        domain,
                        query,
                        expected_answer_text,
                        expected_context_snippet,
                        source_title,
                        source_url,
                        source_type,
                        source_document_id,
                        source_chunk_id
                    )
                    VALUES (
                        :domain,
                        :query,
                        :expected_answer_text,
                        :expected_context_snippet,
                        :source_title,
                        :source_url,
                        :source_type,
                        :source_document_id,
                        :source_chunk_id
                    )
                    ON CONFLICT (query) DO UPDATE SET
                        domain = EXCLUDED.domain,
                        expected_answer_text = EXCLUDED.expected_answer_text,
                        expected_context_snippet = EXCLUDED.expected_context_snippet,
                        source_title = EXCLUDED.source_title,
                        source_url = EXCLUDED.source_url,
                        source_type = EXCLUDED.source_type,
                        source_document_id = EXCLUDED.source_document_id,
                        source_chunk_id = EXCLUDED.source_chunk_id
                    WHERE (
                        {EVALUATION_DATASET_TABLE}.domain,
                        {EVALUATION_DATASET_TABLE}.expected_answer_text,
                        {EVALUATION_DATASET_TABLE}.expected_context_snippet,
                        {EVALUATION_DATASET_TABLE}.source_title,
                        {EVALUATION_DATASET_TABLE}.source_url,
                        {EVALUATION_DATASET_TABLE}.source_type,
                        {EVALUATION_DATASET_TABLE}.source_document_id,
                        {EVALUATION_DATASET_TABLE}.source_chunk_id
                    ) IS DISTINCT FROM (
                        EXCLUDED.domain,
                        EXCLUDED.expected_answer_text,
                        EXCLUDED.expected_context_snippet,
                        EXCLUDED.source_title,
                        EXCLUDED.source_url,
                        EXCLUDED.source_type,
                        EXCLUDED.source_document_id,
                        EXCLUDED.source_chunk_id
                    )
                """),
                rows,
            )
            return int(result.rowcount or 0)

    return 0


def load_faq_documents(
    csv_path: str,
) -> list[Document]:
    documents: list[Document] = []

    with open(
        csv_path,
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row_index, row in enumerate(reader):
            question = row.get(
                "query",
                "",
            ).strip()
            answer = row.get(
                "expected_answer_text",
                "",
            ).strip()

            if not question or not answer:
                continue

            source_url = row.get(
                "source_url",
                "https://www.hkpl.gov.hk/en/ask-a-librarian/faq.html",
            ).strip()
            row_id = row.get(
                "source_row_id",
                str(row_index),
            ).strip()

            document_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{source_url}#faq-{row_id}",
                )
            )

            document = Document(
                text=(
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                ),
                metadata={
                    "document_id": document_id,
                    "original_file_name": (
                        Path(csv_path).name
                    ),
                    "file_name": (
                        Path(csv_path).name
                    ),
                    "file_type": "csv",
                    "source_title": row.get(
                        "source_title",
                        "HKPL Ask a Librarian FAQ",
                    ).strip(),
                    "source": row.get(
                        "source_title",
                        "HKPL Ask a Librarian FAQ",
                    ).strip(),
                    "source_url": source_url,
                    "url": source_url,
                    "source_type": row.get(
                        "source_type",
                        "official_website",
                    ).strip(),
                    "access_level": "public",
                    "document_version": 1,
                    "domain": row.get(
                        "domain",
                        "",
                    ).strip(),
                    "question": question,
                    "snippet": row.get(
                        "expected_context_snippet",
                        "",
                    ).strip(),
                    "row_id": row_id,
                    "row_number": row_index + 2,
                    "section_index": row_index,
                    "chunk_strategy": "atomic",
                },
            )
            document.id_ = (
                f"{document_id}:v1:section:0"
            )
            documents.append(document)

    return documents



def delete_existing_faq_chunks() -> int:
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="source_title",
                value="HKPL Ask a Librarian FAQ",
                operator=FilterOperator.EQ,
            )
        ]
    )
    nodes = vector_store.get_nodes(filters=filters)
    if not nodes:
        return 0

    vector_store.delete_nodes(
        node_ids=[node.node_id for node in nodes]
    )
    return len(nodes)


def clear_hkpl_chunks() -> int:
    """Clear HKPL chunks while preserving benchmark rows in the shared table."""
    table_name = f"data_{VECTOR_TABLE}"
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
    table_name = f"data_{VECTOR_TABLE}"
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
                    COUNT(*) FILTER (WHERE length(trim(text)) < 50) AS short_chunks,
                    COUNT(*) FILTER (
                        WHERE COALESCE(metadata_->>'document_id', '') = ''
                           OR COALESCE(metadata_->>'kb_document_id', '') = ''
                           OR COALESCE(metadata_->>'chunk_id', '') = ''
                           OR COALESCE(metadata_->>'document_type', '') = ''
                           OR COALESCE(metadata_->>'chunk_strategy', '') = ''
                           OR COALESCE(metadata_->>'document_version', '') = ''
                    ) AS missing_metadata
                FROM {table_name}
            """),
            {"embed_dim": EMBED_DIM},
        ).mappings().one()

        type_rows = connection.execute(
            text(f"""
                SELECT
                    COALESCE(metadata_->>'document_type', '(missing)') AS document_type,
                    COALESCE(metadata_->>'chunk_strategy', '(missing)') AS strategy,
                    COUNT(*) AS chunks,
                    ROUND(AVG(length(text)), 1) AS average_characters,
                    MIN(length(text)) AS minimum_characters,
                    MAX(length(text)) AS maximum_characters
                FROM {table_name}
                GROUP BY document_type, strategy
                ORDER BY document_type, strategy
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

        split_records = connection.execute(
            text(f"""
                SELECT
                    metadata_->>'kb_document_id' AS document_id,
                    metadata_->>'original_file_name' AS file_name,
                    metadata_->>'section_index' AS section_index,
                    COUNT(*) AS chunks
                FROM {table_name}
                WHERE metadata_->>'document_type' = 'record_based'
                GROUP BY document_id, file_name, section_index
                HAVING COUNT(*) > 1
                ORDER BY chunks DESC
            """)
        ).mappings().all()

        duplicate_groups = connection.execute(
            text(f"""
                SELECT md5(text) AS content_hash, COUNT(*) AS copies
                FROM {table_name}
                GROUP BY md5(text)
                HAVING COUNT(*) > 1
                ORDER BY copies DESC
                LIMIT 20
            """)
        ).mappings().all()

        typed_chunks = connection.execute(
            text(f"""
                SELECT node_id, text, metadata_
                FROM {table_name}
                WHERE metadata_->>'document_type' IN ('faq', 'directory', 'announcement')
            """)
        ).mappings().all()

    faq_issues: list[str] = []
    directory_issues: list[str] = []
    announcement_issues: list[str] = []
    dated_entry = re.compile(r"(?m)^\(?\s*\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*\)?")

    for row in typed_chunks:
        node_id = row["node_id"]
        content = row["text"] or ""
        metadata = row["metadata_"] or {}
        document_type = metadata.get("document_type")

        if document_type == "faq":
            questions = set(re.findall(r"(?im)^\s*Q(\d+)\s*[:.)]", content))
            answers = set(re.findall(r"(?im)^\s*A(\d+)\s*[:.)]", content))
            labelled_question = bool(re.search(r"(?im)^\s*question\s*:", content))
            labelled_answer = bool(re.search(r"(?im)^\s*answer\s*:", content))
            if (not questions and not labelled_question) or (
                questions and not questions.issubset(answers)
            ) or (
                labelled_question != labelled_answer
            ):
                faq_issues.append(node_id)
        elif document_type == "directory":
            if not metadata.get("library_name"):
                directory_issues.append(node_id)
        elif (
            document_type == "announcement"
            and metadata.get("announcement_entry_index") is not None
            and not dated_entry.search(content)
        ):
            announcement_issues.append(node_id)

    checks = {
        "missing embeddings": int(summary["missing_embeddings"]),
        "wrong embedding dimensions": int(summary["wrong_dimensions"]),
        "chunks shorter than 50 characters": int(summary["short_chunks"]),
        "chunks missing required metadata": int(summary["missing_metadata"]),
        "registry chunk-count mismatches": len(registry_mismatches),
        "stale or orphaned chunks": len(stale_or_orphaned),
        "record rows split across chunks": len(split_records),
        "FAQ pairing issues": len(faq_issues),
        "directory chunks missing library metadata": len(directory_issues),
        "dated announcement chunks missing a date": len(announcement_issues),
    }

    print("=" * 80)
    print("HKPL Knowledge Chunk Audit")
    print("=" * 80)
    print(f"Total chunks       : {summary['total_chunks']}")
    print(f"Vector documents   : {summary['vector_documents']}")
    print(f"Embedding dimension: {EMBED_DIM}")
    print("\nDocument type and strategy distribution:")
    for row in type_rows:
        print(
            f"- {row['document_type']}/{row['strategy']}: "
            f"{row['chunks']} chunks, chars avg={row['average_characters']} "
            f"min={row['minimum_characters']} max={row['maximum_characters']}"
        )

    print("\nInvariant checks:")
    for label, count in checks.items():
        status = "PASS" if count == 0 else "FAIL"
        print(f"- {status}: {label} ({count})")

    if split_records:
        print("\nSplit record rows:")
        for row in split_records[:20]:
            print(
                f"- {row['file_name']} section={row['section_index']} "
                f"chunks={row['chunks']} document={row['document_id']}"
            )
    if duplicate_groups:
        print("\nReview warning: exact duplicate chunk text exists:")
        for row in duplicate_groups:
            print(f"- hash={row['content_hash']} copies={row['copies']}")

    passed = all(count == 0 for count in checks.values())
    print("\nResult:", "PASSED" if passed else "FAILED")
    print(
        "Exact duplicates are warnings because identical official text can "
        "legitimately appear in different sources."
    )
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest FAQ knowledge and/or synchronize the evaluation dataset.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Synchronize evaluation_dataset.csv without changing knowledge chunks.",
    )
    mode.add_argument(
        "--faq-only",
        action="store_true",
        help="Rebuild the FAQ CSV knowledge chunks without importing evaluation rows.",
    )
    mode.add_argument(
        "--rebuild-all",
        action="store_true",
        help=(
            "Clear and rebuild FAQ chunks plus every non-deleted document in "
            "knowledge_documents, including saved crawler HTML."
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
    data_path = os.getenv(
        "DATA_PATH",
        "/app/data/hkpl_faq_clean.csv",
    )
    evaluation_dataset_path = os.getenv(
        "EVALUATION_DATASET_PATH",
        "/app/data/evaluation_dataset.csv",
    )

    if args.audit_chunks:
        if not audit_knowledge_chunks():
            raise SystemExit(1)
        return

    if args.check_rebuild:
        if not Path(data_path).is_file():
            raise FileNotFoundError(
                f"FAQ source file is missing: {data_path}"
            )
        registered_documents = registered_documents_for_rebuild()
        source_counts: dict[str, int] = {}
        for document in registered_documents:
            source_kind = (
                document.get("source_kind")
                or document.get("source_type")
                or "upload"
            )
            source_counts[source_kind] = source_counts.get(source_kind, 0) + 1

        print(f"FAQ source: {data_path}")
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
        and not args.faq_only
    )

    registered_documents: list[dict] = []
    faq_is_registered = False
    if rebuild_all:
        if not Path(data_path).is_file():
            raise FileNotFoundError(
                f"FAQ source file is missing: {data_path}"
            )
        registered_documents = registered_documents_for_rebuild()
        faq_hash = file_content_hash(Path(data_path))
        faq_is_registered = any(
            document.get("content_hash") == faq_hash
            for document in registered_documents
        )
        print(
            f"Full rebuild preflight passed: FAQ source plus "
            f"{len(registered_documents)} registered documents are available."
        )
        removed_chunks = clear_hkpl_chunks()
        print(
            f"Removed {removed_chunks} HKPL chunks from data_{VECTOR_TABLE}; "
            "distractor corpus chunks were preserved."
        )

    if not args.evaluation_only and not (rebuild_all and faq_is_registered):
        removed = delete_existing_faq_chunks()
        print(f"Removed {removed} existing FAQ chunks")

        documents = load_faq_documents(
            data_path
        )
        print(
            f"Loaded {len(documents)} FAQ documents"
        )

        nodes = chunk_documents(
            documents
        )
        print(
            f"Created {len(nodes)} FAQ chunks"
        )

        storage_context = (
            StorageContext.from_defaults(
                vector_store=vector_store
            )
        )

        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        print(
            "Ingested FAQ data into "
            f"data_{VECTOR_TABLE}"
        )
    elif rebuild_all and faq_is_registered:
        print(
            "Skipped standalone FAQ ingestion because the same FAQ source is "
            "registered in knowledge_documents and will be rebuilt there."
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

    if not args.faq_only and not rebuild_all:
        evaluation_rows = ingest_evaluation_dataset(
            evaluation_dataset_path
        )
        print(
            f"Inserted or updated {evaluation_rows} evaluation rows in "
            f"{EVALUATION_DATASET_TABLE}; unchanged rows were skipped"
        )


if __name__ == "__main__":
    main()
