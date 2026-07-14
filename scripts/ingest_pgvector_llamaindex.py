#!/usr/bin/env python3

import argparse
import csv
import os
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
        print(
            f"Clearing data_{VECTOR_TABLE} before rebuilding all knowledge."
        )
        vector_store.clear()

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
