#!/usr/bin/env python3
"""Preview document classification and chunks without creating embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The application normally runs under /app in Docker. This preview command is
# also intended to run directly from a server checkout without root access.
os.environ.setdefault("UPLOAD_DIR", str(PROJECT_ROOT / "uploads"))
os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(PROJECT_ROOT / "models" / "docling"))
os.environ.setdefault("DOCLING_JSON_DIR", str(PROJECT_ROOT / "storage" / "docling"))
os.environ.setdefault(
    "EMBEDDING_TOKENIZER_PATH",
    str(PROJECT_ROOT / "models" / "qwen3-embedding"),
)

from src.infrastructure.db import engine
from src.ingestion.chunking import chunk_documents
from src.ingestion.classification import MAX_BATCH_ITEMS, SAMPLE_CHARACTERS, classify_batch_items_sync
from src.ingestion.readers import load_file
from src.ingestion.registry import list_documents
from src.ingestion.service import OCR_LANGUAGES, UPLOAD_DIR


DOCUMENT_TABLE = "ingestion_preview_documents"
CHUNK_TABLE = "ingestion_preview_chunks"


def ensure_preview_schema() -> None:
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {DOCUMENT_TABLE} (
                run_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source_title TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                classifier_sample TEXT NOT NULL DEFAULT '',
                document_type TEXT,
                classification_source TEXT NOT NULL DEFAULT 'llm',
                status TEXT NOT NULL,
                section_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, document_id)
            )
        """))
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {CHUNK_TABLE} (
                run_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                record_kind TEXT NOT NULL,
                chunk_policy TEXT NOT NULL,
                structure_path JSONB NOT NULL,
                locator JSONB NOT NULL,
                token_count INTEGER NOT NULL,
                evidence_text TEXT NOT NULL,
                search_text TEXT NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, document_id, chunk_id)
            )
        """))


def extract_for_classification(path: Path, record: dict) -> tuple[str, list]:
    documents = load_file(
        path,
        document_id=str(record["document_id"]),
        original_file_name=record["original_file_name"],
        source_title=record.get("source_title") or "",
        source_url=record.get("source_url") or "",
        source_type=record.get("source_type") or "preview",
        access_level=record.get("access_level") or "public",
        document_version=int(record.get("version") or 1),
        content_hash=record.get("content_hash"),
        ocr_languages=OCR_LANGUAGES,
        category=record.get("category"),
        language=record.get("language"),
        effective_date=str(record.get("effective_date") or ""),
        source_kind=record.get("source_kind") or record.get("source_type") or "preview",
        document_type="auto",
        classification_source="llm",
    )
    sample = "\n\n".join(
        str(document.metadata.get("evidence_text") or document.get_content())
        for document in documents
    )[:SAMPLE_CHARACTERS]
    if not sample.strip():
        raise ValueError("No content was extracted for classification.")
    return sample, documents


def save_document_preview(run_id: str, record: dict, sample: str, **values: object) -> None:
    payload = {
        "run_id": run_id,
        "document_id": str(record["document_id"]),
        "source_title": record.get("source_title") or record["original_file_name"],
        "source_url": record.get("source_url") or "",
        "file_name": record["original_file_name"],
        "file_type": record.get("file_type") or Path(record["stored_file_name"]).suffix.lstrip("."),
        "classifier_sample": sample,
        "document_type": values.get("document_type"),
        "status": values.get("status", "pending"),
        "section_count": values.get("section_count", 0),
        "chunk_count": values.get("chunk_count", 0),
        "error_message": values.get("error_message"),
    }
    with engine.begin() as connection:
        connection.execute(text(f"""
            INSERT INTO {DOCUMENT_TABLE} (
                run_id, document_id, source_title, source_url, file_name,
                file_type, classifier_sample, document_type, status,
                section_count, chunk_count, error_message
            ) VALUES (
                :run_id, :document_id, :source_title, :source_url, :file_name,
                :file_type, :classifier_sample, :document_type, :status,
                :section_count, :chunk_count, :error_message
            )
            ON CONFLICT (run_id, document_id) DO UPDATE SET
                document_type = EXCLUDED.document_type,
                status = EXCLUDED.status,
                section_count = EXCLUDED.section_count,
                chunk_count = EXCLUDED.chunk_count,
                error_message = EXCLUDED.error_message
        """), payload)


def save_chunks(run_id: str, document_id: str, nodes: list) -> None:
    rows = []
    for ordinal, node in enumerate(nodes):
        metadata = dict(node.metadata or {})
        rows.append({
            "run_id": run_id,
            "document_id": document_id,
            "chunk_id": node.node_id,
            "ordinal": ordinal,
            "record_kind": metadata.get("record_kind") or "prose",
            "chunk_policy": metadata.get("chunk_policy") or "fallback",
            "structure_path": json.dumps(metadata.get("structure_path") or [], ensure_ascii=False),
            "locator": json.dumps(metadata.get("locator") or {}, ensure_ascii=False),
            "token_count": int(metadata.get("token_count") or 0),
            "evidence_text": metadata.get("evidence_text") or "",
            "search_text": metadata.get("search_text") or node.get_content(),
            "metadata": json.dumps(metadata, ensure_ascii=False, default=str),
        })
    if rows:
        with engine.begin() as connection:
            connection.execute(text(f"""
                INSERT INTO {CHUNK_TABLE} (
                    run_id, document_id, chunk_id, ordinal, record_kind,
                    chunk_policy, structure_path, locator, token_count,
                    evidence_text, search_text, metadata
                ) VALUES (
                    :run_id, :document_id, :chunk_id, :ordinal, :record_kind,
                    :chunk_policy, CAST(:structure_path AS JSONB),
                    CAST(:locator AS JSONB), :token_count, :evidence_text,
                    :search_text, CAST(:metadata AS JSONB)
                )
            """), rows)


def preview_record(
    run_id: str,
    record: dict,
    sample: str,
    document_type: str,
) -> None:
    document_id = str(record["document_id"])
    if document_type == "skip":
        save_document_preview(
            run_id, record, sample, document_type=document_type,
            status="skipped", section_count=0, chunk_count=0,
        )
        return

    path = UPLOAD_DIR / record["stored_file_name"]
    documents = load_file(
        path,
        document_id=document_id,
        original_file_name=record["original_file_name"],
        source_title=record.get("source_title") or "",
        source_url=record.get("source_url") or "",
        source_type=record.get("source_type") or "preview",
        access_level=record.get("access_level") or "public",
        document_version=int(record.get("version") or 1),
        content_hash=record.get("content_hash"),
        ocr_languages=OCR_LANGUAGES,
        category=record.get("category"),
        language=record.get("language"),
        effective_date=str(record.get("effective_date") or ""),
        source_kind=record.get("source_kind") or record.get("source_type") or "preview",
        document_type=document_type,
        classification_source="llm",
    )
    nodes = chunk_documents(documents)
    save_chunks(run_id, document_id, nodes)
    save_document_preview(
        run_id, record, sample, document_type=document_type,
        status="completed", section_count=len(documents), chunk_count=len(nodes),
    )


def run_preview(
    *,
    run_id: str,
    limit: int | None,
    crawler_only: bool,
    classify_only: bool,
) -> None:
    ensure_preview_schema()
    records = [
        record for record in list_documents()
        if not crawler_only or record.get("source_type") == "crawler"
    ]
    records = records[:limit]
    if not records:
        raise RuntimeError("No matching registered documents were found.")

    print(f"Preview run: {run_id}; documents: {len(records)}")
    for start in range(0, len(records), MAX_BATCH_ITEMS):
        batch = records[start:start + MAX_BATCH_ITEMS]
        prepared: list[tuple[dict, str]] = []
        for record in batch:
            path = UPLOAD_DIR / record["stored_file_name"]
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                sample, _documents = extract_for_classification(path, record)
                prepared.append((record, sample))
                save_document_preview(run_id, record, sample, status="classifying")
            except Exception as error:
                save_document_preview(
                    run_id, record, "", status="failed",
                    error_message=str(error)[:2000],
                )

        if not prepared:
            continue
        decisions = classify_batch_items_sync([{
            "id": str(record["document_id"]),
            "title": record.get("source_title") or record["original_file_name"],
            "file_type": record.get("file_type") or "",
            "text": sample,
        } for record, sample in prepared])

        for record, sample in prepared:
            document_id = str(record["document_id"])
            document_type = decisions[document_id]["document_type"]
            try:
                if classify_only:
                    save_document_preview(
                        run_id, record, sample, document_type=document_type,
                        status="classified", section_count=0, chunk_count=0,
                    )
                    print(f"{document_type:6} {record.get('source_url') or record['original_file_name']}")
                    continue
                preview_record(run_id, record, sample, document_type)
                print(f"{document_type:6} {record.get('source_url') or record['original_file_name']}")
            except Exception as error:
                save_document_preview(
                    run_id, record, sample, document_type=document_type,
                    status="failed", error_message=str(error)[:2000],
                )
                print(f"FAILED {record['original_file_name']}: {error}")

    print(f"Preview complete. Inspect run_id={run_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify registered sources and store pre-embedding chunk previews.",
    )
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6])
    parser.add_argument("--limit", type=int, help="Preview at most this many sources.")
    parser.add_argument("--all-sources", action="store_true", help="Include uploads as well as crawler sources.")
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Store 9B classification results without extracting or chunking.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run_preview(
        run_id=arguments.run_id,
        limit=arguments.limit,
        crawler_only=not arguments.all_sources,
        classify_only=arguments.classify_only,
    )
