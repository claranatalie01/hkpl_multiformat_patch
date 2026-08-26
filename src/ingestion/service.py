"""Orchestrate registration, extraction, chunking, embedding, and replacement."""

import logging
import os
from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from sqlalchemy import text

from ..infrastructure.embedding import embed_model
from ..infrastructure.db import engine
from ..infrastructure.vector_store import (
    VECTOR_TABLE_NAME,
    ensure_hybrid_search_schema,
    vector_store,
)
from .chunking import chunk_documents
from .classification import (
    classify_batch_items_resilient_sync,
)
from .document_types import normalize_document_type, validate_document_type
from .readers import (
    SUPPORTED_EXTENSIONS,
    file_content_hash,
    load_file,
)
from .registry import (
    create_document,
    ensure_registry_schema,
    find_completed_duplicate,
    get_document,
    mark_deleted,
    prepare_replacement,
    prepare_reindex,
    restore_document,
    set_document_type,
    update_status,
)
from .write_guard import ensure_corpus_writable


logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", "/app/uploads")
)
OCR_LANGUAGES = os.getenv(
    "OCR_LANGUAGES",
    "eng+chi_tra+chi_sim",
)
KNOWLEDGE_TABLE = VECTOR_TABLE_NAME


def _load_record(record: dict, *, document_type: str | None = None):
    stored_path = UPLOAD_DIR / record["stored_file_name"]
    return load_file(
        stored_path,
        document_id=str(record["document_id"]),
        original_file_name=record["original_file_name"],
        source_title=record.get("source_title") or "",
        source_url=record.get("source_url") or "",
        source_type=record.get("source_type") or "admin_upload",
        access_level=record.get("access_level") or "public",
        document_version=int(record["version"]),
        content_hash=record["content_hash"],
        ocr_languages=OCR_LANGUAGES,
        category=record.get("category"),
        language=record.get("language"),
        effective_date=str(record.get("effective_date") or ""),
        source_kind=record.get("source_kind") or "upload",
        document_type=document_type or record.get("document_type") or "auto",
        classification_source=record.get("classification_source") or None,
    )


def delete_document_chunks(
    document_id: str,
) -> int:
    with engine.begin() as connection:
        result = connection.execute(
            text(f"""
                DELETE FROM {KNOWLEDGE_TABLE}
                WHERE metadata_->>'kb_document_id' = :document_id
            """),
            {"document_id": document_id},
        )
    return int(result.rowcount or 0)


def delete_old_versions(
    document_id: str,
    current_version: int,
) -> int:
    with engine.begin() as connection:
        result = connection.execute(
            text(f"""
                DELETE FROM {KNOWLEDGE_TABLE}
                WHERE metadata_->>'kb_document_id' = :document_id
                  AND metadata_->>'document_version'
                      IS DISTINCT FROM :current_version
            """),
            {
                "document_id": document_id,
                "current_version": str(current_version),
            },
        )
    return int(result.rowcount or 0)


def register_upload(
    *,
    stored_path: Path,
    original_file_name: str,
    mime_type: str = "",
    source_title: str = "",
    source_url: str = "",
    source_type: str = "admin_upload",
    access_level: str = "public",
    category: str | None = None,
    language: str | None = None,
    effective_date: str | None = None,
    source_kind: str = "upload",
    document_type: str = "auto",
    classification_source: str | None = None,
    replace_document_id: str | None = None,
) -> dict:
    """Register a new source or prepare an existing source for replacement."""
    ensure_corpus_writable("register or replace a document")
    ensure_registry_schema()

    extension = stored_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    content_hash = file_content_hash(stored_path)
    document_type = validate_document_type(document_type)
    classification_source = classification_source or (
        "fallback" if normalize_document_type(document_type) == "auto" else "librarian"
    )

    if replace_document_id is None:
        duplicate = find_completed_duplicate(
            content_hash,
            source_url=source_url,
            original_file_name=original_file_name,
        )
        if duplicate:
            return {
                "duplicate": True,
                "document": duplicate,
            }

        record = create_document(
            original_file_name=original_file_name,
            stored_file_name=stored_path.name,
            file_type=extension.lstrip("."),
            mime_type=mime_type,
            content_hash=content_hash,
            source_title=(
                source_title
                or Path(original_file_name).stem
            ),
            source_url=source_url,
            source_type=source_type,
            classification_source=classification_source,
            access_level=access_level,
            category=category,
            language=language,
            effective_date=effective_date,
            source_kind=source_kind,
            document_type=document_type,
        )
    else:
        existing = get_document(
            replace_document_id
        )
        if not existing:
            raise ValueError(
                "Document to replace was not found."
            )

        if (
            existing.get("content_hash")
            == content_hash
            and existing.get("status") == "completed"
            and (
                document_type == "auto"
                or (existing.get("document_type") or "auto") == document_type
            )
        ):
            return {
                "duplicate": True,
                "document": existing,
            }

        record = prepare_replacement(
            replace_document_id,
            original_file_name=original_file_name,
            stored_file_name=stored_path.name,
            file_type=extension.lstrip("."),
            mime_type=mime_type,
            content_hash=content_hash,
            source_title=(
                source_title
                or Path(original_file_name).stem
            ),
            source_url=source_url,
            source_type=source_type,
            classification_source=classification_source,
            access_level=access_level,
            category=category,
            language=language,
            effective_date=effective_date,
            source_kind=source_kind,
            document_type=document_type,
        )

        if record is None:
            raise ValueError(
                "Document could not be prepared for replacement."
            )

    return {
        "duplicate": False,
        "document": record,
    }


def process_registered_document(
    document_id: str,
) -> dict:
    """Extract, chunk, embed, and persist one registered source."""
    ensure_corpus_writable("extract, chunk, or embed a document")
    ensure_registry_schema()

    record = get_document(document_id)
    if not record:
        raise ValueError(
            f"Document {document_id} was not found."
        )

    try:
        if normalize_document_type(record.get("document_type")) == "auto":
            update_status(document_id, "extracting")
            classification_documents = _load_record(record, document_type="auto")
            if not classification_documents:
                raise ValueError("No readable content was extracted for classification.")
            classified = classify_batch_items_resilient_sync([{
                "id": document_id,
                "title": record.get("source_title") or record["original_file_name"],
                "source_url": record.get("source_url") or "",
                "file_type": record.get("file_type") or "",
                "text": "\n\n".join(
                    str(document.metadata.get("evidence_text") or document.get_content())
                    for document in classification_documents
                ),
            }])
            classification = classified[document_id]
            decision = classification["document_type"]
            classification_source = classification.get("classification_source", "llm")
            set_document_type(document_id, decision, classification_source)
            record = {
                **record,
                "document_type": decision,
                "classification_source": classification_source,
            }

        if normalize_document_type(record.get("document_type")) == "skip":
            removed_old_chunks = delete_document_chunks(document_id)
            update_status(document_id, "completed", chunk_count=0, error_message=None)
            return {
                "document_id": document_id,
                "status": "skipped",
                "file_name": record["original_file_name"],
                "document_version": int(record["version"]),
                "sections_extracted": 0,
                "chunks_created": 0,
                "old_chunks_removed": removed_old_chunks,
            }

        update_status(
            document_id,
            "extracting",
        )

        documents = _load_record(record)

        if not documents:
            raise ValueError(
                "No readable content was extracted."
            )

        update_status(
            document_id,
            "chunking",
        )
        nodes = chunk_documents(documents)

        if not nodes:
            raise ValueError(
                "No chunks were created."
            )

        update_status(
            document_id,
            "embedding",
        )

        ensure_hybrid_search_schema()
        storage_context = (
            StorageContext.from_defaults(
                vector_store=vector_store
            )
        )

        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=False,
        )

        removed_old_chunks = delete_old_versions(
            document_id,
            int(record["version"]),
        )

        update_status(
            document_id,
            "completed",
            chunk_count=len(nodes),
            error_message=None,
        )

        result = {
            "document_id": str(
                record["document_id"]
            ),
            "status": "completed",
            "file_name": record[
                "original_file_name"
            ],
            "document_version": int(
                record["version"]
            ),
            "sections_extracted": len(
                documents
            ),
            "chunks_created": len(nodes),
            "old_chunks_removed": (
                removed_old_chunks
            ),
        }

        logger.info(
            "Document ingestion completed: %s",
            result,
        )
        return result

    except Exception as error:
        logger.exception(
            "Document ingestion failed: %s",
            document_id,
        )
        update_status(
            document_id,
            "failed",
            error_message=str(error)[:2000],
        )
        raise


def process_registered_batch(document_ids: list[str]) -> list[dict]:
    """Classify an unlabelled batch once, persist decisions, then ingest it."""
    ensure_corpus_writable("classify and ingest a document batch")
    ensure_registry_schema()
    records: list[dict] = []
    items: list[dict] = []
    persisted_labels: dict[str, str] = {}
    unlabelled_ids: set[str] = set()
    classification_sources: dict[str, str] = {}

    try:
        for document_id in document_ids:
            record = get_document(document_id)
            if not record:
                raise ValueError(f"Document {document_id} was not found.")
            records.append(record)
            selected_type = normalize_document_type(record.get("document_type"))
            if selected_type != "auto":
                persisted_labels[document_id] = selected_type
                continue

            unlabelled_ids.add(document_id)
            update_status(document_id, "extracting")
            documents = _load_record(record, document_type="auto")
            if not documents:
                raise ValueError(f"No readable content was extracted for {document_id}.")

            items.append({
                "id": document_id,
                "title": record.get("source_title") or record["original_file_name"],
                "source_url": record.get("source_url") or "",
                "file_type": record.get("file_type") or "",
                "text": "\n\n".join(
                    str(document.metadata.get("evidence_text") or document.get_content())
                    for document in documents
                ),
            })

        classified = classify_batch_items_resilient_sync(items)
        classification_sources = {
            item_id: decision.get("classification_source", "llm")
            for item_id, decision in classified.items()
        }
        decisions = {
            **persisted_labels,
            **{
                item_id: decision["document_type"]
                for item_id, decision in classified.items()
            },
        }
        if set(decisions) != set(document_ids):
            raise ValueError("Batch classification did not produce one decision per document.")
    except Exception as error:
        for record in records:
            if str(record["document_id"]) not in unlabelled_ids:
                continue
            update_status(
                str(record["document_id"]),
                "failed",
                error_message=str(error)[:2000],
            )
        raise

    for document_id in unlabelled_ids:
        set_document_type(
            document_id,
            decisions[document_id],
            classification_sources.get(document_id, "llm"),
        )

    results: list[dict] = []
    for document_id in document_ids:
        try:
            results.append(process_registered_document(document_id))
        except Exception as error:
            results.append({
                "document_id": document_id,
                "status": "failed",
                "error": str(error)[:2000],
            })
    return results


def ingest_path_sync(
    path: Path,
    *,
    original_file_name: str | None = None,
    mime_type: str = "",
    source_title: str = "",
    source_url: str = "",
    source_type: str = "admin_upload",
    access_level: str = "public",
    category: str | None = None,
    language: str | None = None,
    effective_date: str | None = None,
    source_kind: str = "upload",
    document_type: str = "auto",
    classification_source: str | None = None,
    replace_document_id: str | None = None,
) -> dict:
    """Synchronously register and ingest one saved source path."""
    previous_record = (
        get_document(replace_document_id)
        if replace_document_id is not None
        else None
    )
    registration = register_upload(
        stored_path=path,
        original_file_name=(
            original_file_name
            or path.name
        ),
        mime_type=mime_type,
        source_title=source_title,
        source_url=source_url,
        source_type=source_type,
        access_level=access_level,
        replace_document_id=(
            replace_document_id
        ),
        category=category,
        language=language,
        effective_date=effective_date,
        source_kind=source_kind,
        document_type=document_type,
        classification_source=classification_source,
    )

    if registration["duplicate"]:
        return {
            "status": "duplicate",
            "document_id": str(
                registration["document"][
                    "document_id"
                ]
            ),
            "file_name": registration[
                "document"
            ]["original_file_name"],
        }

    document_id = str(
        registration["document"][
            "document_id"
        ]
    )
    try:
        return process_registered_document(document_id)
    except Exception:
        if previous_record is not None:
            delete_old_versions(document_id, int(previous_record["version"]))
            restore_document(previous_record)
        raise


def delete_registered_document(
    document_id: str,
    *,
    delete_file: bool = True,
) -> dict:
    ensure_corpus_writable("delete a document")
    record = get_document(document_id)
    if not record:
        raise ValueError(
            "Document was not found."
        )

    removed_chunks = delete_document_chunks(
        document_id
    )

    if delete_file:
        stored_path = (
            UPLOAD_DIR
            / record["stored_file_name"]
        )
        stored_path.unlink(
            missing_ok=True
        )

    mark_deleted(document_id)

    return {
        "document_id": document_id,
        "status": "deleted",
        "chunks_removed": removed_chunks,
    }


def reindex_registered_document(
    document_id: str,
    *,
    document_type: str | None = None,
) -> dict:
    ensure_corpus_writable("reindex a document")
    previous_record = get_document(document_id)
    if not previous_record:
        raise ValueError("Document was not found.")
    selected_type = (
        validate_document_type(document_type)
        if document_type is not None
        else None
    )
    record = prepare_reindex(
        document_id,
        selected_type,
        (
            "fallback"
            if selected_type == "auto"
            else "librarian"
            if selected_type is not None
            else None
        ),
    )
    if not record:
        raise ValueError("Document was not found.")
    try:
        return process_registered_document(document_id)
    except Exception:
        delete_old_versions(document_id, int(previous_record["version"]))
        restore_document(previous_record)
        raise
