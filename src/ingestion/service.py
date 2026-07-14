import logging
import os
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from ..infrastructure.embedding import embed_model
from ..infrastructure.vector_store import vector_store
from .chunking import chunk_documents
from .document_types import validate_document_type
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
    update_status,
)


logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", "/app/uploads")
)
OCR_LANGUAGES = os.getenv(
    "OCR_LANGUAGES",
    "eng+chi_tra",
)


def _document_filter(
    document_id: str,
) -> MetadataFilters:
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="document_id",
                value=document_id,
                operator=FilterOperator.EQ,
            )
        ]
    )


def delete_document_chunks(
    document_id: str,
) -> int:
    nodes = vector_store.get_nodes(
        filters=_document_filter(document_id)
    )
    if not nodes:
        return 0

    vector_store.delete_nodes(
        node_ids=[node.node_id for node in nodes]
    )
    return len(nodes)


def delete_old_versions(
    document_id: str,
    current_version: int,
) -> int:
    nodes = vector_store.get_nodes(
        filters=_document_filter(document_id)
    )

    old_node_ids = [
        node.node_id
        for node in nodes
        if int(
            node.metadata.get(
                "document_version",
                0,
            )
        )
        != int(current_version)
    ]

    if old_node_ids:
        vector_store.delete_nodes(
            node_ids=old_node_ids
        )

    return len(old_node_ids)


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
    replace_document_id: str | None = None,
) -> dict:
    ensure_registry_schema()

    extension = stored_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    content_hash = file_content_hash(stored_path)
    document_type = validate_document_type(document_type)

    if replace_document_id is None:
        duplicate = find_completed_duplicate(
            content_hash
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
            and (existing.get("document_type") or "auto") == document_type
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
    ensure_registry_schema()

    record = get_document(document_id)
    if not record:
        raise ValueError(
            f"Document {document_id} was not found."
        )

    stored_path = (
        UPLOAD_DIR
        / record["stored_file_name"]
    )

    try:
        update_status(
            document_id,
            "extracting",
        )

        documents = load_file(
            stored_path,
            document_id=str(record["document_id"]),
            original_file_name=record[
                "original_file_name"
            ],
            source_title=record.get(
                "source_title"
            )
            or "",
            source_url=record.get(
                "source_url"
            )
            or "",
            source_type=record.get(
                "source_type"
            )
            or "admin_upload",
            access_level=record.get(
                "access_level"
            )
            or "public",
            document_version=int(
                record["version"]
            ),
            content_hash=record[
                "content_hash"
            ],
            ocr_languages=OCR_LANGUAGES,
            category=record.get("category"),
            language=record.get("language"),
            effective_date=str(record.get("effective_date") or ""),
            source_kind=record.get("source_kind") or "upload",
            document_type=record.get("document_type") or "auto",
        )

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
    replace_document_id: str | None = None,
) -> dict:
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
    return process_registered_document(
        document_id
    )


def delete_registered_document(
    document_id: str,
    *,
    delete_file: bool = True,
) -> dict:
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
    selected_type = (
        validate_document_type(document_type)
        if document_type is not None
        else None
    )
    record = prepare_reindex(document_id, selected_type)
    if not record:
        raise ValueError("Document was not found.")
    return process_registered_document(document_id)
