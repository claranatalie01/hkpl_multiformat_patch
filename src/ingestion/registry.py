from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import text

from ..infrastructure.db import engine


def ensure_registry_schema() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id UUID PRIMARY KEY,
                    original_file_name TEXT NOT NULL,
                    stored_file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    mime_type TEXT,
                    content_hash TEXT NOT NULL,
                    source_title TEXT,
                    source_url TEXT,
                    source_type TEXT NOT NULL DEFAULT 'admin_upload',
                    access_level TEXT NOT NULL DEFAULT 'public',
                    version INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_knowledge_documents_hash
                ON knowledge_documents(content_hash)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_knowledge_documents_status
                ON knowledge_documents(status)
                """
            )
        )


def _row_to_dict(row: Any) -> dict:
    return dict(row._mapping)


def find_completed_duplicate(
    content_hash: str,
) -> Optional[dict]:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT *
                FROM knowledge_documents
                WHERE content_hash = :content_hash
                  AND status = 'completed'
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ),
            {"content_hash": content_hash},
        ).fetchone()

    return _row_to_dict(row) if row else None


def create_document(
    *,
    original_file_name: str,
    stored_file_name: str,
    file_type: str,
    mime_type: str,
    content_hash: str,
    source_title: str,
    source_url: str,
    source_type: str,
    access_level: str,
) -> dict:
    document_id = str(uuid4())

    with engine.begin() as connection:
        row = connection.execute(
            text(
                """
                INSERT INTO knowledge_documents (
                    document_id,
                    original_file_name,
                    stored_file_name,
                    file_type,
                    mime_type,
                    content_hash,
                    source_title,
                    source_url,
                    source_type,
                    access_level,
                    version,
                    status
                )
                VALUES (
                    :document_id,
                    :original_file_name,
                    :stored_file_name,
                    :file_type,
                    :mime_type,
                    :content_hash,
                    :source_title,
                    :source_url,
                    :source_type,
                    :access_level,
                    1,
                    'uploaded'
                )
                RETURNING *
                """
            ),
            {
                "document_id": document_id,
                "original_file_name": original_file_name,
                "stored_file_name": stored_file_name,
                "file_type": file_type,
                "mime_type": mime_type,
                "content_hash": content_hash,
                "source_title": source_title,
                "source_url": source_url,
                "source_type": source_type,
                "access_level": access_level,
            },
        ).fetchone()

    return _row_to_dict(row)


def prepare_replacement(
    document_id: str,
    *,
    original_file_name: str,
    stored_file_name: str,
    file_type: str,
    mime_type: str,
    content_hash: str,
    source_title: str,
    source_url: str,
    source_type: str,
    access_level: str,
) -> Optional[dict]:
    with engine.begin() as connection:
        row = connection.execute(
            text(
                """
                UPDATE knowledge_documents
                SET
                    original_file_name = :original_file_name,
                    stored_file_name = :stored_file_name,
                    file_type = :file_type,
                    mime_type = :mime_type,
                    content_hash = :content_hash,
                    source_title = :source_title,
                    source_url = :source_url,
                    source_type = :source_type,
                    access_level = :access_level,
                    version = version + 1,
                    status = 'uploaded',
                    chunk_count = 0,
                    error_message = NULL,
                    updated_at = NOW()
                WHERE document_id = :document_id
                  AND status <> 'deleted'
                RETURNING *
                """
            ),
            {
                "document_id": document_id,
                "original_file_name": original_file_name,
                "stored_file_name": stored_file_name,
                "file_type": file_type,
                "mime_type": mime_type,
                "content_hash": content_hash,
                "source_title": source_title,
                "source_url": source_url,
                "source_type": source_type,
                "access_level": access_level,
            },
        ).fetchone()

    return _row_to_dict(row) if row else None


def update_status(
    document_id: str,
    status: str,
    *,
    chunk_count: int | None = None,
    error_message: str | None = None,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE knowledge_documents
                SET
                    status = :status,
                    chunk_count = COALESCE(
                        :chunk_count,
                        chunk_count
                    ),
                    error_message = :error_message,
                    updated_at = NOW()
                WHERE document_id = :document_id
                """
            ),
            {
                "document_id": document_id,
                "status": status,
                "chunk_count": chunk_count,
                "error_message": error_message,
            },
        )


def get_document(
    document_id: str,
) -> Optional[dict]:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT *
                FROM knowledge_documents
                WHERE document_id = :document_id
                """
            ),
            {"document_id": document_id},
        ).fetchone()

    return _row_to_dict(row) if row else None


def list_documents(
    include_deleted: bool = False,
) -> list[dict]:
    where = "" if include_deleted else "WHERE status <> 'deleted'"

    with engine.connect() as connection:
        rows = connection.execute(
            text(
                f"""
                SELECT *
                FROM knowledge_documents
                {where}
                ORDER BY uploaded_at DESC
                """
            )
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def mark_deleted(
    document_id: str,
) -> bool:
    with engine.begin() as connection:
        result = connection.execute(
            text(
                """
                UPDATE knowledge_documents
                SET
                    status = 'deleted',
                    updated_at = NOW()
                WHERE document_id = :document_id
                  AND status <> 'deleted'
                """
            ),
            {"document_id": document_id},
        )

    return result.rowcount > 0
