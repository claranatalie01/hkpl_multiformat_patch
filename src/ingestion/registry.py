"""Maintain the PostgreSQL registry for knowledge-base source documents.

The registry records source identity, file location, hash, version, ingestion
status, and expected chunk count. The actual chunk text and embeddings live in
the separately configured LlamaIndex-managed ``data_<VECTOR_TABLE>`` table.
"""

from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import text

from ..infrastructure.db import engine


def ensure_registry_schema() -> None:
    """Create or evolve registry and compliance tables for existing databases."""
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
                    classification_source TEXT NOT NULL DEFAULT 'fallback',
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
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS prohibited_keywords (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                keyword TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                language TEXT NOT NULL DEFAULT 'en',
                fallback_response TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_by TEXT DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS prohibited_keyword_audit_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                keyword_id UUID,
                action TEXT NOT NULL,
                staff_id TEXT NOT NULL,
                old_value JSONB,
                new_value JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_keyword_id
            ON prohibited_keyword_audit_log(keyword_id)
        """))

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_staff_id
            ON prohibited_keyword_audit_log(staff_id)
        """))

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_action
            ON prohibited_keyword_audit_log(action)
        """))

        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_prohibited_keyword_audit_created_at
            ON prohibited_keyword_audit_log(created_at DESC)
        """))
        
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
        connection.execute(text("""
            ALTER TABLE knowledge_documents
            ADD COLUMN IF NOT EXISTS category TEXT
        """))

        connection.execute(text("""
            ALTER TABLE knowledge_documents
            ADD COLUMN IF NOT EXISTS language TEXT
        """))

        connection.execute(text("""
            ALTER TABLE knowledge_documents
            ADD COLUMN IF NOT EXISTS effective_date DATE
        """))

        connection.execute(text("""
            ALTER TABLE knowledge_documents
            ADD COLUMN IF NOT EXISTS source_kind TEXT
        """))
        connection.execute(text("""
            ALTER TABLE knowledge_documents
            ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'auto'
        """))
        connection.execute(text("""
            ALTER TABLE knowledge_documents
            ADD COLUMN IF NOT EXISTS classification_source TEXT NOT NULL DEFAULT 'fallback'
        """))
        connection.execute(text("""
            UPDATE knowledge_documents
            SET classification_source = 'persisted'
            WHERE document_type <> 'auto'
              AND classification_source = 'fallback'
        """))


def _row_to_dict(row: Any) -> dict:
    return dict(row._mapping)


def find_completed_duplicate(
    content_hash: str,
    *,
    source_url: str = "",
    original_file_name: str = "",
) -> Optional[dict]:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT *
                FROM knowledge_documents
                WHERE content_hash = :content_hash
                  AND status = 'completed'
                  AND (
                    (:source_url <> '' AND source_url = :source_url)
                    OR (
                      :source_url = ''
                      AND COALESCE(source_url, '') = ''
                      AND original_file_name = :original_file_name
                    )
                  )
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ),
            {
                "content_hash": content_hash,
                "source_url": source_url,
                "original_file_name": original_file_name,
            },
        ).fetchone()

    return _row_to_dict(row) if row else None


def find_active_web_document_by_source_url(source_url: str) -> Optional[dict]:
    """Return the current non-deleted crawler/webpage record for a source URL."""
    if not source_url:
        return None

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT *
                FROM knowledge_documents
                WHERE source_url = :source_url
                  AND status <> 'deleted'
                  AND source_type IN ('crawler', 'webpage')
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ),
            {"source_url": source_url},
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
    classification_source: str,
    access_level: str,
    category: str | None = None,
    language: str | None = None,
    effective_date: str | None = None,
    source_kind: str = "upload",
    document_type: str = "auto",
) -> dict:
    """Insert a version-1 registry record with initial status ``uploaded``."""
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
                    classification_source,
                    access_level,
                    version,
                    status,
                    category,
                    language,
                    effective_date,
                    source_kind,
                    document_type
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
                    :classification_source,
                    :access_level,
                    1,
                    'uploaded',
                    :category,
                    :language,
                    :effective_date,
                    :source_kind,
                    :document_type
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
                "classification_source": classification_source,
                "access_level": access_level,
                "category": category,
                "language": language,
                "effective_date": effective_date or None,
                "source_kind": source_kind,
                "document_type": document_type,
                
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
    classification_source: str,
    access_level: str,
    category: str | None = None,
    language: str | None = None,
    effective_date: str | None = None,
    source_kind: str = "upload",
    document_type: str = "auto",
) -> Optional[dict]:
    """Update a registry record for new source content and increment version."""
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
                    classification_source = :classification_source,
                    access_level = :access_level,
                    category = :category,
                    language = :language,
                    effective_date = :effective_date,
                    source_kind = :source_kind,
                    document_type = :document_type,
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
                "classification_source": classification_source,
                "access_level": access_level,
                "category": category,
                "language": language,
                "effective_date": effective_date or None,
                "source_kind": source_kind,
                "document_type": document_type,
            },
        ).fetchone()

    return _row_to_dict(row) if row else None


def prepare_reindex(
    document_id: str,
    document_type: str | None = None,
    classification_source: str | None = None,
) -> Optional[dict]:
    with engine.begin() as connection:
        row = connection.execute(
            text(
                """
                UPDATE knowledge_documents
                SET
                    document_type = COALESCE(:document_type, document_type, 'auto'),
                    classification_source = COALESCE(
                        :classification_source,
                        classification_source,
                        'fallback'
                    ),
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
                "document_type": document_type,
                "classification_source": classification_source,
            },
        ).fetchone()

    return _row_to_dict(row) if row else None


def restore_document(document: dict) -> None:
    """Restore a registry row after a failed in-place replacement."""
    with engine.begin() as connection:
        connection.execute(
            text("""
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
                    classification_source = :classification_source,
                    access_level = :access_level,
                    version = :version,
                    status = :status,
                    chunk_count = :chunk_count,
                    error_message = :error_message,
                    category = :category,
                    language = :language,
                    effective_date = :effective_date,
                    source_kind = :source_kind,
                    document_type = :document_type,
                    updated_at = NOW()
                WHERE document_id = :document_id
            """),
            dict(document),
        )


def set_document_type(
    document_id: str,
    document_type: str,
    classification_source: str,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("""
                UPDATE knowledge_documents
                SET
                    document_type = :document_type,
                    classification_source = :classification_source,
                    updated_at = NOW()
                WHERE document_id = :document_id
                  AND status <> 'deleted'
            """),
            {
                "document_id": document_id,
                "document_type": document_type,
                "classification_source": classification_source,
            },
        )


def update_status(
    document_id: str,
    status: str,
    *,
    chunk_count: int | None = None,
    error_message: str | None = None,
) -> None:
    """Record ingestion progress, final chunk count, or a failure message."""
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
