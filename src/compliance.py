import json
from typing import Optional
from sqlalchemy import text

from .infrastructure.db import engine


def list_prohibited_keywords() -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT *
                FROM prohibited_keywords
                ORDER BY created_at DESC
            """)
        ).fetchall()

    return [dict(row._mapping) for row in rows]


def create_prohibited_keyword(
    *,
    keyword: str,
    category: str,
    language: str,
    fallback_response: str,
    created_by: str = "",
) -> dict:
    with engine.begin() as connection:
        row = connection.execute(
            text("""
                INSERT INTO prohibited_keywords (
                    keyword,
                    category,
                    language,
                    fallback_response,
                    created_by
                )
                VALUES (
                    :keyword,
                    :category,
                    :language,
                    :fallback_response,
                    :created_by
                )
                RETURNING *
            """),
            {
                "keyword": keyword.strip(),
                "category": category,
                "language": language,
                "fallback_response": fallback_response,
                "created_by": created_by,
            },
        ).fetchone()
    created = dict(row._mapping)

    write_keyword_audit_log(
        keyword_id=str(created["id"]),
        action="create",
        staff_id=created_by or "admin",
        old_value=None,
        new_value=created,
    )

    return created


def set_keyword_active(
    keyword_id: str,
    is_active: bool,
    staff_id: str = "admin",
) -> Optional[dict]:
    with engine.begin() as connection:
        old_row = connection.execute(
            text("""
                SELECT *
                FROM prohibited_keywords
                WHERE id = :keyword_id
            """),
            {"keyword_id": keyword_id},
        ).fetchone()

        if not old_row:
            return None

        old_value = dict(old_row._mapping)

        row = connection.execute(
            text("""
                UPDATE prohibited_keywords
                SET
                    is_active = :is_active,
                    updated_at = NOW()
                WHERE id = :keyword_id
                RETURNING *
            """),
            {
                "keyword_id": keyword_id,
                "is_active": is_active,
            },
        ).fetchone()

    updated = dict(row._mapping)

    write_keyword_audit_log(
        keyword_id=keyword_id,
        action="activate" if is_active else "deactivate",
        staff_id=staff_id,
        old_value=old_value,
        new_value=updated,
    )

    return updated


def check_prohibited_keywords(text_value: str) -> Optional[dict]:
    if not text_value:
        return None

    lowered = text_value.lower()

    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT *
                FROM prohibited_keywords
                WHERE is_active = TRUE
                ORDER BY LENGTH(keyword) DESC
            """)
        ).fetchall()

    for row in rows:
        item = dict(row._mapping)
        keyword = item["keyword"].lower().strip()

        if keyword and keyword in lowered:
            return {
                "matched": True,
                "keyword": item["keyword"],
                "category": item["category"],
                "fallback_response": item["fallback_response"],
            }

    return None

def write_keyword_audit_log(
    *,
    keyword_id: str | None,
    action: str,
    staff_id: str,
    old_value: dict | None,
    new_value: dict | None,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("""
                INSERT INTO prohibited_keyword_audit_log (
                    keyword_id,
                    action,
                    staff_id,
                    old_value,
                    new_value
                )
                VALUES (
                    :keyword_id,
                    :action,
                    :staff_id,
                    CAST(:old_value AS JSONB),
                    CAST(:new_value AS JSONB)
                )
            """),
            {
                "keyword_id": keyword_id,
                "action": action,
                "staff_id": staff_id,
                "old_value": json.dumps(old_value, default=str) if old_value else None,
                "new_value": json.dumps(new_value, default=str) if new_value else None,
            },
        )