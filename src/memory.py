from typing import Dict, List

from sqlalchemy import text

from .infrastructure.db import engine


def load_conversation_history(
    session_id: str,
    limit: int = 6,
) -> List[Dict[str, str]]:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT role, content
                FROM conversation_history
                WHERE session_id = :session_id
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {
                "session_id": session_id,
                "limit": limit,
            },
        ).fetchall()

    return [
        {"role": row.role, "content": row.content}
        for row in reversed(rows)
    ]


def save_conversation_turn(
    session_id: str,
    role: str,
    content: str,
) -> None:
    if not session_id or not content:
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO conversation_history
                    (session_id, role, content)
                VALUES
                    (:session_id, :role, :content)
                """
            ),
            {
                "session_id": session_id,
                "role": role,
                "content": content,
            },
        )
