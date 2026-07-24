#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.infrastructure.vector_store import VECTOR_TABLE


CONTROL_TABLE = "knowledge_corpus_control"
LOCK_FUNCTION = "enforce_knowledge_corpus_read_only"
REGISTRY_TABLE = "knowledge_documents"


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


VECTOR_TABLE_NAME = identifier(f"data_{VECTOR_TABLE}")
PROTECTED_TABLES = (REGISTRY_TABLE, VECTOR_TABLE_NAME)


def install_lock() -> None:
    with engine.begin() as connection:
        connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {CONTROL_TABLE} (
                singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
                    CHECK (singleton),
                read_only BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        connection.execute(text(f"""
            INSERT INTO {CONTROL_TABLE} (singleton, read_only)
            VALUES (TRUE, FALSE)
            ON CONFLICT (singleton) DO NOTHING
        """))
        connection.execute(text(f"""
            CREATE OR REPLACE FUNCTION {LOCK_FUNCTION}()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM {CONTROL_TABLE}
                    WHERE singleton = TRUE
                      AND read_only = TRUE
                ) THEN
                    RAISE EXCEPTION
                        'Knowledge corpus is frozen; % on % is blocked',
                        TG_OP,
                        TG_TABLE_NAME
                        USING ERRCODE = '55000';
                END IF;

                IF TG_OP = 'TRUNCATE' THEN
                    RETURN NULL;
                ELSIF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END;
            $$
        """))

        for table_name in PROTECTED_TABLES:
            trigger_name = identifier(f"trg_freeze_{table_name}")
            truncate_trigger_name = identifier(
                f"trg_freeze_truncate_{table_name}"
            )
            connection.execute(text(
                f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name}"
            ))
            connection.execute(text(
                "DROP TRIGGER IF EXISTS "
                f"{truncate_trigger_name} ON {table_name}"
            ))
            connection.execute(text(f"""
                CREATE TRIGGER {trigger_name}
                BEFORE INSERT OR UPDATE OR DELETE
                ON {table_name}
                FOR EACH ROW
                EXECUTE FUNCTION {LOCK_FUNCTION}()
            """))
            connection.execute(text(f"""
                CREATE TRIGGER {truncate_trigger_name}
                BEFORE TRUNCATE
                ON {table_name}
                FOR EACH STATEMENT
                EXECUTE FUNCTION {LOCK_FUNCTION}()
            """))


def set_lock(read_only: bool) -> None:
    install_lock()
    with engine.begin() as connection:
        connection.execute(
            text(f"""
                UPDATE {CONTROL_TABLE}
                SET read_only = :read_only,
                    updated_at = NOW()
                WHERE singleton = TRUE
            """),
            {"read_only": read_only},
        )
        if not read_only:
            for table_name in PROTECTED_TABLES:
                trigger_name = identifier(f"trg_freeze_{table_name}")
                truncate_trigger_name = identifier(
                    f"trg_freeze_truncate_{table_name}"
                )
                connection.execute(text(
                    f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name}"
                ))
                connection.execute(text(
                    "DROP TRIGGER IF EXISTS "
                    f"{truncate_trigger_name} ON {table_name}"
                ))


def lock_status() -> tuple[bool | None, str | None]:
    with engine.connect() as connection:
        exists = connection.execute(
            text("SELECT to_regclass(:table_name)"),
            {"table_name": CONTROL_TABLE},
        ).scalar_one()
        if exists is None:
            return None, None
        row = connection.execute(text(f"""
            SELECT read_only, updated_at::text
            FROM {CONTROL_TABLE}
            WHERE singleton = TRUE
        """)).one_or_none()
    if row is None:
        return None, None
    return bool(row[0]), str(row[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or unfreeze the knowledge registry and vector table at "
            "the PostgreSQL level."
        ),
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--enable", action="store_true")
    action.add_argument("--disable", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required when disabling the database-level corpus lock.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable:
        set_lock(True)
    elif args.disable:
        if not args.yes:
            raise SystemExit(
                "Refusing to unfreeze the knowledge corpus without --yes."
            )
        set_lock(False)

    read_only, updated_at = lock_status()
    if read_only is None:
        print("Database corpus lock: NOT INSTALLED")
        return
    print(
        "Database corpus lock: "
        + ("ENABLED (READ ONLY)" if read_only else "DISABLED (WRITABLE)")
    )
    print(f"Last changed: {updated_at}")
    print("Protected tables:")
    for table_name in PROTECTED_TABLES:
        print(f"- {table_name}")


if __name__ == "__main__":
    main()
