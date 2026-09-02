#!/usr/bin/env python3
"""Re-embed an existing chunk corpus into a separate PGVector table.

This command supports controlled embedding-model A/B experiments. It copies
the exact text, metadata, and stable node IDs from a source vector table,
requests fresh vectors from ``EMBEDDING_URL``, and writes them to the physical
``data_<VECTOR_TABLE>`` table. It never modifies the source table, document
registry, chunking logic, or source files.

The target table is resumable by node ID. A partial run can therefore be
continued without regenerating vectors already committed. Use
``--reset-target --yes`` only when intentionally rebuilding the experimental
table from zero.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.ingestion.write_guard import ensure_corpus_writable


IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://embedding:8080/v1/embeddings",
)
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "hkpl_knowledge_hybrid_jina_v5")
TARGET_TABLE = f"data_{VECTOR_TABLE}"


def identifier(value: str) -> str:
    """Return a validated unquoted PostgreSQL identifier."""

    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def parse_args() -> argparse.Namespace:
    """Parse source, batching, resume, and explicit reset options."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate new embeddings for an existing chunk corpus without "
            "changing its text, metadata, or node IDs."
        )
    )
    parser.add_argument(
        "--source-table",
        default="data_hkpl_knowledge_hybrid",
        help="Physical PGVector table whose chunks will be re-embedded.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of chunk texts sent to the embedding endpoint per request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many new rows during this invocation.",
    )
    parser.add_argument(
        "--reset-target",
        action="store_true",
        help="Drop and recreate the target before embedding.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation for --reset-target.",
    )
    return parser.parse_args()


def relation_exists(table_name: str) -> bool:
    """Return whether a table exists in the current PostgreSQL schema."""

    with engine.connect() as connection:
        return connection.execute(
            text("SELECT to_regclass(:table_name) IS NOT NULL"),
            {"table_name": table_name},
        ).scalar_one()


def prepare_target(source_table: str, target_table: str, reset: bool) -> None:
    """Validate the source and create an empty target with matching schema."""

    if source_table == target_table:
        raise ValueError("Source and target vector tables must be different.")
    if not relation_exists(source_table):
        raise RuntimeError(f"Source vector table does not exist: {source_table}")

    with engine.begin() as connection:
        if reset:
            connection.execute(text(f"DROP TABLE IF EXISTS {target_table}"))
        connection.execute(text(
            f"CREATE TABLE IF NOT EXISTS {target_table} "
            f"(LIKE {source_table} INCLUDING ALL)"
        ))


def audit_node_ids(source_table: str, target_table: str) -> dict[str, int]:
    """Fail when null or duplicate IDs would make resume behavior ambiguous."""

    with engine.connect() as connection:
        source = connection.execute(text(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(*) FILTER (WHERE node_id IS NULL OR node_id = '') AS null_ids,
                COUNT(*) - COUNT(DISTINCT node_id) AS duplicate_ids
            FROM {source_table}
        """)).mappings().one()
        target = connection.execute(text(f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(*) FILTER (WHERE node_id IS NULL OR node_id = '') AS null_ids,
                COUNT(*) - COUNT(DISTINCT node_id) AS duplicate_ids
            FROM {target_table}
        """)).mappings().one()

    summary = {
        "source_rows": int(source["rows"]),
        "source_null_ids": int(source["null_ids"]),
        "source_duplicate_ids": int(source["duplicate_ids"]),
        "target_rows": int(target["rows"]),
        "target_null_ids": int(target["null_ids"]),
        "target_duplicate_ids": int(target["duplicate_ids"]),
    }
    invalid = [
        key for key, value in summary.items()
        if ("null_ids" in key or "duplicate_ids" in key) and value
    ]
    if invalid:
        raise RuntimeError(
            "Stable node IDs are required for resumable re-embedding; "
            + ", ".join(f"{key}={summary[key]}" for key in invalid)
        )
    return summary


def fetch_batch(
    source_table: str,
    target_table: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Load the next source rows whose node IDs are absent from the target."""

    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(text(f"""
            SELECT source.id, source.text, source.metadata_, source.node_id
            FROM {source_table} AS source
            LEFT JOIN {target_table} AS target
              ON target.node_id = source.node_id
            WHERE target.node_id IS NULL
            ORDER BY source.id
            LIMIT :batch_size
        """), {"batch_size": batch_size}).mappings().all()]


def request_embeddings(
    session: requests.Session,
    rows: list[dict[str, Any]],
) -> list[list[float]]:
    """Request ordered 1,024-dimensional vectors for one text batch."""

    response = session.post(
        EMBEDDING_URL,
        json={"input": [str(row["text"]) for row in rows]},
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Embedding service error {response.status_code}: {response.text}"
        )
    items = response.json().get("data", [])
    items = sorted(items, key=lambda item: int(item.get("index", 0)))
    vectors = [item.get("embedding") for item in items]
    if len(vectors) != len(rows):
        raise RuntimeError(
            f"Embedding service returned {len(vectors)} vectors for "
            f"{len(rows)} texts."
        )
    for index, vector in enumerate(vectors):
        if not isinstance(vector, list) or len(vector) != EMBED_DIM:
            actual = len(vector) if isinstance(vector, list) else "invalid"
            raise RuntimeError(
                f"Vector {index} has dimension {actual}; expected {EMBED_DIM}."
            )
    return vectors


def insert_batch(
    target_table: str,
    rows: list[dict[str, Any]],
    vectors: list[list[float]],
) -> None:
    """Commit one batch while preserving source text, metadata, and node IDs."""

    values = []
    for row, vector in zip(rows, vectors):
        metadata = row["metadata_"]
        if not isinstance(metadata, str):
            metadata = json.dumps(metadata, ensure_ascii=False)
        values.append({
            "id": row["id"],
            "text": row["text"],
            "metadata": metadata,
            "node_id": row["node_id"],
            "embedding": json.dumps(vector, separators=(",", ":")),
        })

    with engine.begin() as connection:
        connection.execute(text(f"""
            INSERT INTO {target_table} (
                id, text, metadata_, node_id, embedding
            )
            VALUES (
                :id,
                :text,
                CAST(:metadata AS json),
                :node_id,
                CAST(:embedding AS vector)
            )
        """), values)


def final_audit(source_table: str, target_table: str) -> dict[str, int]:
    """Count copied rows, missing source IDs, and invalid vector dimensions."""

    with engine.connect() as connection:
        row = connection.execute(text(f"""
            SELECT
                (SELECT COUNT(*) FROM {source_table}) AS source_rows,
                (SELECT COUNT(*) FROM {target_table}) AS target_rows,
                (
                    SELECT COUNT(*)
                    FROM {source_table} AS source
                    LEFT JOIN {target_table} AS target
                      ON target.node_id = source.node_id
                    WHERE target.node_id IS NULL
                ) AS missing_rows,
                (
                    SELECT COUNT(*)
                    FROM {target_table}
                    WHERE embedding IS NULL
                       OR vector_dims(embedding) <> :embed_dim
                ) AS invalid_vectors,
                (
                    SELECT COUNT(*)
                    FROM {target_table} AS target
                    LEFT JOIN {source_table} AS source
                      ON source.node_id = target.node_id
                    WHERE source.node_id IS NULL
                ) AS extra_rows,
                (
                    SELECT COUNT(*)
                    FROM {source_table} AS source
                    JOIN {target_table} AS target
                      ON target.node_id = source.node_id
                    WHERE source.text IS DISTINCT FROM target.text
                       OR source.metadata_::jsonb
                          IS DISTINCT FROM target.metadata_::jsonb
                ) AS content_mismatches
        """), {"embed_dim": EMBED_DIM}).mappings().one()
    return {key: int(value) for key, value in row.items()}


def main() -> None:
    """Create or resume the target and report a reproducible final audit."""

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.reset_target and not args.yes:
        raise SystemExit("--reset-target requires --yes.")

    ensure_corpus_writable("build the experimental embedding vector table")
    source_table = identifier(args.source_table)
    target_table = identifier(TARGET_TABLE)
    prepare_target(source_table, target_table, args.reset_target)
    before = audit_node_ids(source_table, target_table)

    print(f"Embedding endpoint: {EMBEDDING_URL}")
    print(f"Source table: {source_table} ({before['source_rows']} rows)")
    print(f"Target table: {target_table} ({before['target_rows']} existing rows)")
    print(f"Embedding dimension: {EMBED_DIM}")

    processed = 0
    with requests.Session() as session:
        while args.limit is None or processed < args.limit:
            remaining = (
                args.batch_size
                if args.limit is None
                else min(args.batch_size, args.limit - processed)
            )
            rows = fetch_batch(source_table, target_table, remaining)
            if not rows:
                break
            vectors = request_embeddings(session, rows)
            insert_batch(target_table, rows, vectors)
            processed += len(rows)
            print(f"Re-embedded rows this run: {processed}")

    audit = final_audit(source_table, target_table)
    print("Re-embedding audit")
    for key, value in audit.items():
        print(f"- {key}: {value}")

    if (
        audit["invalid_vectors"]
        or audit["extra_rows"]
        or audit["content_mismatches"]
    ):
        raise RuntimeError(
            "Target audit failed: vectors, row identity, text, or metadata differ."
        )
    if args.limit is None and audit["missing_rows"]:
        raise RuntimeError("Full run ended before every source row was copied.")


if __name__ == "__main__":
    main()
