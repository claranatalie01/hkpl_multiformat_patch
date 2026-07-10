#!/usr/bin/env python3

import argparse
import os
import re
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.infrastructure.vector_store import VECTOR_TABLE


EVALUATION_DATASET_TABLE = os.getenv("EVALUATION_DATASET_TABLE", "evaluation_dataset")
KNOWLEDGE_TABLE = f"data_{VECTOR_TABLE}"


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\"'“”‘’]", "", value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate evaluation rows against the searchable knowledge chunks.",
    )
    parser.add_argument(
        "--delete-missing-chunks",
        action="store_true",
        help="Delete evaluation rows whose source_chunk_id is missing from the knowledge table.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive cleanup actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                SELECT
                    e.id,
                    e.query,
                    e.expected_answer_text,
                    e.expected_context_snippet,
                    e.source_document_id,
                    e.source_chunk_id,
                    k.text AS source_chunk_text,
                    k.metadata_->>'source_title' AS source_title
                FROM {EVALUATION_DATASET_TABLE} e
                LEFT JOIN {KNOWLEDGE_TABLE} k
                  ON k.metadata_->>'chunk_id' = e.source_chunk_id
                ORDER BY e.id
            """)
        ).fetchall()

    total = len(rows)
    chunk_found = 0
    snippet_found = 0
    answer_found = 0
    missing_chunks = []
    missing_snippets = []
    missing_answers = []

    for row in rows:
        item = dict(row._mapping)
        chunk_text = item.get("source_chunk_text") or ""
        expected_answer = item.get("expected_answer_text") or ""
        expected_snippet = item.get("expected_context_snippet") or ""

        if chunk_text:
            chunk_found += 1
        else:
            missing_chunks.append(item)
            continue

        normalized_chunk = normalize(chunk_text)
        normalized_snippet = normalize(expected_snippet)
        normalized_answer = normalize(expected_answer)

        if normalized_snippet and normalized_snippet in normalized_chunk:
            snippet_found += 1
        else:
            missing_snippets.append(item)

        if normalized_answer and normalized_answer in normalized_chunk:
            answer_found += 1
        else:
            missing_answers.append(item)

    print("=" * 80)
    print("Evaluation Dataset Coverage Check")
    print("=" * 80)
    print(f"Evaluation table      : {EVALUATION_DATASET_TABLE}")
    print(f"Knowledge table       : {KNOWLEDGE_TABLE}")
    print(f"Evaluation rows       : {total}")
    print(f"Expected chunk found  : {chunk_found}/{total} ({chunk_found / total:.2%})" if total else "Expected chunk found  : 0/0")
    print(f"Snippet text found    : {snippet_found}/{total} ({snippet_found / total:.2%})" if total else "Snippet text found    : 0/0")
    print(f"Answer text found     : {answer_found}/{total} ({answer_found / total:.2%})" if total else "Answer text found     : 0/0")

    if missing_chunks:
        print()
        print("Missing expected chunks:")
        for item in missing_chunks[:10]:
            print(f"- id={item['id']} chunk_id={item['source_chunk_id']} query={item['query']}")

    if missing_snippets:
        print()
        print("Expected snippet not found in linked chunk:")
        for item in missing_snippets[:10]:
            print(f"- id={item['id']} chunk_id={item['source_chunk_id']} query={item['query']}")

    if missing_answers:
        print()
        print("Expected answer not found verbatim in linked chunk:")
        for item in missing_answers[:10]:
            print(f"- id={item['id']} answer={item['expected_answer_text']} query={item['query']}")

    if args.delete_missing_chunks:
        if not args.yes:
            raise SystemExit(
                "Refusing to delete rows without --yes. "
                "Re-run with --delete-missing-chunks --yes after reviewing the list."
            )

        missing_ids = [item["id"] for item in missing_chunks]
        if not missing_ids:
            print()
            print("No rows with missing expected chunks to delete.")
            return

        with engine.begin() as connection:
            deleted = connection.execute(
                text(f"""
                    DELETE FROM {EVALUATION_DATASET_TABLE}
                    WHERE id = ANY(:missing_ids)
                """),
                {"missing_ids": missing_ids},
            ).rowcount

        print()
        print(f"Deleted {deleted} rows with missing expected chunks.")


if __name__ == "__main__":
    main()
