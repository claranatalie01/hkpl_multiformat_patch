#!/usr/bin/env python3
"""Validate and optionally repair evaluation rows against stored evidence.

Checks include schema, exact or formatting-tolerant evidence containment,
source/chunk existence, duplicates, and ambiguity. Repair modes can relink or
exclude unresolved labels; the knowledge-vector table remains read-only.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.infrastructure.vector_store import VECTOR_TABLE


EVALUATION_DATASET_TABLE = os.getenv("EVALUATION_DATASET_TABLE", "evaluation_dataset")
KNOWLEDGE_TABLE = f"data_{VECTOR_TABLE}"
EVALUATION_DATASET_PATH = Path(os.getenv(
    "EVALUATION_DATASET_PATH",
    "/app/data/evaluation_dataset.csv",
))


def normalize(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\"'“”‘’]", "", value)
    return value


def compact_normalize(value: str) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold(), flags=re.UNICODE)


def formatting_tolerant_contains(expected: str, actual: str) -> bool:
    normalized_expected = normalize(expected)
    normalized_actual = normalize(actual)
    if normalized_expected and normalized_expected in normalized_actual:
        return True

    compact_expected = compact_normalize(expected)
    compact_actual = compact_normalize(actual)
    return len(compact_expected) >= 40 and compact_expected in compact_actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate evaluation rows against the searchable knowledge chunks.",
    )
    parser.add_argument(
        "--repair-missing-chunks",
        action="store_true",
        help=(
            "Relink stale evaluation rows to current chunks when the stored "
            "snippet, or a unique expected answer, matches the same source."
        ),
    )
    parser.add_argument(
        "--exclude-unresolved",
        action="store_true",
        help=(
            "Remove only unresolved stale rows from the evaluation CSV after "
            "manual review. A timestamped backup and exclusions report are "
            "created. Run the evaluation-only ingestion afterward to sync "
            "the database table."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive cleanup actions.",
    )
    args = parser.parse_args()
    if args.repair_missing_chunks and args.exclude_unresolved:
        parser.error(
            "--repair-missing-chunks and --exclude-unresolved must be run "
            "separately"
        )
    return args


def current_source_candidates(connection, item: dict) -> list[dict]:
    rows = connection.execute(
        text(f"""
            SELECT
                metadata_->>'kb_document_id' AS document_id,
                metadata_->>'chunk_id' AS chunk_id,
                metadata_->>'source_title' AS source_title,
                metadata_->>'source_url' AS source_url,
                text AS chunk_text
            FROM {KNOWLEDGE_TABLE}
            WHERE COALESCE(metadata_->>'corpus_role', 'primary') = 'primary'
              AND (
                    metadata_->>'kb_document_id' = :document_id
                    OR (
                        :source_url <> ''
                        AND metadata_->>'source_url' = :source_url
                    )
              )
            ORDER BY metadata_->>'chunk_id'
        """),
        {
            "document_id": str(item.get("source_document_id") or ""),
            "source_url": str(item.get("source_url") or ""),
        },
    ).mappings().all()
    return [dict(row) for row in rows]


def find_replacement(connection, item: dict) -> tuple[dict | None, str]:
    candidates = current_source_candidates(connection, item)
    if not candidates:
        return None, "source_not_found"

    expected_snippet = normalize(item.get("expected_context_snippet") or "")
    expected_answer = normalize(item.get("expected_answer_text") or "")
    normalized_candidates = [
        (candidate, normalize(candidate.get("chunk_text") or ""))
        for candidate in candidates
    ]

    snippet_matches = [
        candidate
        for candidate, chunk_text in normalized_candidates
        if expected_snippet and expected_snippet in chunk_text
    ]
    if snippet_matches:
        snippet_matches.sort(key=lambda candidate: len(candidate["chunk_text"]))
        return snippet_matches[0], "exact_snippet"

    compact_snippet = compact_normalize(item.get("expected_context_snippet") or "")
    compact_snippet_matches = [
        candidate
        for candidate, chunk_text in normalized_candidates
        if len(compact_snippet) >= 40
        and compact_snippet in compact_normalize(chunk_text)
    ]
    if compact_snippet_matches:
        compact_snippet_matches.sort(
            key=lambda candidate: len(candidate["chunk_text"])
        )
        return compact_snippet_matches[0], "formatting_normalized_snippet"

    answer_matches = [
        candidate
        for candidate, chunk_text in normalized_candidates
        if expected_answer and expected_answer in chunk_text
    ]
    if len(answer_matches) == 1:
        return answer_matches[0], "unique_exact_answer"
    if len(answer_matches) > 1:
        return None, "ambiguous_answer_match"

    compact_answer = compact_normalize(item.get("expected_answer_text") or "")
    compact_answer_matches = [
        candidate
        for candidate, chunk_text in normalized_candidates
        if len(compact_answer) >= 40
        and compact_answer in compact_normalize(chunk_text)
    ]
    if len(compact_answer_matches) == 1:
        return compact_answer_matches[0], "unique_formatting_normalized_answer"
    if len(compact_answer_matches) > 1:
        return None, "ambiguous_formatting_normalized_answer"
    return None, "evidence_not_found"


def synchronize_csv(repaired_rows: list[dict]) -> int:
    if not EVALUATION_DATASET_PATH.is_file() or not repaired_rows:
        return 0

    updates = {
        (str(item["query"]), str(item["source_chunk_id"])): item["replacement"]
        for item in repaired_rows
    }
    with EVALUATION_DATASET_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    updated = 0
    for row in rows:
        replacement = updates.get((
            str(row.get("query") or ""),
            str(row.get("source_chunk_id") or ""),
        ))
        if not replacement:
            continue
        row["source_document_id"] = replacement["document_id"]
        row["source_chunk_id"] = replacement["chunk_id"]
        if replacement.get("source_title"):
            row["source_title"] = replacement["source_title"]
        if replacement.get("source_url"):
            row["source_url"] = replacement["source_url"]
        updated += 1

    temporary_path = EVALUATION_DATASET_PATH.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(EVALUATION_DATASET_PATH)
    return updated


def backup_evaluation_csv() -> Path | None:
    if not EVALUATION_DATASET_PATH.is_file():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = EVALUATION_DATASET_PATH.with_name(
        f"{EVALUATION_DATASET_PATH.stem}.before-validation-{timestamp}.csv"
    )
    shutil.copy2(EVALUATION_DATASET_PATH, backup_path)
    return backup_path


def exclude_unresolved_from_csv(
    unresolved_rows: list[dict],
) -> tuple[int, Path, Path]:
    if not EVALUATION_DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {EVALUATION_DATASET_PATH}"
        )

    unresolved_keys = {
        (
            str(item.get("query") or ""),
            str(item.get("source_chunk_id") or ""),
        )
        for item in unresolved_rows
    }
    with EVALUATION_DATASET_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    excluded = [
        row
        for row in rows
        if (
            str(row.get("query") or ""),
            str(row.get("source_chunk_id") or ""),
        ) in unresolved_keys
    ]
    if len(excluded) != len(unresolved_keys):
        matched_keys = {
            (
                str(row.get("query") or ""),
                str(row.get("source_chunk_id") or ""),
            )
            for row in excluded
        }
        missing_keys = sorted(unresolved_keys - matched_keys)
        raise RuntimeError(
            "Refusing partial cleanup because the database and CSV do not "
            f"match. Missing CSV rows: {missing_keys}"
        )

    kept = [row for row in rows if row not in excluded]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = EVALUATION_DATASET_PATH.with_name(
        f"{EVALUATION_DATASET_PATH.stem}.before-exclusion-{timestamp}.csv"
    )
    report_path = EVALUATION_DATASET_PATH.with_name(
        f"{EVALUATION_DATASET_PATH.stem}.excluded-unresolved-{timestamp}.csv"
    )
    shutil.copy2(EVALUATION_DATASET_PATH, backup_path)

    report_fields = fieldnames + ["exclusion_reason", "excluded_at_utc"]
    with report_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=report_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in excluded:
            key = (
                str(row.get("query") or ""),
                str(row.get("source_chunk_id") or ""),
            )
            reason = next(
                str(item.get("repair_reason") or "evidence_not_found")
                for item in unresolved_rows
                if (
                    str(item.get("query") or ""),
                    str(item.get("source_chunk_id") or ""),
                ) == key
            )
            writer.writerow({
                **row,
                "exclusion_reason": reason,
                "excluded_at_utc": timestamp,
            })

    temporary_path = EVALUATION_DATASET_PATH.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(kept)
    temporary_path.replace(EVALUATION_DATASET_PATH)
    return len(excluded), backup_path, report_path


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
                    e.source_title AS expected_source_title,
                    e.source_url,
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
    repairable_chunks = []
    unresolved_repairs = []

    with engine.connect() as connection:
        for row in rows:
            item = dict(row._mapping)
            chunk_text = item.get("source_chunk_text") or ""
            expected_answer = item.get("expected_answer_text") or ""
            expected_snippet = item.get("expected_context_snippet") or ""

            if chunk_text:
                chunk_found += 1
            else:
                missing_chunks.append(item)
                replacement, reason = find_replacement(connection, item)
                if replacement:
                    repairable_chunks.append({
                        **item,
                        "replacement": replacement,
                        "repair_reason": reason,
                    })
                else:
                    unresolved_repairs.append({
                        **item,
                        "repair_reason": reason,
                    })
                continue

            normalized_chunk = normalize(chunk_text)
            normalized_answer = normalize(expected_answer)

            if formatting_tolerant_contains(expected_snippet, chunk_text):
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
    print(
        f"Answer verbatim found : {answer_found}/{total} "
        f"({answer_found / total:.2%}) [informational]"
        if total
        else "Answer verbatim found : 0/0 [informational]"
    )

    if missing_chunks:
        print()
        print("Missing expected chunks:")
        for item in missing_chunks[:10]:
            print(f"- id={item['id']} chunk_id={item['source_chunk_id']} query={item['query']}")

        print()
        print(f"Safely repairable stale rows: {len(repairable_chunks)}")
        print(f"Unresolved stale rows       : {len(unresolved_repairs)}")
        for item in unresolved_repairs[:10]:
            print(
                f"- id={item['id']} reason={item['repair_reason']} "
                f"query={item['query']}"
            )

    if missing_snippets:
        print()
        print("Expected snippet not found in linked chunk:")
        for item in missing_snippets[:10]:
            print(f"- id={item['id']} chunk_id={item['source_chunk_id']} query={item['query']}")

    if missing_answers:
        print()
        print(
            "Expected answer not found as one contiguous string "
            "(informational; paraphrases and combined facts are allowed):"
        )
        for item in missing_answers[:10]:
            print(f"- id={item['id']} answer={item['expected_answer_text']} query={item['query']}")

    ready = chunk_found == total and snippet_found == total
    print()
    print(
        "Evaluation evidence status: "
        + ("READY" if ready else "REVIEW REQUIRED")
    )
    if not ready:
        print(
            "Readiness requires every row to reference an existing chunk and "
            "contain its expected evidence snippet. Verbatim answer coverage "
            "does not determine readiness."
        )

    if args.repair_missing_chunks:
        if not args.yes:
            raise SystemExit(
                "Refusing to update rows without --yes. Review the repairable "
                "and unresolved counts, then rerun with "
                "--repair-missing-chunks --yes."
            )

        backup_path = backup_evaluation_csv()
        if backup_path:
            print(f"Backed up evaluation CSV to {backup_path}.")

        repaired = 0
        with engine.begin() as connection:
            for item in repairable_chunks:
                replacement = item["replacement"]
                repaired += int(connection.execute(
                    text(f"""
                        UPDATE {EVALUATION_DATASET_TABLE}
                        SET source_document_id = :document_id,
                            source_chunk_id = :chunk_id,
                            source_title = COALESCE(
                                NULLIF(:source_title, ''),
                                source_title
                            ),
                            source_url = COALESCE(
                                NULLIF(:source_url, ''),
                                source_url
                            )
                        WHERE id = :id
                    """),
                    {
                        "id": item["id"],
                        "document_id": replacement["document_id"],
                        "chunk_id": replacement["chunk_id"],
                        "source_title": replacement.get("source_title") or "",
                        "source_url": replacement.get("source_url") or "",
                    },
                ).rowcount or 0)

        print()
        print(f"Repaired {repaired} stale evaluation rows.")
        csv_updates = synchronize_csv(repairable_chunks)
        if EVALUATION_DATASET_PATH.is_file():
            print(
                f"Updated {csv_updates} rows in {EVALUATION_DATASET_PATH}."
            )
        if unresolved_repairs:
            print(
                f"Left {len(unresolved_repairs)} ambiguous or unmatched rows "
                "unchanged for manual review."
            )

    if args.exclude_unresolved:
        if not args.yes:
            raise SystemExit(
                "Refusing to exclude unresolved benchmark rows without --yes. "
                "Confirm that these questions no longer have authoritative "
                "evidence in the frozen corpus first."
            )
        if repairable_chunks:
            raise SystemExit(
                "Refusing to exclude rows while safely repairable references "
                "remain. Run --repair-missing-chunks --yes first."
            )
        if not unresolved_repairs:
            print("No unresolved stale rows to exclude.")
        else:
            excluded, backup_path, report_path = exclude_unresolved_from_csv(
                unresolved_repairs
            )
            print()
            print(f"Excluded {excluded} unresolved rows from the CSV.")
            print(f"Backup: {backup_path}")
            print(f"Exclusions report: {report_path}")
            print(
                "The database table has not been changed. Synchronize it with "
                "ingest_pgvector_llamaindex.py --evaluation-only, then validate "
                "again."
            )

    if (
        not ready
        and not args.repair_missing_chunks
        and not args.exclude_unresolved
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
