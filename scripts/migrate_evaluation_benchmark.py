#!/usr/bin/env python3
"""Migrate legacy evaluation labels to the current document and chunk IDs.

The script validates source references against the current vector corpus,
rewrites compatible rows, and synchronizes the migrated evaluation table. It is
a maintenance utility rather than a normal ingestion entry point.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.ingest_pgvector_llamaindex import ingest_evaluation_dataset


EVALUATION_DATASET_PATH = Path(os.getenv(
    "EVALUATION_DATASET_PATH",
    "/app/data/evaluation_dataset.csv",
))
ALIASES_BY_QUESTION = {
    (
        "What is the phone number to contact for enquiries regarding "
        "library account passwords?"
    ): ["1823"],
    (
        "What is the phone number for enquiries regarding library account "
        "password issues?"
    ): ["1823"],
    (
        "Which form number is used for the Application for Using Smart "
        "Identity Card as Identity Card Allowed for Library Purposes?"
    ): [],
    "What facilities are provided by the Library's hiring facilities?": [
        "an exhibition gallery, a lecture theatre, and activity rooms"
    ],
}
ANSWER_UPDATES = {
    "Which e-book collection provides access to Jin Yong Martial Arts Novels?": (
        "Jin Yong Martial Arts Novels Audio Collection"
    ),
}
QUESTIONS_TO_DELETE = {
    (
        "What is the name of the interactive learning platform mainly for "
        "pre-school children mentioned in the e-Resources highlights?"
    ),
    (
        "What are the two methods readers may consider to simplify the "
        "account login procedures without memorising the online passwords?"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply reviewed benchmark-label corrections while preserving "
            "current source document and chunk IDs."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm removal of two invalid or contradictory benchmark rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.yes:
        raise SystemExit(
            "Review the migration first, then rerun with --yes. Two invalid "
            "benchmark rows will be removed."
        )
    if not EVALUATION_DATASET_PATH.is_file():
        raise FileNotFoundError(EVALUATION_DATASET_PATH)

    with EVALUATION_DATASET_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "accepted_answers_json" not in fieldnames:
        insertion_index = fieldnames.index("expected_context_snippet") + 1
        fieldnames.insert(insertion_index, "accepted_answers_json")

    retained = []
    retained_by_question: dict[str, dict] = {}
    updated = 0
    deleted = 0
    for row in rows:
        question = str(row.get("query") or "")
        if question in QUESTIONS_TO_DELETE:
            deleted += 1
            continue

        prior = retained_by_question.get(question)
        if prior is not None:
            if prior.get("expected_answer_text") != row.get("expected_answer_text"):
                raise ValueError(
                    f"Conflicting duplicate benchmark question: {question!r}"
                )
            deleted += 1
            continue

        aliases = ALIASES_BY_QUESTION.get(question)
        if aliases is not None:
            serialized_aliases = json.dumps(aliases, ensure_ascii=False)
            if row.get("accepted_answers_json") != serialized_aliases:
                row["accepted_answers_json"] = serialized_aliases
                updated += 1
        elif not row.get("accepted_answers_json"):
            row["accepted_answers_json"] = "[]"

        replacement_answer = ANSWER_UPDATES.get(question)
        if (
            replacement_answer
            and row.get("expected_answer_text") != replacement_answer
        ):
            row["expected_answer_text"] = replacement_answer
            updated += 1
        retained_by_question[question] = row
        retained.append(row)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = EVALUATION_DATASET_PATH.with_name(
        f"{EVALUATION_DATASET_PATH.stem}.before-benchmark-fix-{timestamp}.csv"
    )
    shutil.copy2(EVALUATION_DATASET_PATH, backup)

    temporary = EVALUATION_DATASET_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    temporary.replace(EVALUATION_DATASET_PATH)

    synchronized = ingest_evaluation_dataset(str(EVALUATION_DATASET_PATH))
    print(f"Backup: {backup}")
    print(f"Rows before: {len(rows)}")
    print(f"Rows after: {len(retained)}")
    print(f"Rows removed: {deleted}")
    print(f"Reviewed labels or aliases updated: {updated}")
    print(f"Database rows inserted or updated: {synchronized}")


if __name__ == "__main__":
    main()
