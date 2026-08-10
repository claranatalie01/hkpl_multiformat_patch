#!/usr/bin/env python3
"""Normalize evaluation CSV columns and values to the canonical schema.

This maintenance script repairs dataset shape before validation or database
synchronization. It changes evaluation metadata only, never vector chunks.
"""

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
COLUMNS = [
    "domain",
    "query",
    "expected_answer_text",
    "expected_context_snippet",
    "accepted_answers_json",
    "source_title",
    "source_url",
    "source_document_id",
    "source_chunk_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the evaluation CSV to the supported nine-column schema.",
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        actual_columns = list(reader.fieldnames or [])
        rows = list(reader)

    if args.check and actual_columns != COLUMNS:
        raise ValueError(
            "Evaluation CSV columns must exactly match, in order: "
            f"{COLUMNS}. Found: {actual_columns}"
        )

    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"Malformed CSV row {line_number}: too many fields")
        query = " ".join(str(row.get("query") or "").split())
        if not query:
            raise ValueError(f"Missing query at row {line_number}")
        query_key = query.casefold()
        if query_key in seen:
            raise ValueError(f"Duplicate question at row {line_number}: {query!r}")
        seen.add(query_key)

        aliases = json.loads(row.get("accepted_answers_json") or "[]")
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError(f"Invalid accepted answers at row {line_number}")

        item = {column: str(row.get(column) or "").strip() for column in COLUMNS}
        item["query"] = query
        item["accepted_answers_json"] = json.dumps(
            list(dict.fromkeys(alias.strip() for alias in aliases)),
            ensure_ascii=False,
        )
        missing = [
            column
            for column in COLUMNS
            if column not in {"accepted_answers_json", "source_url"}
            and not item[column]
        ]
        if missing:
            raise ValueError(
                f"Empty required fields at row {line_number}: {', '.join(missing)}"
            )
        document_id = item["source_document_id"]
        if not re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            document_id,
        ):
            raise ValueError(
                f"Invalid source_document_id at row {line_number}: "
                f"{document_id!r}"
            )
        if not item["source_chunk_id"].startswith(f"{document_id}:"):
            raise ValueError(
                f"source_chunk_id does not belong to source_document_id "
                f"at row {line_number}"
            )
        cleaned.append(item)

    print(f"Rows: {len(cleaned)}")
    print("Schema: " + ",".join(COLUMNS))
    if args.check:
        print("Result: PASSED (no file changes)")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = args.path.with_name(f"{args.path.stem}.before-schema-{timestamp}.csv")
    shutil.copy2(args.path, backup)

    temporary = args.path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cleaned)
    temporary.replace(args.path)

    print(f"Backup: {backup}")
    print("Result: NORMALIZED")


if __name__ == "__main__":
    main()
