#!/usr/bin/env python3
"""Normalize evaluation CSV columns and values to the canonical schema.

This maintenance script repairs dataset shape before validation or database
synchronization. It changes evaluation metadata only, never vector chunks.
"""

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.schema import (
    EVALUATION_DATASET_COLUMNS,
    has_supported_evaluation_columns,
    parse_json_string_array,
    parse_parallel_evidence,
    serialize_string_array,
)


DEFAULT_PATH = PROJECT_ROOT / "data" / "evaluation_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the evaluation CSV to the multi-chunk evaluation schema.",
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

    if args.check and not has_supported_evaluation_columns(actual_columns):
        raise ValueError(
            "Evaluation CSV columns must exactly match, in order: "
            f"{list(EVALUATION_DATASET_COLUMNS)}. Found: {actual_columns}"
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

        try:
            aliases = parse_json_string_array(
                row.get("accepted_answers_json"),
                field_name="accepted_answers_json",
                strict=True,
                deduplicate=True,
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid accepted answers at row {line_number}: {error}"
            ) from error

        item = {
            column: str(row.get(column) or "").strip()
            for column in EVALUATION_DATASET_COLUMNS
        }
        item["query"] = query
        item["accepted_answers_json"] = serialize_string_array(aliases)
        try:
            evidence = parse_parallel_evidence(
                item,
                context=f"row {line_number}",
                require_document_membership=False,
            )
        except ValueError as error:
            raise ValueError(
                "Evidence and chunk ID arrays must be non-empty parallel "
                f"string arrays at row {line_number}: {error}"
            ) from error
        snippet_values = evidence.snippets
        chunk_id_values = evidence.chunk_ids
        item["expected_context_snippets_json"] = serialize_string_array(
            snippet_values
        )
        item["source_chunk_ids_json"] = serialize_string_array(
            chunk_id_values
        )
        missing = [
            column
            for column in EVALUATION_DATASET_COLUMNS
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
        if any(
            not chunk_id.startswith(f"{document_id}:")
            for chunk_id in chunk_id_values
        ):
            raise ValueError(
                f"source_chunk_ids_json contains a chunk from another "
                f"document at row {line_number}"
            )
        cleaned.append(item)

    print(f"Rows: {len(cleaned)}")
    print("Schema: " + ",".join(EVALUATION_DATASET_COLUMNS))
    if args.check:
        print("Result: PASSED (no file changes)")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = args.path.with_name(f"{args.path.stem}.before-schema-{timestamp}.csv")
    shutil.copy2(args.path, backup)

    temporary = args.path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=EVALUATION_DATASET_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(cleaned)
    temporary.replace(args.path)

    print(f"Backup: {backup}")
    print("Result: NORMALIZED")


if __name__ == "__main__":
    main()
