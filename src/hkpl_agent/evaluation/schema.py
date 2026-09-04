"""Define canonical parsing rules for HKPL evaluation dataset rows.

Evaluation rows retain legacy singular evidence fields for spreadsheet review
and parallel JSON arrays for questions supported by multiple chunks. This
module is the single source of truth for column order and for converting those
JSON/JSONB values into validated Python string lists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EVALUATION_DATASET_COLUMNS: tuple[str, ...] = (
    "domain",
    "query",
    "expected_answer_text",
    "expected_context_snippet",
    "expected_context_snippets_json",
    "accepted_answers_json",
    "source_title",
    "source_url",
    "source_document_id",
    "source_chunk_id",
    "source_chunk_ids_json",
)
"""Canonical evaluation CSV columns, in their required file order."""

LEGACY_EVALUATION_DATASET_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in EVALUATION_DATASET_COLUMNS
    if column not in {"expected_context_snippets_json", "source_chunk_ids_json"}
)
"""Former single-chunk CSV columns accepted during migration and resume."""

SUPPORTED_EVALUATION_COLUMN_SETS: tuple[tuple[str, ...], ...] = (
    EVALUATION_DATASET_COLUMNS,
    LEGACY_EVALUATION_DATASET_COLUMNS,
)
"""All column layouts readers may accept without guessing field meanings."""


@dataclass(frozen=True)
class EvidenceLabels:
    """Validated positional evidence snippets and their matching chunk IDs."""

    snippets: list[str]
    chunk_ids: list[str]


def normalize_evaluation_text(value: str) -> str:
    """Normalize case, whitespace, and quotation marks for label comparison."""

    normalized = str(value or "").casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return re.sub(r"[\"'“”‘’]", "", normalized)


def has_supported_evaluation_columns(columns: Sequence[str]) -> bool:
    """Return whether ``columns`` exactly match a supported ordered schema."""

    return tuple(columns) in SUPPORTED_EVALUATION_COLUMN_SETS


def parse_json_string_array(
    value: Any,
    *,
    field_name: str,
    fallback: Sequence[str] = (),
    strict: bool = True,
    require_non_empty: bool = False,
    deduplicate: bool = False,
) -> list[str]:
    """Parse a CSV JSON string or PostgreSQL JSONB value into clean strings.

    ``fallback`` supports legacy singular fields when the JSON-array field is
    absent. In strict mode malformed JSON, non-list values, and blank/non-string
    members raise ``ValueError``. Tolerant mode filters invalid members and uses
    the fallback when no valid values remain. Deduplication, when requested,
    preserves first-occurrence order.
    """

    missing = value is None or (isinstance(value, str) and not value.strip())
    if missing:
        values: Any = list(fallback)
    elif isinstance(value, str):
        try:
            values = json.loads(value)
        except json.JSONDecodeError as error:
            if strict:
                raise ValueError(f"Invalid {field_name}: {error}") from error
            values = []
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        if strict:
            raise ValueError(f"{field_name} must be a JSON array of strings")
        values = []

    if not isinstance(values, list):
        if strict:
            raise ValueError(f"{field_name} must be a JSON array of strings")
        values = []

    if strict and any(
        not isinstance(item, str) or not item.strip()
        for item in values
    ):
        raise ValueError(
            f"{field_name} must be a JSON array of non-empty strings"
        )

    cleaned = [
        item.strip()
        for item in values
        if isinstance(item, str) and item.strip()
    ]
    if not strict and not cleaned:
        cleaned = [
            item.strip()
            for item in fallback
            if isinstance(item, str) and item.strip()
        ]
    if deduplicate:
        cleaned = list(dict.fromkeys(cleaned))
    if require_non_empty and not cleaned:
        raise ValueError(f"{field_name} must contain at least one string")
    return cleaned


def parse_parallel_evidence(
    row: Mapping[str, Any],
    *,
    context: str,
    strict: bool = True,
    deduplicate_pairs: bool = False,
    require_primary_match: bool = True,
    require_document_membership: bool = True,
) -> EvidenceLabels:
    """Validate the one-to-one relationship between evidence and chunk IDs.

    Legacy rows fall back to ``expected_context_snippet`` and
    ``source_chunk_id``. Array order is retained because each snippet at index
    ``n`` must describe the chunk ID at index ``n``. Optional pair-level
    deduplication removes only duplicate ``(chunk_id, snippet)`` pairs and can
    never misalign the two arrays.
    """

    primary_snippet = str(row.get("expected_context_snippet") or "").strip()
    primary_chunk_id = str(row.get("source_chunk_id") or "").strip()
    snippets = parse_json_string_array(
        row.get("expected_context_snippets_json"),
        field_name="expected_context_snippets_json",
        fallback=[primary_snippet],
        strict=strict,
        require_non_empty=True,
    )
    chunk_ids = parse_json_string_array(
        row.get("source_chunk_ids_json"),
        field_name="source_chunk_ids_json",
        fallback=[primary_chunk_id],
        strict=strict,
        require_non_empty=True,
    )
    if len(snippets) != len(chunk_ids):
        raise ValueError(
            "expected_context_snippets_json and source_chunk_ids_json must "
            f"be parallel arrays for {context}."
        )

    if deduplicate_pairs:
        # Length equality was checked above, so plain ``zip`` is safe and
        # keeps this schema-only helper usable by lightweight audit tooling.
        pairs = list(dict.fromkeys(zip(chunk_ids, snippets)))
        chunk_ids = [chunk_id for chunk_id, _ in pairs]
        snippets = [snippet for _, snippet in pairs]

    if require_primary_match and (
        snippets[0] != primary_snippet or chunk_ids[0] != primary_chunk_id
    ):
        raise ValueError(
            "Singular evidence fields must equal the first items in their "
            f"JSON arrays for {context}."
        )

    document_id = str(row.get("source_document_id") or "").strip()
    if require_document_membership and document_id and any(
        not chunk_id.startswith(f"{document_id}:") for chunk_id in chunk_ids
    ):
        raise ValueError(
            "Every source_chunk_ids_json value must belong to "
            f"source_document_id for {context}."
        )
    return EvidenceLabels(snippets=snippets, chunk_ids=chunk_ids)


def parse_accepted_answers(value: Any, primary_answer: str) -> list[str]:
    """Return a primary answer followed by tolerant, unique answer aliases."""

    aliases = parse_json_string_array(
        value,
        field_name="accepted_answers_json",
        strict=False,
    )
    answers: list[str] = []
    for raw_answer in (primary_answer, *aliases):
        answer = str(raw_answer).strip()
        if answer and answer not in answers:
            answers.append(answer)
    return answers


def serialize_string_array(values: Sequence[str]) -> str:
    """Serialize validated strings as compact, Unicode-preserving JSON."""

    return json.dumps(list(values), ensure_ascii=False)
