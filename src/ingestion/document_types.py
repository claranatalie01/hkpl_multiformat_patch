"""Validate source labels and select deterministic structure-aware policies."""

from __future__ import annotations

from typing import Final


# These labels exist only where they change ingestion behavior. Domain metadata
# such as "event" or "policy" belongs in the existing category field.
DOCUMENT_TYPES: Final[dict[str, tuple[str, str]]] = {
    "faq": ("FAQ / Q&A", "Keep each question with its answer."),
    "record": ("Atomic record", "Keep one notice, event, or branch profile together."),
    "prose": ("Structured prose", "Chunk by Docling headings and document structure."),
}
CLASSIFIER_TYPES: Final[tuple[str, ...]] = ("faq", "record", "prose", "skip")

LEGACY_ALIASES: Final[dict[str, str]] = {
    "announcement": "record",
    "notice": "record",
    "event": "record",
    "directory": "record",
    "branch": "record",
    "policy": "prose",
    "news": "prose",
    "structured": "prose",
    "record_based": "table",
}


def document_type_options() -> list[dict[str, str]]:
    options = [{
        "value": "auto",
        "label": "Unlabelled (9B classification)",
        "description": "Classify extracted content with the non-reasoning generation model.",
        "chunk_strategy": "llm_classified",
        "upload_endpoint": "/admin/documents/upload",
    }]
    for value, (label, description) in DOCUMENT_TYPES.items():
        options.append({
            "value": value,
            "label": label,
            "description": description,
            "chunk_strategy": chunk_policy_for(value),
            "upload_endpoint": "/admin/documents/upload",
        })
    return options


def validate_document_type(document_type: str | None) -> str:
    value = (document_type or "auto").strip().lower()
    allowed = {"auto", "table", "skip", *DOCUMENT_TYPES, *LEGACY_ALIASES}
    if value not in allowed:
        raise ValueError(
            f"Unknown document_type {document_type!r}. Allowed values: "
            f"{', '.join(sorted(allowed))}"
        )
    return value


def normalize_document_type(document_type: str | None) -> str:
    value = validate_document_type(document_type)
    return LEGACY_ALIASES.get(value, value)


def resolve_record_kind(
    metadata: dict | None = None,
    *,
    structural_kind: str | None = None,
) -> str:
    metadata = metadata or {}
    physical = str(structural_kind or metadata.get("structural_kind") or "").lower()
    if physical in {"table", "table_row"}:
        return "table"
    selected = normalize_document_type(
        str(metadata.get("document_type") or metadata.get("record_kind") or "auto")
    )
    return "prose" if selected == "auto" else selected


def chunk_policy_for(document_type: str, structural_kind: str | None = None) -> str:
    selected = normalize_document_type(document_type)
    physical = str(structural_kind or "").lower()
    if selected == "auto":
        return "fallback"
    if selected == "skip":
        return "no_chunks"
    if physical in {"table", "table_row"} or selected == "table":
        return "table_rows"
    if selected in {"faq", "record"}:
        return "atomic_record"
    return "hierarchical"
