from __future__ import annotations

import json
import os
import re
from copy import deepcopy


# Add new types here, or append/override them at deployment time with
# DOCUMENT_TYPE_RULES_JSON. Patterns are regular expressions, matched
# case-insensitively against extracted text.
DOCUMENT_TYPE_RULES = {
    "faq": {
        "label": "FAQ / question and answer",
        "description": "Question-and-answer material. Each numbered Q&A block is embedded separately.",
        "chunk_strategy": "faq_entries",
        "admin_selectable": True,
        "patterns": [
            r"\bquestion\s*:.*?\banswer\s*:",
            r"(?:^|\n)\s*q(\d+)\s*[:.)].*?(?:^|\n)\s*a\1\s*[:.)]",
        ],
    },
    "directory": {
        "label": "Directory",
        "description": "Branch, contact, or service listings that should be split into directory entries.",
        "chunk_strategy": "directory_sections",
        "admin_selectable": True,
        "patterns": [r"\bdistrict\b.*\bpublic library\b.*\btel\s*:"],
    },
    "announcement": {
        "label": "Announcement / notice",
        "description": "Time-sensitive notices, closures, suspensions, and service announcements.",
        "chunk_strategy": "announcement_entries",
        "admin_selectable": True,
        "match_title_or_heading": True,
        "title_or_heading_only": True,
        "patterns": [
            r"\bannouncements?\b",
            r"\bnotices?\b",
            r"\btemporary closure\b",
            r"\bsuspension\b",
        ],
    },
    "policy": {
        "label": "Policy / rules",
        "description": "Policies, rules, regulations, and guidance documents.",
        "chunk_strategy": "prose",
        "admin_selectable": True,
        "patterns": [
            r"\bpolicy\b",
            r"\brules?\b",
            r"\bregulations?\b",
            r"\bguidelines?\b",
        ],
    },
    "structured": {
        "label": "Structured sections",
        "description": "Documents with Markdown-style section headings; each section is prepared separately.",
        "chunk_strategy": "marked_sections",
        "admin_selectable": False,
        "patterns": [r"(?:^|\n)##\s+"],
    },
    "record_based": {
        "label": "Record-based data",
        "description": "CSV, spreadsheet, JSON, or XML records that should remain atomic where possible.",
        "chunk_strategy": "atomic",
        "admin_selectable": False,
        "file_types": ["csv", "xlsx", "xlsm", "json", "jsonl", "xml"],
        "patterns": [],
    },
    "news": {
        "label": "News article",
        "description": "One article per document, split at sentence boundaries with its title repeated in every chunk.",
        "chunk_strategy": "prose",
        "admin_selectable": False,
        "patterns": [],
    },
    "prose": {
        "label": "General prose",
        "description": "Narrative webpages, PDFs, and documents split into overlapping prose chunks.",
        "chunk_strategy": "prose",
        "admin_selectable": True,
        "patterns": [],
    },
}


def get_document_type_rules() -> dict:
    rules = deepcopy(DOCUMENT_TYPE_RULES)
    raw_overrides = os.getenv("DOCUMENT_TYPE_RULES_JSON", "").strip()
    if not raw_overrides:
        return rules

    overrides = json.loads(raw_overrides)
    if not isinstance(overrides, dict):
        raise ValueError("DOCUMENT_TYPE_RULES_JSON must contain a JSON object.")

    for name, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"Document type rule {name!r} must be a JSON object.")
        rules.setdefault(name, {}).update(values)
    return rules


def document_type_options() -> list[dict]:
    options = [{
        "value": "auto",
        "label": "Automatic detection (recommended)",
        "description": "Detect the type from file format and extracted text.",
        "upload_endpoint": "/admin/documents/upload",
    }]
    for name, rule in get_document_type_rules().items():
        if not rule.get("admin_selectable", False):
            continue
        options.append({
            "value": name,
            "label": rule.get("label", name.replace("_", " ").title()),
            "description": rule.get("description", ""),
            "chunk_strategy": rule.get("chunk_strategy", "prose"),
            "upload_endpoint": "/admin/documents/upload",
        })
    return options


def validate_document_type(document_type: str | None) -> str:
    normalized = (document_type or "auto").strip().lower()
    if normalized == "auto" or normalized in get_document_type_rules():
        return normalized
    allowed = ", ".join(["auto", *get_document_type_rules().keys()])
    raise ValueError(f"Unknown document_type {document_type!r}. Allowed values: {allowed}")


def detect_document_type(text: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    explicit = validate_document_type(metadata.get("document_type"))
    if explicit != "auto":
        return explicit

    file_type = str(metadata.get("file_type", "")).lower().lstrip(".")
    rules = get_document_type_rules()

    # File formats with explicit record semantics take precedence over words
    # appearing inside a row. For example, a CSV answer may contain the text
    # "Question:" without making the whole CSV an FAQ webpage.
    for name, rule in rules.items():
        if file_type and file_type in rule.get("file_types", []):
            return name

    for name, rule in rules.items():
        patterns = rule.get("patterns", [])
        if not patterns:
            continue

        title_and_heading = "\n".join(filter(None, [
            str(metadata.get("source_title", "")),
            str(metadata.get("section_heading", "")),
        ]))
        if rule.get("match_title_or_heading") and any(
            re.search(pattern, title_and_heading, re.IGNORECASE | re.DOTALL)
            for pattern in patterns
        ):
            return name
        if rule.get("title_or_heading_only"):
            continue

        minimum_matches = int(rule.get("minimum_text_matches", 1))
        match_count = sum(
            len(re.findall(pattern, text or "", re.IGNORECASE | re.DOTALL))
            for pattern in patterns
        )
        if match_count >= minimum_matches:
            return name
    return "prose"


def chunk_strategy_for(document_type: str) -> str:
    return get_document_type_rules().get(document_type, {}).get("chunk_strategy", "prose")
