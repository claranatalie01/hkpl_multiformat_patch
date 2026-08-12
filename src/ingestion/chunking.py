"""Convert structured evidence records into stable, token-bounded vector nodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Iterable

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode

from .document_types import chunk_policy_for, resolve_record_kind
from .tokenizer import DEFAULT_MAX_TOKENS, get_embedding_tokenizer


MAX_TOKENS = int(os.getenv("CHUNK_SIZE", str(DEFAULT_MAX_TOKENS)))
FALLBACK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
CHUNKER_VERSION = os.getenv("CHUNKER_VERSION", "hkpl-structure-v2")


def get_text(document: Document) -> str:
    return getattr(document, "text", None) or document.get_content()


def normalize_search_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return []


def build_search_text(metadata: dict[str, Any], evidence_text: str) -> str:
    """Build retrieval text from useful source context only."""
    components: list[str] = []

    def add(value: str | None) -> None:
        value = normalize_search_text(value or "")
        if value and value not in components:
            components.append(value)

    add(str(metadata.get("source_title") or ""))
    for heading in _text_list(metadata.get("structure_path")):
        add(heading)
    for alias in _text_list(metadata.get("search_aliases")):
        add(alias)
    header = normalize_search_text(str(metadata.get("record_header") or ""))
    normalized_evidence = normalize_search_text(evidence_text)
    if header and not normalized_evidence.startswith(header):
        add(header)
    add(evidence_text)
    return "\n\n".join(components)


def _locator(metadata: dict[str, Any]) -> dict[str, Any]:
    existing = metadata.get("locator")
    if isinstance(existing, dict) and existing:
        return existing
    for key, locator_type in (
        ("page_number", "page"),
        ("slide_number", "slide"),
        ("row_number", "row"),
        ("section_index", "section"),
    ):
        if metadata.get(key) is not None:
            return {"type": locator_type, locator_type: metadata[key]}
    return {"type": "document"}


def _token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    underlying = tokenizer.get_tokenizer()
    encoded = underlying(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if not isinstance(offsets, list):
        raise ValueError("The embedding tokenizer must provide character offsets.")
    return [
        (int(start), int(end))
        for start, end in offsets
        if int(end) > int(start)
    ]


def _split_evidence(
    evidence_text: str,
    metadata: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
) -> list[str]:
    search_text = build_search_text(metadata, evidence_text)
    if tokenizer.count_tokens(search_text) <= max_tokens:
        return [evidence_text]

    repeat_context = str(
        metadata.get("repeat_context")
        or metadata.get("question")
        or metadata.get("table_header")
        or ""
    ).strip()
    body = evidence_text
    if repeat_context and normalize_search_text(body).startswith(
        normalize_search_text(repeat_context)
    ):
        body = body[len(repeat_context):].lstrip("\r\n :")

    probe = "\n\n".join(part for part in (repeat_context, "x") if part)
    prefix_tokens = tokenizer.count_tokens(build_search_text(metadata, probe)) - 1
    available = max_tokens - prefix_tokens
    if available <= FALLBACK_OVERLAP:
        raise ValueError("Document context leaves no room for a 512/64 fallback chunk.")

    offsets = _token_offsets(tokenizer, body)
    if not offsets:
        return [evidence_text]
    parts: list[str] = []
    step = available - FALLBACK_OVERLAP
    for start in range(0, len(offsets), step):
        end = min(start + available, len(offsets))
        part = body[offsets[start][0]:offsets[end - 1][1]]
        if part.strip():
            parts.append(part)
        if end == len(offsets):
            break
    if repeat_context:
        parts = [f"{repeat_context}\n{part}" for part in parts]
    return parts


def _chunk_id(
    metadata: dict[str, Any],
    locator: dict[str, Any],
    part_number: int,
    evidence_text: str,
) -> str:
    source_version = str(
        metadata.get("source_version_id")
        or f"{metadata.get('kb_document_id') or metadata.get('document_id') or 'source'}"
        f":v{metadata.get('document_version', 1)}"
    )
    locator_json = json.dumps(locator, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    locator_hash = hashlib.sha256(locator_json.encode("utf-8")).hexdigest()[:12]
    evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()[:16]
    return f"{source_version}:l{locator_hash}:p{part_number}:{evidence_hash}"


def chunk_documents(
    documents: list[Document],
    *,
    tokenizer: Any | None = None,
    max_tokens: int = MAX_TOKENS,
) -> list[BaseNode]:
    """Build exact-evidence nodes with stable provenance and retrieval text."""
    tokenizer = tokenizer or get_embedding_tokenizer(max_tokens)
    nodes: list[BaseNode] = []

    for document in documents:
        metadata = dict(document.metadata or {})
        evidence_text = str(metadata.get("evidence_text") or get_text(document)).strip()
        if not evidence_text:
            continue

        structural_kind = str(metadata.get("structural_kind") or "")
        selected_type = str(metadata.get("document_type") or "auto")
        record_kind = resolve_record_kind(metadata, structural_kind=structural_kind)
        policy = str(
            metadata.get("chunk_policy")
            or chunk_policy_for(selected_type, structural_kind)
        )
        locator = _locator(metadata)
        parent_record_id = str(metadata.get("parent_record_id") or "")
        if not parent_record_id:
            identity = (
                f"{metadata.get('source_version_id') or metadata.get('document_id')}|"
                f"{json.dumps(locator, ensure_ascii=False, sort_keys=True)}"
            )
            parent_record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

        metadata.update({
            "dataset": metadata.get("dataset") or "hkpl",
            "corpus": metadata.get("corpus") or "hkpl",
            "corpus_role": metadata.get("corpus_role") or "primary",
            "record_kind": record_kind,
            "chunk_policy": policy,
            "structure_path": _text_list(metadata.get("structure_path")),
            "locator": locator,
            "branch_ids": _text_list(metadata.get("branch_ids")),
            "parser_version": metadata.get("parser_version") or "deterministic-v2",
            "chunker_version": CHUNKER_VERSION,
            "parent_record_id": parent_record_id,
            "classification_source": metadata.get("classification_source") or (
                "fallback" if selected_type == "auto" else "librarian"
            ),
        })

        parts = _split_evidence(
            evidence_text,
            metadata,
            tokenizer,
            max_tokens,
        )
        for part_number, evidence_part in enumerate(parts, start=1):
            search_text = build_search_text(metadata, evidence_part)
            token_count = int(tokenizer.count_tokens(search_text))
            if token_count > max_tokens:
                raise ValueError(f"Chunk exceeds {max_tokens} tokens: {token_count}")
            part_metadata = {
                **metadata,
                "evidence_text": evidence_part,
                "search_text": search_text,
                "chunk_policy": "oversized_leaf" if len(parts) > 1 else policy,
                "part_number": part_number,
                "part_count": len(parts),
                "token_count": token_count,
                "chunk_size": max_tokens,
                "chunk_overlap": FALLBACK_OVERLAP if len(parts) > 1 else 0,
            }
            chunk_id = _chunk_id(part_metadata, locator, part_number, evidence_part)
            part_metadata.update({"chunk_id": chunk_id, "chunk_index": part_number - 1})
            excluded_keys = list(part_metadata)
            nodes.append(TextNode(
                id_=chunk_id,
                text=search_text,
                metadata=part_metadata,
                excluded_embed_metadata_keys=excluded_keys,
                excluded_llm_metadata_keys=excluded_keys,
            ))
    return nodes
