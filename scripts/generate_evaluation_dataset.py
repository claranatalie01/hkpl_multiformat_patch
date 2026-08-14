#!/usr/bin/env python3
"""Generate reviewable evaluation questions from HKPL vector or preview chunks.

By default, eligible primary-corpus chunks are read from
``data_hkpl_knowledge``. Passing ``--preview-run-id`` instead reads the
non-embedded candidates in ``ingestion_preview_chunks`` for one explicit
preview run. Chunks are sent to the answer model to propose questions, accepted
answers, exact evidence, and source IDs. Progress is checkpointed; output
remains a candidate until reviewed and promoted. This script does not create
document embeddings.
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.llm_client import http_llm


OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation_dataset.csv"

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 1800
QUESTIONS_PER_CHUNK = 1
MAX_CHUNKS_PER_DOCUMENT = 8
FIELDNAMES = [
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
]
LEGACY_FIELDNAMES = [
    column for column in FIELDNAMES
    if column not in {"expected_context_snippets_json", "source_chunk_ids_json"}
]


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[\"'“”‘’]", "", value)
    return value


def clean_json_response(raw: str):
    raw = raw.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def row_source_chunk_ids(row: dict) -> list[str]:
    """Return every evidence chunk consumed by one evaluation row.

    New rows store all evidence IDs in ``source_chunk_ids_json``. Legacy rows
    have only ``source_chunk_id``. Invalid optional JSON falls back to the
    singular ID so resume checkpoints remain safe and backward compatible.
    """
    raw_ids = row.get("source_chunk_ids_json")
    try:
        values = json.loads(raw_ids) if raw_ids else []
    except (json.JSONDecodeError, TypeError):
        values = []
    chunk_ids = [
        str(value).strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]
    primary_id = str(row.get("source_chunk_id") or "").strip()
    if not chunk_ids and primary_id:
        chunk_ids = [primary_id]
    return list(dict.fromkeys(chunk_ids))


def infer_domain(title: str, text_value: str) -> str:
    joined = f"{title} {text_value}".lower()

    if "opening hour" in joined or "public holidays" in joined:
        return "opening_hours"
    if "event" in joined or "activities" in joined or "venue:" in joined:
        return "events"
    if "library list" in joined or "district" in joined:
        return "branch_information"
    if "password" in joined:
        return "account_password"
    if "e-resource" in joined or "e-book" in joined or "ebook" in joined:
        return "e_resources"
    if "notice" in joined or "temporary closure" in joined:
        return "notices"
    if "hong kong central library" in joined:
        return "hkcl_information"

    return "general"


def load_vector_chunks(
    *,
    excluded_chunk_ids: set[str] | None = None,
    limit: int | None = None,
    max_chunks_per_document: int | None = MAX_CHUNKS_PER_DOCUMENT,
) -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                WITH ranked_chunks AS (
                    SELECT
                        split_part(metadata_->>'chunk_id', ':', 1) AS document_id,
                        COALESCE(metadata_->>'chunk_id', '') AS chunk_id,
                        COALESCE(metadata_->>'source_title', '') AS source_title,
                        COALESCE(metadata_->>'source_url', metadata_->>'url', '') AS source_url,
                        COALESCE(metadata_->>'file_name', '') AS file_name,
                        COALESCE(metadata_->>'section_heading', '') AS section_heading,
                        text,
                        ROW_NUMBER() OVER (
                            PARTITION BY split_part(metadata_->>'chunk_id', ':', 1)
                            ORDER BY metadata_->>'chunk_id'
                        ) AS rn
                    FROM data_hkpl_knowledge
                    WHERE text IS NOT NULL
                      AND LENGTH(TRIM(text)) >= :min_chars
                      AND COALESCE(NULLIF(metadata_->>'dataset', ''), 'hkpl') = 'hkpl'
                      AND COALESCE(NULLIF(metadata_->>'corpus_role', ''), 'primary') = 'primary'
                )
                SELECT *
                FROM ranked_chunks
                WHERE (
                    :max_chunks_per_document IS NULL
                    OR rn <= :max_chunks_per_document
                )
                ORDER BY document_id, chunk_id
            """),
            {
                "min_chars": MIN_CHUNK_CHARS,
                "max_chunks_per_document": max_chunks_per_document,
            },
        ).fetchall()

    chunks = []

    for row in rows:
        item = dict(row._mapping)
        source_title = item.get("source_title") or item.get("file_name") or "HKPL knowledge base"

        chunks.append(
            {
                "document_id": item.get("document_id") or "",
                "chunk_id": item.get("chunk_id") or "",
                "source_title": source_title,
                "source_url": item.get("source_url") or "",
                "file_name": item.get("file_name") or "",
                "section_heading": item.get("section_heading") or "",
                "text": item["text"][:MAX_CHUNK_CHARS],
            }
        )

    excluded_chunk_ids = excluded_chunk_ids or set()
    chunks = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in excluded_chunk_ids
    ]
    return chunks[:limit] if limit is not None else chunks


def load_preview_chunks(
    preview_run_id: str,
    *,
    excluded_chunk_ids: set[str] | None = None,
    limit: int | None = None,
    max_chunks_per_document: int | None = MAX_CHUNKS_PER_DOCUMENT,
) -> list[dict]:
    """Load exact evidence from one completed, non-embedded preview run.

    Preview rows deliberately use ``evidence_text`` rather than ``search_text``.
    The latter may prepend titles or headings for retrieval, while evaluation
    snippets must remain exact quotations from the underlying evidence.
    """
    with engine.connect() as connection:
        run_summary = connection.execute(
            text("""
                SELECT
                    COUNT(*) AS documents,
                    COUNT(*) FILTER (WHERE status = 'completed') AS completed
                FROM ingestion_preview_documents
                WHERE run_id = :run_id
            """),
            {"run_id": preview_run_id},
        ).mappings().one()
        if int(run_summary["documents"]) == 0:
            raise ValueError(
                f"Preview run {preview_run_id!r} was not found in "
                "ingestion_preview_documents."
            )
        if int(run_summary["completed"]) == 0:
            raise ValueError(
                f"Preview run {preview_run_id!r} has no completed documents."
            )

        rows = connection.execute(
            text("""
                WITH ranked_chunks AS (
                    SELECT
                        c.document_id,
                        c.chunk_id,
                        d.source_title,
                        d.source_url,
                        d.file_name,
                        COALESCE(c.structure_path->>-1, '') AS section_heading,
                        c.evidence_text AS text,
                        ROW_NUMBER() OVER (
                            PARTITION BY c.document_id
                            ORDER BY c.ordinal, c.chunk_id
                        ) AS rn
                    FROM ingestion_preview_chunks c
                    JOIN ingestion_preview_documents d
                      ON d.run_id = c.run_id
                     AND d.document_id = c.document_id
                    WHERE c.run_id = :run_id
                      AND d.status = 'completed'
                      AND c.evidence_text IS NOT NULL
                      AND LENGTH(TRIM(c.evidence_text)) >= :min_chars
                )
                SELECT *
                FROM ranked_chunks
                WHERE (
                    :max_chunks_per_document IS NULL
                    OR rn <= :max_chunks_per_document
                )
                ORDER BY document_id, rn, chunk_id
            """),
            {
                "run_id": preview_run_id,
                "min_chars": MIN_CHUNK_CHARS,
                "max_chunks_per_document": max_chunks_per_document,
            },
        ).fetchall()

    chunks = []
    for row in rows:
        item = dict(row._mapping)
        chunks.append({
            "document_id": str(item.get("document_id") or ""),
            "chunk_id": str(item.get("chunk_id") or ""),
            "source_title": (
                item.get("source_title")
                or item.get("file_name")
                or "HKPL preview"
            ),
            "source_url": item.get("source_url") or "",
            "file_name": item.get("file_name") or "",
            "section_heading": item.get("section_heading") or "",
            "text": str(item["text"])[:MAX_CHUNK_CHARS],
        })

    excluded_chunk_ids = excluded_chunk_ids or set()
    chunks = [
        chunk for chunk in chunks
        if chunk["chunk_id"] not in excluded_chunk_ids
    ]
    return chunks[:limit] if limit is not None else chunks


def load_existing_rows(output_file: Path) -> list[dict]:
    if not output_file.is_file():
        raise FileNotFoundError(
            f"Cannot resume because the candidate does not exist: {output_file}"
        )
    with output_file.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        actual_columns = list(reader.fieldnames or [])
        if actual_columns not in (FIELDNAMES, LEGACY_FIELDNAMES):
            raise ValueError(
                f"Cannot resume candidate with columns {actual_columns}; "
                f"expected {FIELDNAMES}."
            )
        rows = [dict(row) for row in reader if row.get("query")]
    for row in rows:
        row.setdefault("accepted_answers_json", "[]")
        row.setdefault(
            "expected_context_snippets_json",
            json.dumps([row["expected_context_snippet"]], ensure_ascii=False),
        )
        row.setdefault(
            "source_chunk_ids_json",
            json.dumps([row["source_chunk_id"]], ensure_ascii=False),
        )
    return rows


def save_rows(output_file: Path, rows: list[dict]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_file)


def progress_file(output_file: Path) -> Path:
    return output_file.with_suffix(output_file.suffix + ".progress.json")


def save_progress(
    output_file: Path,
    processed_chunk_ids: set[str],
    *,
    all_chunks: bool,
    limit_chunks: int | None,
    target_questions: int | None,
    preview_run_id: str | None,
) -> None:
    path = progress_file(output_file)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "all_chunks": all_chunks,
                "limit_chunks": limit_chunks,
                "target_questions": target_questions,
                "preview_run_id": preview_run_id,
                "processed_chunk_ids": sorted(processed_chunk_ids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_progress(
    output_file: Path,
    existing_rows: list[dict],
    *,
    all_chunks: bool,
    limit_chunks: int | None,
    target_questions: int | None,
    preview_run_id: str | None,
) -> set[str]:
    path = progress_file(output_file)
    if not path.is_file():
        return {
            str(row.get("source_chunk_id") or "")
            for row in existing_rows
            if row.get("source_chunk_id")
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("all_chunks") != all_chunks:
        raise ValueError(
            "Resume options differ from the original run: --all-chunks mismatch."
        )
    if payload.get("limit_chunks") != limit_chunks:
        raise ValueError(
            "Resume options differ from the original run: --limit-chunks mismatch."
        )
    if payload.get("target_questions") != target_questions:
        raise ValueError(
            "Resume options differ from the original run: "
            "--target-questions mismatch."
        )
    if payload.get("preview_run_id") != preview_run_id:
        raise ValueError(
            "Resume options differ from the original run: "
            "--preview-run-id mismatch."
        )
    chunk_ids = payload.get("processed_chunk_ids")
    if not isinstance(chunk_ids, list) or not all(
        isinstance(chunk_id, str) and chunk_id for chunk_id in chunk_ids
    ):
        raise ValueError(f"Invalid progress checkpoint: {path}")
    return set(chunk_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation candidates from HKPL vector chunks."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted candidate using saved chunk checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=(
            "Candidate CSV path. Use a candidate filename during review; "
            f"default: {OUTPUT_FILE}"
        ),
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Generate from at most this many new HKPL chunks.",
    )
    parser.add_argument(
        "--target-questions",
        type=int,
        default=None,
        help=(
            "Stop after this many deduplicated evaluation questions have "
            "been generated. More chunks may be processed because invalid "
            "or duplicate candidates do not count toward the target."
        ),
    )
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help=(
            "Generate candidates from every eligible primary HKPL chunk. "
            "This is slower and still requires human label review."
        ),
    )
    parser.add_argument(
        "--preview-run-id",
        default=None,
        help=(
            "Read chunks from this exact ingestion preview run instead of "
            "data_hkpl_knowledge. Always use a separate --output file."
        ),
    )
    args = parser.parse_args()
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be positive")
    if args.target_questions is not None and args.target_questions < 1:
        parser.error("--target-questions must be positive")
    if args.preview_run_id and args.output.resolve() == OUTPUT_FILE.resolve():
        parser.error(
            "--preview-run-id requires a separate --output path so the active "
            "evaluation_dataset.csv cannot be overwritten"
        )
    return args


async def generate_questions_for_chunk(
    chunk: dict,
    related_chunks: list[dict],
) -> list[dict]:
    domain = infer_domain(chunk["source_title"], chunk["text"])

    prompt = f"""
You are creating an evaluation dataset for a Retrieval-Augmented Generation system for Hong Kong Public Libraries.

Generate exactly {QUESTIONS_PER_CHUNK} factual evaluation question from the official HKPL chunks below.

The first chunk is the anchor. The remaining chunks are sibling chunks from
the SAME webpage/document and are supplied so that repeated sessions are not
mistaken for separate, incomplete facts.

VERY IMPORTANT RULES:
- The question must have ONE and ONLY ONE complete correct answer.
- Search ALL supplied chunks for other occurrences of the same named event,
  workshop, exhibition, branch session, service, or venue before writing the
  answer.
- If the question asks about a named item without limiting it to one session,
  expected_answer_text MUST combine every matching date, time, venue, and
  branch found in the supplied chunks. Include every supporting chunk ID and
  one exact evidence snippet from each supporting chunk.
- If you intentionally ask about only one occurrence of a repeated event,
  roving exhibition, workshop, branch session, or multi-venue activity, the
  question MUST state the exact venue/branch AND date/month that identifies
  that occurrence.
- Do NOT generate generic repeated questions such as:
  "When and where is the roving exhibition held?"
  "When and where is this event held?"
  "Where is the activity held?"
- Instead, generate specific questions such as:
  "When is the roving exhibition titled 'Blissful Moments Between Pages' held at Sham Shui Po Public Library?"
  "What are the dates for 'Blissful Moments Between Pages' at Ma On Shan Public Library?"
- Each question must be answerable only from the supplied chunks.
- Do not invent facts.
- Avoid vague questions.
- Prefer useful public-service questions.
- The expected_answer_text must be concise but complete.
- expected_context_snippets must contain exact contiguous phrases copied from
  their corresponding source_chunk_ids. The two arrays must be parallel and
  must include every chunk needed for the complete answer.
- Return ONLY valid JSON array.
- Do not include markdown.

JSON format:
[
  {{
    "domain": "{domain}",
    "query": "...",
    "expected_answer_text": "...",
    "expected_context_snippets": ["..."],
    "source_title": "{chunk['source_title']}",
    "source_url": "{chunk['source_url']}",
    "source_document_id": "{chunk['document_id']}",
    "source_chunk_ids": ["{chunk['chunk_id']}"]
  }}
]

Official HKPL chunks:
{chr(10).join(
    f'''--- CHUNK {index + 1} ---
Chunk ID: {related["chunk_id"]}
Source title: {related["source_title"]}
Section: {related["section_heading"]}
Text:
{related["text"]}'''
    for index, related in enumerate(related_chunks)
)}
"""

    raw = await http_llm(prompt, temperature=0.0, max_tokens=1400)

    try:
        items = clean_json_response(raw)
    except Exception as exc:
        print(f"Failed to parse JSON for chunk: {chunk['chunk_id']}")
        print(f"Source: {chunk['source_title']}")
        print(f"Error: {exc}")
        print(raw[:500])
        return []

    if not isinstance(items, list) or not items:
        print("  Skipped: the model returned no evaluation candidate.")
        return []

    if len(items) > QUESTIONS_PER_CHUNK:
        print(
            f"  Model returned {len(items)} candidates; only the first "
            f"{QUESTIONS_PER_CHUNK} is allowed for one anchor."
        )
        items = items[:QUESTIONS_PER_CHUNK]

    output = []

    for item in items:
        query = str(item.get("query", "")).strip()
        answer = str(item.get("expected_answer_text", "")).strip()
        raw_snippets = item.get("expected_context_snippets")
        raw_chunk_ids = item.get("source_chunk_ids")
        if not isinstance(raw_snippets, list) or not isinstance(raw_chunk_ids, list):
            print("  Rejected candidate: multi-chunk evidence arrays were missing.")
            continue
        snippets = [str(value).strip() for value in raw_snippets if str(value).strip()]
        chunk_ids = [str(value).strip() for value in raw_chunk_ids if str(value).strip()]
        available_chunks = {related["chunk_id"]: related for related in related_chunks}

        if not query or not answer or not snippets or len(snippets) != len(chunk_ids):
            print(
                "  Skipped candidate: question, answer, or evidence snippet "
                "was empty."
            )
            continue
        if not query.endswith("?") or len(query) < 15:
            print(f"  Rejected malformed question: {query!r}")
            continue
        if any(chunk_id not in available_chunks for chunk_id in chunk_ids):
            print("  Rejected candidate because it cited an unavailable chunk ID.")
            continue
        if any(
            normalize_text(snippet)
            not in normalize_text(available_chunks[chunk_id]["text"])
            for snippet, chunk_id in zip(snippets, chunk_ids, strict=True)
        ):
            print("  Rejected candidate because an evidence snippet was not present in its cited chunk.")
            continue

        # A quoted event/activity title is a reliable cross-chunk key. If that
        # title occurs in sibling chunks, all of those chunks must be labeled;
        # otherwise a one-session answer could be judged against a three-session
        # webpage, which is the exact ambiguity this generator must avoid.
        quoted_subjects = re.findall(r"[\"'“‘]([^\"'”’]{10,})[\"'”’]", query)
        question_key = normalize_text(query)
        has_exact_date_or_month = bool(re.search(
            r"\b(?:20\d{2}[/.-]\d{1,2}(?:[/.-]\d{1,2})?|"
            r"\d{1,2}[/.-]\d{1,2}|january|february|march|april|may|june|"
            r"july|august|september|october|november|december)\b",
            question_key,
        ))
        has_specific_location = bool(re.search(
            r"\b(?:[a-z]+[ -]){1,6}(?:public library|library branch)\b",
            question_key,
        ))
        session_is_explicitly_scoped = has_exact_date_or_month and has_specific_location
        incomplete_subject = None
        for subject in quoted_subjects:
            subject_key = normalize_text(subject)
            matching_ids = {
                chunk_id
                for chunk_id, related in available_chunks.items()
                if subject_key in normalize_text(related["text"])
            }
            if (
                not session_is_explicitly_scoped
                and not matching_ids.issubset(set(chunk_ids))
            ):
                incomplete_subject = subject
                break
        if incomplete_subject:
            print(
                "  Rejected candidate because the quoted subject appears in "
                f"additional sibling chunks: {incomplete_subject!r}."
            )
            continue

        # Preserve order while removing exact duplicate evidence pairs.
        evidence_pairs = list(dict.fromkeys(zip(chunk_ids, snippets, strict=True)))
        chunk_ids = [pair[0] for pair in evidence_pairs]
        snippets = [pair[1] for pair in evidence_pairs]

        output.append(
            {
                "domain": item.get("domain") or domain,
                "query": query,
                "expected_answer_text": answer,
                "expected_context_snippet": snippets[0],
                "expected_context_snippets_json": json.dumps(snippets, ensure_ascii=False),
                "accepted_answers_json": "[]",
                "source_title": item.get("source_title") or chunk["source_title"],
                "source_url": item.get("source_url") or chunk["source_url"],
                "source_document_id": chunk["document_id"],
                "source_chunk_id": chunk_ids[0],
                "source_chunk_ids_json": json.dumps(chunk_ids, ensure_ascii=False),
            }
        )

    return output


def remove_ambiguous_duplicates(rows: list[dict]) -> list[dict]:
    """Remove duplicate questions and enforce one evaluation row per chunk.

    A multi-chunk row reserves every cited evidence chunk. Later rows that
    overlap any reserved chunk are removed, even when their question wording
    differs. This prevents sibling anchors and extra model outputs from turning
    the same evidence into several evaluation examples.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(normalize_text(row["query"]), []).append(row)

    cleaned: list[dict] = []
    for group in grouped.values():
        if len(group) == 1:
            cleaned.append(max(
                group,
                key=lambda row: len(row_source_chunk_ids(row)),
            ))
            continue
        answer_keys = {
            normalize_text(row["expected_answer_text"])
            for row in group
        }
        if len(answer_keys) == 1:
            cleaned.append(max(
                group,
                key=lambda row: len(row_source_chunk_ids(row)),
            ))
            continue

        print()
        print("Removed all conflicting versions of ambiguous question:")
        print("Question:", group[0]["query"])
        for index, row in enumerate(group, start=1):
            print(f"Answer {index}:", row["expected_answer_text"])

    non_overlapping: list[dict] = []
    used_chunk_ids: set[str] = set()
    for row in cleaned:
        evidence_ids = set(row_source_chunk_ids(row))
        overlap = evidence_ids.intersection(used_chunk_ids)
        if overlap:
            print()
            print("Removed question because its evidence chunk was already used:")
            print("Question:", row["query"])
            print("Reused chunk IDs:", ", ".join(sorted(overlap)))
            continue
        non_overlapping.append(row)
        used_chunk_ids.update(evidence_ids)

    return non_overlapping


async def main() -> None:
    args = parse_args()
    output_file = args.output.resolve()
    existing_rows = load_existing_rows(output_file) if args.resume else []
    if existing_rows:
        before_resume_cleanup = len(existing_rows)
        existing_rows = remove_ambiguous_duplicates(existing_rows)
        if len(existing_rows) != before_resume_cleanup:
            print(
                "Removed "
                f"{before_resume_cleanup - len(existing_rows)} duplicate or "
                "evidence-overlapping checkpoint rows before resuming."
            )
    processed_chunk_ids = (
        load_progress(
            output_file,
            existing_rows,
            all_chunks=args.all_chunks,
            limit_chunks=args.limit_chunks,
            target_questions=args.target_questions,
            preview_run_id=args.preview_run_id,
        )
        if args.resume
        else set()
    )
    # Older checkpoints recorded only the anchor. Add every evidence chunk
    # already represented by resumed rows so those siblings cannot become new
    # anchors and produce paraphrased duplicates.
    consumed_chunk_ids = {
        chunk_id
        for existing_row in existing_rows
        for chunk_id in row_source_chunk_ids(existing_row)
    }
    processed_chunk_ids.update(consumed_chunk_ids)
    remaining_limit = (
        max(args.limit_chunks - len(processed_chunk_ids), 0)
        if args.limit_chunks is not None
        else None
    )
    if args.preview_run_id:
        context_chunks = load_preview_chunks(
            args.preview_run_id,
            max_chunks_per_document=(
                None if args.all_chunks else MAX_CHUNKS_PER_DOCUMENT
            ),
        )
        source_description = (
            "ingestion_preview_chunks "
            f"for run {args.preview_run_id!r}"
        )
    else:
        context_chunks = load_vector_chunks(
            max_chunks_per_document=(
                None if args.all_chunks else MAX_CHUNKS_PER_DOCUMENT
            ),
        )
        source_description = "data_hkpl_knowledge"
    chunks_by_document: dict[str, list[dict]] = {}
    for candidate in context_chunks:
        chunks_by_document.setdefault(candidate["document_id"], []).append(candidate)
    chunks = [
        candidate for candidate in context_chunks
        if candidate["chunk_id"] not in processed_chunk_ids
    ]
    if remaining_limit is not None:
        chunks = chunks[:remaining_limit]
    print(
        f"Loaded {len(chunks)} new HKPL chunks from {source_description}."
    )
    print(
        "Selection: primary HKPL chunks with at least "
        f"{MIN_CHUNK_CHARS} characters; "
        + (
            "all eligible chunks per document."
            if args.all_chunks
            else f"at most {MAX_CHUNKS_PER_DOCUMENT} chunks per document."
        )
    )
    if args.resume:
        print(f"Resuming with {len(existing_rows)} checkpointed evaluation rows.")
    else:
        save_rows(output_file, [])
        save_progress(
            output_file,
            set(),
            all_chunks=args.all_chunks,
            limit_chunks=args.limit_chunks,
            target_questions=args.target_questions,
            preview_run_id=args.preview_run_id,
        )
        print(f"Initialized candidate checkpoint: {output_file}")

    generated_rows = []
    target_rows: list[dict] | None = None
    try:
        for index, chunk in enumerate(chunks, start=1):
            if chunk["chunk_id"] in processed_chunk_ids:
                continue
            print(
                f"[{index}/{len(chunks)}] "
                f"{chunk['source_title']} | {chunk['section_heading']} | "
                f"{chunk['chunk_id']}"
            )

            siblings = chunks_by_document[chunk["document_id"]]
            related_chunks = [
                chunk,
                *(
                    sibling
                    for sibling in siblings
                    if sibling["chunk_id"] != chunk["chunk_id"]
                    and sibling["chunk_id"] not in consumed_chunk_ids
                ),
            ]
            rows = await generate_questions_for_chunk(chunk, related_chunks)
            generated_rows.extend(rows)
            processed_chunk_ids.add(chunk["chunk_id"])
            consumed_evidence_ids = {
                chunk_id
                for row in rows
                for chunk_id in row_source_chunk_ids(row)
            }
            consumed_chunk_ids.update(consumed_evidence_ids)
            processed_chunk_ids.update(consumed_evidence_ids)
            save_rows(output_file, [*existing_rows, *generated_rows])
            save_progress(
                output_file,
                processed_chunk_ids,
                all_chunks=args.all_chunks,
                limit_chunks=args.limit_chunks,
                target_questions=args.target_questions,
                preview_run_id=args.preview_run_id,
            )
            print(
                f"  Generated {len(rows)} question(s); checkpoint rows="
                f"{len(existing_rows) + len(generated_rows)}, processed chunks="
                f"{len(processed_chunk_ids)}."
            )
            if consumed_evidence_ids.difference({chunk["chunk_id"]}):
                print(
                    "  Consumed sibling evidence chunks; they will not be "
                    "used as later anchors: "
                    + ", ".join(sorted(
                        consumed_evidence_ids.difference({chunk["chunk_id"]})
                    ))
                )
            if args.target_questions is not None:
                deduplicated = remove_ambiguous_duplicates(
                    [*existing_rows, *generated_rows]
                )
                if len(deduplicated) >= args.target_questions:
                    target_rows = deduplicated[:args.target_questions]
                    save_rows(output_file, target_rows)
                    print(
                        f"Reached target of {args.target_questions} "
                        "deduplicated questions."
                    )
                    break
    finally:
        save_rows(
            output_file,
            target_rows or [*existing_rows, *generated_rows],
        )
        save_progress(
            output_file,
            processed_chunk_ids,
            all_chunks=args.all_chunks,
            limit_chunks=args.limit_chunks,
            target_questions=args.target_questions,
            preview_run_id=args.preview_run_id,
        )

    all_rows = target_rows or [*existing_rows, *generated_rows]
    before = len(all_rows)
    all_rows = remove_ambiguous_duplicates(all_rows)
    after = len(all_rows)

    print()
    print(f"Rows before deduplication: {before}")
    print(f"Rows after deduplication : {after}")
    print(f"Dropped rows             : {before - after}")

    save_rows(output_file, all_rows)

    print()
    print(f"Saved {len(all_rows)} evaluation rows to:")
    print(output_file)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Generation interrupted. Completed chunks were checkpointed.")
        raise SystemExit(130) from None
