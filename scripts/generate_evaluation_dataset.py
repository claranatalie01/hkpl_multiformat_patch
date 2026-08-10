#!/usr/bin/env python3
"""Generate reviewable evaluation questions from existing HKPL vector chunks.

Eligible primary-corpus chunks are read from ``data_hkpl_knowledge`` and sent
to the answer model to propose questions, accepted answers, exact evidence, and
source IDs. Progress is checkpointed; output remains a candidate until reviewed
and promoted. This script does not create document embeddings.
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
    "accepted_answers_json",
    "source_title",
    "source_url",
    "source_document_id",
    "source_chunk_id",
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


def load_chunks(
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


def load_existing_rows(output_file: Path) -> list[dict]:
    if not output_file.is_file():
        raise FileNotFoundError(
            f"Cannot resume because the candidate does not exist: {output_file}"
        )
    with output_file.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        actual_columns = list(reader.fieldnames or [])
        if actual_columns != FIELDNAMES:
            raise ValueError(
                f"Cannot resume candidate with columns {actual_columns}; "
                f"expected {FIELDNAMES}."
            )
        rows = [dict(row) for row in reader if row.get("query")]
    for row in rows:
        row.setdefault("accepted_answers_json", "[]")
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
) -> None:
    path = progress_file(output_file)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "all_chunks": all_chunks,
                "limit_chunks": limit_chunks,
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
        "--all-chunks",
        action="store_true",
        help=(
            "Generate candidates from every eligible primary HKPL chunk. "
            "This is slower and still requires human label review."
        ),
    )
    args = parser.parse_args()
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be positive")
    return args


async def generate_questions_for_chunk(chunk: dict) -> list[dict]:
    domain = infer_domain(chunk["source_title"], chunk["text"])

    prompt = f"""
You are creating an evaluation dataset for a Retrieval-Augmented Generation system for Hong Kong Public Libraries.

Generate exactly {QUESTIONS_PER_CHUNK} factual evaluation question from the official HKPL chunk below.

VERY IMPORTANT RULES:
- The question must have ONE and ONLY ONE correct answer.
- If the chunk is about repeated events, roving exhibitions, workshops, branch sessions, or multiple venues, the question MUST include the exact venue/branch and date/month.
- Do NOT generate generic repeated questions such as:
  "When and where is the roving exhibition held?"
  "When and where is this event held?"
  "Where is the activity held?"
- Instead, generate specific questions such as:
  "When is the roving exhibition titled 'Blissful Moments Between Pages' held at Sham Shui Po Public Library?"
  "What are the dates for 'Blissful Moments Between Pages' at Ma On Shan Public Library?"
- Each question must be answerable ONLY from this chunk.
- Do not invent facts.
- Avoid vague questions.
- Prefer useful public-service questions.
- The expected_answer_text must be concise but complete.
- The expected_context_snippet must be an exact contiguous phrase from the chunk.
- Return ONLY valid JSON array.
- Do not include markdown.

JSON format:
[
  {{
    "domain": "{domain}",
    "query": "...",
    "expected_answer_text": "...",
    "expected_context_snippet": "...",
    "source_title": "{chunk['source_title']}",
    "source_url": "{chunk['source_url']}",
    "source_document_id": "{chunk['document_id']}",
    "source_chunk_id": "{chunk['chunk_id']}"
  }}
]

Official HKPL chunk:
Source title: {chunk["source_title"]}
Section: {chunk["section_heading"]}

{chunk["text"]}
"""

    raw = await http_llm(prompt, temperature=0.0, max_tokens=650)

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

    output = []

    for item in items:
        query = str(item.get("query", "")).strip()
        answer = str(item.get("expected_answer_text", "")).strip()
        snippet = str(item.get("expected_context_snippet", "")).strip()

        if not query or not answer or not snippet:
            print(
                "  Skipped candidate: question, answer, or evidence snippet "
                "was empty."
            )
            continue
        if not query.endswith("?") or len(query) < 15:
            print(f"  Rejected malformed question: {query!r}")
            continue
        if normalize_text(snippet) not in normalize_text(chunk["text"]):
            print(
                "  Rejected candidate because expected_context_snippet is "
                "not present in the source chunk."
            )
            continue

        output.append(
            {
                "domain": item.get("domain") or domain,
                "query": query,
                "expected_answer_text": answer,
                "expected_context_snippet": snippet,
                "accepted_answers_json": "[]",
                "source_title": item.get("source_title") or chunk["source_title"],
                "source_url": item.get("source_url") or chunk["source_url"],
                "source_document_id": chunk["document_id"],
                "source_chunk_id": chunk["chunk_id"],
            }
        )

    return output


def remove_ambiguous_duplicates(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(normalize_text(row["query"]), []).append(row)

    cleaned: list[dict] = []
    for group in grouped.values():
        if len(group) == 1:
            cleaned.append(group[0])
            continue
        answer_keys = {
            normalize_text(row["expected_answer_text"])
            for row in group
        }
        if len(answer_keys) == 1:
            cleaned.append(group[0])
            continue

        print()
        print("Removed all conflicting versions of ambiguous question:")
        print("Question:", group[0]["query"])
        for index, row in enumerate(group, start=1):
            print(f"Answer {index}:", row["expected_answer_text"])

    return cleaned


async def main() -> None:
    args = parse_args()
    output_file = args.output.resolve()
    existing_rows = load_existing_rows(output_file) if args.resume else []
    processed_chunk_ids = (
        load_progress(
            output_file,
            existing_rows,
            all_chunks=args.all_chunks,
            limit_chunks=args.limit_chunks,
        )
        if args.resume
        else set()
    )
    remaining_limit = (
        max(args.limit_chunks - len(processed_chunk_ids), 0)
        if args.limit_chunks is not None
        else None
    )
    chunks = load_chunks(
        excluded_chunk_ids=processed_chunk_ids,
        limit=remaining_limit,
        max_chunks_per_document=(None if args.all_chunks else MAX_CHUNKS_PER_DOCUMENT),
    )
    print(
        f"Loaded {len(chunks)} new HKPL chunks from data_hkpl_knowledge; "
        "distractor corpora were excluded."
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
        )
        print(f"Initialized candidate checkpoint: {output_file}")

    generated_rows = []
    try:
        for index, chunk in enumerate(chunks, start=1):
            print(
                f"[{index}/{len(chunks)}] "
                f"{chunk['source_title']} | {chunk['section_heading']} | "
                f"{chunk['chunk_id']}"
            )

            rows = await generate_questions_for_chunk(chunk)
            generated_rows.extend(rows)
            processed_chunk_ids.add(chunk["chunk_id"])
            save_rows(output_file, [*existing_rows, *generated_rows])
            save_progress(
                output_file,
                processed_chunk_ids,
                all_chunks=args.all_chunks,
                limit_chunks=args.limit_chunks,
            )
            print(
                f"  Generated {len(rows)} question(s); checkpoint rows="
                f"{len(existing_rows) + len(generated_rows)}, processed chunks="
                f"{len(processed_chunk_ids)}."
            )
    finally:
        save_rows(output_file, [*existing_rows, *generated_rows])
        save_progress(
            output_file,
            processed_chunk_ids,
            all_chunks=args.all_chunks,
            limit_chunks=args.limit_chunks,
        )

    all_rows = [*existing_rows, *generated_rows]
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
