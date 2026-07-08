#!/usr/bin/env python3

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
from src.nodes import http_llm


OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation_dataset.csv"

MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 1800
QUESTIONS_PER_CHUNK = 1
MAX_CHUNKS_PER_DOCUMENT = 8


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


def load_chunks() -> list[dict]:
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
                )
                SELECT *
                FROM ranked_chunks
                WHERE rn <= :max_chunks_per_document
                ORDER BY document_id, chunk_id
            """),
            {
                "min_chars": MIN_CHUNK_CHARS,
                "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT,
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

    return chunks


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
- The expected_context_snippet must be an exact or near-exact phrase from the chunk.
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
    "source_type": "generated_from_kb",
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

    output = []

    for item in items:
        query = str(item.get("query", "")).strip()
        answer = str(item.get("expected_answer_text", "")).strip()
        snippet = str(item.get("expected_context_snippet", "")).strip()

        if not query or not answer or not snippet:
            continue

        output.append(
            {
                "domain": item.get("domain") or domain,
                "query": query,
                "expected_answer_text": answer,
                "expected_context_snippet": snippet,
                "source_title": item.get("source_title") or chunk["source_title"],
                "source_url": item.get("source_url") or chunk["source_url"],
                "source_type": "generated_from_kb",
                "source_document_id": chunk["document_id"],
                "source_chunk_id": chunk["chunk_id"],
            }
        )

    return output


def remove_ambiguous_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    cleaned: list[dict] = []

    for row in rows:
        query_key = normalize_text(row["query"])
        answer_key = normalize_text(row["expected_answer_text"])

        if query_key not in seen:
            seen[query_key] = row
            cleaned.append(row)
            continue

        previous = seen[query_key]
        previous_answer_key = normalize_text(previous["expected_answer_text"])

        if previous_answer_key == answer_key:
            continue

        print()
        print("Dropped ambiguous duplicate question:")
        print("Question:", row["query"])
        print("Answer A:", previous["expected_answer_text"])
        print("Answer B:", row["expected_answer_text"])

    return cleaned


async def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from data_hkpl_knowledge.")

    all_rows = []

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"[{index}/{len(chunks)}] "
            f"{chunk['source_title']} | {chunk['section_heading']} | {chunk['chunk_id']}"
        )

        rows = await generate_questions_for_chunk(chunk)
        all_rows.extend(rows)

        print(f"  Generated {len(rows)} question(s).")

    before = len(all_rows)
    all_rows = remove_ambiguous_duplicates(all_rows)
    after = len(all_rows)

    print()
    print(f"Rows before deduplication: {before}")
    print(f"Rows after deduplication : {after}")
    print(f"Dropped rows             : {before - after}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "domain",
        "query",
        "expected_answer_text",
        "expected_context_snippet",
        "source_title",
        "source_url",
        "source_type",
        "source_document_id",
        "source_chunk_id",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"Saved {len(all_rows)} evaluation rows to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())