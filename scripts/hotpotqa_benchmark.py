#!/usr/bin/env python3

import argparse
import hashlib
import sys
from itertools import islice
from pathlib import Path
from urllib.parse import quote

from datasets import load_dataset
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.infrastructure.embedding import embed_model
from src.infrastructure.vector_store import VECTOR_TABLE, vector_store


DATASET_NAME = "hotpotqa"
DATASET_REPOSITORY = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
TABLE_NAME = f"data_{VECTOR_TABLE}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest HotpotQA paragraphs as distractor noise in the shared "
            "HKPL PGVector table."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="Download a deterministic subset and replace HotpotQA vectors.",
    )
    prepare.add_argument("--limit", type=int, default=1000)
    prepare.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def load_examples(offset: int, limit: int) -> list[dict]:
    if offset < 0 or limit < 1:
        raise ValueError("Offset must be non-negative and limit must be positive.")

    dataset = load_dataset(
        DATASET_REPOSITORY,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )
    examples = [
        dict(example)
        for example in islice(dataset, offset, offset + limit)
    ]

    if len(examples) != limit:
        raise ValueError(
            f"Requested {limit} examples from offset {offset}, but the "
            f"dataset returned {len(examples)}."
        )
    return examples


def paragraph_id(title: str, paragraph: str) -> str:
    digest = hashlib.sha256(
        f"{title.strip()}\n{paragraph.strip()}".encode("utf-8")
    ).hexdigest()[:24]
    return f"hotpotqa:{digest}"


def build_distractor_nodes(examples: list[dict]) -> list[TextNode]:
    paragraphs: dict[str, dict] = {}
    for example in examples:
        example_id = str(example.get("_id") or example.get("id") or "")
        context = example.get("context") or {}
        if isinstance(context, dict):
            context_rows = zip(
                context.get("title") or [],
                context.get("sentences") or [],
            )
        else:
            context_rows = context
        for title, sentences in context_rows:
            paragraph = "".join(str(sentence) for sentence in sentences).strip()
            if not paragraph:
                continue
            chunk_id = paragraph_id(str(title), paragraph)
            record = paragraphs.setdefault(
                chunk_id,
                {
                    "title": str(title),
                    "text": paragraph,
                    "source_example_ids": set(),
                },
            )
            if example_id:
                record["source_example_ids"].add(example_id)

    nodes = []
    for chunk_id, record in paragraphs.items():
        title = record["title"]
        metadata = {
            "dataset": DATASET_NAME,
            "corpus": DATASET_NAME,
            "kb_document_id": chunk_id,
            "document_id": chunk_id,
            "chunk_id": chunk_id,
            "source_title": title,
            "source_url": (
                "https://en.wikipedia.org/wiki/"
                + quote(title.replace(" ", "_"))
            ),
            "source_type": "distractor_benchmark",
            "document_type": "prose",
            "chunk_strategy": "atomic",
            "document_version": 1,
            "hotpotqa_source_example_ids": sorted(
                record["source_example_ids"]
            ),
        }
        nodes.append(
            TextNode(
                id_=chunk_id,
                text=f"Title: {title}\n\n{record['text']}",
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
            )
        )
    return nodes


def replace_hotpotqa_vectors(nodes: list[TextNode]) -> int:
    with engine.begin() as connection:
        deleted = connection.execute(
            text(f"""
                DELETE FROM {TABLE_NAME}
                WHERE metadata_->>'dataset' = :dataset
            """),
            {"dataset": DATASET_NAME},
        )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return int(deleted.rowcount or 0)


def prepare(args: argparse.Namespace) -> None:
    examples = load_examples(args.offset, args.limit)
    nodes = build_distractor_nodes(examples)
    deleted = replace_hotpotqa_vectors(nodes)
    print(f"Removed previous HotpotQA distractor vectors: {deleted}")
    print(f"HotpotQA examples sampled: {len(examples)}")
    print(f"Unique HotpotQA distractor paragraphs: {len(nodes)}")
    print(f"Shared vector table: {TABLE_NAME}")
    print("No HotpotQA questions or expected answers were stored for evaluation.")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)


if __name__ == "__main__":
    main()
