#!/usr/bin/env python3

import asyncio
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.observability import setup_phoenix_tracing
from src.retrieval import retrieve_nodes

setup_phoenix_tracing()

tracer = trace.get_tracer("hkpl-retrieval-evaluation")

DATASET = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
OUTPUT = PROJECT_ROOT / "data" / "retrieval_results.csv"


def reciprocal_rank(expected: str, retrieved: list[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc == expected:
            return 1.0 / rank
    return 0.0


async def evaluate() -> None:
    df = pd.read_csv(DATASET)

    results = []
    hit1 = 0
    hit3 = 0
    hit5 = 0
    rr_total = 0.0
    total = len(df)

    print("=" * 80)
    print("Evaluating retrieval...")
    print("=" * 80)

    for index, row in df.iterrows():
        query = str(row["query"])
        expected_document = str(row.get("source_document_id", ""))
        expected_chunk = str(row.get("source_chunk_id", ""))

        print(f"[{index + 1}/{total}] {query}")

        with tracer.start_as_current_span("evaluate_retrieval_row") as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                "RETRIEVER",
            )
            span.set_attribute(SpanAttributes.INPUT_VALUE, query)
            span.set_attribute("eval.question", query)
            span.set_attribute("eval.expected_document", expected_document)
            span.set_attribute("eval.expected_chunk", expected_chunk)

            try:
                nodes = await retrieve_nodes(query)
            except Exception as error:
                span.record_exception(error)
                span.set_attribute("eval.failed", True)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(error))

                print("=" * 80)
                print("FAILED QUERY")
                print(query)
                print(error)
                print("=" * 80)

                results.append(
                    {
                        "query": query,
                        "expected_document": expected_document,
                        "expected_chunk": expected_chunk,
                        "retrieved_document_1": "",
                        "retrieved_document_2": "",
                        "retrieved_document_3": "",
                        "retrieved_chunk_1": "",
                        "retrieved_chunk_2": "",
                        "retrieved_chunk_3": "",
                        "title_1": "",
                        "title_2": "",
                        "title_3": "",
                        "score_1": "",
                        "score_2": "",
                        "score_3": "",
                        "hit@1": False,
                        "hit@3": False,
                        "hit@5": False,
                        "reciprocal_rank": 0.0,
                        "error": str(error),
                    }
                )
                continue

            retrieved_documents = []
            retrieved_chunks = []
            retrieved_titles = []
            retrieved_scores = []

            for item in nodes:
                metadata = item.node.metadata or {}
                chunk_id = metadata.get("chunk_id", "")

                document_id = (
                    metadata.get("kb_document_id")
                    or metadata.get("document_id")
                    or chunk_id.split(":")[0]
                    or ""
                )

                retrieved_documents.append(document_id)
                retrieved_chunks.append(chunk_id)
                retrieved_titles.append(metadata.get("source_title", ""))
                retrieved_scores.append(float(item.score or 0.0))

            h1 = (
                len(retrieved_documents) >= 1
                and retrieved_documents[0] == expected_document
            )
            h3 = expected_document in retrieved_documents[:3]
            h5 = expected_document in retrieved_documents[:5]
            rr = reciprocal_rank(expected_document, retrieved_documents)

            hit1 += int(h1)
            hit3 += int(h3)
            hit5 += int(h5)
            rr_total += rr

            output_payload = {
                "retrieved_documents": retrieved_documents,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_titles": retrieved_titles,
                "scores": retrieved_scores,
                "hit_at_1": h1,
                "hit_at_3": h3,
                "hit_at_5": h5,
                "reciprocal_rank": rr,
            }

            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE,
                json.dumps(output_payload, ensure_ascii=False),
            )
            span.set_attribute("retrieval.documents", json.dumps(retrieved_documents))
            span.set_attribute("retrieval.chunks", json.dumps(retrieved_chunks))
            span.set_attribute("retrieval.titles", json.dumps(retrieved_titles))
            span.set_attribute("retrieval.scores", json.dumps(retrieved_scores))
            span.set_attribute("eval.hit_at_1", bool(h1))
            span.set_attribute("eval.hit_at_3", bool(h3))
            span.set_attribute("eval.hit_at_5", bool(h5))
            span.set_attribute("eval.reciprocal_rank", float(rr))

            results.append(
                {
                    "query": query,
                    "expected_document": expected_document,
                    "expected_chunk": expected_chunk,
                    "retrieved_document_1": retrieved_documents[0] if len(retrieved_documents) > 0 else "",
                    "retrieved_document_2": retrieved_documents[1] if len(retrieved_documents) > 1 else "",
                    "retrieved_document_3": retrieved_documents[2] if len(retrieved_documents) > 2 else "",
                    "retrieved_chunk_1": retrieved_chunks[0] if len(retrieved_chunks) > 0 else "",
                    "retrieved_chunk_2": retrieved_chunks[1] if len(retrieved_chunks) > 1 else "",
                    "retrieved_chunk_3": retrieved_chunks[2] if len(retrieved_chunks) > 2 else "",
                    "title_1": retrieved_titles[0] if len(retrieved_titles) > 0 else "",
                    "title_2": retrieved_titles[1] if len(retrieved_titles) > 1 else "",
                    "title_3": retrieved_titles[2] if len(retrieved_titles) > 2 else "",
                    "score_1": retrieved_scores[0] if len(retrieved_scores) > 0 else "",
                    "score_2": retrieved_scores[1] if len(retrieved_scores) > 1 else "",
                    "score_3": retrieved_scores[2] if len(retrieved_scores) > 2 else "",
                    "hit@1": h1,
                    "hit@3": h3,
                    "hit@5": h5,
                    "reciprocal_rank": rr,
                    "error": "",
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 80)
    print("Retrieval Evaluation Summary")
    print("=" * 80)
    print(f"Questions          : {total}")
    print(f"Hit@1              : {hit1 / total:.2%}")
    print(f"Recall@3 (Hit@3)   : {hit3 / total:.2%}")
    print(f"Recall@5 (Hit@5)   : {hit5 / total:.2%}")
    print(f"MRR                : {rr_total / total:.4f}")
    print()
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(evaluate())