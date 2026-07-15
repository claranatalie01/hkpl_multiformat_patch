#!/usr/bin/env python3

import asyncio
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

from llama_index.core.evaluation import HitRate, MRR
from opentelemetry import trace
from opentelemetry.trace import format_span_id
from openinference.semconv.trace import SpanAttributes
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.observability import setup_phoenix_tracing
from src.infrastructure.db import engine
from src.phoenix_annotations import (
    log_document_relevance_annotations,
    log_span_annotations,
)
from src.rag_diagnosis import diagnose_rag
from src.retrieval import retrieve_nodes
from src.infrastructure.vector_store import VECTOR_TABLE
from src.tracing_helpers import set_json_attribute, set_span_io

setup_phoenix_tracing()

tracer = trace.get_tracer("hkpl-retrieval-evaluation")

SOURCE_CSV = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
OUTPUT = PROJECT_ROOT / "data" / "retrieval_results.csv"
SUMMARY = PROJECT_ROOT / "data" / "retrieval_summary.json"
EVALUATION_DATASET_TABLE = os.getenv("EVALUATION_DATASET_TABLE", "evaluation_dataset")


def llamaindex_hit_rate(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    if not expected_ids or not retrieved_ids:
        return 0.0
    return float(
        HitRate().compute(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
        ).score
    )


def llamaindex_mrr(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    if not expected_ids or not retrieved_ids:
        return 0.0
    return float(
        MRR().compute(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
        ).score
    )


async def evaluate() -> None:
    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                SELECT
                    domain,
                    query,
                    expected_answer_text,
                    expected_context_snippet,
                    source_title,
                    source_url,
                    source_type,
                    source_document_id,
                    source_chunk_id
                FROM {EVALUATION_DATASET_TABLE}
                WHERE EXISTS (
                    SELECT 1
                    FROM data_{VECTOR_TABLE} k
                    WHERE k.metadata_->>'chunk_id' = {EVALUATION_DATASET_TABLE}.source_chunk_id
                )
                ORDER BY id
            """)
        ).fetchall()

    rows = [dict(row._mapping) for row in rows]
    if not rows:
        raise RuntimeError(
            f"No valid rows found in Postgres table {EVALUATION_DATASET_TABLE}. "
            f"Run scripts/ingest_pgvector_llamaindex.py and scripts/validate_evaluation_dataset.py first."
        )

    results = []
    hit1 = 0
    hit3 = 0
    hit5 = 0
    rr_total = 0.0
    total = len(rows)

    print("=" * 80)
    print("Evaluating retrieval...")
    print("=" * 80)
    print(f"Evaluation questions loaded from table: {EVALUATION_DATASET_TABLE}")
    print(f"Retriever searches vector table: data_{VECTOR_TABLE}")
    print("=" * 80)

    for index, row in enumerate(rows):
        query = str(row["query"])
        expected_document = str(row.get("source_document_id", ""))
        expected_chunk = str(row.get("source_chunk_id", ""))

        print(f"[{index + 1}/{total}] {query}")

        with tracer.start_as_current_span("HKPL Retrieval Evaluation Query") as span:
            root_span_id = format_span_id(span.get_span_context().span_id)

            set_span_io(
                span,
                "CHAIN",
                input_value=query,
            )
            span.set_attribute("eval.root_span_id", root_span_id)
            span.set_attribute("eval.dataset", "hkpl")
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
                        "expected_document_in_vector": False,
                        "expected_chunk_in_vector": False,
                        "expected_chunk_after_rerank": False,
                        "diagnosis": "retrieval_evaluation_failed",
                        "diagnosis_recommendation": str(error),
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

            expected_document_ids = [expected_document] if expected_document else []
            h1 = llamaindex_hit_rate(expected_document_ids, retrieved_documents[:1])
            h3 = llamaindex_hit_rate(expected_document_ids, retrieved_documents[:3])
            h5 = llamaindex_hit_rate(expected_document_ids, retrieved_documents[:5])
            rr = llamaindex_mrr(expected_document_ids, retrieved_documents)

            retrieval_trace = getattr(retrieve_nodes, "last_trace", {})
            retriever_span_id = retrieval_trace.get("retriever_span_id", "")
            vector_candidates = retrieval_trace.get("vector_candidates_before_rerank", [])
            after_rerank = retrieval_trace.get("final_chunks_after_rerank", [])
            vector_documents = [item.get("document_id", "") for item in vector_candidates]
            vector_chunks = [item.get("chunk_id", "") for item in vector_candidates]
            reranked_chunks = [item.get("chunk_id", "") for item in after_rerank]

            expected_document_in_vector = expected_document in vector_documents
            expected_chunk_in_vector = expected_chunk in vector_chunks
            expected_chunk_after_rerank = expected_chunk in reranked_chunks

            diagnostic = diagnose_rag(
                expected_document_id=expected_document,
                expected_chunk_id=expected_chunk,
                vector_candidates=vector_candidates,
                after_rerank=after_rerank,
                chunks_sent_to_llm=reranked_chunks,
                correctness_score=5.0,
                faithfulness_score=1.0,
                relevancy_score=1.0,
            )

            hit1 += h1
            hit3 += h3
            hit5 += h5
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
                "expected_document_in_vector": expected_document_in_vector,
                "expected_chunk_in_vector": expected_chunk_in_vector,
                "expected_chunk_after_rerank": expected_chunk_after_rerank,
                "diagnosis": diagnostic["diagnosis"],
                "recommendation": diagnostic["recommendation"],
            }

            set_span_io(span, "CHAIN", output_value=output_payload)
            span.set_attribute("retrieval.documents", json.dumps(retrieved_documents))
            span.set_attribute("retrieval.chunks", json.dumps(retrieved_chunks))
            span.set_attribute("retrieval.titles", json.dumps(retrieved_titles))
            span.set_attribute("retrieval.scores", json.dumps(retrieved_scores))
            set_json_attribute(span, "retrieval.vector_candidate_documents", vector_documents)
            set_json_attribute(span, "retrieval.vector_candidate_chunks", vector_chunks)
            set_json_attribute(span, "retrieval.after_rerank_chunks", reranked_chunks)
            span.set_attribute("eval.hit_at_1", float(h1))
            span.set_attribute("eval.hit_at_3", float(h3))
            span.set_attribute("eval.hit_at_5", float(h5))
            span.set_attribute("eval.reciprocal_rank", float(rr))
            span.set_attribute("eval.metric_source", "llama_index")
            span.set_attribute(
                "eval.expected_document_in_vector",
                bool(expected_document_in_vector),
            )
            span.set_attribute("eval.expected_chunk_in_vector", bool(expected_chunk_in_vector))
            span.set_attribute(
                "eval.expected_chunk_after_rerank",
                bool(expected_chunk_after_rerank),
            )
            span.set_attribute("rag.diagnosis", diagnostic["diagnosis"])
            span.set_attribute("rag.recommendation", diagnostic["recommendation"])

            log_span_annotations(
                root_span_id,
                [
                    {
                        "name": "Hit@1",
                        "annotator_kind": "CODE",
                        "label": "hit" if h1 else "miss",
                        "score": h1,
                        "explanation": f"LlamaIndex HitRate over top 1. Expected document: {expected_document}",
                        "identifier": "hkpl-hit-at-1",
                    },
                    {
                        "name": "Hit@3",
                        "annotator_kind": "CODE",
                        "label": "hit" if h3 else "miss",
                        "score": h3,
                        "explanation": f"LlamaIndex HitRate over top 3. Expected document: {expected_document}",
                        "identifier": "hkpl-hit-at-3",
                    },
                    {
                        "name": "Hit@5",
                        "annotator_kind": "CODE",
                        "label": "hit" if h5 else "miss",
                        "score": h5,
                        "explanation": f"LlamaIndex HitRate over top 5. Expected document: {expected_document}",
                        "identifier": "hkpl-hit-at-5",
                    },
                ],
            )
            log_document_relevance_annotations(
                retriever_span_id=retriever_span_id,
                retrieved_documents=vector_candidates,
                expected_document_id=expected_document,
                expected_chunk_id=expected_chunk,
            )

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
                    "hit@1": bool(h1),
                    "hit@3": bool(h3),
                    "hit@5": bool(h5),
                    "reciprocal_rank": rr,
                    "expected_document_in_vector": expected_document_in_vector,
                    "expected_chunk_in_vector": expected_chunk_in_vector,
                    "expected_chunk_after_rerank": expected_chunk_after_rerank,
                    "diagnosis": diagnostic["diagnosis"],
                    "diagnosis_recommendation": diagnostic["recommendation"],
                    "error": "",
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "total_questions": total,
        "hit_at_1": hit1 / total if total else 0.0,
        "recall_at_3": hit3 / total if total else 0.0,
        "recall_at_5": hit5 / total if total else 0.0,
        "mrr": rr_total / total if total else 0.0,
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in results)),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with tracer.start_as_current_span("HKPL Retrieval Evaluation Summary") as span:
        set_span_io(
            span,
            "EVALUATOR",
            input_value={
                "dataset_table": EVALUATION_DATASET_TABLE,
                "source_csv": str(SOURCE_CSV),
                "result_file": str(OUTPUT),
                "questions": total,
            },
            output_value=summary,
        )
        span.set_attribute("eval.total_questions", int(total))
        span.set_attribute("eval.dataset", "hkpl")
        span.set_attribute("eval.hit_at_1", float(summary["hit_at_1"]))
        span.set_attribute("eval.recall_at_3", float(summary["recall_at_3"]))
        span.set_attribute("eval.recall_at_5", float(summary["recall_at_5"]))
        span.set_attribute("eval.mrr", float(summary["mrr"]))
        set_json_attribute(span, "eval.diagnosis_counts", summary["diagnosis_counts"])

    print()
    print("=" * 80)
    print("Retrieval Evaluation Summary")
    print("=" * 80)
    print(f"Questions          : {total}")
    print(f"Hit@1              : {summary['hit_at_1']:.2%}")
    print(f"Recall@3 (Hit@3)   : {summary['recall_at_3']:.2%}")
    print(f"Recall@5 (Hit@5)   : {summary['recall_at_5']:.2%}")
    print(f"MRR                : {summary['mrr']:.4f}")
    print("Diagnosis counts   :")
    for diagnosis, count in summary["diagnosis_counts"].items():
        print(f"  {diagnosis}: {count}")
    print()
    print(f"Saved to {OUTPUT}")
    print(f"Saved summary to {SUMMARY}")


if __name__ == "__main__":
    asyncio.run(evaluate())
