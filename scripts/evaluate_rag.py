#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from statistics import mean

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import format_span_id
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.infrastructure.vector_store import VECTOR_TABLE
from src.llm_client import http_llm
from src.observability import setup_phoenix_tracing
from src.phoenix_annotations import (
    log_document_relevance_annotations,
    log_rag_answer_annotations,
    log_span_annotations,
)
from src.retrieval import retrieve_nodes
from src.token_counting import LLM_TOKENIZER_NAME, LLM_TOKENIZER_URL, count_tokens
from src.tracing_helpers import set_json_attribute, set_llm_attributes, set_span_io

HKPL_EVALUATION_TABLE = os.getenv("EVALUATION_DATASET_TABLE", "evaluation_dataset")
RESULTS_PATH = Path(
    os.getenv(
        "RAG_EVALUATION_RESULTS_PATH",
        "/app/data/rag_evaluation/results.csv",
    )
)
SUMMARY_PATH = Path(
    os.getenv(
        "RAG_EVALUATION_SUMMARY_PATH",
        "/app/data/rag_evaluation/summary.json",
    )
)
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "32768"))
EVALUATION_MAX_TOKENS = int(os.getenv("EVALUATION_MAX_TOKENS", "1024"))
CUTOFFS = (1, 3, 5)

setup_phoenix_tracing()
tracer = trace.get_tracer("hkpl-rag-noise-evaluation")


class QwenEvaluationLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=LLM_CONTEXT_WINDOW,
            num_output=EVALUATION_MAX_TOKENS,
            model_name="qwen3.5-9b-http",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("Use asynchronous evaluation.")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = await http_llm(
            prompt,
            temperature=0.0,
            max_tokens=EVALUATION_MAX_TOKENS,
            enable_thinking=False,
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming is not used by evaluation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HKPL questions while searching the combined HKPL and "
            "HotpotQA distractor vector corpus."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic limit on HKPL evaluation questions.",
    )
    question_filter = parser.add_mutually_exclusive_group()
    question_filter.add_argument(
        "--question-contains",
        default="",
        help="Evaluate only questions containing this case-insensitive text.",
    )
    question_filter.add_argument(
        "--question-exact",
        default="",
        help="Evaluate one question using a case-insensitive exact match.",
    )
    return parser.parse_args()


def report_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    is_full_evaluation = (
        args.limit is None
        and not args.question_contains
        and not args.question_exact
    )
    if is_full_evaluation:
        return RESULTS_PATH, SUMMARY_PATH

    tags = ["hkpl"]
    if args.limit is not None:
        tags.append(f"limit-{args.limit}")

    question_filter = args.question_exact or args.question_contains
    if question_filter:
        match_type = "exact" if args.question_exact else "contains"
        question_slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            question_filter.casefold(),
        ).strip("-")[:48]
        tags.append(f"{match_type}-{question_slug or 'question'}")

    tag = ".".join(tags)
    results_path = RESULTS_PATH.with_name(
        f"{RESULTS_PATH.stem}.{tag}{RESULTS_PATH.suffix}"
    )
    summary_path = SUMMARY_PATH.with_name(
        f"{SUMMARY_PATH.stem}.{tag}{SUMMARY_PATH.suffix}"
    )
    return results_path, summary_path


def safe_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL table name: {value!r}")
    return value


def load_hkpl_rows(limit: int | None) -> list[dict]:
    table = safe_table_name(HKPL_EVALUATION_TABLE)
    vector_table = safe_table_name(f"data_{VECTOR_TABLE}")
    limit_clause = "LIMIT :limit" if limit is not None else ""
    parameters = {"limit": limit} if limit is not None else {}
    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                SELECT
                    id,
                    domain,
                    query,
                    expected_answer_text,
                    source_document_id,
                    source_chunk_id
                FROM {table}
                WHERE EXISTS (
                    SELECT 1
                    FROM {vector_table} knowledge
                    WHERE knowledge.metadata_->>'chunk_id' = {table}.source_chunk_id
                )
                ORDER BY id
                {limit_clause}
            """),
            parameters,
        ).mappings().all()

    return [
        {
            "evaluation_id": f"hkpl:{row['id']}",
            "dataset": "hkpl",
            "domain": str(row.get("domain") or ""),
            "question": str(row["query"]),
            "expected_answer": str(row["expected_answer_text"]),
            "expected_document_ids": [str(row["source_document_id"])],
            "expected_chunk_ids": [str(row["source_chunk_id"])],
        }
        for row in rows
    ]


def load_corpus_counts() -> dict[str, int]:
    vector_table = safe_table_name(f"data_{VECTOR_TABLE}")
    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                SELECT
                    COALESCE(NULLIF(metadata_->>'dataset', ''), 'hkpl') AS dataset,
                    COUNT(*) AS vectors
                FROM {vector_table}
                GROUP BY dataset
                ORDER BY dataset
            """)
        ).mappings().all()
    return {str(row["dataset"]): int(row["vectors"]) for row in rows}


def ranking_metrics(expected_ids: list[str], retrieved_ids: list[str], k: int) -> dict:
    expected = set(expected_ids)
    selected = set(retrieved_ids[:k])
    matched = expected.intersection(selected)
    return {
        "hit": float(bool(matched)),
        "recall": len(matched) / len(expected) if expected else 0.0,
        "complete": float(bool(expected) and expected.issubset(selected)),
    }


def reciprocal_rank(expected_ids: list[str], retrieved_ids: list[str]) -> float:
    expected = set(expected_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def ranked_chunk_ids(documents: list[dict]) -> list[str]:
    return [str(document.get("chunk_id") or "") for document in documents]


def distractor_metrics(documents: list[dict], k: int) -> dict:
    selected = documents[:k]
    hotpotqa_count = sum(
        str(document.get("metadata", {}).get("dataset") or "").lower()
        == "hotpotqa"
        for document in selected
    )
    return {
        "count": hotpotqa_count,
        "rate": hotpotqa_count / len(selected) if selected else 0.0,
    }


def build_context(nodes) -> tuple[str, list[str], list[dict]]:
    with tracer.start_as_current_span("build_context") as span:
        parts = []
        contexts = []
        sources = []
        for rank, item in enumerate(nodes, start=1):
            node = item.node
            metadata = node.metadata or {}
            content = node.get_content()
            chunk_id = str(metadata.get("chunk_id") or "")
            title = str(metadata.get("source_title") or "")
            parts.append(f"[Source {rank}: {title}]\n{content}")
            contexts.append(content)
            sources.append(
                {
                    "rank": rank,
                    "document_id": str(
                        metadata.get("kb_document_id")
                        or metadata.get("document_id")
                        or chunk_id
                    ),
                    "chunk_id": chunk_id,
                    "title": title,
                    "score": float(item.score or 0.0),
                }
            )
        context = "\n\n".join(parts)
        set_span_io(
            span,
            "CHAIN",
            input_value={"num_nodes": len(nodes)},
            output_value={"sources": sources, "context_chars": len(context)},
        )
        set_json_attribute(span, "rag.context_sources", sources)
        return context, contexts, sources


async def generate_answer(question: str, context: str) -> tuple[str, dict]:
    prompt = f"""You are a retrieval-grounded question answering assistant.

Answer the question using only the retrieved context. Combine evidence from
multiple sources when required. Do not invent information. If the retrieved
context does not contain enough evidence, say: "I don't have that information
in my knowledge base."

Retrieved context:
{context}

Question:
{question}

Answer:
"""
    with tracer.start_as_current_span("LLM") as span:
        started = time.perf_counter()
        answer = await http_llm(
            prompt,
            temperature=0.0,
            max_tokens=EVALUATION_MAX_TOKENS,
            enable_thinking=False,
        )
        prompt_tokens, prompt_estimated, tokenizer = await count_tokens(
            prompt,
            LLM_TOKENIZER_URL,
            LLM_TOKENIZER_NAME,
        )
        completion_tokens, completion_estimated, _ = await count_tokens(
            answer,
            LLM_TOKENIZER_URL,
            LLM_TOKENIZER_NAME,
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "is_estimated": prompt_estimated or completion_estimated,
            "tokenizer": tokenizer,
        }
        set_llm_attributes(
            span=span,
            model_name="qwen3.5-9b-http",
            prompt=prompt,
            response=answer,
            temperature=0.0,
            max_tokens=EVALUATION_MAX_TOKENS,
            usage=usage,
        )
        span.set_attribute(
            "llm.latency_seconds",
            round(time.perf_counter() - started, 4),
        )
        return answer, usage


def evaluator_score(result) -> float:
    score = getattr(result, "score", None)
    if score is not None:
        return float(score)
    return 1.0 if getattr(result, "passing", False) else 0.0


def evaluator_reason(result) -> str:
    return str(
        getattr(result, "feedback", None)
        or getattr(result, "reason", None)
        or ""
    )


@asynccontextmanager
async def suppress_auto_instrumentation():
    try:
        from opentelemetry.context import attach, detach, set_value

        token = attach(set_value("suppress_instrumentation", True))
        try:
            yield
        finally:
            detach(token)
    except Exception:
        yield


async def run_evaluator(name: str, payload: dict, call) -> tuple[float, str, bool]:
    with tracer.start_as_current_span(name) as span:
        set_span_io(span, "EVALUATOR", input_value=payload)
        try:
            async with suppress_auto_instrumentation():
                result = await call()
        except Exception as error:
            span.record_exception(error)
            reason = f"Evaluator failed: {error}"
            set_span_io(
                span,
                "EVALUATOR",
                output_value={"score": None, "reason": reason},
            )
            return 0.0, reason, True
        score = evaluator_score(result)
        reason = evaluator_reason(result)
        set_span_io(span, "EVALUATOR", output_value={"score": score, "reason": reason})
        return score, reason, False


def diagnose(
    expected_ids: list[str],
    vector_ids: list[str],
    reranked_ids: list[str],
    context_ids: list[str],
    correctness: float,
    faithfulness: float,
    relevancy: float,
) -> tuple[str, str]:
    expected = set(expected_ids)
    if not expected.issubset(vector_ids):
        missing = expected.difference(vector_ids)
        return (
            "retrieval_problem",
            f"PGVector missed {len(missing)} of {len(expected)} expected chunks.",
        )
    if not expected.issubset(reranked_ids):
        missing = expected.difference(reranked_ids)
        return (
            "reranker_problem",
            f"Reranking removed {len(missing)} of {len(expected)} expected chunks.",
        )
    if not expected.issubset(context_ids):
        return (
            "context_building_problem",
            "Expected reranked evidence was not included in the LLM context.",
        )
    if correctness < 3.0:
        return (
            "llm_generation_problem",
            "Expected evidence reached the LLM, but answer correctness was low.",
        )
    if correctness >= 4.0 and (faithfulness < 0.5 or relevancy < 0.5):
        return (
            "evaluator_or_dataset_issue",
            "Correctness is high while another judge is low; inspect evaluator output and labels.",
        )
    return (
        "working_correctly",
        "Retrieval, reranking, context construction, and generation passed.",
    )


def metric_annotations(prefix: str, metrics: dict, mrr: float) -> list[dict]:
    annotations = []
    for cutoff in CUTOFFS:
        for metric in ("hit", "recall", "complete"):
            score = float(metrics[cutoff][metric])
            annotations.append(
                {
                    "name": f"{prefix} {metric.title()}@{cutoff}",
                    "annotator_kind": "CODE",
                    "label": "pass" if score >= 1.0 else "fail",
                    "score": score,
                    "explanation": (
                        f"{metric.title()}@{cutoff} against all expected chunks."
                    ),
                }
            )
    annotations.append(
        {
            "name": f"{prefix} MRR",
            "annotator_kind": "CODE",
            "label": "found" if mrr > 0 else "not_found",
            "score": float(mrr),
            "explanation": "Reciprocal rank of the first expected chunk.",
        }
    )
    return annotations


def distractor_annotations(prefix: str, metrics: dict) -> list[dict]:
    return [
        {
            "name": f"{prefix} Distractor Rate@{cutoff}",
            "annotator_kind": "CODE",
            "label": "clean" if metrics[cutoff]["rate"] == 0.0 else "contaminated",
            "score": float(metrics[cutoff]["rate"]),
            "explanation": (
                "Fraction of selected documents originating from the "
                "HotpotQA distractor corpus."
            ),
        }
        for cutoff in CUTOFFS
    ]


async def evaluate_row(row: dict, evaluators: tuple) -> dict:
    question = row["question"]
    expected_answer = row["expected_answer"]
    expected_ids = row["expected_chunk_ids"]
    with tracer.start_as_current_span("RAG Evaluation Query") as span:
        root_span_id = format_span_id(span.get_span_context().span_id)
        set_span_io(span, "CHAIN", input_value=question)
        span.set_attribute("eval.dataset", row["dataset"])
        span.set_attribute("eval.evaluation_id", row["evaluation_id"])
        span.set_attribute("eval.question", question)
        span.set_attribute("eval.expected_answer", expected_answer)
        set_json_attribute(span, "eval.expected_chunk_ids", expected_ids)

        started = time.perf_counter()
        nodes = await retrieve_nodes(question)
        retrieval_trace = getattr(retrieve_nodes, "last_trace", {})
        vector_documents = retrieval_trace.get(
            "vector_candidates_before_rerank",
            [],
        )
        reranked_documents = retrieval_trace.get(
            "final_chunks_after_rerank",
            [],
        )
        vector_ids = ranked_chunk_ids(vector_documents)
        reranked_ids = ranked_chunk_ids(reranked_documents)
        retrieval_metrics = {
            cutoff: ranking_metrics(expected_ids, vector_ids, cutoff)
            for cutoff in CUTOFFS
        }
        reranker_metrics = {
            cutoff: ranking_metrics(expected_ids, reranked_ids, cutoff)
            for cutoff in CUTOFFS
        }
        retrieval_distractors = {
            cutoff: distractor_metrics(vector_documents, cutoff)
            for cutoff in CUTOFFS
        }
        reranker_distractors = {
            cutoff: distractor_metrics(reranked_documents, cutoff)
            for cutoff in CUTOFFS
        }
        retrieval_mrr = reciprocal_rank(expected_ids, vector_ids)
        reranker_mrr = reciprocal_rank(expected_ids, reranked_ids)
        all_candidate_metrics = ranking_metrics(
            expected_ids,
            vector_ids,
            len(vector_ids),
        )
        expected_set = set(expected_ids)
        retrieved_expected_ids = [
            chunk_id for chunk_id in vector_ids if chunk_id in expected_set
        ]
        reranked_expected_ids = [
            chunk_id for chunk_id in reranked_ids if chunk_id in expected_set
        ]

        context, contexts, sources = build_context(nodes)
        context_ids = [source["chunk_id"] for source in sources]
        answer, usage = await generate_answer(question, context)
        context_tokens, context_estimated, tokenizer = await count_tokens(
            context,
            LLM_TOKENIZER_URL,
            LLM_TOKENIZER_NAME,
        )
        rag_latency_seconds = round(time.perf_counter() - started, 4)
        evaluation_started = time.perf_counter()

        correctness, correctness_reason, correctness_failed = await run_evaluator(
            "correctness_evaluator",
            {
                "question": question,
                "generated_answer": answer,
                "expected_answer": expected_answer,
            },
            lambda: evaluators[0].aevaluate(
                query=question,
                response=answer,
                reference=expected_answer,
            ),
        )
        faithfulness, faithfulness_reason, faithfulness_failed = await run_evaluator(
            "faithfulness_evaluator",
            {"generated_answer": answer, "contexts": contexts},
            lambda: evaluators[1].aevaluate(response=answer, contexts=contexts),
        )
        relevancy, relevancy_reason, relevancy_failed = await run_evaluator(
            "relevancy_evaluator",
            {"question": question, "generated_answer": answer},
            lambda: evaluators[2].aevaluate(
                query=question,
                response=answer,
                contexts=contexts,
            ),
        )
        evaluator_failed = (
            correctness_failed or faithfulness_failed or relevancy_failed
        )
        evaluation_latency_seconds = round(
            time.perf_counter() - evaluation_started,
            4,
        )
        hallucination = (
            0.0
            if evaluator_failed
            else max(0.0, min(1.0, 1.0 - faithfulness))
        )
        if evaluator_failed:
            diagnosis = "evaluation_failed"
            recommendation = (
                "At least one LlamaIndex judge failed. Retrieval and reranking "
                "metrics remain valid; rerun answer evaluation for this row."
            )
        else:
            diagnosis, recommendation = diagnose(
                expected_ids,
                vector_ids,
                reranked_ids,
                context_ids,
                correctness,
                faithfulness,
                relevancy,
            )

        token_usage = retrieval_trace.get("token_usage", {})
        retriever_tokens = int(token_usage.get("retriever_query_tokens", 0))
        reranker_tokens = int(token_usage.get("reranker_input_tokens", 0))
        pipeline_tokens = retriever_tokens + reranker_tokens + usage["total_tokens"]
        tokens_estimated = (
            bool(token_usage.get("is_estimated", False))
            or bool(usage.get("is_estimated", False))
            or context_estimated
        )

        result = {
            "evaluation_id": row["evaluation_id"],
            "dataset": row["dataset"],
            "domain": row.get("domain", ""),
            "difficulty": row.get("difficulty", ""),
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "expected_chunk_ids": json.dumps(expected_ids),
            "expected_chunk_count": len(expected_ids),
            "retrieved_expected_chunk_ids": json.dumps(retrieved_expected_ids),
            "missing_retrieval_chunk_ids": json.dumps(
                sorted(expected_set.difference(vector_ids))
            ),
            "reranked_expected_chunk_ids": json.dumps(reranked_expected_ids),
            "missing_reranker_chunk_ids": json.dumps(
                sorted(expected_set.difference(reranked_ids))
            ),
            "retrieval_mrr": retrieval_mrr,
            "reranker_mrr": reranker_mrr,
            "retrieval_candidate_count": len(vector_ids),
            "retrieval_recall_all_candidates": all_candidate_metrics["recall"],
            "retrieval_complete_all_candidates": all_candidate_metrics["complete"],
            "correctness": correctness,
            "correctness_normalized": max(0.0, min(1.0, correctness / 5.0)),
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "hallucination": hallucination,
            "evaluator_failed": evaluator_failed,
            "correctness_reason": correctness_reason,
            "faithfulness_reason": faithfulness_reason,
            "relevancy_reason": relevancy_reason,
            "diagnosis": diagnosis,
            "recommendation": recommendation,
            "retriever_query_tokens": retriever_tokens,
            "reranker_input_tokens": reranker_tokens,
            "context_tokens": context_tokens,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "llm_total_tokens": usage["total_tokens"],
            "pipeline_total_tokens": pipeline_tokens,
            "tokens_are_estimated": tokens_estimated,
            "tokenizer": tokenizer,
            "rag_latency_seconds": rag_latency_seconds,
            "evaluation_latency_seconds": evaluation_latency_seconds,
            "total_latency_seconds": round(time.perf_counter() - started, 4),
        }
        for prefix, metrics in (
            ("retrieval", retrieval_metrics),
            ("reranker", reranker_metrics),
        ):
            for cutoff in CUTOFFS:
                for metric in ("hit", "recall", "complete"):
                    result[f"{prefix}_{metric}_at_{cutoff}"] = metrics[cutoff][metric]
        for prefix, metrics in (
            ("retrieval", retrieval_distractors),
            ("reranker", reranker_distractors),
        ):
            for cutoff in CUTOFFS:
                result[f"{prefix}_hotpotqa_count_at_{cutoff}"] = metrics[cutoff][
                    "count"
                ]
                result[f"{prefix}_distractor_rate_at_{cutoff}"] = metrics[cutoff][
                    "rate"
                ]

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
        span.set_attribute("eval.correctness", correctness)
        span.set_attribute("eval.faithfulness", faithfulness)
        span.set_attribute("eval.relevancy", relevancy)
        span.set_attribute("eval.hallucination", hallucination)
        span.set_attribute("eval.evaluator_failed", evaluator_failed)
        span.set_attribute("eval.retrieval_mrr", retrieval_mrr)
        span.set_attribute("eval.reranker_mrr", reranker_mrr)
        span.set_attribute(
            "eval.retrieval.recall_all_candidates",
            float(all_candidate_metrics["recall"]),
        )
        span.set_attribute(
            "eval.retrieval.complete_all_candidates",
            float(all_candidate_metrics["complete"]),
        )
        for prefix, metrics in (
            ("retrieval", retrieval_metrics),
            ("reranker", reranker_metrics),
        ):
            for cutoff in CUTOFFS:
                for metric in ("hit", "recall", "complete"):
                    span.set_attribute(
                        f"eval.{prefix}.{metric}_at_{cutoff}",
                        float(metrics[cutoff][metric]),
                    )
        for prefix, metrics in (
            ("retrieval", retrieval_distractors),
            ("reranker", reranker_distractors),
        ):
            for cutoff in CUTOFFS:
                span.set_attribute(
                    f"eval.{prefix}.distractor_rate_at_{cutoff}",
                    float(metrics[cutoff]["rate"]),
                )
        span.set_attribute("rag.diagnosis", diagnosis)
        span.set_attribute("rag.latency_seconds", rag_latency_seconds)
        span.set_attribute(
            "eval.latency_seconds",
            evaluation_latency_seconds,
        )
        span.set_attribute("rag.token_count.total_pipeline", pipeline_tokens)
        span.set_attribute("rag.token_count.is_estimated", tokens_estimated)
        set_json_attribute(span, "rag.evaluation_output", result)
        set_json_attribute(
            span,
            "eval.missing_retrieval_chunk_ids",
            sorted(expected_set.difference(vector_ids)),
        )
        set_json_attribute(
            span,
            "eval.missing_reranker_chunk_ids",
            sorted(expected_set.difference(reranked_ids)),
        )

        log_document_relevance_annotations(
            retriever_span_id=retrieval_trace.get("retriever_span_id", ""),
            retrieved_documents=vector_documents,
            expected_document_id="",
            expected_chunk_id="",
            expected_chunk_ids=expected_ids,
        )
        log_span_annotations(
            root_span_id,
            metric_annotations("Retrieval", retrieval_metrics, retrieval_mrr)
            + metric_annotations("Reranker", reranker_metrics, reranker_mrr)
            + distractor_annotations("Retrieval", retrieval_distractors)
            + distractor_annotations("Reranker", reranker_distractors),
        )
        if evaluator_failed:
            log_span_annotations(
                root_span_id,
                [
                    {
                        "name": "Evaluation Status",
                        "annotator_kind": "CODE",
                        "label": "failed",
                        "score": 0.0,
                        "explanation": recommendation,
                    }
                ],
            )
        else:
            log_rag_answer_annotations(
                root_span_id=root_span_id,
                correctness_score=correctness,
                correctness_reason=correctness_reason,
                faithfulness_score=faithfulness,
                faithfulness_reason=faithfulness_reason,
                relevancy_score=relevancy,
                relevancy_reason=relevancy_reason,
                diagnosis=diagnosis,
                recommendation=recommendation,
            )
        return result


def failed_result(row: dict, error: Exception) -> dict:
    result = {
        "evaluation_id": row["evaluation_id"],
        "dataset": row["dataset"],
        "domain": row.get("domain", ""),
        "difficulty": row.get("difficulty", ""),
        "question": row["question"],
        "expected_answer": row["expected_answer"],
        "generated_answer": "",
        "expected_chunk_ids": json.dumps(row["expected_chunk_ids"]),
        "expected_chunk_count": len(row["expected_chunk_ids"]),
        "retrieved_expected_chunk_ids": "[]",
        "missing_retrieval_chunk_ids": json.dumps(row["expected_chunk_ids"]),
        "reranked_expected_chunk_ids": "[]",
        "missing_reranker_chunk_ids": json.dumps(row["expected_chunk_ids"]),
        "retrieval_mrr": 0.0,
        "reranker_mrr": 0.0,
        "retrieval_candidate_count": 0,
        "retrieval_recall_all_candidates": 0.0,
        "retrieval_complete_all_candidates": 0.0,
        "correctness": 0.0,
        "correctness_normalized": 0.0,
        "faithfulness": 0.0,
        "relevancy": 0.0,
        "hallucination": 1.0,
        "evaluator_failed": True,
        "correctness_reason": f"Evaluation failed: {error}",
        "faithfulness_reason": "",
        "relevancy_reason": "",
        "diagnosis": "evaluation_failed",
        "recommendation": str(error),
        "retriever_query_tokens": 0,
        "reranker_input_tokens": 0,
        "context_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_total_tokens": 0,
        "pipeline_total_tokens": 0,
        "tokens_are_estimated": True,
        "tokenizer": "",
        "rag_latency_seconds": 0.0,
        "evaluation_latency_seconds": 0.0,
        "total_latency_seconds": 0.0,
    }
    for prefix in ("retrieval", "reranker"):
        for cutoff in CUTOFFS:
            for metric in ("hit", "recall", "complete"):
                result[f"{prefix}_{metric}_at_{cutoff}"] = 0.0
            result[f"{prefix}_hotpotqa_count_at_{cutoff}"] = 0
            result[f"{prefix}_distractor_rate_at_{cutoff}"] = 0.0
    return result


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"total_questions": 0}

    def average(field: str) -> float:
        return mean(float(row[field]) for row in results)

    judged_results = [row for row in results if not row["evaluator_failed"]]

    def judged_average(field: str) -> float:
        if not judged_results:
            return 0.0
        return mean(float(row[field]) for row in judged_results)

    summary = {
        "total_questions": len(results),
        "answer_evaluated_questions": len(judged_results),
        "retrieval_mrr": average("retrieval_mrr"),
        "reranker_mrr": average("reranker_mrr"),
        "average_retrieval_candidate_count": average(
            "retrieval_candidate_count"
        ),
        "retrieval_recall_all_candidates": average(
            "retrieval_recall_all_candidates"
        ),
        "retrieval_complete_all_candidates": average(
            "retrieval_complete_all_candidates"
        ),
        "average_correctness": judged_average("correctness"),
        "average_correctness_normalized": judged_average("correctness_normalized"),
        "average_faithfulness": judged_average("faithfulness"),
        "average_relevancy": judged_average("relevancy"),
        "average_hallucination": judged_average("hallucination"),
        "average_rag_latency_seconds": average("rag_latency_seconds"),
        "average_evaluation_latency_seconds": average(
            "evaluation_latency_seconds"
        ),
        "average_total_latency_seconds": average("total_latency_seconds"),
        "average_retriever_query_tokens": average("retriever_query_tokens"),
        "average_reranker_input_tokens": average("reranker_input_tokens"),
        "average_context_tokens": average("context_tokens"),
        "average_prompt_tokens": average("prompt_tokens"),
        "average_completion_tokens": average("completion_tokens"),
        "average_llm_total_tokens": average("llm_total_tokens"),
        "average_pipeline_total_tokens": average("pipeline_total_tokens"),
        "working_correctly_rate": (
            sum(row["diagnosis"] == "working_correctly" for row in results)
            / len(results)
        ),
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in results)),
    }
    for prefix in ("retrieval", "reranker"):
        for cutoff in CUTOFFS:
            for metric in ("hit", "recall", "complete"):
                field = f"{prefix}_{metric}_at_{cutoff}"
                summary[field] = average(field)
            summary[f"{prefix}_average_hotpotqa_count_at_{cutoff}"] = average(
                f"{prefix}_hotpotqa_count_at_{cutoff}"
            )
            summary[f"{prefix}_distractor_rate_at_{cutoff}"] = average(
                f"{prefix}_distractor_rate_at_{cutoff}"
            )
    return summary


def log_summary_span(summary: dict) -> None:
    with tracer.start_as_current_span("HKPL Noise Evaluation Summary") as span:
        set_span_io(
            span,
            "EVALUATOR",
            input_value={
                "evaluation_dataset": "hkpl",
                "distractor_dataset": "hotpotqa",
                "vector_table": summary["vector_table"],
            },
            output_value=summary,
        )
        span.set_attribute("eval.dataset", "hkpl")
        span.set_attribute("eval.distractor_dataset", "hotpotqa")
        set_json_attribute(
            span,
            "eval.search_corpus_vectors",
            summary["search_corpus_vectors"],
        )
        span.set_attribute(
            "eval.total_questions",
            int(summary["total_questions"]),
        )
        for metric, value in summary.items():
            if isinstance(value, (int, float)):
                span.set_attribute(f"eval.{metric}", float(value))


async def main() -> None:
    args = parse_args()
    results_path, summary_path = report_paths(args)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")

    rows = load_hkpl_rows(args.limit)
    if not rows:
        raise RuntimeError("No valid HKPL evaluation rows were loaded.")
    corpus_counts = load_corpus_counts()
    if args.question_contains:
        needle = args.question_contains.casefold()
        rows = [row for row in rows if needle in row["question"].casefold()]
        if not rows:
            raise RuntimeError(
                f"No evaluation question contains {args.question_contains!r}."
            )
    elif args.question_exact:
        expected_question = args.question_exact.casefold().strip()
        rows = [
            row
            for row in rows
            if row["question"].casefold().strip() == expected_question
        ]
        if not rows:
            raise RuntimeError(
                f"No evaluation question exactly matches {args.question_exact!r}."
            )
    print(f"Loaded {len(rows)} HKPL evaluation rows.")
    print(
        f"Retriever searches combined vector table: data_{VECTOR_TABLE} "
        "(HKPL + HotpotQA distractors)"
    )
    print(f"Search corpus vectors: {corpus_counts}")
    print(f"Phoenix project: {os.getenv('PHOENIX_PROJECT_NAME', 'hkpl-rag')}")

    judge = QwenEvaluationLLM()
    evaluators = (
        CorrectnessEvaluator(llm=judge),
        FaithfulnessEvaluator(llm=judge),
        RelevancyEvaluator(llm=judge),
    )
    results = []
    for position, row in enumerate(rows, start=1):
        print(
            f"[{position}/{len(rows)}] [{row['dataset']}] {row['question']}"
        )
        try:
            result = await evaluate_row(row, evaluators)
        except Exception as error:
            print(f"FAILED: {error}")
            result = failed_result(row, error)
        results.append(result)
        print(
            "  "
            f"retrieval_complete@5={result['retrieval_complete_at_5']:.0f} "
            f"reranker_complete@5={result['reranker_complete_at_5']:.0f} "
            f"correctness={result['correctness']:.2f} "
            f"diagnosis={result['diagnosis']}"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = summarize(results)
    summary.update({
        "phoenix_project": os.getenv("PHOENIX_PROJECT_NAME", "hkpl-rag"),
        "vector_table": f"data_{VECTOR_TABLE}",
        "evaluation_dataset": "hkpl",
        "distractor_dataset": "hotpotqa",
        "search_corpus_vectors": corpus_counts,
        "metric_definitions": {
            "hit_at_k": "At least one expected chunk appears in the top K.",
            "recall_at_k": "Fraction of expected chunks appearing in the top K.",
            "complete_at_k": "All expected chunks appear in the top K.",
            "complete_all_candidates": (
                "All expected chunks appear anywhere in the vector candidates "
                "provided to the reranker."
            ),
            "mrr": "Reciprocal rank of the first expected chunk.",
            "hallucination": "One minus the LlamaIndex faithfulness score.",
            "distractor_rate_at_k": (
                "Fraction of the top K documents whose dataset metadata is "
                "hotpotqa."
            ),
        },
    })
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_summary_span(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
