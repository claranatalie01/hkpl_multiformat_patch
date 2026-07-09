#!/usr/bin/env python3

import asyncio
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from src.nodes import http_llm
from src.observability import setup_phoenix_tracing
from src.rag_diagnosis import diagnose_rag
from src.retrieval import retrieve_nodes
from src.tracing_helpers import set_llm_attributes, set_span_io, set_json_attribute

setup_phoenix_tracing()
tracer = trace.get_tracer("hkpl-answer-evaluation")

EVAL_FILE = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_results.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "generation_summary.json"


class QwenLlamaIndexLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=1024,
            model_name="qwen3.5-9b-http",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("Use async evaluation only.")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = await http_llm(prompt, temperature=0.0, max_tokens=1024)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming is not used for evaluation.")


def load_dataset() -> list[dict]:
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_context(nodes) -> tuple[str, list[str], list[dict]]:
    with tracer.start_as_current_span("build_context") as span:
        set_span_io(span, "CHAIN", input_value={"num_nodes": len(nodes)})

        parts = []
        contexts = []
        sources = []

        for index, item in enumerate(nodes, start=1):
            node = item.node
            text = node.get_content()
            metadata = node.metadata or {}
            chunk_id = metadata.get("chunk_id", "")

            formatted = f"[Source {index}]\n{text}"
            parts.append(formatted)
            contexts.append(text)

            sources.append(
                {
                    "source_title": metadata.get("source_title", ""),
                    "source_url": metadata.get("source_url") or metadata.get("url", ""),
                    "document_id": (
                        metadata.get("kb_document_id")
                        or metadata.get("document_id")
                        or chunk_id.split(":")[0]
                        or ""
                    ),
                    "chunk_id": chunk_id,
                    "score": float(item.score or 0.0),
                    "text_preview": text[:700],
                }
            )

        context = "\n\n".join(parts)
        chunks_used = [source["chunk_id"] for source in sources]

        output_payload = {
            "chunks_sent_to_llm": chunks_used,
            "source_titles": [source["source_title"] for source in sources],
            "context_chars": len(context),
            "context_preview": context[:1500],
        }

        set_json_attribute(span, "rag.chunks_sent_to_llm", chunks_used)
        set_json_attribute(span, "rag.context_sources", sources)
        span.set_attribute("rag.context_chars", len(context))
        set_span_io(span, "CHAIN", output_value=output_payload)

        return context, contexts, sources


async def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are the official Hong Kong Public Libraries assistant.

Answer the user's question using ONLY the retrieved context.
Do not invent information.
If the context does not contain the answer, say:
"I don't have that information in my knowledge base."

Retrieved context:
{context}

Question:
{query}

Answer:
"""

    with tracer.start_as_current_span("LLM") as span:
        start = time.time()
        answer = await http_llm(prompt, temperature=0.0, max_tokens=512)
        latency = time.time() - start

        set_llm_attributes(
            span=span,
            model_name="qwen3.5-9b-http",
            prompt=prompt,
            response=answer,
            temperature=0.0,
            max_tokens=512,
        )
        span.set_attribute("llm.latency_seconds", round(latency, 4))
        span.set_attribute("rag.query", query)
        span.set_attribute("rag.context_chars", len(context))
        span.set_attribute("rag.generated_answer", answer)

        return answer


def source_match(expected_document_id: str, sources: list[dict]) -> bool:
    if not expected_document_id:
        return False

    for source in sources:
        document_id = source.get("document_id", "")
        chunk_id = source.get("chunk_id", "")

        if expected_document_id == document_id:
            return True

        if chunk_id.startswith(expected_document_id):
            return True

    return False


def chunk_match(expected_chunk_id: str, sources: list[dict]) -> bool:
    if not expected_chunk_id:
        return False

    return any(expected_chunk_id == source.get("chunk_id", "") for source in sources)


def get_eval_score(result) -> float:
    if hasattr(result, "score") and result.score is not None:
        return float(result.score)

    if hasattr(result, "passing"):
        return 1.0 if result.passing else 0.0

    return 0.0


def get_eval_reason(result) -> str:
    if hasattr(result, "feedback") and result.feedback:
        return str(result.feedback)

    if hasattr(result, "reason") and result.reason:
        return str(result.reason)

    return ""


async def run_evaluator_span(name: str, input_payload: dict, evaluator_call):
    with tracer.start_as_current_span(name) as span:
        set_span_io(span, "EVALUATOR", input_value=input_payload)
        result = await evaluator_call()
        score = get_eval_score(result)
        reason = get_eval_reason(result)

        set_span_io(
            span,
            "EVALUATOR",
            output_value={
                "score": score,
                "reason": reason,
            },
        )

        span.set_attribute(f"eval.{name}.score", float(score))
        span.set_attribute(f"eval.{name}.reason", reason)
        return result, score, reason


async def evaluate_one(
    row: dict,
    correctness_evaluator,
    faithfulness_evaluator,
    relevancy_evaluator,
) -> dict:
    query = row["query"]
    expected_answer = row.get("expected_answer_text", "")
    expected_document_id = row.get("source_document_id", "")
    expected_chunk_id = row.get("source_chunk_id", "")

    with tracer.start_as_current_span("HKPL RAG Query") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
        set_span_io(
            span,
            "CHAIN",
            input_value={
                "question": query,
                "expected_answer": expected_answer,
                "expected_document_id": expected_document_id,
                "expected_chunk_id": expected_chunk_id,
            },
        )

        span.set_attribute("eval.question", query)
        span.set_attribute("eval.expected_answer", expected_answer)
        span.set_attribute("eval.expected_source_document_id", expected_document_id)
        span.set_attribute("eval.expected_source_chunk_id", expected_chunk_id)

        start = time.time()

        nodes = await retrieve_nodes(query)
        context, contexts, sources = build_context(nodes)
        chunks_sent_to_llm = [source.get("chunk_id", "") for source in sources]

        answer = await generate_answer(query, context)
        retrieval_generation_latency = time.time() - start

        correctness_result, correctness_score, correctness_reason = await run_evaluator_span(
            "correctness_evaluator",
            {
                "query": query,
                "generated_answer": answer,
                "expected_answer": expected_answer,
            },
            lambda: correctness_evaluator.aevaluate(
                query=query,
                response=answer,
                reference=expected_answer,
            ),
        )

        faithfulness_result, faithfulness_score, faithfulness_reason = await run_evaluator_span(
            "faithfulness_evaluator",
            {
                "generated_answer": answer,
                "contexts_preview": [context[:700] for context in contexts],
            },
            lambda: faithfulness_evaluator.aevaluate(
                response=answer,
                contexts=contexts,
            ),
        )

        relevancy_result, relevancy_score, relevancy_reason = await run_evaluator_span(
            "relevancy_evaluator",
            {
                "query": query,
                "generated_answer": answer,
                "contexts_preview": [context[:700] for context in contexts],
            },
            lambda: relevancy_evaluator.aevaluate(
                query=query,
                response=answer,
                contexts=contexts,
            ),
        )

        source_match_value = source_match(expected_document_id, sources)
        chunk_match_value = chunk_match(expected_chunk_id, sources)

        retrieval_trace = getattr(retrieve_nodes, "last_trace", {})
        vector_candidates = retrieval_trace.get("vector_candidates_before_rerank", [])
        after_rerank = retrieval_trace.get("final_chunks_after_rerank", [])

        diagnostic = diagnose_rag(
            expected_document_id=expected_document_id,
            expected_chunk_id=expected_chunk_id,
            vector_candidates=vector_candidates,
            after_rerank=after_rerank,
            chunks_sent_to_llm=chunks_sent_to_llm,
            correctness_score=correctness_score,
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score,
        )

        diagnosis = diagnostic["diagnosis"]

        with tracer.start_as_current_span(f"rag_diagnosis:{diagnosis}") as diagnosis_span:
            set_span_io(
                diagnosis_span,
                "CHAIN",
                input_value={
                    "question": query,
                    "expected_answer": expected_answer,
                    "generated_answer": answer,
                },
                output_value=diagnostic,
            )

            for key, value in diagnostic.items():
                if isinstance(value, (str, int, float, bool)):
                    diagnosis_span.set_attribute(f"rag.{key}", value)
                else:
                    set_json_attribute(diagnosis_span, f"rag.{key}", value)

        output_payload = {
            "query": query,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "correctness": correctness_score,
            "faithfulness": faithfulness_score,
            "relevancy": relevancy_score,
            "source_match_at_3": source_match_value,
            "chunk_match_at_3": chunk_match_value,
            "latency_seconds": round(retrieval_generation_latency, 4),
            "diagnosis": diagnosis,
            "recommendation": diagnostic["recommendation"],
        }

        set_span_io(span, "CHAIN", output_value=output_payload)

        span.set_attribute("eval.correctness", float(correctness_score))
        span.set_attribute("eval.faithfulness", float(faithfulness_score))
        span.set_attribute("eval.relevancy", float(relevancy_score))
        span.set_attribute("eval.source_match_at_3", bool(source_match_value))
        span.set_attribute("eval.chunk_match_at_3", bool(chunk_match_value))
        span.set_attribute("eval.latency_seconds", float(round(retrieval_generation_latency, 4)))
        span.set_attribute("rag.diagnosis", diagnosis)
        span.set_attribute("rag.recommendation", diagnostic["recommendation"])

        top_source = sources[0] if sources else {}

        return {
            "domain": row.get("domain", ""),
            "query": query,
            "expected_answer_text": expected_answer,
            "generated_answer": answer,
            "expected_document_id": expected_document_id,
            "expected_chunk_id": expected_chunk_id,
            "top_source_title": top_source.get("source_title", ""),
            "top_document_id": top_source.get("document_id", ""),
            "top_chunk_id": top_source.get("chunk_id", ""),
            "source_match_at_3": source_match_value,
            "chunk_match_at_3": chunk_match_value,
            "correctness_score": correctness_score,
            "correctness_reason": correctness_reason,
            "faithfulness_score": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "relevancy_score": relevancy_score,
            "relevancy_reason": relevancy_reason,
            "diagnosis": diagnosis,
            "diagnosis_recommendation": diagnostic["recommendation"],
            "latency_seconds": round(retrieval_generation_latency, 4),
        }


async def main() -> None:
    rows = load_dataset()

    print(f"Loaded {len(rows)} evaluation rows.")

    llm = QwenLlamaIndexLLM()

    correctness_evaluator = CorrectnessEvaluator(llm=llm)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)

    results = []

    for index, row in enumerate(rows, start=1):
        print("=" * 80)
        print(f"[{index}/{len(rows)}] {row['query']}")

        try:
            result = await evaluate_one(
                row,
                correctness_evaluator,
                faithfulness_evaluator,
                relevancy_evaluator,
            )
            results.append(result)

            print(f"Correctness : {result['correctness_score']}")
            print(f"Faithfulness: {result['faithfulness_score']}")
            print(f"Relevancy   : {result['relevancy_score']}")
            print(f"Source@3    : {result['source_match_at_3']}")
            print(f"Diagnosis   : {result['diagnosis']}")
            print(f"Latency     : {result['latency_seconds']}s")

        except Exception as error:
            print(f"FAILED: {error}")

            results.append(
                {
                    "domain": row.get("domain", ""),
                    "query": row.get("query", ""),
                    "expected_answer_text": row.get("expected_answer_text", ""),
                    "generated_answer": "",
                    "expected_document_id": row.get("source_document_id", ""),
                    "expected_chunk_id": row.get("source_chunk_id", ""),
                    "top_source_title": "",
                    "top_document_id": "",
                    "top_chunk_id": "",
                    "source_match_at_3": False,
                    "chunk_match_at_3": False,
                    "correctness_score": 0.0,
                    "correctness_reason": f"Evaluation failed: {error}",
                    "faithfulness_score": 0.0,
                    "faithfulness_reason": "",
                    "relevancy_score": 0.0,
                    "relevancy_reason": "",
                    "diagnosis": "evaluation_failed",
                    "diagnosis_recommendation": str(error),
                    "latency_seconds": 0.0,
                }
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    total = len(results)

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in results) / total if total else 0.0

    def rate(key: str) -> float:
        return sum(bool(row[key]) for row in results) / total if total else 0.0

    summary = {
        "total_questions": total,
        "average_correctness": avg("correctness_score"),
        "average_faithfulness": avg("faithfulness_score"),
        "average_relevancy": avg("relevancy_score"),
        "source_match_at_3": rate("source_match_at_3"),
        "chunk_match_at_3": rate("chunk_match_at_3"),
        "average_latency_seconds": avg("latency_seconds"),
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in results)),
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("Answer Generation Evaluation Summary")
    print("=" * 80)
    print(f"Questions              : {summary['total_questions']}")
    print(f"Average correctness    : {summary['average_correctness']:.4f}")
    print(f"Average faithfulness   : {summary['average_faithfulness']:.4f}")
    print(f"Average relevancy      : {summary['average_relevancy']:.4f}")
    print(f"Source match@3         : {summary['source_match_at_3']:.2%}")
    print(f"Chunk match@3          : {summary['chunk_match_at_3']:.2%}")
    print(f"Average latency        : {summary['average_latency_seconds']:.4f}s")
    print("Diagnosis counts       :")
    for diagnosis, count in summary["diagnosis_counts"].items():
        print(f"  {diagnosis}: {count}")
    print()
    print(f"Saved results to: {OUTPUT_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
