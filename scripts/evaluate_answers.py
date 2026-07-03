#!/usr/bin/env python3

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from src.retrieval import retrieve_nodes
from src.nodes import http_llm


EVAL_FILE = PROJECT_ROOT / "data" / "evaluation_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_results.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "generation_summary.json"

MAX_CONTEXT_CHARS = 8000


class QwenLlamaIndexLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32000,
            num_output=1024,
            model_name="qwen3.5-9b-http",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        raise NotImplementedError("Use async evaluation only.")

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = await http_llm(
            prompt,
            temperature=0.0,
            max_tokens=1024,
        )
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming is not used for evaluation.")


def load_dataset() -> list[dict]:
    with EVAL_FILE.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_context(nodes) -> tuple[str, list[str], list[dict]]:
    parts = []
    contexts = []
    sources = []
    used_chars = 0

    for index, item in enumerate(nodes, start=1):
        node = item.node
        text = node.get_content()
        metadata = node.metadata or {}

        formatted = f"[Source {index}]\n{text}"

        if used_chars + len(formatted) > MAX_CONTEXT_CHARS:
            break

        parts.append(formatted)
        contexts.append(text)
        used_chars += len(formatted)

        sources.append(
            {
                "source_title": metadata.get("source_title", ""),
                "source_url": metadata.get("source_url") or metadata.get("url", ""),
                "document_id": metadata.get("kb_document_id")
                or metadata.get("document_id", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "score": item.score or 0.0,
            }
        )

    return "\n\n".join(parts), contexts, sources


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

    return await http_llm(
        prompt,
        temperature=0.0,
        max_tokens=512,
    )


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

    return any(
        expected_chunk_id == source.get("chunk_id", "")
        for source in sources
    )


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

    start = time.time()

    nodes = await retrieve_nodes(query)
    context, contexts, sources = build_context(nodes)

    answer = await generate_answer(query, context)

    retrieval_and_generation_latency = time.time() - start

    correctness_result = await correctness_evaluator.aevaluate(
        query=query,
        response=answer,
        reference=expected_answer,
    )

    faithfulness_result = await faithfulness_evaluator.aevaluate(
        response=answer,
        contexts=contexts,
    )

    relevancy_result = await relevancy_evaluator.aevaluate(
        query=query,
        response=answer,
        contexts=contexts,
    )

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
        "source_match_at_3": source_match(expected_document_id, sources),
        "chunk_match_at_3": chunk_match(expected_chunk_id, sources),
        "correctness_score": get_eval_score(correctness_result),
        "correctness_reason": get_eval_reason(correctness_result),
        "faithfulness_score": get_eval_score(faithfulness_result),
        "faithfulness_reason": get_eval_reason(faithfulness_result),
        "relevancy_score": get_eval_score(relevancy_result),
        "relevancy_reason": get_eval_reason(relevancy_result),
        "latency_seconds": round(retrieval_and_generation_latency, 4),
    }


async def main():
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
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

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
    print()
    print(f"Saved results to: {OUTPUT_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    asyncio.run(main())