#!/usr/bin/env python3

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import TextNode
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import format_span_id
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.db import engine
from src.infrastructure.embedding import embed_model
from src.infrastructure.vector_store import VECTOR_TABLE, vector_store
from src.nodes import http_llm
from src.observability import setup_phoenix_tracing
from src.phoenix_annotations import (
    log_document_relevance_annotations,
    log_rag_answer_annotations,
    log_span_annotations,
)
from src.retrieval import retrieve_nodes
from src.token_counting import LLM_TOKENIZER_NAME, LLM_TOKENIZER_URL, count_tokens
from src.tracing_helpers import set_json_attribute, set_llm_attributes, set_span_io

DATASET_NAME = "hotpotqa"
DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/"
    "hotpot_dev_distractor_v1.json?download=true"
)
DATASET_PATH = Path(
    os.getenv(
        "HOTPOTQA_DATASET_PATH",
        "/app/data/hotpotqa/hotpot_dev_distractor_v1.json",
    )
)
EVALUATION_TABLE = os.getenv(
    "HOTPOTQA_EVALUATION_TABLE",
    "hotpotqa_evaluation",
)
RESULTS_PATH = Path(
    os.getenv(
        "HOTPOTQA_RESULTS_PATH",
        "/app/data/hotpotqa/results.csv",
    )
)
SUMMARY_PATH = Path(
    os.getenv(
        "HOTPOTQA_SUMMARY_PATH",
        "/app/data/hotpotqa/summary.json",
    )
)
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "32768"))
TABLE_NAME = f"data_{VECTOR_TABLE}"

setup_phoenix_tracing()
tracer = trace.get_tracer("hotpotqa-benchmark")


class QwenEvaluationLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=LLM_CONTEXT_WINDOW,
            num_output=1024,
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
            max_tokens=1024,
            enable_thinking=False,
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming is not used by the benchmark.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest and evaluate HotpotQA in the existing HKPL PGVector table."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Download a deterministic dev subset and replace HotpotQA vectors.",
    )
    prepare.add_argument("--limit", type=int, default=1000)
    prepare.add_argument("--offset", type=int, default=0)
    prepare.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    prepare.add_argument("--force-download", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate retrieval/reranking and optionally answer generation.",
    )
    evaluate.add_argument("--limit", type=int, default=100)
    evaluate.add_argument("--answers", action="store_true")
    evaluate.add_argument(
        "--llama-evaluators",
        action="store_true",
        help="Run LlamaIndex correctness, faithfulness, and relevancy evaluators.",
    )
    return parser.parse_args()


def download_dataset(source_url: str, force: bool) -> None:
    if DATASET_PATH.is_file() and not force:
        return

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = DATASET_PATH.with_suffix(DATASET_PATH.suffix + ".part")
    with requests.get(source_url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    output.write(block)
    temporary_path.replace(DATASET_PATH)


def load_examples(offset: int, limit: int) -> list[dict]:
    with DATASET_PATH.open(encoding="utf-8") as source:
        examples = json.load(source)

    if not isinstance(examples, list):
        raise ValueError("HotpotQA source must contain a JSON list.")
    if offset < 0 or limit < 1:
        raise ValueError("Offset must be non-negative and limit must be positive.")
    selected = examples[offset : offset + limit]
    if not selected:
        raise ValueError(
            f"No examples selected from offset {offset} with limit {limit}."
        )
    return selected


def paragraph_id(title: str, paragraph: str) -> str:
    digest = hashlib.sha256(
        f"{title.strip()}\n{paragraph.strip()}".encode("utf-8")
    ).hexdigest()[:24]
    return f"hotpotqa:{digest}"


def build_nodes_and_rows(examples: list[dict]) -> tuple[list[TextNode], list[dict]]:
    paragraphs: dict[str, dict] = {}
    evaluation_rows: list[dict] = []

    for example in examples:
        example_id = str(example["_id"])
        title_to_chunk: dict[str, str] = {}

        for title, sentences in example.get("context", []):
            paragraph = "".join(str(sentence) for sentence in sentences).strip()
            if not paragraph:
                continue

            chunk_id = paragraph_id(str(title), paragraph)
            title_to_chunk[str(title)] = chunk_id
            record = paragraphs.setdefault(
                chunk_id,
                {
                    "title": str(title),
                    "text": paragraph,
                    "example_ids": set(),
                },
            )
            record["example_ids"].add(example_id)

        supporting_facts = example.get("supporting_facts", [])
        gold_titles = list(dict.fromkeys(str(item[0]) for item in supporting_facts))
        missing_titles = [title for title in gold_titles if title not in title_to_chunk]
        if missing_titles:
            raise ValueError(
                f"Example {example_id} is missing gold contexts: {missing_titles}"
            )

        evaluation_rows.append(
            {
                "example_id": example_id,
                "question": str(example["question"]),
                "expected_answer": str(example["answer"]),
                "question_type": str(example.get("type", "")),
                "difficulty": str(example.get("level", "")),
                "gold_titles": gold_titles,
                "gold_chunk_ids": [title_to_chunk[title] for title in gold_titles],
                "supporting_facts": supporting_facts,
            }
        )

    nodes: list[TextNode] = []
    for chunk_id, record in paragraphs.items():
        title = record["title"]
        node = TextNode(
            id_=chunk_id,
            text=record["text"],
            metadata={
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
                "source_type": "benchmark",
                "document_type": "prose",
                "chunk_strategy": "atomic",
                "document_version": 1,
                "hotpotqa_example_ids": sorted(record["example_ids"]),
            },
        )
        nodes.append(node)

    return nodes, evaluation_rows


def create_evaluation_table() -> None:
    with engine.begin() as connection:
        connection.execute(
            text(f"""
                CREATE TABLE IF NOT EXISTS {EVALUATION_TABLE} (
                    example_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    expected_answer TEXT NOT NULL,
                    question_type TEXT NOT NULL DEFAULT '',
                    difficulty TEXT NOT NULL DEFAULT '',
                    gold_titles JSONB NOT NULL,
                    gold_chunk_ids JSONB NOT NULL,
                    supporting_facts JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        )


def replace_hotpotqa_data(nodes: list[TextNode], rows: list[dict]) -> int:
    create_evaluation_table()
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

    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {EVALUATION_TABLE}"))
        for row in rows:
            connection.execute(
                text(f"""
                    INSERT INTO {EVALUATION_TABLE} (
                        example_id,
                        question,
                        expected_answer,
                        question_type,
                        difficulty,
                        gold_titles,
                        gold_chunk_ids,
                        supporting_facts
                    ) VALUES (
                        :example_id,
                        :question,
                        :expected_answer,
                        :question_type,
                        :difficulty,
                        CAST(:gold_titles AS JSONB),
                        CAST(:gold_chunk_ids AS JSONB),
                        CAST(:supporting_facts AS JSONB)
                    )
                """),
                {
                    **row,
                    "gold_titles": json.dumps(row["gold_titles"]),
                    "gold_chunk_ids": json.dumps(row["gold_chunk_ids"]),
                    "supporting_facts": json.dumps(row["supporting_facts"]),
                },
            )
    return int(deleted.rowcount or 0)


def load_evaluation_rows(limit: int) -> list[dict]:
    create_evaluation_table()
    with engine.connect() as connection:
        rows = connection.execute(
            text(f"""
                SELECT *
                FROM {EVALUATION_TABLE}
                ORDER BY example_id
                LIMIT :limit
            """),
            {"limit": limit},
        ).mappings().all()
    return [dict(row) for row in rows]


def ranked_chunk_ids(documents: list[dict]) -> list[str]:
    return [str(document.get("chunk_id", "")) for document in documents]


def retrieval_metrics(gold: list[str], retrieved: list[str], k: int) -> dict:
    expected = set(gold)
    selected = retrieved[:k]
    matched = expected.intersection(selected)
    return {
        "hit": float(bool(matched)),
        "recall": len(matched) / len(expected) if expected else 0.0,
        "complete": float(bool(expected) and expected.issubset(selected)),
    }


def reciprocal_rank(gold: list[str], retrieved: list[str]) -> float:
    expected = set(gold)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_scores(expected: str, generated: str) -> tuple[float, float]:
    expected_tokens = normalize_answer(expected).split()
    generated_tokens = normalize_answer(generated).split()
    exact_match = float(expected_tokens == generated_tokens)
    if not expected_tokens or not generated_tokens:
        return exact_match, float(expected_tokens == generated_tokens)

    common = Counter(expected_tokens) & Counter(generated_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return exact_match, 0.0
    precision = overlap / len(generated_tokens)
    recall = overlap / len(expected_tokens)
    return exact_match, 2 * precision * recall / (precision + recall)


async def generate_answer(question: str, nodes) -> tuple[str, dict, list[str]]:
    contexts = [node.node.get_content() for node in nodes]
    formatted_context = "\n\n".join(
        f"[Source {index}]\n{context}"
        for index, context in enumerate(contexts, start=1)
    )
    prompt = f"""You are answering a multi-hop question using retrieved evidence.
Use only the supplied sources. Combine evidence across sources when needed.
If the sources do not contain the answer, say that the evidence is insufficient.
Give a concise answer.

Sources:
{formatted_context}

Question: {question}
Answer:"""

    with tracer.start_as_current_span("LLM") as span:
        started = time.perf_counter()
        answer = await http_llm(
            prompt,
            temperature=0.0,
            max_tokens=256,
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
            max_tokens=256,
            usage=usage,
        )
        span.set_attribute(
            "llm.latency_seconds",
            round(time.perf_counter() - started, 4),
        )
    return answer, usage, contexts


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


async def evaluate(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise ValueError("Evaluation limit must be positive.")
    rows = load_evaluation_rows(args.limit)
    if not rows:
        raise RuntimeError("No HotpotQA rows found. Run the prepare command first.")

    evaluators = None
    if args.llama_evaluators:
        if not args.answers:
            raise ValueError("--llama-evaluators requires --answers.")
        judge = QwenEvaluationLLM()
        evaluators = (
            CorrectnessEvaluator(llm=judge),
            FaithfulnessEvaluator(llm=judge),
            RelevancyEvaluator(llm=judge),
        )

    results: list[dict] = []
    for position, row in enumerate(rows, start=1):
        question = row["question"]
        gold_chunk_ids = list(row["gold_chunk_ids"])
        print(f"[{position}/{len(rows)}] {question}")

        with tracer.start_as_current_span("HotpotQA RAG Query") as span:
            root_span_id = format_span_id(span.get_span_context().span_id)
            set_span_io(span, "CHAIN", input_value=question)
            span.set_attribute("eval.dataset", DATASET_NAME)
            span.set_attribute("eval.example_id", row["example_id"])
            set_json_attribute(span, "eval.gold_titles", row["gold_titles"])
            set_json_attribute(span, "eval.gold_chunk_ids", gold_chunk_ids)

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
            retrieval_tokens = retrieval_trace.get("token_usage", {})
            retriever_query_tokens = int(
                retrieval_tokens.get("retriever_query_tokens", 0)
            )
            reranker_input_tokens = int(
                retrieval_tokens.get("reranker_input_tokens", 0)
            )
            vector_ids = ranked_chunk_ids(vector_documents)
            reranked_ids = ranked_chunk_ids(reranked_documents)

            vector_metrics = {
                cutoff: retrieval_metrics(gold_chunk_ids, vector_ids, cutoff)
                for cutoff in (1, 3, 5, 10)
            }
            rerank_metrics = {
                cutoff: retrieval_metrics(gold_chunk_ids, reranked_ids, cutoff)
                for cutoff in (1, 3, 5)
            }
            vector_at_10 = vector_metrics[10]
            rerank_at_5 = rerank_metrics[5]
            result = {
                "example_id": row["example_id"],
                "question": question,
                "expected_answer": row["expected_answer"],
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
                "gold_chunk_count": len(gold_chunk_ids),
                "vector_mrr": reciprocal_rank(gold_chunk_ids, vector_ids),
                "rerank_mrr": reciprocal_rank(gold_chunk_ids, reranked_ids),
                "generated_answer": "",
                "answer_exact_match": 0.0,
                "answer_f1": 0.0,
                "correctness": 0.0,
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "retriever_query_tokens": retriever_query_tokens,
                "reranker_input_tokens": reranker_input_tokens,
                "context_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "pipeline_total_tokens": (
                    retriever_query_tokens + reranker_input_tokens
                ),
                "latency_seconds": 0.0,
                "diagnosis": "",
            }
            for cutoff, metrics in vector_metrics.items():
                for metric, value in metrics.items():
                    result[f"vector_{metric}_at_{cutoff}"] = value
            for cutoff, metrics in rerank_metrics.items():
                for metric, value in metrics.items():
                    result[f"rerank_{metric}_at_{cutoff}"] = value

            correctness_reason = "Answer evaluation was not requested."
            faithfulness_reason = correctness_reason
            relevancy_reason = correctness_reason
            if args.answers:
                answer, usage, contexts = await generate_answer(question, nodes)
                context_tokens, _, _ = await count_tokens(
                    "\n\n".join(contexts),
                    LLM_TOKENIZER_URL,
                    LLM_TOKENIZER_NAME,
                )
                exact_match, answer_f1 = answer_scores(
                    row["expected_answer"],
                    answer,
                )
                result.update(
                    {
                        "generated_answer": answer,
                        "answer_exact_match": exact_match,
                        "answer_f1": answer_f1,
                        "context_tokens": context_tokens,
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "pipeline_total_tokens": (
                            retriever_query_tokens
                            + reranker_input_tokens
                            + usage["total_tokens"]
                        ),
                    }
                )

            rag_latency = round(time.perf_counter() - started, 4)

            if args.answers and evaluators:
                correctness_result = await evaluators[0].aevaluate(
                    query=question,
                    response=answer,
                    reference=row["expected_answer"],
                )
                faithfulness_result = await evaluators[1].aevaluate(
                    response=answer,
                    contexts=contexts,
                )
                relevancy_result = await evaluators[2].aevaluate(
                    query=question,
                    response=answer,
                    contexts=contexts,
                )
                result.update(
                    {
                        "correctness": evaluator_score(correctness_result),
                        "faithfulness": evaluator_score(faithfulness_result),
                        "relevancy": evaluator_score(relevancy_result),
                    }
                )
                correctness_reason = evaluator_reason(correctness_result)
                faithfulness_reason = evaluator_reason(faithfulness_result)
                relevancy_reason = evaluator_reason(relevancy_result)

            if not vector_at_10["complete"]:
                diagnosis = "retrieval_problem"
            elif not rerank_at_5["complete"]:
                diagnosis = "reranker_problem"
            elif args.answers and result["answer_f1"] < 0.5:
                diagnosis = "llm_generation_problem"
            else:
                diagnosis = "working_correctly"
            result["diagnosis"] = diagnosis
            result["latency_seconds"] = rag_latency

            span.set_attribute(SpanAttributes.OUTPUT_VALUE, result["generated_answer"])
            span.set_attribute("eval.vector_recall_at_10", vector_at_10["recall"])
            span.set_attribute("eval.vector_complete_at_10", vector_at_10["complete"])
            span.set_attribute("eval.rerank_recall_at_5", rerank_at_5["recall"])
            span.set_attribute("eval.rerank_complete_at_5", rerank_at_5["complete"])
            span.set_attribute("eval.answer_exact_match", result["answer_exact_match"])
            span.set_attribute("eval.answer_f1", result["answer_f1"])
            span.set_attribute(
                "rag.token_count.retriever_query",
                result["retriever_query_tokens"],
            )
            span.set_attribute(
                "rag.token_count.reranker_input",
                result["reranker_input_tokens"],
            )
            span.set_attribute("rag.token_count.context", result["context_tokens"])
            span.set_attribute("rag.token_count.prompt", result["prompt_tokens"])
            span.set_attribute(
                "rag.token_count.completion",
                result["completion_tokens"],
            )
            span.set_attribute(
                "rag.token_count.total_pipeline",
                result["pipeline_total_tokens"],
            )
            span.set_attribute("rag.diagnosis", diagnosis)
            set_json_attribute(span, "rag.evaluation_output", result)

            retriever_span_id = retrieval_trace.get("retriever_span_id", "")
            log_document_relevance_annotations(
                retriever_span_id=retriever_span_id,
                retrieved_documents=vector_documents,
                expected_document_id="",
                expected_chunk_id="",
                expected_chunk_ids=gold_chunk_ids,
            )
            annotations = [
                {
                    "name": "Supporting Context Recall@10",
                    "annotator_kind": "CODE",
                    "label": "complete" if vector_at_10["complete"] else "incomplete",
                    "score": vector_at_10["recall"],
                    "explanation": "Fraction of gold supporting paragraphs retrieved.",
                },
                {
                    "name": "Reranker Complete@5",
                    "annotator_kind": "CODE",
                    "label": "complete" if rerank_at_5["complete"] else "incomplete",
                    "score": rerank_at_5["complete"],
                    "explanation": "Whether every gold paragraph survived reranking.",
                },
            ]
            if args.answers:
                annotations.append(
                    {
                        "name": "Answer F1",
                        "annotator_kind": "CODE",
                        "label": (
                            "correct" if result["answer_f1"] >= 0.8 else "incorrect"
                        ),
                        "score": result["answer_f1"],
                        "explanation": "Official-style normalized token F1.",
                    }
                )
            log_span_annotations(root_span_id, annotations)
            if evaluators:
                log_rag_answer_annotations(
                    root_span_id=root_span_id,
                    correctness_score=result["correctness"],
                    correctness_reason=correctness_reason,
                    faithfulness_score=result["faithfulness"],
                    faithfulness_reason=faithfulness_reason,
                    relevancy_score=result["relevancy"],
                    relevancy_reason=relevancy_reason,
                    diagnosis=diagnosis,
                    recommendation=(
                        "Inspect the failed retrieval stage shown by the diagnosis."
                        if diagnosis != "working_correctly"
                        else "All evaluated stages passed."
                    ),
                )

            results.append(result)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    numeric_fields = [
        "vector_mrr",
        "rerank_mrr",
        "retriever_query_tokens",
        "reranker_input_tokens",
        "pipeline_total_tokens",
        "latency_seconds",
    ]
    numeric_fields.extend(
        f"vector_{metric}_at_{cutoff}"
        for cutoff in (1, 3, 5, 10)
        for metric in ("hit", "recall", "complete")
    )
    if args.answers:
        numeric_fields.extend(
            [
                "answer_exact_match",
                "answer_f1",
                "context_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ]
        )
    if args.llama_evaluators:
        numeric_fields.extend(
            [
                "correctness",
                "faithfulness",
                "relevancy",
            ]
        )
    numeric_fields.extend(
        f"rerank_{metric}_at_{cutoff}"
        for cutoff in (1, 3, 5)
        for metric in ("hit", "recall", "complete")
    )
    summary = {
        "dataset": DATASET_NAME,
        "vector_table": TABLE_NAME,
        "total_questions": len(results),
        "answer_generation_enabled": bool(args.answers),
        "llama_evaluators_enabled": bool(args.llama_evaluators),
        **{
            f"average_{field}": sum(float(row[field]) for row in results)
            / len(results)
            for field in numeric_fields
        },
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in results)),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def prepare(args: argparse.Namespace) -> None:
    download_dataset(args.source_url, args.force_download)
    examples = load_examples(args.offset, args.limit)
    nodes, rows = build_nodes_and_rows(examples)
    deleted = replace_hotpotqa_data(nodes, rows)
    print(f"Removed previous HotpotQA chunks: {deleted}")
    print(f"HotpotQA evaluation examples: {len(rows)}")
    print(f"Unique HotpotQA paragraphs: {len(nodes)}")
    print(f"Shared vector table: {TABLE_NAME}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
