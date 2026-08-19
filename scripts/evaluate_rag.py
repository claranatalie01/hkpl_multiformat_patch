#!/usr/bin/env python3
"""Evaluate retrieval, reranking, evidence coverage, and generated answers.

Rows from the approved evaluation table are run through the live RAG pipeline.
The script exports per-question results and aggregate diagnostics, records
Phoenix traces, and attributes failures to retrieval, reranking, context, or
answer generation. It reads the vector corpus but does not ingest documents.
"""

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
from src.corpus import is_distractor_metadata
from src.llm_client import http_llm, http_llm_with_usage
from src.observability import setup_phoenix_tracing
from src.phoenix_annotations import (
    log_document_relevance_annotations,
    log_rag_answer_annotations,
    log_span_annotations,
    normalize_evidence_text,
)
from src.retrieval import get_last_retrieval_trace, retrieve_nodes
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
EVALUATION_MAX_TOKENS = int(os.getenv("EVALUATION_MAX_TOKENS", "2048"))
DEFAULT_REASONING_BUDGET = 1000
RETRIEVER_HIT_CUTOFF = 10
RERANKER_HIT_CUTOFF = 5
CUTOFFS = (1, 3, 5, 10)

tracer = trace.get_tracer("hkpl-rag-noise-evaluation")

STRICT_CORRECTNESS_TEMPLATE = """
You are evaluating a factual question-answering system.

Compare the generated answer with every accepted reference answer. A generated
answer is correct if it matches any accepted reference answer in meaning.

Scoring rules:
- 5: Fully correct. Every required number, date component, name, location, and
  requested list item is present and correct, with no contradiction.
- 4: Correct core answer with only harmless wording differences or irrelevant
  extra detail. No required factual component is missing.
- 3: Partially correct or ambiguous. Use this when a required component or
  qualifier is missing, or when the answer selects one interpretation without
  resolving an ambiguity. For example, giving only a month and year when an
  exact day is required.
- 2: Relevant, but the core fact is wrong, or the answer refuses despite the
  reference supplying the answer.
- 1: Irrelevant or entirely incorrect.

Do not reward verbosity. Do not treat a partial date, number, identifier, phone
number, email address, or list as fully correct. Output the numeric score alone
on the first line. Provide a brief reason on the second line.

Question:
{query}

Accepted reference answers:
{reference_answer}

Generated answer:
{generated_answer}
"""

STRICT_FAITHFULNESS_TEMPLATE = """
Determine whether the answer is fully grounded in the retrieved context.

Rules:
- Answer YES only if every factual claim in the answer is directly supported
  by one or more passages in the context.
- Answer NO if any factual claim is unsupported, contradicted, more specific
  than the context, or supplied from outside knowledge.
- A correct core answer with unsupported additional claims must be NO.
- Ignore tone, formatting, and harmless restatement.
- Return only YES or NO.

Answer to check:
{query_str}

Retrieved context:
{context_str}

Verdict:
"""

STRICT_RELEVANCY_TEMPLATE = """
Determine whether the generated response directly answers the question and is
supported by the retrieved context.

Rules:
- Evidence may be combined from multiple passages in the context.
- Answer YES when the response directly addresses every requested item and the
  supporting information exists anywhere in the combined context.
- Answer NO when the response does not answer the question, misses a requested
  item, contradicts the context, or depends on unsupported information.
- Irrelevant additional context must not cause a correct response to fail.
- Return only YES or NO.

Question and generated response:
{query_str}

Combined retrieved context:
{context_str}

Verdict:
"""


class QwenEvaluationLLM(CustomLLM):
    enable_thinking: bool = False
    thinking_budget_tokens: int = DEFAULT_REASONING_BUDGET

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
            enable_thinking=self.enable_thinking,
            thinking_budget_tokens=self.thinking_budget_tokens,
        )
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("Streaming is not used by evaluation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HKPL questions while searching the combined HKPL and "
            "distractor vector corpora."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic limit on HKPL evaluation questions.",
    )
    parser.add_argument(
        "--phoenix-project",
        default=os.getenv("PHOENIX_PROJECT_NAME", "hkpl-rag"),
        help="Phoenix project that receives this evaluation run's traces.",
    )
    parser.add_argument(
        "--allow-incomplete-dataset",
        action="store_true",
        help=(
            "Allow diagnostic runs when some evaluation rows have stale "
            "evidence links. Full benchmark runs should never use this."
        ),
    )
    parser.add_argument(
        "--allow-missing-distractors",
        action="store_true",
        help=(
            "Allow diagnostic runs without both HotpotQA and Webz News. "
            "Full robustness benchmarks should never use this."
        ),
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
    parser.add_argument(
        "--rerun-answer-failures-from",
        type=Path,
        default=None,
        help=(
            "Evaluate only rows whose answer_pass value is false in a prior "
            "results CSV. The prior results file is never modified."
        ),
    )
    parser.add_argument(
        "--answer-reasoning",
        action="store_true",
        help="Enable bounded reasoning for answer generation in this run.",
    )
    parser.add_argument(
        "--evaluator-reasoning",
        action="store_true",
        help=(
            "Enable bounded reasoning for the evaluator. Leave this off when "
            "comparing with a no-reasoning baseline."
        ),
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=DEFAULT_REASONING_BUDGET,
        help="Maximum thinking tokens per reasoning-enabled LLM call (default: 1000).",
    )
    return parser.parse_args()


def report_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    is_full_evaluation = (
        args.limit is None
        and not args.question_contains
        and not args.question_exact
        and args.rerun_answer_failures_from is None
        and not args.answer_reasoning
        and not args.evaluator_reasoning
    )
    if is_full_evaluation:
        return RESULTS_PATH, SUMMARY_PATH

    tags = ["hkpl"]
    if args.limit is not None:
        tags.append(f"limit-{args.limit}")
    if args.rerun_answer_failures_from is not None:
        tags.append("answer-failures")
    if args.answer_reasoning:
        tags.append(f"answer-reasoning-{args.reasoning_budget}")
    if args.evaluator_reasoning:
        tags.append(f"evaluator-reasoning-{args.reasoning_budget}")

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


def prior_failure_selection(results_path: Path) -> dict:
    """Load failed answer IDs and baseline counts without changing the CSV."""
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Prior evaluation results file does not exist: {results_path}"
        )

    failed_ids: set[str] = set()
    evaluated_questions = 0
    passed_questions = 0
    with results_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"evaluation_id", "answer_pass"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Prior results CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            evaluated_questions += 1
            answer_pass = str(row.get("answer_pass") or "").strip().casefold()
            if answer_pass in {"true", "1", "yes"}:
                passed_questions += 1
            else:
                failed_ids.add(str(row["evaluation_id"]).strip())

    if not failed_ids:
        raise RuntimeError(
            f"No failed answers were found in prior results: {results_path}"
        )
    return {
        "source": str(results_path),
        "evaluated_questions": evaluated_questions,
        "passed_questions": passed_questions,
        "failed_questions": len(failed_ids),
        "failed_ids": failed_ids,
    }


def safe_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL table name: {value!r}")
    return value


def parse_accepted_answers(value, primary_answer: str) -> list[str]:
    if isinstance(value, str):
        try:
            aliases = json.loads(value or "[]")
        except json.JSONDecodeError:
            aliases = []
    elif isinstance(value, list):
        aliases = value
    else:
        aliases = []

    answers = [primary_answer, *aliases]
    return list(dict.fromkeys(
        str(answer).strip() for answer in answers if str(answer).strip()
    ))


def parse_json_string_list(value, fallback: list[str]) -> list[str]:
    """Read a JSON/JSONB string list, using legacy singular labels if empty."""
    if isinstance(value, str):
        try:
            values = json.loads(value or "[]")
        except json.JSONDecodeError:
            values = []
    elif isinstance(value, list):
        values = value
    else:
        values = []
    cleaned = [
        str(item).strip() for item in values
        if isinstance(item, str) and item.strip()
    ]
    # Do not deduplicate here: repeated text in different chunks must retain
    # the same positional relationship as source_chunk_ids_json.
    return cleaned or fallback


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
                    expected_context_snippet,
                    expected_context_snippets_json,
                    accepted_answers_json,
                    source_document_id,
                    source_chunk_id,
                    source_chunk_ids_json
                FROM {table}
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(
                        CASE
                            WHEN jsonb_array_length({table}.source_chunk_ids_json) > 0
                            THEN {table}.source_chunk_ids_json
                            ELSE jsonb_build_array({table}.source_chunk_id)
                        END
                    ) AS expected_chunks(chunk_id)
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM {vector_table} knowledge
                        WHERE knowledge.metadata_->>'chunk_id' = expected_chunks.chunk_id
                    )
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
            "accepted_answers": parse_accepted_answers(
                row.get("accepted_answers_json"),
                str(row["expected_answer_text"]),
            ),
            "expected_context_snippet": str(
                row.get("expected_context_snippet") or ""
            ),
            "expected_context_snippets": parse_json_string_list(
                row.get("expected_context_snippets_json"),
                [str(row.get("expected_context_snippet") or "")],
            ),
            "expected_document_ids": [str(row["source_document_id"])],
            "expected_chunk_ids": parse_json_string_list(
                row.get("source_chunk_ids_json"),
                [str(row["source_chunk_id"])],
            ),
        }
        for row in rows
    ]


def load_hkpl_coverage() -> dict[str, int | float]:
    """Count intended labels and labels still linked to searchable chunks."""
    table = safe_table_name(HKPL_EVALUATION_TABLE)
    vector_table = safe_table_name(f"data_{VECTOR_TABLE}")
    with engine.connect() as connection:
        row = connection.execute(
            text(f"""
                SELECT
                    COUNT(*) AS intended_questions,
                    COUNT(*) FILTER (
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM jsonb_array_elements_text(
                                CASE
                                    WHEN jsonb_array_length({table}.source_chunk_ids_json) > 0
                                    THEN {table}.source_chunk_ids_json
                                    ELSE jsonb_build_array({table}.source_chunk_id)
                                END
                            ) AS expected_chunks(chunk_id)
                            WHERE NOT EXISTS (
                                SELECT 1
                                FROM {vector_table} knowledge
                                WHERE knowledge.metadata_->>'chunk_id' = expected_chunks.chunk_id
                            )
                        )
                    ) AS valid_questions
                FROM {table}
            """)
        ).mappings().one()

    intended = int(row["intended_questions"])
    valid = int(row["valid_questions"])
    return {
        "intended_questions": intended,
        "valid_questions": valid,
        "evaluation_dataset_coverage": (
            valid / intended if intended else 0.0
        ),
    }


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


def evidence_match(
    documents: list[dict],
    expected_snippets: list[str],
    k: int,
    accepted_answers: list[str] | None = None,
) -> dict:
    required_phrases = list(dict.fromkeys(
        normalize_evidence_text(snippet)
        for snippet in expected_snippets
        if normalize_evidence_text(snippet)
    ))
    if not required_phrases:
        required_phrases = [
            normalize_evidence_text(answer)
            for answer in accepted_answers or []
            if len(re.sub(r"\W+", "", normalize_evidence_text(answer))) >= 4
        ]
    matched_ids: list[str] = []
    matched_phrases: set[str] = set()
    for document in documents[:k]:
        document_text = normalize_evidence_text(
            document.get("text") or document.get("text_preview") or ""
        )
        document_matches = {
            phrase for phrase in required_phrases if phrase in document_text
        }
        if document_matches:
            matched_ids.append(str(document.get("chunk_id") or ""))
            matched_phrases.update(document_matches)
    return {
        "hit": float(bool(matched_ids)),
        "complete": float(
            bool(required_phrases)
            and set(required_phrases).issubset(matched_phrases)
        ),
        "matched_chunk_ids": matched_ids,
    }


def reference_match_mode(exact_match: bool, evidence_match_found: bool) -> str:
    if exact_match:
        return "exact_chunk"
    if evidence_match_found:
        return "equivalent_evidence"
    return "missing"


def distractor_metrics(documents: list[dict], k: int) -> dict:
    selected = documents[:k]
    datasets = Counter(
        str(document.get("metadata", {}).get("dataset") or "unknown").lower()
        for document in selected
        if is_distractor_metadata(document.get("metadata"))
    )
    distractor_count = sum(datasets.values())
    return {
        "count": distractor_count,
        "rate": distractor_count / len(selected) if selected else 0.0,
        "by_dataset": dict(sorted(datasets.items())),
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


async def generate_answer(
    question: str,
    context: str,
    *,
    enable_thinking: bool = False,
    thinking_budget_tokens: int = DEFAULT_REASONING_BUDGET,
) -> tuple[str, dict]:
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
        llm_response = await http_llm_with_usage(
            prompt,
            temperature=0.0,
            max_tokens=EVALUATION_MAX_TOKENS,
            enable_thinking=enable_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        answer = llm_response.text
        usage = llm_response.usage
        if llm_response.reasoning_text and not usage["reasoning_tokens"]:
            reasoning_tokens, _, _ = await count_tokens(
                llm_response.reasoning_text,
                LLM_TOKENIZER_URL,
                LLM_TOKENIZER_NAME,
            )
            usage["reasoning_tokens"] = reasoning_tokens
        if not usage["total_tokens"]:
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
                "reasoning_tokens": 0,
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
    retrieval_match_mode: str,
    reranker_match_mode: str,
    context_match_mode: str,
    correctness: float,
    faithfulness: float,
    relevancy: float,
) -> tuple[str, str]:
    answer_is_correct = (
        correctness >= 4.0
        and faithfulness >= 0.5
        and relevancy >= 0.5
    )
    if retrieval_match_mode == "missing":
        if answer_is_correct:
            return (
                "correct_answer_evidence_label_miss",
                "The answer is correct, but neither the labeled chunk nor a "
                "recognized accepted-answer passage was detected in the "
                "retrieval pool. Review alternative evidence and benchmark "
                "labels.",
            )
        return (
            "retrieval_problem",
            "PGVector retrieved neither the labeled chunk nor recognized "
            "accepted-answer evidence, and the final answer was not fully correct.",
        )
    if reranker_match_mode == "missing":
        if answer_is_correct:
            return (
                "correct_answer_reranker_evidence_miss",
                "The answer is correct, but recognized benchmark evidence was "
                "not retained in the final reranked context.",
            )
        return (
            "reranker_problem",
            "Reranking removed both the labeled chunk and equivalent gold evidence.",
        )
    if context_match_mode == "missing":
        return (
            "context_building_problem",
            "Neither the labeled chunk nor equivalent gold evidence reached the LLM context.",
        )
    if correctness < 4.0:
        return (
            "llm_generation_problem",
            "Expected evidence reached the LLM, but the answer was partial, "
            "ambiguous, or incorrect.",
        )
    if correctness >= 4.0 and relevancy < 0.5:
        return (
            "irrelevant_answer",
            "The answer may contain correct facts but does not adequately "
            "address the question.",
        )
    if correctness >= 4.0 and faithfulness < 0.5:
        return (
            "ungrounded_answer",
            "The answer may be factually correct, but at least one claim is "
            "not supported by the retrieved context.",
        )
    return (
        "working_correctly",
        "Retrieval, reranking, context construction, and generation passed "
        f"(retrieval={retrieval_match_mode}, reranker={reranker_match_mode}).",
    )


def answer_outcome(
    correctness: float,
    faithfulness: float,
    relevancy: float,
    evaluator_failed: bool,
) -> str:
    if evaluator_failed:
        return "not_evaluated"
    if correctness < 3.0:
        return "incorrect"
    if correctness < 4.0:
        return "partial_or_ambiguous"
    if relevancy < 0.5:
        return "irrelevant"
    if faithfulness < 0.5:
        return "ungrounded"
    return "correct"


def evidence_outcome(
    retrieval_match_mode: str,
    reranker_match_mode: str,
    context_match_mode: str,
) -> str:
    if retrieval_match_mode == "missing":
        return "missing_from_retrieval_pool"
    if reranker_match_mode == "missing":
        return "removed_by_reranker"
    if context_match_mode == "missing":
        return "missing_from_llm_context"
    return "reached_llm_context"


def focused_pipeline_annotations(
    retrieval_match_mode: str,
    reranker_match_mode: str,
    reranker_distractors_at_5: dict,
) -> list[dict]:
    retrieval_hit = retrieval_match_mode != "missing"
    reranker_hit = reranker_match_mode != "missing"
    return [
        {
            "name": "Retriever Hit@10",
            "annotator_kind": "CODE",
            "label": "pass" if retrieval_hit else "fail",
            "score": float(retrieval_hit),
            "explanation": (
                "Whether the exact labeled chunk or equivalent labeled "
                "evidence appeared among the ten vector candidates."
            ),
        },
        {
            "name": "Reranker Hit@5",
            "annotator_kind": "CODE",
            "label": "pass" if reranker_hit else "fail",
            "score": float(reranker_hit),
            "explanation": (
                "Whether the exact labeled chunk or equivalent labeled "
                "evidence appeared among the five reranked context chunks."
            ),
        },
        {
            "name": "Distractor Rate@5",
            "annotator_kind": "CODE",
            "label": (
                "clean"
                if reranker_distractors_at_5["rate"] == 0.0
                else "contaminated"
            ),
            "score": float(reranker_distractors_at_5["rate"]),
            "explanation": (
                "Fraction of the five final context chunks originating from "
                "configured distractor corpora."
            ),
        },
    ]


async def evaluate_row(
    row: dict,
    evaluators: tuple,
    *,
    answer_reasoning: bool = False,
    reasoning_budget: int = DEFAULT_REASONING_BUDGET,
) -> dict:
    question = row["question"]
    expected_answer = row["expected_answer"]
    accepted_answers = row.get("accepted_answers") or [expected_answer]
    expected_ids = row["expected_chunk_ids"]
    expected_snippet = row.get("expected_context_snippet", "")
    expected_snippets = row.get("expected_context_snippets") or [expected_snippet]
    with tracer.start_as_current_span("RAG Evaluation Query") as span:
        root_span_id = format_span_id(span.get_span_context().span_id)
        set_span_io(span, "CHAIN", input_value=question)
        span.set_attribute("eval.dataset", row["dataset"])
        span.set_attribute("eval.evaluation_id", row["evaluation_id"])
        span.set_attribute("eval.question", question)
        span.set_attribute("eval.expected_answer", expected_answer)
        set_json_attribute(span, "eval.accepted_answers", accepted_answers)
        span.set_attribute("eval.expected_context_snippet", expected_snippet)
        set_json_attribute(span, "eval.expected_context_snippets", expected_snippets)
        set_json_attribute(span, "eval.expected_chunk_ids", expected_ids)

        started = time.perf_counter()
        nodes = await retrieve_nodes(question, include_distractors=True)
        retrieval_trace = get_last_retrieval_trace()
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
        retrieval_evidence_metrics = {
            cutoff: evidence_match(
                vector_documents,
                expected_snippets,
                cutoff,
                accepted_answers,
            )
            for cutoff in CUTOFFS
        }
        reranker_evidence_metrics = {
            cutoff: evidence_match(
                reranked_documents,
                expected_snippets,
                cutoff,
                accepted_answers,
            )
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
        all_candidate_evidence = evidence_match(
            vector_documents,
            expected_snippets,
            len(vector_documents),
            accepted_answers,
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
        retrieval_at_10_match_mode = reference_match_mode(
            bool(retrieval_metrics[RETRIEVER_HIT_CUTOFF]["complete"]),
            bool(
                retrieval_evidence_metrics[RETRIEVER_HIT_CUTOFF]["complete"]
            ),
        )
        retrieval_match_mode = reference_match_mode(
            bool(all_candidate_metrics["complete"]),
            bool(all_candidate_evidence["complete"]),
        )
        reranker_match_mode = reference_match_mode(
            bool(reranker_metrics[RERANKER_HIT_CUTOFF]["complete"]),
            bool(reranker_evidence_metrics[RERANKER_HIT_CUTOFF]["complete"]),
        )
        context_match_mode = reference_match_mode(
            set(expected_ids).issubset(context_ids),
            bool(
                evidence_match(
                    reranked_documents,
                    expected_snippets,
                    len(reranked_documents),
                    accepted_answers,
                )["complete"]
            ),
        )
        answer, usage = await generate_answer(
            question,
            context,
            enable_thinking=answer_reasoning,
            thinking_budget_tokens=reasoning_budget,
        )
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
                "accepted_answers": accepted_answers,
            },
            lambda: evaluators[0].aevaluate(
                query=question,
                response=answer,
                reference="\n".join(
                    f"- {answer}" for answer in accepted_answers
                ),
            ),
        )
        faithfulness, faithfulness_reason, faithfulness_failed = await run_evaluator(
            "faithfulness_evaluator",
            {"generated_answer": answer, "contexts": contexts},
            lambda: evaluators[1].aevaluate(
                response=answer,
                contexts=[context],
            ),
        )
        relevancy, relevancy_reason, relevancy_failed = await run_evaluator(
            "relevancy_evaluator",
            {"question": question, "generated_answer": answer},
            lambda: evaluators[2].aevaluate(
                query=question,
                response=answer,
                contexts=[context],
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
                retrieval_match_mode,
                reranker_match_mode,
                context_match_mode,
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
        final_answer_outcome = answer_outcome(
            correctness,
            faithfulness,
            relevancy,
            evaluator_failed,
        )

        result = {
            "evaluation_id": row["evaluation_id"],
            "dataset": row["dataset"],
            "domain": row.get("domain", ""),
            "difficulty": row.get("difficulty", ""),
            "question": question,
            "expected_answer": expected_answer,
            "accepted_answers": json.dumps(
                accepted_answers,
                ensure_ascii=False,
            ),
            "expected_context_snippet": expected_snippet,
            "expected_context_snippets": json.dumps(
                expected_snippets, ensure_ascii=False
            ),
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
            "retrieval_evidence_hit_all_candidates": all_candidate_evidence["hit"],
            "retrieval_evidence_chunk_ids_all_candidates": json.dumps(
                all_candidate_evidence["matched_chunk_ids"]
            ),
            "retrieval_reference_match_mode": retrieval_match_mode,
            "retrieval_reference_match_mode_at_10": (
                retrieval_at_10_match_mode
            ),
            "reranker_reference_match_mode": reranker_match_mode,
            "context_reference_match_mode": context_match_mode,
            "correctness": correctness,
            "correctness_normalized": max(0.0, min(1.0, correctness / 5.0)),
            "answer_outcome": final_answer_outcome,
            "answer_pass": final_answer_outcome == "correct",
            "answer_reasoning_enabled": answer_reasoning,
            "reasoning_budget_tokens": (
                reasoning_budget if answer_reasoning else 0
            ),
            "evidence_outcome": evidence_outcome(
                retrieval_match_mode,
                reranker_match_mode,
                context_match_mode,
            ),
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
            "reasoning_tokens": usage["reasoning_tokens"],
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
            ("retrieval", retrieval_evidence_metrics),
            ("reranker", reranker_evidence_metrics),
        ):
            for cutoff in CUTOFFS:
                result[f"{prefix}_evidence_hit_at_{cutoff}"] = metrics[cutoff]["hit"]
                result[f"{prefix}_evidence_chunk_ids_at_{cutoff}"] = json.dumps(
                    metrics[cutoff]["matched_chunk_ids"]
                )
        for prefix, metrics in (
            ("retrieval", retrieval_distractors),
            ("reranker", reranker_distractors),
        ):
            for cutoff in CUTOFFS:
                result[f"{prefix}_distractor_count_at_{cutoff}"] = metrics[cutoff][
                    "count"
                ]
                result[f"{prefix}_distractor_rate_at_{cutoff}"] = metrics[cutoff][
                    "rate"
                ]
                result[f"{prefix}_distractor_datasets_at_{cutoff}"] = json.dumps(
                    metrics[cutoff]["by_dataset"],
                    sort_keys=True,
                )

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
        span.set_attribute(
            "eval.retrieval.evidence_hit_all_candidates",
            float(all_candidate_evidence["hit"]),
        )
        span.set_attribute("eval.retrieval.reference_match_mode", retrieval_match_mode)
        span.set_attribute("eval.reranker.reference_match_mode", reranker_match_mode)
        span.set_attribute("eval.context.reference_match_mode", context_match_mode)
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
            ("retrieval", retrieval_evidence_metrics),
            ("reranker", reranker_evidence_metrics),
        ):
            for cutoff in CUTOFFS:
                span.set_attribute(
                    f"eval.{prefix}.evidence_hit_at_{cutoff}",
                    float(metrics[cutoff]["hit"]),
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
                set_json_attribute(
                    span,
                    f"eval.{prefix}.distractor_datasets_at_{cutoff}",
                    metrics[cutoff]["by_dataset"],
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
            expected_context_snippet=expected_snippet,
        )
        log_span_annotations(
            root_span_id,
            focused_pipeline_annotations(
                retrieval_at_10_match_mode,
                reranker_match_mode,
                reranker_distractors[5],
            ),
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
        "accepted_answers": json.dumps(
            row.get("accepted_answers") or [row["expected_answer"]],
            ensure_ascii=False,
        ),
        "expected_context_snippet": row.get("expected_context_snippet", ""),
        "expected_context_snippets": json.dumps(
            row.get("expected_context_snippets")
            or [row.get("expected_context_snippet", "")],
            ensure_ascii=False,
        ),
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
        "retrieval_evidence_hit_all_candidates": 0.0,
        "retrieval_evidence_chunk_ids_all_candidates": "[]",
        "retrieval_reference_match_mode": "missing",
        "retrieval_reference_match_mode_at_10": "missing",
        "reranker_reference_match_mode": "missing",
        "context_reference_match_mode": "missing",
        "correctness": 0.0,
        "correctness_normalized": 0.0,
        "answer_outcome": "not_evaluated",
        "answer_pass": False,
        "answer_reasoning_enabled": False,
        "reasoning_budget_tokens": 0,
        "evidence_outcome": "not_evaluated",
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
        "reasoning_tokens": 0,
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
            result[f"{prefix}_evidence_hit_at_{cutoff}"] = 0.0
            result[f"{prefix}_evidence_chunk_ids_at_{cutoff}"] = "[]"
            result[f"{prefix}_distractor_count_at_{cutoff}"] = 0
            result[f"{prefix}_distractor_rate_at_{cutoff}"] = 0.0
            result[f"{prefix}_distractor_datasets_at_{cutoff}"] = "{}"
    return result


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(results: list[dict]) -> dict:
    if not results:
        return {
            "hit_rate": {},
            "average_token_usage": {},
            "quality_guardrails": {},
            "latency_seconds": {},
            "robustness": {},
            "reliability": {},
            "per_domain": {},
        }

    def average(field: str) -> float:
        return mean(float(row[field]) for row in results)

    judged_results = [row for row in results if not row["evaluator_failed"]]

    def judged_average(field: str) -> float:
        if not judged_results:
            return 0.0
        return mean(float(row[field]) for row in judged_results)

    def summarize_group(group: list[dict]) -> dict:
        judged_group = [row for row in group if not row["evaluator_failed"]]

        def group_average(field: str, rows: list[dict] = group) -> float:
            if not rows:
                return 0.0
            return mean(float(row[field]) for row in rows)

        def judged_group_average(field: str) -> float:
            return group_average(field, judged_group)

        def reference_hit(field: str) -> float:
            if not group:
                return 0.0
            return mean(
                1.0 if row[field] != "missing" else 0.0
                for row in group
            )

        return {
            "questions": len(group),
            "retriever_hit_at_10": reference_hit(
                "retrieval_reference_match_mode_at_10"
            ),
            "reranker_hit_at_5": reference_hit(
                "reranker_reference_match_mode"
            ),
            "average_correctness": judged_group_average("correctness"),
            "average_faithfulness": judged_group_average("faithfulness"),
            "average_relevancy": judged_group_average("relevancy"),
            "average_pipeline_total_tokens": group_average(
                "pipeline_total_tokens"
            ),
        }

    rag_latencies = [float(row["rag_latency_seconds"]) for row in results]
    domains = sorted({str(row.get("domain") or "unspecified") for row in results})
    summary = {
        "hit_rate": {
            "retriever_at_10": mean(
                1.0
                if row["retrieval_reference_match_mode_at_10"] != "missing"
                else 0.0
                for row in results
            ),
            "reranker_at_5": mean(
                1.0
                if row["reranker_reference_match_mode"] != "missing"
                else 0.0
                for row in results
            ),
        },
        "average_token_usage": {
            "embedding_query_tokens": average("retriever_query_tokens"),
            "reranker_input_tokens": average("reranker_input_tokens"),
            "context_tokens": average("context_tokens"),
            "llm_prompt_tokens": average("prompt_tokens"),
            "llm_completion_tokens": average("completion_tokens"),
            "llm_reasoning_tokens": average("reasoning_tokens"),
            "llm_total_tokens": average("llm_total_tokens"),
            "pipeline_total_tokens": average("pipeline_total_tokens"),
        },
        "quality_guardrails": {
            "answer_pass_rate": (
                mean(
                    1.0 if row["answer_pass"] else 0.0
                    for row in judged_results
                )
                if judged_results
                else 0.0
            ),
            "average_correctness": judged_average("correctness"),
            "average_faithfulness": judged_average("faithfulness"),
            "average_relevancy": judged_average("relevancy"),
        },
        "latency_seconds": {
            "rag_p50": percentile(rag_latencies, 0.50),
            "rag_p95": percentile(rag_latencies, 0.95),
        },
        "robustness": {
            "distractor_rate_at_5": average(
                "reranker_distractor_rate_at_5"
            ),
        },
        "reliability": {
            "answer_evaluated_questions": len(judged_results),
            "evaluator_failure_rate": 1.0 - (
                len(judged_results) / len(results)
            ),
            "working_correctly_rate": (
                sum(
                    row["diagnosis"] == "working_correctly"
                    for row in results
                )
                / len(results)
            ),
            "diagnosis_counts": dict(
                Counter(row["diagnosis"] for row in results)
            ),
            "answer_outcome_counts": dict(
                Counter(row["answer_outcome"] for row in results)
            ),
        },
        "per_domain": {
            domain: summarize_group([
                row
                for row in results
                if str(row.get("domain") or "unspecified") == domain
            ])
            for domain in domains
        },
    }
    return summary


def log_summary_span(summary: dict, diagnostics: dict) -> None:
    with tracer.start_as_current_span("HKPL Noise Evaluation Summary") as span:
        set_span_io(
            span,
            "EVALUATOR",
            input_value={
                "evaluation_dataset": diagnostics["metadata"]["dataset"],
                "distractor_datasets": diagnostics["corpus"][
                    "distractor_datasets"
                ],
                "vector_table": diagnostics["corpus"]["vector_table"],
            },
            output_value=summary,
        )
        span.set_attribute("eval.dataset", diagnostics["metadata"]["dataset"])
        set_json_attribute(
            span,
            "eval.distractor_datasets",
            diagnostics["corpus"]["distractor_datasets"],
        )
        set_json_attribute(
            span,
            "eval.search_corpus_vectors",
            diagnostics["corpus"]["search_vectors"],
        )
        set_json_attribute(
            span,
            "eval.average_token_usage",
            summary["average_token_usage"],
        )
        set_json_attribute(
            span,
            "eval.per_domain",
            diagnostics["per_domain"],
        )
        span.set_attribute(
            "eval.evaluated_questions",
            int(summary["questions"]["evaluated_questions"]),
        )
        for metric, value in summary["hit_rate"].items():
            span.set_attribute(f"eval.hit_rate.{metric}", float(value))
        for metric, value in summary["average_token_usage"].items():
            span.set_attribute(f"eval.tokens.average.{metric}", float(value))


async def main() -> None:
    args = parse_args()
    os.environ["PHOENIX_PROJECT_NAME"] = args.phoenix_project
    setup_phoenix_tracing(project_name=args.phoenix_project)
    results_path, summary_path = report_paths(args)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.reasoning_budget < 1:
        raise ValueError("--reasoning-budget must be positive.")

    coverage = load_hkpl_coverage()
    if (
        coverage["valid_questions"] != coverage["intended_questions"]
        and not args.allow_incomplete_dataset
    ):
        raise RuntimeError(
            "Evaluation dataset is not fully linked to the current HKPL "
            f"corpus: {coverage['valid_questions']}/"
            f"{coverage['intended_questions']} valid. Repair or regenerate "
            "the benchmark before evaluation. Use "
            "--allow-incomplete-dataset only for targeted diagnostics."
        )
    # When retrying failures, apply --limit after selecting failed IDs so that
    # it means "first N failures", not "failures among the first N rows".
    rows = load_hkpl_rows(
        None if args.rerun_answer_failures_from is not None else args.limit
    )
    if not rows:
        raise RuntimeError("No valid HKPL evaluation rows were loaded.")
    corpus_counts = load_corpus_counts()
    missing_distractors = [
        dataset
        for dataset in ("hotpotqa", "webz_news")
        if corpus_counts.get(dataset, 0) <= 0
    ]
    if missing_distractors and not args.allow_missing_distractors:
        raise RuntimeError(
            "Full RAG evaluation requires both distractor corpora. Missing: "
            + ", ".join(missing_distractors)
            + ". Use --allow-missing-distractors only for diagnostics."
        )
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
    prior_selection = None
    if args.rerun_answer_failures_from is not None:
        prior_selection = prior_failure_selection(
            args.rerun_answer_failures_from
        )
        failed_ids = prior_selection["failed_ids"]
        rows = [row for row in rows if row["evaluation_id"] in failed_ids]
        if not rows:
            raise RuntimeError(
                "None of the failed evaluation IDs from the prior results "
                "exist in the current valid benchmark."
            )
        missing_ids = failed_ids.difference(
            row["evaluation_id"] for row in rows
        )
        if missing_ids:
            print(
                "Warning: "
                f"{len(missing_ids)} prior failed row(s) are not present in "
                "the current valid benchmark and will not be rerun."
            )
        if args.limit is not None:
            rows = rows[:args.limit]
    print(
        f"Evaluation dataset coverage: {coverage['valid_questions']}/"
        f"{coverage['intended_questions']} "
        f"({coverage['evaluation_dataset_coverage']:.2%})"
    )
    print(f"Loaded {len(rows)} HKPL evaluation rows for this run.")
    print(
        f"Retriever searches combined vector table: data_{VECTOR_TABLE} "
        "(HKPL + all configured distractor corpora)"
    )
    print(f"Search corpus vectors: {corpus_counts}")
    print(f"Phoenix project: {os.getenv('PHOENIX_PROJECT_NAME', 'hkpl-rag')}")
    print(
        "Reasoning: "
        f"answer={'on' if args.answer_reasoning else 'off'}, "
        f"evaluator={'on' if args.evaluator_reasoning else 'off'}, "
        "budget="
        f"{args.reasoning_budget if (args.answer_reasoning or args.evaluator_reasoning) else 0}"
    )

    judge = QwenEvaluationLLM(
        enable_thinking=args.evaluator_reasoning,
        thinking_budget_tokens=args.reasoning_budget,
    )
    evaluators = (
        CorrectnessEvaluator(
            llm=judge,
            eval_template=STRICT_CORRECTNESS_TEMPLATE,
        ),
        FaithfulnessEvaluator(
            llm=judge,
            eval_template=STRICT_FAITHFULNESS_TEMPLATE,
        ),
        RelevancyEvaluator(
            llm=judge,
            eval_template=STRICT_RELEVANCY_TEMPLATE,
        ),
    )
    results = []
    for position, row in enumerate(rows, start=1):
        print(
            f"[{position}/{len(rows)}] [{row['dataset']}] {row['question']}"
        )
        try:
            result = await evaluate_row(
                row,
                evaluators,
                answer_reasoning=args.answer_reasoning,
                reasoning_budget=args.reasoning_budget,
            )
        except Exception as error:
            print(f"FAILED: {error}")
            result = failed_result(row, error)
        results.append(result)
        print(
            "  "
            f"retriever_hit@10="
            f"{int(result['retrieval_reference_match_mode_at_10'] != 'missing')} "
            f"reranker_hit@5="
            f"{int(result['reranker_reference_match_mode'] != 'missing')} "
            f"pool_reference={result['retrieval_reference_match_mode']}->"
            f"{result['reranker_reference_match_mode']} "
            f"correctness={result['correctness']:.2f} "
            f"faithfulness={result['faithfulness']:.2f} "
            f"relevancy={result['relevancy']:.2f} "
            f"answer={result['answer_outcome']} "
            f"diagnosis={result['diagnosis']}"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    metrics = summarize(results)
    summary = {
        "questions": {
            "evaluated_questions": len(results),
        },
        "hit_rate": metrics["hit_rate"],
        "average_token_usage": metrics["average_token_usage"],
        "quality": {
            "answer_pass_rate": metrics["quality_guardrails"][
                "answer_pass_rate"
            ],
            "average_correctness": metrics["quality_guardrails"][
                "average_correctness"
            ],
            "average_faithfulness": metrics["quality_guardrails"][
                "average_faithfulness"
            ],
        },
        "latency_seconds": metrics["latency_seconds"],
    }
    if prior_selection is not None:
        recovered_answers = sum(
            1 for result in results if result["answer_pass"]
        )
        projected_passes = (
            prior_selection["passed_questions"] + recovered_answers
        )
        summary["failure_retry"] = {
            "baseline_evaluated_questions": prior_selection[
                "evaluated_questions"
            ],
            "baseline_passed_questions": prior_selection["passed_questions"],
            "baseline_failed_questions": prior_selection["failed_questions"],
            "retried_questions": len(results),
            "recovered_answers": recovered_answers,
            "retry_pass_rate": recovered_answers / len(results),
            "projected_combined_passed_questions": projected_passes,
            "projected_combined_answer_pass_rate": (
                projected_passes / prior_selection["evaluated_questions"]
            ),
        }
    diagnostics = {
        "metadata": {
            "dataset": "hkpl",
            "phoenix_project": os.getenv(
                "PHOENIX_PROJECT_NAME",
                "hkpl-rag",
            ),
            "answer_reasoning_enabled": args.answer_reasoning,
            "evaluator_reasoning_enabled": args.evaluator_reasoning,
            "reasoning_budget_tokens": (
                args.reasoning_budget
                if args.answer_reasoning or args.evaluator_reasoning
                else 0
            ),
            "rerun_answer_failures_from": (
                str(args.rerun_answer_failures_from)
                if args.rerun_answer_failures_from is not None
                else ""
            ),
        },
        "dataset_validation": coverage,
        "average_relevancy": metrics["quality_guardrails"][
            "average_relevancy"
        ],
        "robustness": metrics["robustness"],
        "reliability": metrics["reliability"],
        "per_domain": metrics["per_domain"],
        "corpus": {
            "vector_table": f"data_{VECTOR_TABLE}",
            "search_vectors": corpus_counts,
            "distractor_datasets": sorted(
                dataset for dataset in corpus_counts if dataset != "hkpl"
            ),
        },
    }
    diagnostics_path = summary_path.with_name(
        f"{summary_path.stem}.diagnostics{summary_path.suffix}"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_summary_span(summary, diagnostics)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved diagnostics to: {diagnostics_path}")


if __name__ == "__main__":
    asyncio.run(main())
