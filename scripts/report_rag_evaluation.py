#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from statistics import mean

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.observability import setup_phoenix_tracing
from src.tracing_helpers import set_json_attribute, set_span_io

VECTOR_TABLE = os.getenv("VECTOR_TABLE", "hkpl_knowledge")
HKPL_RETRIEVAL_PATH = Path(
    os.getenv(
        "HKPL_RETRIEVAL_SUMMARY_PATH",
        "/app/data/retrieval_summary.json",
    )
)
HKPL_ANSWER_PATH = Path(
    os.getenv(
        "HKPL_ANSWER_SUMMARY_PATH",
        "/app/data/generation_summary.json",
    )
)
HOTPOTQA_PATH = Path(
    os.getenv(
        "HOTPOTQA_SUMMARY_PATH",
        "/app/data/hotpotqa/summary.json",
    )
)
OUTPUT_PATH = Path(
    os.getenv(
        "COMBINED_EVALUATION_SUMMARY_PATH",
        "/app/data/combined_evaluation_summary.json",
    )
)


def load_summary(path: Path, name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {name} summary: {path}. Run its evaluation first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def macro(*values: float) -> float:
    return mean(float(value) for value in values)


def working_rate(summary: dict) -> float:
    total = int(summary.get("total_questions", 0))
    if total == 0:
        return 0.0
    return float(summary.get("diagnosis_counts", {}).get("working_correctly", 0)) / total


def weighted_average(
    left_value: float,
    left_count: int,
    right_value: float,
    right_count: int,
) -> float:
    total = left_count + right_count
    if total == 0:
        return 0.0
    return (
        float(left_value) * left_count + float(right_value) * right_count
    ) / total


def main() -> None:
    hkpl_retrieval = load_summary(HKPL_RETRIEVAL_PATH, "HKPL retrieval")
    hkpl_answers = load_summary(HKPL_ANSWER_PATH, "HKPL answer")
    hotpotqa = load_summary(HOTPOTQA_PATH, "HotpotQA")

    if not hotpotqa.get("answer_generation_enabled", False):
        raise ValueError(
            "HotpotQA summary does not include answer generation. "
            "Run hotpotqa_benchmark.py evaluate --answers first."
        )

    hkpl_count = int(hkpl_answers["total_questions"])
    hkpl_retrieval_count = int(hkpl_retrieval["total_questions"])
    hotpot_count = int(hotpotqa["total_questions"])
    macro_metrics = {
        "evidence_recall_at_5": macro(
            hkpl_retrieval["recall_at_5"],
            hotpotqa["average_vector_recall_at_5"],
        ),
        "vector_mrr": macro(
            hkpl_retrieval["mrr"],
            hotpotqa["average_vector_mrr"],
        ),
        "normalized_answer_quality": macro(
            float(hkpl_answers["average_correctness"]) / 5.0,
            hotpotqa["average_answer_f1"],
        ),
        "working_correctly_rate": macro(
            working_rate(hkpl_answers),
            working_rate(hotpotqa),
        ),
        "average_latency_seconds": macro(
            hkpl_answers["average_latency_seconds"],
            hotpotqa["average_latency_seconds"],
        ),
        "average_pipeline_total_tokens": macro(
            hkpl_answers["average_pipeline_total_tokens"],
            hotpotqa["average_pipeline_total_tokens"],
        ),
    }
    weighted_metrics = {
        "evidence_recall_at_5": weighted_average(
            hkpl_retrieval["recall_at_5"],
            hkpl_retrieval_count,
            hotpotqa["average_vector_recall_at_5"],
            hotpot_count,
        ),
        "vector_mrr": weighted_average(
            hkpl_retrieval["mrr"],
            hkpl_retrieval_count,
            hotpotqa["average_vector_mrr"],
            hotpot_count,
        ),
        "normalized_answer_quality": weighted_average(
            float(hkpl_answers["average_correctness"]) / 5.0,
            hkpl_count,
            hotpotqa["average_answer_f1"],
            hotpot_count,
        ),
        "working_correctly_rate": weighted_average(
            working_rate(hkpl_answers),
            hkpl_count,
            working_rate(hotpotqa),
            hotpot_count,
        ),
        "average_latency_seconds": weighted_average(
            hkpl_answers["average_latency_seconds"],
            hkpl_count,
            hotpotqa["average_latency_seconds"],
            hotpot_count,
        ),
        "average_pipeline_total_tokens": weighted_average(
            hkpl_answers["average_pipeline_total_tokens"],
            hkpl_count,
            hotpotqa["average_pipeline_total_tokens"],
            hotpot_count,
        ),
    }
    summary = {
        "phoenix_project": os.getenv("PHOENIX_PROJECT_NAME", "hkpl-rag"),
        "vector_table": f"data_{VECTOR_TABLE}",
        "total_answer_questions": hkpl_count + hotpot_count,
        "datasets": {
            "hkpl": {
                "retrieval": hkpl_retrieval,
                "answers": hkpl_answers,
            },
            "hotpotqa": hotpotqa,
        },
        "macro_average": macro_metrics,
        "question_weighted_average": weighted_metrics,
        "notes": {
            "macro_average": (
                "Each dataset has equal weight, regardless of question count."
            ),
            "question_weighted_average": (
                "Each question has equal weight, so the larger HotpotQA set "
                "has more influence."
            ),
            "normalized_answer_quality": (
                "Mean of HKPL correctness divided by 5 and HotpotQA token F1."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    setup_phoenix_tracing()
    tracer = trace.get_tracer("combined-rag-evaluation")
    with tracer.start_as_current_span("Combined RAG Evaluation Summary") as span:
        set_span_io(
            span,
            "EVALUATOR",
            input_value={
                "datasets": ["hkpl", "hotpotqa"],
                "vector_table": f"data_{VECTOR_TABLE}",
            },
            output_value=summary,
        )
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "EVALUATOR")
        span.set_attribute("eval.dataset", "combined")
        span.set_attribute("eval.total_questions", hkpl_count + hotpot_count)
        for name, value in macro_metrics.items():
            span.set_attribute(f"eval.macro.{name}", float(value))
        for name, value in weighted_metrics.items():
            span.set_attribute(f"eval.weighted.{name}", float(value))
        set_json_attribute(span, "eval.datasets", ["hkpl", "hotpotqa"])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved combined summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
