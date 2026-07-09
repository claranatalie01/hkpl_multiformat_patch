import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def phoenix_annotations_enabled() -> bool:
    return os.getenv("PHOENIX_LOG_EVAL_ANNOTATIONS", "true").lower() == "true"


def phoenix_base_url() -> str:
    return os.getenv("PHOENIX_BASE_URL", "http://phoenix:6006")


def normalize_score(value: float, maximum: float = 1.0) -> float:
    if maximum <= 0:
        return 0.0
    score = float(value or 0.0) / maximum
    return max(0.0, min(1.0, score))


def document_is_relevant(
    document: dict[str, Any],
    expected_document_id: str,
    expected_chunk_id: str,
) -> bool:
    chunk_id = document.get("chunk_id", "")
    document_id = document.get("document_id", "")

    if expected_chunk_id and chunk_id == expected_chunk_id:
        return True

    if expected_document_id and (
        document_id == expected_document_id
        or chunk_id.startswith(expected_document_id)
    ):
        return True

    return False


def log_span_annotations(span_id: str, annotations: list[dict[str, Any]]) -> None:
    if not phoenix_annotations_enabled() or not span_id:
        return

    try:
        import pandas as pd
        from phoenix.client import Client

        rows = []
        for annotation in annotations:
            rows.append(
                {
                    "span_id": span_id,
                    "annotation_name": annotation["name"],
                    "annotator_kind": annotation.get("annotator_kind", "LLM"),
                    "label": annotation.get("label"),
                    "score": annotation.get("score"),
                    "explanation": annotation.get("explanation"),
                    "metadata": annotation.get("metadata"),
                }
            )

        Client(base_url=phoenix_base_url()).spans.log_span_annotations_dataframe(
            dataframe=pd.DataFrame(rows),
        )

    except Exception:
        logger.exception("Failed to log Phoenix span annotations.")


def log_document_relevance_annotations(
    *,
    retriever_span_id: str,
    retrieved_documents: list[dict[str, Any]],
    expected_document_id: str,
    expected_chunk_id: str,
) -> None:
    if not phoenix_annotations_enabled() or not retriever_span_id or not retrieved_documents:
        return

    try:
        import pandas as pd
        from phoenix.client import Client

        rows = []

        for position, document in enumerate(retrieved_documents):
            relevant = document_is_relevant(
                document,
                expected_document_id=expected_document_id,
                expected_chunk_id=expected_chunk_id,
            )
            expected_label = (
                "expected chunk"
                if expected_chunk_id and document.get("chunk_id") == expected_chunk_id
                else "expected document"
                if relevant
                else "not expected"
            )

            rows.append(
                {
                    "span_id": retriever_span_id,
                    "document_position": position,
                    "label": "relevant" if relevant else "irrelevant",
                    "score": 1.0 if relevant else 0.0,
                    "explanation": (
                        f"{expected_label}; "
                        f"document_id={document.get('document_id', '')}; "
                        f"chunk_id={document.get('chunk_id', '')}"
                    ),
                }
            )

        Client(base_url=phoenix_base_url()).spans.log_document_annotations_dataframe(
            dataframe=pd.DataFrame(rows),
            annotation_name="Relevance",
            annotator_kind="CODE",
        )

    except Exception:
        logger.exception("Failed to log Phoenix document relevance annotations.")


def log_rag_answer_annotations(
    *,
    root_span_id: str,
    correctness_score: float,
    correctness_reason: str,
    faithfulness_score: float,
    faithfulness_reason: str,
    relevancy_score: float,
    relevancy_reason: str,
    diagnosis: str,
    recommendation: str,
) -> None:
    correctness = normalize_score(correctness_score, maximum=5.0)
    faithfulness = normalize_score(faithfulness_score)
    relevancy = normalize_score(relevancy_score)
    hallucination = 1.0 - faithfulness

    log_span_annotations(
        root_span_id,
        [
            {
                "name": "Q&A Correctness",
                "annotator_kind": "LLM",
                "label": "correct" if correctness >= 0.8 else "incorrect",
                "score": correctness,
                "explanation": correctness_reason,
            },
            {
                "name": "Faithfulness",
                "annotator_kind": "LLM",
                "label": "faithful" if faithfulness >= 0.5 else "unfaithful",
                "score": faithfulness,
                "explanation": faithfulness_reason,
            },
            {
                "name": "Relevancy",
                "annotator_kind": "LLM",
                "label": "relevant" if relevancy >= 0.5 else "irrelevant",
                "score": relevancy,
                "explanation": relevancy_reason,
            },
            {
                "name": "Hallucination",
                "annotator_kind": "CODE",
                "label": "hallucinated" if hallucination > 0.5 else "grounded",
                "score": hallucination,
                "explanation": faithfulness_reason,
            },
            {
                "name": "RAG Diagnosis",
                "annotator_kind": "CODE",
                "label": diagnosis,
                "score": 1.0 if diagnosis == "working_correctly" else 0.0,
                "explanation": recommendation,
            },
        ],
    )
