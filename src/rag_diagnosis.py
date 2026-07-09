from typing import Any


def get_chunk_ids(items: list[dict[str, Any]]) -> list[str]:
    return [item.get("chunk_id", "") for item in items]


def diagnose_rag(
    expected_document_id: str,
    expected_chunk_id: str,
    vector_candidates: list[dict[str, Any]],
    after_rerank: list[dict[str, Any]],
    chunks_sent_to_llm: list[str],
    correctness_score: float,
    faithfulness_score: float,
    relevancy_score: float,
) -> dict[str, Any]:
    vector_chunk_ids = get_chunk_ids(vector_candidates)
    reranked_chunk_ids = get_chunk_ids(after_rerank)

    expected_doc_in_vector = any(
        expected_document_id
        and (
            item.get("document_id") == expected_document_id
            or item.get("chunk_id", "").startswith(expected_document_id)
        )
        for item in vector_candidates
    )

    expected_chunk_in_vector = expected_chunk_id in vector_chunk_ids
    expected_chunk_after_rerank = expected_chunk_id in reranked_chunk_ids
    expected_chunk_sent_to_llm = expected_chunk_id in chunks_sent_to_llm

    if not expected_doc_in_vector:
        diagnosis = "retrieval_problem"
        recommendation = "Expected document was not retrieved by PGVector. Check embeddings, query wording, ingestion, or increase SIMILARITY_TOP_K."
    elif expected_doc_in_vector and not expected_chunk_in_vector:
        diagnosis = "chunk_level_retrieval_problem"
        recommendation = "Correct document was retrieved, but exact expected chunk was not. Check chunking, overlap, or expected_chunk_id."
    elif expected_chunk_in_vector and not expected_chunk_after_rerank:
        diagnosis = "reranker_problem"
        recommendation = "Expected chunk was retrieved but removed after reranking. Inspect reranker scores or increase RERANK_TOP_N."
    elif expected_chunk_after_rerank and not expected_chunk_sent_to_llm:
        diagnosis = "context_building_problem"
        recommendation = "Expected chunk survived reranking but was not sent to the LLM. Check context construction."
    elif expected_chunk_sent_to_llm and correctness_score < 3:
        diagnosis = "llm_generation_problem"
        recommendation = "Correct evidence reached the LLM, but answer quality was poor. Improve answer prompt."
    elif correctness_score >= 4 and (faithfulness_score < 0.5 or relevancy_score < 0.5):
        diagnosis = "evaluator_or_dataset_issue"
        recommendation = "Answer appears mostly correct but evaluator score is low. Check evaluator strictness or dataset wording."
    else:
        diagnosis = "working_correctly"
        recommendation = "Retrieval, reranking, context construction, and generation look acceptable."

    return {
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "expected_document_id": expected_document_id,
        "expected_chunk_id": expected_chunk_id,
        "expected_doc_in_vector": expected_doc_in_vector,
        "expected_chunk_in_vector": expected_chunk_in_vector,
        "expected_chunk_after_rerank": expected_chunk_after_rerank,
        "expected_chunk_sent_to_llm": expected_chunk_sent_to_llm,
        "vector_candidate_chunk_ids": vector_chunk_ids,
        "after_rerank_chunk_ids": reranked_chunk_ids,
        "chunks_sent_to_llm": chunks_sent_to_llm,
        "correctness_score": correctness_score,
        "faithfulness_score": faithfulness_score,
        "relevancy_score": relevancy_score,
    }