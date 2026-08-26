"""Verify hybrid-retrieval configuration and reciprocal-rank fusion behavior.

These focused unit tests protect the vector store's lexical-search settings,
exact-term extraction, stable candidate deduplication, and cross-pool ranking
used by ``src.retrieval`` before reranking and context construction.
"""

from __future__ import annotations

import unittest

from llama_index.core.schema import NodeWithScore, TextNode

from src.infrastructure.vector_store import vector_store
from src.retrieval import _exact_terms, reciprocal_rank_fuse


def candidate(node_id: str, score: float = 0.0) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            id_=node_id,
            text=f"Evidence for {node_id}",
            metadata={"chunk_id": node_id},
        ),
        score=score,
    )


class HybridRetrievalTests(unittest.TestCase):
    def test_vector_store_has_indexed_text_search_enabled(self) -> None:
        self.assertTrue(vector_store.hybrid_search)
        self.assertEqual(vector_store.text_search_config, "english")

    def test_literal_term_extraction_is_conservative(self) -> None:
        self.assertEqual(
            _exact_terms('Loans for "The Hobbit" with BIB ID 3239705'),
            ["the hobbit", "3239705"],
        )
        self.assertEqual(_exact_terms("ordinary semantic question"), [])

    def test_rrf_rewards_cross_pool_evidence_and_deduplicates(self) -> None:
        fused = reciprocal_rank_fuse({
            "dense": [candidate("dense-only", 0.9), candidate("shared", 0.8)],
            "text": [candidate("shared", 0.7), candidate("text-only", 0.6)],
            "trigram": [candidate("shared", 0.95), candidate("shared", 0.5)],
        })

        self.assertEqual([node.node_id for node in fused], [
            "shared",
            "dense-only",
            "text-only",
        ])
        scores = fused[0].node.metadata["retrieval_scores"]
        self.assertEqual(set(scores), {"dense", "text", "trigram", "fused"})

    def test_rrf_uses_stable_node_id_tie_break(self) -> None:
        fused = reciprocal_rank_fuse({
            "dense": [candidate("b")],
            "text": [candidate("a")],
        })
        self.assertEqual([node.node_id for node in fused], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
