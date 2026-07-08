import json
import logging
import os
import time
from typing import List

import aiohttp
from opentelemetry import trace

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .infrastructure.embedding import embed_model
from .infrastructure.vector_store import VECTOR_TABLE, vector_store
from src.tracing_helpers import set_span_io


logger = logging.getLogger(__name__)
tracer = trace.get_tracer("hkpl-retrieval")

RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8080/reranking")
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANKER_TIMEOUT_SECONDS = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "120"))

Settings.embed_model = embed_model

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

vector_retriever = index.as_retriever(
    similarity_top_k=SIMILARITY_TOP_K,
)


def node_to_trace_dict(
    node: NodeWithScore,
    rank: int,
    score_name: str,
) -> dict:
    metadata = node.node.metadata or {}
    chunk_id = metadata.get("chunk_id", "")

    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "document_id": (
            metadata.get("kb_document_id")
            or metadata.get("document_id")
            or chunk_id.split(":")[0]
            or ""
        ),
        "source_title": metadata.get("source_title", ""),
        "source_url": metadata.get("source_url") or metadata.get("url", ""),
        score_name: float(node.score or 0.0),
        "text_preview": node.node.get_content()[:500],
    }


class HTTPReranker:
    def __init__(self, reranker_url: str, top_n: int = 3):
        self.reranker_url = reranker_url
        self.top_n = top_n

    async def arerank(
        self,
        nodes: List[NodeWithScore],
        query: str,
    ) -> List[NodeWithScore]:
        with tracer.start_as_current_span("reranker") as span:
            before_rerank = [
                node_to_trace_dict(
                    node=node,
                    rank=index + 1,
                    score_name="vector_score",
                )
                for index, node in enumerate(nodes)
            ]

            set_span_io(
                span,
                "RERANKER",
                input_value={
                    "query": query,
                    "top_n": self.top_n,
                    "candidate_count": len(nodes),
                    "before_rerank": before_rerank,
                },
            )

            span.set_attribute(
                "rag.before_rerank",
                json.dumps(before_rerank, ensure_ascii=False),
            )

            if not nodes:
                set_span_io(
                    span,
                    "RERANKER",
                    output_value={
                        "after_rerank": [],
                        "reason": "no_candidates",
                    },
                )
                return []

            documents = [
                node.node.get_content()
                for node in nodes
            ]

            span.set_attribute(
                "reranker.input_doc_lengths",
                json.dumps([len(doc) for doc in documents]),
            )

            timeout = aiohttp.ClientTimeout(total=RERANKER_TIMEOUT_SECONDS)
            start = time.time()

            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.reranker_url,
                        json={
                            "query": query,
                            "documents": documents,
                        },
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            logger.error(
                                "Reranker error HTTP %s: %s",
                                response.status,
                                body,
                            )

                            fallback = nodes[: self.top_n]

                            after_rerank = [
                                node_to_trace_dict(
                                    node=node,
                                    rank=index + 1,
                                    score_name="fallback_vector_score",
                                )
                                for index, node in enumerate(fallback)
                            ]

                            span.set_attribute("reranker.failed", True)
                            span.set_attribute("reranker.http_status", response.status)
                            span.set_attribute("reranker.error_body", body)
                            span.set_attribute(
                                "rag.after_rerank",
                                json.dumps(after_rerank, ensure_ascii=False),
                            )

                            set_span_io(
                                span,
                                "RERANKER",
                                output_value={
                                    "failed": True,
                                    "fallback": "vector_order",
                                    "after_rerank": after_rerank,
                                },
                            )

                            return fallback

                        payload = await response.json()

            except Exception as error:
                logger.exception("Reranker request failed; using vector order.")

                fallback = nodes[: self.top_n]

                after_rerank = [
                    node_to_trace_dict(
                        node=node,
                        rank=index + 1,
                        score_name="fallback_vector_score",
                    )
                    for index, node in enumerate(fallback)
                ]

                span.record_exception(error)
                span.set_attribute("reranker.failed", True)
                span.set_attribute(
                    "rag.after_rerank",
                    json.dumps(after_rerank, ensure_ascii=False),
                )

                set_span_io(
                    span,
                    "RERANKER",
                    output_value={
                        "failed": True,
                        "fallback": "vector_order",
                        "error": str(error),
                        "after_rerank": after_rerank,
                    },
                )

                return fallback

            span.set_attribute(
                "reranker.latency_seconds",
                round(time.time() - start, 4),
            )

            results = payload.get("results", []) if isinstance(payload, dict) else payload

            if not isinstance(results, list) or not results:
                fallback = nodes[: self.top_n]

                after_rerank = [
                    node_to_trace_dict(
                        node=node,
                        rank=index + 1,
                        score_name="fallback_vector_score",
                    )
                    for index, node in enumerate(fallback)
                ]

                span.set_attribute(
                    "rag.after_rerank",
                    json.dumps(after_rerank, ensure_ascii=False),
                )

                set_span_io(
                    span,
                    "RERANKER",
                    output_value={
                        "failed": True,
                        "fallback": "empty_reranker_result",
                        "after_rerank": after_rerank,
                    },
                )

                return fallback

            ranked: list[NodeWithScore] = []

            for position, item in enumerate(results):
                if not isinstance(item, dict):
                    continue

                try:
                    candidate_index = int(item.get("index", position))
                except (TypeError, ValueError):
                    continue

                if not 0 <= candidate_index < len(nodes):
                    continue

                score = item.get("relevance_score", item.get("score", 0.0))

                candidate = nodes[candidate_index]
                candidate.score = float(score or 0.0)
                ranked.append(candidate)

            if not ranked:
                fallback = nodes[: self.top_n]

                after_rerank = [
                    node_to_trace_dict(
                        node=node,
                        rank=index + 1,
                        score_name="fallback_vector_score",
                    )
                    for index, node in enumerate(fallback)
                ]

                span.set_attribute(
                    "rag.after_rerank",
                    json.dumps(after_rerank, ensure_ascii=False),
                )

                set_span_io(
                    span,
                    "RERANKER",
                    output_value={
                        "failed": True,
                        "fallback": "invalid_reranker_result",
                        "after_rerank": after_rerank,
                    },
                )

                return fallback

            ranked.sort(key=lambda node: node.score or 0.0, reverse=True)
            ranked = ranked[: self.top_n]

            after_rerank = [
                node_to_trace_dict(
                    node=node,
                    rank=index + 1,
                    score_name="rerank_score",
                )
                for index, node in enumerate(ranked)
            ]

            span.set_attribute(
                "rag.after_rerank",
                json.dumps(after_rerank, ensure_ascii=False),
            )

            set_span_io(
                span,
                "RERANKER",
                output_value={
                    "after_rerank": after_rerank,
                },
            )

            return ranked


reranker = HTTPReranker(
    reranker_url=RERANKER_URL,
    top_n=RERANK_TOP_N,
)


async def retrieve_nodes(query: str) -> List[NodeWithScore]:
    with tracer.start_as_current_span("retrieve_nodes") as span:
        set_span_io(
            span,
            "RETRIEVER",
            input_value={
                "query": query,
                "similarity_top_k": SIMILARITY_TOP_K,
                "rerank_top_n": RERANK_TOP_N,
            },
        )

        start = time.time()
        candidates = await vector_retriever.aretrieve(query)
        vector_latency = time.time() - start

        vector_candidates = [
            node_to_trace_dict(
                node=node,
                rank=index + 1,
                score_name="vector_score",
            )
            for index, node in enumerate(candidates)
        ]

        span.set_attribute(
            "rag.vector_candidates_before_rerank",
            json.dumps(vector_candidates, ensure_ascii=False),
        )
        span.set_attribute(
            "retrieval.vector_latency_seconds",
            round(vector_latency, 4),
        )
        span.set_attribute("retrieval.candidate_count", len(candidates))

        reranked = await reranker.arerank(candidates, query)

        final_chunks = [
            node_to_trace_dict(
                node=node,
                rank=index + 1,
                score_name="final_score",
            )
            for index, node in enumerate(reranked)
        ]

        span.set_attribute(
            "rag.chunks_after_rerank",
            json.dumps(final_chunks, ensure_ascii=False),
        )

        set_span_io(
            span,
            "RETRIEVER",
            output_value={
                "PGVector retrieved before rerank": vector_candidates,
                "Final chunks after rerank": final_chunks,
            },
        )

        return reranked


logger.info(
    "Retrieval configured: table=data_%s, vector_top_k=%s, rerank_top_n=%s",
    VECTOR_TABLE,
    SIMILARITY_TOP_K,
    RERANK_TOP_N,
)