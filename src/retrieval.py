import json
import logging
import os
import time
from typing import List

import aiohttp
from opentelemetry import trace
from opentelemetry.trace import format_span_id

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .infrastructure.embedding import embed_model
from .infrastructure.vector_store import VECTOR_TABLE, vector_store
from src.tracing_helpers import (
    set_document_list_attributes,
    set_json_attribute,
    set_span_io,
)
from src.token_counting import (
    EMBEDDING_TOKENIZER_NAME,
    EMBEDDING_TOKENIZER_URL,
    RERANKER_TOKENIZER_NAME,
    RERANKER_TOKENIZER_URL,
    count_many_tokens,
    count_tokens,
)


logger = logging.getLogger(__name__)
tracer = trace.get_tracer("hkpl-retrieval")

RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8080/reranking")
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANKER_TIMEOUT_SECONDS = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "120"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "Qwen3-Reranker-0.6B")

Settings.embed_model = embed_model

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

vector_retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)


def node_to_trace_dict(
    node: NodeWithScore,
    rank: int,
    score_name: str,
) -> dict:

    metadata = node.node.metadata or {}

    chunk_id = metadata.get("chunk_id", "")

    return {

        "rank": rank,

        "document_id":
            metadata.get("kb_document_id")
            or metadata.get("document_id")
            or chunk_id.split(":")[0]
            or "",

        "chunk_id": chunk_id,

        "title":
            metadata.get("source_title")
            or metadata.get("file_name")
            or "HKPL",

        "url":
            metadata.get("source_url")
            or metadata.get("url")
            or "",

        "page":
            metadata.get("page_number"),

        "section":
            metadata.get("section_heading"),

        "score":
            float(node.score or 0.0),

        "score_name": score_name,

        "text":
            node.node.get_content(),

        "text_preview":
            node.node.get_content()[:700],

        "metadata": metadata,
    }


class HTTPReranker:
    def __init__(self, reranker_url: str, top_n: int = 3):
        self.reranker_url = reranker_url
        self.top_n = top_n
        self.last_token_usage = {
            "reranker_input_tokens": 0,
            "is_estimated": False,
        }

    async def arerank(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        with tracer.start_as_current_span("Reranker") as span:
            before_rerank = [
                node_to_trace_dict(node, i + 1, "vector_score")
                for i, node in enumerate(nodes)
            ]

            set_span_io(
                span,
                "RERANKER",
                input_value={
                    "query": query,
                    "candidate_count": len(nodes),
                    "top_n": self.top_n,
                    "before_rerank": before_rerank,
                },
            )
            set_json_attribute(span, "rag.before_rerank", before_rerank)
            set_document_list_attributes(span, "reranker.input_documents", before_rerank)
            span.set_attribute("reranker.query", query)
            span.set_attribute("reranker.top_k", int(self.top_n))
            span.set_attribute("reranker.model_name", RERANKER_MODEL_NAME)
            span.set_attribute("reranker.input_document_count", len(before_rerank))

            if not nodes:
                self.last_token_usage = {
                    "reranker_input_tokens": 0,
                    "is_estimated": False,
                }
                set_span_io(span, "RERANKER", output_value={"after_rerank": []})
                set_document_list_attributes(span, "reranker.output_documents", [])
                return []

            documents = [node.node.get_content() for node in nodes]
            set_json_attribute(span, "reranker.input_doc_lengths", [len(doc) for doc in documents])
            reranker_pair_texts = [f"{query}\n\n{document}" for document in documents]
            reranker_input_tokens, reranker_tokens_estimated, reranker_tokenizer = await count_many_tokens(
                reranker_pair_texts,
                RERANKER_TOKENIZER_URL,
                RERANKER_TOKENIZER_NAME,
            )
            span.set_attribute("reranker.token_count.input", int(reranker_input_tokens))
            span.set_attribute("reranker.token_count.total", int(reranker_input_tokens))
            span.set_attribute("reranker.token_count.is_estimated", bool(reranker_tokens_estimated))
            span.set_attribute("reranker.token_count.tokenizer", reranker_tokenizer)
            self.last_token_usage = {
                "reranker_input_tokens": int(reranker_input_tokens),
                "is_estimated": bool(reranker_tokens_estimated),
            }

            timeout = aiohttp.ClientTimeout(total=RERANKER_TIMEOUT_SECONDS)
            start = time.time()

            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.reranker_url,
                        json={"query": query, "documents": documents},
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            fallback = nodes[: self.top_n]
                            after_rerank = [
                                node_to_trace_dict(node, i + 1, "fallback_vector_score")
                                for i, node in enumerate(fallback)
                            ]
                            span.set_attribute(
                                "reranker.input_document_count",
                                len(before_rerank),
                            )

                            span.set_attribute(
                                "reranker.output_document_count",
                                len(after_rerank),
                            )

                            span.set_attribute("reranker.failed", True)
                            span.set_attribute("reranker.http_status", response.status)
                            span.set_attribute("reranker.error_body", body)
                            set_json_attribute(span, "rag.after_rerank", after_rerank)
                            set_document_list_attributes(
                                span,
                                "reranker.output_documents",
                                after_rerank,
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
                fallback = nodes[: self.top_n]
                after_rerank = [
                    node_to_trace_dict(node, i + 1, "fallback_vector_score")
                    for i, node in enumerate(fallback)
                ]

                span.record_exception(error)
                span.set_attribute("reranker.failed", True)
                set_json_attribute(span, "rag.after_rerank", after_rerank)
                set_document_list_attributes(span, "reranker.output_documents", after_rerank)

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

            span.set_attribute("reranker.latency_seconds", round(time.time() - start, 4))

            results = payload.get("results", []) if isinstance(payload, dict) else payload

            if not isinstance(results, list) or not results:
                fallback = nodes[: self.top_n]
                after_rerank = [
                    node_to_trace_dict(node, i + 1, "fallback_vector_score")
                    for i, node in enumerate(fallback)
                ]

                set_json_attribute(span, "rag.after_rerank", after_rerank)
                set_document_list_attributes(span, "reranker.output_documents", after_rerank)
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
                    node_to_trace_dict(node, i + 1, "fallback_vector_score")
                    for i, node in enumerate(fallback)
                ]

                set_json_attribute(span, "rag.after_rerank", after_rerank)
                set_document_list_attributes(span, "reranker.output_documents", after_rerank)
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
                node_to_trace_dict(node, i + 1, "rerank_score")
                for i, node in enumerate(ranked)
            ]

            set_json_attribute(span, "rag.after_rerank", after_rerank)
            set_document_list_attributes(span, "reranker.output_documents", after_rerank)
            span.set_attribute("reranker.output_document_count", len(after_rerank))
            set_span_io(span, "RERANKER", output_value={"after_rerank": after_rerank})

            return ranked


reranker = HTTPReranker(reranker_url=RERANKER_URL, top_n=RERANK_TOP_N)


async def retrieve_nodes(query: str) -> List[NodeWithScore]:
    with tracer.start_as_current_span("Retriever") as span:
        retriever_span_id = format_span_id(span.get_span_context().span_id)

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
        query_tokens, query_tokens_estimated, query_tokenizer = await count_tokens(
            query,
            EMBEDDING_TOKENIZER_URL,
            EMBEDDING_TOKENIZER_NAME,
        )
        span.set_attribute("retrieval.token_count.query", int(query_tokens))
        span.set_attribute("retrieval.token_count.total", int(query_tokens))
        span.set_attribute("retrieval.token_count.is_estimated", bool(query_tokens_estimated))
        span.set_attribute("retrieval.token_count.tokenizer", query_tokenizer)

        candidates = await vector_retriever.aretrieve(query)
        vector_latency = time.time() - start

        vector_candidates = [
            node_to_trace_dict(node, i + 1, "vector_score")
            for i, node in enumerate(candidates)
        ]
        span.set_attribute(
            "retrieval.top_k",
            len(vector_candidates),
        )

        span.set_attribute(
            "retrieval.query",
            query,
        )
        span.set_attribute("retrieval.span_id", retriever_span_id)

        span.set_attribute("retrieval.vector_latency_seconds", round(vector_latency, 4))
        span.set_attribute("retrieval.candidate_count", len(candidates))
        set_json_attribute(span, "rag.vector_candidates_before_rerank", vector_candidates)
        set_document_list_attributes(span, "retrieval.documents", vector_candidates)
        set_span_io(
            span,
            "RETRIEVER",
            input_value=query,
            output_value={
                "documents": vector_candidates,
            },
        )

    reranked = await reranker.arerank(candidates, query)

    final_chunks = [
        node_to_trace_dict(node, i + 1, "final_score")
        for i, node in enumerate(reranked)
    ]
    reranker_token_usage = getattr(reranker, "last_token_usage", {})

    retrieve_nodes.last_trace = {
        "retriever_span_id": retriever_span_id,
        "vector_candidates_before_rerank": vector_candidates,
        "final_chunks_after_rerank": final_chunks,
        "token_usage": {
            "retriever_query_tokens": int(query_tokens),
            "reranker_input_tokens": int(
                reranker_token_usage.get("reranker_input_tokens", 0)
            ),
            "is_estimated": bool(query_tokens_estimated)
            or bool(reranker_token_usage.get("is_estimated", False)),
        },
    }

    return reranked


retrieve_nodes.last_trace = {
    "retriever_span_id": "",
    "vector_candidates_before_rerank": [],
    "final_chunks_after_rerank": [],
}

logger.info(
    "Retrieval configured: table=data_%s, vector_top_k=%s, rerank_top_n=%s",
    VECTOR_TABLE,
    SIMILARITY_TOP_K,
    RERANK_TOP_N,
)
