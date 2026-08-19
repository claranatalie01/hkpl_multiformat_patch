"""Retrieve candidate vectors and rerank them for the RAG answer context.

The incoming question is embedded with the same model used at ingestion,
PGVectorStore returns the nearest chunks, and the local reranker reorders and
filters candidates. Live retrieval restricts results to primary-corpus rows.
"""

import asyncio
import logging
import os
import re
import time
from contextvars import ContextVar
from typing import List

import aiohttp
from opentelemetry import trace
from opentelemetry.trace import format_span_id

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from sqlalchemy import text

from .corpus import (
    PRIMARY_CORPUS_ROLE,
    is_distractor_metadata,
)
from .infrastructure.embedding import embed_model
from .infrastructure.db import engine
from .infrastructure.vector_store import (
    VECTOR_TABLE,
    VECTOR_TABLE_NAME,
    ensure_hybrid_search_schema,
    vector_store,
)
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
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", os.getenv("SIMILARITY_TOP_K", "30")))
LEXICAL_TOP_K = int(os.getenv("LEXICAL_TOP_K", "30"))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))
RERANKER_TIMEOUT_SECONDS = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "120"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "Qwen3-Reranker-0.6B")

Settings.embed_model = embed_model

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

vector_retriever = index.as_retriever(similarity_top_k=DENSE_TOP_K)
live_vector_retriever = index.as_retriever(
    similarity_top_k=DENSE_TOP_K,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(
                key="corpus_role",
                value=PRIMARY_CORPUS_ROLE,
            )
        ]
    ),
)

_reranker_token_usage: ContextVar[dict] = ContextVar(
    "reranker_token_usage",
    default={"reranker_input_tokens": 0, "is_estimated": False},
)
_retrieval_trace: ContextVar[dict] = ContextVar(
    "retrieval_trace",
    default={
        "retriever_span_id": "",
        "dense_candidates": [],
        "text_candidates": [],
        "trigram_candidates": [],
        "fused_candidates": [],
        "vector_candidates_before_rerank": [],
        "final_chunks_after_rerank": [],
        "token_usage": {},
    },
)


_LIVE_FILTERS = MetadataFilters(
    filters=[
        MetadataFilter(
            key="corpus_role",
            value=PRIMARY_CORPUS_ROLE,
        )
    ]
)

_QUOTED_TERM = re.compile(
    r'"([^"\r\n]{2,})"|“([^”\r\n]{2,})”|「([^」\r\n]{2,})」|『([^』\r\n]{2,})』'
)
_IDENTIFIER_TERM = re.compile(
    r"(?<![\w-])(?=[A-Za-z0-9-]{3,}(?![\w-]))"
    r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+"
)


def _exact_terms(query: str) -> list[str]:
    """Return only high-confidence literal phrases and identifiers."""
    terms = [
        next(value for value in match.groups() if value is not None).strip()
        for match in _QUOTED_TERM.finditer(query)
    ]
    terms.extend(match.group(0) for match in _IDENTIFIER_TERM.finditer(query))
    return list(dict.fromkeys(term.casefold() for term in terms if term.strip()))


async def _text_search_candidates(
    query: str,
    *,
    include_distractors: bool,
) -> list[NodeWithScore]:
    result = await vector_store.aquery(VectorStoreQuery(
        query_str=query,
        mode=VectorStoreQueryMode.TEXT_SEARCH,
        similarity_top_k=LEXICAL_TOP_K,
        sparse_top_k=LEXICAL_TOP_K,
        filters=None if include_distractors else _LIVE_FILTERS,
    ))
    nodes = list(result.nodes or [])
    similarities = list(result.similarities or [])
    return [
        NodeWithScore(
            node=node,
            score=float(similarities[index] if index < len(similarities) else 0.0),
        )
        for index, node in enumerate(nodes)
    ]


def _trigram_candidates(
    query: str,
    *,
    include_distractors: bool,
) -> list[NodeWithScore]:
    patterns = [f"%{term}%" for term in _exact_terms(query)]
    role_predicate = (
        "TRUE"
        if include_distractors
        else "COALESCE(metadata_->>'corpus_role', 'primary') = :primary_role"
    )
    statement = text(f"""
        SELECT
            node_id,
            text,
            metadata_,
            CASE
                WHEN lower(text) LIKE ANY(CAST(:patterns AS text[])) THEN 1
                ELSE 0
            END AS exact_match,
            GREATEST(
                similarity(lower(text), lower(:query)),
                word_similarity(lower(:query), lower(text))
            ) AS trigram_score
        FROM {VECTOR_TABLE_NAME}
        WHERE {role_predicate}
          AND (
              lower(text) % lower(:query)
              OR lower(:query) <% lower(text)
              OR lower(text) LIKE ANY(CAST(:patterns AS text[]))
          )
        ORDER BY exact_match DESC, trigram_score DESC, node_id
        LIMIT :limit
    """)
    with engine.connect() as connection:
        rows = connection.execute(statement, {
            "query": query,
            "patterns": patterns,
            "primary_role": PRIMARY_CORPUS_ROLE,
            "limit": LEXICAL_TOP_K,
        }).mappings().all()
    return [
        NodeWithScore(
            node=TextNode(
                id_=str(row["node_id"]),
                text=str(row["text"]),
                metadata=dict(row["metadata_"] or {}),
            ),
            score=float(row["trigram_score"] or 0.0) + float(row["exact_match"] or 0),
        )
        for row in rows
    ]


def reciprocal_rank_fuse(
    pools: dict[str, list[NodeWithScore]],
    *,
    top_k: int = FUSION_TOP_K,
    rrf_k: int = RRF_K,
) -> list[NodeWithScore]:
    """Fuse incomparable dense, text, and trigram scores by rank."""
    entries: dict[str, dict] = {}
    for pool_name, nodes in pools.items():
        seen: set[str] = set()
        for rank, candidate in enumerate(nodes, start=1):
            node_id = str(candidate.node.node_id)
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            entry = entries.setdefault(node_id, {
                "node": candidate,
                "score": 0.0,
                "best_rank": rank,
                "scores": {},
            })
            entry["score"] += 1.0 / (rrf_k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            entry["scores"][pool_name] = {
                "rank": rank,
                "score": float(candidate.score or 0.0),
            }

    ordered = sorted(
        entries.items(),
        key=lambda item: (-item[1]["score"], item[1]["best_rank"], item[0]),
    )[:top_k]
    fused: list[NodeWithScore] = []
    for _, entry in ordered:
        candidate = entry["node"]
        candidate.score = float(entry["score"])
        candidate.node.metadata["retrieval_scores"] = {
            **entry["scores"],
            "fused": float(entry["score"]),
        }
        fused.append(candidate)
    return fused


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

    async def arerank(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        with tracer.start_as_current_span("Reranker") as span:
            before_rerank = [
                node_to_trace_dict(node, i + 1, "fused_score")
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
                _reranker_token_usage.set({
                    "reranker_input_tokens": 0,
                    "is_estimated": False,
                })
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
            _reranker_token_usage.set({
                "reranker_input_tokens": int(reranker_input_tokens),
                "is_estimated": bool(reranker_tokens_estimated),
            })

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
                                node_to_trace_dict(node, i + 1, "fallback_fused_score")
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
                    node_to_trace_dict(node, i + 1, "fallback_fused_score")
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
                    node_to_trace_dict(node, i + 1, "fallback_fused_score")
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
                    node_to_trace_dict(node, i + 1, "fallback_fused_score")
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


def get_last_retrieval_trace() -> dict:
    """Return diagnostics for the current async request context."""
    return _retrieval_trace.get()


async def retrieve_nodes(
    query: str,
    *,
    include_distractors: bool = False,
) -> List[NodeWithScore]:
    with tracer.start_as_current_span("Retriever") as span:
        retriever_span_id = format_span_id(span.get_span_context().span_id)

        set_span_io(
            span,
            "RETRIEVER",
            input_value={
                "query": query,
                "dense_top_k": DENSE_TOP_K,
                "lexical_top_k": LEXICAL_TOP_K,
                "fusion_top_k": FUSION_TOP_K,
                "rerank_top_n": RERANK_TOP_N,
                "include_distractors": include_distractors,
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
        span.set_attribute(
            "retrieval.token_count.is_estimated",
            bool(query_tokens_estimated),
        )
        span.set_attribute("retrieval.token_count.tokenizer", query_tokenizer)

        await asyncio.to_thread(ensure_hybrid_search_schema)
        dense_retriever = (
            vector_retriever
            if include_distractors
            else live_vector_retriever
        )
        dense_candidates, text_candidates, trigram_candidates = await asyncio.gather(
            dense_retriever.aretrieve(query),
            _text_search_candidates(
                query,
                include_distractors=include_distractors,
            ),
            asyncio.to_thread(
                _trigram_candidates,
                query,
                include_distractors=include_distractors,
            ),
        )
        pools = {
            "dense": list(dense_candidates),
            "text": list(text_candidates),
            "trigram": list(trigram_candidates),
        }
        filtered_distractors = 0
        if not include_distractors:
            for pool_name, nodes in pools.items():
                filtered_distractors += sum(
                    1
                    for node in nodes
                    if is_distractor_metadata(node.node.metadata)
                )
                pools[pool_name] = [
                    node
                    for node in nodes
                    if not is_distractor_metadata(node.node.metadata)
                ]

        candidates = reciprocal_rank_fuse(pools)
        retrieval_latency = time.time() - start
        dense_trace = [
            node_to_trace_dict(node, i + 1, "dense_score")
            for i, node in enumerate(pools["dense"])
        ]
        text_trace = [
            node_to_trace_dict(node, i + 1, "text_score")
            for i, node in enumerate(pools["text"])
        ]
        trigram_trace = [
            node_to_trace_dict(node, i + 1, "trigram_score")
            for i, node in enumerate(pools["trigram"])
        ]
        vector_candidates = [
            node_to_trace_dict(node, i + 1, "fused_score")
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

        span.set_attribute("retrieval.latency_seconds", round(retrieval_latency, 4))
        span.set_attribute("retrieval.candidate_count", len(candidates))
        span.set_attribute(
            "retrieval.raw_candidate_count",
            sum(len(nodes) for nodes in pools.values()),
        )
        span.set_attribute("retrieval.filtered_distractor_count", filtered_distractors)
        set_json_attribute(span, "rag.dense_candidates", dense_trace)
        set_json_attribute(span, "rag.text_candidates", text_trace)
        set_json_attribute(span, "rag.trigram_candidates", trigram_trace)
        set_json_attribute(span, "rag.fused_candidates", vector_candidates)
        # Backward-compatible name: evaluation treats this as the complete
        # pre-rerank retrieval list, which is now the fused candidate list.
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
    reranker_token_usage = _reranker_token_usage.get()

    _retrieval_trace.set({
        "retriever_span_id": retriever_span_id,
        "dense_candidates": dense_trace,
        "text_candidates": text_trace,
        "trigram_candidates": trigram_trace,
        "fused_candidates": vector_candidates,
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
    })

    return reranked

logger.info(
    "Retrieval configured: table=data_%s, dense_top_k=%s, lexical_top_k=%s, "
    "fusion_top_k=%s, rerank_top_n=%s",
    VECTOR_TABLE,
    DENSE_TOP_K,
    LEXICAL_TOP_K,
    FUSION_TOP_K,
    RERANK_TOP_N,
)
