import logging
import os
from typing import List

import aiohttp

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .infrastructure.embedding import embed_model
from .infrastructure.vector_store import VECTOR_TABLE, vector_store


logger = logging.getLogger(__name__)

RERANKER_URL = os.getenv(
    "RERANKER_URL",
    "http://reranker:8080/reranking",
)
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
RERANKER_TIMEOUT_SECONDS = float(
    os.getenv("RERANKER_TIMEOUT_SECONDS", "120")
)

Settings.embed_model = embed_model

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
)

vector_retriever = index.as_retriever(
    similarity_top_k=SIMILARITY_TOP_K,
)


class HTTPReranker:
    """Explicit retrieve-then-rerank client for llama.cpp."""

    def __init__(
        self,
        reranker_url: str,
        top_n: int = 3,
    ):
        self.reranker_url = reranker_url
        self.top_n = top_n

    async def arerank(
        self,
        nodes: List[NodeWithScore],
        query: str,
    ) -> List[NodeWithScore]:
        if not nodes:
            return []

        documents = [node.node.get_content() for node in nodes]
        timeout = aiohttp.ClientTimeout(
            total=RERANKER_TIMEOUT_SECONDS
        )

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
                        logger.warning(
                            "Reranker returned HTTP %s; using vector order.",
                            response.status,
                        )
                        return nodes[: self.top_n]

                    payload = await response.json()
        except Exception:
            logger.exception(
                "Reranker request failed; using vector order."
            )
            return nodes[: self.top_n]

        results = (
            payload.get("results", [])
            if isinstance(payload, dict)
            else payload
        )

        if not isinstance(results, list) or not results:
            return nodes[: self.top_n]

        ranked: list[NodeWithScore] = []

        for position, item in enumerate(results):
            if not isinstance(item, dict):
                continue

            raw_index = item.get("index", position)
            try:
                candidate_index = int(raw_index)
            except (TypeError, ValueError):
                continue

            if not 0 <= candidate_index < len(nodes):
                continue

            score = item.get(
                "relevance_score",
                item.get("score", 0.0),
            )

            candidate = nodes[candidate_index]
            candidate.score = float(score or 0.0)
            ranked.append(candidate)

        if not ranked:
            return nodes[: self.top_n]

        ranked.sort(
            key=lambda node: node.score or 0.0,
            reverse=True,
        )
        return ranked[: self.top_n]


reranker = HTTPReranker(
    reranker_url=RERANKER_URL,
    top_n=RERANK_TOP_N,
)


async def retrieve_nodes(query: str) -> List[NodeWithScore]:
    candidates = await vector_retriever.aretrieve(query)
    return await reranker.arerank(candidates, query)


logger.info(
    "Retrieval configured: table=data_%s, vector_top_k=%s, rerank_top_n=%s",
    VECTOR_TABLE,
    SIMILARITY_TOP_K,
    RERANK_TOP_N,
)
