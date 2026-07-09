import asyncio
import os
from typing import List, Sequence

import aiohttp
import requests
from dotenv import load_dotenv
from pydantic import Field, PrivateAttr

from llama_index.core.base.embeddings.base import BaseEmbedding


load_dotenv()


class LlamaCppEmbedding(BaseEmbedding):
    """OpenAI-compatible embedding client for the local llama.cpp server."""

    embedding_url: str = Field(description="llama.cpp embedding endpoint")
    request_timeout: float = Field(default=120.0)

    _sync_session: requests.Session = PrivateAttr(default_factory=requests.Session)

    def __init__(
        self,
        embedding_url: str,
        request_timeout: float = 120.0,
        **kwargs,
    ):
        super().__init__(
            embedding_url=embedding_url,
            request_timeout=request_timeout,
            **kwargs,
        )

    @staticmethod
    def _parse_embeddings(payload: dict) -> list[list[float]]:
        items = payload.get("data", [])
        if not items:
            raise RuntimeError("Embedding service returned no vectors.")

        items = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def _request_sync(self, inputs: Sequence[str]) -> list[list[float]]:
        response = self._sync_session.post(
            self.embedding_url,
            json={"input": list(inputs)},
            timeout=self.request_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding service error {response.status_code}: "
                f"{response.text}"
            )
        return self._parse_embeddings(response.json())

    async def _request_async(
        self,
        inputs: Sequence[str],
    ) -> list[list[float]]:
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.embedding_url,
                json={"input": list(inputs)},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Embedding service error {response.status}: "
                        f"{await response.text()}"
                    )
                return self._parse_embeddings(await response.json())

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._request_sync([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._request_sync([text])[0]

    def _get_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        if not texts:
            return []
        return self._request_sync(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._request_async([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._request_async([text]))[0]

    async def _aget_text_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        if not texts:
            return []
        return await self._request_async(texts)


EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://embedding:8080/v1/embeddings",
)

embed_model = LlamaCppEmbedding(
    embedding_url=EMBEDDING_URL,
    request_timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120")),
)
