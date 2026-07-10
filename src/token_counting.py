from __future__ import annotations

import asyncio
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)

LLM_TOKENIZER_URL = os.getenv("LLM_TOKENIZER_URL", "http://llm:8080/tokenize")
EMBEDDING_TOKENIZER_URL = os.getenv("EMBEDDING_TOKENIZER_URL", "http://embedding:8080/tokenize")
RERANKER_TOKENIZER_URL = os.getenv("RERANKER_TOKENIZER_URL", "http://reranker:8080/tokenize")

LLM_TOKENIZER_NAME = os.getenv("LLM_TOKENIZER_NAME", "llama.cpp:qwen3.5-9b")
EMBEDDING_TOKENIZER_NAME = os.getenv("EMBEDDING_TOKENIZER_NAME", "llama.cpp:qwen3-embedding-0.6b")
RERANKER_TOKENIZER_NAME = os.getenv("RERANKER_TOKENIZER_NAME", "llama.cpp:qwen3-reranker-0.6b")

TOKENIZER_TIMEOUT_SECONDS = float(os.getenv("TOKENIZER_TIMEOUT_SECONDS", "30"))


def estimate_token_count(text: str) -> int:
    if not text:
        return 0

    # Conservative fallback for local tokenizer endpoint failures.
    return max(1, round(len(text) / 4))


async def count_tokens(
    text: str,
    tokenizer_url: str | None = None,
    tokenizer_name: str | None = None,
) -> tuple[int, bool, str]:
    """Return token count, whether it is estimated, and tokenizer name."""
    if not text:
        return 0, False, tokenizer_name or ""

    url = tokenizer_url or LLM_TOKENIZER_URL
    name = tokenizer_name or url

    timeout = aiohttp.ClientTimeout(total=TOKENIZER_TIMEOUT_SECONDS)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json={"content": text}) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {body[:300]}")

                payload = await response.json()

    except Exception as error:
        logger.warning("Falling back to estimated token count for %s: %s", name, error)
        return estimate_token_count(text), True, name

    tokens = payload.get("tokens", []) if isinstance(payload, dict) else []

    if not isinstance(tokens, list):
        logger.warning("Unexpected tokenizer response from %s: %s", name, payload)
        return estimate_token_count(text), True, name

    return len(tokens), False, name


async def count_many_tokens(
    texts: list[str],
    tokenizer_url: str | None = None,
    tokenizer_name: str | None = None,
) -> tuple[int, bool, str]:
    url = tokenizer_url or LLM_TOKENIZER_URL
    name = tokenizer_name or url

    results = await asyncio.gather(
        *(count_tokens(text, url, name) for text in texts),
    )

    total = sum(count for count, _, _ in results)
    is_estimated = any(estimated for _, estimated, _ in results)
    return total, is_estimated, name
