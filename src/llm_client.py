"""Call the local OpenAI-compatible answer model and normalize its usage data.

This client is used for query rewriting, answer generation, and evaluation
candidate/judge calls. It generates language; document and query embeddings are
handled separately by ``infrastructure.embedding``.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import aiohttp
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)

LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16000"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
LLM_TOKENIZER_NAME = os.getenv("LLM_TOKENIZER_NAME", "llama.cpp:qwen3.5-9b")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: dict
    reasoning_text: str = ""


async def http_llm_with_usage(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 1000,
    response_format: dict[str, Any] | None = None,
) -> LLMResponse:
    max_tokens = max_tokens or LLM_MAX_TOKENS
    payload = {
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        # llama.cpp supports a per-request thinking budget. Sending zero keeps
        # ordinary requests in no-thinking mode while allowing selected
        # benchmark retries to opt into a bounded reasoning pass.
        "thinking_budget_tokens": (
            thinking_budget_tokens if enable_thinking else 0
        ),
    }
    if response_format is not None:
        payload["response_format"] = response_format
    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            LLM_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(
                    f"LLM service error {response.status}: {body}"
                )
            data = await response.json()

    logger.debug("LLM raw response: %s", data)
    message = data["choices"][0]["message"]
    content = message["content"]
    reasoning_content = str(message.get("reasoning_content") or "")
    if not content or not content.strip():
        content = "I'm sorry, I couldn't generate a proper answer. Please try again."
    raw_usage = data.get("usage") or {}
    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or 0)
    completion_details = raw_usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(
            raw_usage.get("total_tokens") or prompt_tokens + completion_tokens
        ),
        "reasoning_tokens": reasoning_tokens,
        "is_estimated": False,
        "tokenizer": LLM_TOKENIZER_NAME,
    }
    return LLMResponse(
        text=content,
        usage=usage,
        reasoning_text=reasoning_content,
    )


async def http_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
    thinking_budget_tokens: int = 1000,
    response_format: dict[str, Any] | None = None,
) -> str:
    response = await http_llm_with_usage(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        response_format=response_format,
    )
    return response.text
