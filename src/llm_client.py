import logging
import os
from dataclasses import dataclass

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


async def http_llm_with_usage(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
) -> LLMResponse:
    max_tokens = max_tokens or LLM_MAX_TOKENS
    payload = {
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
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
    content = data["choices"][0]["message"]["content"]
    if not content or not content.strip():
        content = "I'm sorry, I couldn't generate a proper answer. Please try again."
    raw_usage = data.get("usage") or {}
    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or 0)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(
            raw_usage.get("total_tokens") or prompt_tokens + completion_tokens
        ),
        "is_estimated": False,
        "tokenizer": LLM_TOKENIZER_NAME,
    }
    return LLMResponse(text=content, usage=usage)


async def http_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
) -> str:
    response = await http_llm_with_usage(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    return response.text
