import logging
import os

import aiohttp
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)

LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "16000"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))


async def http_llm(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    enable_thinking: bool = False,
) -> str:
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
        return "I'm sorry, I couldn't generate a proper answer. Please try again."
    return content
