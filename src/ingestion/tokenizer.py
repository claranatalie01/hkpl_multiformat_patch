from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKENIZER_PATH = "/app/models/qwen3-embedding"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def get_embedding_tokenizer(max_tokens: int = DEFAULT_MAX_TOKENS):
    """Load the pinned embedding tokenizer locally unless explicitly opted in."""
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    configured_path = Path(
        os.getenv("EMBEDDING_TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH)
    )
    allow_remote = _env_true("ALLOW_REMOTE_MODEL_DOWNLOADS")

    if configured_path.exists():
        model_name: str = str(configured_path)
        local_files_only = True
    elif allow_remote:
        model_name = os.getenv(
            "EMBEDDING_TOKENIZER_MODEL",
            DEFAULT_TOKENIZER_MODEL,
        )
        local_files_only = False
    else:
        raise RuntimeError(
            "The Qwen3 embedding tokenizer is not available at "
            f"{configured_path}. Prefetch the pinned model into that directory "
            "or explicitly set ALLOW_REMOTE_MODEL_DOWNLOADS=true for development."
        )

    return HuggingFaceTokenizer.from_pretrained(
        model_name=model_name,
        max_tokens=max_tokens,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
