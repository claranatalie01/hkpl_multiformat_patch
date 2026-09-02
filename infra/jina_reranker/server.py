"""HTTP adapter for the official Jina Reranker v3 GGUF implementation.

The HKPL RAG application expects a Jina-compatible ``POST /reranking`` API,
whereas ``jinaai/jina-reranker-v3-GGUF`` is distributed as a GGUF backbone,
an external MLP projector, and a Python ``rerank.py`` implementation.  This
service downloads those pinned artifacts, executes the official listwise
scoring code with the required llama.cpp fork, and exposes the result using
the same JSON contract as the existing Qwen reranker service.

This container is intended only for the controlled PoC A/B experiment.  The
model is CC BY-NC 4.0 and its CLI-based implementation starts a
``llama-embedding`` subprocess for every reranking request, so its latency is
not directly representative of an optimized persistent production server.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from tokenizers import Tokenizer


MODEL_REPOSITORY = os.getenv(
    "JINA_MODEL_REPOSITORY",
    "jinaai/jina-reranker-v3-GGUF",
)
MODEL_REVISION = os.getenv(
    "JINA_MODEL_REVISION",
    "4bbace80cf59987f6fec850519012341c06810d5",
)
MODEL_FILENAME = os.getenv(
    "JINA_MODEL_FILENAME",
    "jina-reranker-v3-Q8_0.gguf",
)
TOKENIZER_REPOSITORY = os.getenv(
    "JINA_TOKENIZER_REPOSITORY",
    "jinaai/jina-reranker-v3",
)
TOKENIZER_REVISION = os.getenv(
    "JINA_TOKENIZER_REVISION",
    "d7d7e73b6ea138ced340b83865931b5dfb6c97aa",
)
MODEL_CACHE_DIRECTORY = Path(os.getenv("JINA_MODEL_CACHE", "/models"))
LLAMA_EMBEDDING_PATH = os.getenv(
    "JINA_LLAMA_EMBEDDING_PATH",
    "/usr/local/bin/llama-embedding",
)
MAX_DOCUMENTS = int(os.getenv("JINA_MAX_DOCUMENTS", "64"))


class RerankRequest(BaseModel):
    """Request accepted by the Jina-compatible reranking endpoints."""

    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    model: str | None = None
    instruction: str | None = None


class TokenizeRequest(BaseModel):
    """Request used by the HKPL token-accounting client."""

    content: str


@dataclass
class JinaRuntime:
    """Loaded model adapter and tokenizer shared by all HTTP requests."""

    reranker: Any
    tokenizer: Tokenizer
    lock: asyncio.Lock


def _download(filename: str, repository: str, revision: str) -> str:
    """Download one revision-pinned artifact into the persistent cache."""

    token = os.getenv("HF_TOKEN", "").strip() or None
    return hf_hub_download(
        repo_id=repository,
        filename=filename,
        revision=revision,
        cache_dir=str(MODEL_CACHE_DIRECTORY),
        token=token,
    )


def _load_module(path: str) -> ModuleType:
    """Load Jina's revision-pinned official ``rerank.py`` implementation."""

    spec = importlib.util.spec_from_file_location("jina_v3_official_rerank", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Jina reranker implementation from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_runtime() -> JinaRuntime:
    """Download and initialize all artifacts required for Jina v3 scoring."""

    MODEL_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = _download(MODEL_FILENAME, MODEL_REPOSITORY, MODEL_REVISION)
    projector_path = _download(
        "projector.safetensors",
        MODEL_REPOSITORY,
        MODEL_REVISION,
    )
    implementation_path = _download("rerank.py", MODEL_REPOSITORY, MODEL_REVISION)
    tokenizer_path = _download(
        "tokenizer.json",
        TOKENIZER_REPOSITORY,
        TOKENIZER_REVISION,
    )

    module = _load_module(implementation_path)
    reranker = module.GGUFReranker(
        model_path=model_path,
        projector_path=projector_path,
        llama_embedding_path=LLAMA_EMBEDDING_PATH,
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return JinaRuntime(
        reranker=reranker,
        tokenizer=tokenizer,
        lock=asyncio.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before readiness succeeds and release it on shutdown."""

    app.state.runtime = await asyncio.to_thread(build_runtime)
    yield
    app.state.runtime = None


app = FastAPI(
    title="HKPL Jina Reranker v3 GGUF Adapter",
    version="1.0.0",
    lifespan=lifespan,
)


def _runtime(request: Request) -> JinaRuntime:
    """Return the initialized runtime or a stable service-unavailable error."""

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Jina reranker is not ready")
    return runtime


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    """Report readiness after model, projector, and tokenizer initialization."""

    _runtime(request)
    return {
        "status": "ok",
        "model": f"{MODEL_REPOSITORY}:Q8_0",
    }


@app.post("/tokenize")
async def tokenize(payload: TokenizeRequest, request: Request) -> dict[str, list[int]]:
    """Return token IDs for evaluation token-usage accounting."""

    runtime = _runtime(request)
    encoding = runtime.tokenizer.encode(payload.content)
    return {"tokens": encoding.ids}


async def _rerank(payload: RerankRequest, request: Request) -> dict[str, Any]:
    """Execute one bounded listwise reranking request."""

    if len(payload.documents) > MAX_DOCUMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_DOCUMENTS} documents may be reranked per request",
        )
    if any(not document.strip() for document in payload.documents):
        raise HTTPException(status_code=422, detail="Documents must not be empty")

    runtime = _runtime(request)
    top_n = min(payload.top_n or len(payload.documents), len(payload.documents))

    try:
        async with runtime.lock:
            results = await asyncio.to_thread(
                runtime.reranker.rerank,
                payload.query,
                payload.documents,
                top_n,
                False,
                payload.instruction,
            )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {error}") from error

    # The HKPL client only needs the original candidate index and score.  Do
    # not send full document text back across the container boundary.
    normalized_results = [
        {
            "index": int(result["index"]),
            "relevance_score": float(result["relevance_score"]),
        }
        for result in results
    ]

    return {
        "model": MODEL_REPOSITORY,
        "results": normalized_results,
    }


@app.post("/reranking")
@app.post("/rerank")
@app.post("/v1/rerank")
@app.post("/v1/reranking")
async def rerank(payload: RerankRequest, request: Request) -> dict[str, Any]:
    """Expose aliases supported by llama.cpp and the Jina reranking API."""

    return await _rerank(payload, request)
