import json
import os
from typing import Any

from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes

from src.token_counting import estimate_token_count


def to_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def set_span_io(
    span,
    span_kind: str,
    input_value: Any = None,
    output_value: Any = None,
) -> None:
    span.set_status(Status(StatusCode.OK))
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        span_kind,
    )

    if input_value is not None:
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            input_value if isinstance(input_value, str) else to_json(input_value),
        )

    if output_value is not None:
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            output_value if isinstance(output_value, str) else to_json(output_value),
        )


def set_json_attribute(
    span,
    key: str,
    value: Any,
) -> None:
    span.set_attribute(
        key,
        to_json(value),
    )


def get_float_env(name: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def set_document_list_attributes(
    span,
    prefix: str,
    documents: list[dict],
) -> None:
    span.set_status(Status(StatusCode.OK))

    for index, document in enumerate(documents):
        metadata = {
            "rank": document.get("rank"),
            "document_id": document.get("document_id", ""),
            "chunk_id": document.get("chunk_id", ""),
            "title": document.get("title") or document.get("source_title", ""),
            "url": document.get("url") or document.get("source_url", ""),
            "page": document.get("page"),
            "section": document.get("section"),
            "score_name": document.get("score_name", ""),
            **(document.get("metadata") or {}),
        }

        document_id = (
            document.get("chunk_id")
            or document.get("document_id")
            or f"document-{index + 1}"
        )

        span.set_attribute(f"{prefix}.{index}.document.id", str(document_id))
        span.set_attribute(
            f"{prefix}.{index}.document.content",
            document.get("text") or document.get("text_preview", ""),
        )
        span.set_attribute(
            f"{prefix}.{index}.document.score",
            float(document.get("score") or 0.0),
        )
        span.set_attribute(
            f"{prefix}.{index}.document.metadata",
            to_json(metadata),
        )


def node_document_payload(
    *,
    rank: int,
    text: str,
    score: float,
    document_id: str = "",
    chunk_id: str = "",
    source_title: str = "",
    source_url: str = "",
) -> dict:
    return {
        "rank": rank,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_title": source_title,
        "source_url": source_url,
        "score": float(score or 0.0),
        "text": text,
        "text_preview": text[:700],
    }


def set_documents_attribute(
    span,
    key: str,
    documents: list[dict],
) -> None:
    set_json_attribute(span, key, documents)


def set_llm_attributes(
    *,
    span,
    model_name: str,
    prompt: str,
    response: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    usage: dict | None = None,
) -> None:
    span.set_status(Status(StatusCode.OK))
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        "LLM",
    )

    span.set_attribute("llm.model_name", model_name)

    span.set_attribute(
        "llm.invocation_parameters",
        to_json(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ),
    )

    span.set_attribute(
        "llm.input_messages.0.message.role",
        "user",
    )
    span.set_attribute(
        "llm.input_messages.0.message.content",
        prompt,
    )
    span.set_attribute(
        "llm.prompts.0.prompt.text",
        prompt,
    )

    span.set_attribute(
        SpanAttributes.INPUT_VALUE,
        prompt,
    )

    if response is not None:
        span.set_attribute(
            "llm.output_messages.0.message.role",
            "assistant",
        )
        span.set_attribute(
            "llm.output_messages.0.message.content",
            response,
        )
        span.set_attribute(
            "llm.choices.0.completion.text",
            response,
        )
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            response,
        )

    prompt_tokens = estimate_token_count(prompt)
    completion_tokens = estimate_token_count(response or "")
    prompt_estimated = True
    completion_estimated = True
    tokenizer_name = "estimated"

    if usage and usage.get("prompt_tokens") is not None:
        prompt_tokens = int(usage["prompt_tokens"])
        prompt_estimated = bool(usage.get("is_estimated", False))
        tokenizer_name = str(usage.get("tokenizer", tokenizer_name))

    if usage and usage.get("completion_tokens") is not None:
        completion_tokens = int(usage["completion_tokens"])
        completion_estimated = bool(usage.get("is_estimated", False))
        tokenizer_name = str(usage.get("tokenizer", tokenizer_name))

    total_tokens = (
        int(usage["total_tokens"])
        if usage and usage.get("total_tokens") is not None
        else prompt_tokens + completion_tokens
    )

    span.set_attribute("llm.token_count.prompt", prompt_tokens)
    span.set_attribute("llm.token_count.completion", completion_tokens)
    span.set_attribute("llm.token_count.total", total_tokens)
    span.set_attribute("llm.token_count.is_estimated", prompt_estimated or completion_estimated)
    span.set_attribute("llm.token_count.tokenizer", tokenizer_name)

    prompt_cost_per_1k = get_float_env("LLM_PROMPT_COST_PER_1K_USD")
    completion_cost_per_1k = get_float_env("LLM_COMPLETION_COST_PER_1K_USD")

    if prompt_cost_per_1k or completion_cost_per_1k:
        prompt_cost = prompt_tokens * prompt_cost_per_1k / 1000
        completion_cost = completion_tokens * completion_cost_per_1k / 1000
        total_cost = prompt_cost + completion_cost

        span.set_attribute("llm.cost.prompt", prompt_cost)
        span.set_attribute("llm.cost.completion", completion_cost)
        span.set_attribute("llm.cost.total", total_cost)
        span.set_attribute("llm.cost.prompt_details.input", prompt_cost)
        span.set_attribute("llm.cost.completion_details.output", completion_cost)
        span.set_attribute("llm.cost.is_estimated", True)
