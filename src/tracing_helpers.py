import json
from typing import Any

from openinference.semconv.trace import SpanAttributes


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


def set_document_list_attributes(
    span,
    prefix: str,
    documents: list[dict],
) -> None:
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

    if usage:
        if usage.get("prompt_tokens") is not None:
            span.set_attribute(
                "llm.token_count.prompt",
                int(usage["prompt_tokens"]),
            )

        if usage.get("completion_tokens") is not None:
            span.set_attribute(
                "llm.token_count.completion",
                int(usage["completion_tokens"]),
            )

        if usage.get("total_tokens") is not None:
            span.set_attribute(
                "llm.token_count.total",
                int(usage["total_tokens"]),
            )
