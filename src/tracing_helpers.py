import json
from typing import Any

from openinference.semconv.trace import SpanAttributes


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def set_span_io(
    span,
    span_kind: str,
    input_value: Any = None,
    output_value: Any = None,
) -> None:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind)

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