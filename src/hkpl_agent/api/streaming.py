"""Small formatting helpers for server-sent event responses."""


def format_sse(event: str, data: str) -> str:
    """Encode one named event while preserving multiline payloads."""

    lines = str(data).splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n"

