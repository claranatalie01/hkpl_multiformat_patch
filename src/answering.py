"""Build the shared evidence-grounded prompt used by live and evaluated RAG.

Keeping answer instructions and source formatting here prevents the benchmark
from testing a different prompt than the public workflow.  The module performs
no retrieval or model calls; callers remain responsible for selecting evidence,
enforcing token limits, and invoking the private generation endpoint.
"""

from __future__ import annotations

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have that information in my knowledge base."
)


def format_source_block(
    content: str,
    rank: int,
    *,
    title: str = "",
) -> str:
    """Format one selected evidence chunk with a stable source label."""

    label = f"Source {rank}"
    if title.strip():
        label += f": {title.strip()}"
    return f"[{label}]\n{content.strip()}"


def answer_completion_budget(
    visible_answer_tokens: int,
    *,
    enable_thinking: bool,
    thinking_budget_tokens: int,
) -> int:
    """Reserve visible-answer tokens in addition to optional hidden reasoning."""

    if visible_answer_tokens <= 0:
        raise ValueError("visible_answer_tokens must be positive")
    if thinking_budget_tokens < 0:
        raise ValueError("thinking_budget_tokens cannot be negative")
    return visible_answer_tokens + (
        thinking_budget_tokens if enable_thinking else 0
    )


def build_grounded_answer_prompt(
    *,
    question: str,
    context: str,
    current_datetime: str = "",
    library_name: str = "",
    library_code: str = "",
) -> str:
    """Return the canonical prompt for answering from approved evidence only."""

    request_hints: list[str] = []
    if current_datetime:
        request_hints.append(f"Current date and time: {current_datetime}.")
    if library_name:
        branch = f"Current library: {library_name}"
        if library_code:
            branch += f" (code: {library_code})"
        request_hints.append(branch + ".")
    hints = "\n".join(request_hints)
    if hints:
        hints = f"\nRequest context:\n{hints}\n"

    return f"""You are the Hong Kong Public Libraries question-answering assistant.

Answer the question using only the retrieved context.

Rules:
- Output only the final answer. Do not reveal analysis, reconsideration, or
  self-correction.
- Be concise. Normally answer in one to three sentences; use a short list only
  when the question requests multiple items.
- For a yes/no question, begin with "Yes" or "No" and give the decisive
  supporting fact. Never include contradictory yes and no conclusions.
- Identify every constraint and relationship in the question. Use evidence
  that satisfies those constraints together.
- Prefer evidence matching the exact entity, location, date, and requested
  relationship over a broader heading, group, or date range.
- Do not infer that a fact about a group applies to every member, or that a fact
  about one member applies to the group.
- Combine multiple passages when needed, but never invent missing information.
- If the context is insufficient, answer exactly:
  "{INSUFFICIENT_EVIDENCE_ANSWER}"
{hints}
Retrieved context:
{context}

Question:
{question}

Answer:
"""
