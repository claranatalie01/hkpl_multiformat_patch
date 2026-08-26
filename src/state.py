"""Typed state contracts passed between LangGraph workflow nodes.

The state records request context, routing decisions, retrieved chunks and
sources, safety flags, generated output, and the selected library branch.
It contains transient workflow data rather than database behavior.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ConversationTurn(TypedDict):
    role: str
    content: str


class LibraryBotState(TypedDict):
    messages: List[BaseMessage]

    session_id: str
    conversation_history: List[ConversationTurn]

    intent: str
    request_type: Literal[
        "normal_info",
        "rag_search",
    ]

    original_query: str
    rewritten_query: str

    retrieved_chunks: List[str]
    retrieved_scores: List[float]
    retrieved_sources: List[Dict[str, Any]]
    generated_answer: str

    is_output_safe: bool
    end_conversation: bool

    current_library_code: Optional[str]
    current_library_name: Optional[str]
