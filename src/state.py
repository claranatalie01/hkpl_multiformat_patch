from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ConversationTurn(TypedDict):
    role: str
    content: str


class LibraryBotState(TypedDict):
    messages: List[BaseMessage]

    session_id: str
    conversation_history: List[ConversationTurn]

    input_type: Literal["text", "voice"]
    stt_confidence: float

    intent: str
    request_type: Literal[
        "sensitive_reject",
        "normal_info",
        "tool_use",
        "rag_search",
        "mcp_tool",
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
    user_memory: Dict[str, Any]
