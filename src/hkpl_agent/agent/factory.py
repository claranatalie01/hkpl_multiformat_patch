"""Construct the initial typed state supplied to the LangGraph workflow."""

from langchain_core.messages import HumanMessage


def build_initial_state(
    *,
    question: str,
    session_id: str,
    conversation_history: list,
    request_type: str = "normal_info",
    current_library: dict | None = None,
) -> dict:
    """Return a complete initial state for one chat or admin-test request."""

    return {
        "messages": [HumanMessage(content=question)],
        "session_id": session_id,
        "conversation_history": conversation_history,
        "original_query": question,
        "rewritten_query": question,
        "intent": "",
        "request_type": request_type,
        "retrieved_chunks": [],
        "retrieved_scores": [],
        "retrieved_sources": [],
        "generated_answer": "",
        "is_output_safe": True,
        "end_conversation": False,
        "current_library_code": current_library["code"] if current_library else None,
        "current_library_name": current_library["name"] if current_library else None,
    }

