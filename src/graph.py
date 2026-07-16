from typing import Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    add_citations_node,
    generate_answer_node,
    intent_router_node,
    output_safety_filter_node,
    rag_pipeline_node,
    rewrite_query_node,
    safety_and_intent_node,
    save_conversation_node,
    voice_to_text_node,
)
from .state import LibraryBotState


def route_by_input_type(
    state: LibraryBotState,
) -> Literal["voice_path", "direct_to_safety"]:
    if (
        state["input_type"] == "voice"
        and state.get("stt_confidence", 1.0) < 0.85
    ):
        return "voice_path"
    return "direct_to_safety"


def after_voice(
    state: LibraryBotState,
) -> Literal["safety"]:
    return "safety"


def after_safety(
    state: LibraryBotState,
) -> Literal["end", "continue"]:
    if state.get("end_conversation", False):
        return "end"
    return "continue"


def after_intent(
    state: LibraryBotState,
) -> Literal["rag_path", "direct_path"]:
    if state.get("request_type", "rag_search") == "normal_info":
        return "direct_path"
    return "rag_path"


def route_safety_decision(
    state: LibraryBotState,
) -> Literal["show", "block"]:
    return (
        "show"
        if state.get("is_output_safe", True)
        else "block"
    )


builder = StateGraph(LibraryBotState)

builder.add_node("voice_to_text", voice_to_text_node)
builder.add_node("safety", safety_and_intent_node)
builder.add_node("intent_router", intent_router_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("rag_pipeline", rag_pipeline_node)
builder.add_node("generate_answer", generate_answer_node)
builder.add_node("add_citations", add_citations_node)
builder.add_node("output_safety_filter", output_safety_filter_node)
builder.add_node("save_conversation", save_conversation_node)

builder.add_conditional_edges(
    START,
    route_by_input_type,
    {
        "voice_path": "voice_to_text",
        "direct_to_safety": "safety",
    },
)

builder.add_conditional_edges(
    "voice_to_text",
    after_voice,
    {"safety": "safety"},
)

builder.add_conditional_edges(
    "safety",
    after_safety,
    {
        "end": END,
        "continue": "intent_router",
    },
)

builder.add_conditional_edges(
    "intent_router",
    after_intent,
    {
        "rag_path": "rewrite_query",
        "direct_path": "generate_answer",
    },
)

builder.add_edge("rewrite_query", "rag_pipeline")
builder.add_edge("rag_pipeline", "generate_answer")

builder.add_edge("generate_answer", "add_citations")
builder.add_edge("add_citations", "output_safety_filter")

builder.add_conditional_edges(
    "output_safety_filter",
    route_safety_decision,
    {
        "show": "save_conversation",
        "block": END,
    },
)

builder.add_edge("save_conversation", END)

compiled_workflow = builder.compile()
