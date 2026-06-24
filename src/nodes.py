import os
import logging
import json
import asyncio
import re
import time
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from gliner2 import GLiNER2
from .state import LibraryBotState
from .retrieval import retrieve_nodes
from .memory import save_conversation_turn
import aiohttp

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)          # ✅ moved after logger definition

def get_current_datetime():
    """Returns the current date and time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------------------------------------------------
# HTTP endpoint for LLM (answer generation)
# ----------------------------------------------------------------------
LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")

async def http_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen3.5-9b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(LLM_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"LLM service error {resp.status}: {text}")
            data = await resp.json()
            logger.debug(f"LLM raw response: {data}")
            response = data["choices"][0]["message"]["content"]
            if not response or len(response.strip()) == 0:
                response = "I'm sorry, I couldn't generate a proper answer. Please try again."
            return response

# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
async def voice_to_text_node(state: LibraryBotState) -> dict:
    """Mock voice transcription node (kept as placeholder)."""
    logger.info("[Node] Voice-to-Text Conversion (mock)")
    transcribed = "請問沙田圖書館幾點關門？"
    return {"messages": [HumanMessage(content=transcribed)]}

# ----------------------------------------------------------------------
# GLiGuard safety classifier
# ----------------------------------------------------------------------
safety_model = None

def get_safety_model():
    global safety_model
    if safety_model is None:
        safety_model = GLiNER2.from_pretrained("fastino/gliguard-LLMGuardrails-300M")
        safety_model.to("cpu")
    return safety_model

async def safety_and_intent_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Safety Classifier (GLiGuard)")
    user_input = state["messages"][-1].content
    model = get_safety_model()

    toxicity_labels = [
        "violence_and_weapons", "non_violent_crime", "sexual_content",
        "hate_and_discrimination", "self_harm_and_suicide", "pii_exposure",
        "misinformation", "copyright_violation", "child_safety",
        "political_manipulation", "unethical_conduct", "regulated_advice",
        "privacy_violation", "other", "benign",
    ]
    toxicity_task = {
        "labels": toxicity_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    jailbreak_labels = [
        "prompt_injection", "jailbreak_attempt", "policy_evasion",
        "instruction_override", "system_prompt_exfiltration", "data_exfiltration",
        "roleplay_bypass", "hypothetical_bypass", "obfuscated_attack",
        "multi_step_attack", "social_engineering", "benign",
    ]
    jailbreak_task = {
        "labels": jailbreak_labels,
        "multi_label": True,
        "cls_threshold": 0.4,
    }
    schema = {
        "prompt_safety": ["safe", "unsafe"],
        "prompt_toxicity": toxicity_task,
        "jailbreak_detection": jailbreak_task,
    }

    try:
        result = model.classify_text(user_input, schema, threshold=0.5)
        logger.debug(f"GLiGuard raw result: {result}")

        safety = result.get("prompt_safety", "safe")
        detected_categories = []

        toxicity = result.get("prompt_toxicity", [])
        if isinstance(toxicity, list):
            detected_categories.extend(toxicity)

        jailbreak = result.get("jailbreak_detection", [])
        if isinstance(jailbreak, list):
            detected_categories.extend(jailbreak)

        # Remove benign + duplicates.
        detected_categories = sorted({
            cat for cat in detected_categories
            if cat and cat != "benign"
        })

        # High-risk categories: block immediately.
        # These are categories where allowing the request could be unsafe.
        hard_block_categories = {
            "violence_and_weapons",
            "sexual_content",
            "hate_and_discrimination",
            "self_harm_and_suicide",
            "pii_exposure",
            "child_safety",
            "political_manipulation",
            "privacy_violation",
            "prompt_injection",
            "jailbreak_attempt",
            "policy_evasion",
            "instruction_override",
            "system_prompt_exfiltration",
            "data_exfiltration",
            "obfuscated_attack",
        }

        # Ambiguous categories: log for monitoring, but do not block by themselves.
        # These caused false positives for valid HKPL questions such as:
        # "What is Adobe Digital Editions?" and "What is it and how do I use it?"
        soft_flag_categories = {
            "non_violent_crime",
            "copyright_violation",
            "misinformation",
            "regulated_advice",
            "unethical_conduct",
            "multi_step_attack",
            "hypothetical_bypass",
            "roleplay_bypass",
            "social_engineing",
            "social_engineering",
            "other",
        }

        hard_hits = [
            cat for cat in detected_categories
            if cat in hard_block_categories
        ]

        soft_hits = [
            cat for cat in detected_categories
            if cat in soft_flag_categories
        ]

        # Final safety decision:
        # - hard-hit categories are blocked.
        # - soft-hit categories are only logged.
        # - prompt_safety is kept for logging, but not trusted alone because
        #   GLiGuard can return contradictory outputs such as unsafe + benign.
        is_unsafe = len(hard_hits) > 0

        # Keep this alias so the existing log_entry / debugging code remains clear.
        blocking_hits = hard_hits

        logger.info(
            f"Safety={safety}; categories={detected_categories}; "
            f"hard_hits={hard_hits}; soft_hits={soft_hits}; is_unsafe={is_unsafe}"
        )

    except Exception as e:
        logger.error(f"GLiGuard classification error: {e}")
        is_unsafe = False
        detected_categories = []
        hard_hits = []
        soft_hits = []
        blocking_hits = []

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input[:500],
        "safety": "unsafe" if is_unsafe else "safe",
        "categories": detected_categories,
        "hard_hits": hard_hits,
        "soft_hits": soft_hits,
    }

    if is_unsafe:
        with open("safety_intent_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.warning(f"Unsafe input blocked: {user_input[:100]} | categories: {detected_categories}")
        if any(cat in ["self_harm_and_suicide", "self_harm", "suicide"] for cat in detected_categories):
            safe_msg = (
                "I'm really sorry you're feeling this way. Please know that you're not alone. "
                "If you are in distress, please reach out to the Samaritans Hong Kong 24‑hour hotline at 2896 0000, "
                "or contact a mental health professional. Your well‑being is very important."
            )
        elif "political_manipulation" in detected_categories:
            safe_msg = (
                "I'm here to help with library services, book information, and general library questions. "
                "I can't discuss political topics. Is there something about the library I can help you with?"
            )
        elif any(cat in ["prompt_injection", "jailbreak_attempt", "instruction_override"] for cat in detected_categories):
            safe_msg = (
                "I can only follow instructions related to library services. "
                "Please ask a genuine question about library hours, book searches, or library policies."
            )
        elif "violence_and_weapons" in detected_categories:
            safe_msg = (
                "I cannot provide information that promotes or glorifies violence. "
                "If you need help with library resources, I'm happy to assist."
            )
        elif "hate_and_discrimination" in detected_categories:
            safe_msg = (
                "I strive to be respectful and helpful to everyone. "
                "Please ask a library‑related question without using offensive language."
            )
        else:
            safe_msg = (
                "I'm unable to process that request. Please ask a library‑related question, such as "
                "library hours, book availability, or how to borrow materials."
            )
        return {
            "messages": [AIMessage(content=safe_msg)],
            "is_output_safe": True,
            "end_conversation": True
        }
    return {}

# ----------------------------------------------------------------------
# Intent router (only distinguishes greetings from everything else)
# ----------------------------------------------------------------------
async def intent_router_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Intent Router")

    user_query = state["messages"][-1].content.strip()

    prompt = f"""
Classify the user's message into exactly one of these labels:

- greeting: greetings such as hello, hi, good morning
- thanks: thank you messages
- library_question: any question about HKPL services, library materials, e-resources, borrowing, collections, classification schemes, opening hours, reference services, accounts, apps, or library help
- other: anything else that is safe but not a greeting or thanks

Return only valid JSON:
{{"intent": "greeting" | "thanks" | "library_question" | "other"}}

User message:
{user_query}
"""

    try:
        raw = await http_llm(prompt, temperature=0.0, max_tokens=80)
        raw = raw.strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed = json.loads(raw)
        intent = parsed.get("intent", "library_question")

    except Exception as e:
        logger.warning(f"Intent classification failed, defaulting to RAG: {e}")
        intent = "library_question"

    if intent in ["greeting", "thanks"]:
        request_type = "normal_info"
    else:
        request_type = "rag_search"

    logger.info(f"Intent classified as: {intent}; request_type={request_type}")

    return {
        "intent": intent,
        "request_type": request_type,
    }
# ----------------------------------------------------------------------
# RAG pipeline using LlamaIndex retriever (includes reranking)
# ----------------------------------------------------------------------
async def rag_pipeline_node(state: LibraryBotState) -> dict:
    # Log that the RAG retrieval node has started
    logger.info("[Node] RAG Pipeline (LlamaIndex + PGVectorStore + reranking)")

    # Use rewritten query if available; otherwise use the latest user message
    query = state.get("rewritten_query") or state["messages"][-1].content
    logger.debug(f"Retrieval query: {query}")

    # Measure retrieval time for debugging and performance monitoring
    start = time.time()

    # Retrieve relevant nodes using LlamaIndex retriever
    nodes = await retrieve_nodes(query)

    # Calculate retrieval duration
    elapsed = time.time() - start
    logger.info(f"Retrieval took {elapsed:.3f} seconds")

    # Store retrieved chunk texts
    chunk_texts = []

    # Store similarity/reranking scores
    scores = []

    # Store metadata used later for citations/debugging
    sources = []

    # Convert LlamaIndex NodeWithScore objects into plain Python data
    for i, node in enumerate(nodes):
        # Metadata was saved during ingestion
        metadata = node.node.metadata or {}

        # Use 0.0 if score is missing
        score = node.score if node.score is not None else 0.0

        # Store the actual retrieved text
        chunk_texts.append(node.node.text)

        # Store the retrieval/reranking score
        scores.append(score)

        # Store source metadata for citations and traceability
        sources.append(
            {
                "chunk_index": i,
                "score": score,
                "document_id": metadata.get("document_id", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "source": metadata.get("source", ""),
                "source_title": metadata.get(
                    "source_title",
                    metadata.get("file_name", "HKPL knowledge base"),
                ),
                "url": metadata.get("url", ""),
                "source_url": metadata.get(
                    "source_url",
                    metadata.get("url", ""),
                ),
                "file_name": metadata.get("file_name", ""),
                "file_type": metadata.get("file_type", ""),
                "page_number": metadata.get("page_number"),
                "slide_number": metadata.get("slide_number"),
                "section_heading": metadata.get("section_heading", ""),
                "sheet_name": metadata.get("sheet_name", ""),
                "row_number": metadata.get("row_number"),
                "domain": metadata.get("domain", ""),
                "question": metadata.get("question", ""),
                "document_version": metadata.get("document_version"),
                "access_level": metadata.get("access_level", "public"),
            }
        )


    # Build context string passed to the answer generation node
    if chunk_texts:
        context = "\n\n".join(
            f"[Source {i + 1}]\n{text}"
            for i, text in enumerate(chunk_texts)
        )
    else:
        context = "No relevant documents found."

    # Log retrieval details for debugging
    logger.debug(f"Retrieved {len(nodes)} nodes. Scores: {scores}")

    # Log the top retrieved chunk and source
    if chunk_texts:
        logger.debug(f"Top chunk text: {chunk_texts[0][:300]}")
        logger.debug(f"Top source: {sources[0]}")

    # Return retrieval results into LangGraph state
    return {
        "retrieved_chunks": chunk_texts,
        "retrieved_context": context,
        "retrieved_scores": scores,
        "retrieved_sources": sources,
    }

def format_citations(sources: list[dict]) -> str:
    """Create readable citations for FAQs, PDFs, Office files, and tables."""
    seen = set()
    citation_lines = []

    for source in sources:
        title = (
            source.get("source_title")
            or source.get("file_name")
            or "HKPL knowledge base"
        )
        url = source.get("source_url") or source.get("url", "")
        page = source.get("page_number")
        slide = source.get("slide_number")
        section = source.get("section_heading", "")
        sheet = source.get("sheet_name", "")
        row = source.get("row_number")
        domain = source.get("domain", "")
        question = source.get("question", "")

        key = (
            title,
            url,
            page,
            slide,
            section,
            sheet,
            row,
            question,
        )
        if key in seen:
            continue
        seen.add(key)

        details = []
        if page:
            details.append(f"page {page}")
        if slide:
            details.append(f"slide {slide}")
        if section:
            details.append(f'section "{section}"')
        if sheet:
            details.append(f'sheet "{sheet}"')
        if row:
            details.append(f"row {row}")
        if domain:
            details.append(domain)

        line = f"- {title}"
        if details:
            line += f" ({', '.join(details)})"
        if question:
            line += f": {question}"
        if url:
            line += f"\n  {url}"

        citation_lines.append(line)

    if not citation_lines:
        return ""

    return "\n\nSources:\n" + "\n".join(citation_lines[:1])


async def faithfulness_check_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Faithfulness Check")

    answer = state.get("generated_answer") or state["messages"][-1].content
    context = state.get("retrieved_context", "")

    if not context or context == "No relevant documents found.":
        fallback = (
            "I don't have enough verified information in my knowledge base "
            "to answer that reliably. Please try rephrasing or ask about a "
            "specific HKPL service."
        )

        return {
            "faithfulness_passed": False,
            "faithfulness_reason": "No retrieved context available.",
            "messages": [AIMessage(content=fallback)],
            "generated_answer": fallback,
        }

    prompt = f"""
You are checking whether an HKPL assistant answer is fully supported by the retrieved context.

Return ONLY valid JSON.

Format:
{{"supported": true, "reason": "short reason"}}
or
{{"supported": false, "reason": "short reason"}}

Rules:
- Return supported=true if the answer is broadly supported by the retrieved context.
- Do not reject because of minor wording differences.
- Do not reject if the answer merges equivalent details from multiple retrieved sources.
- Return supported=false only if the answer gives a clearly wrong instruction, unsupported phone number, unsupported service name, unsupported requirement, or unsupported URL.

Retrieved context:
{context}

Assistant answer:
{answer}

JSON:
"""

    try:
        raw = await http_llm(prompt, temperature=0.0, max_tokens=256)
        raw = raw.strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        result = json.loads(raw)
        supported = bool(result.get("supported", False))
        reason = result.get("reason", "")

    except Exception as e:
        logger.error(f"Faithfulness check failed: {e}")
        supported = True
        reason = "Faithfulness checker failed; answer allowed."

    logger.info(f"Faithfulness result: supported={supported}, reason={reason}")

    return {
        "faithfulness_passed": supported,
        "faithfulness_reason": reason,
    }


def _is_fallback_answer(answer: str) -> bool:
    lowered = answer.lower()
    fallback_markers = [
        "i don't have enough verified information",
        "i don't have enough confidence",
        "i don't have that information in my knowledge base",
        "i couldn't find that information",
    ]
    return any(marker in lowered for marker in fallback_markers)


async def add_citations_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Add Citations")

    answer = state.get("generated_answer") or state["messages"][-1].content

    if state.get("request_type") == "normal_info":
        return {
            "messages": [AIMessage(content=answer)],
            "generated_answer": answer,
        }

    sources = state.get("retrieved_sources", [])
    citations = format_citations(sources)

    if not _is_fallback_answer(answer) and citations and "Sources:" not in answer:
        answer = answer.strip() + citations

    return {
        "messages": [AIMessage(content=answer)],
        "generated_answer": answer,
    }


async def rewrite_query_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Query Rewriter")

    question = state["messages"][-1].content
    history = state.get("conversation_history", [])

    if not history:
        return {
            "original_query": question,
            "rewritten_query": question,
        }

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in history
    )

    prompt = f"""
You rewrite follow-up library questions into standalone search queries.

Rules:
- Use the conversation history only to resolve references like "it", "that", "them", "do I need it".
- Do not answer the question.
- Do not add information not implied by the history.
- Output only the rewritten query.

Conversation history:
{history_text}

Current user question:
{question}

Standalone search query:
"""

    try:
        rewritten = await http_llm(prompt, temperature=0.0, max_tokens=128)
        rewritten = rewritten.strip()

        if not rewritten:
            rewritten = question

    except Exception as e:
        logger.error(f"Query rewrite failed: {e}")
        rewritten = question

    logger.debug(f"Original query: {question}")
    logger.debug(f"Rewritten query: {rewritten}")

    return {
        "original_query": question,
        "rewritten_query": rewritten,
    }

async def save_conversation_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Save Conversation History")

    session_id = state.get("session_id", "")
    user_question = state.get("original_query") or state["messages"][0].content
    assistant_answer = state["messages"][-1].content

    try:
        save_conversation_turn(session_id, "user", user_question)
        save_conversation_turn(session_id, "assistant", assistant_answer)
    except Exception as e:
        logger.error(f"Failed to save conversation history: {e}")

    return {}

# ----------------------------------------------------------------------
# Generate answer (using HTTP LLM)
# ----------------------------------------------------------------------
async def generate_answer_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Generate Answer")
    request_type = state.get("request_type", "rag_search")
    question = state["messages"][-1].content

    if request_type == "normal_info":
        system_msg = "You are a helpful library assistant. Keep answers short and friendly."
        full_prompt = f"{system_msg}\n\nUser: {question}"
        response = await http_llm(full_prompt, temperature=0.7)
        return {
            "messages": [AIMessage(content=response)],
            "generated_answer": response,
        }

    retrieved_chunks = state.get("retrieved_chunks", [])
    max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

    selected_context_parts = []
    used_chars = 0

    for index, chunk in enumerate(retrieved_chunks):
        formatted_chunk = f"[Source {index + 1}]\n{chunk}"

        if (
            selected_context_parts
            and used_chars + len(formatted_chunk) > max_context_chars
        ):
            break

        selected_context_parts.append(formatted_chunk)
        used_chars += len(formatted_chunk)

    context = "\n\n".join(selected_context_parts)

    if not context:
        fallback = "I'm sorry, I couldn't find that information. Could you rephrase or ask about a specific library branch (e.g., Shatin Library)?"
        return {
            "messages": [AIMessage(content=fallback)],
            "generated_answer": fallback,
        }
    scores = state.get("retrieved_scores", [])

    if not scores or max(scores) < float(os.getenv("RERANKER_THRESHOLD", "0.50")):
        fallback = "I don't have enough confidence to answer that. Could you rephrase your question or ask about a specific library service?"
        return {
            "messages": [AIMessage(content=fallback)],
            "generated_answer": fallback,
        }

    library_name = state.get("current_library_name")
    library_code = state.get("current_library_code")
    current_time = get_current_datetime()
    user_memory = state.get("user_memory", {})

    location_hint = f"The user is currently at or near **{library_name}** (code: {library_code}). " if library_name else ""
    date_hint = f"Current date and time: {current_time}. " if current_time else ""
    memory_hint = f"User context: {json.dumps(user_memory)}" if user_memory else ""

    # ✅ Build the system prompt first
    system_prompt = f"""You are the official HKPL (Hong Kong Public Libraries) assistant.  
    **IMPORTANT:** Do NOT include any reasoning, thinking, or analysis in your response. Output only the final answer.
    {date_hint}{location_hint}{memory_hint}

    **Instructions:**
    1. Answer based **only** on the provided context. Do not invent facts.
    2. **Generalise across phrasing:** The user's question may use different words, but if the intent matches information in the context, answer it. Treat paraphrases as equivalent to the original question in the context.
    3. **Extract fully:** If the context contains the exact answer, state it clearly and completely.
    4. **Handle partial information:** If the context provides only part of the answer, give what is available and politely note what is missing.
    5. **Handle empty or irrelevant context:** If the context is empty or does not address the question at all, say: "I don't have that information in my knowledge base. Please try rephrasing or ask about a specific library service."
    6. **Be concise:** Keep answers short (1-3 sentences), but include all essential facts.
    7. **Lists:** If the question asks for a list, present it in bullet points.
    8. **Stay specific**: Do not broaden the answer beyond the retrieved FAQ. If the context is about e-resources, answer only about e-resources. Do not add advice for unrelated issues.

    **Context:**
    {context}

    **Question:** {question}
    **Answer:**"""


    logger.debug(f"Context length: {len(context)} characters")
    logger.debug(f"System prompt (first 500 chars): {system_prompt[:500]}...")

    response = await http_llm(system_prompt, temperature=0.0)

    # Store the generated answer separately so later nodes can inspect it.
    return {
        "messages": [AIMessage(content=response)],
        "generated_answer": response,
    }

# ----------------------------------------------------------------------
# Output safety filter
# ----------------------------------------------------------------------
async def output_safety_filter_node(state: LibraryBotState) -> dict:
    logger.info("[Node] Output Safety Filter")
    answer = state["messages"][-1].content
    blocked_phrases = ["self-harm", "suicide", "kill yourself"]
    if any(phrase in answer.lower() for phrase in blocked_phrases):
        return {"is_output_safe": False, "messages": [AIMessage(content="I cannot provide that answer. Please contact library staff or call the Samaritans at 2896 0000 for immediate help.")]}
    return {
        "is_output_safe": True,
        "messages": [AIMessage(content=answer)],
        "generated_answer": answer,
    }