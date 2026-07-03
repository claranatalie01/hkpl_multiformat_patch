#!/usr/bin/env python3

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://phoenix:6006"

from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

tracer_provider = register(
    project_name="hkpl-rag",
    endpoint="http://phoenix:6006/v1/traces",
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

from src.retrieval import retrieve_nodes
from src.nodes import http_llm


async def run_one_question(question: str):
    nodes = await retrieve_nodes(question)

    context = "\n\n".join(
        f"[Source {i+1}]\n{node.node.get_content()}"
        for i, node in enumerate(nodes)
    )

    prompt = f"""
You are the official Hong Kong Public Libraries assistant.

Answer using only the retrieved context.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    answer = await http_llm(prompt, temperature=0.0, max_tokens=512)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for i, node in enumerate(nodes, start=1):
        metadata = node.node.metadata or {}
        print(f"{i}. {metadata.get('source_title', '')} | score={node.score}")


async def main():
    questions = [
        "Where can I read e-books?",
        "What are the rules for changing my HKPL account password?",
        "What is the telephone number of Hong Kong Central Library?",
        "When and where is the Parent-Child Story Theatre event?",
    ]

    for question in questions:
        await run_one_question(question)


if __name__ == "__main__":
    asyncio.run(main())