import os
import faiss
import numpy as np
import phoenix as px

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from phoenix.otel import register
from opentelemetry import trace


# -----------------------------
# 1. Start Phoenix tracing
# -----------------------------
px.launch_app()

tracer_provider = register(
    project_name="rag-phoenix-trial",
    endpoint="http://localhost:6006/v1/traces",
)

tracer = trace.get_tracer(__name__)


# -----------------------------
# 2. Setup models
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 3. Example documents
# Replace this with your PDF chunks later
# -----------------------------
documents = [
    "Central Library opens from 10 AM to 9 PM from Monday to Saturday.",
    "Users can read e-books through the Hong Kong Public Libraries e-Books platform.",
    "Library members can borrow books using their library card or smart ID.",
    "Some libraries close earlier on public holidays.",
]


# -----------------------------
# 4. Build FAISS vector index
# -----------------------------
doc_embeddings = embed_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


# -----------------------------
# 5. RAG functions with Phoenix spans
# -----------------------------
def retrieve(query: str, top_k: int = 3):
    with tracer.start_as_current_span("retrieval") as span:
        query_embedding = embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = index.search(query_embedding, top_k)

        retrieved_docs = [documents[i] for i in indices[0]]
        scores = distances[0].tolist()

        span.set_attribute("query", query)
        span.set_attribute("retrieved_chunks", str(retrieved_docs))
        span.set_attribute("retrieval_scores", str(scores))

        return retrieved_docs, scores


def generate_answer(query: str, context_chunks: list[str]):
    with tracer.start_as_current_span("llm_generation") as span:
        context = "\n".join(context_chunks)

        prompt = f"""
You are a helpful library assistant.
Answer only using the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer only from the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        span.set_attribute("context", context)
        span.set_attribute("answer", answer)

        return answer


def simple_groundedness_eval(answer: str, context_chunks: list[str]):
    with tracer.start_as_current_span("groundedness_eval") as span:
        context = "\n".join(context_chunks)

        eval_prompt = f"""
You are checking whether an answer is grounded in the context.

Context:
{context}

Answer:
{answer}

Return only one of:
GROUNDED
NOT_GROUNDED
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
        )

        verdict = response.choices[0].message.content.strip()

        span.set_attribute("eval_verdict", verdict)

        return verdict


def rag_chat(query: str):
    with tracer.start_as_current_span("rag_chat_trace") as span:
        span.set_attribute("user_query", query)

        chunks, scores = retrieve(query)
        answer = generate_answer(query, chunks)
        verdict = simple_groundedness_eval(answer, chunks)

        span.set_attribute("final_answer", answer)
        span.set_attribute("groundedness", verdict)

        return {
            "query": query,
            "retrieved_chunks": chunks,
            "answer": answer,
            "groundedness": verdict,
        }


# -----------------------------
# 6. Trial query
# -----------------------------
if __name__ == "__main__":
    result = rag_chat("Where can I read e-books?")

    print("\nQuestion:", result["query"])
    print("\nRetrieved chunks:")
    for chunk in result["retrieved_chunks"]:
        print("-", chunk)

    print("\nAnswer:", result["answer"])
    print("\nGroundedness:", result["groundedness"])