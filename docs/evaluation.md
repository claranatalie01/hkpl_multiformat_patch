# Evaluation datasets and Phoenix

## What is evaluated

Evaluation separates the RAG stages:

| Metric | Question it answers |
|---|---|
| Retriever Hit@K | Did the vector/hybrid candidate pool contain labelled or equivalent evidence? |
| Reranker Hit@N | Did that evidence survive reranking into the final candidate set? |
| Distractor Rate@N | How much final context came from configured distractor corpora? |
| Relevancy | Does the response directly answer the query using the combined context? |
| Faithfulness | Are answer claims supported by the context actually given to the LLM? |
| Q&A Correctness | Does the answer agree with the reviewed reference answer? |
| RAG Diagnosis | Which stage best explains the combined outcome? |

Faithfulness can be `1` while Retriever Hit is `0`: the answer may be supported
by an alternative retrieved chunk even though the benchmark's labelled chunk
or accepted evidence was not detected. That result is a benchmark-label or
evidence-matching issue, not necessarily hallucination.

## Generate a candidate dataset

Generation reads existing vector chunks; it does not crawl or add vectors.
Only `dataset=hkpl`, `corpus_role=primary` chunks are eligible by default.
Short, duplicate, ambiguous, or unsupported candidates can be rejected.

```text
pgvector HKPL chunks
    -> eligibility filter
    -> group by source document
    -> choose anchor and relevant sibling evidence
    -> local LLM proposes question and answer
    -> deterministic evidence and schema checks
    -> deduplicate and checkpoint
    -> candidate CSV
    -> validate every evidence ID against pgvector
```

Generate exactly 100 rows:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py prepare-candidate \
  --output data/evaluation_dataset_100.csv \
  --target-questions 100 \
  --all-chunks
```

The generator can use multiple chunks from one document for one comprehensive
question. Evidence snippets and source chunk IDs must be parallel arrays: item
`n` in `expected_context_snippets_json` must be supported by item `n` in
`source_chunk_ids_json`.

Validate without modifying the file:

```bash
docker compose run --rm \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py validate-candidate \
  --candidate data/evaluation_dataset_100.csv
```

Review candidate rows before promotion. LLM-generated questions are useful test
seeds, not automatically credible ground truth. A human should verify the
question is unambiguous, the answer is complete, dates/venues are scoped, and
every evidence reference is correct.

## Run evaluation in Phoenix

Run a smoke test against an explicit evaluation and vector table:

```bash
docker compose run --rm \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_100 \
  -e VECTOR_TABLE=hkpl_knowledge \
  -e RAG_EVALUATION_RESULTS_PATH=/app/data/rag_evaluation/smoke.csv \
  -e RAG_EVALUATION_SUMMARY_PATH=/app/data/rag_evaluation/smoke.json \
  langgraph-agent \
  python scripts/rag_benchmark_workflow.py evaluate \
  --limit 3 \
  --phoenix-project hkpl-rag-smoke
```

Run one exact question. Use shell-safe outer double quotes when the question
contains an apostrophe:

```bash
docker compose run --rm \
  -e EVALUATION_DATASET_TABLE=evaluation_dataset_100 \
  -e VECTOR_TABLE=hkpl_knowledge \
  langgraph-agent \
  python scripts/evaluate_rag.py \
  --question-exact "Which collection lists children's literature?" \
  --phoenix-project hkpl-rag-debug
```

For the hybrid table, use:

```bash
-e VECTOR_TABLE=hkpl_knowledge_hybrid
```

This setting must be supplied to the one-off `docker compose run` command. A
running agent container's environment does not automatically change the
environment of an already issued command.

## Answer reasoning and token budgets

Ordinary RAG should use deterministic answer generation with reasoning off.
Evaluation can enable bounded reasoning for diagnosis:

```bash
python scripts/rag_benchmark_workflow.py evaluate \
  --answer-reasoning \
  --reasoning-budget 500
```

The evaluator adds the thinking budget to the visible answer budget so hidden
reasoning does not consume all output space. Larger budgets cost latency and do
not guarantee better factual selection; retrieval and context quality still
control what the model can answer.

## Where results are stored

- The selected evaluation questions are read from the SQL table named by
  `EVALUATION_DATASET_TABLE`.
- Per-row metrics go to `RAG_EVALUATION_RESULTS_PATH`.
- Aggregate metrics go to `RAG_EVALUATION_SUMMARY_PATH`.
- Traces and annotations go to the selected Phoenix project.
- Generated CSV/JSON results under `data/rag_evaluation` are ignored by Git.
