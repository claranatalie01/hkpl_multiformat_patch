# HKPL RAG Data and Evaluation Documentation

**Document status:** Current working reference  
**Last code/documentation audit:** 7 August 2026  
**Scope:** HKPL knowledge corpus, distractor corpora, evaluation dataset,
evaluation outputs, metrics, and data-governance controls.

## 1. Purpose

This document explains what data the HKPL RAG system uses, how the evaluation
dataset was produced, how each metric is calculated, and how benchmark results
should be interpreted. It is the data reference for reproducing or reviewing
the current benchmark.

The benchmark measures four distinct stages:

1. whether dense retrieval finds labelled or equivalent evidence;
2. whether reranking preserves that evidence;
3. whether the LLM generates a correct, relevant, and grounded answer;
4. how many tokens and how much latency the RAG pipeline uses.

## 2. Data inventory

All searchable content is stored in one PGVector table:

```text
data_hkpl_knowledge
```

The `metadata_->>'dataset'` value identifies the logical corpus inside that
table. An absent or empty dataset value is treated as `hkpl`.

### Recorded benchmark corpus snapshot

| Logical corpus | Role | Vectors |
|---|---|---:|
| `hkpl` | Primary knowledge used to answer questions | 2,147 |
| `hotpotqa` | Retrieval distractor/noise | 9,769 |
| `webz_news` | Retrieval distractor/noise | 3,783 |
| **Total** |  | **15,699** |

These counts describe the corpus reported during the current 528-question
benchmark. Always run the status command before a new official evaluation and
record any changed counts.

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py status
```

### Corpus roles

- `hkpl` contains authoritative HKPL pages and uploaded HKPL material.
- `hotpotqa` and `webz_news` are not sources of evaluation questions.
- Distractors exist only to test whether retrieval remains focused in a noisy,
  shared vector table.
- An HKPL benchmark row must link to an HKPL primary chunk, never a distractor.

## 3. Data lineage

The required data flow is:

```text
Authoritative HKPL documents
  → extraction and structure-aware chunking
  → embeddings in data_hkpl_knowledge
  → corpus audit
  → candidate questions generated from eligible HKPL chunks
  → automated evidence validation
  → manual semantic review
  → active evaluation_dataset.csv
  → HotpotQA and Webz distractors in the shared table
  → RAG evaluation and Phoenix traces
```

Evaluation questions are generated **after** the documents have been ingested.
This makes every question traceable to evidence that existed in the searchable
corpus at generation time.

### Current benchmark lineage

The current benchmark was produced through this progression:

```text
2,141 HKPL vectors at initial candidate-generation time
  → 1,085 eligible chunks selected
  → Qwen generated 540 candidate rows
  → validation and manual cleanup
  → 532 active rows
  → removal of later unresolved/stale labels
  → 528 evidence-valid active questions
```

The later recorded corpus contains 2,147 HKPL vectors. The difference from the
initial 2,141-vector generation snapshot must not be interpreted as six new
evaluation questions; vectors and evaluation questions are not one-to-one.

### Why 2,147 vectors do not produce 2,147 questions

A vector is a searchable chunk, not an evaluation item. Candidate generation
uses only chunks that satisfy the configured eligibility rules:

- dataset is `hkpl`;
- corpus role is `primary`;
- trimmed chunk text contains at least 120 characters;
- default generation selects at most eight chunks per document;
- only the first 1,800 characters of an eligible chunk are supplied to the
  question generator;
- the question generator may return no valid question for navigation text,
  duplicate text, incomplete fragments, lists without a stable fact, or
  otherwise unsuitable content;
- duplicate, ambiguous, time-sensitive, stale, or unsupported candidates are
  removed during validation and review.

`--all-chunks` removes the eight-chunks-per-document cap, but it does not make
every chunk suitable for a benchmark question.

## 4. Evaluation dataset

### Authoritative representations

| Representation | Location | Purpose |
|---|---|---|
| Active CSV | `data/evaluation_dataset.csv` | Reviewed, portable benchmark source |
| Active PostgreSQL table | `evaluation_dataset` | Runtime copy used by evaluation |
| Candidate CSV | `data/evaluation_dataset.candidate.csv` | Generated rows awaiting review |
| Candidate PostgreSQL table | `evaluation_dataset_100` | Validation copy of the 100-row candidate |

The active CSV and table should contain the same 528 questions before running
the current benchmark.

### CSV data dictionary

The CSV columns must appear in this exact order:

| Column | Required | Meaning |
|---|:---:|---|
| `domain` | Yes | Reporting category such as `events`, `e_resources`, or `opening_hours` |
| `query` | Yes | Unique user-style evaluation question |
| `expected_answer_text` | Yes | Primary concise and complete reference answer |
| `expected_context_snippet` | Yes | Exact contiguous evidence passage from the linked chunk |
| `accepted_answers_json` | Yes | JSON array containing additional fully equivalent answers |
| `source_title` | Yes | Human-readable title of the authoritative source |
| `source_url` | No | Source URL when one is available |
| `source_document_id` | Yes | Stable UUID identifying the source document |
| `source_chunk_id` | Yes | Exact versioned chunk containing the labelled evidence |

Example:

```csv
events,On what date did the Hong Kong Central Library open to the public?,17 May 2001,The Library was open to the public on 17 May 2001,[],Hong Kong Public Libraries - Introduction,https://www.hkpl.gov.hk/en/about-us/intro/intro.html,d85949f9-6adc-4527-b59c-e0067a469a9d,d85949f9-6adc-4527-b59c-e0067a469a9d:v7:s0:c4:834858bf092a43d1
```

### Meaning of `accepted_answers_json=[]`

`[]` means there are no additional accepted aliases. It does **not** mean that
the row has no expected answer. The evaluator still uses
`expected_answer_text` as the primary reference answer.

Only add an alias when it is completely equivalent. Do not add partial answers
merely to increase the pass rate. For example, `May 2001` must not be accepted
for a question whose required answer is `17 May 2001`.

### Why document and chunk IDs are necessary

The answer alone can measure final-answer similarity, but it cannot determine
where a pipeline failure occurred.

- `source_document_id` supplies document-level provenance and allows a stale
  chunk to be searched within the correct source document.
- `source_chunk_id` identifies the exact evidence unit expected from retrieval.
- Together with `expected_context_snippet`, they distinguish retrieval,
  reranking, context-building, generation, and evidence-label failures.

A chunk ID is versioned. When a document is replaced or reingested, an old
chunk such as `:v11:...` may be deleted and replaced by `:v12:...`. A benchmark
still pointing at the deleted version becomes stale even when similar text
exists elsewhere.

## 5. Dataset validation

Run validation before every official evaluation:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/normalize_evaluation_schema.py \
  --path /app/data/evaluation_dataset.csv \
  --check

docker compose run --rm langgraph-agent \
  uv run python scripts/validate_evaluation_dataset.py
```

The expected current state is:

```text
Evaluation rows:       528
Expected chunk found:  528/528
Snippet text found:    528/528
Evaluation status:     READY
```

Validation fields mean:

- **Expected chunk found:** the exact `source_chunk_id` exists in
  `data_hkpl_knowledge`.
- **Snippet text found:** the linked chunk contains the labelled evidence
  snippet, allowing formatting-tolerant matching.
- **Answer verbatim found:** the complete expected answer happens to occur as
  one contiguous string. This is informational because a valid answer can
  combine facts or paraphrase the evidence.
- **READY:** every row has an existing chunk and its evidence snippet is found.

`Answer verbatim found` is not a readiness requirement and is not an answer
correctness score.

## 6. Retrieval and generation configuration

| Stage | Current configuration |
|---|---|
| Embedding | Qwen3-Embedding-0.6B GGUF Q8_0, 1,024 dimensions |
| Vector retrieval | LlamaIndex with PGVector, top 10 candidates |
| Reranking | Qwen3-Reranker-0.6B GGUF Q8_0, final top 5 |
| Live-answer confidence threshold | 0.30; used by the chat graph, not `evaluate_rag.py` |
| Answer model | Qwen3.5-9B GGUF Q6_K |
| LLM context window | 32,768 tokens |
| Evaluation output limit | 2,048 completion tokens |
| Normal answer mode | Reasoning disabled per request |
| Selective retry mode | Up to 1,000 reasoning tokens per answer call |
| Evaluators | LlamaIndex correctness, faithfulness, and relevancy using Qwen3.5-9B |
| Parallel LLM slots | 1 |

Although llama.cpp is started with reasoning support enabled, normal requests
send a per-request thinking budget of zero. Reasoning is used only when the
evaluation command includes `--answer-reasoning`.

The evaluator uses the same model family and endpoint as the answer generator.
This is economical but not an independent judge. Material score changes and
borderline answers should therefore receive manual review, or later be checked
with a stronger independent evaluator.

## 7. Primary metrics

### Summary metrics

| Metric | Definition | Direction |
|---|---|:---:|
| `evaluated_questions` | Number of benchmark rows processed | Context |
| `retriever_at_10` | Fraction where the labelled chunk or recognized equivalent evidence appears in the ten vector candidates | Higher |
| `reranker_at_5` | Fraction where the labelled chunk or recognized equivalent evidence remains in the final five chunks | Higher |
| `answer_pass_rate` | Fraction passing correctness, relevancy, faithfulness, and evaluator reliability | Higher |
| `average_correctness` | Mean judge score on a 1–5 scale | Higher |
| `average_faithfulness` | Mean groundedness verdict, normally 0 or 1 | Higher |
| `rag_p50` | Median retrieval-through-answer latency; evaluator latency excluded | Lower |
| `rag_p95` | 95th-percentile retrieval-through-answer latency; evaluator latency excluded | Lower |

### Evidence-hit matching

A retrieval or reranking hit is recognized when either:

1. the exact labelled `source_chunk_id` is present; or
2. a retrieved chunk contains the normalized expected context snippet or a
   sufficiently meaningful accepted-answer phrase.

The second rule allows equivalent current evidence to count even if the exact
labelled chunk was not retrieved. It is still string-based evidence
recognition, not a full semantic entailment evaluator.

### Answer pass rule

An answer passes only when all conditions are true:

```text
correctness >= 4.0 out of 5
faithfulness >= 0.5
relevancy >= 0.5
no evaluator failure
```

The outcome labels are:

| Outcome | Meaning | Pass? |
|---|---|:---:|
| `correct` | Complete, relevant, and grounded | Yes |
| `partial_or_ambiguous` | Required detail missing or ambiguity unresolved | No |
| `incorrect` | Core answer incorrect | No |
| `ungrounded` | One or more claims unsupported by retrieved context | No |
| `irrelevant` | Does not adequately answer the question | No |
| `not_evaluated` | One or more evaluators failed | No |

### Diagnosis logic

```text
Did retrieval find labelled/equivalent evidence?
├── No + answer failed → retrieval_problem
├── No + answer passed → correct_answer_evidence_label_miss
└── Yes
    └── Did reranking preserve the evidence?
        ├── No + answer failed → reranker_problem
        ├── No + answer passed → correct_answer_reranker_evidence_miss
        └── Yes
            ├── evidence missing from built context → context_building_problem
            ├── correctness < 4 → llm_generation_problem
            ├── relevancy < 0.5 → irrelevant_answer
            ├── faithfulness < 0.5 → ungrounded_answer
            └── all checks pass → working_correctly
```

`answer_pass_rate` and `working_correctly_rate` are intentionally different.
A correct grounded answer may pass even when the benchmark's specific evidence
label was missed.

## 8. Token accounting

| Field | Meaning |
|---|---|
| `embedding_query_tokens` | Tokens used to embed the query |
| `reranker_input_tokens` | Query-plus-document tokens submitted to the reranker |
| `context_tokens` | Tokens in the final context supplied to the answer model |
| `llm_prompt_tokens` | Complete answer-generation prompt tokens |
| `llm_completion_tokens` | All answer-model completion tokens, including reasoning when enabled |
| `llm_reasoning_tokens` | Reasoning portion of the completion when exposed or countable |
| `llm_total_tokens` | Answer-model prompt plus completion tokens |
| `pipeline_total_tokens` | Embedding query + reranker input + answer-model total |

Evaluator tokens are not included in `pipeline_total_tokens`. Consequently,
this field describes the RAG answer pipeline, not the total compute consumed by
the three LlamaIndex judging calls.

## 9. Benchmark results

### Full 528-question baseline: reasoning off

| Metric | Result |
|---|---:|
| Evaluated questions | 528 |
| Retriever Hit@10 | 0.9432 (498/528) |
| Reranker Hit@5 | 0.9223 (487/528) |
| Answer pass rate | 0.8352 (441/528) |
| Average correctness | 4.6345/5 |
| Average faithfulness | 0.9621 |
| RAG p50 latency | 2.1807 seconds |
| RAG p95 latency | 4.6980 seconds |
| Average pipeline tokens | 2,677.47 |
| Average LLM total tokens | 1,037.93 |

Interpretation: retrieval and grounding are strong, but 87 answers failed at
least one answer-pass condition. The Phoenix project contains 529 traces
because the run records 528 question traces plus one aggregate summary trace.
Phoenix root-span latency includes evaluation work, while `rag_p50` and
`rag_p95` exclude the evaluator calls.

### Selective retry: reasoning up to 1,000 tokens

Only the 87 baseline rows with `answer_pass=false` were rerun. Generator
reasoning was enabled; evaluator reasoning remained off.

| Metric | Result |
|---|---:|
| Retried questions | 87 |
| Recovered answers | 21 |
| Retry pass rate | 0.2414 (21/87) |
| Remaining failed answers | 66 |
| Projected combined passes | 462/528 |
| Projected combined pass rate | 0.8750 |
| Absolute projected improvement | 3.98 percentage points |
| Average reasoning tokens | 977.64 |
| Average pipeline tokens | 3,720.89 |
| RAG p50 latency | 18.5574 seconds |
| RAG p95 latency | 33.7300 seconds |

The retry subset is intentionally difficult, so its 73.56% Retriever Hit@10,
64.37% Reranker Hit@5, and 24.14% answer pass rate must not be compared as if
they were another random 528-question benchmark.

The 87.50% result is a **projected combined result**, not a fresh independent
full benchmark. It uses benchmark labels to identify failures after the first
run. A production system does not have expected answers, so deployment requires
a runtime routing rule based on signals available at inference time.

## 10. Result artifacts

| Artifact | Contents |
|---|---|
| `results.csv` | Per-question retrieval, generation, evaluator, token, latency, and diagnosis fields |
| `summary.json` | Concise primary benchmark metrics |
| `summary.diagnostics.json` | Diagnosis counts, answer outcomes, per-domain results, robustness, reliability, and corpus snapshot |
| Phoenix project | Trace hierarchy, retrieved documents, reranked context, LLM calls, and annotations |

The protected no-reasoning baseline should be retained on the remote benchmark
host or approved artifact storage as:

```text
data/rag_evaluation/results.no-reasoning.528.csv
```

Generated evaluation reports are ignored by Git and are not bundled with a
clean clone. The benchmark tables below record a completed run; reproduce or
retrieve its artifacts from the remote benchmark host when row-level evidence
is required.

The selective reasoning retry is written separately using a tagged filename,
for example:

```text
data/rag_evaluation/results.hkpl.answer-failures.answer-reasoning-1000.csv
data/rag_evaluation/summary.hkpl.answer-failures.answer-reasoning-1000.json
data/rag_evaluation/summary.hkpl.answer-failures.answer-reasoning-1000.diagnostics.json
```

## 11. Reproducible operating procedure

### Confirm that the corpus is frozen

The application-level guard is configured through:

```text
KNOWLEDGE_CORPUS_READ_ONLY=true
```

Confirm the database-level lock as well:

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/manage_corpus_lock.py --status
```

Expected status:

```text
Database corpus lock: ENABLED (READ ONLY)
```

The database lock protects both `knowledge_documents` and
`data_hkpl_knowledge` against insert, update, delete, and truncate operations.
The evaluation CSV and evaluation table are separate and can still be changed
by benchmark-management commands.

To intentionally mutate the knowledge corpus, both locks must be addressed:
disable the database lock with `scripts/manage_corpus_lock.py --disable --yes`
and run the specific write command with
`KNOWLEDGE_CORPUS_READ_ONLY=false`. Re-enable the database lock immediately
afterward. The full maintenance sequence is documented in
`IMPLEMENTATION_GUIDE.md`.

### Run the normal full benchmark

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py \
  evaluate \
  --phoenix-project hkpl-no-reasoning-full-528-v1
```

### Preserve the baseline

```bash
cp data/rag_evaluation/results.csv \
  data/rag_evaluation/results.no-reasoning.528.csv
```

### Rerun only failed answers with reasoning

```bash
docker compose run --rm langgraph-agent \
  uv run python scripts/rag_benchmark_workflow.py \
  evaluate \
  --rerun-answer-failures-from \
  /app/data/rag_evaluation/results.no-reasoning.528.csv \
  --answer-reasoning \
  --reasoning-budget 1000 \
  --phoenix-project hkpl-failed-87-reasoning-1000
```

Do not add `--evaluator-reasoning` when comparing against the no-reasoning
baseline. Changing both the answer model behavior and judge behavior would make
the cause of a score change unclear.

## 12. Data ownership and filesystem permissions

`docker compose run` currently executes as root inside the container. Files
created in the bind-mounted `data/rag_evaluation` directory can therefore be
owned by `root` on Ubuntu.

If the host user cannot copy or manage the results and has no `sudo` access,
repair ownership through Docker:

```bash
docker compose run --rm --no-deps \
  --entrypoint sh langgraph-agent \
  -c "chown -R $(id -u):$(id -g) /app/data/rag_evaluation"
```

This changes filesystem ownership only. It does not modify result contents,
the evaluation dataset, or the vector corpus.

## 13. Data-quality and interpretation limitations

- Candidate questions are generated by an LLM and require human semantic
  review even when schema and evidence validation pass.
- Event dates, opening hours, fees, policies, and service availability are
  time-sensitive. A frozen benchmark measures the frozen corpus snapshot, not
  necessarily the current HKPL website.
- `source_chunk_id` is version-specific; intentional corpus refreshes require
  benchmark revalidation and possibly regeneration.
- Evidence-equivalence detection is normalized string matching, so some true
  semantic alternatives may still be reported as evidence-label misses.
- The generator and evaluator currently use Qwen3.5-9B. This can create shared
  model bias and is weaker than evaluation with an independent judge.
- Small domains cannot support reliable domain-wide conclusions.
- The selective reasoning result is diagnostic. It is not a deployable routing
  policy until failure detection uses only runtime-available signals.
- Reasoning consumed nearly the entire 1,000-token budget and increased median
  RAG latency substantially; quality gains must be weighed against this cost.

## 14. Recommended next work

1. Manually review the 21 reasoning-recovered answers for completeness and
   grounding.
2. Inspect the 66 remaining failures using `diagnosis_counts` and the per-row
   result CSV.
3. Improve retrieval for `retrieval_problem` cases rather than applying more
   generation reasoning to missing evidence.
4. Improve reranking for `reranker_problem` cases.
5. Use reasoning primarily for cases where useful evidence reaches the LLM but
   generation is partial, ambiguous, or irrelevant.
6. Test a 512-token reasoning budget to measure whether most of the 21
   recoveries can be retained with lower latency.
7. Define a production-time confidence trigger, then evaluate that routing
   policy across all 528 questions without using expected answers to select
   retries.
8. Record a corpus snapshot identifier, vector counts, code commit, model
   versions, benchmark checksum, and configuration for every official run.
