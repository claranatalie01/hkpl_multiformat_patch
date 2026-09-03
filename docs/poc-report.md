# HKPL Agentic RAG PoC — high-level overview

**Source repository:**
[claranatalie01/hkpl_multiformat_patch](https://github.com/claranatalie01/hkpl_multiformat_patch)

## Objective and verdict

This PoC demonstrates an internally hosted, multilingual RAG service for public
HKPL webpages and documents, covering ingestion, retrieval, reranking, grounded
generation, input safety, evaluation, and observability. It does not yet prove
production capacity, policy acceptance, patron integration, or access
isolation.

**Verdict:** use Qwen3 Embedding 0.6B Q8_0, dense top 10, Qwen3 Reranker
0.6B Q8_0 top 5, and Qwen3.5-9B Q6_K generation with reasoning disabled. It
provides the best tested balance of grounding, cost, and latency.

## Implemented workflow

```text
approved HKPL webpage, PDF, or upload
  -> register/hash/version source
  -> deterministic reader or Docling/OCR extraction
  -> structure-aware chunks (512 tokens; 64-token prose overlap)
  -> local 1,024-dimensional embeddings
  -> PostgreSQL/pgvector
  -> input rules + GLiGuard
  -> dense top 10 -> rerank top 5 -> bounded context
  -> Qwen3.5 answer -> grounding/output checks + citations
  -> OpenTelemetry/Phoenix + offline evaluation
```

Structure-aware chunking preserves atomic records and follows headings for
prose. Exact evidence and stable source/version/locator IDs support evaluation
and citations. The answer LLM does not create vectors; pgvector is used instead
of FAISS. LangGraph is bounded and has no write-capable model-selected tools.

## Evaluation evidence

- Corpus: 16,220 chunks — 2,668 HKPL primary, 9,769 HotpotQA distractor, and
  3,783 Webz News distractor chunks.
- Benchmark: 128 questions with expected answers, evidence snippets, and stable
  chunk IDs; human-review status must accompany a release.
- Generation: eligible HKPL chunks are grouped by document; a local LLM sees
  anchor/sibling evidence, after which deterministic checks, deduplication, and
  human review precede promotion.
- Control: the Jina table retained every text, metadata record, and node ID and
  changed only embeddings. Questions, generation, reasoning-off policy, and
  top-10/top-5 budgets were otherwise held constant.
- Runtime: local llama.cpp/GGUF on two RTX 2080 Ti GPUs; reported latency is
  single-run PoC evidence, not a concurrent-capacity result.

## RAG experiment results

QE = Qwen embedding; QR = Qwen reranker; JE5 = Jina v5 embedding; JR = Jina
reranker. All model variants shown are Q8_0.

| Configuration | Retrieval Hit@10 | Reranker Hit@5 | Answer pass | Correctness /5 | Faithfulness | Pipeline tokens | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **QE + dense + QR** | **90.63%** | **83.59%** | 87.50% | 4.578 | **97.66%** | **3,782.9** | 1.93 / 3.05 s |
| QE + hybrid + QR | 89.06% | 82.03% | 87.50% | 4.563 | 96.09% | 4,420.2 | 5.72 / 7.38 s |
| QE + dense + JR3 | **90.63%** | 82.03% | **89.84%** | **4.625** | **97.66%** | 3,953.3 | 8.30 / 13.09 s |
| QE + dense + JR3.5 | **90.63%** | 80.47% | 89.06% | **4.625** | 96.09% | 3,926.7 | **1.87 / 2.94 s** |
| JE5 + dense + QR | 87.50% | 81.25% | 88.28% | 4.539 | 93.75% | 4,015.9 | 2.00 / 3.16 s |
| JE5 + dense + JR3.5 | 87.50% | 79.69% | 85.94% | 4.477 | 92.19% | 4,010.5 | 1.90 / 3.04 s |

Embedding-query usage was 25.7 tokens and answer completions averaged 49-53
tokens in every run; reasoning usage was zero.

Hybrid used 16.8% more tokens and increased p50 latency by 196% without an
answer gain. Qwen embeddings retrieved four more labelled chunks than Jina v5.
Jina v3 had the most passing answers but was 4.3 times slower. Jina v3.5 is the
closest challenger, but its two-answer gain came with four fewer reranked hits
and lower faithfulness. The all-Jina stack was weaker. **QE + dense + QR
therefore remains the baseline**, pending repeated E1/E4 runs and manual review
of their disagreements.

## Input-safety experiment

WildGuardTest contained 1,725 rows: 1,699 scorable (945 safe, 754 unsafe) and 26
without labels. All policies completed every scorable row.

| Policy | Accuracy | Unsafe recall | Unsafe F1 | FPR / FNR | Mean latency |
|---|---:|---:|---:|---:|---:|
| **GLiGuard** | 82.52% | **86.60%** | 81.47% | 20.74% / **13.40%** | **16.7 ms** |
| Qwen3Guard loose | 88.70% | 80.50% | 86.34% | **4.76%** / 19.50% | 248.3 ms |
| Qwen3Guard strict | **90.23%** | 85.54% | **88.60%** | 6.03% / 14.46% | 248.3 ms |

GLiGuard is the latency-first PoC choice: it was about 15 times faster and had
the highest unsafe recall. Its 20.74% false-positive rate requires deterministic
rules plus HKPL-specific multilingual and jailbreak calibration before use.

## Limitations and next actions

1. Version results with the dataset, corpus fingerprint, configuration, models,
   Git commit, and Phoenix project.
2. Reject runs with connection warnings, `reranker.failed=true`, or
   `fallback_vector_score`; these indicate dense-order fallback.
3. Repeat E1/E4 and manually review retrieval, reranking, and answer
   disagreements.
4. Add multilingual, ambiguity, freshness, unanswerable, injection, citation,
   and restricted-access tests; calibrate thresholds on held-out data.
5. Complete security, source governance, model-license review,
   failure-recovery, and 100-request capacity testing; confirm the contractual
   requirements version before production acceptance.

Implementation and reproduction details are maintained in
[architecture.md](architecture.md), [ingestion.md](ingestion.md),
[evaluation.md](evaluation.md), [operations.md](operations.md), and
[development.md](development.md).
