# HKPL Agentic RAG proof of concept

## Purpose and scope

This proof of concept (PoC) tests whether an internally hosted, multilingual
RAG service can answer public questions from approved Hong Kong Public
Libraries webpages and documents with traceable evidence, acceptable latency,
and an input safety guard. It covers ingestion, retrieval, reranking, grounded
generation, evaluation, and development observability. It does not prove
production capacity, patron-system integration, or formal policy acceptance.

The requirements filename refers to Version 1.2 while its cover refers to
Version 1.1 (June 2026); the contractual version remains to be confirmed.

## Implemented pipeline

```text
approved HKPL webpage, PDF, or upload
        |
        v
register + hash + save original source
        |
        v
deterministic reader or Docling extraction/OCR
        |
        v
structure-aware, token-bounded chunking
        |
        v
Qwen3-Embedding-0.6B (1,024 dimensions)
        |
        v
PostgreSQL 16 + pgvector
        |
user -> prohibited rules -> GLiGuard -> intent/query rewrite
        |
        v
dense retrieval (top 10) -> Qwen3-Reranker-0.6B (top 5)
        |
        v
context pack -> Qwen3.5-9B Q6_K -> output check + citations
        |
        v
OpenTelemetry/Phoenix traces and offline evaluation
```

The online LangGraph is bounded; it has no open-ended planning loop or
write-capable model-selected tools. PostgreSQL/pgvector is the vector database;
FAISS is not used. Phoenix is a development and evaluation viewer rather than
the production audit record.

## Ingestion and chunking

The crawler and protected administration endpoints both delegate to the same
ingestion service. The service registers the source, selects a reader, creates
chunks, embeds their retrieval text, writes vectors, and updates document
status. A failed run is recorded as failed instead of being reported complete.

Readers preserve structure before applying a size limit:

| Source | Current treatment |
|---|---|
| HKPL HTML | Special readers preserve events, opening-hour rows, branch profiles, forms, and FAQs; other pages use structured fallback extraction. |
| PDF, DOCX, PPTX, Markdown, text, images | Docling extraction; OCR is available for image-only content. |
| CSV/XLSX | Deterministic row records with headers and row locators. |
| JSON/JSONL/XML | Deterministic record extraction. |

An explicit `faq`, `record`, or `prose` document type controls the structural
policy. With `auto`, a bounded non-reasoning classifier selects `faq`,
`record`, `prose`, or `skip`, and the decision is saved before final parsing.

Final chunks use the embedding tokenizer and the current limits:

```env
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

- FAQ pairs, event/notice records, branch records, and table rows are kept
  atomic where they fit.
- Prose follows headings and Docling leaf structure.
- Oversized evidence is split at paragraph, list, or sentence boundaries
  first; raw tokenizer offsets are the fallback.
- Token fallback uses 64-token overlap, except atomic table rows use no
  overlap so values are not duplicated across rows.
- `evidence_text` preserves the exact text used for answers and citations.
- `search_text` adds useful title, heading, alias, or repeated-header context
  and is the text sent to the embedding model.
- Chunk IDs include the source version, locator hash, part number, and evidence
  hash, making provenance and changed content visible.

Changing chunking code does not change stored vectors. The intended corpus
must be reindexed or reingested. See [ingestion.md](ingestion.md) for commands
and SQL inspection.

## Evaluation-dataset construction

`generate_evaluation_dataset.py` creates candidate labels from existing chunks;
it does not crawl, chunk, embed, or alter the vector corpus.

```text
configured vector table or one preview run
        |
        v
select HKPL primary chunks with at least 120 characters
        |
        v
group chunks by source document
        |
        v
anchor + available sibling chunks (first 1,800 characters each)
        |
        v
local LLM proposes zero or one question, answer, and evidence list
        |
        v
schema, chunk-ID, exact-snippet, completeness, and ambiguity checks
        |
        v
question/evidence deduplication + resumable checkpoint
        |
        v
candidate CSV -> human review -> validation -> SQL promotion
```

Only `dataset=hkpl`, `corpus_role=primary` vector rows are eligible by default.
Without `--all-chunks`, at most eight chunks per document are considered. Each
LLM prompt receives an anchor and sibling chunks from the same document, but a
document is not limited to one question. Evidence chunks cited by an accepted
row are reserved and cannot later become anchors in the same case/language
slice; siblings that were not cited remain eligible.

The generator uses temperature zero, reasoning off, and a strict JSON schema.
It rejects missing fields, unavailable chunk IDs, non-contiguous evidence,
non-parallel snippet/ID arrays, and multi-evidence cases with too little
evidence. For repeated named events, an unscoped question must cite every
matching sibling occurrence; a one-session question must identify its date or
month and branch/venue. Normalized duplicate questions keep the most complete
consistent row, conflicting answers are removed, and reused evidence is
removed within the same evaluation slice.

The target count is applied after validation and deduplication, so more than
100 anchors may be processed to obtain 100 rows. The LLM output is only a
candidate: a reviewer must confirm that the question is useful and
unambiguous, the answer is complete, and all source labels are correct. See
[evaluation.md](evaluation.md) for the complete rules and commands.

## RAG experiment

The 128-row benchmark was evaluated against the same pgvector corpus with
reasoning disabled. Its human-review status should be recorded with the release
artifact. The promoted PoC profile is:

```env
RETRIEVAL_MODE=dense
DENSE_TOP_K=10
RERANK_TOP_N=5
MAX_CONTEXT_TOKENS=4000
ANSWER_MAX_TOKENS=256
```

| Metric | Broad hybrid | Optimized dense |
|---|---:|---:|
| Retriever Hit@10 | 89.06% | **90.63% (116/128)** |
| Reranker Hit@5 | 81.25% | **83.59% (107/128)** |
| Answer pass rate | 71.88% | **87.50% (112/128)** |
| Average correctness (out of 5) | **4.648** | 4.578 |
| Average faithfulness | 81.25% | **97.66%** |
| Average pipeline tokens | 8,054.7 | **3,782.9** |
| RAG p50 latency | 6.29 s | **1.93 s** |
| RAG p95 latency | 8.27 s | **3.05 s** |

The optimized dense configuration reduced median latency by about 69%, p95 by
about 63%, and pipeline tokens by about 53%. It also improved labelled retrieval,
reranking, answer pass rate, and faithfulness. Average correctness decreased by
0.070, so the result is not an improvement on every metric.

The observed improvement belongs to the complete optimized configuration, not
only the retrieval algorithm: candidate limits, context limits, Phoenix batch
export, and llama.cpp settings also changed. A controlled `10/10/10/5` hybrid
run is required to isolate dense-versus-hybrid causality.

## Guardrail experiment

The input-guard comparison used
[AllenAI WildGuardMix/WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix),
task `prompt_harm_label`.

| Dataset accounting | Count |
|---|---:|
| Total rows | 1,725 |
| Scorable rows | 1,699 |
| Missing gold labels | 26 |
| Safe prompts | 945 |
| Unsafe prompts | 754 |

Every policy processed all 1,699 scorable prompts with no parsing or runtime
errors.

| Model/policy | Accuracy | Unsafe precision | Unsafe recall | Unsafe F1 | FPR | FNR | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLiGuard | 82.52% | 76.91% | **86.60%** | 81.47% | 20.74% | **13.40%** | **16.7 ms** |
| Qwen3Guard loose | 88.70% | **93.10%** | 80.50% | 86.34% | **4.76%** | 19.50% | 248.3 ms |
| Qwen3Guard strict | **90.23%** | 91.88% | 85.54% | **88.60%** | 6.03% | 14.46% | 248.3 ms |

GLiGuard detected 653 unsafe prompts, missed 101, and incorrectly blocked 196
safe prompts. Qwen strict detected 645, missed 109, and incorrectly blocked 57.
The strict policy maps `Controversial` to unsafe; compared with loose mode it
caught 38 additional unsafe prompts at the cost of 12 additional false blocks.

### Guardrail decision

The latency-first PoC uses `fastino/gliguard-LLMGuardrails-300M` on the configured
GPU. It was about 15 times faster than Qwen3Guard and had the highest unsafe
recall. This is not a claim that it is the best classifier overall: Qwen strict
had materially better accuracy, F1, and false-positive rate. The absolute mean
latency saving was about 232 ms, not a 15-times speed-up of the complete RAG
pipeline.

GLiGuard's 20.74% false-positive rate is the main concern. It remains one layer
behind deterministic prohibited rules; selected high-risk categories block,
ambiguous categories are logged, and model/GPU failure fails closed. Before
production, thresholds and category mapping require HKPL-specific English,
Traditional Chinese, Simplified Chinese, Cantonese, mixed-language, jailbreak,
and benign-library validation. WildGuardTest alone does not represent HKPL
policy acceptance.

## What worked

- One shared ingestion service prevents crawler and administrator uploads from
  producing incompatible vectors.
- Structure-aware records and exact evidence metadata improve traceability.
- Dense top-10 retrieval gave the best tested speed/quality balance.
- Smaller reranker and context inputs reduced both latency and distraction.
- Phoenix exposed retriever, reranker, context, LLM, and judge behavior.
- GLiGuard provided low-latency, high-recall input screening with full test
  coverage.

## What did not work or remains incomplete

- Broad hybrid retrieval added latency and noise in the tested configuration.
- Dense retrieval missed labelled evidence for 12 questions.
- Reranking removed labelled evidence for nine questions that retrieval found.
- Sixteen answers did not pass the combined answer criteria.
- The generated benchmark can inherit LLM and corpus sampling bias and still
  requires human review.
- GLiGuard blocked about 21% of safe WildGuardTest prompts.
- The current repository does not contain the guardrail benchmark runner or
  immutable raw guardrail result artifact; these are required for independent
  reproduction.
- Production concurrency, access isolation, multilingual acceptance, source
  freshness, and human governance are not yet demonstrated.

## Recommendation and next steps

1. Keep dense retrieval as the PoC default and retain hybrid mode as an
   inactive experiment/fallback.
2. Review the 12 retriever misses, nine reranker losses, and 16 answer failures
   by diagnosis and language/domain.
3. Add exact identifier/title, misspelling, multilingual, unanswerable, stale,
   conflicting-source, and prompt-injection slices.
4. Version the evaluation CSV, corpus fingerprint, configuration snapshot,
   model revisions, Git commit, Phoenix project, and raw result files together.
5. Add the guardrail benchmark runner and machine-readable results to CI-owned
   evaluation artifacts.
6. Calibrate GLiGuard on an HKPL-specific set; retain Qwen strict as the quality
   comparator or optional adjudicator for uncertain cases.
7. Run concurrent capacity, security, citation-integrity, and human-review
   testing before any production recommendation.
