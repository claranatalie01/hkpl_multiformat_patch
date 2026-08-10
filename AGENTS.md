# HKPL RAG Agent Architecture

## Requirements baseline

This document defines the target architecture for the Hong Kong Public Libraries public RAG agent. The requirements source is `HKPL_AI_Chatbot_Proposal_v1.2.pdf`; its cover identifies itself as Version 1.1, June 2026, so the contractual version must be confirmed before formal acceptance.

The service answers questions from approved HKPL documents and webpages in English, Traditional Chinese, Simplified Chinese, Cantonese-style written Chinese, and mixed-language input. Answers are grounded in traceable evidence, fail safely when evidence is insufficient, and remain under human control.

### In scope

- Public document and webpage RAG.
- Multi-turn text conversations with short-lived session context.
- Branch-aware clarification and retrieval using a canonical branch registry.
- Approved-source ingestion, versioning, retrieval, citations, safety, evaluation, and production operations.
- Text received directly from a channel or from a trusted upstream transcription service.

### Deferred

- Management application and staff review UI.
- Live SLS catalogue, opening-hours, event, patron, reservation, or transaction APIs.
- MCP and model-selected tools.
- Speech-to-text and text-to-speech.
- Authenticated patron data, personalisation, notifications, and autonomous learning.

The agent contains no write-capable tools and never learns from patron conversations automatically. Future SLS integrations use typed, application-owned, read-only clients rather than arbitrary model-selected tools.

## Architectural principles

1. **Bounded workflow, not autonomous agency.** LangGraph expresses a finite, inspectable state machine with bounded retries and no open-ended planning loop.
2. **Evidence before generation.** HKPL facts come only from approved evidence supplied to the generator. Missing or weak evidence produces clarification or an approved fallback.
3. **Authorization before ranking.** Publication and access constraints are SQL predicates and database permissions, never prompt instructions or post-retrieval filters.
4. **Sources are untrusted data.** Retrieved text cannot issue instructions, select tools, alter policy, or obtain credentials.
5. **Last known good remains live.** Failed ingestion never removes or partially replaces an active version.
6. **Human-governed releases.** Models, prompts, policies, sources, and retrieval configuration are versioned, evaluated, and approved.
7. **Private and reproducible deployment.** Production performs no runtime model download and sends no patron content to public AI services.
8. **Minimal platform.** The system is a modular monolith plus an ingestion worker, model servers, PostgreSQL, object storage, and telemetry. It does not add a separate vector database, broker, MCP server, or retrieval microservice.

## System topology

```text
web/mobile/kiosk channel backend
        |
        v
LCSD gateway/WAF -- TLS, body limits, rate limits, channel identity
        |
        v
stateless FastAPI agent replicas
        |
        +-- bounded LangGraph workflow
        +-- read-only active_public_chunks view
        +-- private generation/embedding/reranking/guard endpoints
        +-- short-lived session store in PostgreSQL
        +-- metadata-only OpenTelemetry

approved upload or allowlisted webpage
        |
        v
immutable source object + PostgreSQL ingestion job
        |
        v
isolated ingestion worker
        |
        v
extract -> normalize -> chunk -> embed -> validate -> approve -> atomic activate
```

Deployable units are limited to `agent-api`, `ingestion-worker`, private model-serving processes, PostgreSQL with pgvector, approved object storage, and the LCSD-approved telemetry collector. Retrieval remains an in-process module of `agent-api`.

## Technology decisions

| Concern | Decision | Rationale |
|---|---|---|
| Application | Python 3.11, FastAPI, Pydantic | Provides mature, typed asynchronous HTTP contracts with a small operational footprint. |
| Orchestration | LangGraph with typed state | Makes routing and failure paths explicit without permitting autonomous behavior. |
| Persistence | PostgreSQL 16+, pgvector, pg_trgm | One transactional system supports metadata, sessions, versioning, dense retrieval, and lexical matching at the expected scale. |
| Database access | SQLAlchemy 2 async with asyncpg; Alembic migrations | One asynchronous access path and explicit, repeatable schema ownership. |
| Complex-document extraction | Docling | Produces structured content and provenance for PDF, DOCX, HTML, Markdown, and text. |
| Tabular extraction | Deterministic CSV/XLSX row parsing | Preserves sheet, row, headers, record boundaries, and exact values. |
| Generation baseline | Official Qwen3.5-9B | Strong multilingual PoC baseline that fits quantized local development. |
| Embedding baseline | Qwen3-Embedding-0.6B, 1024 dimensions | Multilingual, instruction-aware retrieval with practical local resource use. |
| Reranking baseline | Qwen3-Reranker-0.6B | Multilingual cross-encoder reranking aligned with the embedding family. |
| Semantic safety baseline | Qwen3Guard-0.6B behind a typed adapter | A small multilingual model whose policy mapping remains application-owned. |
| Local serving | llama.cpp-compatible pinned quantized artifacts | Supports the two available RTX 2080 Ti GPUs for development and PoC work. |
| Production serving | vLLM on benchmark-sized LCSD GPU capacity | Provides continuous batching, admission-aware serving, metrics, and standard HTTP APIs. |
| Observability | OpenTelemetry; Phoenix only for restricted development/evaluation | Keeps instrumentation portable and production content capture off by default. |

Exact model revisions, tokenizers, quantization, context limits, and serving image digests are release artifacts. A larger model is promoted only when the HKPL evaluation suite shows a material quality gain within the latency and capacity budget.

NeMo Guardrails is a PoC comparison candidate, not the initial security boundary. It may replace parts of the semantic rail implementation only if it improves multilingual false-negative and false-positive results without violating the latency budget. Database authorization, source approval, request validation, network controls, and citation validation always remain application-owned.

## Online workflow

The graph executes the following finite stages:

1. **Validate request.** Enforce schema, byte and token limits, Unicode validity, supported channel, locale, server-issued session, and allowlisted branch identifier.
2. **Apply input safety.** Run deterministic policy rules followed by the semantic guard. A missing policy or unavailable guard fails closed with a localized approved response.
3. **Load session context.** Load only the bounded, unexpired conversation window belonging to the server-issued session.
4. **Understand the query.** One schema-constrained model call returns a `QueryPlan` containing the route, standalone retrieval query, language/script, requested answer language, branch entities, and clarification need.
5. **Route.** The only routes are `static`, `policy`, `clarify`, `rag`, and `fallback`. Greetings and thanks use deterministic localized templates.
6. **Resolve branch context.** An explicit branch in the current message wins. Otherwise, a previously confirmed branch or trusted kiosk branch may resolve “here” or “this library”. Ambiguity produces clarification.
7. **Retrieve evidence.** Run dense and lexical retrieval concurrently against authorized active content, fuse ranks, and rerank.
8. **Pack context.** Select diverse evidence within the embedding and generation token budgets. Record exactly which evidence enters the prompt.
9. **Decide answerability.** Apply thresholds calibrated on held-out HKPL questions. Empty, weak, conflicting, expired, or irrelevant evidence produces clarification or fallback.
10. **Generate.** Use real system and user message roles. Evidence is delimited as untrusted data. Ordinary RAG uses temperature zero, reasoning disabled, and a bounded answer length.
11. **Validate.** Check output schema, policy, PII, cited evidence membership, locator integrity, grounding, and requested language. At most one bounded repair attempt is allowed.
12. **Respond and persist.** Emit only safety-approved content and structured citations, then store the minimum session state and metadata-only telemetry.

No unvalidated answer token is exposed. Progress events may be sent while processing, but the first `answer` event contains approved content.

## Public API

### `POST /v1/chat/sessions`

Creates a cryptographically random opaque session owned by the server. Web sessions use Secure, HttpOnly, SameSite cookies. Mobile and kiosk channel backends use an audience-bound session token. Clients cannot choose a session identifier.

### `POST /v1/chat/messages`

Accepts:

```json
{
  "text": "string",
  "locale": "en|zh-Hant|zh-Hans|null",
  "channel": "web|mobile|kiosk",
  "branch_id": "validated canonical identifier or null"
}
```

`text` may be a trusted upstream transcript, but the RAG agent has no voice flag or transcription fallback. The contract contains no arbitrary memory, prompt, free-form library object, coordinates, model setting, or tool instruction.

The response uses server-sent events with these event types:

- `status`: non-sensitive progress state.
- `answer`: approved answer text.
- `citations`: structured evidence objects.
- `clarification`: a bounded question needed to continue.
- `fallback`: approved no-answer or failure response and reason code.
- `error`: stable public error code without internals.
- `done`: response ID, outcome, corpus version, and policy version.

### Health

- `GET /health/live` reports whether the API process is alive.
- `GET /health/ready` verifies migrations, database connectivity, active public corpus availability, required policy configuration, and required model endpoints.

## Internal records

### `QueryPlan`

| Field | Meaning |
|---|---|
| `route` | `static`, `policy`, `clarify`, `rag`, or `fallback` |
| `standalone_query` | Retrieval query with conversational references resolved |
| `input_language` | Detected language and script |
| `answer_language` | Language/script to mirror in the response |
| `branch_ids` | Canonical branch identifiers, never free text in prompts |
| `entities` | Bounded service, event, resource, and date entities |
| `clarification` | Localized clarification text or null |

### `Evidence`

| Field | Meaning |
|---|---|
| `chunk_id` | Stable chunk identifier within a source version |
| `source_id` / `source_version_id` | Immutable provenance identifiers |
| `title` / `canonical_url` | Public citation identity |
| `locator` | Page, section, sheet/row, or webpage anchor |
| `text` | Exact selected evidence text |
| `language` / `branch_ids` | Ranking and applicability metadata |
| `retrieval_scores` | Dense, lexical, fused, and reranker diagnostics |

### `GroundedAnswer`

| Field | Meaning |
|---|---|
| `response_id` | Server-generated identifier |
| `outcome` | `answered`, `clarify`, or `fallback` |
| `answer` | Localized approved text |
| `cited_evidence_ids` | IDs drawn only from the packed context |
| `citations` | Public source/version/title/URL/locator/excerpt objects |
| `language` | Actual answer language/script |
| `fallback_reason` | Stable reason code or null |
| `corpus_version` / `policy_version` | Reproducibility metadata |

## Knowledge and publication model

### Core tables

- `knowledge_sources`: canonical URI, type, title, owner, trust tier, access scope, connector policy, and active version pointer.
- `knowledge_versions`: immutable checksum, source ID, effective dates, fetch/upload facts, parser and configuration versions, approval facts, and state.
- `knowledge_chunks`: version ID, ordinal, original text, normalized search variants, embedding, language, branch applicability, locator, and structured provenance.
- `ingestion_jobs`: idempotency key, source/version, state, lease owner, heartbeat, attempt count, deadline, and error summary.
- `chat_sessions` and `chat_messages`: server-owned expiry and the minimum raw context required for active conversation.

Version states are `staged`, `validated`, `approved`, `active`, `superseded`, and `failed`. A source has at most one active version. Activation updates the source pointer and version states in one transaction. Query replicas see either the previous complete version or the new complete version, never a partial mixture.

The `active_public_chunks` view joins only approved, active, public, currently effective versions. The public agent database role can select this view and session tables but cannot read base knowledge tables or restricted content. Migration, ingestion-writer, and public-query roles are separate.

### Approval rules

- A webpage already registered as an approved official HKPL/LCSD source may activate automatically after all validation gates pass.
- A newly registered source, uploaded document, policy document, compliance template, or connector-policy change requires human approval.
- Failed validation leaves the existing active version untouched.
- Raw patron conversations never become knowledge sources.

## Ingestion

The ingestion worker leases durable PostgreSQL jobs with `FOR UPDATE SKIP LOCKED`, heartbeats its lease, and uses bounded retry with an idempotency key. FastAPI request workers do not parse or embed documents.

The pipeline is:

1. Acquire from an approved upload or HTTPS allowlisted connector.
2. Enforce exact origin/path rules, DNS and resolved-IP checks, redirect revalidation, content-type, byte, page/archive, and time limits.
3. Store the immutable original and checksum in approved object storage.
4. Extract a normalized Docling document or deterministic tabular records in a non-root, no-network, resource-limited worker.
5. Preserve original text and create search-only Unicode and Traditional/Simplified Chinese variants.
6. Chunk by structure: headings for prose; atomic FAQ, policy, event, branch, table row, and spreadsheet records; repeated headers for split tables.
7. Retain short but meaningful content and independent provenance. Content equality never erases a distinct source locator.
8. Embed version-specific chunks.
9. Run extraction, metadata, duplicate-lineage, retrieval-smoke, and safety checks.
10. Record approval and atomically activate the version.

Parser output and model artifacts are pinned and mirrored internally. Production workers have no outbound internet except the allowlisted fetch path.

## Multilingual retrieval

- The original patron query is preserved and never translated wholesale to English.
- Embedding instructions are English while query content remains in its original language.
- Search variants include original script, Unicode-normalized text, Traditional Chinese, Simplified Chinese, approved branch/service aliases, and exact identifiers.
- Language is a ranking preference rather than a hard evidence filter; authoritative evidence in another language remains eligible.
- Explicit branch names override session context. Branch filters apply only when the question is branch-specific.

Candidate generation runs concurrently:

- Dense pgvector search: top 30.
- PostgreSQL full-text search for tokenized English: top 30.
- `pg_trgm`, exact title/alias, and identifier matching for Chinese, mixed-script, names, and misspellings: included in the lexical pool.
- Reciprocal-rank fusion: top 20.
- Cross-encoder reranking: top 8 before context packing.

These are PoC baselines, not environment variables changed ad hoc. A configuration version records promoted values, and the evaluation suite controls promotion.

Context packing retains the evidence-to-citation mapping. Citations are created from the packed evidence list, never the larger candidate list. If a citation does not resolve to a packed evidence ID, the answer is rejected.

## Safety and privacy

The application-owned safety gateway layers:

1. Request schema, size, Unicode, and rate validation.
2. Deterministic scope and compliance policies with LCSD-approved multilingual templates.
3. Multilingual semantic classification for harmful content, jailbreaks, sensitive intent, and PII.
4. Database-enforced publication and access isolation.
5. Retrieval-content inspection and instruction/data separation.
6. Schema-constrained generation using true message roles.
7. Output policy, PII, grounding, language, and citation validation.

Required failure behavior:

- Missing policy, guard failure, or malformed guard output: fixed localized fallback; generation is not called.
- Missing or invalid retrieval authorization: zero evidence; never retry without filters.
- Retrieval/model failure or insufficient evidence: approved fallback.
- Output, grounding, PII, or citation failure: discard generated text and return an approved fallback.
- Telemetry failure: bounded service operation may continue, but raw logging is never enabled as a fallback.
- Ingestion or audit failure: publication is blocked and the previous active version remains served.

The LLM process has no credentials, database access, tool execution, or internet access. Retrieved content is data, not instruction.

Session defaults are 30 minutes idle and 24 hours absolute for web/mobile; kiosk sessions expire after five minutes idle or explicit reset. Expiry deletes raw session text. Routine traces contain request ID, hashed session ID, route, model/config/source versions, stage latency, scores, decision codes, token counts, fallback reason, and citation IDs—not raw queries, prompts, answers, coordinates, or chunk text. Redacted review-sample persistence remains disabled until LCSD approves its roles and retention schedule.

## Evaluation and release gates

Evaluation invokes the same public workflow used in production, including history, branch resolution, safety, retrieval, generation, citations, and fallback behavior.

The versioned, human-reviewed dataset contains stable source/version/locator references and evidence snippets. Missing or mismatched evidence fails the run; it is never silently excluded. Required slices include English, Traditional Chinese, Simplified Chinese, Cantonese text, mixed language, cross-language retrieval, branch ambiguity, multi-turn references, stale and unanswerable questions, source conflicts, direct and indirect prompt injection, sensitive-policy false positives, and access-isolation attacks.

Metrics include extraction integrity, Recall@k, MRR/nDCG, reranker gain, answer correctness, faithfulness, citation precision and coverage, abstention quality, language adherence, safety false-positive/negative rates, and latency by stage. Deterministic checks, a separately versioned local judge, and sampled human review are used; the answer generator is never its sole judge.

A release requires:

- 100% dataset-reference integrity.
- 100% rejection of restricted, inactive, unapproved, expired, and superseded evidence in isolation tests.
- 100% citation membership in the exact packed context.
- No critical safety-policy bypass in the approved adversarial suite.
- No regression beyond the approved tolerance for each multilingual quality slice.
- p95 time to first safety-approved answer content at or below 40 seconds under 100 simultaneous requests.
- Under 1% infrastructure errors during the capacity test.
- Controlled overload using timely `429` or `503` responses with `Retry-After`, rather than unbounded queueing.

## Runtime and operations

### Local/PoC profile

The two RTX 2080 Ti GPUs are development hardware. Pinned, internally mirrored quantized artifacts run through llama.cpp-compatible endpoints. Generation normally occupies one GPU; embedding, reranking, and guard workloads share the other subject to measured memory limits. This profile demonstrates behavior and small-load evaluation, not the production capacity commitment.

### Production profile

vLLM serves official pinned model revisions on LCSD-approved GPU infrastructure. Replica count, tensor/data parallelism, quantization, batch limits, context size, and admission limits are outputs of the multilingual quality and 100-request capacity tests. API replicas remain stateless and scale independently from model servers and ingestion workers.

All environments expose the same internal generation, embedding, reranking, and guard client contracts. Production images and model/tokenizer artifacts are pinned by immutable digest, scanned, mirrored internally, and loaded before readiness succeeds. Internal model, database, and telemetry ports are not host-public.

OpenTelemetry measures request volume, stage latency, time to first approved answer, fallback and citation rates, retrieval score distributions, model queue/batch/token/GPU metrics, ingestion lease age and failures, and active corpus/config versions. Phoenix is a restricted development and evaluation viewer, not the production audit record.

Alembic owns every schema change. A clean database must migrate, create the public view and roles, load approved configuration, and reach readiness without requiring prior ingestion or import-time DDL.

## Governing references

- [HK Digital Policy Office Ethical AI Framework](https://www.digitalpolicy.gov.hk/en/our_work/data_governance/policies_standards/ethical_ai_framework/)
- [PCPD AI Model Personal Data Protection Framework](https://www.pcpd.org.hk/english/resources_centre/publications/files/ai_protection_framework.pdf)
- [pgvector hybrid search guidance](https://github.com/pgvector/pgvector#hybrid-search)
- [PostgreSQL pg_trgm](https://www.postgresql.org/docs/current/pgtrgm.html)
- [Docling supported formats](https://docling-project.github.io/docling/usage/supported_formats/)
- [Qwen3 Embedding and Reranker](https://github.com/QwenLM/Qwen3-Embedding)
- [vLLM serving](https://docs.vllm.ai/en/stable/)
- [NeMo Guardrails rail types](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/rail-types)
