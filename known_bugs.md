# Known Bugs: HKPL RAG Agent

## Scope

This ledger contains current defects that still require explicit action before the RAG agent can be treated as a production public service. Each entry states the root problem rather than every observed symptom. Defects confined to mechanisms that the target architecture retires are intentionally excluded.

Severity definitions:

- **P0:** stop-ship security, privacy, or data-exposure risk.
- **P1:** release-blocking correctness, reliability, evaluation, or capacity defect.
- **P2:** important operational or maintainability defect.

## P0 defects

### BUG-001 — Previously committed credentials remain in repository history

- **Subsystem:** Secrets and configuration
- **Evidence:** Commit `7d85fbc` removed `.env` from the index, added it to `.gitignore`, and added a placeholder-only `.env.example`. Earlier commits still contain the former non-empty credential configuration.
- **Impact:** Anyone with repository-history access may obtain the previously committed values until they are rotated and the history is purged. The current branch no longer tracks `.env`.
- **Acceptance criteria:**
  - Rotate every previously committed credential against its backing service.
  - Purge secret values from repository history using an approved coordinated procedure.
  - Keep `.env` untracked and retain a value-free `.env.example` or equivalent configuration reference.
  - Load production secrets from an LCSD-approved secret store or runtime-mounted secret.
  - Run a repository-history secret scan and record a clean result.

### BUG-002 — Anonymous retrieval does not enforce publication or access authorization

- **Subsystem:** Retrieval authorization
- **Evidence:** `src/retrieval.py:56-66` constructs the live retriever with only a `corpus_role=primary` metadata filter. Ingestion stores `access_level`, version, and status information, and `src/nodes.py:401-417` returns some of it as citation metadata, but retrieval does not require `public`, active, approved, current, or effective content.
- **Impact:** Restricted, partial, stale, failed, or superseded chunks can be selected for an anonymous patron answer. Post-retrieval filtering cannot undo disclosure to the reranker or generator.
- **Acceptance criteria:**
  - The public database role can read only an `active_public_chunks` view or equivalent database-enforced policy.
  - Authorization, active-version, approval, and effective-date constraints apply before vector and lexical ranking.
  - Missing or invalid scope returns zero rows and never retries unfiltered.
  - Integration tests prove that restricted, inactive, expired, failed, and superseded fixtures never reach retrieval, reranking, generation, citations, or telemetry.

### BUG-003 — Client-controlled session and context fields cross trust boundaries

- **Subsystem:** Sessions and request validation
- **Evidence:** `main.py:292-300` accepts caller-selected `session_id`, `library_code`, coordinates, and arbitrary `user_memory`. `main.py:434-469` loads history and places these values into graph state. `src/memory.py:8-29` selects history using only the supplied session ID. `src/nodes.py:675-700` interpolates library and memory values into the generation prompt.
- **Impact:** A caller can collide with another session, influence hidden prompt context, cause cross-session disclosure, inject instructions, or consume unbounded storage/model input.
- **Acceptance criteria:**
  - Sessions are server-issued, cryptographically random, audience-bound, and expire under the approved idle and absolute TTLs.
  - Public requests cannot submit arbitrary memory, system context, free-form library objects, or model settings.
  - Branch identifiers are validated against the canonical registry; kiosk context is trusted only from its authenticated channel.
  - Request text, history window, field lengths, and token counts have enforced limits.
  - Isolation tests demonstrate that one client cannot load or influence another session.

### BUG-004 — The safety boundary can fail open

- **Subsystem:** Input and output safety
- **Evidence:** `src/nodes.py:114-149` loads and calls the safety model before the classification exception boundary. `src/nodes.py:166-218` logs `prompt_safety` but bases blocking only on selected category hits. `src/nodes.py:238-245` explicitly converts classifier exceptions into `is_unsafe = False`. The output filter performs only prohibited-keyword and small phrase checks at `src/nodes.py:703-727`.
- **Impact:** Model-loading failures, classifier outages, contradictory results, malformed outputs, and unsupported unsafe categories can continue to retrieval and generation or reach patrons.
- **Acceptance criteria:**
  - Safety dependency failure, timeout, malformed output, or explicit unsafe result returns an approved localized fallback without calling generation.
  - Input and output safety policies use versioned deterministic rules plus a typed semantic guard result.
  - Direct, obfuscated, multilingual, and mixed-language adversarial tests cover every approved category and failure mode.
  - No response is emitted before output safety and grounding checks approve it.

### BUG-005 — Prompt and retrieval injection are not contained

- **Subsystem:** Prompt construction and grounding
- **Evidence:** `src/llm_client.py:33` sends the assembled prompt as a single `user` message. `src/nodes.py:625-700` concatenates retrieved text, current context, arbitrary memory, instructions, and the patron question into that prompt without a typed trust boundary. Retrieved webpage or document instructions are not identified as untrusted data.
- **Impact:** Direct user text or poisoned source content can override intended behavior, exfiltrate prompt/context information, alter the answer format, or defeat grounding and policy instructions.
- **Acceptance criteria:**
  - System policy, patron input, and untrusted evidence use distinct message roles and structured delimiters.
  - The LLM has no credentials, database access, internet access, or tools.
  - Query understanding and generation use schema-constrained outputs with strict parsing.
  - Ingestion and runtime tests cover direct and indirect prompt injection, system-prompt extraction, instruction override, and malicious retrieved content.
  - Invalid grounding or citation bindings discard the generated text and return an approved fallback.

### BUG-006 — URL ingestion permits SSRF and unbounded resource use

- **Subsystem:** Web ingestion
- **Evidence:** `src/ingestion/webpage.py:11-31` validates only the `http`/`https` scheme, follows redirects, and applies no hostname allowlist, resolved-IP restrictions, response-size cap, content-type check, or redirect revalidation. `main.py:249-290` exposes this path through an administrative endpoint. `scripts/crawl_hkpl_site.py:206-213` validates the requested domain but not every final redirect target or response byte count.
- **Impact:** The service can be induced to contact loopback, private, link-local, metadata, or otherwise unauthorized destinations and can download responses large enough to exhaust memory, disk, or worker capacity.
- **Acceptance criteria:**
  - Only registered HTTPS origins and paths are fetchable.
  - DNS and every resolved/final redirect IP reject loopback, private, link-local, multicast, reserved, and metadata ranges.
  - Redirect count, request deadline, response bytes, content type, archive expansion, and document pages are bounded.
  - Network egress independently restricts the worker to approved destinations.
  - Tests cover DNS rebinding, redirects to private addresses, oversized/chunked bodies, incorrect content types, slow responses, and redirect loops.

### BUG-007 — Administrative ingestion authentication fails open

- **Subsystem:** Privileged API security
- **Evidence:** `main.py:150-168` treats an empty `ADMIN_API_KEY` as authorization disabled. `docker-compose.yml:188` permits that empty default. All privileged operations share one secret and accept request-supplied actor strings rather than an authenticated identity.
- **Impact:** A missing environment value exposes document ingestion, replacement, deletion, and compliance operations. A shared key cannot provide role separation or trustworthy audit attribution.
- **Acceptance criteria:**
  - Production startup or readiness fails when privileged authentication is not configured.
  - Privileged calls use LCSD identity, authenticated actor IDs, and least-privilege roles.
  - Public and privileged routes are separated at the gateway and application layers.
  - Audit records derive actor identity from authentication and cannot be supplied by the request body.
  - Tests prove that absent, invalid, expired, and insufficient credentials fail closed.

## P1 defects

### BUG-008 — Raw patron and source content escapes the privacy boundary

- **Subsystem:** Privacy, logging, and telemetry
- **Evidence:** `src/tracing_helpers.py:19-80` records span input and output values. `src/retrieval.py:91-129` includes complete chunk text and metadata in trace records, and later spans attach full queries and documents. `src/memory.py` stores raw conversations without the selected idle/absolute expiry policy. Phoenix is host-published in `docker-compose.yml:202-205`.
- **Impact:** Patron questions, answers, source text, and possible personal data are duplicated into systems with different access and retention controls, increasing disclosure and compliance risk.
- **Acceptance criteria:**
  - Routine production logs and traces contain metadata and pseudonymous identifiers only.
  - Raw session messages expire after 30 minutes idle/24 hours absolute for web/mobile and five minutes idle for kiosks.
  - Raw diagnostic capture is disabled by default and requires an approved, time-bounded incident workflow.
  - Telemetry endpoints are internal and authenticated.
  - Automated tests inspect emitted spans/logs and reject raw queries, prompts, answers, coordinates, and chunk text.

### BUG-009 — A clean database cannot reliably bootstrap the agent

- **Subsystem:** Schema and startup
- **Evidence:** `src/retrieval.py:49-54` constructs the vector index and immediately calls `normalize_corpus_roles()`. `src/corpus.py:31-85` updates the vector table, while `postgres-init/init.sql` does not create that table. Schema definitions are split across Docker initialization, runtime DDL, and library-managed lazy creation.
- **Impact:** A fresh environment can fail during import before ingestion creates the vector table. Different startup paths can produce different schemas, preventing repeatable deployment and rollback.
- **Acceptance criteria:**
  - One Alembic migration chain owns all application tables, indexes, extensions, views, constraints, and roles.
  - Application import performs no DDL or corpus mutation.
  - A clean database migrates and reaches readiness with an empty corpus.
  - CI tests fresh migration, upgrade from the previous schema, and rollback/restore procedures.

### BUG-010 — Knowledge publication is neither atomic nor approval-gated

- **Subsystem:** Ingestion and corpus integrity
- **Evidence:** `src/ingestion/service.py:190-287` updates registry/version state, inserts vectors, deletes other versions, and changes status across separate operations. `src/corpus.py:103-120` and `scripts/ingest_pgvector_llamaindex.py:794-823` delete an existing corpus before a replacement is fully extracted and embedded. Newly indexed content becomes eligible without a distinct validated/approved/active publication boundary.
- **Impact:** Failure or concurrency can expose a partial version, remove the last good corpus, let a stale job delete newer content, or publish unreviewed material.
- **Acceptance criteria:**
  - Source versions are immutable and progress through staged, validated, approved, active, superseded, or failed states.
  - Jobs are durable, leased, idempotent, and reject stale completions.
  - All extraction, chunking, embedding, and smoke checks finish before activation.
  - One transaction switches the active version; failure leaves the previous version queryable.
  - New uploads and policy sources require authenticated approval, while pre-approved official connectors follow their recorded activation policy.
  - Concurrency and failure-injection tests prove last-known-good behavior.

### BUG-011 — Evaluation results cannot support a release decision

- **Subsystem:** RAG evaluation
- **Evidence:** `data/evaluation_dataset.csv` contains 175 rows tied to UUID/chunk identifiers from a prior corpus and is English-only. `scripts/evaluate_rag.py:161-203` retains only labels whose chunk currently exists, silently shrinking the evaluated set. `scripts/evaluate_rag.py:547-627` invokes a separate retrieve/context/generate path instead of the deployed graph. `scripts/evaluate_rag.py:67-88` and `scripts/evaluate_rag.py:321-378` use the same LLM service for generation and judging.
- **Impact:** Fresh environments cannot reproduce the evaluation; missing gold evidence is hidden instead of scored as failure; multilingual, session, branch, safety, citation, and fallback behavior is not tested; same-model judging biases results.
- **Acceptance criteria:**
  - Gold labels use stable source/version/locator references and verified evidence snippets.
  - Dataset-reference integrity must be 100%; any missing evidence fails the run.
  - Evaluation calls the real public workflow and covers all required multilingual, conversational, authorization, safety, stale-content, and no-answer slices.
  - Deterministic metrics, a separately versioned local judge, and sampled human review are combined.
  - Per-row infrastructure failures and total outages make the command and CI job fail non-zero.

### BUG-012 — Production capacity and overload behavior are unproven

- **Subsystem:** Model serving and performance
- **Evidence:** `docker-compose.yml:87-104` configures one generation server with `--parallel 1`. The API has no explicit generation admission controller, per-session concurrency limit, or bounded overload queue. No checked-in load test demonstrates the proposal's 100 simultaneous requests.
- **Impact:** Concurrent traffic can queue beyond the 40-second target, exhaust GPU/context memory, or fail unpredictably. Local progress events do not prove time to first approved answer content.
- **Acceptance criteria:**
  - A production-like vLLM deployment is sized using the selected model, context, output, guard, and retrieval configuration.
  - A repeatable test sustains 100 simultaneous representative requests with p95 time to first approved answer at or below 40 seconds and under 1% infrastructure errors.
  - Admission control returns timely `429` or `503` with `Retry-After` when capacity is exhausted.
  - Queue time, model time, token counts, batching, GPU memory, cancellation, and timeout behavior are measured and reported.

### BUG-013 — There is no automated regression suite

- **Subsystem:** Testing and release control
- **Evidence:** The repository contains no `tests/` tree or automated test files. Validation is concentrated in operational scripts that frequently report problems without failing their process.
- **Impact:** Changes to ingestion, retrieval, safety, prompts, sessions, citations, configuration, or dependencies can regress silently. Security and multilingual behavior have no executable release gate.
- **Acceptance criteria:**
  - CI runs unit tests for deterministic logic and integration tests against fresh PostgreSQL migrations and stubbed model endpoints.
  - Golden tests cover extraction/chunk provenance, hybrid retrieval, access isolation, multi-turn rewriting, branch clarification, citation binding, fail-closed safety, and session expiry.
  - The versioned multilingual end-to-end evaluation and 100-request load test are documented release gates.
  - Any failed check exits non-zero.

## P2 defects

### BUG-014 — Async request paths execute blocking work

- **Subsystem:** FastAPI concurrency and ingestion jobs
- **Evidence:** `main.py:249-290` performs synchronous webpage fetch/storage and ingestion from an async endpoint. `main.py:457` loads conversation history through synchronous database code during chat handling. `main.py:515-660` delegates long-lived ingestion to in-process FastAPI `BackgroundTasks`, which is neither durable nor isolated from API lifecycle.
- **Impact:** Blocking I/O stalls the event loop, increases tail latency, and reduces concurrent chat capacity. Process restarts lose background jobs or leave ambiguous state.
- **Acceptance criteria:**
  - Request-path database and HTTP operations use pooled asynchronous clients.
  - Document parsing, OCR, embedding, and publication run only in durable worker jobs.
  - Job leases, heartbeats, retry limits, deadlines, cancellation, and stale-worker rejection are tested.
  - Event-loop lag and API latency remain within the approved limits during ingestion activity.

### BUG-015 — Production artifacts and internal services are not hardened or reproducible

- **Subsystem:** Supply chain and deployment
- **Evidence:** `docker-compose.yml` uses mutable model/Phoenix image tags, downloads Hugging Face artifacts at runtime, publishes PostgreSQL, embedding, reranker, generator, agent, and Phoenix ports to the host, and defaults the database password. `Dockerfile.agent:13` uses a mutable `uv:latest` source image. The guard model is also downloaded lazily by `src/nodes.py:37-44`.
- **Impact:** Builds can change without source changes, production startup depends on external registries, compromised artifacts are harder to detect, and internal services are reachable beyond their required trust boundaries.
- **Acceptance criteria:**
  - Containers, models, tokenizers, quantizations, and parser artifacts are pinned by immutable revision and digest, scanned, and mirrored internally.
  - Production performs no outbound model or package download.
  - Internal services are reachable only on private networks from explicitly authorized workloads.
  - Containers run non-root with read-only filesystems, dropped capabilities, and resource limits where supported.
  - An SBOM, vulnerability scan, artifact manifest, and reproducible deployment record accompany each release.
