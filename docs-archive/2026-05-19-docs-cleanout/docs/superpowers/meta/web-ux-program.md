# Web UX Composer MVP — Program of Work

**Created:** 2026-03-28
**Branch:** RC4.2-UX (released as RC-5)
**Parent Epic:** Semi-Autonomous Platform (`elspeth-rapid-ea33f5`)
**Status:** Phase 1 ready for execution; Phases 2-6 require plan body sync + L3 decomposition

---

## Vision

A modular monolith web application for chat-first pipeline authoring. Users
describe data processing pipelines in natural language, a server-side LLM
composes them via tool calls, and users review, validate, and execute through
a React SPA. First increment towards the full Semi-Autonomous Platform.

---

## Document Hierarchy

```
Layer 1 — Main Spec + Main Plan (architecture-level)
  │
  ├── Layer 2 — Sub-Specs + Sub-Plans (per-phase, 6 each)
  │     │
  │     └── Layer 3 — Task-Plans (per-task within large sub-plans, as needed)
  │           Only created for sub-plans >1000 lines
  │           Decomposed at execution time, not upfront
  │
  └── Reviews — Seam reviews, panel reviews, re-reviews
```

**Decomposition rule:** Sub-plans exceeding 1000 lines must be broken into
task-level plans matching the internal seams of their parent sub-plan.

---

## Main Documents

| Type | Path | Lines | Purpose |
|------|------|-------|---------|
| Main Spec | `specs/2026-03-28-web-ux-composer-mvp-design.md` | 649 | Architecture, data model, API surface, UX, project structure |
| Main Plan | `plans/2026-03-28-web-ux-composer-mvp.md` | 1,707 | Phase overview, file map, review findings, warnings |
| Seam Contracts | `specs/2026-03-28-web-ux-seam-contracts.md` | — | Inter-module boundary contracts, type conversions, error shapes, ownership |

---

## Sub-Specs (Layer 2)

| # | Phase | Path | Lines | Scope |
|---|-------|------|-------|-------|
| 1 | Foundation | `specs/2026-03-28-web-ux-sub1-foundation-design.md` | 204 | Plugin manager extraction, WebSettings, app factory, CLI, `[webui]` extra |
| 2 | Auth & Sessions | `specs/2026-03-28-web-ux-sub2-auth-sessions-design.md` | 852 | AuthProvider protocol, 3 providers, session persistence, CRUD API, file upload |
| 3 | Catalog | `specs/2026-03-28-web-ux-sub3-catalog-design.md` | 166 | CatalogService wrapping PluginManager, catalog REST API |
| 4 | Composer | `specs/2026-03-28-web-ux-sub4-composer-design.md` | 692 | CompositionState, LLM tool-use loop, YAML generator, HTTP error shapes |
| 5 | Execution | `specs/2026-03-28-web-ux-sub5-execution-design.md` | 447 | Dry-run validation, background execution, thread safety, WebSocket progress |
| 6 | Frontend | `specs/2026-03-28-web-ux-sub6-frontend-design.md` | 508 | React SPA, chat, inspector, accessibility, empty states, error messages |

**Total sub-spec lines:** 2,869

---

## Sub-Plans (Layer 2)

| # | Phase | Path | Lines | Tasks | Needs L3 Decomposition? |
|---|-------|------|-------|-------|------------------------|
| 1 | Foundation | `plans/2026-03-28-web-ux-sub1-foundation.md` | 880 | 4 | No |
| 2 | Auth & Sessions | `plans/2026-03-28-web-ux-sub2-auth-sessions.md` | 4,021 | 12+ | **Yes** |
| 3 | Catalog | `plans/2026-03-28-web-ux-sub3-catalog.md` | 804 | 2 | No |
| 4 | Composer | `plans/2026-03-28-web-ux-sub4-composer.md` | 3,635 | 8+ | **Yes** |
| 5 | Execution | `plans/2026-03-28-web-ux-sub5-execution.md` | 2,784 | 8+ | **Yes** |
| 6 | Frontend | `plans/2026-03-28-web-ux-sub6-frontend.md` | 3,717 | 15+ | **Yes** |

**Total sub-plan lines:** 15,841

---

## Dependency Graph

```
Sub-Plan 1 (Foundation)         ← start here
  ├── Sub-Plan 2 (Auth & Sessions)
  │     ├── Sub-Plan 4 (Composer)  ← also needs Sub-Plan 3
  │     │     └── Sub-Plan 5 (Execution) ← also needs Sub-Plan 2
  │     │           └── Sub-Plan 6 (Frontend) ← needs all of 2-5
  └── Sub-Plan 3 (Catalog)        ← can run parallel with Sub-Plan 2
```

**Parallel opportunities:**
- Sub-Plans 1→2 and 1→3 can run in parallel after Phase 1 completes
- Sub-Plan 4 requires both 2 and 3

---

## Review History

### Round 1 — Initial Plan Review (5 reviewers)

| Reviewer | Focus | Findings |
|----------|-------|----------|
| Architecture Critic | Module boundaries, extraction path, DI | 2 blocking, 4 recommendations |
| Systems Thinker | Feedback loops, thread safety, archetypes | 3 blocking, 6 warnings |
| Python Engineer | Async/sync, protocols, frozen dataclasses | 5 critical, 5 warnings |
| Quality Engineer | Test pyramid, edge cases, security | 1 blocking, 6 warnings |
| UX Specialist | Layout, composing indicator, accessibility | 1 critical, 4 major |

**Result:** 8 blocking issues identified and fixed in the main plan.

### Round 2 — Seam Review (5 reviewers)

Cross-spec interface analysis after sub-specs were written.

**Result:** 18 seam issues found (7 critical, 11 high). All 18 fixed in specs.

### Round 3 — Re-Review (5 reviewers)

Verification pass after seam fixes applied.

**Result:** 18/18 original issues confirmed resolved. 6 new issues (R1-R6)
found and fixed. Spec-plan drift identified and addressed in Round 4.

### Round 4 — Expert Panel Review (5 reviewers)

Five-reviewer panel examining L1/L2 seam interfaces after all sub-specs completed.

| Reviewer | Focus | Findings |
|----------|-------|----------|
| Architecture Critic | Data model seams, Protocol boundaries, DI wiring | 11 issues (3 HIGH, 6 MEDIUM, 2 LOW) |
| Systems Thinker | Feedback loops, concurrency, error cascades, v2 extraction | 6 risks (1 CRITICAL, 1 HIGH, 1 MEDIUM, 3 LOW) |
| API Reviewer | REST/WS contracts, DTO shapes, error envelopes | 12 issues (2 CRITICAL, 5 HIGH, 4 MEDIUM, 1 LOW) |
| Quality Engineer | Integration test gaps, contract tests, round-trip coverage | 9 test gaps (3 CRITICAL, 3 HIGH, 3 MEDIUM) |
| Security Analyst | STRIDE analysis, IDOR, file upload, prompt injection, WS auth | 32 threats (4 CRITICAL, 15 HIGH, 5 MEDIUM, 3 LOW) |

**Result:** 4 critical issues fixed in specs (C1: async/sync boundary B8, C2: from_dict
reconstruction, C3: path injection S1/S2, C4: secret_key enforcement S3). Seam
contracts document created (`specs/2026-03-28-web-ux-seam-contracts.md`). 5 spec
amendments applied (H1, H2, H3, M1, H8p). All 6 sub-plans updated with Round 4
Review Amendments appendix. Plan body sync pending (see sync prompt below).

**Plan sync prompt:** `meta/prompt-sync-plans-with-r4-spec-delta.md` — self-contained
prompt for an agent to update plan bodies to match the post-R4 spec state.

---

## Round 4 Security Rules

These are hard rules established by the security review. They are non-negotiable.

| Rule | Fix ID | Description | Enforcement Point |
|------|--------|-------------|-------------------|
| **S1** | C3 | `landscape_url` is removed from `PipelineMetadata` and the `set_metadata` LLM tool. The LLM must never control audit infrastructure. Landscape URL comes exclusively from `WebSettings`. | Sub-4 `state.py`, `yaml_generator.py` |
| **S2** | C3 | Source plugin options containing `path` or `file` keys are restricted to paths under `{data_dir}/uploads/`. Enforced at **both** composition-time (Sub-4 `set_source` tool, for LLM feedback) and validation-time (Sub-5 `validate_pipeline()` step 1, authoritative). Uses `Path.resolve()` + `is_relative_to()` to defeat `../` traversal. | Sub-4 `tools.py`, Sub-5 `validation.py` |
| **S3** | C4 | `WebSettings.secret_key` must not be the default value (`"change-me-in-production"`) in non-test environments. Startup raises `SystemExit` with a clear message. Test environments (pytest in `sys.modules` or `ELSPETH_ENV=test`) are exempt. | Sub-2 app startup guard |

---

## Web Surface Trust Boundary Mapping

The web application introduces new trust boundaries that must be mapped to
ELSPETH's three-tier trust model (see CLAUDE.md § Data Manifesto). This mapping
is **not** deferred — it applies to v1.

**Clarification:** "Governance tiers" (deferred to v2) means *approval workflows*
(e.g., "this pipeline needs manager sign-off before execution"). This is distinct
from ELSPETH's *data trust tiers* (Tier 1/2/3), which apply to v1 and are mapped
below.

| Boundary | Data | Trust Tier | Handling Rule | Enforcement |
|----------|------|-----------|---------------|-------------|
| User chat input | Natural language prompt | **Tier 3** (external, zero trust) | Passed to LLM unsanitised; LLM output is constrained by tool schema and path allowlist (S2). No coercion — the LLM either produces valid tool calls or text. | Sub-4 tool validation |
| File upload | User-provided files | **Tier 3** (external, zero trust) | Path-sanitised (B5), size-checked, restricted to `{data_dir}/uploads/` (S2). Content is not trusted — source plugins handle Tier 3 data per existing model. | Sub-2 upload handler |
| Auth tokens (from IdP or login) | JWT / OIDC token | **Tier 3 at boundary** | Validated by AuthProvider protocol. Once validated, `UserIdentity` is Tier 1. Invalid tokens → 401, never coerced. | Sub-2 auth middleware |
| Session DB (sessions, messages, states, runs) | Application state | **Tier 1** (our data, full trust) | Written by our code, read by our code. Bad data = crash (standard Tier 1 rules). | Sub-2 SessionService |
| CompositionState (in-memory) | Pipeline configuration | **Tier 1** (our data) | Frozen dataclass, immutable, versioned. Created by our tool executor, not by external input directly. The LLM proposes mutations via tools; the tool executor validates and applies them. | Sub-4 tool executor |
| CompositionState (serialised in DB) | JSON columns | **Tier 1** (our data) | Serialised via `to_dict()`, deserialised via `from_dict()`. `_version` envelope for schema evolution. Bad data on read = crash (Tier 1). | Sub-2 `from_record()` |
| Generated YAML | Pipeline config for engine | **Tier 1** (our data) | Produced deterministically from CompositionState by `generate_yaml()`. Consumed by `load_settings()` + real engine validation. Not user-editable in v1. | Sub-4 `yaml_generator.py` |
| LLM API responses | Tool call results from LLM provider | **Tier 3** (external) | LiteLLM normalises responses. Malformed tool calls → error message returned to LLM for self-correction. Unknown tool names → ValueError (offensive). | Sub-4 `service.py` |
| WebSocket progress events | RunEvent payloads | **Tier 1** (our data) | Generated by Orchestrator (our code), bridged via ProgressBroadcaster. Not user-modifiable. | Sub-5 `progress.py` |
| Landscape audit DB | Audit records | **Tier 1** (our data, full trust) | Standard ELSPETH Tier 1 handling. URL comes from WebSettings only (S1). | Engine (existing) |

**Key principle:** The web surface does NOT introduce new Tier 2 data. User input
(Tier 3) is validated at the boundary and becomes Tier 1 application state. The
existing Tier 2 (pipeline data post-source) is unchanged — it flows through the
engine as before.

---

## Server-Side Validation Gate (Audit Gap)

**Finding R5-4:** The Execute gate (Stage 2 validation must pass before execution)
is enforced only in the frontend. `POST /api/sessions/{id}/execute` does not check
whether the bound CompositionState has passed Stage 2 validation. This means:

1. A direct API call bypasses the gate entirely.
2. Validation status is lost on page refresh.
3. There is no audit record of "this pipeline was validated before execution."

For a framework whose headline guarantee is auditability, this is a known gap.

**Current state (v1):** The `composition_states` table stores `is_valid` and
`validation_errors` from Stage 1 (composition-time). Stage 2 results
(`ValidationResult` from `validate_pipeline()`) are returned to the frontend but
**not persisted server-side**. The Execute button state is derived from
`executionStore.validationResult` (ephemeral frontend state).

**Required for audit integrity:** The execute endpoint should verify that a Stage 2
validation has passed for the specific `CompositionState` version being executed.
This requires:

- A `validation_results` table (or column on `runs`) recording the `ValidationResult`
  for the `state_id` that was validated.
- A server-side check in `execute()`: reject with 422 if no passing Stage 2
  validation exists for the bound `state_id`.
- The audit trail then records: "state version N was validated at time T, execution
  started at time T+1 against state version N."

**Disposition:** Promoted from casual deferral to **Spike 6**. This is a design
spike because it touches Sub-2 (data model), Sub-5 (execute endpoint), and
potentially Sub-4 (how validation results are stored after dry-run). Estimated
effort: 2-3 hours for the spike, half a day for implementation.

---

## Round 4 Seam Contract Ownership

Each inter-module boundary has a single owner responsible for the contract and its
tests. Full contracts: `specs/2026-03-28-web-ux-seam-contracts.md`.

| Seam | Boundary | Owner | Key Contract |
|------|----------|-------|-------------|
| **A** | CompositionState lifecycle | Sub-4 | `to_dict()`/`from_dict()` round-trip; Sub-2's `to_record()`/`from_record()` use these; `metadata`↔`metadata_` column mapping |
| **B** | Composer ↔ Session route | Sub-4 | `compose(message, messages, state)` — route handler mediates, pre-fetches `messages`; calls `state.validate()` before `save_composition_state()` |
| **C** | Catalog ↔ Composer tools | Sub-4 | Protocol uses singular (`"source"`), REST uses plural (`sources`); route handler translates |
| **D** | Execution ↔ Sessions run lifecycle | Sub-5 | `_call_async()` bridges sync thread → async SessionService; Run state transitions; `error_type: "run_already_active"` on 409 |
| **E** | WebSocket progress | Sub-5 | RunEvent wire format; close codes: 1000 (terminal), 1006 (reconnect), 1011 (error), 4001 (auth) |
| **F** | Validation gate | Sub-6 | `clearValidation()` on ANY `compositionState.version` change (composer, revert, session switch); Stage 1 vs Stage 2 separate renderers |
| **G** | Auth config → Frontend | Sub-2 | Error envelope: `{detail, error_type?}`; all errors use `detail` not `message` |
| **H** | Source path security | Sub-5 | Path allowlist at composition-time (Sub-4, feedback) + validation-time (Sub-5, authoritative) |

---

## Round 4 Finding Dispositions

### Fixed in Specs

| ID | Issue | Fix Applied |
|----|-------|-------------|
| **C1/B8** | `_run_pipeline()` calls async SessionService from sync thread | `_call_async()` helper using `run_coroutine_threadsafe` (Sub-5) |
| **C2** | No `from_dict()` reconstruction for CompositionState | Factory methods on all `*Spec` types + round-trip invariant (Sub-4) |
| **C3/S1** | LLM can set `landscape_url` to attacker-controlled DB | Removed field from PipelineMetadata and YAML generator (Sub-4) |
| **C3/S2** | LLM can configure sources reading arbitrary server files | Path allowlist at composition + validation boundaries (Sub-4, Sub-5) |
| **C4/S3** | Default `secret_key` only produces a warning | Hard crash (`SystemExit`) in non-test environments (Sub-2) |
| **H1** | `compose()` parameter `session` type unspecified | Changed to `messages: list[ChatMessageRecord]`, pre-fetched by route handler (Sub-4) |
| **H2** | Wire field `state` vs store field `compositionState` | Documented mapping in Sub-6 store action |
| **H3** | Revert doesn't clear validation gate | `revertToVersion()` calls `clearValidation()` before updating state (Sub-6) |
| **M1** | `/state/yaml` endpoint not in Sub-2 inventory | Added to Sub-2 endpoint table with 501 stub until Sub-4 |
| **H8p** | No rate limiting on POST /messages | `composer_rate_limit_per_minute: int = 10` in WebSettings (Sub-1) |

### Addressed by Seam Contracts (no further spec change needed)

| ID | Issue | Seam Contract |
|----|-------|---------------|
| **H4** | `metadata`/`metadata_` column mapping undocumented | Seam A |
| **H5** | `tool_calls` JSON schema undefined | Cross-cutting section |
| **H6** | WebSocket close codes unspecified | Seam E |
| **H7** | Stage 1 vs Stage 2 validation error conflation | Seam F |
| **M2** | Error envelope inconsistent (`detail` vs `message`) | Seam G |
| **M3** | Singular/plural `plugin_type` namespace | Seam C |
| **M5** | `is_valid` not populated before `save_composition_state()` | Seam B |

### Accepted for v1 (documented, no action)

| ID | Issue | Rationale |
|----|-------|-----------|
| **M4** | `validate()` takes state, `execute()` takes IDs | Intentional asymmetry: validate is stateless, execute owns lifecycle |
| **M10** | SQLite B6 race (check-then-insert) | Asyncio serialises handlers in single-process mode; PostgreSQL partial index is the production fix |
| **M11** | `_run_pipeline` monolith incompatible with Temporal | v2 architecture decision; add ADR when v2 planning begins |
| **M12** | ProgressBroadcaster requires full replacement for Redis | RunEvent shape is portable, broadcaster implementation is not; document in v2 planning |

### Deferred to Pre-Deployment Checklist

| ID | Issue | Trigger |
|----|-------|---------|
| **M6** | No file type validation on uploads | Before any internet-facing deployment |
| **M7** | JWT in WebSocket query parameter logged by proxies | Document risk in deployment guide; ticket-based auth for production (v1.1) |
| **M8** | JWT in localStorage vulnerable to XSS | Add CSP headers; httpOnly cookie migration is v1.1 |
| **M9** | No token revocation mechanism | Document in deployment guide; 24h expiry is acceptable for MVP |
| **H8-login** | No rate limiting on login endpoint | Before any internet-facing deployment |

### Design Spikes Needed (not yet specced)

| Spike | Scope | Estimated Effort | When |
|-------|-------|-----------------|------|
| **Spike 1** | Messages rate limiting — per-user vs per-session, interaction with composer timeout, in-flight request counting | 1-2 hours | Before Sub-4 implementation |
| **Spike 2** | Login rate limiting + account lockout — per-IP vs per-username, lockout state storage, progressive backoff | 1 hour | Before internet-facing deployment |
| **Spike 3** | File upload type validation — extension vs magic bytes, XLSX detection, downstream plugin validation overlap | 1 hour | Before internet-facing deployment |
| **Spike 4** | WebSocket ticket-based auth — ticket exchange endpoint, one-time-use tickets, TTL, storage mechanism | 2-3 hours | v1.1 |
| **Spike 5** | v2 extraction architecture — Temporal activity decomposition, Redis broadcaster replacement, partial unique index migration | 4-6 hours | v2 planning phase |
| **Spike 6** | Server-side validation gate — `validation_results` table, execute endpoint pre-check, audit trail for "validated before execution" | 2-3 hours | Before Sub-5 implementation (audit integrity) |

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Modular monolith (not microservices) | Single deployable for v1; Protocol interfaces designed for later extraction |
| Server-side LLM composer | Backend owns the tool loop; model can't bypass platform controls |
| FastAPI lifespan for async services | Python 3.12+ compatible; ProgressBroadcaster needs `get_running_loop()` |
| SQLAlchemy Core (not ORM) | Consistent with existing Landscape; SQLite dev / Postgres prod |
| WebSocket auth via query parameter | Browser WebSocket API doesn't support custom headers; close code 4001 for auth failure |
| Indeterminate progress bar | Engine doesn't know total row count in advance; counters provide signal |
| CompositionState as frozen dataclass | Domain model authority in `composer/state.py`; `to_dict()` for serialisation |
| 404 for IDOR (not 403) | Prevents leaking resource existence |

---

## Known Deferred Items

These are documented decisions to defer, not forgotten gaps:

| Item | Deferred To | Reason | R4 Ref |
|------|-------------|--------|--------|
| Governance tiers (approval workflows) | v2 | Means *who can authorise execution* (approval chains, sign-off gates), NOT data trust tiers. Data trust tiers are mapped in v1 — see § Web Surface Trust Boundary Mapping. Deferred because no users yet; server-side validation gate (Spike 6) is the v1 control point. | R5-1 |
| Live preview execution | v2 | Dry-run validation is the MVP preview | — |
| Graph editing | v2 | Chat-first is the primary interaction | — |
| Multi-tenant isolation | v2 | Single-tenant target (2-5 users) | — |
| Redis Streams telemetry | v2 | In-process WebSocket forwarding for v1; broadcaster requires full replacement, not transport swap | M12 |
| Temporal workflows | v2 | `_run_pipeline` requires activity decomposition; SessionService is the integration point | M11 |
| Frontend tests | v1.1 | W17 — add Vitest + React Testing Library. **High blast radius:** W3 validation gate, WS close code discrimination, cross-store interactions are untested. Consider a thin smoke suite (chat send → compose → inspect) during v1 to cover the critical path. | QE, R5-5 |
| Stage 2 validation persistence | **Spike 6** | **Promoted from v1.1.** Frontend-only gate has no audit trail and is bypassed by direct API calls. Server-side enforcement needed for audit integrity. See § Server-Side Validation Gate above. | R5-4 |
| Stuck run reaper | v1.1 | `_on_pipeline_done` callback is the v1 safety net | — |
| WebSocket ticket-based auth | v1.1 | JWT in query param is logged by proxies; ticket exchange is non-trivial (Spike 4) | M7 |
| httpOnly cookie token storage | v1.1 | localStorage is XSS-vulnerable; add CSP headers in v1, migrate cookies in v1.1 | M8 |
| Token revocation / blacklist | v1.1 | 24h expiry is acceptable for MVP; add blacklist before wider deployment | M9 |
| File upload type allowlist | Pre-deploy | Path allowlist (S2) mitigates the main attack vector; type validation before internet exposure | M6 |
| Login rate limiting + lockout | Pre-deploy | 2-5 trusted users for MVP; add before any internet-facing deployment (Spike 2) | H8 |

---

## Execution Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Foundation | R4 appendix added, body sync pending | 880 lines, 4 tasks |
| 2. Auth & Sessions | R4 appendix added, body sync pending, needs L3 decomposition | 4,021 lines, 12+ tasks |
| 3. Catalog | R4 appendix added, body sync pending | 804 lines, 2 tasks |
| 4. Composer | R4 appendix added, body sync pending, needs L3 decomposition | 3,635 lines, 8+ tasks |
| 5. Execution | R4 appendix added, body sync pending, needs L3 decomposition | 2,784 lines, 8+ tasks |
| 6. Frontend | R4 appendix added, body sync pending, needs L3 decomposition | 3,717 lines, 15+ tasks |

**Next actions:**
1. Run plan body sync using `meta/prompt-sync-plans-with-r4-spec-delta.md`
2. Execute Sub-Plan 1 (Foundation)
3. Spike 1 (messages rate limiting) — before Sub-4 implementation
4. Spike 6 (server-side validation gate) — before Sub-5 implementation
