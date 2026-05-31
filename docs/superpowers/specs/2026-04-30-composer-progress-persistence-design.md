# Composer Progress Persistence — Design

**Ticket:** [elspeth-90b4542b63](filigree:elspeth-90b4542b63) — Composer progress persistence — tool-call breadcrumbs and partial drafts survive long-running failures
**Parent epic:** [elspeth-528bde62bb](filigree:elspeth-528bde62bb) — Composer LLM evaluation remediation
**Related future epic:** [elspeth-f0460a6594](filigree:elspeth-f0460a6594) — Composer async/background execution model (deferred to Future release)
**Date:** 2026-04-30
**Status:** Proposed (revision 5 — addresses the four-reviewer plan-review pass on the Phase 2 plan derived from rev 4: reality, architecture, quality, systems. Closes the four BLOCKERs summarized in §12.2 and re-derives the redaction design from the actual `tools.py` architecture.)
**Branch:** RC5-UX (or successor)
**Tier-artifact match:** **M-tier** change delivered as a four-phase plan. Revision 1 labelled XS; revision 2 grew to S; revision 3 retained S; revision 4 demotes to M and explicitly splits delivery (see §11) because the `SessionsTransaction` primitive (Phase 1), the redaction framework (Phase 2), the compose-loop persistence (Phase 3), and the frontend recovery surface (Phase 4) are independently reviewable, independently testable, and independently deployable. Each phase is a separate PR against the parent epic.

**Path conventions.** Throughout this spec, `web/...` and `sessions/...` are shorthand for `src/elspeth/web/...` and `src/elspeth/web/sessions/...` respectively. Full paths appear the first time a file is cited per major section; later references in the same section may use the shorthand. Test paths are written full from repo root (`tests/...`). Implementers cloning the repo and following a citation should prepend `src/elspeth/` when the path begins with `web/`.

**Architectural pivot in revision 4.** Revision 3 proposed an async `SessionsTransaction` context manager interleaved with `asyncio.shield`-protected audit inserts. The first-principles panel established that (a) the current `SessionsService` is built on a sync `Engine` dispatched via `_run_sync`/`run_sync_in_worker`, so a multi-await transaction context cannot share a connection without pinning a worker thread; (b) `asyncio.shield` does not deliver "the inner write completes before the outer await returns" — it detaches the shielded coroutine and lets the outer await re-raise `CancelledError` immediately; (c) the proposed `try/except/finally` block reads an unbound local in the `BaseException` path, raising `NameError` that replaces the in-flight `CancelledError`. Revision 4 resolves all three by collapsing each per-turn write (assistant row + N tool rows + corresponding `composition_states` rows + sequence-number reservations) into **one sync block dispatched through `_run_sync`**. This eliminates the cross-async-boundary race surface entirely. The `SessionsTransaction` primitive becomes a sync-only context yielded inside the `_run_sync` worker. See ADR row "Transaction primitive shape" in §3 and the rewritten §5.2 / §5.7.

**Architectural pivot in revision 5.** Revision 4's redaction design (§4.2 / §4.4) was derived from the false premise that "Composer tools declare arguments via Pydantic `BaseModel` subclasses" (rev-4 §4.4.1, line 649). The four-reviewer plan-review pass on the rev-4-derived Phase 2 plan established by direct citation against `src/elspeth/web/composer/tools.py` (HEAD `f5115fd5`, 5481 lines) that (a) **no tool declares a Pydantic argument model**: `tools.py` imports Pydantic only for `ValidationError`, and every handler takes `arguments: dict[str, Any]`; (b) dispatch is via **six function-pointer registries** at lines 5250–5314 (`_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, `_BLOB_DISCOVERY_TOOLS`, `_BLOB_MUTATION_TOOLS`, `_SECRET_DISCOVERY_TOOLS`, `_SECRET_MUTATION_TOOLS`), with three tools (`preview_pipeline`, `diff_pipeline`, `set_pipeline`) carrying extended signatures handled inline in `execute_tool()`; (c) there is **no `class ComposerTool` base** anywhere in the project — the rev-4 §4.4.2 `ClassVar` pattern is fictional. Revision 5 replaces the implicit class-hierarchy registry with an explicit **manifest** (`MANIFEST: dict[str, ToolRedaction]` keyed by tool name, mirroring the existing `_TOOL_REQUIRED_PATHS: dict[str, ...]` precedent at `src/elspeth/web/composer/service.py:702`). Each `ToolRedaction` entry is *either* type-driven (an `argument_model: type[BaseModel]` whose fields carry `Sensitive[T]` annotations — Phase 2 promotes ~6–8 sensitive-touching tools to this shape with full `model.model_validate` dispatch validation; promoted handlers catch `pydantic.ValidationError` and re-raise as `ToolArgumentError` caught at `service.py:2480` for ARG_ERROR routing) *or* declarative (the rev-4 `ToolRedactionPolicy` shape, retained for tools whose argument surface is purely structural — Phase 2 covers the remaining ~29–31 tools this way, enumerated explicitly per registry). The walker and the adequacy guard share **one** traversal iterator (§4.2.4) yielding `(path, field_type, metadata, value_provider)` tuples, parametrically descending into `BaseModel`, `list[BaseModel]`, `dict[str, BaseModel]`, `tuple[BaseModel, ...]`, `Optional[BaseModel]`, `Union[*, BaseModel, *]` arms, and `Annotated[T, *, Sensitive(), *]` regardless of marker position; future schema changes touch the iterator and both consumers atomically. The walker is at the **Tier-3 → Tier-1 boundary**: unknown response keys are **fail-closed-redacted** with a fixed sentinel string (`"<redacted-unknown-response-key>"` — no length disclosure) and a counter increments; only **internal-invariant violations** (manifest entry missing for a dispatched tool name, summarizer raises, summarizer returns non-`str`) crash via `AuditIntegrityError`, in line with CLAUDE.md plugin-ownership policy. The `@elspeth/security` CODEOWNERS team referenced in rev-4 §4.4.5 cannot exist on `johnm-dta/elspeth` (personal-account repo; GitHub teams require an organisation), so rev-5 drops the team-routing control and promotes the policy-hash snapshot test plus a `policy-weaken-justified` PR-label gate enforced by a CI step (§4.4.5) to primary control — defense by mechanism, not defense by paperwork. See the rewritten §4.2 and §4.4, the amended §9 RSK-03 (summarizer-exception crash discipline), the amended §11 Phase 2 scope, and the rev-5 reviewer-finding traceability at §12.2.

---

## 1. Goals and Non-Goals

### 1.1 Goal

When a long-running compose request fails (wall-clock timeout, turn-budget
exhaustion, plugin crash, runtime-preflight rejection, or any other
`ComposerServiceError` subclass that carries `partial_state`), the user can
resume from the LLM's accumulated work rather than starting over.
Specifically:

1. The user sees, on the failure response, a complete record of every tool
   call the LLM made and every response those tools returned.
2. The user can apply the partial draft to their editor in one click, with a
   diff against current state.
3. Tool arguments and responses are persisted to `chat_messages` with a
   centrally-enforced redaction policy that each tool declares locally.
4. Replaying a failed compose by reviewing chat history reproduces the exact
   sequence the LLM saw, **byte-identical after redaction-sentinel
   substitution**, ordered by per-session monotonic sequence.

### 1.2 Non-Goals

- Async / background-job execution. Filed as future epic
  [elspeth-f0460a6594](filigree:elspeth-f0460a6594).
- Resume-by-replay (re-running tool calls automatically). The user resumes
  from the persisted state; they do not re-invoke the LLM with the partial
  transcript.
- Streaming progress beyond what `_active` and `composer-progress` already
  provide.
- Migration of existing `chat_messages`. Pre-release per CLAUDE.md no-legacy
  policy — direct cutover.

### 1.3 In-Scope Failure Paths

The three existing route helpers in `web/sessions/routes.py`:

- `_handle_convergence_error` — wall-clock timeout, turn-budget exhaustion.
- `_handle_plugin_crash` — plugin bug surfaced through composer.
- `_handle_runtime_preflight_failure` — final-gate runtime preflight reject.

Plus any future failure path that surfaces a `ComposerServiceError` subclass
carrying `partial_state` (the contract is the exception type, not the
specific failure mode).

### 1.4 Quantified NFRs

| NFR | Target | Verification |
|---|---|---|
| Per-turn DB write overhead (sanity bound) | p95 ≤ 250 ms with N ≤ 8 tool calls per assistant turn, measured in CI on standard infra | `tests/integration/web/test_compose_loop_latency_sanity.py` (new) — order-of-magnitude bound, not a tight budget |
| Per-turn DB write overhead (tight target) | p95 ≤ 25 ms with N ≤ 8 on dedicated bench host | Nightly bench job, not gated on CI; failure raises a tracking observation, not a build break |
| Recovery panel time-to-interactive | ≤ 500 ms with ≤ 50 tool rows in transcript | Frontend perf test (new); same sanity-bound discipline as latency NFR |
| Redaction-summarizer failure rate (rev 5: SLO tightened to 0) | **0 events expected**; any non-zero value is an incident, not a budget. The walker raises `AuditIntegrityError` on summarizer exception or non-`str` return (§9 RSK-03; §4.2.6 boundary table). The rev-4 "≤ 0.1% in 24h" budget is removed because silent sentinel-substitution is no longer the disposition — system-code summarizers must be total and `str`-returning by contract; non-zero events represent system bugs, not pathological inputs. | OTel counter `composer.redaction.summarizer_errors_total`; SLO threshold = 0 |
| Tier-1 audit-write failure rate (tool succeeded, audit insert raised non-IntegrityError) | **0 events expected**; any non-zero value alerts | OTel counter `composer.audit.tool_row_tier1_violation_total`; SLO threshold = 0. This is the load-bearing primacy violation. |
| Audit-state-rollback during atomic-pair commit | **0 events expected**; any non-zero value alerts | OTel counter `composer.audit.state_rolled_back_during_persist_total`; SLO threshold = 0 |
| Audit-row insert failure during tool-exception unwind (best-effort path) | telemetry-only; no SLO. The tool exception is the dominant signal; the audit best-effort write is ancillary. | OTel counter `composer.audit.tool_row_persist_failed_during_unwind_total` (no alert). Discussed in §5.5 row 7 / §5.2 audit-failure primacy. |
| Tool-call integrity violation (LLM emitted duplicate `tool_call_id` or other CHECK violation) | provider-dependent; SLO is "the violation is detected and the request fails fast." Counter exists for visibility, not for budgeting. | OTel counter `composer.audit.tool_row_integrity_violation_total`; surfaces RSK-12 in production. |
| Per-turn LLM tool-call cap | Hard limit of 16 tool calls per assistant turn (configurable; default 16). Exceeding the cap fails the request fast with a discriminated reason code. | New cap in `_compose_loop`; CL-PP-12 covers (§8.2). Defends against prompt-injection-induced amplification (RSK-13). |
| Redacted-transcript fidelity | The persisted `chat_messages` corpus is byte-identical to the canonical redacted form: every `chat_messages.content` and every `chat_messages.tool_calls` JSON entry equals the output of `apply_redaction_policy(...)` / `apply_response_redaction(...)` on the original tool call / response, with declared sensitive keys substituted by sentinel or summarizer output and non-declared keys passed through unchanged. **This is fidelity of the redacted record, not a claim that the original LLM-visible content is recoverable.** Replay for debugging means re-reading the audit-grade transcript; it does NOT mean re-feeding the transcript to the LLM and expecting identical tool calls (summarizer outputs are lossy by design). | Property test (§8.3); explicit in §8.3.2 post-conditions |
| Audit-ahead-of-state invariant violation rate (forward direction) | **0 events**; single violation = audit incident, not an SLO budget | Property test (§8.3) + post-condition assertion in `_compose_loop` |
| State-ahead-of-audit invariant violation rate (backward direction) | **0 events**; verified at schema level via the new `composition_states.provenance` discriminator (§4.1) | Property test (§8.3) post-condition; schema-level introspection in `tests/integration/web/test_inv_audit_ahead_backward.py` (new) |

The latency NFR is split deliberately into a CI-friendly sanity bound and
a tight nightly-bench target. CI runs on shared infrastructure with
unbounded noise; tight thresholds at p95 are notorious flake sources.

---

## 2. Context — What Already Exists

The original ticket text (filed 2026-04-27) was written before the
remediation work in commits 4fce0cae (RC reason codes), 1ad03ddd (in-flight
observability + cancel-race drain), and 83e6228d (redacted blob path
sentinel) landed. The post-merge state is:

| Original concern | Status today |
|---|---|
| "No partial state survives a failure." | **Mostly fixed.** `ComposerConvergenceError`, `ComposerPluginCrashError`, `ComposerRuntimePreflightError` each carry `partial_state: CompositionState`, captured iff `state.version > 0` (see `web/composer/protocol.py:84-288`). Three route handlers persist that partial state and return it in 422/500 responses. |
| "Wall-clock timeout has no breadcrumbs." | **Fixed.** Inner LLM-call `asyncio.wait_for` timeouts convert to `ComposerConvergenceError.capture(state)` at `web/composer/service.py:1226-1231`. |
| "No in-flight visibility." | **Fixed.** `composer.requests.inflight` UpDownCounter, `composer.request.terminal.total` Counter, `GET /api/sessions/_active` cross-session enumeration (commit 1ad03ddd). |
| "Distinct failure causes look the same." | **Fixed.** Discriminated reason codes on `ComposerProgressEvent`: `convergence_wall_clock_timeout`, `convergence_discovery_budget`, `convergence_composition_budget`, `client_cancelled`, `runtime_preflight_failed` (commit 4fce0cae). |
| "Only final assistant text is persisted." | **Open.** The compose loop today appends each assistant turn to an in-memory `llm_messages` list (`src/elspeth/web/composer/service.py:_compose_loop`, defined at line 573; assistant append at ~lines 675-691) but does **not** insert per-turn rows into `chat_messages`. Only the final message lands, via the route layer's calls to `SessionsService.add_message(...)` at `src/elspeth/web/sessions/routes.py:1487` and `:1883`. (The method definition itself lives at `src/elspeth/web/sessions/service.py:276-325`; revision 3's "call site at sessions/service.py:283-320" was a misidentification — that range is the method body, not a caller.) The `chat_messages.tool_calls` JSON column exists at `src/elspeth/web/sessions/models.py:68` but is populated only on that final insert. |
| "No tool-result rows in chat history." | **Open.** The schema permits `role='tool'` (`web/sessions/models.py:81`) but no insert site exists. Tool *responses* are not persisted. |
| "Frontend has no recovery surface." | **Open.** The data is in DB and in failure response body, but no UX renders it as a "draft you can pick up." |
| "Tool argument redaction is unverified." | **Open.** No central redaction layer exists for `chat_messages.tool_calls` JSON content; the existing `redact_source_storage_path` helper is path-specific. |

**This design closes four open gaps**: per-turn assistant row persistence,
per-tool-call response row persistence, central redaction layer at the
write seam, and the frontend recovery surface.

**Existing exception-handling around `execute_tool()` (clarification for §5.2 implementers).**
Reality-check observed three `except` clauses around the tool-execution call site in `service.py:_compose_loop`:

1. `except ToolArgumentError` (line 867) — Tier-3 boundary signal; **catches and continues** the loop to the next tool call. The compose loop is contractually obligated not to re-raise this (`web/composer/protocol.py:291`, docstring at :299-303).
2. `except (AssertionError, MemoryError, RecursionError, SystemError)` (line 907) — fail-fast classes; **catches and re-raises**. These are interpreter-level invariant violations that must not be wrapped or recovered from.
3. `except Exception as tool_exc` (line 942) — plugin-bug surface; **catches and converts to `ComposerPluginCrashError.capture(tool_exc, state=state, initial_version=initial_version) from tool_exc`**, then re-raises the wrapped error. This is the load-bearing branch that feeds the `_handle_plugin_crash` route helper. Revision 4's §5.2 preserves this wrap explicitly — earlier revisions described only ToolArgumentError + a generic `except Exception` arm, which would have lost the `capture()` wrap and broken the route-layer dispatch path that this entire spec exists to populate. Implementers MUST keep the `ComposerPluginCrashError.capture(...) from tool_exc` line; the new audit-row insertion happens BEFORE the capture-and-raise, not in place of it.

The async / background-job direction is captured in the future epic and
does not constrain this work.

---

## 3. Approach Decisions (inlined mini-ADRs)

| Decision | Choice | Alternatives | Reversibility | Rollback | Cost driver | Review by |
|---|---|---|---|---|---|---|
| Transaction primitive shape (NEW in rev 4) | **TX1.** Single sync block per turn dispatched via `_run_sync`. The compose loop builds redacted payloads in async land, then hands a fully-populated work item to a sync function that opens ONE `engine.begin()` transaction, reserves sequence numbers, writes assistant + N tool + state rows, and commits. `SessionsTransaction` becomes a sync context manager yielded inside that worker. `asyncio.shield` and the `try/except/finally` interleaving disappear. | **TX2.** Async `SessionsTransaction` with a connection pinned to a single worker thread for the txn lifetime (incompatible with SQLite's cross-thread connection rule, complex on PostgreSQL pool semantics). **TX3.** Migrate `SessionsService` to `async_engine` as a prerequisite ticket (correct long-term direction, but materially out of scope for this work). | TX1 is reversible at the API boundary — if a future ticket migrates SessionsService to async_engine, the inner sync block becomes a single async-with body without compose-loop-side changes. | If sync-block design causes unexpected blocking issues, isolate the worker pool dedicated to compose persistence (existing `run_sync_in_worker` pattern accepts a pool selector). | Cost: existing `_run_sync` worker pool capacity. Bench shows N≤8 tool calls per turn at ≤25ms p95 fits comfortably. | RC 5.1 architecture review; concurrency review by Web subsystem owner |
| Tool-response persistence shape | **A1.** One `role='tool'` `chat_messages` row per tool response, correlated to its assistant turn via `tool_call_id`. | A2 (embed responses in assistant row's `tool_calls` JSON) | One-way after first compose run executes — tool rows accumulate. Pre-release the corpus is empty so revisable. | Drop the tool rows + `tool_call_id` column; revert to A2 shape. Costs one schema migration + audit-data-loss event. | Per-call audit row growth (1 + N rows per turn vs 1) | RC 5.1 readiness review |
| Frontend recovery UX | **F2.** Diff-and-confirm modal showing pipeline diff + tool transcript with explicit Apply / Discard buttons. | F1 (auto-apply) destructive of unsaved edits; F3 (reload-to-recover) clunky. | Reversible — frontend-only change, no DB shape implications. | Hide the panel; failure path falls back to existing toast/banner. | Frontend dev + accessibility cost | RC 5.1 frontend review |
| Redaction primitive (rev 4 → rev 5 reshape) | **R5.** **Manifest-keyed dispatch.** Each tool name has exactly one entry in `MANIFEST: Mapping[str, ToolRedaction]` (§4.2.1). Each entry is *either* type-driven (an `argument_model: type[BaseModel]` whose fields carry `Sensitive[T]` annotations — Phase 2 promotes ~6–8 sensitive-touching tools to this shape with `Model.model_validate` dispatch validation; promoted handlers catch `pydantic.ValidationError` and re-raise as `ToolArgumentError` caught at `service.py:2480` for ARG_ERROR routing) *or* declarative (`policy: ToolRedactionPolicy` with explicit `sensitive_argument_keys` / `argument_summarizers` / `known_response_keys` / structured reason). Both shapes set is a construction-time `ValueError` (§4.2.7). There is no `EXEMPT_FROM_TYPE_DRIVEN_REDACTION` ClassVar; there is no `class ComposerTool` base. The walker and the adequacy guard share **one** traversal iterator (§4.2.5). The structural Level-4 leverage (Meadows hierarchy) is the manifest itself: the registration root for redaction policy is a single object reviewed by every consumer. | R4 (rev-4 design): implicit class-based registry assumed via `EXEMPT_FROM_TYPE_DRIVEN_REDACTION: ClassVar[bool]` on a fictional `class ComposerTool` base. Plan-review BLOCKER B1 established the class hierarchy does not exist; `tools.py` dispatches via six function-pointer registries at lines 5250–5314. | One-way after Phase 2 lands. Reverting requires re-introducing a class hierarchy (which has never existed) AND restoring the rev-4 `EXEMPT_FROM_*` ClassVar pattern. | Remove the manifest, re-introduce string-keyed declarative policy at module level, accept that the type-driven Sensitive[T] leverage point is lost. Costs auditability discontinuity and a structural regression to the rev-3 declaration-burden-normalisation feedback loop (rev-3 RSK-02). | Per-tool manifest-entry definition (small, structural; ~6–8 type-driven + ~29–31 declarative across the 37-tool surface) + five-assertion adequacy guard (§4.4) + label-gate CI step (§4.4.5) + content-keyed policy-hash snapshot (§4.4.3) | RC 5.1 security review; redaction-policy weakening lands only with `policy-weaken-justified` PR label and a "Redaction policy weakening rationale" section in the PR body, enforced by the label-gate CI step |
| Tool argument redaction (legacy escape valve) | **R3-LEGACY.** Retained only for tools that cannot or have not yet adopted `Sensitive[T]`. Same shape as rev 3 (`sensitive_argument_keys`, `sensitive_response_keys`, `argument_summarizers`, `known_response_keys`, `handles_no_sensitive_data`, `handles_no_sensitive_data_reason_struct`). The `handles_no_sensitive_data_reason_struct` is now a structured dataclass (see §4.2), not a free-text string, so adequacy-guard checks reason quality programmatically rather than via stop-list pattern-matching. | R3-FREE-TEXT (rev-3 free-text reason) — fragile against plausible-but-wrong reasoning. | One-way per tool — once migrated to `Sensitive[T]`, removing the type annotation requires re-declaring policy. | Per-tool: re-declare ToolRedactionPolicy and remove the type annotation. | Per-tool author declaration burden + structured reason discipline | RC 5.1 security review |
| Migration | **None for chat_messages content; pre-deploy `DELETE` for chat_messages rows on staging.** Pre-release per CLAUDE.md no-legacy policy. New columns ship as part of the schema. Because staging deploys (`elspeth.foundryside.dev` per project memory) have generated `chat_messages` rows under the rev-3-era schema, the new NOT NULL columns (`sequence_no`, `provenance` on `composition_states`, `writer_principal` on `chat_messages`) cannot be added by `ALTER ... ADD COLUMN ... NOT NULL` without a default. The migration plan is therefore: (1) dev: empty database, no migration concern; (2) staging: pre-deploy step `DELETE FROM chat_messages` and `DELETE FROM composition_states` (or `DROP TABLE` + recreate via Alembic) before applying schema; (3) production (when reached): no existing corpus exists yet because composer is still pre-release. | A multi-stage migration with `server_default='0'` followed by backfill assigning sequence numbers in `created_at` order, then `ALTER COLUMN ... DROP DEFAULT`. Rejected because pre-release deployments contain test data only and the no-legacy policy authorises destructive resets at this stage. | One-way — first eval run after deploy produces rows in the new shape. | Delete the chat_messages corpus; pre-release acceptable. | No backward-compat shim cost; pre-deploy DELETE step is a known operation, not a project-novel one. | Once a real corpus exists |

---

## 4. Data Model

### 4.1 Schema changes

#### 4.1.1 `chat_messages`

```python
chat_messages_table = Table(
    "chat_messages",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("role", String, nullable=False),
    Column("content", Text, nullable=False),
    Column("raw_content", Text, nullable=True),
    Column("tool_calls", JSON, nullable=True),
    Column("tool_call_id", String, nullable=True),    # NEW (rev 3)
    Column("sequence_no", Integer, nullable=False),   # NEW (rev 3) — monotonic per session
    Column(                                            # NEW (rev 4) — actor attribution
        "writer_principal",
        String,
        nullable=False,
    ),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("composition_state_id", String, nullable=True),
    Column(                                            # NEW (rev 3); composite FK below (Phase 1)
        "parent_assistant_id",
        String,
        nullable=True,
    ),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_chat_messages_composition_state_session",
    ),
    # Phase 1: composite same-session FK on parent_assistant_id closes the
    # cross-session lineage hole — a tool row in session B cannot reference
    # an assistant row in session A. ON DELETE CASCADE removes child tool
    # rows when the assistant is deleted, preventing orphan tool rows from
    # accumulating in the audit DB.
    ForeignKeyConstraint(
        ["parent_assistant_id", "session_id"],
        ["chat_messages.id", "chat_messages.session_id"],
        name="fk_chat_messages_parent_assistant_session",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "id",
        "session_id",
        name="uq_chat_messages_id_session",
    ),
    CheckConstraint(
        "role IN ('user', 'assistant', 'system', 'tool', 'audit')",
        name="ck_chat_messages_role",
    ),
    CheckConstraint(                                                              # NEW (rev 3)
        "(role = 'tool') = (tool_call_id IS NOT NULL)",
        name="ck_chat_messages_tool_call_id_role",
    ),
    CheckConstraint(                                                              # NEW (rev 3)
        "(role = 'tool') = (parent_assistant_id IS NOT NULL)",
        name="ck_chat_messages_parent_role",
    ),
    CheckConstraint(                                                              # NEW (rev 4)
        # writer_principal is one of the five expected sources; future writers
        # require a destructive session-DB recreation per
        # ``project_db_migration_policy`` (no Alembic in this project) that
        # extends this CHECK with the new value.
        "writer_principal IN ('compose_loop', 'route_user_message', 'route_system_message', 'admin_tool', 'session_fork')",
        name="ck_chat_messages_writer_principal",
    ),
    Index(
        "ix_chat_messages_session_sequence",
        "session_id",
        "sequence_no",
        unique=True,
    ),                                                                             # NEW (rev 3)
    Index(
        "ix_chat_messages_session_tool_call_id",
        "session_id",
        "tool_call_id",
    ),                                                                             # NEW (rev 3)
)
```

`writer_principal` records which subsystem produced the row. It is required
(NOT NULL) because every persisted row must be attributable to a subsystem
identity; an unattributed row violates CLAUDE.md's "explain(recorder, run_id,
token_id)" attributability standard. Permitted values:

- `compose_loop` — the per-turn writes from `_compose_loop` covered by this
  spec (assistant rows, tool rows, and any system rows the loop produces).
- `route_user_message` — the route-layer insert that writes the originating
  user message before the compose loop runs (existing call site at
  `routes.py:1487`).
- `route_system_message` — the route-layer insert for system-prompt seeding
  (existing call site at `routes.py:1883`).
- `admin_tool` — reserved for future admin tooling. No current writer.
- `session_fork` — the Phase 1 fork-copy writer that re-emits inherited
  history into a forked session without losing actor attribution. Distinct
  from the original writer because the row is a fork-time copy, not a
  fresh authoring event.

The `role='audit'` value is an internal-only role for breadcrumb rows
that have no real OpenAI tool-response or assistant parent (LLM-call
audit envelopes, pre-flight redaction failures, etc.). They MUST be
filtered out of any user-facing chat response and any composer
prompt-history rebuild — enforced at
``_is_composer_audit_tool_message`` /
``_composer_conversation_messages`` and the public messages route.

The CHECK constraint pins the enum at the database level; a new writer
identity requires a destructive session-DB recreation per
``project_db_migration_policy`` (no Alembic in this project) that
extends the CHECK, which forces architectural review of any new write
surface. This is the mechanical mitigation for the
"single-writer-per-session structurally enforced" assertion in §5.7 —
instead of relying on convention, the database refuses unrecognised
principal strings.

#### 4.1.2 `composition_states` — provenance discriminator (NEW in rev 4)

```python
composition_states_table = Table(
    "composition_states",
    metadata,
    # ... existing columns ...
    Column(                                                                     # NEW (rev 4)
        "provenance",
        String,
        nullable=False,
    ),
    # ... existing constraints ...
    CheckConstraint(                                                            # NEW (rev 4)
        "provenance IN ('tool_call', 'convergence_persist', "
        "'plugin_crash_persist', 'preflight_persist', "
        "'session_seed', 'session_fork')",
        name="ck_composition_states_provenance",
    ),
)
```

`provenance` records which code path committed the row. Permitted values
(the CHECK constraint is the closed enum — extending it requires a
corresponding amendment of this section):

- `tool_call` — written by `SessionsService._persist_compose_turn`
  (`web/sessions/service.py` ~L866) as part of the atomic per-tool write
  (assistant + tool + state rows in one transaction). This is the ONLY
  value for which the backward-direction INV-AUDIT-AHEAD invariant
  applies: every `('tool_call', version > 0)` row MUST have a
  corresponding `chat_messages` row with `role='tool'` and matching
  `composition_state_id`. **ACTIVE writer in Phase 1.**
- `convergence_persist` — written by `_handle_convergence_error`
  (`web/sessions/routes.py` ~L1135) when `ComposerConvergenceError`
  captures `partial_state` after the compose loop hits its turn budget
  or wall-clock deadline. Distinguishes "state recorded after a
  convergence-budget exhaustion" from "initial seed state on session
  creation" so an auditor counting convergence failures gets the right
  answer. **ACTIVE writer in Phase 1** (promoted from DORMANT by
  elspeth-obs-f217c634aa: writer call site already existed but was
  shadowing under `session_seed` due to a hardcoded label in
  `save_composition_state`; the fix threads `provenance` through the
  public API as a required keyword argument).
- `plugin_crash_persist` — written by `_handle_plugin_crash`
  (`web/sessions/routes.py` ~L1273) when `ComposerPluginCrashError`
  captures `partial_state` after a downstream tool plugin raised mid-
  loop. Distinguishes plugin-crash partial state from convergence
  partial state — different remediations (bug fix vs. retry/budget
  tuning) — so an auditor querying plugin failures gets a clean count.
  **ACTIVE writer in Phase 1** (promoted from DORMANT by
  elspeth-obs-f217c634aa under the same shadowing fix as above).
- `preflight_persist` — written by `_handle_runtime_preflight_failure`
  (`web/sessions/routes.py` ~L1500) when `ComposerRuntimePreflightError`
  captures `partial_state` because the runtime preflight rejected the
  composed pipeline. Distinguishes preflight-detected misconfiguration
  from runtime execution failures so the audit DB can attribute the
  rejection class correctly. **ACTIVE writer in Phase 1** (promoted
  from DORMANT by elspeth-obs-f217c634aa under the same shadowing fix
  as above).
- `session_seed` — initial state row written when a session is created
  with seed configuration (`SessionsService.create_session`;
  branch-from-message reseed in `set_active_state`). Note: two
  additional `routes.py` call sites continue to write `session_seed`
  under their pre-fix behaviour (post-compose state advance in
  `_send_message` and the fork source-storage rewrite in
  `fork_session_at_message`). Both are pre-existing mis-attributions
  separate from elspeth-obs-f217c634aa; relabelling them is a
  follow-up that requires its own observation, spec amendment, and
  governance ticket. **ACTIVE writer in Phase 1.**
- `session_fork` — written by `SessionsService.fork_session_at_message`
  (`web/sessions/service.py` ~L2260) when a user forks a session from
  an earlier message: the helper copies the source session's state
  forward into the new session under this distinct label. Cross-session
  copy-forward is meaningfully distinct from intra-session reseed, and
  the audit DB needs to tell them apart (`session_seed` is a single-
  session originating event; `session_fork` derives from another
  session's prior state). **ACTIVE writer in Phase 1.** Added in the
  Phase 1 plan supersession marker; this revision of §4.1.2 closes the
  drift between that addition and the spec text.

**Why this is the load-bearing addition.** Without `provenance`, the
backward-direction INV-AUDIT-AHEAD post-condition ("every committed
`composition_states` row that resulted from a tool call has a
corresponding `chat_messages` row") could only be evaluated by the test
harness — no schema field witnessed which rows came from the tool-call
path versus the route-helper persist paths. With `provenance`, the
post-condition is a pure SQL predicate:

```sql
SELECT cs.id
  FROM composition_states cs
  LEFT JOIN chat_messages cm
    ON cm.composition_state_id = cs.id AND cm.role = 'tool'
 WHERE cs.provenance = 'tool_call' AND cs.version > 0
   AND cm.id IS NULL;
-- Expected: empty result. Any row is a backward-direction violation.
```

`tests/integration/web/test_inv_audit_ahead_backward.py` (new) runs this
query at the end of every property-test trace and at the end of every
CL-PP-* integration scenario. Closes QA finding F-1.

#### 4.1.3 Partial unique index (unchanged from rev 3, full text retained)

Plus, in a separate DDL block (because SQLAlchemy partial unique
constraints require dialect-specific syntax — see Risk RSK-09):

```sql
-- Partial unique constraint: tool_call_id must be unique within (session_id, tool_role) scope.
-- Both SQLite (3.8.0+) and PostgreSQL parse this identically.
CREATE UNIQUE INDEX uq_chat_messages_tool_call_id
    ON chat_messages (session_id, tool_call_id)
    WHERE role = 'tool';
```

**Database-enforced invariants:**

- `tool_call_id` is non-null iff `role='tool'`. (`ck_chat_messages_tool_call_id_role`.)
- `parent_assistant_id` is non-null iff `role='tool'` (`ck_chat_messages_parent_role`). The same-session composite FK `(parent_assistant_id, session_id) -> (chat_messages.id, chat_messages.session_id)` (`fk_chat_messages_parent_assistant_session`, backed by the `uq_chat_messages_id_session` composite uniqueness target) ensures a tool row's parent assistant lives in the same session, with `ON DELETE CASCADE` so deleting an assistant row removes its tool rows.
- `(session_id, sequence_no)` is unique — every row in a session has a unique sequence number, monotonically increasing in commit order. **Sequence numbers are ordering keys, not counts.** Gaps are permitted (e.g., when an atomic-pair transaction rolls back after the next free sequence number was reserved). The property test post-condition asserts strict monotonicity within a session, NOT density. Closes architect H-3.
- `(session_id, tool_call_id)` is unique among `role='tool'` rows. Cross-turn collisions (same `tool_call_id` reused by the LLM provider in a different turn) are rejected as a Tier-3 input-validation failure: the compose loop crashes the request rather than silently mis-correlating.
- `writer_principal` is one of the four enumerated values; the database refuses any other string at write time.
- `composition_states.provenance` is one of the six enumerated values (see §4.1.2); backward-direction INV-AUDIT-AHEAD applies only to rows with `provenance='tool_call'`.

`composition_state_id` FK behaviour — see §4.5.

### 4.2 Redaction primitives — manifest-keyed dispatch with type-driven and declarative shapes

**Location.** `src/elspeth/web/composer/redaction.py` (L3, alongside the
existing `redact_source_storage_path` helper). Tools are L3; their
redaction primitives belong in the same layer.

**Architecture (rev 5).** The unit of redaction policy is **the tool
name**, not a Python class. There is no `class ComposerTool` base;
`tools.py` dispatches via six function-pointer registries at
`src/elspeth/web/composer/tools.py:5250–5314`, and that dispatch shape
is preserved. Redaction layers a **manifest** —
`MANIFEST: dict[str, ToolRedaction]` keyed by tool name — alongside it.
This mirrors the existing `_TOOL_REQUIRED_PATHS: dict[str, ...]`
precedent at `src/elspeth/web/composer/service.py:702`: the same project
idiom applied to a different concern. The manifest is the single
registration root for redaction; the adequacy guard walks it, the
runtime walker dispatches through it, and the policy-hash snapshot
covers every entry.

Each `ToolRedaction` entry chooses **exactly one** of two shapes:

- **Type-driven** — the entry carries `argument_model: type[BaseModel]`
  (and optionally `response_model: type[BaseModel]`). The model's fields
  carry `Sensitive[T]` annotations (§4.2.2) where redaction is required;
  the runtime walker uses Pydantic `model_fields[name].metadata` to
  detect markers. Phase 2 (§11) promotes ~6–8 sensitive-touching tools
  to this shape. Promoted tools' handlers also gain
  `Model.model_validate(arguments)` at the dispatch boundary; promoted
  handlers catch `pydantic.ValidationError` and re-raise as
  `ToolArgumentError` (per `tools.py:2668–2801` pattern), which is
  caught at `service.py:2480` and routes to `ARG_ERROR` — so the
  type-driven primitive delivers two guarantees: dispatch-time validation
  AND persistence-time redaction.
- **Declarative** — the entry carries `sensitive_argument_keys`,
  `argument_summarizers`, `known_response_keys`, and (when
  `handles_no_sensitive_data=True`) a structured
  `HandlesNoSensitiveDataReason` (§4.2.3). Phase 2 covers the remaining
  ~29–31 tools this way; their dispatch surface remains
  `arguments: dict[str, Any]` because their argument schema is
  structural (graph IDs, node names, plugin-key strings) and does not
  benefit from a redaction-specific Pydantic model.

Declaring both shapes in one entry is a **construction-time error**
(§4.2.7). The adequacy guard rejects it; the manifest dataclass
`__post_init__` raises `ValueError`.

**Walker is a Tier-3 → Tier-1 boundary** (§4.2.6). The walker accepts
the LLM's raw decoded JSON `dict` (Tier-3 input — may carry malformed
keys, unexpected types, prompt-injection payloads) and produces the
canonical redacted dict that lands in `chat_messages.tool_calls` /
`chat_messages.content` (Tier-1 audit row — must be pristine). The
boundary discipline is: **fail-closed redact** unknown keys with a
fixed sentinel, count, and continue (Tier-3 inputs do not crash
audit); but **internal-invariant violations** (manifest entry missing
for a dispatched tool name, summarizer raises, summarizer returns
non-`str`) **crash via `AuditIntegrityError`** (Tier-1 invariants must
be honoured; CLAUDE.md plugin-ownership policy applies because the
manifest and summarizer are system code).

#### 4.2.1 `ToolRedaction` — the manifest-entry dataclass (NEW in rev 5)

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from elspeth.contracts.freeze import freeze_fields

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolRedaction:
    """One manifest entry. Each entry is keyed by tool name in MANIFEST.

    Exactly one of the two shapes must be populated:
      • type-driven  — argument_model is not None
      • declarative  — argument_model is None AND policy is not None

    Both populated → ValueError (precedence is undefined; use one shape).
    Neither populated → ValueError (every tool must have a redaction
    declaration; the adequacy guard cannot consult what doesn't exist).

    response_model is only valid alongside argument_model. Declarative
    entries express response shape via policy.known_response_keys.
    """

    argument_model: type["BaseModel"] | None = None
    response_model: type["BaseModel"] | None = None
    policy: "ToolRedactionPolicy | None" = None

    def __post_init__(self) -> None:
        type_driven = self.argument_model is not None
        declarative = self.policy is not None

        if type_driven and declarative:
            raise ValueError(
                "ToolRedaction declared both argument_model and policy; "
                "each manifest entry must choose exactly one shape. "
                "If a tool has a Pydantic argument model with Sensitive[T] "
                "annotations, the model is the single source of truth — "
                "remove the policy. If the argument surface is purely "
                "structural and does not benefit from Sensitive[T], "
                "remove the argument_model and declare the policy."
            )
        if not type_driven and not declarative:
            raise ValueError(
                "ToolRedaction declared neither argument_model nor policy; "
                "every manifest entry must declare its redaction shape. "
                "If the tool genuinely handles no sensitive material, set "
                "policy=ToolRedactionPolicy(handles_no_sensitive_data=True, "
                "handles_no_sensitive_data_reason_struct=...) — the "
                "structured reason is part of the audit trail."
            )
        if self.response_model is not None and not type_driven:
            raise ValueError(
                "response_model requires argument_model to also be set "
                "(declarative entries express response shape via "
                "policy.known_response_keys)."
            )


# The manifest itself. Module-level so the adequacy guard, the runtime
# walker, and the snapshot test consume the same object. Construction
# happens at import time; failure raises immediately and is caught by
# pytest collection or the import statement in the dispatcher.
MANIFEST: Mapping[str, ToolRedaction] = MappingProxyType({
    "set_source": ToolRedaction(argument_model=SetSourceArgumentsModel, ...),
    "list_sources": ToolRedaction(policy=ToolRedactionPolicy(
        handles_no_sensitive_data=True,
        handles_no_sensitive_data_reason_struct=...,
    )),
    # ... one entry per tool name; enumeration in §4.7 ...
})
```

#### 4.2.2 `Sensitive[T]` — type-driven redaction metadata (rev 4, refined for rev 5)

`Sensitive[T]` remains the field-level annotation that drives mechanical
redaction inside type-driven manifest entries. Rev 5 refines its role:
the marker is **redaction-only metadata**, not a dispatch-validation
gate by itself. The dispatch-validation guarantee comes from
`Model.model_validate(arguments)` invoked by the promoted handler
(§4.2.1, type-driven shape); the `Sensitive[T]` marker tells the walker
*which fields of that validated model* to redact.

```python
from collections.abc import Callable
from typing import Annotated, Any, TypeVar

T = TypeVar("T")


class _SensitiveMarker:
    """Annotated metadata marker. Presence on a Pydantic field of a
    manifest-entry argument_model or response_model indicates the field's
    value MUST be redacted at the persistence boundary. The tool author
    has no opt-out path — redaction is mechanical, not declarative.

    summarizer: optional per-field replacement function. If None, the
        sentinel '<redacted>' is substituted. If present, the function
        receives the original value and returns the replacement string.
        See §4.2.6 for the summarizer contract (must not raise; must
        return str — violations crash via AuditIntegrityError).
    """

    __slots__ = ("summarizer",)

    def __init__(self, summarizer: Callable[[Any], str] | None = None) -> None:
        self.summarizer = summarizer


def Sensitive(  # noqa: N802 — capitalised to read as a type alias at use sites
    *, summarizer: Callable[[Any], str] | None = None
) -> _SensitiveMarker:
    """Field-level annotation requesting redaction at the persistence
    boundary. Used as Pydantic field metadata via ``Annotated``.

    Example (the set_source tracer-bullet model in §4.7):

        from typing import Annotated
        from pydantic import BaseModel

        class SetSourceArgumentsModel(BaseModel):
            plugin: str    # not sensitive — plugin name is structural
            options: Annotated[
                dict[str, Any],
                Sensitive(summarizer=redact_source_storage_path),
            ]
            on_success: str | None = None
            label: str | None = None    # not sensitive
    """
    return _SensitiveMarker(summarizer=summarizer)
```

**`extra="forbid"` is required on all promoted argument models
(rev-2 M_adequacy_mechanical_enforcement).** Every type-driven manifest
entry's `argument_model` MUST set `model_config = ConfigDict(extra=
"forbid")`. Pydantic 2.x defaults to `extra="ignore"`, which silently
drops extra keys — creating a discrepancy between `arguments_canonical`
(which records the raw LLM arguments including the extra key) and what
the redaction walker walked. `extra="forbid"` eliminates this
discrepancy by rejecting unexpected keys at `model_validate` time,
routing them to `ARG_ERROR` (Tier-3 quarantine). The fifth adequacy
assertion (§4.4.2) enforces this at CI time.

Example:

```python
from pydantic import BaseModel, ConfigDict

class SetSourceArgumentsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plugin: str
    options: Annotated[dict[str, Any], Sensitive(summarizer=redact_source_storage_path)]
    on_success: str | None = None
    label: str | None = None
```

**Why this is a Level-4 (Meadows hierarchy) intervention rather than a
Level-5 / Level-6 mitigation.** For tools with type-driven entries the
redaction policy is expressed in the type system. There is no "I declare
this tool has no sensitive data" flag for fields whose model uses
`Sensitive[T]`; the persistence layer reads the model's `Annotated`
metadata and acts on it. The declaration-burden-normalisation feedback
loop (rev-3 RSK-02) cannot run because there is nothing to declare. New
tools that adopt the type-driven shape inherit the redaction guarantee
mechanically.

The walker's recursion semantics (which container types descend, in
what order, with what marker-position rules) are defined once in the
shared traversal iterator (§4.2.5) so that the adequacy guard
(CI-time) and the runtime walker (persistence-time) cannot diverge.

#### 4.2.3 Declarative legacy escape valve — `ToolRedactionPolicy` and `HandlesNoSensitiveDataReason` (rev 4, refined for rev 5)

For tools whose argument surface is purely structural (graph IDs, node
names, plugin-key strings) and does not benefit from a
redaction-specific Pydantic model, the rev-4 declarative shape is
retained as the **declarative manifest-entry shape** — *not* as a
"legacy" fallback to be migrated. Some tools genuinely have nothing
sensitive to declare; for those, the declarative shape with
`handles_no_sensitive_data=True` and a structured reason is the correct
permanent representation. Phase 2 (§11) declares one entry per tool;
the ~29–31 tools without sensitive arguments use this shape.

**Summarizer purity requirement (rev-2 BLOCKER_C).** Summarizers
registered in manifest entries (both type-driven `Sensitive[T]`
summarizers and declarative `argument_summarizers` values) MUST be
**pure functions of their argument**. Each summarizer is a callable
taking one argument and returning a `str`. The summarizer's behaviour
MUST NOT depend on module-level mutable state, instance state, or any
closure-captured variable that may change after the summarizer is
registered in `MANIFEST`. **Rationale:** the policy snapshot hashes
the summarizer's callable identity (fully-qualified name). Replacement
of a summarizer with a new callable correctly flips the hash;
in-place behavioural change via mutated closed-over state is a known
false-negative class for the snapshot mechanism (see §4.4.3). This is
acceptable because the no-legacy-code policy combined with the
offensive-programming discipline means closure-captured mutable state
is itself a code smell — but reviewers and snapshot-tooling cannot
detect it automatically. Implementers MUST NOT introduce module-level
state that summarizers close over.

Rev 5 refinements over rev 4:

1. The shape lives **inside** a manifest entry's `policy` field
   (§4.2.1), not on a tool class. There is no `EXEMPT_FROM_*` ClassVar.
2. **Mass-copy uniqueness** is enforced by the adequacy guard
   (§4.4.4): no two tools' `why_arguments_safe` (or `why_responses_safe`)
   strings may exact-match. Closes W7.
3. **`last_reviewed_iso` is replaced by `policy_text_hash`** — a
   SHA-256 over the structured reason's textual content. Review is
   triggered by *content change*, not by calendar tick. The adequacy
   guard fails if the manifest's `policy_text_hash` differs from the
   committed snapshot value without the corresponding
   `policy-weaken-justified` PR label. Closes W2 (calendar
   synchronicity at migration → mass batch expiry → date-bump ritual).

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class HandlesNoSensitiveDataReason:
    """Structured justification required for handles_no_sensitive_data=True.
    Replaces rev-3's free-text string. Adequacy-guard validates each field.

    sensitive_data_locations: where sensitive material actually lives if
        not in this tool's arguments or responses (e.g., 'server-side
        secret resolver', 'request headers stripped before tool dispatch').
        Must be non-empty.

    why_arguments_safe: prose explanation of why every argument, including
        any string-typed ones, is safe to persist verbatim. Adequacy-guard
        checks length >= 32 chars and that the text does not exact-match
        any other tool's why_arguments_safe (mass-copy uniqueness, §4.4.4).

    why_responses_safe: same as above, for responses.
    """

    sensitive_data_locations: tuple[str, ...]
    why_arguments_safe: str
    why_responses_safe: str

    def __post_init__(self) -> None:
        if not self.sensitive_data_locations:
            raise ValueError(
                "sensitive_data_locations is empty; declare at least one location "
                "where sensitive material related to this tool exists, OR migrate "
                "the tool's arguments to a Pydantic model with Sensitive[T] "
                "annotations (the type-driven manifest-entry shape, §4.2.1)."
            )
        for label, value in (("why_arguments_safe", self.why_arguments_safe),
                             ("why_responses_safe", self.why_responses_safe)):
            if len(value.strip()) < 32:
                raise ValueError(
                    f"{label} is shorter than 32 characters; the structured "
                    f"justification requires concrete reasoning, not a placeholder."
                )
        freeze_fields(self, "sensitive_data_locations")


@dataclass(frozen=True, slots=True)
class ToolRedactionPolicy:
    """Declarative redaction policy. Used inside the `policy` field of a
    manifest entry whose argument surface is purely structural (no
    Pydantic argument model declared). The type-driven shape (§4.2.1
    with argument_model set) is preferred for any tool that has, or
    would benefit from, a redaction-bearing Pydantic argument model.

    sensitive_argument_keys: keys in the tool-call argument dict whose
        values must be replaced before the tool call is persisted to
        chat_messages.tool_calls JSON.

    sensitive_response_keys: keys in the tool's response dict whose values
        must be replaced before the tool's response is persisted to
        chat_messages.content (as JSON).

    known_response_keys: REQUIRED when handles_no_sensitive_data=False.
        The complete set of keys the tool may legitimately emit in its
        response. At persistence time, every key in the actual response
        is checked against this allowlist; unknown keys are fail-closed
        redacted with the FIXED sentinel '<redacted-unknown-response-key>'
        (no length disclosure — closes W6) and the counter
        composer.redaction.unknown_response_key_total increments.
        Closes security I-1 (response-shape drift defeating static policy).

    argument_summarizers: optional per-key replacement functions for keys
        in sensitive_argument_keys. Summarizer contract (§4.2.6): the
        function MUST NOT raise on any argument value reachable through
        the dispatch boundary; it MUST return str. Violations crash via
        AuditIntegrityError (system-code discipline, CLAUDE.md plugin
        ownership).

    handles_no_sensitive_data: explicit "this tool reviewed and asserts
        no sensitive material in arguments or responses" flag.

    handles_no_sensitive_data_reason_struct: structured justification
        REQUIRED when handles_no_sensitive_data=True.

    NOTE on freeze: argument_summarizers values are Callables; deep_freeze
    passes Callables through unchanged (verified against
    src/elspeth/contracts/freeze.py:78). Identity-equality of summarizer
    callables is the policy contract.
    """

    sensitive_argument_keys: tuple[str, ...] = ()
    sensitive_response_keys: tuple[str, ...] = ()
    known_response_keys: tuple[str, ...] = ()
    argument_summarizers: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)
    handles_no_sensitive_data: bool = False
    handles_no_sensitive_data_reason_struct: HandlesNoSensitiveDataReason | None = None

    def __post_init__(self) -> None:
        # Validators run BEFORE freeze_fields so they read mutable state.
        # If any raise, the dataclass __init__ raises and the object is
        # never returned — atomic construction failure.

        orphan_summarizers = set(self.argument_summarizers) - set(self.sensitive_argument_keys)
        if orphan_summarizers:
            raise ValueError(
                f"argument_summarizers keys {sorted(orphan_summarizers)} are not declared in "
                f"sensitive_argument_keys; orphan summarizers indicate a policy bug."
            )

        if self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason_struct is None:
            raise ValueError(
                "handles_no_sensitive_data=True requires a non-None "
                "handles_no_sensitive_data_reason_struct. Build a "
                "HandlesNoSensitiveDataReason instance with concrete fields."
            )
        if not self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason_struct is not None:
            raise ValueError(
                "handles_no_sensitive_data_reason_struct is only meaningful "
                "when handles_no_sensitive_data=True."
            )

        if not self.handles_no_sensitive_data and not self.known_response_keys:
            raise ValueError(
                "known_response_keys must be declared (non-empty) when "
                "handles_no_sensitive_data=False. The allowlist defends "
                "against response-shape drift; unknown keys at persistence "
                "time are fail-closed redacted with a fixed sentinel."
            )

        freeze_fields(
            self,
            "sensitive_argument_keys",
            "sensitive_response_keys",
            "known_response_keys",
            "argument_summarizers",
        )
```

#### 4.2.4 `RedactionTelemetry` — typed Protocol for OTel emissions (NEW in rev 5)

The walker increments OTel counters at three points: unknown response
keys (fail-closed redact), summarizer-error path (does not exist in rev
5; the code path crashes — see §4.2.6 and §9 RSK-03), and
manifest-entry lookup (records each dispatch by shape). Rev 4 left the
telemetry surface as duck-typed `telemetry=None` parameters. Rev 5
defines a `Protocol` so the type checker enforces the contract;
this closes W4. The walker accepts a `RedactionTelemetry` instance,
never `None`. A no-op implementation is provided for tests.

```python
from typing import Protocol


class RedactionTelemetry(Protocol):
    """OTel counter surface for the redaction walker.

    Implementations live in src/elspeth/web/composer/telemetry.py
    (production: wraps the project's structured-counter helper). Tests
    use NoopRedactionTelemetry (counts to a dict; assertable). The
    walker never accepts None — callers must pass a real telemetry
    instance, even in tests.
    """

    def unknown_response_key_redacted(self, *, tool_name: str) -> None:
        """Counter: composer.redaction.unknown_response_key_total."""
        ...

    def manifest_dispatch(self, *, tool_name: str, shape: str) -> None:
        """Counter: composer.redaction.manifest_dispatch_total{shape}.

        shape ∈ {'type_driven', 'declarative'}. Records walker
        dispatch by manifest-entry shape; useful for migration
        progress visibility.
        """
        ...

    def summarizer_error(self, *, tool_name: str) -> None:
        """Counter: composer.redaction.summarizer_errors_total.

        Incremented immediately BEFORE raising AuditIntegrityError on
        summarizer exception or non-str return (rev-2
        M_telemetry_implementation). The counter fires before the raise
        so OTel scrapes see it even if the crash kills the request
        before other telemetry flushes.
        """
        ...
```

#### 4.2.5 Shared traversal iterator (NEW in rev 5)

The adequacy guard (§4.4) and the runtime walker (§4.2.6) share **one**
traversal iterator. This is the structural mitigation for plan-review
B2 (the rev-4 spec promised recursion into `list[BaseModel]`,
`dict[str, BaseModel]` etc., but the walker omitted it; the adequacy
guard could not detect the gap because the two were defined
independently).

The iterator is a generator over a Pydantic model class. It yields one
`TraversalNode` per field encountered, descending parametrically into
container types. Both consumers walk the same nodes:

```python
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class TraversalNode:
    """One field encountered while walking a model schema.

    path: dotted path from the root model to this field, e.g.
        "options.path" or "items[*].secret". '[*]' marks list-element
        descent; '{*}' marks dict-value descent. Adequacy-guard error
        messages cite these paths verbatim.

    field_type: the resolved Python type of the field (post-Annotated
        unwrap). For list[X], dict[str, X], tuple[X, ...] etc. the
        iterator descends into X and the path captures the indirection.

    metadata: tuple of all metadata objects from any Annotated[...]
        wrapping. Walker checks isinstance(m, _SensitiveMarker); guard
        checks the same; both reach the same decision.

    value_provider: callable that returns the value at this path given
        the root dict. None for the adequacy guard (no value). Walker
        passes the LLM dict as the root and the provider extracts.
        Decoupling field-introspection from value-extraction is what
        lets one iterator serve both consumers.
    """

    path: str
    field_type: type
    metadata: tuple[Any, ...]
    value_provider: Callable[[dict[str, Any]], Any] | None = None


def walk_model_schema(
    model: type[BaseModel],
    *,
    with_values: bool = False,
) -> Iterator[TraversalNode]:
    """Yield one TraversalNode per field, descending into:

      • BaseModel subclasses (recurse into model_fields)
      • list[X] where X is BaseModel or non-scalar (descend into X)
      • dict[str, X] where X is BaseModel or non-scalar (descend into X)
      • tuple[X, ...] where X is BaseModel or non-scalar (descend into X)
      • Optional[X] / Union[*, X, *] (descend into every non-None arm)
      • Annotated[T, *, marker, *] regardless of marker position in the
        Annotated args (the iterator scans the full tuple, not just
        args[0])

    Does NOT descend into:
      • scalar types (int, float, bool, str, bytes, datetime, enum)
      • Any, object (the adequacy guard fails on these per §4.4.1)
      • cycles (Pydantic's model-resolution depth limit applies)

    with_values=False: yields nodes with value_provider=None (adequacy
    guard).
    with_values=True: yields nodes with value_provider set (walker).
    """
    ...
```

The iterator is **the** definition of which container types
participate in redaction. Adding a new container type (e.g. a future
`frozenset[BaseModel]`) is a single edit to this function and is
covered by both consumers automatically. Test coverage in §8.1 includes
all ten container shapes (original six plus four rev-2 additions:
duplicate-marker error, `list[list[BaseModel]]` compound nesting,
`Annotated[T, Field(...), Sensitive()]` with `FieldInfo` in metadata,
three-arm Union with non-BaseModel arms) plus
marker-position-not-first-in-Annotated plus nested-three-levels.

#### 4.2.6 Walker semantics — Tier-3 → Tier-1 boundary disposition (NEW in rev 5)

Two redaction entry points consume the manifest:

```python
def redact_tool_call_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Produce the redacted arguments dict that lands in
    chat_messages.tool_calls JSON. Called once per tool invocation
    inside the per-turn audit-write block (§5.2).

    Tier-3 → Tier-1 boundary discipline:
      • Unknown manifest-entry shape branches → crash (internal-invariant).
      • Manifest entry missing for tool_name → crash AuditIntegrityError
        (registry-consistency invariant; the dispatch-time check at
        compose-loop entry asserts MANIFEST.keys() == registered tool
        names exactly).
      • Type-driven entry: validate against argument_model. Promoted
        handlers MUST catch pydantic.ValidationError and re-raise as
        ToolArgumentError (pattern at tools.py:2668, 2761, 2767, 2773,
        2787, 2801); ToolArgumentError is then caught at service.py:2480
        and routes to ARG_ERROR. A bare ValidationError escaping a
        handler hits the catch-all at service.py:2564 and surfaces as
        ComposerPluginCrashError → HTTP 500, which is the wrong
        disposition for Tier-3 input. The walker does not swallow it.
      • Declarative entry: walk arguments by sensitive_argument_keys.
        Keys not in the dict are no-ops; keys present are summarized
        or sentinel-substituted.
      • Summarizer raises → immediately call
        telemetry.summarizer_error(tool_name=tool_name) BEFORE raising
        AuditIntegrityError chained from the underlying exception
        (registered in TIER_1_ERRORS so `except Exception` cannot
        swallow). System-code discipline: summarizer is contractually
        required to never raise. Counter fires before raise so OTel
        scrapes see it even when the request dies.
      • Summarizer returns non-str → same telemetry.summarizer_error
        call BEFORE AuditIntegrityError with a typed message indicating
        which tool, which key, which return type.
    """
    ...


def redact_tool_call_response(
    tool_name: str,
    response: dict[str, Any],
    *,
    telemetry: RedactionTelemetry,
) -> dict[str, Any]:
    """Produce the redacted response dict that lands in
    chat_messages.content (as JSON). Tier-3 → Tier-1 boundary same as
    above, plus:

      • Type-driven entry with response_model declared: walk via
        model_fields metadata. Keys not in the model are unknown.
      • Declarative entry: walk by sensitive_response_keys ∪
        known_response_keys allowlist.
      • Unknown response key (not in any declared set) → substitute
        the FIXED sentinel '<redacted-unknown-response-key>'; emit
        telemetry.unknown_response_key_redacted(tool_name=...).
        No length disclosure (closes W6 / spec §8.1 RSK-03 weak echo).
        Continue walking (Tier-3 input does not crash audit).
    """
    ...
```

The boundary discipline is summarised:

| Failure mode | Trust tier | Walker disposition |
|---|---|---|
| Unknown response key (key in input dict, not in any declared set) | Tier-3 (LLM-supplied response data) | Fixed-sentinel substitute, counter increment, continue |
| **Unknown tool name (LLM hallucination)** | **Tier-3** | **Walker not invoked.** Dispatcher fall-through at `tools.py:5731` returns `ToolResult(success=False, data={"error": "Unknown tool: <name>"})` without raising. `dispatch_with_audit` enters its SUCCESS branch (no exception caught) and records `ComposerToolStatus.SUCCESS` with the failure payload in `result_canonical`. The compose loop continues; the LLM receives the failure payload as a `role=tool` message and can self-correct. The audit record carries the full semantic outcome. **This is the canonical SUCCESS-with-semantic-failure pattern** — documented in `ComposerToolStatus.SUCCESS`'s docstring (`contracts/composer_audit.py:34-37`). Pinned by `test_compose_loop_unknown_tool_name.py::TestUnknownToolNameComposeLoopAuditShape::test_unknown_tool_name_audit_shape`. |
| Argument fails Pydantic validation (type-driven) | Tier-3 | Walker not invoked; promoted handler catches `pydantic.ValidationError` and re-raises as `ToolArgumentError` (per `tools.py:2668–2801`); `ToolArgumentError` caught at `service.py:2480`; ARG_ERROR path runs |
| Manifest entry missing for dispatched tool name | Tier-1 (registry consistency) | `AuditIntegrityError` |
| Summarizer raises | Tier-1 (system code; CLAUDE.md plugin policy) | `telemetry.summarizer_error(tool_name=…)` fired; then `AuditIntegrityError` chained |
| Summarizer returns non-str | Tier-1 | `telemetry.summarizer_error(tool_name=…)` fired; then `AuditIntegrityError` |
| Argument summarizer key declared but argument key absent in input | Tier-3 | No-op (key absence is not a fault) |

#### 4.2.7 Sensitive[T] vs declarative precedence (NEW in rev 5)

Each manifest entry chooses **exactly one** shape (§4.2.1
`__post_init__`). Both shapes set is a `ValueError` at construction
time; neither set is also a `ValueError`. There is no precedence rule
to define because there is no path in which both apply. Closes W8.

Rationale: a manifest entry with `argument_model=X` AND a `policy` with
`sensitive_argument_keys=("y",)` would describe two parallel,
potentially divergent policies for the same arguments. The walker
would have to define which wins; the adequacy guard would have to walk
both; the snapshot would have to capture both hashes. Each of those is
a divergence-risk surface. Forbidding the configuration eliminates the
class of bugs.

#### 4.2.8 `arguments_canonical` — posture (a): Intentional raw (NEW in rev-5 rev-2 iteration)

`ComposerToolInvocation.arguments_canonical` (defined at
`src/elspeth/contracts/composer_audit.py`) stores the RFC 8785
canonical JSON of the raw LLM-supplied arguments, computed by
`begin_dispatch` / `begin_dispatch_or_arg_error` at `service.py:1930`
**BEFORE redaction runs**. Phase 2 builds `redacted_assistant_tool_calls`
and `redacted_tool_rows` for `chat_messages`, but `arguments_canonical`
is a separate persistence surface (MCP JSONL sidecar + web-composer
`BufferingRecorder`).

**Selected posture: (a) Intentional raw.** The `arguments_canonical`
field is NOT redacted. Rationale:

- `arguments_canonical` is the integrity-hash input for
  `arguments_hash` in `ComposerToolInvocation`. Redacting it would
  break the "hash verifies what the LLM actually sent" property — a
  Tier-1 invariant documented in the `ComposerToolInvocation` docstring:
  "a verifier reading this record back from durable storage MUST
  recompute the digest and crash on mismatch."
- The redacted view lands in `chat_messages.tool_calls` /
  `chat_messages.content` per §4.2 — that is the in-DB conversation
  reconstruction surface.
- The audit sidecar (MCP JSONL + `BufferingRecorder`) is a narrower-
  visibility surface that retains forensic completeness.

**Phase 3 MUST NOT thread redacted arguments through `begin_dispatch`.
Redaction is applied ONLY to `chat_messages` surfaces.** Tier-1 access
controls on the audit sidecar are the load-bearing protection for this
surface; Phase 3 must verify they are appropriate before wiring (see
§11 Phase 3 preconditions).

### 4.3 Sentinel rules

- Plain sensitive key → value replaced by literal string `"<redacted>"`.
- Key with summarizer → value replaced by `summarizer(original_value)`.
  Example: `lambda b: f"<inline-blob:{len(b)}-bytes>"`.
- Existing `redact_source_storage_path` continues to handle source paths
  in `partial_state`. Unchanged by this work.

### 4.4 Adequate-redaction guard

The adequacy guard is the CI-time enforcement that **every tool name
present in the six dispatch registries at `tools.py:5250–5314` has
exactly one corresponding entry in `MANIFEST` (§4.2.1)** and that the
entry's redaction declaration covers every field where redaction may be
required. Revision 5 re-derives the guard against the manifest
architecture; revision 4's class-walk approach is replaced because the
class hierarchy it assumed does not exist.

The guard is a single pytest module
(`tests/unit/web/composer/test_adequacy_guard.py`) with five
assertions:

1. **Registry-manifest set equality** (§4.4.1).
2. **Per-entry shape walk** (§4.4.2).
3. **Mass-copy uniqueness** of declarative reasons (§4.4.4).
4. **Policy-hash snapshot equality** (§4.4.3).
5. **`extra="forbid"` on type-driven entries** (§4.4.2 fifth assertion;
   rev-2 M_adequacy_mechanical_enforcement M.2).

The five assertions share the same shared traversal iterator (§4.2.5)
that the runtime walker uses. Future schema changes touch the iterator
and both consumers atomically (closes plan-review B2: walker-vs-guard
divergence).

#### 4.4.1 Registry-manifest set equality

Tool names are the join key. The guard asserts:

```python
def test_manifest_covers_every_dispatch_registry_entry() -> None:
    """Every name in any of the six dispatch registries has exactly one
    MANIFEST entry. No orphans on either side. Source of truth: the
    actual dispatch dicts at src/elspeth/web/composer/tools.py:5250–5314.

    The registry-side names come from a small helper that imports the
    six dicts and unions their keys, plus the three names with extended
    signatures handled inline in execute_tool() (preview_pipeline,
    diff_pipeline, set_pipeline). The compose-loop advisor escape-hatch
    name 'request_advisor_hint' is intercepted before execute_tool()
    (service.py:2070 onward) and is not in the dispatch registries; it
    is enumerated separately as a manifest entry.
    """
    registry_names = _collect_registry_names()  # union of six dicts + 3 inline + advisor
    manifest_names = set(MANIFEST.keys())

    only_in_registry = registry_names - manifest_names
    only_in_manifest = manifest_names - registry_names
    assert not only_in_registry, (
        f"Tools registered for dispatch but missing MANIFEST entry: "
        f"{sorted(only_in_registry)}. Add a ToolRedaction entry."
    )
    assert not only_in_manifest, (
        f"MANIFEST entries with no dispatch registration: "
        f"{sorted(only_in_manifest)}. Either delete the entry or add "
        f"the tool to its appropriate dispatch dict."
    )
```

The test reads the dispatch dicts from `tools.py` directly. A new tool
added to `_DISCOVERY_TOOLS` (or any registry) without a manifest entry
fails CI on the next test run. A manifest entry without a dispatch
registration also fails — this catches stale entries left behind after
a tool is renamed or removed. Closes plan-review B1 (precondition
gap).

#### 4.4.2 Per-entry shape walk

For each manifest entry the guard traverses the entry's declared
shape:

| Entry shape | Guard action |
|---|---|
| `argument_model is not None` (type-driven) | Walk `argument_model` via `walk_model_schema(model, with_values=False)` (§4.2.5). For each `TraversalNode`, apply the field-shape rule below. If `response_model` is also set, walk it too. Assert `entry.argument_model.model_config.get("extra") == "forbid"` (fifth adequacy assertion — see below). |
| `policy is not None` (declarative) | If `policy.handles_no_sensitive_data=True`: pass (the structured reason covers it; no schema to walk). If `False`: assert `policy.known_response_keys` is non-empty AND `policy.sensitive_argument_keys ⊆ known_response_keys ∪ argument_summarizers.keys()` (internal consistency), and that no orphan summarizers exist. The guard does NOT perform AST inspection of handler internals — that is implementation coupling. Tools requiring mechanical key-coverage guarantees MUST be promoted to type-driven Pydantic argument models with `extra="forbid"` (see fifth adequacy assertion below). |

The field-shape rule for type-driven entries:

| Field shape | Adequacy rule |
|---|---|
| `Annotated[T, *, Sensitive(...), *]` (marker anywhere in the annotation tuple, any `T`) | Pass — type-driven redaction will handle it at runtime. |
| `BaseModel` subclass | The iterator already descends; nodes for nested fields are produced and rule-checked individually. |
| `list[T]` / `dict[str, T]` / `tuple[T, ...]` where `T` is `BaseModel` or non-scalar | The iterator descends into `T`. If `T` is itself `Any` or `object`, fail. |
| `Optional[T]` / `Union[A, B, ...]` | Iterator descends into every non-`None` arm. Each arm rule-checked individually. If any arm is `Any`/`object`, fail. |
| `Any`, `object`, untyped fields | **Fail.** `Any`-typed fields are inspection-resistant; the model author must narrow the type. |
| `str`, `bytes`, `Annotated[str, ...]` *without* `Sensitive` | Pass IF the tool's manifest entry has `argument_model` only (the model author asserts the field is non-sensitive by omission of the marker). The model is reviewed via the policy-hash snapshot (§4.4.3); a removed `Sensitive()` annotation flips the snapshot and triggers CODEOWNERS / label review. |
| `int`, `float`, `bool`, `datetime`, enum members | Pass — scalars not covered by the redaction policy. |

The iterator descends parametrically; adding a new container type
(e.g. `frozenset[BaseModel]`) is a single edit (§4.2.5) covered by
both the guard and the walker.

**Fifth adequacy assertion — `extra="forbid"` on type-driven entries
(rev-2 M_adequacy_mechanical_enforcement).** For every type-driven
manifest entry, the guard asserts
`entry.argument_model.model_config.get("extra") == "forbid"`. Pydantic
2.x defaults to `extra="ignore"`, which silently drops extra keys from
`arguments_canonical`. If the LLM supplies an extra key, it enters
`arguments_canonical` (the integrity-hash input) but is absent from
the validated model's fields — creating a discrepancy between what the
audit sidecar recorded and what the redaction walker walked. `extra=
"forbid"` makes the model reject unexpected keys at `model_validate`
time, converting the discrepancy to a `ValidationError` routed to
`ARG_ERROR` (Tier-3 quarantine). All promoted argument models MUST set
`model_config = ConfigDict(extra="forbid")` (§4.2.2 requirement).
The snapshot includes this as a field so the hash flips on `extra`
policy changes.

#### 4.4.3 Policy-hash snapshot test (rev 4 → broadened in rev 5)

Revision 4's snapshot covered only legacy-policy tools
(`EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True`). Rev 5 broadens the
snapshot to **every manifest entry**, type-driven and declarative
alike (closes plan-review M9 / W12: type-driven Sensitive[T] removal
was previously undetectable).

The snapshot is committed to
`tests/unit/web/composer/redaction_policy_snapshot.json`. Each
manifest entry contributes a SHA-256 over its canonical shape:

```python
def _entry_hash(name: str, entry: ToolRedaction) -> str:
    if entry.argument_model is not None:
        # Type-driven: hash over the schema-walk produced by the
        # shared iterator. Two type-driven entries with identical
        # shapes have identical hashes; a removed Sensitive() marker,
        # an added field, a renamed field, a changed summarizer
        # identity all flip the hash.
        nodes = list(walk_model_schema(entry.argument_model))
        if entry.response_model is not None:
            nodes.extend(walk_model_schema(entry.response_model))
        canon_payload = [_canonicalise_node(n) for n in nodes]
    else:
        policy = entry.policy
        canon_payload = {
            "sensitive_argument_keys": sorted(policy.sensitive_argument_keys),
            "sensitive_response_keys": sorted(policy.sensitive_response_keys),
            "known_response_keys": sorted(policy.known_response_keys),
            "summarizer_keys": sorted(policy.argument_summarizers.keys()),
            "handles_no_sensitive_data": policy.handles_no_sensitive_data,
            "reason_text_hash": (
                _reason_text_hash(policy.handles_no_sensitive_data_reason_struct)
                if policy.handles_no_sensitive_data_reason_struct is not None
                else None
            ),
        }
    canon = json.dumps(canon_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()
```

`_canonicalise_node` produces a deterministic representation of a
`TraversalNode`: `{"path": ..., "type_name": ..., "metadata":
[...marker types and summarizer fully-qualified names...]}`.
Summarizer identity is captured by fully-qualified name, not by
function `id()`, so reloading the module does not flip the snapshot.

The CI test compares `{name: _entry_hash(name, entry) for name, entry
in MANIFEST.items()}` against the committed JSON file. Any difference
fails the test. Policy changes therefore require an explicit commit to
the snapshot file. The label-gate CI step (§4.4.5) requires a
direction-appropriate label on any change that flips the snapshot;
without the label, CI is red.

**Hash semantics and known false-negative class (rev-2 BLOCKER_C).**
The snapshot hash captures **callable identity** (via fully-qualified
name), not behavioural correctness. This means:

- **Replacement flips the hash correctly.** Registering a new
  function in place of an existing summarizer (a new callable
  object at a different FQDN or same FQDN post-refactor) flips the
  hash and triggers the label-gate.
- **Known false-negative: in-place closure-state mutation.** If a
  summarizer closes over a module-level mutable variable, and that
  variable is mutated without replacing the summarizer callable,
  the hash is unchanged and the behavioural drift is undetected.
  This is a documented limitation. The ELSPETH no-legacy-code +
  offensive-programming discipline makes module-level mutable state
  a code smell that should be caught in review. Implementers MUST
  NOT introduce module-level state that summarizers close over.

Closes security T-3 (programmatic detection of policy weakening) and
plan-review M9 / W12 (type-driven coverage in snapshot).

#### 4.4.4 Mass-copy uniqueness of declarative reasons (NEW in rev 5)

Closes plan-review M5 / W7. The adequacy guard asserts:

```python
def test_handles_no_sensitive_data_reasons_are_unique() -> None:
    """Two tools must not share an exact-match why_arguments_safe (or
    why_responses_safe) string. Mass-copy is a signal that someone
    bulk-applied a placeholder reason during migration.

    Whitespace-normalised exact match. The 32-char minimum (§4.2.3)
    plus this uniqueness assertion together force concrete reasoning.
    """
    arg_reasons: dict[str, list[str]] = collections.defaultdict(list)
    resp_reasons: dict[str, list[str]] = collections.defaultdict(list)

    for name, entry in MANIFEST.items():
        if entry.policy is None:
            continue
        struct = entry.policy.handles_no_sensitive_data_reason_struct
        if struct is None:
            continue
        arg_reasons[" ".join(struct.why_arguments_safe.split())].append(name)
        resp_reasons[" ".join(struct.why_responses_safe.split())].append(name)

    duplicates = {
        ("arguments", text, names)
        for text, names in arg_reasons.items()
        if len(names) > 1
    } | {
        ("responses", text, names)
        for text, names in resp_reasons.items()
        if len(names) > 1
    }
    assert not duplicates, (
        "Mass-copy detected in handles_no_sensitive_data reasons: "
        f"{duplicates}"
    )
```

Closes the calendar-synchronicity-at-migration concern (W2 / plan
review). The `last_reviewed_iso` field of rev 4 is **removed** from
`HandlesNoSensitiveDataReason` (§4.2.3); review is triggered by
content change via the policy-hash snapshot, not by calendar tick. A
manifest entry whose reason text and shape never change does not need
periodic review — the underlying tool's behaviour did not change.

#### 4.4.5 Label-gate CI step (rev 5; replaces CODEOWNERS team routing)

Rev 4 §4.4.5 routed `redaction.py`, `tools.py`, the snapshot file, and
`test_redaction_policy.py` to a `@elspeth/security` GitHub team.
**That team cannot exist on this repository:** the project lives at
`https://github.com/johnm-dta/elspeth.git` (a personal-account remote;
GitHub teams require an organisation). A `CODEOWNERS` file naming a
non-existent team is silently no-op — CODEOWNERS-as-control would be
defense by paperwork.

Rev 5 replaces the team routing with a **CI-enforced PR-label gate**:

1. **The policy-hash snapshot is the primary control** (§4.4.3). Any
   change to a manifest entry — Sensitive[T] annotation removed,
   declarative key set narrowed, summarizer replaced, structured
   reason rewritten — flips the snapshot.
2. **A CI step** at `.github/workflows/composer-redaction-gate.yml`
   compares the snapshot file on the PR head against `main`. If they
   differ, the step performs a **direction-aware** label check (rev-2
   BLOCKER_B fix):

   - Compute the `sensitive_path_count` per manifest entry (the count
     of `Sensitive`-annotated field paths, as enumerated by
     `walk_model_schema`) in both the PR-head snapshot and the main
     snapshot.
   - If the total count across all changed entries strictly **decreased**
     (a weakening), only `policy-weaken-justified` is valid.
     `policy-strengthen` is rejected with an error enumerating which
     entries decreased.
   - If the total count strictly **increased or stayed the same** with a
     hash change (a strengthening or neutral semantic shift), only
     `policy-strengthen` is valid. `policy-weaken-justified` is rejected
     with an error noting "snapshot diff shows no coverage reduction; do
     not use `policy-weaken-justified` for this change".
   - The CI failure message enumerates which manifest entries changed
     direction so reviewers can sanity-check.
   - The `policy-weaken-justified` label requires a section titled
     "Redaction policy weakening rationale" in the PR body (grep-asserted
     via `gh api`).

   The snapshot may include a per-entry `sensitive_path_count` field to
   make the direction check a pure JSON diff without requiring a live
   Python run in CI. If so, Task 12 (snapshot bootstrap) emits this
   field, and the guard's fifth adequacy assertion (§4.4) verifies it
   matches a live walk.

3. **Without a label**, the CI step fails. There is no override
   mechanism; the change does not merge.
4. **No `.github/CODEOWNERS` file is created** as part of this work.
   The repo's ownership model remains "the operator reviews everything"
   until the project migrates to an organisation-owned repo. At that
   point the team-routing variant becomes available and the gate can
   be augmented (filed as a follow-up; not blocking).
5. **Single-owner governance note.** On a multi-person repository,
   reviewers in CODEOWNERS would gate label application via review
   approval. On a single-owner / personal-account repository, the
   label gate is **self-bypassing** — the repository owner can apply
   any label without external review. This is an acknowledged
   governance gap. Operators on single-owner repos MUST treat
   snapshot-hash flips as warranting a manual sanity-check of the diff
   direction before merging. See `docs/guides/redaction-policy-changes.md`.

The label-gate is mechanism over policy: a label is enforced by CI,
not by a reviewer's discipline. CODEOWNERS-as-team-routing would
add a second checkpoint *after* a CI gate already exists. Closes
plan-review B4 / W9 / M10.

### 4.5 `composition_state_id` FK behaviour, `IntegrityError`, and `OperationalError` semantics

Under §5.2's single-sync-block design, every per-turn write
(assistant row + tool rows + state rows + sequence-number reservations)
runs inside ONE database transaction. The FK race the original §4.5
described (state row rolled back between tool execution and tool-row
insert) is **not reachable** — every row is committed atomically with
the others, or none is.

Places where DB-layer exceptions can fire on a per-turn write:

| Source | Class | Cause | Correct response |
|---|---|---|---|
| `fk_chat_messages_composition_state_session` (assistant row references a pre-committed state) | `IntegrityError` | Externally-deleted `composition_states` row (admin tooling, race with cleanup). The assistant row references an *already-committed* state row that vanished between read and write. | Crash — Tier-1 invariant violation. |
| `uq_chat_messages_tool_call_id` (partial unique) | `IntegrityError` | LLM provider re-used a `tool_call_id` across turns within a session. | Crash — Tier-3 input validation failure surfaced as a database integrity error. The compose request fails fast; the LLM is misbehaving. Counter `composer.audit.tool_row_integrity_violation_total` increments. |
| `ix_chat_messages_session_sequence` (unique) | `IntegrityError` | Sequence-number reservation race. Should be unreachable per §5.7's single-writer-per-session rule + `writer_principal` CHECK + the advisory-lock acquisition. | Crash — concurrency bug. |
| `ck_chat_messages_*` (CHECK constraints) | `IntegrityError` | Caller bug. | Crash — internal bug. |
| `ck_chat_messages_writer_principal` | `IntegrityError` | An unrecognised writer attempted to insert. | Crash — schema-level rejection of unknown subsystem identity. |
| `ck_composition_states_provenance` | `IntegrityError` | An unrecognised provenance value attempted to insert. | Crash — schema-level rejection. |
| Connection drop, pool exhaustion, network partition | `OperationalError` | Infrastructure issue, not a logical-state issue. | Counter `composer.audit.tool_row_persist_failed_total` increments; primacy disposition applies (see §5.2). |
| Disk full, fsync failure on commit | `OperationalError` (driver-specific) | Storage layer failure during COMMIT. | Same as connection drop — counter + primacy disposition. |

**Persistence-layer disposition for these classes:**

- **`IntegrityError`** is treated as a Tier-1 invariant violation. The
  helper increments `composer.audit.tool_row_integrity_violation_total`
  and propagates. There is no recovery path; CLAUDE.md offensive-programming
  policy requires informative crashes for invalid states. (Revision 4
  corrects rev-3's prose contradiction: the §5.2 helper does have an
  `except IntegrityError: ... raise` arm; the prose now reads
  "treats IntegrityError as terminal" rather than "does not catch
  IntegrityError," which is honest about the mechanism.)

- **`OperationalError`** is treated under audit-failure primacy (§5.2):
  if a tool exception is in flight, the `OperationalError` is logged
  (permitted use under CLAUDE.md), the
  `composer.audit.tool_row_persist_failed_during_unwind_total` counter
  increments, and the tool exception propagates. If no tool exception is
  in flight, it is a Tier-1 audit invariant violation:
  `composer.audit.tool_row_tier1_violation_total` increments and the
  helper raises.

(Earlier revisions of this spec proposed catching `IntegrityError` and
re-attempting with `composition_state_id=NULL`. That would silently
swallow real bugs including duplicate `tool_call_id`, CHECK violations,
and writer-principal violations. Removed in revision 3 and remains
removed in revision 4.)

### 4.6 Retention path

`chat_messages` rows for a session are eligible for archival under the
same retention policy as `composition_states`. Today, deleting a session
cascades to all chat_messages rows (existing FK). Extending the
`elspeth purge --retention-days N` CLI to operate on web sessions
without manual session deletion is filed as
[elspeth-RETENTION-WEB] (to be filed during implementation; see §10
open question OQ-1).

### 4.7 Initial policy declarations

Final list assembled during implementation; the adequacy guard (§4.4)
enforces correctness. Representative examples:

| Tool | sensitive_argument_keys | sensitive_response_keys | summarizers | handles_no_sensitive_data | reason |
|---|---|---|---|---|---|
| `wire_secret_ref` | `()` | `()` | none | `True` | "Secret *names* are inventory metadata, not values; resolved values never appear in arguments or responses (the resolver is server-side and audit-recorded separately)." |
| `set_source` | `("path",)` | `()` | `path` → `redact_source_storage_path` | `False` | n/a |
| `create_blob` | `("content",)` | `()` | `content` → `<inline-blob:{n}-bytes>` | `False` | n/a |
| `patch_*` | depends on plugin schema; declared per-tool | `()` | as needed | `False` | n/a |

The `wire_secret_ref` reason illustrates the discipline: explicit,
concrete, refers to where the sensitive material *actually* lives and
why the tool's surface is safe.

---

## 5. Persistence Boundary

### 5.1 Composer service must hold a `SessionsService` handle

Today the composer service does not have a `SessionsService` reference.
It will gain one via constructor injection.

**Lifetime.** `SessionsService` is constructed at app startup with the
shared SQLAlchemy `Engine` and lives for the app lifetime (see existing
`web/dependencies.py`). The composer service has the same lifetime.
Constructor injection at startup; no per-request rewiring. This matches
the existing pattern for `WebSecretService`, `BlobServiceImpl`, etc.

### 5.2 Insertion sites in the compose loop

#### 5.2.1 Loop shape — single-sync-block per turn

Revision 4 abandons the rev-3 `try/except/finally` + `asyncio.shield` +
async `SessionsTransaction` design. The new shape: each per-turn write
collects its inputs in async land (the LLM call, the tool execution
outcomes), then dispatches **one** sync function via `_run_sync` that
performs every database write — assistant row, N tool rows, N
`composition_states` rows where state advanced, plus all sequence-number
reservations — inside a single `engine.begin()` transaction.

Three properties this design delivers that the rev-3 design did not:

1. **Atomicity per turn.** Either the entire turn lands or none of it
   does. The bidirectional INV-AUDIT-AHEAD invariant (§5.3) is mechanical:
   one transaction, one outcome.
2. **No cancellation race surface.** Cancellation arriving while the
   sync worker is mid-transaction has no effect on the worker — Python
   threads do not interrupt sync code unless cooperatively. The async
   side either entered the worker (and waits for completion) or did not
   (no DB work happened). There is no third state. `asyncio.shield` is
   not needed and is removed from the design.
3. **No `finally`/`NameError` interaction.** No cross-await `finally`
   block exists. `response_for_persistence` is built in straight-line
   code in the worker, never read across an await boundary, and never
   referenced under `BaseException` paths.

The compose loop body (`web/composer/service.py:_compose_loop`, defined
at line 573) becomes:

```python
# (previous LLM call code unchanged; assistant_message obtained from LLM)

# Step 0 — enforce the per-turn tool-call cap (RSK-13 / §1.4 NFR).
if len(assistant_message.tool_calls) > self._max_tool_calls_per_turn:
    raise ComposerConvergenceError.capture(
        state,
        reason="tool_call_cap_exceeded",  # new ComposerProgressEvent reason code
        observed=len(assistant_message.tool_calls),
        cap=self._max_tool_calls_per_turn,
    )

# Step 1 — execute every tool call in async land, accumulating outcomes.
# This is where async / I/O / cancellation can legitimately occur. No
# audit work has been written yet; cancellation here is safe — the LLM
# response is already recorded in memory, the DB is unchanged, and the
# next turn (if any) will re-enter this loop.
tool_outcomes: list[_ToolOutcome] = []
plugin_crash: BaseException | None = None

for tool_call in assistant_message.tool_calls:
    pre_version = state.version
    try:
        response = await execute_tool(tool_call, state)
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=response,
            error_class=None,
            error_message=None,
            pre_version=pre_version,
            post_version=state.version,
        ))
    except ToolArgumentError as exc:
        # Tier-3 boundary signal — the loop's existing contract continues.
        # service.py:867 — catches and continues to the next tool_call.
        # protocol.py:299-303 (revised in revision 4 to enumerate all three
        # except clauses on the docstring).
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=None,
            error_class="ToolArgumentError",
            error_message=str(exc),
            pre_version=pre_version,
            post_version=state.version,
        ))
        # Do NOT re-raise; the loop continues. Compose loop is contractually
        # obligated to feed the error back to the LLM as the tool response.
    except (AssertionError, MemoryError, RecursionError, SystemError):
        # service.py:907 — interpreter-level invariant violations. Not
        # recoverable, not wrappable. Re-raise BEFORE writing audit rows;
        # the partial state we have is from before this branch was entered
        # so it is the legitimate snapshot to persist via the route helper.
        raise
    except Exception as tool_exc:
        # service.py:942-980 — plugin-bug surface. Wrap in
        # ComposerPluginCrashError.capture and break out of the for-loop
        # so the audit-write step (Step 2 below) still runs for the calls
        # that already succeeded. The ComposerPluginCrashError is raised
        # AFTER the audit-write step.
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=None,
            error_class=type(tool_exc).__name__,
            error_message=str(tool_exc),
            pre_version=pre_version,
            post_version=state.version,
        ))
        plugin_crash = ComposerPluginCrashError.capture(
            tool_exc,
            state=state,
            initial_version=initial_version,
        )
        # Note: `from tool_exc` chaining is preserved — capture() sets __cause__
        # internally. Verified against service.py:942-980.
        break  # exit the for-loop; we still need to write audit rows below.

# Step 2 — redact in async land, then dispatch ONE sync write.
# All redaction is pure / non-blocking; building the redacted payload in
# async-land keeps the sync worker's wall-clock window narrow.
redacted_assistant_tool_calls = tuple(
    # Rev 5: dispatch via MANIFEST[tool_name] inside the walker.
    # No lookup_tool_class helper, no MissingToolError class
    # (see §5.7.5 — Tier-3 unknown tool name does not reach here;
    # only successfully-dispatched tools produce tool_calls in the
    # post-execute_tool list). Walker takes the already-decoded
    # arguments dict (M3: single parse) and the redaction telemetry
    # instance.
    redact_tool_call_arguments(tc.function.name, decoded_args[tc.id], telemetry=redaction_telemetry)
    for tc in assistant_message.tool_calls
)
redacted_tool_rows = tuple(
    RedactedToolRow(
        tool_call_id=outcome.call.id,
        content=_serialize_response_via_walker(outcome, telemetry=redaction_telemetry),
        composition_state_payload=(
            # B1 (Phase 1 plan-review synthesis): no caller-supplied
            # ``version`` here. ``StatePayload`` carries the per-column
            # ``CompositionStateData`` plus a ``derived_from_state_id``;
            # the new state row's ``version`` is allocated inside
            # ``_insert_composition_state`` under the held
            # ``_session_write_lock`` via
            # ``SELECT COALESCE(MAX(version), 0) + 1 ...`` (§5.7.1).
            #
            # Direct constructor; no ``from_composition_state`` factory
            # exists in Phase 1 because no caller needed one. A future
            # phase MAY add ``StatePayload.from_composition_state(state)``
            # if a use case emerges that benefits from the indirection.
            StatePayload(
                data=CompositionStateData(
                    source=state.source,
                    nodes=state.nodes,
                    edges=state.edges,
                    outputs=state.outputs,
                    metadata_=state.metadata_,
                    is_valid=state.is_valid,
                    validation_errors=state.validation_errors,
                    composer_meta=state.composer_meta,
                ),
                # The pre-call state id (predecessor lineage) is
                # supplied by the compose loop's surrounding context;
                # ``_ToolOutcome`` itself carries ``pre_version`` /
                # ``post_version`` for the version-advanced predicate
                # but not the state id. Phase 3 wires the actual
                # caller; Phase 1 surfaces the StatePayload contract.
                derived_from_state_id=pre_state_id_for(outcome),
            )
            if outcome.post_version > outcome.pre_version
            else None
        ),
    )
    for outcome in tool_outcomes
)

# Production async callers MUST go through the protocol's async
# dispatcher: ``persist_compose_turn_async`` opens an
# ``asyncio.shield``-wrapped worker thread and runs the sync primitive
# under it (commit-wins cancellation contract — see §5.2.2 and the
# concrete implementation at
# ``SessionServiceImpl.persist_compose_turn_async``). The sync method
# ``persist_compose_turn`` stays concrete-only and async-loop guarded;
# direct invocation from a coroutine raises ``RuntimeError`` rather
# than blocking the event loop on synchronous DB I/O.
#
# Cancellation of the outer task while this is in flight has no effect —
# the sync worker runs to completion. If the outer task is cancelled
# before this line is reached, no DB work has been done; no invariant
# is violated.
audit_outcome = await self.sessions_service.persist_compose_turn_async(
    session_id=session_id,
    assistant_content=assistant_message.content or "",
    raw_content=raw_assistant_content,  # B2: pre-redaction LLM output
    redacted_assistant_tool_calls=redacted_assistant_tool_calls,
    redacted_tool_rows=redacted_tool_rows,
    parent_composition_state_id=current_state_id,
    # Stale-state guard input: the latest state id observed before
    # the LLM call. ``persist_compose_turn`` re-reads
    # ``MAX(version)`` under the session write lock and raises
    # ``StaleComposeStateError`` if a different state has landed
    # since (e.g. a concurrent fork or admin write).
    expected_current_state_id=current_state_id,
    writer_principal="compose_loop",
    plugin_crash_pending=plugin_crash is not None,
)

# Step 3 — dispatch by audit outcome and any pending plugin crash.
# AuditOutcome has two valid shapes (see §5.2.2 below):
# (1) success — assistant_id set, unwind_audit_failed=False;
# (2) tool failed AND audit unwind failed — assistant_id=None,
#     unwind_audit_failed=True.
# The third historical shape (tier1_violation flag) is REMOVED:
# Tier-1 audit-write failures raise ``AuditIntegrityError``
# directly inside the sync worker rather than returning a flag for
# the caller to re-raise (CLAUDE.md primacy: Tier-1 anomalies crash
# unconditionally; ``@tier_1_error`` registration prevents
# ``except Exception`` blocks from swallowing it).

if plugin_crash is not None:
    # The audit row for the crashing tool was written successfully (or the
    # audit failure was logged under primacy and ``unwind_audit_failed``
    # is set on the returned outcome). Now re-raise the captured
    # ComposerPluginCrashError so _handle_plugin_crash receives it.
    raise plugin_crash

# (loop continues to next turn)
```

Where `_ToolOutcome`, `RedactedToolRow`, `StatePayload`, and
`AuditOutcome` are dataclasses defined in
`src/elspeth/web/sessions/_persist_payload.py` (the
``CompositionStateData`` input DTO that ``StatePayload`` composes lives
in `src/elspeth/web/sessions/protocol.py`). Both modules are L3 and
adhere to layer rules.

#### 5.2.2 The sync write function — `SessionsService.persist_compose_turn`

```python
class SessionsService:
    def persist_compose_turn(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,            # B2 audit attribution
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,     # stale-state guard input
        writer_principal: str,  # "compose_loop"
        plugin_crash_pending: bool,
    ) -> AuditOutcome:
        """Synchronous, single-transaction persistence of one compose turn.

        Concrete sync primitive. Production async callers MUST invoke
        ``await self.persist_compose_turn_async(...)`` through
        :class:`SessionServiceProtocol`; that dispatcher uses ``_run_sync``
        under the hood. Calling this sync primitive directly from async
        land would block the event loop because the body opens a
        synchronous SQLAlchemy transaction. The implementation guards via
        ``asyncio.get_running_loop()`` and raises ``RuntimeError`` if a
        running loop is detected (closes synthesised review finding
        SA-7 / M1).

        Order of work (load-bearing):

        1. Pre-DB transcript validation
           (``_validate_tool_call_id_set_equality``). Pure function of
           caller args; runs BEFORE ``_engine.begin()`` so a contract
           violation cannot leave a half-written audit trail. Raises
           :class:`ToolCallIDMismatchError` (defined in
           ``elspeth.web.sessions.protocol`` alongside
           :class:`StaleComposeStateError`).
        2. Open transaction; acquire ``_session_write_lock`` (PostgreSQL
           uses ``pg_advisory_xact_lock`` two-arg form; SQLite uses a
           process-wide RLock — see §5.7.1).
        3. Cross-session guard on ``parent_composition_state_id`` (B5).
        4. Stale-state guard on ``expected_current_state_id``: re-read the
           session's latest committed composition state id and raise
           :class:`StaleComposeStateError` if it does not match.
        5. Reserve sequence range for assistant + N tool rows under the
           held lock.
        6. Insert assistant row (with optional ``raw_content`` — B2
           audit-attribution column for assistant messages whose visible
           ``content`` was rewritten by runtime preflight redaction).
        7. For each tool row: optionally allocate the new
           ``composition_states`` row's ``version`` under the held lock
           (``SELECT COALESCE(MAX(version), 0) + 1 ...``) and insert it
           via ``_insert_composition_state``, then insert the tool chat
           row referencing the new state id. ``StatePayload`` does NOT
           carry a caller-supplied ``version`` (B1 fix — see §5.2 /
           §5.2.1).

        Atomicity contract:

        - Either every row (assistant + N tool + N state) commits, or
          none does.
        - INV-AUDIT-AHEAD bidirectional is delivered structurally: state
          rows and tool rows share a transaction.

        Audit-failure primacy:

        - Tool succeeded (``plugin_crash_pending=False``) + audit raised
          non-Integrity error => raise ``AuditIntegrityError`` from the
          sync worker, chained from the underlying ``OperationalError``
          via ``raise ... from audit_exc``. The exception is registered
          in ``TIER_1_ERRORS`` (via the ``@tier_1_error`` decoration on
          :class:`AuditIntegrityError`) so ``except Exception`` blocks
          cannot silently swallow it. There is NO ``tier1_violation``
          field on ``AuditOutcome``; the caller has no opportunity to
          ignore the failure (closes synthesised review finding H1).
        - Tool failed (``plugin_crash_pending=True``) + audit raised
          non-Integrity error => log permitted (audit-system failure
          under CLAUDE.md primacy), counter
          ``composer.audit.tool_row_persist_failed_during_unwind_total
          += 1``, return ``AuditOutcome(assistant_id=None,
          unwind_audit_failed=True)``. Caller proceeds to raise the
          captured ``ComposerPluginCrashError``; the unwind flag tells
          the caller "your raise should also record this audit failure"
          (the counter + slog inside ``persist_compose_turn`` have
          already done so).
        - IntegrityError of any class => counter
          ``composer.audit.tool_row_integrity_violation_total += 1``;
          raise (no recovery; CLAUDE.md offensive-programming). Catch is
          OUTSIDE ``with self._engine.begin()`` so the context's
          ``__exit__`` rolls back the transaction BEFORE the counter
          increment, preventing a partial audit row from surviving.
        """
        # Async-loop guard (closes SA-7 / M1).
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop; we are in a worker thread or sync test.
        else:
            raise RuntimeError(
                "persist_compose_turn must be dispatched via "
                "await self.persist_compose_turn_async(...) -- "
                "calling it directly from a coroutine blocks the event "
                "loop on synchronous DB I/O."
            )

        # Step 1: pre-DB transcript validation (Q-F1 / Step 3c).
        _validate_tool_call_id_set_equality(
            redacted_assistant_tool_calls=redacted_assistant_tool_calls,
            redacted_tool_rows=redacted_tool_rows,
        )

        with self._engine.begin() as conn:
            with self._session_write_lock(conn, session_id):
                # Step 3: same-session guard on parent_composition_state_id.
                if parent_composition_state_id is not None:
                    _assert_state_in_session(
                        conn,
                        state_id=parent_composition_state_id,
                        expected_session_id=session_id,
                        caller="persist_compose_turn",
                    )

                # Step 4: stale-state guard.
                current_state_id = conn.execute(
                    select(composition_states_table.c.id)
                    .where(composition_states_table.c.session_id == session_id)
                    .order_by(composition_states_table.c.version.desc())
                    .limit(1)
                ).scalar_one_or_none()
                if current_state_id != expected_current_state_id:
                    raise StaleComposeStateError(
                        "persist_compose_turn: current composition state "
                        f"changed for session_id={session_id!r}; "
                        f"expected={expected_current_state_id!r}, "
                        f"actual={current_state_id!r}. Refusing to persist "
                        "a compose result based on a stale state."
                    )

                # Step 5: reserve sequence range under the held lock.
                base_seq = self._reserve_sequence_range(
                    conn, session_id,
                    count=1 + len(redacted_tool_rows),
                )

                # Step 6: insert assistant row (with optional raw_content).
                assistant_id = self._insert_chat_message(
                    conn,
                    session_id=session_id,
                    role="assistant",
                    content=assistant_content,
                    raw_content=raw_content,
                    tool_calls=redacted_assistant_tool_calls,
                    sequence_no=base_seq,
                    writer_principal=writer_principal,
                    composition_state_id=parent_composition_state_id,
                    tool_call_id=None,
                    parent_assistant_id=None,
                )

                # Step 7: insert each tool row + state row (when state advanced).
                # ``_insert_composition_state`` allocates ``version`` under
                # the held ``_session_write_lock`` (B1 fix); StatePayload
                # carries no caller-supplied version. Both helpers use the
                # shared ``_enveloped_state_column(...)`` helper used by
                # ``_insert_composition_state``, ``save_composition_state``,
                # and ``fork_session``.
                for offset, tool_row in enumerate(redacted_tool_rows, start=1):
                    state_id: str | None = None
                    if tool_row.composition_state_payload is not None:
                        state_id = self._insert_composition_state(
                            conn,
                            session_id=session_id,
                            payload=tool_row.composition_state_payload,
                            provenance="tool_call",
                        )
                    self._insert_chat_message(
                        conn,
                        session_id=session_id,
                        role="tool",
                        content=tool_row.content,
                        raw_content=None,
                        tool_calls=None,
                        sequence_no=base_seq + offset,
                        writer_principal=writer_principal,
                        composition_state_id=state_id,
                        tool_call_id=tool_row.tool_call_id,
                        parent_assistant_id=assistant_id,
                    )

                # Transaction commits on context exit.
                return AuditOutcome(
                    assistant_id=assistant_id,
                    unwind_audit_failed=False,
                )

        # IntegrityError / OperationalError disposition lives OUTSIDE the
        # ``with self._engine.begin()`` block deliberately — see the
        # docstring "Audit-failure primacy" notes above. Order is
        # load-bearing: rollback first (context __exit__), then the
        # counter increment, then the exception re-raises (or, in the
        # ``plugin_crash_pending`` branch, the AuditOutcome with
        # ``unwind_audit_failed=True`` is returned).

    async def persist_compose_turn_async(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: str,
        plugin_crash_pending: bool,
    ) -> AuditOutcome:
        """Async dispatcher for :meth:`persist_compose_turn` — the public
        async entry point on :class:`SessionServiceProtocol`.

        Bridges to the sync primitive via ``_run_sync`` (worker-thread
        dispatch shielded from caller cancellation). Commit-wins
        cancellation contract: on caller cancellation the worker
        continues to completion; the transaction either commits durably
        or rolls back atomically. **Callers MUST NOT retry on
        ``CancelledError``** — retrying risks a duplicate
        ``tool_call_id`` INSERT that fires a fabricated Tier-1 counter
        increment.
        """
        return await self._run_sync(
            self.persist_compose_turn,
            session_id=session_id,
            assistant_content=assistant_content,
            raw_content=raw_content,
            redacted_assistant_tool_calls=redacted_assistant_tool_calls,
            redacted_tool_rows=redacted_tool_rows,
            parent_composition_state_id=parent_composition_state_id,
            expected_current_state_id=expected_current_state_id,
            writer_principal=writer_principal,
            plugin_crash_pending=plugin_crash_pending,
        )
```

#### 5.2.3 What this design eliminates

- **The rev-3 `finally`/`NameError` interaction.** No `finally` block;
  `response_for_persistence` is unused. Closes Python engineer Q1.
- **The rev-3 `asyncio.shield` reasoning.** Cancellation cannot reach
  inside the sync worker. Closes architect C-2 and Python engineer Q2.
- **The rev-3 missing `ComposerPluginCrashError.capture` wrap.** The
  capture is explicit on line 60 of the §5.2.1 listing above. Closes
  Python engineer C-5.
- **The rev-3 advisory-lock-across-await race.** The advisory lock is
  acquired and released within a single `engine.begin()` transaction;
  no `await` punctuates the protected window. Closes architect C-1.

#### 5.2.4 Cancellation semantics — explicit table

| Cancellation arrives… | Effect on DB state |
|---|---|
| Before Step 1 (no LLM call yet) | No DB writes; `CancelledError` propagates to the route helper which records nothing for this turn. INV-AUDIT-AHEAD undisturbed (no half-state). |
| During Step 1 (mid tool execution) | Tool execution may raise `CancelledError` itself; the for-loop's `except (AssertionError, ...)` arm does NOT catch it (CancelledError is not in the list and was promoted to `BaseException` in Python 3.8). The `except Exception` arm also does NOT catch it. CancelledError propagates out of the loop; no DB writes have occurred. INV-AUDIT-AHEAD undisturbed. |
| Between Step 1 and Step 2 (after all tool calls completed, before _run_sync dispatched) | No DB writes; CancelledError propagates. INV-AUDIT-AHEAD undisturbed. |
| During Step 2 (sync worker is running) | The async `await self._run_sync(...)` raises `CancelledError` immediately; the sync worker continues running detached (this is the standard `_run_sync` semantics; the worker pool does not propagate cancellation into running threads). The transaction either commits or rolls back based on its internal outcome. The route helper observes `CancelledError`, treats it as a `client_cancelled` reason, and reads the actual chat_messages state via a fresh query (which sees the committed-or-not result). |
| After Step 2 commit, before Step 3 | Audit row committed; CancelledError propagates. INV-AUDIT-AHEAD undisturbed (forward direction satisfied; backward direction satisfied because the state row committed in the same transaction). |
| Between Step 3 raise and route helper read | Route helper observes the exception, queries chat_messages directly, and computes `tool_responses_persisted` from the database state, not from in-flight bookkeeping. F-3 (read-consistency note in §6.1) applies. |

#### 5.2.5 Why per-turn (not per-tool-row) atomicity is correct

Putting all of one turn in a single transaction means a late failure
within the turn rolls back the whole turn — an unfortunate property
because earlier successful tool rows are lost. The compensating
correctness property is that the bidirectional invariant holds without
any cross-transaction coordination. The trade-off: a single LLM turn is
atomic. A previous turn's writes (committed in their own per-turn
transaction) are unaffected. **The grain of atomicity is the LLM turn,
not the tool call.** This matches how an auditor reasons about the
record: "what did this turn produce?" is the natural question, not
"what did this individual tool call produce in isolation?"

If a future requirement demands per-tool-row atomicity (e.g., a turn with
60 tool calls where partial commit is preferable), the implementer can
split `persist_compose_turn` into N+1 single-row transactions while
preserving the bidirectional invariant inside each one. This change is
local to the sync function; the loop body is unaffected. The phased
delivery plan in §11 calls this out explicitly.

### 5.3 Bidirectional audit-ahead-of-state invariant (INV-AUDIT-AHEAD)

The invariant is **bidirectional**:

1. **`chat_messages` may be ahead of `composition_states`** (showing what
   was attempted) but never behind (claiming work that did not land).
2. **`composition_states` must NOT be ahead of `chat_messages` for tool-driven
   state changes.** Every committed `composition_states` row with
   `provenance='tool_call'` and `version > 0` must have a corresponding
   `chat_messages` row with `role='tool'`, written in the same
   transaction.

This invariant derives directly from CLAUDE.md's auditability standard:
*"no inference — if it's not recorded, it didn't happen."* If
`composition_states` advances past `chat_messages` for a tool-driven
change, the database asserts that work happened (the state changed)
without recording the evidence (the tool row). That is the canonical
fabrication failure mode CLAUDE.md forbids.

**Mechanical enforcement (revision 4):**

- `execute_tool()` does not commit `composition_states` directly; it
  mutates an in-memory `CompositionState`.
- The single sync transaction inside `persist_compose_turn` (§5.2.2)
  writes the tool row, the corresponding `composition_states` row (when
  state advanced), and reserves sequence numbers — all under one
  `engine.begin()` context. There is NO multi-await coordination, NO
  `asyncio.shield`, and NO opportunity for cancellation to tear the
  transaction.
- The `composition_states.provenance` discriminator (§4.1.2) makes the
  backward-direction post-condition a pure SQL predicate at test time
  rather than a test-harness bookkeeping assertion. Closes QA F-1.
- The route-helper persist paths (`_handle_convergence_error`,
  `_handle_plugin_crash`, `_handle_runtime_preflight_failure`) write
  `composition_states` rows with `provenance='convergence_persist'`,
  `'plugin_crash_persist'`, and `'preflight_persist'` respectively. These
  rows are explicitly outside the bidirectional invariant — they record
  partial state captured by an exception, not state changes driven by
  tool calls.
- The property test (§8.3) and a dedicated integration test
  (`tests/integration/web/test_inv_audit_ahead_backward.py`, new) assert
  the post-condition by running the SQL predicate at trace end.

### 5.4 `partial_state` redaction symmetry

The existing `partial_state` persistence in
`_handle_convergence_error` / `_handle_plugin_crash` /
`_handle_runtime_preflight_failure` already redacts via
`redact_source_storage_path` for blob source paths. To preserve the
"redact at write, never at read" principle uniformly, the same redaction
policy that applies to `chat_messages.tool_calls` JSON also applies to
`partial_state.source.options` and node options before persistence.
Composition state rows with raw paths are an existing inconsistency this
spec does NOT extend; they will be addressed in a follow-up issue
(see §10 OQ-2). The new code path uniformly redacts.

### 5.5 Failure mode interaction

Revision 4's single-sync-block design simplifies cancellation reasoning
materially. The grain of atomicity is the LLM turn; either everything in
a turn lands or nothing does. The table reflects that.

| # | Failure | What persists | What the user sees |
|---|---|---|---|
| 1 | Tool returns successfully (every call in turn succeeds) | assistant row + N tool rows + state rows for every call where state advanced; all committed atomically | Normal continuation. |
| 2 | Tool raises `ToolArgumentError` | Captured in the loop's `_ToolOutcome` list; persisted as a tool row with error_class='ToolArgumentError'; loop continues to next tool_call | Conversation continues; the LLM gets the error tool row as feedback. |
| 3 | Tool raises `(AssertionError, MemoryError, RecursionError, SystemError)` | NOTHING persisted for this turn — the for-loop re-raises before the dispatch step. The route helper's existing partial-state persist runs (`_handle_*` writes a `composition_states` row with the corresponding `provenance` value). | 500 response. The interpreter-level invariant violation is the dominant signal; persistence of in-flight tool outcomes is sacrificed. Documented loss; no claim that this case preserves audit detail. |
| 4 | Tool raises `Exception` (plugin bug, non-ToolArgumentError) | assistant row + N-1 normal tool rows + 1 error tool row for the crashing call + state rows for any earlier calls that advanced state, all atomically; the captured `ComposerPluginCrashError` is then raised AFTER the audit write | Existing `_handle_plugin_crash` runs; 500 response with `partial_state`. Recovery panel shows the crash row. |
| 5 | `asyncio.CancelledError` during tool execution (mid Step 1) | NOTHING persisted for this turn — no DB writes have occurred when cancellation arrives in Step 1. Previous turns' commits are unaffected. | Route helper observes `client_cancelled` reason; existing `_active`-side handler emits the cancellation telemetry; `tool_responses_persisted=0` for this turn. |
| 6 | `asyncio.CancelledError` between Step 1 and Step 2 (after for-loop completes, before `_run_sync` dispatched) | NOTHING persisted for this turn. | Same as row 5. |
| 7 | `asyncio.CancelledError` arrives while `_run_sync` is executing | The sync worker continues to completion (cancellation does not propagate into running threads in CPython's worker pool). The transaction either commits (if it reached COMMIT before the worker's natural exit) or rolls back. The async caller observes `CancelledError` immediately. The route helper queries chat_messages directly to compute `tool_responses_persisted` from committed state. Either: (a) the turn committed atomically (everything persisted) and `tool_responses_persisted == N`; or (b) the worker rolled back and `tool_responses_persisted == 0`. No third state. | Recovery panel shows the actual committed state, not in-flight bookkeeping. |
| 8 | DB write fails on `INSERT` (any constraint) — `IntegrityError` | Counter `composer.audit.tool_row_integrity_violation_total` increments; transaction rolls back; sync worker raises `IntegrityError`; async caller raises | Crash; surfaces real bug class (RSK-12 LLM misbehaviour, sequence race, writer_principal violation, provenance violation). |
| 9 | DB INSERT succeeded, COMMIT fails (`OperationalError`) — disk full, fsync failure, connection dropped between INSERT and COMMIT, no plugin crash in flight | Counter `composer.audit.tool_row_tier1_violation_total` increments; sync worker raises `AuditIntegrityError` chained from the underlying `OperationalError` (no flag-return — see §5.2.2); async caller propagates | 500 response; `partial_state_save_failed=true` carries through; Tier-1 audit invariant violation telemetry alerts. |
| 10 | DB INSERT succeeded, COMMIT fails (`OperationalError`) — plugin crash in flight | Counter `composer.audit.tool_row_persist_failed_during_unwind_total` increments; sync worker logs (permitted under primacy) and returns `AuditOutcome(assistant_id=None, unwind_audit_failed=True)`; async caller raises the captured `ComposerPluginCrashError` | Plugin crash path runs as in row 4; the audit-failure-during-unwind is visible via telemetry counter only. |
| 11 | Advisory lock acquisition fails (PostgreSQL pool exhausted, deadlock detector aborts) | Transaction never opens; sync worker raises `OperationalError`; primacy disposition applies as in rows 9/10 | Same dispatch as the post-INSERT commit failure. |
| 12 | Per-turn tool-call cap exceeded (>16 by default) | NO tool execution attempted; loop raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")` BEFORE the for-loop runs | `_handle_convergence_error` runs; 500 response with the new reason code; partial_state captured pre-cap with state.version unchanged. |

### 5.6 Atomicity grain — per turn, not per tool row

Revision 3 split each tool row into its own transaction so that a late
failure would not roll back earlier audit records (forward-direction
preservation). Revision 4 instead uses **per-turn atomicity**: every row
produced by one assistant turn lands together, or none does.

The trade-off and rationale:

| Property | Rev-3 (per-row) | Rev-4 (per-turn) |
|---|---|---|
| Forward direction | Preserved across multiple per-row commits | Preserved trivially (all rows commit together) |
| Backward direction | Required `asyncio.shield` + atomic-pair (tool row + state row) per call | Preserved trivially (one transaction; provenance discriminator makes the post-condition a SQL predicate) |
| Cancellation race surface | Multiple await boundaries between writes | Zero await boundaries within the transaction |
| Atomicity scope | One tool call | One LLM turn |
| Cost of late failure | Lose only the failed call's row | Lose all rows from the failed turn |
| Auditor's natural unit | "Did this tool call succeed?" | "What did this turn produce?" |

The atomicity-grain change is the load-bearing simplification of revision
4. The audit-grade question an investigator asks is "what did this LLM
turn try and what landed?" — naturally aligned with per-turn atomicity.
Per-call atomicity made sense in rev 3 only because the rev-3 mechanism
(async `SessionsTransaction` with shield) couldn't safely span multiple
calls without making the cancellation surface intractable. Rev 4's sync
worker has no cancellation surface inside the transaction, so the
mechanism aligns with the natural atomicity grain instead of fighting
against it.

If a future requirement demands per-call atomicity (e.g., 60-call turns
where partial commits are valuable), the change is local to
`persist_compose_turn`: split the loop body into N+1 single-row
transactions. The bidirectional invariant inside each transaction still
holds. No compose-loop changes required. This is filed as a forward-looking
note, not a current requirement.

### 5.7 Infrastructure additions on `SessionsService`

#### 5.7.1 The sync persistence primitive

Revision 4's primary new method on `SessionsService` is **synchronous**
because the entire per-turn persistence runs inside a `_run_sync` worker.
Async wrappers exist for non-compose-loop callers (route helpers, admin
tooling) but the compose-loop write path goes straight through the sync
function.

```python
class SessionsService:
    def __init__(
        self,
        engine: Engine,
        *,
        data_dir: Path,
        telemetry: _SessionsTelemetry,
        log: structlog.BoundLogger,
    ) -> None:
        self._engine = engine                  # sync Engine — see CLAUDE.md and existing service.py:18
        self._data_dir = data_dir
        self._telemetry = telemetry
        self._log = log

    # ── New in rev 4 ────────────────────────────────────────────────

    def persist_compose_turn(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: str,           # one of the five CHECK-permitted values
        plugin_crash_pending: bool,
    ) -> AuditOutcome:
        """Concrete-only synchronous primitive. Implementation in §5.2.2.
        Production async callers go through the protocol's
        :meth:`persist_compose_turn_async` dispatcher; calling this sync
        method directly from async land raises ``RuntimeError`` (the
        ``asyncio.get_running_loop()`` guard).
        """
        ...

    async def persist_compose_turn_async(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: str,
        plugin_crash_pending: bool,
    ) -> AuditOutcome:
        """Public async entry point on
        :class:`SessionServiceProtocol`. Bridges to the sync primitive
        above via ``_run_sync`` (worker-thread dispatch shielded from
        caller cancellation). Commit-wins cancellation contract — see
        §5.2.2.
        """
        ...

    # ── Helpers private to persist_compose_turn ─────────────────────

    def _acquire_session_advisory_lock(self, conn: Connection, session_id: str) -> None:
        """Acquire a session write lock for the duration of the current
        transaction. Released automatically on COMMIT or ROLLBACK.

        SQLite: no-op. SQLite serialisation is owned by
        ``_session_write_lock`` below (process-wide RLock keyed on
        ``(database_url, session_id)``); this helper exists only for the
        PostgreSQL advisory-lock SQL and remains no-op on SQLite so
        callers can test the dialect-specific SQL separately.

        PostgreSQL: ``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
        hashtext(session_id))`` — the **two-argument** form (B3 from the
        Phase 1 plan-review synthesis). The classid namespace is reserved
        in ``src/elspeth/contracts/advisory_locks.py`` as
        ``ELSPETH_SESSIONS_LOCK_CLASSID`` and is on-the-wire ABI under
        change control; do not open-code the literal here, always import
        the constant. The classid value is **NOT** a deployment knob —
        two ELSPETH instances on the same PostgreSQL cluster (including
        different versions during a rolling deploy) MUST share the same
        value or they will not mutually exclude each other.

        Hash-function notes:

        - ``pg_advisory_xact_lock(int, int)`` requires two signed int4
          arguments. ``hashtext(text)`` returns int4 directly. Do not
          use ``hashtextextended(...)::int``: PostgreSQL integer casts
          are range-checked and may fail before the lock is acquired.
          (B3 amendment: rev-3 spec text used the one-arg
          ``hashtextextended`` form. The two-arg form with the reserved
          classid namespaces ELSPETH's session-write locks distinctly
          from any other advisory-lock space on the cluster.)
        - Birthday collisions become probable around ~65k *concurrent*
          sessions hashing to the same classid slot. This is benign —
          the unique index ``ix_chat_messages_session_sequence`` is the
          correctness guarantee; the advisory lock is a contention
          reducer ahead of it. Collisions cause spurious serialisation
          between two unrelated sessions, never duplicate rows or lost
          writes.
        """

    @contextlib.contextmanager
    def _session_write_lock(self, conn: Connection, session_id: str) -> Iterator[None]:
        """Serialise same-session sequence/version allocators.

        PostgreSQL uses the transaction-scoped advisory lock above.
        SQLite uses a process-wide per-session ``threading.RLock`` keyed
        on ``(str(self._engine.url), session_id)`` around the whole
        allocator + insert sequence — the staging deployment is
        single-process today; cross-process serialisation will arrive
        with the Phase 3 Postgres-default switch.

        Every caller that performs ``SELECT MAX(...) + 1`` for
        ``chat_messages.sequence_no`` or ``composition_states.version``
        MUST wrap that read and every dependent INSERT in this context.
        ``_assert_session_write_lock_held`` enforces the precondition
        mechanically (per-thread ``ContextVar`` token keyed on
        ``(id(conn), session_id)``); lock-requiring helpers crash
        immediately if called without that token in the same
        transaction.
        """

    def _reserve_sequence_range(
        self, conn: Connection, session_id: str, *, count: int
    ) -> int:
        """Reserve `count` consecutive sequence numbers for `session_id`.

        MUST be called inside ``_session_write_lock(conn, session_id)``
        (enforced by ``_assert_session_write_lock_held``). Inside the
        same transaction, performs:

            SELECT COALESCE(MAX(sequence_no), 0) FROM chat_messages
             WHERE session_id = ?

        and returns max+1; the caller writes rows at
        max+1, max+2, ... max+count.

        Note: gaps in sequence_no are permitted (transaction rollback
        after reservation leaves the next caller's MAX+1 higher than the
        first successful row's sequence_no). Sequence_no is an ordering
        key, not a count.
        """

    def _insert_chat_message(self, conn: Connection, /, **fields) -> str:
        """Single-row insert into chat_messages with the supplied fields.
        Caller must already hold the advisory lock and have reserved the
        sequence_no.
        """

    def _insert_composition_state(
        self, conn: Connection, *,
        session_id: str,
        payload: StatePayload,
        provenance: str,                 # one of the six CHECK-permitted values
    ) -> str:
        """Single-row insert into composition_states with the supplied
        provenance discriminator. Caller MUST already hold
        ``_session_write_lock(conn, session_id)`` (enforced by
        ``_assert_session_write_lock_held``); the helper allocates the
        new row's ``version`` under that lock via
        ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
        WHERE session_id = :sid`` (B1 — see §5.2 / §5.2.1).

        All per-column writes use the shared
        ``_enveloped_state_column(...)`` helper (replaces earlier
        method-local ``_enveloped(...)`` snippets in
        ``save_composition_state`` and ``fork_session``); existing rev-3
        inline inserts in ``service.py`` are refactored to call this
        helper rather than emit raw INSERTs.
        """

    # ── Existing async wrappers — kept for non-compose callers ─────

    async def add_message(
        self,
        *,
        session_id: str,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        tool_call_id: str | None = None,                 # NEW (rev 3)
        parent_assistant_id: str | None = None,          # NEW (rev 3)
        writer_principal: str,                           # NEW (rev 4) — REQUIRED
        composition_state_id: str | None = None,
    ) -> str:
        """Insert a chat_messages row. Used by route-layer callers that
        do NOT need to write multiple rows atomically (user message,
        system seed). Internally calls _run_sync to dispatch a one-row
        transaction that acquires the advisory lock, reserves a sequence
        number, and inserts. Returns the inserted row's id.

        BREAKING CHANGE in rev 4: writer_principal is now REQUIRED. All
        existing callers must be updated:
        - src/elspeth/web/sessions/routes.py:1487 (route_user_message)
          → add ``writer_principal="route_user_message"``
        - src/elspeth/web/sessions/routes.py:1883 (route_system_message)
          → add ``writer_principal="route_system_message"``

        Implementers MUST update both call sites in the same PR that
        changes the signature; partial migration leaves the route layer
        broken.
        """
```

#### 5.7.2 Removed: async `SessionsTransaction` and `atomic_transaction`

Revision 3's `atomic_transaction` async context manager and
`SessionsTransaction` handle are NOT introduced in revision 4. The
single-sync-block design eliminates the multi-await transaction need
(architect C-1). Implementers should NOT add these primitives; doing so
would re-introduce the cross-await coordination problem the rev-4 pivot
exists to avoid.

#### 5.7.3 Single-writer-per-session — structurally enforced

Revision 4 strengthens the rev-3 "single writer per session" claim from
convention to mechanism via the `writer_principal` CHECK constraint
(§4.1.1) and the advisory lock. Concretely:

- Only `compose_loop` writes assistant and tool rows.
- Only `route_user_message` / `route_system_message` write user/system
  rows during route handling.
- A future admin tool that wishes to write rows must add `admin_tool` to
  the CHECK enum via schema migration AND coordinate with the compose
  loop via the advisory lock (which protects against any concurrent
  writer regardless of identity).
- The CHECK constraint refuses any other writer_principal string at
  write time, so a forgotten parameter or a misnamed subsystem fails
  fast at the database boundary.

#### 5.7.4 Telemetry module

The `_telemetry` container lives in
`src/elspeth/web/sessions/telemetry.py`, exposing the named OTel
counters introduced in §1.4:

- `composer.audit.tool_row_tier1_violation_total`
- `composer.audit.state_rolled_back_during_persist_total`
- `composer.audit.tool_row_persist_failed_during_unwind_total`
- `composer.audit.tool_row_integrity_violation_total`
- `composer.redaction.summarizer_errors_total`
- `composer.redaction.unknown_response_key_total`

The module is located under `web/sessions/` rather than
`web/composer/` (the original spec path) because the audit-row
counters live alongside their only writer (`SessionServiceImpl`).
Co-locating the counter definitions with the writer keeps "registered"
and "exercised" in lock-step at the ownership boundary; composer code
that needs to read the counters imports the sessions-owned container
from app wiring rather than the reverse direction.

Tests inject a fake counter via the `telemetry=` constructor parameter
on `SessionServiceImpl` for assertable behaviour. The injected fake
counter is itself asserted against by the property-test post-conditions
(§8.3.2 closes QA F-4).

#### 5.7.5 `MANIFEST` lookup and redaction-layer integration (rev 5)

(Rev 4 framed this section around a fictional `lookup_tool_class(tool_name) -> type[ComposerTool]` helper and a `MissingToolError` exception. Plan-review B1 / W3 established that no class hierarchy exists. Rev 5 restates the integration around the manifest.)

`MANIFEST: Mapping[str, ToolRedaction]` (§4.2.1) is the single
registration root for redaction policy. The runtime walker
(`redact_tool_call_arguments`, `redact_tool_call_response`; §4.2.6)
dispatches via `MANIFEST[tool_name]`. There is no `lookup_tool_class`
helper and no `MissingToolError` exception class.

Boundary disposition (per §4.2.6 table):

- **Tier-3 input**: an LLM-supplied tool name with no entry in any of
  the six dispatch registries at `tools.py:5250–5314` is handled via
  the dispatcher's fall-through at `tools.py:5731` (`return
  _failure_result(state, f"Unknown tool: {tool_name}")`). The
  dispatcher returns a failure `ToolResult` **without raising** — no
  exception escapes. `dispatch_with_audit` therefore enters its SUCCESS
  branch and records `ComposerToolStatus.SUCCESS` with the failure
  payload (`result.data["error"] = "Unknown tool: <name>"`) in
  `result_canonical`. The compose loop continues; the LLM receives the
  failure payload as a `role=tool` message and can self-correct. This
  is the canonical **SUCCESS-with-semantic-failure** pattern: the
  dispatch itself completed successfully, and the audit record carries
  the full semantic outcome so an auditor can read it. The pattern is
  documented in the `ComposerToolStatus.SUCCESS` docstring
  (`contracts/composer_audit.py:34-37`). Pinned by
  `test_compose_loop_unknown_tool_name.py::TestUnknownToolNameComposeLoopAuditShape::test_unknown_tool_name_audit_shape`.
  (The JSON-decode / non-dict / missing-required-paths pre-dispatch
  gate at `service.py:1836–1870` is a separate, earlier site that
  handles malformed arguments for a tool name that was at least
  recognized; the unknown-tool-name fall-through is handled by
  `tools.py:5731`, not `service.py:1836–1870`.) **The walker is
  not invoked for unknown tool names** — `execute_tool` returns before
  dispatch completes, so there is no dispatched-tool record to redact.
- **Tier-1 invariant**: a manifest entry missing for a tool name
  that *did* dispatch successfully is a registry-consistency
  violation. The walker raises `AuditIntegrityError` chained from a
  `KeyError` on `MANIFEST[tool_name]`. The adequacy guard (§4.4.1)
  ensures this never fires in practice — registry-manifest set
  equality is asserted at CI time. If it ever fires, the registry
  itself is corrupt; offensive-programming policy applies and the
  audit trail must not proceed.

The two cases are distinguished by *whether dispatch succeeded*:
Tier-3 unknown-tool-name input never reaches the walker; only a
dispatched-but-not-manifest-registered tool name does. This matches
the trust-tier discipline in CLAUDE.md: external input (LLM tool
names) is Tier-3 and absence-tolerant; internal registry consistency
is Tier-1 and crash-on-violation.

---

## 6. Route Handling and Response Shape

### 6.1 Existing `_handle_*` helpers — minimal change

The three sibling handlers stay; they continue to persist `partial_state`
and emit it on the response body. They get one new field:

```python
response_body = {
    # existing fields:
    "error": ...,
    "reason": ...,                              # discriminated reason code (4fce0cae)
    "headline": ..., "evidence": [...],         # user-facing recovery copy
    "partial_state": redacted_partial_state,    # the partial CompositionState (existing)

    # new:
    "failed_turn": {
        "assistant_message_id": "...",          # FK back to chat_messages
        "tool_calls_attempted": 4,              # count from assistant.tool_calls
        "tool_responses_persisted": 3,          # count of role='tool' rows linked
        "transcript_url": "/api/sessions/{sid}/messages?since={user_message_id}&include_tool_rows=true",
    },
}
```

`tool_calls_attempted - tool_responses_persisted` surfaces "the LLM tried
4 tools, only 3 finished" without forcing a separate round-trip.

**Read consistency note.** `tool_responses_persisted` is computed by a
SELECT after the shielded audit writes have completed. The route helpers
await the surrounding compose-loop coroutine fully before computing the
count, so the shielded writes have committed by then. This is verified
by CL-PP-8 (mid-loop cancellation race).

### 6.2 Transcript fetch endpoint

The existing `GET /api/sessions/{sid}/messages` returns chat history. It
gains:

- New response fields per row: `tool_call_id`, `parent_assistant_id`,
  `sequence_no` (mirror the new columns).
- New query parameter: `include_tool_rows: bool = False`. Default `false`
  keeps the live chat panel's existing behavior (no tool rows interleaved
  into user/assistant flow). Recovery panel sets `include_tool_rows=true`.
- Default ordering: `(sequence_no ASC)` — monotonic per session.

### 6.3 Auth and redaction reuse

- Session ownership check unchanged for the default behaviour
  (`include_tool_rows=false`).
- **`include_tool_rows=true` requires audit-grade access logging** (NEW in
  rev 4 — closes security I-5). The transcript view exposes redacted-but-
  metadata-bearing rows (path summaries, blob byte counts, secret names,
  tool argument shapes); operationally these are more sensitive than the
  user/assistant chat content the default view exposes. The route helper
  performs:
  1. Session ownership check (existing).
  2. Emit an access-log row to a new `audit_access_log` table with
     fields `(timestamp, session_id, requesting_principal, request_path,
     query_args, ip_address)`. The access-log table is append-only (no
     `UPDATE` or `DELETE` paths exposed via SessionsService) and uses the
     same `writer_principal` CHECK pattern (`audit_grade_view`).
  3. Increment OTel counter `composer.audit.audit_grade_view_total`.
  4. Return the requested rows.

  The new table schema:

  ```python
  audit_access_log_table = Table(
      "audit_access_log",
      metadata,
      Column("id", String, primary_key=True),
      Column("timestamp", DateTime(timezone=True), nullable=False),
      Column(
          "session_id",
          String,
          # ON DELETE CASCADE so ``archive_session`` can delete sessions
          # that have audit-grade view rows.
          ForeignKey("sessions.id", ondelete="CASCADE"),
          nullable=False,
      ),
      Column("requesting_principal", String, nullable=False),  # auth subject
      Column("request_path", String, nullable=False),
      Column("query_args", JSON, nullable=False),
      Column("ip_address", String, nullable=True),
      Column("writer_principal", String, nullable=False),
      CheckConstraint(
          "writer_principal IN ('audit_grade_view', 'admin_tool')",
          name="ck_audit_access_log_writer_principal",
      ),
      Index("ix_audit_access_log_session_timestamp", "session_id", "timestamp"),
  )
  ```

- The audit_access_log is itself a Tier-1 audit surface. Its growth is
  bounded by recovery-panel access frequency; retention is filed under the
  same `[elspeth-RETENTION-WEB]` follow-up as `chat_messages` (§4.6).

- **Redact at write, never at read.** Persisted shape is the canonical safe
  shape; route handlers do not re-redact. This eliminates the entire class
  of bug "future read path forgets to redact."

### 6.4 What does NOT change

- The wall-clock 180s budget.
- The `_active` cross-session enumeration endpoint.
- OTel `composer.requests.inflight` gauge / `composer.request.terminal.total` counter.
- The `/composer-progress` endpoint.

---

## 7. Frontend Recovery Surface

### 7.1 Component layout

New directory `src/elspeth/web/frontend/src/components/recovery/`:

- `RecoveryPanel.tsx` — the modal.
- `RecoveryDiff.tsx` — pipeline diff section.
- `RecoveryTranscript.tsx` — tool transcript section.

New hook `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts` —
manages open/closed state and apply/discard semantics.

### 7.2 Trigger

`useComposer.sendMessage()` already has an `onError` path. The recovery
panel opens iff response body carries `partial_state` AND `failed_turn`.
The boolean-pair gating matrix is exhaustive — see §8 frontend tests.

### 7.3 Visual layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Recover composer draft — Turn 12 of compose run [reason badge]   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Headline copy (reason-keyed, from response body)                │
│  Evidence bullets                                                │
│                                                                  │
│  ─── Pipeline diff ──────────────────────────────────────────    │
│  [+ source: csv_users     ]                                      │
│  [+ transform: classify   ]                                      │
│  [~ sink: results.json    ]   options.path changed               │
│                                                                  │
│  ─── Tool transcript (3 of 4 completed) ────────────────────     │
│  ▸ list_transforms        ✓ 12 returned                          │
│  ▸ upsert_node(id=classify) ✓ state v2 → v3                      │
│  ▸ upsert_node(id=results) ✓ state v3 → v4                       │
│  ▸ wire_secret_ref         ✗ tool did not complete               │
│                                                                  │
│  [Apply partial draft]   [Discard]   [View raw transcript]       │
└──────────────────────────────────────────────────────────────────┘
```

### 7.4 Diff calculation

Client-side. Iterate `source / nodes / edges` in current editor state vs
`partial_state`; produce structured deltas. No new backend work.

### 7.5 Tool transcript

Pulled from
`GET /api/sessions/{sid}/messages?since={user_message_id}&include_tool_rows=true`.
Each `role='tool'` row contributes one entry; the assistant row's
redacted `tool_calls` JSON contributes the per-call argument summary.

### 7.6 Apply / discard / view raw

- **Apply.** Frontend overwrites in-memory editor state with `partial_state`.
  No server round-trip — `composition_states` already has the row.
  **Concurrent-edit guard**: if the editor's current state has been
  modified since the failed compose started (tracked by a client-side
  edit-counter), the Apply button shows a "this will overwrite your
  unsaved edits" confirmation. Test coverage in §8.4.
- **Discard.** Closes the panel; editor state unchanged. The DB record
  remains for audit (discard is a UI choice, not a data-deletion command).
- **View raw transcript.** Expands to a read-only view of every redacted
  tool call/response JSON.

### 7.7 Accessibility

- Focus-trapped modal (existing `Modal` pattern).
- Apply requires explicit click; no auto-apply on Enter.
- Reason badge has both colour and text label (no colour-only signalling).

### 7.8 What does NOT change

- Live chat panel (`components/chat/`) keeps its current default
  `include_tool_rows=false` — tool rows do not interleave into the live
  conversation by default. They appear only in the recovery panel and the
  raw-transcript view.

The frontend will get its own follow-up review pass per the user's note;
this design captures the contract and the surface, not the final visual
polish.

---

## 8. Testing Strategy

### 8.1 Backend — unit

- **Phase 2 redaction test suite (rev 5 — see plan tasks for the per-file canonical list).** The unit-test surface for the rev-5 redaction framework is split across multiple files (replacing the rev-4 single `test_redaction_policy.py`):
  - `tests/unit/web/composer/test_walk_model_schema.py` — ten container-shape coverage (original six: BaseModel, list, dict, tuple, Optional, Union; plus four rev-2 additions per `M_adequacy_mechanical_enforcement` M.4: duplicate-marker error, `list[list[BaseModel]]` compound nesting, `Annotated[T, Field(...), Sensitive()]` with `FieldInfo` in metadata, three-arm Union with non-BaseModel arms); marker-not-first-in-Annotated; three-level nesting; `with_values=True` extraction. Closes plan-review B2 plus rev-2 quality MAJOR-1.
  - `tests/unit/web/composer/test_tool_redaction_dataclass.py` — both-shapes-set / neither-shape-set / response_model-without-argument_model construction errors. Closes plan-review W8.
  - `tests/unit/web/composer/test_redaction_telemetry.py` — typed Protocol contract; Noop and OTel impls. Closes plan-review W4.
  - `tests/unit/web/composer/test_redact_set_source.py` — tracer-bullet end-to-end. Validates `Sensitive[T]` integration on one path before bulk migration.
  - `tests/unit/web/composer/test_handles_no_sensitive_data_reason.py` — empty locations, < 32-char reason, freeze-guard, idempotent `__post_init__`. (No `last_reviewed_iso` validator — calendar-keyed review removed in rev 5; closes W2.)
  - `tests/unit/web/composer/test_tool_redaction_policy.py` — orphan summarizer, `handles_no_sensitive_data` XOR reason, missing `known_response_keys`.
  - `tests/unit/web/composer/test_redact_tool_call_response.py` — known-key passthrough; sensitive-key substitution; **fixed-sentinel `<redacted-unknown-response-key>`** for unknown keys (no length disclosure; closes W6); summarizer-raises crashes via `AuditIntegrityError` (closes M2 / W5); summarizer-non-`str`-return crashes (closes W5).
  - `tests/unit/web/composer/test_redact_tool_call_arguments.py` — full disposition table from §4.2.6; type-driven walks via shared iterator; declarative walks via `sensitive_argument_keys`; manifest-entry-missing-for-dispatched-tool-name crashes (Tier-1 registry violation distinct from Tier-3 LLM-hallucinated tool name).
  - `tests/unit/web/composer/test_adequacy_guard.py` — five assertions per §4.4 (registry-manifest set equality; per-entry shape walk; mass-copy uniqueness; policy-hash snapshot equality; `extra="forbid"` on type-driven entries). Sanity bound: < 5 s for the current 37-tool set.
  - `tests/unit/web/composer/test_promote_*.py` — one per promoted tool (`set_source`, `create_blob`, `update_blob`, `set_source_from_blob`, `set_pipeline`, `apply_pipeline_recipe`, optionally `patch_*_options`). Each asserts `ToolArgumentError` raised on invalid input (with `pydantic.ValidationError` as `__cause__`, per the §4.2.6 promoted-handler pattern at `tools.py:2668, 2761, 2767, 2773, 2787, 2801`) AND existing handler-behaviour regression for valid input.
  - `tests/unit/web/composer/test_compose_loop_unknown_tool_name.py` — two pins: (1) `TestUnknownToolNameComposeLoopAuditShape::test_unknown_tool_name_audit_shape` pins the **SUCCESS-with-semantic-failure** audit shape for LLM-supplied unknown tool names: the dispatcher fall-through at `tools.py:5731` returns `ToolResult(success=False, data={"error": "Unknown tool: …"})` without raising; `dispatch_with_audit` records `ComposerToolStatus.SUCCESS` (not `ARG_ERROR`) with the failure payload in `result_canonical`; the compose loop continues. Closes plan-review M7 / W3. (2) `test_redact_tool_call_arguments_raises_for_unknown_tool` pins the Phase 3 call-order precondition (redaction walker must not be called before the unknown-tool check). Closes rev-3 M2.
  - `tests/unit/web/composer/test_redaction_completeness_property.py` — Hypothesis property test: for each manifest entry with `argument_model`, for generated valid payloads, no raw value of any `Sensitive`-annotated field appears in `json.dumps(redact_tool_call_arguments(…))`. Closes rev-2 BLOCKER_A quality MAJOR-4.
  - `tests/unit/web/composer/test_walker_guard_parity.py` — behavioural parity test asserting `walk_model_schema(M, with_values=False)` and `walk_model_schema(M, with_values=True)` produce identical path-sets and marker-presence for each manifest entry's argument model. Closes rev-2 M_walker_guard_parity.
  - `tests/unit/web/composer/redaction_policy_snapshot.json` — committed hash snapshot covering every manifest entry (type-driven + declarative; broadened from rev 4's legacy-only coverage). Closes plan-review M9 / W12.

  The rev-4 single-file `test_redaction_policy.py` does not survive into rev 5: the assertion surface is too broad to live in one file readable for review. Each rev-5 test module covers one mechanism; the adequacy guard is the integration point. The rev-4 "fallback summarizer-error sentinel" path is removed (closes M2 / W5; see §9 RSK-03 amendment); the rev-4 length-disclosing `<redacted-unknown-key:{n}-bytes>` sentinel is replaced by the fixed `<redacted-unknown-response-key>` form (closes W6).

- `tests/unit/web/sessions/test_chat_messages.py` (executed against
  in-memory SQLite — closes QA F-8)
  - Assert new `tool_call_id`, `parent_assistant_id`, `sequence_no`,
    `writer_principal` columns exist.
  - Assert `(role='tool') = (tool_call_id IS NOT NULL)` check rejects.
  - Assert `(role='tool') = (parent_assistant_id IS NOT NULL)` check rejects.
  - Assert composite index `(session_id, sequence_no)` is unique.
  - Assert partial unique index `(session_id, tool_call_id) WHERE role='tool'` rejects duplicates.
  - Assert `writer_principal` CHECK rejects unknown writer identities.
  - Assert `composition_states.provenance` CHECK rejects unknown
    provenance values.
  - Assert `ON DELETE CASCADE` from session removes all rows; from
    assistant row removes child tool rows (orphan prevention).

- `tests/unit/web/composer/test_compose_loop_persistence.py` (executed
  against in-memory SQLite)
  - Drive a fake LLM emitting 3 tool calls in one turn.
  - Assert: 1 assistant row with redacted `tool_calls`; 3 `role='tool'`
    rows with matching `tool_call_id`; ordered by `sequence_no`;
    `composition_state_id` set on rows where `state.version` advanced;
    `parent_assistant_id` matches the assistant row's `id`;
    `writer_principal='compose_loop'` for every row;
    `composition_states.provenance='tool_call'` for every advanced row.

- `tests/unit/web/composer/test_composer_holds_sessions_service.py`
  - Assert composer service constructor accepts and stores a
    `SessionsService` handle (the dependency that was missing pre-rev-3).

- `tests/unit/web/sessions/test_persist_compose_turn.py` (NEW in rev 4 —
  replaces the rev-3 `test_sessions_transaction.py`; the async
  `SessionsTransaction` is no longer introduced).
  - Assert `persist_compose_turn` is sync and writes all rows in one
    transaction (verified by injecting a SQLAlchemy event listener
    that counts `BEGIN`/`COMMIT` per turn — exactly one of each).
  - Assert rollback on injected `IntegrityError` rolls back ALL rows
    (no partial-turn state visible after the rollback).
  - Assert advisory lock is acquired and released (PostgreSQL only;
    inspect `pg_locks` between INSERT and COMMIT).
  - Assert that ``persist_compose_turn`` raises ``AuditIntegrityError``
    (chained from the underlying ``OperationalError``) when no plugin
    crash is pending and ``OperationalError`` is injected on COMMIT —
    Tier-1 audit-write failures raise inside the sync worker rather
    than returning a flag (§5.2.2; closes synthesised review finding
    H1).
  - Assert ``AuditOutcome.unwind_audit_failed=True`` when a plugin
    crash is pending and ``OperationalError`` is injected on COMMIT.

- `tests/unit/web/composer/test_audit_failure_primacy.py`
  - Tool succeeds + audit fails (non-IntegrityError on COMMIT) →
    ``AuditIntegrityError`` raised from the sync worker (chained from
    the underlying ``OperationalError`` via ``raise ... from``); counter
    ``composer.audit.tool_row_tier1_violation_total`` increments;
    ``AuditIntegrityError`` is registered in ``TIER_1_ERRORS`` so
    ``except Exception`` blocks cannot swallow it.
  - Tool fails + audit fails (non-IntegrityError on COMMIT) →
    ``AuditOutcome(assistant_id=None, unwind_audit_failed=True)``
    returned; counter
    ``composer.audit.tool_row_persist_failed_during_unwind_total``
    increments (different counter); log permitted under primacy
    (audit-system failure); caller raises the captured
    ``ComposerPluginCrashError``.
  - Audit `IntegrityError` (any constraint) → counter increments;
    caller raises (no recovery; no primacy disposition).
  - Tool-call cap exceeded → caller raises
    `ComposerConvergenceError(reason="tool_call_cap_exceeded")`
    BEFORE any tool execution.

- `tests/unit/web/composer/test_sensitive_marker.py` (NEW in rev 4)
  - Assert `Sensitive[T]` returns a `_SensitiveMarker` instance.
  - Assert reading `Annotated[str, Sensitive(summarizer=f)]` field
    metadata via Pydantic's `model_fields` API yields the marker.
  - Assert recursion descends into nested `BaseModel` fields,
    `dict[str, Sensitive[T]]`, and `list[Sensitive[T]]`.
  - Assert summarizer identity is preserved through `freeze_fields`
    (verified against `src/elspeth/contracts/freeze.py:78`).

### 8.2 Backend — integration

Extend `tests/integration/pipeline/test_composer_llm_eval_characterization.py`:

- **CL-PP-1: Convergence error mid-loop.** Force budget exhaustion at
  turn N. Assert assistant rows for 0..N exist; tool rows for completed
  calls exist; response body's `failed_turn.tool_calls_attempted` and
  `tool_responses_persisted` match observed counts; `partial_state`
  matches the latest `composition_states` row.
- **CL-PP-2: Plugin crash mid-loop.** A patched `execute_tool` raises
  `RuntimeError` on the second of three tool calls. Assert: assistant
  row + 1 normal tool row + 1 error tool row for the crashing call all
  committed atomically; the captured `ComposerPluginCrashError`
  propagates AFTER the audit write; `_handle_plugin_crash` runs;
  response body carries `partial_state` and `failed_turn`. (Rev-4
  collapse of rev-3 CL-PP-2a/2b: with single-sync-block atomicity, the
  audit-precedes-raise vs raise-precedes-audit distinction no longer
  exists — both are committed in the same transaction by construction.)
- **CL-PP-3: Wall-clock timeout during tool execution.** Force
  `asyncio.TimeoutError` in a tool. Assert the convergence error
  captured `partial_state`; audit trail consistent with captured state;
  shielded finally block produced the row even under cancellation.
- **CL-PP-4a: DB write fails on assistant row.** Inject a
  `add_message` failure on the assistant insert. Assert
  `partial_state_save_failed=true` propagates; no tool rows written.
- **CL-PP-4b: DB write fails on Nth tool row, tool succeeded.** Failure
  on the 2nd of 3 tool rows; tool itself returned successfully. Assert
  helper raises (Tier-1 audit failure with no in-flight tool exception).
- **CL-PP-4c: DB write fails on Nth tool row, tool also failed.**
  Failure on the same row; tool raised `Exception`. Assert helper logs,
  counter increments, helper returns; tool exception propagates.
- **CL-PP-4d: DB write fails on `composition_states` insert.**
  Inject `IntegrityError` on the `_insert_composition_state` call (e.g.,
  via a malformed `provenance` value). Assert the entire turn's
  transaction rolls back: no chat_messages rows visible, no
  composition_states rows visible. The bidirectional invariant holds
  by construction (single transaction).
- **CL-PP-5: Redaction summarizer raises mid-write.** End-to-end through
  the route layer. Assert fallback sentinel is in the persisted content;
  counter increments.
- **CL-PP-6: Cross-session leakage.** Same `tool_call_id` in two
  sessions. Assert no FK collision; no row from one session links to
  the other.
- **CL-PP-7: Mid-loop cancellation race.** Cancellation arrives between
  assistant-row write and a tool-row write. Assert both rows present
  due to `asyncio.shield`; route helper's `tool_responses_persisted`
  count matches the actual DB state.
- **CL-PP-8: Duplicate tool_call_id from misbehaving LLM.** LLM emits
  the same `tool_call_id` in two assistant turns of the same session.
  Assert the second write raises `IntegrityError` and the request fails
  fast — no silent recovery.

- **CL-PP-9: Mixed redaction policy (NEW in rev 4 — QA F-2).** A tool
  with both type-driven `Sensitive[T]` annotated fields AND non-sensitive
  fields. Drive a turn whose tool call carries values for every field.
  Assert the persisted `chat_messages.tool_calls` JSON: sensitive fields
  show sentinel/summarizer output; non-sensitive fields are
  byte-identical to input; structural shape preserved.

- **CL-PP-10: INSERT succeeded, COMMIT failed (NEW in rev 4 — QA F-2).**
  Inject an `OperationalError` at COMMIT (e.g., via a SQLAlchemy event
  hook on `commit` that raises). Two sub-cases:
  - **CL-PP-10a:** No plugin crash in flight. Assert
    ``persist_compose_turn`` raises ``AuditIntegrityError`` (chained
    from the injected ``OperationalError``);
    ``composer.audit.tool_row_tier1_violation_total`` increments;
    caller propagates; no rows visible in chat_messages (transaction
    rolled back). Tier-1 raise (no flag-return) — see §5.2.2.
  - **CL-PP-10b:** Plugin crash in flight. Assert
    ``AuditOutcome(assistant_id=None, unwind_audit_failed=True)``
    returned;
    ``composer.audit.tool_row_persist_failed_during_unwind_total``
    increments; log entry emitted (permitted under primacy); caller
    raises the captured ``ComposerPluginCrashError``.

- **CL-PP-11: Concurrent multi-session writes (NEW in rev 4 — QA F-2).**
  Two compose loops on session A and session B, sharing one PostgreSQL
  connection pool. The advisory lock uses the two-arg
  ``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
  hashtext(session_id))`` form (§5.7.1). Phase 1 lands the structural
  test against testcontainer Postgres
  (``tests/integration/web/test_compose_loop_concurrent_sessions.py``);
  the deliberate hash-collision variant remains a Phase 3 follow-up
  because it requires a seeded fixture that selects session_ids whose
  ``hashtext`` outputs collide modulo the classid slot. Assert: no
  deadlock; sequence_no values are independently monotonic per session;
  both transactions commit; the advisory-lock collision serialises but
  does not error.

- **CL-PP-12: Tool-call cap exceeded (NEW in rev 4 — RSK-13).** Drive
  the LLM to emit 17 tool calls in one assistant turn (cap is 16).
  Assert the loop raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")`
  BEFORE any tool execution; no DB writes for the over-cap turn; route
  helper emits the new reason code; counter
  `composer.tool_call_cap_exceeded_total` increments.

- **CL-PP-13: Unknown response key fail-closed (rev 4 → rev 5 framing).**
  A declarative-manifest-entry tool (with explicit `known_response_keys`)
  returns a response containing a key not in the allowlist. Assert the
  unknown key's value is replaced with the FIXED sentinel
  `<redacted-unknown-response-key>` (rev-5 sentinel; rev-4's
  `<redacted-unknown-key:{n}-bytes>` length-disclosing form is replaced
  per plan-review W6) in the persisted content;
  `composer.redaction.unknown_response_key_total` increments. The tool
  call still completes successfully — fail-closed redaction is not a
  failure, it is the policy.

### 8.3 Backend — property test (bidirectional INV-AUDIT-AHEAD)

Hypothesis-style stateful machine over a model of (LLM emissions,
tool executions, cancellations, retries, redaction policies).

#### 8.3.1 Strategy contracts

Strategies live in `tests/property/web/composer/strategies.py`.

**`st_tool_call`** — value space: well-formed `ToolCall` instances with
non-empty function names matching the registered tool registry, and
argument dicts whose keys match the tool's argument schema. Distribution:
70% well-formed; 30% with sentinel-marked sensitive keys for redaction
exercise. Coverage assertion: every test trace touches at least 3 distinct
tool names.

**`st_argument_dict`** — value space: dicts with string, int, bool, and
nested-dict values to depth 3. Includes orphan summarizer keys (rejected
at construction) and well-formed cases. Coverage: every test trace
exercises both well-formed and orphan-summarizer inputs.

**`st_redaction_policy`** — value space: tuples of (`sensitive_argument_keys`,
`sensitive_response_keys`, `argument_summarizers`,
`handles_no_sensitive_data`, `handles_no_sensitive_data_reason`).
Generates: empty policy + non-empty schema (rejected by adequacy guard);
non-empty policy; `handles_no_sensitive_data=True` with concrete reason;
`handles_no_sensitive_data=True` with stop-list reason (rejected).
Coverage: every adequacy-guard rejection branch exercised.

**`st_failure_injection_point`** — value space: enum of (`tool_returns`,
`tool_raises_ToolArgumentError`,
`tool_raises_AssertionError` (NEW in rev 4 — covers §5.5 row 3),
`tool_raises_Exception`,
`audit_raises_IntegrityError`,
`audit_raises_OperationalError_on_insert`,
`audit_raises_OperationalError_on_commit` (NEW in rev 4 — covers CL-PP-10),
`advisory_lock_unavailable` (NEW in rev 4 — covers CL-PP-11/§5.5 row 11),
`tool_call_cap_exceeded` (NEW in rev 4 — covers CL-PP-12),
`unknown_response_key` (NEW in rev 4 — covers CL-PP-13)).
Coverage: every §5.5 row reached at least once per test campaign;
mapping documented in `tests/property/web/composer/strategies.py` as
a docstring table (so future drift between §5.5 and the strategies is
catchable). Coverage assertions use Hypothesis `@example(...)`
decorators rather than relying on shrink-distribution heuristics, so
"reached at least once" is mechanically guaranteed (closes QA F-6).

**`st_cancellation_arrival_time`** — value space: enum of
(`before_step_1` (no LLM call yet),
`during_tool_execution` (mid Step 1),
`between_step_1_and_step_2` (after for-loop, before _run_sync dispatched),
`during_run_sync_before_insert` (worker entered, INSERTs not started),
`during_run_sync_between_insert_and_commit` (NEW in rev 4 — QA F-3; INSERT issued, COMMIT not yet),
`during_run_sync_during_commit` (COMMIT in flight),
`during_advisory_lock_acquisition` (NEW in rev 4 — QA F-3; pg_advisory_xact_lock waiting),
`after_commit_before_response_yielded` (NEW in rev 4 — QA F-3; route helper has not yet computed tool_responses_persisted),
`after_response_yielded`).
Coverage: every CL-PP-7-class race window exercised. The expanded enum
maps onto the §5.5 failure-mode rows 5-11; the property test verifies
the union is exhaustive across runs (any newly-discovered race window
shows up as a property-test counterexample, not silently).

**`st_session_state`** — value space: synthetic `CompositionState`
instances with `version ∈ [0, 50]` and 0-20 nodes. Coverage:
`version=0` (no partial state) and `version > 0` cases.

#### 8.3.2 Post-conditions (asserted after each trace)

```
Forward direction (audit can be ahead of state):
  count(role='assistant' rows for turn) == 1
  count(role='tool' rows for turn) <= N  (N = len(assistant.tool_calls))
  on failure: count(role='tool') == failed_turn.tool_responses_persisted
  every role='tool' row has tool_call_id matching exactly one entry
    in the assistant.tool_calls array

Backward direction (state never ahead of audit):
  for every committed composition_states row r where r.version > 0
    AND r resulted from a tool call (vs route-level convergence-error persist):
      there exists a chat_messages row m with role='tool' AND
      m.composition_state_id = r.id

Ordering & uniqueness:
  for every session, sequence_no values are STRICTLY MONOTONIC
    (no duplicates), but GAPS ARE PERMITTED — a transaction rollback
    after sequence reservation leaves the next caller's MAX+1 higher
    than the first successful row. Sequence_no is an ordering key, not
    a count. (Revision 4 weakens the rev-3 "densely monotonic" claim
    per architect H-3; the rev-3 claim was unachievable under reachable
    IntegrityError-during-atomic-pair traces.)
  for every (session_id, tool_call_id) where role='tool': exactly one row
  for every assistant row a, every child tool row t:
    t.created_at >= a.created_at
    t.parent_assistant_id == a.id
    t.sequence_no > a.sequence_no  (strict ordering within a turn)
    t.composition_state_id is NULL OR
      composition_state(t.composition_state_id).version >= composition_state(a.composition_state_id).version

Redaction:
  redacted output is structurally equal to input EXCEPT for declared
    sensitive_argument_keys / sensitive_response_keys
  redacted output is always a string-serializable JSON value
  redaction never raises through the persistence boundary

Cancellation specific:
  if cancellation_arrived_during_tool_execution=True:
    EITHER the tool row exists with the in-flight content,
    OR the assistant row never existed (no assistant row implies no tool rows).
  No third state.

Audit-failure primacy:
  if audit_raises and no tool exception in flight:
    the test framework observes the audit exception propagating out of the loop.
  if audit_raises and tool exception in flight:
    the test framework observes the tool exception (not the audit one) at the route layer.
  IntegrityError on chat_messages always propagates regardless of in-flight tool state.

OTel counter post-conditions (NEW in rev 4 — closes QA F-4):
  Across the full property-test campaign, the following counters must
  satisfy their SLO claims (§1.4):

  composer.audit.tool_row_tier1_violation_total == 0
    -- The Tier-1 violation MUST NOT fire under any test trace; if it
       does, either the test injected a Tier-1 fault (audit fail with no
       tool exception) and the test asserts the counter incremented,
       OR the production code has a bug.

  composer.audit.state_rolled_back_during_persist_total == 0
    -- The atomic-pair race is structurally impossible in the rev-4
       sync-block design; the counter exists for production observability
       but is never expected to fire.

  composer.audit.tool_row_persist_failed_during_unwind_total ==
    count(traces where audit raised OperationalError AND plugin crash was in flight)
    -- This is a telemetry-only counter, no SLO; the assertion verifies
       primacy disposition is consistent.

  composer.audit.tool_row_integrity_violation_total ==
    count(traces where IntegrityError was injected)
    -- The counter increments exactly once per IntegrityError trace.

  composer.redaction.summarizer_errors_total ==
    count(traces where summarizer raises was injected)

  composer.redaction.unknown_response_key_total ==
    count(traces where unknown_response_key was injected for legacy-escape-valve tools)

  composer.tool_call_cap_exceeded_total ==
    count(traces where tool_call_cap_exceeded was injected)

Schema-level backward-direction post-condition (NEW in rev 4 — closes QA F-1):
  After every trace, run the SQL predicate from §4.1.2:
    SELECT cs.id
      FROM composition_states cs
      LEFT JOIN chat_messages cm
        ON cm.composition_state_id = cs.id AND cm.role = 'tool'
     WHERE cs.provenance = 'tool_call' AND cs.version > 0
       AND cm.id IS NULL;
  Assert empty result set. The post-condition is now witnessable at the
  schema level rather than via test-harness bookkeeping.
```

### 8.4 Frontend

- `RecoveryPanel.test.tsx` — render with synthetic 422 response; assert
  diff section, transcript section, button states, accessibility hooks
  (`aria-modal`, focus trap, reason badge text label).
- `useRecoveryPanel.test.ts` — apply / discard semantics; **discard does
  NOT delete the DB record** (asserted via mock service expectation);
  apply does NOT round-trip to server.
- `useRecoveryPanel.test.ts` — **gating matrix.** Four boolean states of
  `(partial_state, failed_turn) ∈ {present, absent}²` — only the
  (present, present) case opens the panel; the other three fall back
  to the existing toast path.
- `useRecoveryPanel.test.ts` — **concurrent-edit-on-Apply.** User
  modifies the editor between failure response and Apply click; Apply
  shows confirmation dialog; only on confirm does it overwrite.
- `useComposer.test.ts` extension — when response body carries
  `partial_state` + `failed_turn`, the recovery panel opens; when it
  doesn't, the existing toast path runs.

Full Playwright/E2E round-trip is deferred to
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa) (final staging replay).

### 8.5 Verification scope (VER) — explicit VER/VAL boundary

This ticket (across all four phases — see §11) closes when:

1. Backend test set above is green.
2. Frontend test set above is green.
3. RC5-UX (or successor) branch CI passes including
   `enforce_tier_model.py` and `enforce_freeze_guards.py`.
4. All scenarios listed in §8.2 (CL-PP-1, 2, 3, 4a, 4b, 4c, 4d, 5, 6,
   7, 8, 9, 10a, 10b, 11, 12, 13 — sixteen total in rev 4) are present
   in `test_composer_llm_eval_characterization.py` and pass. CL-PP-11
   runs only against testcontainer PostgreSQL.
5. `tests/integration/web/test_inv_audit_ahead_backward.py` schema-level
   backward-direction test passes.
6. The OTel-counter post-conditions in §8.3.2 hold across the property-
   test campaign.
7. Policy-hash snapshot test (§4.4.3) passes.

**This ticket does NOT validate that users can actually recover from a
failure.** That validation is owned by
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa), the final staging
replay. VER is the contract; VAL is the user need; they are
intentionally separated. **The VAL ticket is a hard blocker on RC 5.1
release** (not a follow-up backlog item) so the VER/VAL split remains
honest — a "VER passed, VAL deferred indefinitely" outcome would
constitute VER-without-VAL theatre. Confirm
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa) carries the
`blocks-rc5.1` label before this ticket's Phase 4 closes.

#### 8.5.1 CI lane structure — testcontainer marker and aggregation

Phase 1C splits the CI matrix into a Docker-enabled lane and a set of
non-Docker lanes:

- **Docker-enabled lane.** Pulls the
  ``testcontainers[postgres]`` image and runs the full pytest suite
  including tests marked ``@pytest.mark.testcontainer``. The
  ``testcontainer`` marker is registered in ``pyproject.toml`` and
  carries the documented intent "Requires a Docker daemon and the
  ``testcontainers`` extra." Only this lane exercises CL-PP-11 and any
  future PostgreSQL-only assertions.
- **Non-Docker lanes** (Linux/macOS without a Docker socket, lint-only
  jobs, etc.). These runners cannot pull the Postgres image and MUST
  deselect the marker explicitly via ``pytest -m "not testcontainer"``.
  Implicit deselection (e.g. relying on import-time skip) is
  insufficient — pytest collects markered tests by default and would
  fail the lane on import errors before the skip fires.
- **``ci-success`` aggregation job.** A single GitHub Actions job that
  ``needs:`` every lane and inspects ``needs.<job>.result`` for each
  one. The aggregation passes only when every required lane reports
  ``success`` (and explicitly tolerates ``skipped`` only for lanes that
  declare themselves optional). Branch-protection requires
  ``ci-success`` rather than the individual lane jobs so a Docker
  outage on a single runner cannot silently mask a Phase 3 regression
  by causing the lane to be skipped without failing the merge gate.

The marker registration, the deselection expression, and the
aggregation job are all required by Schedule 1C Task 16; their
absence is a Phase 1C blocker, not a hardening follow-up.

### 8.6 Test path integrity — explicit composer rule

CLAUDE.md's "never bypass production code paths in tests" rule is about
not bypassing the production engine path (`ExecutionGraph.from_plugin_instances`,
`instantiate_plugins_from_config`); the composer is a different surface.
Revision 4 spells out the equivalent rule for composer tests:

**Composer integration tests MUST instantiate the full route → service →
SessionsService stack against either:**

- An in-memory SQLite database constructed via the production-equivalent
  helpers — ``create_session_engine("sqlite:///:memory:",
  poolclass=StaticPool)`` followed by ``initialize_session_schema(engine)``
  — suitable for unit-level integration that exercises check constraints,
  partial unique indexes, and CASCADE semantics. **Bare
  ``sqlalchemy.create_engine("sqlite:///:memory:")`` +
  ``metadata.create_all()`` is forbidden**: it bypasses the schema
  initialiser's PRAGMA wiring (most importantly ``PRAGMA foreign_keys =
  ON``), the ``StaticPool`` configuration that keeps the in-memory DB
  alive across connections, and any future schema-bootstrap steps that
  ``initialize_session_schema`` adds. Tests that bypass these helpers
  silently disagree with production on FK enforcement and on
  multi-connection lifetime, which is exactly the class of drift
  ``elspeth.web.sessions.engine`` exists to prevent. OR
- A testcontainer PostgreSQL database, for tests that need
  ``pg_advisory_xact_lock`` (the two-arg form keyed on
  ``ELSPETH_SESSIONS_LOCK_CLASSID`` — see §5.7.1), partial-unique-index
  dialect-specific behaviour, or multi-session concurrency (CL-PP-11).
  Constructed identically: ``create_session_engine(pg.get_connection_url())``
  + ``initialize_session_schema(engine)``. The pattern is captured in
  ``tests/integration/web/test_compose_loop_concurrent_sessions.py``.

**Mocking is permitted ONLY at:**

1. The LLM boundary (`ChaosLLM` fixture) — replaces the OpenRouter HTTP
   call with deterministic LLM emissions including structured `tool_calls`
   JSON arrays.
2. The tool-execution boundary (a fake `ComposerTool` registry exposing
   tools whose argument/response models and behaviour are controlled by
   the test).
3. Time (for testing wall-clock convergence; existing pattern via
   `time.monotonic` injection).
4. The OTel counter (replaced by an assertion-friendly fake; see §5.7.4).

**Mocking is FORBIDDEN at:**

- `SessionsService.persist_compose_turn` — the sync persistence primitive
  must execute against a real database; mocking it bypasses the
  bidirectional invariant verification that this spec exists to deliver.
- `SessionsService.add_message` for route-helper tests — same rationale.
- Any helper inside `persist_compose_turn` (`_acquire_session_advisory_lock`,
  `_reserve_sequence_range`, `_insert_chat_message`,
  `_insert_composition_state`) — these helpers exist to be exercised, not
  to be mocked away.
- The redaction layer (`apply_redaction_policy`, `redact_tool_call`) —
  these have their own unit tests in §8.1; integration tests must
  exercise the real redaction code path so end-to-end fidelity is
  asserted.

§8.1 unit tests that assert column existence, check-constraint behaviour,
and partial-unique-index rejection MUST execute against a real SQLite
engine (not metadata-only introspection — closes QA F-8). A
metadata-only test would pass against any declared schema regardless of
whether the database actually enforces the declarations.

CL-PP-11 specifically requires testcontainer PostgreSQL because SQLite
has no analogue for `pg_advisory_xact_lock` and the multi-session
concurrency test cannot run against the in-memory dialect.

### 8.7 Test data hygiene and fixture extension

All tests use the existing `chaos*` fixtures under `tests/`
(specifically `ChaosLLM` for composer LLM mocking). No live OpenRouter
calls in CI. The `elspeth-xdist-auto` plugin shipped inside
`src/elspeth/testing/` is separate from the project's own test suite,
per CLAUDE.md.

**`ChaosLLM` extension scope check (rev 4 — closes QA F-5).** Before
implementation begins, the implementer runs:

```bash
grep -rn "class ChaosLLM\|def.*tool_calls\|multi_turn" tests/ src/elspeth/testing/
```

and answers two questions in the Phase 1 PR description:

1. Does `ChaosLLM` currently support multi-turn state correlated to
   specific tool-call IDs? Required for CL-PP-1 (budget exhaustion mid
   loop), CL-PP-2 (causal-ordered plugin crash), and CL-PP-7
   (cancellation race).
2. Does `ChaosLLM` currently emit structured `tool_calls` JSON arrays
   matching the OpenRouter / OpenAI shape? Required for CL-PP-* across
   the board.

If either answer is "no, and extension is < 200 LOC," the extension
lands in the same Phase 1 PR. If extension is > 200 LOC, the
implementer files a precursor Filigree ticket for `ChaosLLM` extension
and Phase 1 of this design becomes blocked on that precursor ticket.
The "verify on day 1" rev-3 wording is replaced by an explicit
go/no-go decision recorded in the PR description.

---

## 9. Risks and Mitigations

| ID | Risk | Likelihood | Impact | Trigger | Mitigation | Owner |
|---|---|---|---|---|---|---|
| RSK-01 | Tool author ships a tool without redaction primitive (neither `Sensitive[T]` nor a legacy `ToolRedactionPolicy`). | Low | High (silent leakage) | Adequacy-guard CI test fires red on PR. | Recursive adequacy guard (§4.4) covers nested types; type-driven primitive (`Sensitive[T]`) is preferred; legacy policy is the explicit-exempt path with structured justification. | Implementing engineer per PR; reviewer sign-off |
| RSK-02 (rev 5: control set restated) | Declarative manifest entries with `handles_no_sensitive_data=True` normalize over time (Shifting the Burden) — the structured-reason field becomes a default copy-paste. | Low | Medium (erosion of audit safety) | Mass-copy uniqueness assertion (§4.4.4) fires on any two tools sharing exact-match `why_arguments_safe` or `why_responses_safe` text. Policy-hash snapshot diff (§4.4.3) without `policy-weaken-justified` PR label fails the label-gate CI step (§4.4.5). | (Rev-5 controls, replacing rev-4's CODEOWNERS-team routing which is non-functional on this repo:) Type-driven manifest entries (§4.2.1 with `argument_model`) are preferred for any tool whose argument surface benefits from a redaction-bearing Pydantic model. Declarative entries (§4.2.3) are the permanent shape for tools whose surface is structural. Adequacy-guard mass-copy uniqueness check forces concrete reasoning; policy-hash snapshot is content-keyed (review on change, not on calendar tick — closes W2); label-gate CI step is the merge control. | Implementing engineer per PR; label-gate CI step |
| RSK-03 (rev 5: discipline tightened) | Redaction summarizer raises on pathological input, OR returns a non-`str` value. | Very low (rev 5; was Low) | Critical (rev 5; was Medium) — Tier-1 audit invariant violation | Either: (a) summarizer raises during a per-turn audit-write, OR (b) summarizer returns a non-`str` value. Both surface inside `redact_tool_call_arguments` / `redact_tool_call_response` (§4.2.6). | **Persistence wrapper does NOT catch summarizer exceptions and does NOT coerce non-`str` returns.** Per CLAUDE.md plugin-ownership policy and Tier-1 discipline (§4.2.6), the summarizer is system code; an exception or wrong return type is a system bug, not a Tier-3 input fault. The walker raises `AuditIntegrityError` chained from the underlying exception (registered in `TIER_1_ERRORS` so `except Exception` cannot swallow). Property test (§8.1) asserts: (i) every committed summarizer returns `str` for every input value reachable through its declared field type; (ii) every summarizer is total (does not raise) on the same domain. The "fallback sentinel" path of rev 4 is removed — silent sentinel-substitution would mask the bug, and the audit row would record "we redacted this" when in fact the redaction implementation was broken. Closes plan-review M2 / M3 / W5 (Tier-1 silent-sentinel and non-str coercion). The rev-4 `composer.redaction.summarizer_errors_total` counter is retained for production observability; SLO threshold is 0 (any non-zero value is an incident, not a budget). | Implementing engineer per PR; SRE on incident |
| RSK-04 | Per-turn sync transaction slows down the loop. | Low | Low (latency NFR §1.4) | CI sanity bound (p95 ≤ 250 ms with N ≤ 8) red. | CI sanity bound + nightly tight bench; bounded write per turn; SQLite/PostgreSQL handle small transactions well; per-turn (not per-row) atomicity reduces transaction count. | Implementing engineer |
| RSK-05 | Frontend diff blows up on very large `partial_state`. | Low | Low (UX nuisance) | Recovery panel TTI exceeds 500 ms in CI sanity test. | Diff helper iterates fields rather than diffing entire JSON; UI shows "large diff — expand" disclosure for thousands of nodes. | Frontend follow-up |
| RSK-06 | New `tool_call_id` index slows large message-table writes. | Very low | Low | Insert latency regression in benchmarks. | Composite single-column index; SQLite/PostgreSQL handle this trivially. | n/a |
| RSK-07 | Audit-ahead-of-state invariant violated during cancellation. | Very low (rev 4 — was Low; mechanism now structural) | Critical (auditability standard breach) | Property test failure; or schema-level post-condition assertion fires. | Single-sync-block design (§5.2.1) eliminates cross-await coordination; cancellation cannot tear a transaction; backward direction provable from `composition_states.provenance` discriminator. CL-PP-3, CL-PP-7, expanded `st_cancellation_arrival_time` cover the cancel paths. | Implementing engineer; verified by property test |
| RSK-08 | `chat_messages` and `audit_access_log` tables grow unboundedly without retention. | Medium (over time) | Low (pre-release) | Table size exceeds 1 GB in dev/staging. | Cascade-delete with sessions today; retention extension filed under `[elspeth-RETENTION-WEB]` (§10 OQ-1). | RC 5.1 production hardening |
| RSK-09 | Partial unique index syntax differs across DB dialects. | Low | Low | DDL fails on a target dialect. | Use SQL `CREATE UNIQUE INDEX ... WHERE ...` (SQLite 3.8.0+; PostgreSQL); SQLAlchemy DDL emit hook for both dialects. | Implementing engineer |
| RSK-10 | State-rollback race during atomic-pair commit. | **Negligible (rev 4 — structurally impossible)** | n/a | Counter retained for production observability only; never expected to fire. | Single-sync-block design means there is no atomic-pair-across-await window. The counter `composer.audit.state_rolled_back_during_persist_total` exists but the §5.2 code path cannot increment it. Kept in the schema so a future re-introduction of multi-transaction grain (per-call atomicity) has the observability point ready. | n/a |
| RSK-11 | Audit-write failure when no tool exception in flight (Tier-1 violation). | Very low | Critical | OTel counter `composer.audit.tool_row_tier1_violation_total` non-zero. | §5.2.2 sync function raises `AuditIntegrityError` (registered in `TIER_1_ERRORS` so `except Exception` blocks cannot swallow it) chained from the underlying `OperationalError`; caller propagates; CL-PP-10a asserts. | RC 5.1 SRE |
| RSK-12 | LLM provider re-uses `tool_call_id` across turns within a session. | Medium (provider-dependent) | High (silent mis-correlation absent the partial-unique index) | Partial unique index rejects insert; CL-PP-8 fires; counter `composer.audit.tool_row_integrity_violation_total` increments. | Crash on duplicate; do not silently recover. Per-provider observation: OpenRouter/OpenAI ids are message-scoped per current spec but not contractually forever. | Implementing engineer |
| RSK-13 (NEW rev 4) | LLM-driven tool-call amplification via prompt injection. | Medium (LLM-dependent) | High (storage growth, Tier-3 input attack) | Per-turn cap of 16 tool calls fires; `composer.tool_call_cap_exceeded_total` increments. | Hard cap per assistant turn (configurable); CL-PP-12 covers; `_handle_convergence_error` returns the new reason code. | Security review at RC 5.1 |
| RSK-14 (rev 5: control restated) | Redaction-policy weakening lands without explicit justification (T-3). | Low | High (silent audit-safety regression) | Policy-hash snapshot test fails on PR (§4.4.3); the label-gate CI step (§4.4.5) requires either `policy-strengthen` or `policy-weaken-justified` PR label and a "Redaction policy weakening rationale" section in the PR description for the latter. | Snapshot test (§4.4.3) + label-gate CI step (§4.4.5). The rev-4 CODEOWNERS-team routing to `@elspeth/security` is dropped because the team cannot exist on `johnm-dta/elspeth` (personal-account repo; see W9 / §12.2). The label gate is the merge control. | Implementing engineer; label-gate CI step |
| RSK-15 (NEW rev 4) | Tool response-shape drift defeats static `sensitive_response_keys` (security I-1). | Medium (over time) | High (silent leakage of new keys) | `composer.redaction.unknown_response_key_total` non-zero. | `known_response_keys` allowlist with fail-closed redaction (§4.2.2); CL-PP-13 covers; counter alerts on first occurrence in production. | Implementing engineer; tool authors when extending response shapes |
| RSK-16 (NEW rev 4) | Actor attribution missing on rows from a future writer (security R-1). | Low | Medium (audit-trail gap for unattributable rows) | A new writer attempts to insert without a registered `writer_principal` value; CHECK constraint refuses. | `writer_principal` CHECK constraint (§4.1.1); new writers require schema migration that adds the value to the CHECK enum AND coordinates with the advisory lock. | Architecture review for any new writer |
| RSK-17 (NEW rev 4) | Audit-grade transcript view exposed without access logging (security I-5). | Low | Medium (forensic gap on access patterns) | `audit_access_log` row missing for an `include_tool_rows=true` request. | Route helper writes the access-log row before returning the response; integration test asserts the row exists for every audit-grade query. | Security review at RC 5.1 |

---

## 10. Open Questions

- **OQ-1 (resolved Phase 1C, Task 19).** Filigree ticket
  [elspeth-63012b19a5](filigree:elspeth-63012b19a5) — "chat_messages
  and audit_access_log retention CLI extension". P3, labels
  ``cluster:composer-progress-persistence`` and ``from-design-spec``.
  Extends the ``elspeth purge --retention-days`` CLI to operate on web
  session ``chat_messages`` and ``audit_access_log`` tables in addition
  to its existing pipeline-payload scope. Web tables grow only through
  cascade-delete with sessions today; an explicit retention path is
  needed before the production composer corpus exists. Cited as
  ``[elspeth-RETENTION-WEB]`` in §4.6, §6.3, and RSK-08; resolution
  cited in the Phase 1C closure PR description.
- **OQ-2.** `composition_states` redaction symmetry: today the partial
  state is persisted with raw paths; the new code path uniformly
  redacts. The historical asymmetry is filed as a follow-up issue
  during implementation. Document the planned ID in the PR description.
- **OQ-3 (NEW rev 4).** Audit-table integrity binding (per-row hash
  chain). Filed as a separate follow-up; mechanism sketch below so the
  implementer can pick it up without re-deriving the design. Not a
  blocker for the present ticket; the present design's auditability
  posture is structural (CHECK constraints, provenance discriminator,
  writer-principal attribution, access logging) rather than
  cryptographic.

  **Mechanism sketch (for the follow-up issue):**

  Add `chat_messages.integrity_hash` (`String(64)`, NOT NULL) and
  `chat_messages.prev_integrity_hash` (`String(64)`, NULL only for the
  first row in a session). On `INSERT`:

  ```python
  prev_hash = (
      session_first_row_hash if first_row
      else SELECT integrity_hash FROM chat_messages
           WHERE session_id = ? ORDER BY sequence_no DESC LIMIT 1
  )
  canonical = canonical_json({
      "session_id": ...,
      "sequence_no": ...,
      "role": ...,
      "content": ...,
      "tool_calls": ...,
      "tool_call_id": ...,
      "parent_assistant_id": ...,
      "writer_principal": ...,
      "composition_state_id": ...,
      "created_at": iso8601(...),
      "prev_integrity_hash": prev_hash,
  })
  integrity_hash = sha256(canonical).hexdigest()
  ```

  Optional HMAC-keyed variant: replace `sha256` with
  `hmac.new(deployment_key, canonical, sha256).hexdigest()`. The HMAC
  variant requires the key to detect tampering; the plain SHA-256
  variant only detects mutation if the original chain is preserved
  externally (e.g., periodic offsite backup of the latest hash).

  Add a `verify_chain(session_id) -> bool` helper that walks the
  session's rows in `sequence_no` order and recomputes each hash,
  comparing against the stored value. Schedule a periodic verifier
  (every N hours) that runs `verify_chain` for every session and emits
  an alert on mismatch.

  Same chain pattern applies to `composition_states` and
  `audit_access_log`.

- **OQ-4 (NEW rev 4).** Pre-deploy session-DB recreation step for
  staging (``elspeth.foundryside.dev``). Per
  ``project_db_migration_policy``, ELSPETH does not ship Alembic
  migrations or schema-version probes — schema changes are landed by
  destroying and recreating the session DB, never by row-level DELETE
  + structural ALTER. The §3 Migration ADR row authorises the
  recreation: the rev-3-era schema generated rows that cannot satisfy
  the new NOT NULL columns without a fabricated default, and a
  row-level ``DELETE FROM chat_messages`` followed by structural ALTER
  would smuggle backfilled defaults into the audit trail (Tier-1
  fabrication). The runbook step is:

  1. Stop ``elspeth-web.service`` (operator action; the staging
     deployment is single-process — see ``project_staging_deployment``).
  2. ``mv sessions.db sessions.db.archive-YYYY-MM-DD`` so the prior
     audit corpus is preserved offline rather than overwritten.
     ``DELETE`` is forbidden — operator-initiated archive only.
  3. Start the service. ``initialize_session_schema`` recreates the
     tables with the rev-4 column set on first connection.
  4. Verify in the same PR that the staging-deploy runbook captures
     this archive/delete/restart procedure in writing.

  The recreation discards the staging audit corpus by design — staging
  is pre-release and has no users; deferring the schema change until
  there are users would be the opposite of the project's "no legacy
  code" policy.

All four questions are administrative (file a ticket, cite an ID, update
a runbook); none blocks design or implementation.

---

## 11. Phased Delivery (NEW in rev 4)

Revision 4 demotes the tier-artifact match from S to M and splits
delivery across four phases. Each phase is an independently reviewable
PR against the parent epic
([elspeth-528bde62bb](filigree:elspeth-528bde62bb)). Phases are
sequential because each builds on the previous; running phases
concurrently would re-create the rev-3 tier-artifact mismatch.

### Phase 1 — `SessionsService.persist_compose_turn` sync primitive

> **Supersession notice — Task 19 closed (2026-05-09).** The Phase 1
> spec amendments listed below have been applied in-place to §4.1.1,
> §4.5, §5.2 / §5.2.1, §5.2.2, §5.7.1, §6.3, §8.5.1, §8.6, and §10
> OQ-4. The old Phase 1A/1B/1C continuation handovers are retired; the current
> implementation and this spec body are the authoritative source for future
> Phase 3 code work that builds on the primitive landed here. The list below is
> retained as a historical pointer to what the spec body now reflects:
>
> 1. `chat_messages.role` includes the internal value `"audit"`.
>    The plan keeps `"audit"` as a stored role but excludes it from
>    public route responses and composer prompt history; do not copy
>    spec wording that surfaces audit rows on the public surface.
> 2. `chat_messages.writer_principal` includes `"session_fork"`. The
>    plan's CHECK enum is the five-value form
>    `('compose_loop', 'route_user_message', 'route_system_message',
>    'admin_tool', 'session_fork')`, broadening the spec's earlier
>    four-value enum. Task 14 separately bans fork-time DEFAULTING to
>    `"session_fork"`: fork copies of source-session chat rows preserve
>    each source row's stored `writer_principal`, and role-keyed
>    fallback helpers are forbidden. `"session_fork"` remains a valid
>    value because (a) the schema must accept it for any future writer
>    that legitimately produces fork-only rows, and (b) the
>    `composition_states.provenance` enum (a separate column) uses
>    `"session_fork"` for the new seed state row created at fork time.
> 3. `parent_assistant_id` is enforced by a composite same-session FK:
>    `(parent_assistant_id, session_id) -> (chat_messages.id,
>    chat_messages.session_id)`. Use the plan's exact composite-FK
>    definition; do not infer a single-column FK from earlier text.
> 4. `_insert_composition_state` accepts `CompositionStateData`
>    directly and allocates versions under
>    `_session_write_lock`. No 1A caller supplies a precomputed state
>    version; do not copy spec wording that hands the caller a
>    pre-computed version.
> 5. PostgreSQL session write locks use
>    `pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
>    hashtext(session_id))`. The plan defers PostgreSQL operational
>    proof to Schedule 1C; for SQLite-current 1A, use the plan's
>    SQLite advisory-lock shape, not earlier `hashtextextended`
>    wording from this spec.
> 6. `SessionServiceImpl.persist_compose_turn` accepts optional
>    `raw_content` and `expected_current_state_id`, remains
>    concrete-only, and is wrapped by the protocol-public async
>    `SessionServiceProtocol.persist_compose_turn_async`. Use the
>    plan's exact signature, not any earlier sync-protocol or
>    raw_content-required wording.
> 7. ``AuditOutcome`` (renamed from ``_AuditOutcome`` in the Pre-Phase-3
>    hygiene pass once the type crossed the protocol boundary) has only
>    ``assistant_id`` and ``unwind_audit_failed``; Tier-1 audit-write
>    failures raise ``AuditIntegrityError`` from inside the sync worker
>    rather than returning a flag. Do not copy earlier four-field
>    shapes that included ``tier1_violation`` / ``tier1_violation_exc``
>    or that swallowed Tier-1 failures.
> 8. `composition_states.provenance` enum is the SIX-value form
>    `('tool_call', 'convergence_persist', 'plugin_crash_persist',
>    'preflight_persist', 'session_seed', 'session_fork')`. The spec
>    body asserts a five-value enum at §4.1.2 (line 241), §1.4
>    (line 308), and §6 (line 1305); all three are stale. The plan
>    adds `session_fork` for the new state row written when a fork
>    copies the source session's snapshot, and broadens
>    `session_seed` from "initial state row at session creation" to
>    "any state row written outside the compose loop's tool-call
>    path" — which now also covers `save_composition_state`
>    route-level saves and `set_active_state` within-session
>    reverts, not only the original session-creation seed case. Use
>    the plan's six-value list and the broadened `session_seed`
>    definition; do not copy the spec body's five-value enum or its
>    narrow `session_seed` definition. Site-by-site mapping of
>    which writer supplies which value is in plan §1017-1051 and is
>    enforced by Task 10's writer cutover.
>
> Session tests use `create_session_engine(..., StaticPool)` plus
> `initialize_session_schema()`, never bare `metadata.create_all()`.
> Task 19 of the Phase 1 plan rewrites the affected sections of this
> spec after the implementation lands; until then, treat the plan as
> the source of truth for Phase 1 mechanics.

**Scope.** Add the new schema columns (`writer_principal`,
`composition_states.provenance`, `audit_access_log` table — with
``ON DELETE CASCADE`` on ``audit_access_log.session_id``); add the
sync persistence primitive on `SessionsService` (with `raw_content`,
`expected_current_state_id`, the two-field ``AuditOutcome``, and the
async dispatcher ``persist_compose_turn_async`` on the protocol);
update existing `add_message` callers (`routes.py:1487`, `:1883`) to
pass `writer_principal`; wire ``_acquire_session_advisory_lock`` to
the two-arg ``pg_advisory_xact_lock(ELSPETH_SESSIONS_LOCK_CLASSID,
hashtext(session_id))`` form (B3) with the SQLite per-session RLock
fallback; refactor existing inline state-row inserts at
`web/sessions/service.py:395-418` and `:828-850` to call
`_insert_composition_state` and the shared
``_enveloped_state_column`` helper.

**Done when.** §8.1 unit tests pass against in-memory SQLite (real
database constructed via ``create_session_engine`` +
``initialize_session_schema``, not bare ``metadata.create_all``);
CL-PP-11 passes against testcontainer PostgreSQL on the Docker-enabled
CI lane (with the ``testcontainer`` marker registered, non-Docker lanes
deselecting it via ``-m "not testcontainer"``, and the ``ci-success``
aggregation job inspecting ``needs.<job>.result`` for every required
lane — see §8.5.1); ``enforce_tier_model.py`` and
``enforce_freeze_guards.py`` green; staging runbook updated for the
pre-deploy archive/delete/restart procedure (§10 OQ-4).

**Out of scope for Phase 1.** No compose-loop changes; no redaction
framework; no frontend.

### Phase 2 — Redaction framework: manifest dispatch, `Sensitive[T]` promotion wave, adequacy guard (rev 5)

**Scope (rev 5 reshape).** Introduce the `ToolRedaction` manifest
dataclass (§4.2.1) and the module-level `MANIFEST` keyed by tool name.
Add the `Sensitive[T]` field-level annotation (§4.2.2) and the
declarative `ToolRedactionPolicy` / `HandlesNoSensitiveDataReason`
manifest-entry shape (§4.2.3). Promote a wave of ~6–8 sensitive-touching
tools (the "Sensitive[T] promotion wave" — `set_source`, `create_blob`,
`update_blob`, `set_source_from_blob`, `set_pipeline`,
`apply_pipeline_recipe`, and any `patch_*_options` whose option dict
can carry secrets) to type-driven manifest entries:

  - Each promoted tool gains a redaction-bearing Pydantic argument
    model with `Sensitive[T]` annotations on the relevant fields.
  - The handler's dispatch path validates `Model.model_validate(arguments)`
    at the dispatch boundary. **Promoted handlers MUST catch
    `pydantic.ValidationError` and re-raise as `ToolArgumentError`**
    (pattern at `tools.py:2668, 2761, 2767, 2773, 2787, 2801`);
    `ToolArgumentError` is then caught at `service.py:2480` and routes
    to `ARG_ERROR`. A bare `ValidationError` escaping a handler hits the
    catch-all at `service.py:2564` and surfaces as
    `ComposerPluginCrashError` → HTTP 500, which is the wrong disposition
    for Tier-3 input.
  - The handler reads typed attributes from the validated instance,
    not `arguments["key"]` — replaces ~6–8 instances of the loose-dict
    pattern.

The remaining ~29–31 tools in the six dispatch registries get
declarative manifest entries (`ToolRedactionPolicy` with explicit
`sensitive_argument_keys`, `argument_summarizers`, `known_response_keys`
or `handles_no_sensitive_data=True` with structured reason). These
tools' handler signatures and dispatch paths are not changed by Phase
2 — their argument schema is structural and does not benefit from a
redaction-bearing Pydantic model.

Add the shared traversal iterator (§4.2.5); the runtime walker
`redact_tool_call_arguments` / `redact_tool_call_response` (§4.2.6);
the typed `RedactionTelemetry` Protocol (§4.2.4); the five-assertion
adequacy guard (§4.4); the broadened policy-hash snapshot covering
**every** manifest entry (§4.4.3); and the label-gate CI step
(§4.4.5). No `.github/CODEOWNERS` file is created.

**Done when.**

- `tests/unit/web/composer/test_adequacy_guard.py` passes all five
  assertions (registry-manifest set equality; per-entry shape walk;
  mass-copy uniqueness; policy-hash snapshot equality; `extra="forbid"`
  on type-driven entries).
- §8.1 redaction-policy unit tests pass, including coverage of
  `list[BaseModel]`, `dict[str, BaseModel]`, `tuple[BaseModel, ...]`,
  `Optional[BaseModel]`, `Union[*, BaseModel, *]`, marker-not-first-in-
  Annotated, duplicate-marker-error, three-level-nested redaction.
- Every promoted tool's dispatch path validates via
  `Model.model_validate(arguments)`; promoted handlers catch
  `pydantic.ValidationError` and re-raise as `ToolArgumentError` (per
  `tools.py:2668–2801` pattern); `ToolArgumentError` is caught at
  `service.py:2480`; the LLM receives the Tier-3 error response and
  continues.
- Every manifest entry is either type-driven (with `argument_model`
  set) or declarative (with `policy` set, never both).
- `redaction_policy_snapshot.json` covers all manifest entries
  (type-driven + declarative); the snapshot CI step asserts equality
  to the committed file.
- `composer-redaction-gate.yml` CI step passes on the Phase 2 PR
  (label-gate succeeds because the snapshot is being added on a
  greenfield, with a rationale section in the PR description if
  required by branch protection).
- `composer.redaction.summarizer_errors_total` SLO threshold is
  asserted to be 0 in production telemetry config (§9 RSK-03).
- compose-loop unknown-tool-name routing test confirms LLM-supplied
  unknown tool names route to ARG_ERROR via the existing path, NOT
  to a `MissingToolError` crash (closes plan-review M7 / W3).
- **Integration scenario pinned end-to-end (rev-2 BLOCKER_A).** At
  least one integration-level test exercises the full redaction →
  serialization → mock-persistence path for a `Sensitive[T]`-annotated
  field: invoke `redact_tool_call_arguments("set_source",
  args_with_canary_sensitive_value, telemetry=NoopRedactionTelemetry())`,
  pass the result through `json.dumps`, and assert (a) the canary value
  does not appear in the serialized form, (b) the field key DOES appear,
  (c) the redacted value matches the expected summarizer output. The
  persistence layer itself is mocked (Phase 3 wires the real one); the
  assertion is on the JSON that WOULD reach `chat_messages.tool_calls`.
- **Hypothesis completeness property test passes (rev-2 BLOCKER_A).**
  `tests/unit/web/composer/test_redaction_completeness_property.py`
  verifies: for each manifest entry with `argument_model`, for Hypothesis-
  generated valid payloads, no raw value of any `Sensitive`-annotated
  field appears in `json.dumps(redact_tool_call_arguments(…), sort_keys=True)`.
  `settings(max_examples=50, deadline=None)`. This is the load-bearing
  test that proves no `Sensitive[T]` field value reaches serialization
  unchanged.

**Out of scope for Phase 2.** No compose-loop persistence (Phase 3 ships
that); no frontend; no `.github/CODEOWNERS` file (label-gate is
sufficient until org migration); promoting non-sensitive-touching
tools to Pydantic argument models (the operator's "remove loose dicts
is always ongoing" direction continues outside Phase 2).

**Phase 3 preconditions (from Phase 2, rev-2 BLOCKER_A).** Phase 3
MUST observe these constraints when wiring the redaction layer into the
compose loop:

1. **MANIFEST-membership ordering.** `redact_tool_call_arguments` MUST
   be called only AFTER MANIFEST membership is confirmed for the
   dispatched tool name. `redact_tool_call_arguments(tool_name=…)`
   raises `AuditIntegrityError` when `tool_name` is absent from
   `MANIFEST` — this is reserved for system-internal registry-consistency
   violations, NOT for Tier-3 LLM hallucinations. Unknown LLM-supplied
   tool names MUST continue to route through the existing `_failure_result`
   path at `tools.py:5731` BEFORE redaction is attempted. Violating this
   ordering converts a graceful Tier-3 quarantine into a Tier-1 crash.
2. **`arguments_canonical` is NOT redacted.** Phase 3 MUST NOT thread
   redacted arguments through `begin_dispatch` / `begin_dispatch_or_arg_error`
   at `service.py:1930`. The `arguments_canonical` field retains raw
   LLM-supplied arguments (posture (a) — see §4.2.8). Redacted views
   land in `chat_messages.tool_calls` / `chat_messages.content` only.
3. **Tier-1 access controls on the audit sidecar.** The MCP JSONL sidecar
   and `BufferingRecorder` retain forensic completeness via raw
   `arguments_canonical`. Phase 3 must verify these surfaces have
   appropriate access controls before wiring.

### Phase 3 — Compose-loop persistence + tool-call cap

**Scope.** Modify `_compose_loop` (§5.2.1) to (a) enforce the per-turn
tool-call cap; (b) accumulate tool outcomes; (c) dispatch the protocol's
``persist_compose_turn_async`` (which bridges to the sync primitive via
``_run_sync``, shielded from caller cancellation); (d) raise
`ComposerPluginCrashError` after the audit write completes; (e) honour
``AuditOutcome.unwind_audit_failed`` for the unwind-path counter, and
let the sync primitive's ``AuditIntegrityError`` propagate (Tier-1
audit-write failures raise inside the worker rather than returning a
flag — see §5.2.2). Add the route-helper `failed_turn` field to 422/500
response bodies. Extend `GET /api/sessions/{sid}/messages` with
`include_tool_rows` parameter and the audit-grade access-log emission
(§6.3).

**Done when.** All §8.2 CL-PP-* scenarios pass (including new
CL-PP-9/10/11/12/13); §8.3 property test passes with the schema-level
backward-direction post-condition; OTel counter assertions match the
§1.4 SLO claims; `composer.audit.tool_row_tier1_violation_total`
remains 0 across the test campaign.

**Out of scope for Phase 3.** Frontend recovery panel; integrity-hash
chain (OQ-3 follow-up).

### Phase 4 — Frontend recovery panel

**Scope.** Implement `RecoveryPanel.tsx`, `RecoveryDiff.tsx`,
`RecoveryTranscript.tsx`, `useRecoveryPanel.ts` per §7. Wire to the
`failed_turn` field in error responses. Add concurrent-edit guard. Add
accessibility hooks (focus trap, reason badge text label).

**Done when.** §8.4 frontend test scenarios pass; manual smoke
against staging confirms diff renders for non-trivial pipelines and
transcript renders ≥50 rows without layout collapse; accessibility
audit (lighthouse / axe) passes.

**Out of scope for Phase 4.** Playwright/E2E round-trip is deferred to
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa) (final staging
replay). VAL of "the user can actually recover" is owned by that
ticket.

### Cross-phase considerations

- Each phase's PR description must cite the previous phase's commit as
  a dependency. Reviewers can merge them in order without re-reviewing
  earlier phases.
- The `[elspeth-RETENTION-WEB]` (OQ-1) and integrity-hash chain (OQ-3)
  follow-ups are filed during Phase 1 and 3 respectively; their
  delivery is independent of this ticket's close-out.
- If a phase's review surfaces a design issue that affects later phases,
  prefer fixing in-phase rather than carrying the change forward — the
  M-tier framing exists to keep each phase tractable.

---

## 12. References

- CLAUDE.md — auditability standard, three-tier trust model, plugin
  ownership, no-legacy-code policy, frozen-dataclass deep-freeze contract,
  defensive-vs-offensive programming policy.
- `engine-patterns-reference` skill — composite primary keys, schema
  contracts, secret handling, layer architecture & dependency analysis.
- `tier-model-deep-dive` skill — coercion rules, operation wrapping,
  fabrication decision test, web-server section.
- `logging-telemetry-policy` skill — audit primacy, telemetry-only
  exemptions, primacy test, permitted logger uses.
- Source report — `docs/composer/evidence/composer-llm-eval-2026-04-28.md`.
- Predecessor commits:
  - `5c17d380` — blob source path normalization.
  - `96c730d2` — secret-resolver routing for env markers.
  - `83e6228d` — redacted blob path sentinel preservation.
  - `48a4ab7f` — aggregation end-of-source trigger contract.
  - `7747b721` — batch-aware required input fields rejection.
  - `3844454f` — grouped batch-stats rollups.
  - `4fce0cae` — discriminated reason codes on `ComposerProgressEvent`.
  - `1ad03ddd` — in-flight composer observability + cancel-race drain.
  - `b21e9f1a` — inline blob user attribution.
- Predecessor verification: §2 cites the observed state of the codebase
  as of 2026-04-30, post-merge of all of the above. All cited symbols,
  classes, methods, commits, OTel counters, CI scripts, and test paths
  exist exactly as claimed (verified by reality-check pass on revision 3).
- Panel reviews:
  - Revision 1 reviewed 2026-04-30 by four reviewers (solution architect,
    systems thinker, Python engineer, QA analyst).
  - Revision 2 reviewed by the same four reviewers.
  - Revision 3 incorporated findings from both passes.
  - **Revision 4 (this revision)** addresses a six-reviewer first-principles
    panel that re-examined revision 3 from scratch (solution architect,
    systems-thinking pattern recognizer, Python engineer, QA analyst,
    security threat analyst, reality/hallucination check). The panel's
    no-go verdict on revision 3 motivated the architectural pivot to
    the single-sync-block design and the type-driven redaction primitive.

### 12.1 Revision 4 reviewer-finding traceability

Each accepted finding from the revision-3 panel has been mapped to a
spec section. Implementers can use this table to confirm coverage of a
specific concern without re-reading the entire panel response.

| Finding | Reviewer | Resolved in |
|---|---|---|
| C-1 SessionsTransaction not buildable on sync engine | Architect | §3 ADR (TX1); §5.2.1; §5.7.1; §5.7.2 |
| C-2 asyncio.shield semantics | Architect / Python engineer | §5.2.3 (shield removed entirely) |
| C-3 finally/NameError replaces CancelledError | Architect / Python engineer | §5.2.1 (no finally block); §5.2.4 cancellation table |
| C-4 Tier-artifact mismatch | Architect | Header (M-tier); §11 phased delivery |
| C-5 ComposerPluginCrashError.capture wrap omitted | Python engineer | §5.2.1 explicit wrap |
| H-1 Migration story | Architect | §3 ADR Migration row; §10 OQ-4 |
| H-2 Audit-fail counter contradictory | Architect | §1.4 NFR table (split into three counters) |
| H-3 Dense-monotonic sequence_no unachievable | Architect | §4.1.1 invariant prose (gaps permitted); §8.3.2 weakened post-condition |
| H-4 Adequacy-guard normalization | Architect | §4.4 hardening; §4.4.3 policy-hash snapshot; §4.4.5 CODEOWNERS |
| F-1 Backward-direction not provable from schema | QA | §4.1.2 provenance discriminator; §1.4 NFR; §8.3.2 SQL post-condition |
| F-2 Missing scenarios | QA | CL-PP-9, CL-PP-10a/b, CL-PP-11 added |
| F-3 Cancellation enum non-exhaustive | QA | §8.3.1 expanded enum |
| F-4 0-events SLOs not asserted by VER | QA | §8.3.2 OTel post-conditions |
| F-5 ChaosLLM extension hand-waving | QA | §8.7 explicit go/no-go check |
| F-6 Hypothesis coverage flake risk | QA | §8.3.1 @example decorators |
| F-7 Composer test-path-integrity rule unstated | QA | §8.6 explicit rule |
| F-8 Mocked-DB risk on schema tests | QA | §8.1 in-memory SQLite requirement; §8.6 forbidden mocks |
| T-1 Audit-trail integrity binding | Security | §10 OQ-3 (filed as follow-up; not blocking) |
| T-3 Policy-weakening detection | Security | §4.4.3 policy-hash snapshot |
| T-4 Adequacy guard nested-type traversal | Security | §4.4.1 recursive table |
| R-1 Actor attribution | Security | §4.1.1 writer_principal |
| I-1 Response-shape drift | Security | §4.2.2 known_response_keys; CL-PP-13 |
| I-2 wire_secret_ref reasoning | Security | §4.7 reasoning retained but flagged for security review per RSK-02 |
| I-3 Summarizer exception echo | Security | §8.1 test asserts exception-class-only sentinel; RSK-03 mitigation |
| I-5 Audit-grade transcript auth | Security | §6.3 audit_access_log + RSK-17 |
| D-1 LLM-driven amplification | Security | §1.4 cap; CL-PP-12; RSK-13 |
| Systems: type-driven redaction | Systems thinker | §4.2.1 Sensitive[T] |
| Systems: NFR goal drift | Systems thinker | §1.4 honest "redacted-transcript fidelity" wording |
| Reality: ToolArgumentError ONLY claim | Reality check | §2 status table enumerates all three except clauses |
| Reality: sessions/service.py:283-320 misidentified | Reality check | §2 corrected (call sites are routes.py:1487, :1883) |
| Reality: path shorthand | Reality check | Header conventions note |

### 12.2 Revision 5 reviewer-finding traceability

The four-reviewer plan-review pass on the rev-4-derived Phase 2 plan returned
`CHANGES_REQUESTED` with four BLOCKERs and twelve warnings.
Revision 5 closes each finding at the spec level so the plan rewrite
can be derived cleanly. Implementers should consult this table
alongside §12.1.

| Finding | Reviewer convergence | Resolved in |
|---|---|---|
| **B1** Class-based `ComposerTool` hierarchy is fictional; `tools.py` is six function-pointer dispatch dicts at lines 5250–5314 | reality + architecture + quality + systems (4-of-4) | Header rev-5 architectural pivot; §4.2 manifest-keyed dispatch; §4.4.1 registry-manifest set-equality assertion; §11 Phase 2 scope reshape |
| **B2** Walker omits recursion into `list[BaseModel]` / `dict[str, BaseModel]`; adequacy guard cannot detect the gap | quality + systems (2-of-2) | §4.2.5 shared traversal iterator (one definition consumed by both); §4.4.2 per-entry shape walk uses the same iterator; §8.1 test coverage of all ten container shapes (rev-5 original six + rev-2 M.4 four additions for compound nesting / `Field()`+`Annotated` / 3-arm Union / duplicate-marker error) |
| **B3** Spec §4.4.1 line 649 falsely claims tools declare arguments via Pydantic `BaseModel` subclasses | reality + quality + architecture | §4.4.1 rewritten against the actual function-pointer dispatch; rev-5 architectural pivot in header makes the correction load-bearing |
| **B4** Task 13 unconditionally invokes `gh pr create`, violating operator PR-confirmation discipline | quality | §11 Phase 2 done-when removes PR-open from scope; plan rewrite ends at "gate green; await operator PR-open instruction" (per Phase 1B/1C convention) |
| **W1** `ComposerTool` name collides with existing `ComposerToolInvocation` / `ComposerToolStatus` / `ComposerToolRecorder` L0 family | architecture + reality | No `ComposerTool` name introduced; manifest-entry dataclass is `ToolRedaction` (§4.2.1) |
| **W2** Calendar synchronicity at migration → 365-day mass batch expiry → date-bump ritual | systems + architecture | §4.4.4 replaces calendar-keyed review with content-hash-keyed review via the policy-hash snapshot (§4.4.3); `last_reviewed_iso` field removed from `HandlesNoSensitiveDataReason` |
| **W3** `MissingToolError` for LLM-hallucinated tool name treats Tier-3 input as crash condition | systems | §4.2.6 walker boundary table (new row: unknown tool name → SUCCESS-with-semantic-failure); §5.7.5 (audit status clarified); `tools.py:5731` fall-through returns `ToolResult(success=False, data={"error": "Unknown tool: …"})` without raising; `dispatch_with_audit` records `ComposerToolStatus.SUCCESS`; compose loop continues. Pinned by `test_compose_loop_unknown_tool_name.py::TestUnknownToolNameComposeLoopAuditShape::test_unknown_tool_name_audit_shape`. (The `service.py:1836–1870` JSON-decode pre-dispatch gate is a separate, earlier site.) Closed (Task 17 option a). |
| **W4** Telemetry call is duck-typed (`telemetry=None` default with attribute access) | quality | §4.2.4 typed `RedactionTelemetry` Protocol; walker accepts a real instance, never `None` |
| **W5** Summarizer non-string return value silently flows into Tier-1 audit row | quality | §4.2.6 walker disposition table; §9 RSK-03 amended (crash on non-`str` return) |
| **W6** `len(repr(value))` discloses structural metadata (RSK-03 weak echo) | quality | §4.2.3 sentinel `<redacted-unknown-response-key>` is fixed-form; §4.2.6 boundary table; no length disclosure |
| **W7** Mass-copy uniqueness check absent | quality | §4.4.4 explicit assertion in adequacy guard |
| **W8** `Sensitive[T]` vs `ToolRedactionPolicy` precedence undefined | quality | §4.2.7 + §4.2.1 `__post_init__` makes both-shapes-set a construction-time `ValueError`; precedence cannot arise |
| **W9** `@elspeth/security` CODEOWNERS team likely does not exist | reality + architecture + systems | §4.4.5 drops team routing; promotes label-gate CI step to primary control; no `.github/CODEOWNERS` file is created |
| **W10** Task 9 under-scoped to a single heading covering ~30 tools across 3 signature shapes | architecture | Plan rewrite splits manifest-entry creation into discrete tasks per registry (DISCOVERY, MUTATION, BLOB-DISCOVERY, BLOB-MUTATION, SECRET-DISCOVERY, SECRET-MUTATION); the Sensitive[T] promotion wave is a separate task per promoted tool |
| **W11** Default-form `hasattr` / `getattr` violations in plan tasks | reality | All §4.2 / §4.4 code sketches use direct typed-attribute access; `__post_init__` validators raise `ValueError` instead of catching attribute lookups; plan tasks reference these sketches verbatim |
| **W12** Snapshot covers only legacy-policy tools; new tools have a false-pass window | systems | §4.4.3 broadened to cover every manifest entry, type-driven and declarative; bootstrap snapshot enumerates all tools at creation |
| **M1** `test_redaction_policy.py` is *create*, not *replace* | reality | Plan rewrite says "create" |
| **M2** `sorted()` on non-string keys may error at runtime | quality | §4.4.3 canonicalisation routes through `sorted()` only on `str`-typed key sets (manifest keys are tool names, declared `dict[str, ...]`); §4.2.1 manifest type asserts string-typed keys |
| **M3** `redact_tool_call` double-parses JSON | systems | §4.2.6 entry points accept already-decoded `dict[str, Any]`; the compose loop decodes once at `service.py:1838` and passes the parsed structure to the walker |
| **M4** Adequacy guard CI scaling unaddressed | systems | Plan rewrite includes a sanity-bound assertion: adequacy guard < 5 s for 37 tools (the current full set); regression budget for future expansion is filed as a follow-up issue, not Phase 2 scope |

### 12.3 Revision 5 rev-2 plan-review finding traceability

The four-reviewer plan-review pass on the rev-5-derived Phase 2 plan rev-2
returned `CHANGES_REQUESTED` with three new BLOCKERs and five bundled MAJOR
groups. The rev-2 rewrite closed all four prior BLOCKERs (B1–B4) and all twelve
prior warnings (W1–W12). This section documents the rev-2 findings and where
each is addressed by this spec iteration.

| Finding ID | Title | Resolved in |
|---|---|---|
| **BLOCKER_A** | Phase 2 → Phase 3 contract integrity unproven at the persistence boundary | §4.2.6 (corrected `ValidationError` routing claim + ToolArgumentError wrapping pattern); §4.2.8 (arguments_canonical posture selection); §11 Phase 2 done-when (cross-boundary integration scenario criterion added); plan Task 4, 13–15 (ToolArgumentError wrap pattern + serialization boundary test); plan Task 17 (escalation rule); plan Task 19 (Hypothesis completeness property test) |
| **BLOCKER_B** | Label-gate does not enforce direction (weakening can be mislabelled as strengthening) | §4.4.5 (direction-aware comparator replaces symmetric jq check); plan Task 18 (rewritten workflow + direction-misclassification tests) |
| **BLOCKER_C** | Summarizer behavioural weakening is undetectable (hash tracks callable identity, not closure-state mutations) | §4.2.3 (summarizer purity assumption documented); §4.4.3 (known false-negative class for closure-state mutations documented); plan Task 19 (hash-semantics tests — replacement flips hash; mutation does not) |
| **M_adequacy_mechanical_enforcement** | Adequacy guard does not mechanically enforce all invariants the spec claims | §4.2.2 (`extra="forbid"` requirement added to type-driven argument models); §4.4.2 (AST inspection removed; replaced by promoted-model requirement for tools needing key-coverage guarantees; fifth adequacy assertion for `extra="forbid"` added); plan Task 1 (four additional walker container-shape tests); plan Task 10 (AST inspection step removed) |
| **M_telemetry_implementation** | Telemetry implementation is fictional (`_increment_counter` helper does not exist) and one counter unwired | §4.2.4 (`summarizer_error` method added to Protocol); plan Task 3 (rewritten around module-level `create_counter()` + `.add()` pattern per `service.py:135, 148, 824, 868, 1172, 1182`; `summarizer_error` counter wired before `AuditIntegrityError` raise) |
| **M_walker_guard_parity** | Walker↔guard parity asserted structurally but not behaviourally | plan Task 19 (walker-guard parity test added to `tests/unit/web/composer/test_walker_guard_parity.py`) |
| **M_governance_single_owner** | Label-gate is self-bypassing on a single-owner repo | `docs/guides/redaction-policy-changes.md` (new; documents single-owner governance note) |
| **M_blob_summarizer_type_variability** | Blob-tool summarizer type-safety not validated by tracer bullet | plan Task 13 (explicit type-variability verification step added) |
| **m_script_dir_missing** | Task 12 creates `scripts/composer/` directory which does not exist | plan Task 12 (script relocated to `scripts/cicd/bootstrap_redaction_snapshot.py`) |
| **m_citation_hygiene** | `service.py:1836-1870` citation misapplied throughout; that range is only the pre-dispatch JSON-decode/non-dict gate | §4.2.6, §4.4.2, §5.7.5, §8.1, §11 Phase 2 (all citations corrected; `service.py:2480` used for ToolArgumentError catch; `service.py:2564` for Exception catch-all) |

---

## 13. Glossary

- **Composer.** The LLM-driven pipeline-authoring service in
  `web/composer/`. Loops over LLM calls, executes tool calls against
  `CompositionState`, returns the final composed pipeline YAML.
- **Compose loop.** The bounded LLM-and-tool iteration inside the
  composer service, governed by wall-clock timeout, discovery-turn
  budget, composition-turn budget, and explicit termination on a
  no-tool-call assistant message.
- **Partial state.** A `CompositionState` snapshot captured by a
  failure-bearing exception (`ComposerConvergenceError`,
  `ComposerPluginCrashError`, `ComposerRuntimePreflightError`) iff
  `state.version > 0`. Persisted by the route helper and returned in
  the failure response body.
- **Tool call / tool response.** One round-trip in the compose loop:
  LLM emits a tool call; composer executes it against `CompositionState`;
  composer returns the response to the LLM (and persists it as a
  `role='tool'` row).
- **Audit-ahead-of-state (INV-AUDIT-AHEAD).** The bidirectional invariant
  that `chat_messages` may be ahead of `composition_states` (showing
  what was attempted) but never behind, AND `composition_states` must
  not be ahead of `chat_messages` (every committed state-mutating tool
  has its row). Derived from CLAUDE.md "no inference — if it's not
  recorded, it didn't happen."
- **Sequence number (`sequence_no`).** Monotonic per-session integer
  reserved atomically inside `add_message`'s transaction via session-scoped
  advisory lock. Replay fidelity depends on it; clock-based ordering is
  insufficient (sub-millisecond collisions and clock jumps).
- **`ToolArgumentError`.** The Tier-3 boundary signal documented at
  `web/composer/protocol.py:291` — the only exception class the compose
  loop catches around `execute_tool()` and continues past. The compose
  loop is contractually obligated to *not* re-raise `ToolArgumentError`.
- **Audit-failure primacy.** When an audit write fails, the disposition
  depends on whether a tool exception is in flight. Tool succeeded +
  audit fails → raise (Tier-1 invariant violation). Tool fails + audit
  fails → log permitted, increment counter, let tool exception propagate.
  See §5.2.2 sync helper.
- **`Sensitive[T]`** (rev 4; rev-5 framing). Type-driven redaction
  metadata, used as Pydantic field metadata via
  `Annotated[T, Sensitive(summarizer=...)]` on fields of a manifest
  entry's `argument_model` or `response_model` (§4.2.2). The walker
  reads `model_fields[name].metadata` to detect markers; promoted
  tools' dispatch validates `Model.model_validate(arguments)` at the
  dispatch boundary so the model also serves as the validation gate.
- **`ToolRedaction`** (rev 5). The manifest-entry dataclass (§4.2.1).
  Each entry carries either an `argument_model` (type-driven shape)
  or a `policy: ToolRedactionPolicy` (declarative shape) — exactly
  one. Construction-time `ValueError` if both or neither is set.
- **Manifest** (rev 5). `MANIFEST: Mapping[str, ToolRedaction]` —
  module-level dict keyed by tool name; the single registration root
  for redaction policy. Mirrors the `_TOOL_REQUIRED_PATHS:
  dict[str, ...]` precedent at `service.py:702`. The adequacy guard
  walks it; the runtime walker dispatches through it; the policy-hash
  snapshot covers every entry.
- **`ToolRedactionPolicy`** (rev 4; rev-5 framing). The declarative
  manifest-entry shape (§4.2.3) for tools whose argument surface is
  structural (graph IDs, node names, plugin-key strings). Carries
  `sensitive_argument_keys`, `argument_summarizers`,
  `known_response_keys`, and (when `handles_no_sensitive_data=True`)
  a structured `HandlesNoSensitiveDataReason`. Not a "legacy" shape
  in rev 5 — it is the correct permanent representation for tools
  whose surface does not benefit from a redaction-bearing Pydantic
  model.
- **`RedactionTelemetry` Protocol** (rev 5). Typed Protocol for the
  walker's OTel emissions (§4.2.4). Replaces rev-4's duck-typed
  `telemetry=None` parameter. Walker accepts a real instance, never
  `None`.
- **Shared traversal iterator** (rev 5). The single generator
  (§4.2.5) that yields one `TraversalNode` per field of a Pydantic
  model schema, descending into `BaseModel`, `list[BaseModel]`,
  `dict[str, BaseModel]`, `tuple[BaseModel, ...]`, `Optional[BaseModel]`,
  `Union[..., BaseModel, ...]`, and `Annotated[T, *, Sensitive(), *]`
  regardless of marker position. Consumed by both the adequacy guard
  (CI-time) and the runtime walker (persistence-time) so they cannot
  diverge. Closes plan-review B2 (walker-vs-guard divergence).
- **Label-gate CI step** (rev 5). The merge control (§4.4.5) that
  enforces `policy-strengthen` or `policy-weaken-justified` PR labels
  on any change that flips the policy-hash snapshot. Replaces rev-4's
  `@elspeth/security` CODEOWNERS team routing, which is non-functional
  on `johnm-dta/elspeth` (personal-account repo; GitHub teams require
  an organisation). Closes plan-review W9 / M10.
- **`writer_principal`** (rev 4). Required column on `chat_messages`
  recording which subsystem wrote the row. Permitted values are
  CHECK-constrained: `compose_loop`, `route_user_message`,
  `route_system_message`, `admin_tool`. New writers require schema
  migration that extends the CHECK enum.
- **Provenance discriminator** (rev 4). Required column on
  `composition_states` recording which code path committed the row.
  Permitted values: `tool_call`, `convergence_persist`,
  `plugin_crash_persist`, `preflight_persist`, `session_seed`. Backward-
  direction INV-AUDIT-AHEAD applies only to `provenance='tool_call'`
  rows; the other values record route-helper persists of partial state.
- **Per-turn atomicity** (rev 4). The grain at which audit-trail rows
  commit. One LLM turn produces one transaction containing the
  assistant row + N tool rows + N state rows + sequence-number
  reservations. Either all of them land or none does. Replaces the
  rev-3 per-tool-row atomicity grain.
- **Single-sync-block design** (rev 4). The rev-4 architectural pivot.
  Per-turn writes are dispatched via `_run_sync` to a sync function
  that runs the entire transaction in one `engine.begin()` context.
  Eliminates `asyncio.shield`, the `try/except/finally` block, and the
  cross-await coordination problem.
- **Audit-grade transcript view** (rev 4). The
  `GET /api/sessions/{sid}/messages?include_tool_rows=true` response
  variant. Triggers an `audit_access_log` row write per access.
  Considered a Tier-1 surface; access requires session ownership AND
  emits its own access log.
- **Adequacy guard** (rev 4; rev-5 framing; rev-5 rev-2 iteration adds
  fifth assertion). The CI-time test (§4.4) that asserts: (1) every
  tool name in the six dispatch registries at `tools.py:5250–5314` has
  exactly one `MANIFEST` entry; (2) every entry's redaction declaration
  covers every field where redaction may be required, walked via the
  shared traversal iterator (§4.2.5); (3) no two declarative reasons are
  exact-match copies; (4) the policy-hash snapshot equals the committed
  file; (5) every type-driven entry's argument model sets
  `extra="forbid"`. All five assertions live in
  `tests/unit/web/composer/test_adequacy_guard.py`.
- **Policy-hash snapshot** (rev 4; rev-5 broadened). A SHA-256 hash
  per manifest entry — type-driven and declarative alike (rev-5
  broadening; rev 4 covered only declarative). For type-driven entries
  the hash is over the schema-walk produced by the shared iterator
  (so a removed `Sensitive()` marker flips the hash); for declarative
  entries the hash is over the canonical key sets plus the structured
  reason's text hash. Committed at
  `tests/unit/web/composer/redaction_policy_snapshot.json`. Changes
  require the label-gate CI step to pass (§4.4.5). Closes plan-review
  M9 / W12.
- **Tool-call cap** (rev 4). Per-turn hard limit on the number of tool
  calls an assistant turn may emit. Default 16. Exceeding the cap
  raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")`
  before any tool execution. Defends against prompt-injection-induced
  amplification.
- **`AuditOutcome`** (rev 4; renamed from ``_AuditOutcome`` in the
  Pre-Phase-3 hygiene pass once the type crossed the protocol
  boundary). The dataclass returned by
  ``SessionsService.persist_compose_turn``. Two valid shapes:
  (1) success — ``assistant_id`` populated, ``unwind_audit_failed=False``;
  (2) tool failed AND audit unwind failed — ``assistant_id=None``,
  ``unwind_audit_failed=True`` (caller proceeds to raise the captured
  plugin crash). There is NO ``tier1_violation`` field: Tier-1
  audit-write failures raise ``AuditIntegrityError`` directly inside
  the sync worker (registered in ``TIER_1_ERRORS``). See
  ``src/elspeth/web/sessions/_persist_payload.py`` for the canonical
  definition; ``RedactedToolRow`` and ``StatePayload`` live alongside it.
