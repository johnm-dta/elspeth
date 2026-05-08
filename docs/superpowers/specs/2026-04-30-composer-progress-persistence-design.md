# Composer Progress Persistence — Design

**Ticket:** [elspeth-90b4542b63](filigree:elspeth-90b4542b63) — Composer progress persistence — tool-call breadcrumbs and partial drafts survive long-running failures
**Parent epic:** [elspeth-528bde62bb](filigree:elspeth-528bde62bb) — Composer LLM evaluation remediation
**Related future epic:** [elspeth-f0460a6594](filigree:elspeth-f0460a6594) — Composer async/background execution model (deferred to Future release)
**Date:** 2026-04-30
**Status:** Proposed (revision 4 — addresses six-reviewer first-principles panel: solution architect, systems-thinking pattern recognizer, Python engineer, QA analyst, security threat analyst, reality/hallucination check)
**Branch:** RC5-UX (or successor)
**Tier-artifact match:** **M-tier** change delivered as a four-phase plan. Revision 1 labelled XS; revision 2 grew to S; revision 3 retained S; revision 4 demotes to M and explicitly splits delivery (see §11) because the `SessionsTransaction` primitive (Phase 1), the redaction framework (Phase 2), the compose-loop persistence (Phase 3), and the frontend recovery surface (Phase 4) are independently reviewable, independently testable, and independently deployable. Each phase is a separate PR against the parent epic.

**Path conventions.** Throughout this spec, `web/...` and `sessions/...` are shorthand for `src/elspeth/web/...` and `src/elspeth/web/sessions/...` respectively. Full paths appear the first time a file is cited per major section; later references in the same section may use the shorthand. Test paths are written full from repo root (`tests/...`). Implementers cloning the repo and following a citation should prepend `src/elspeth/` when the path begins with `web/`.

**Architectural pivot in revision 4.** Revision 3 proposed an async `SessionsTransaction` context manager interleaved with `asyncio.shield`-protected audit inserts. The first-principles panel established that (a) the current `SessionsService` is built on a sync `Engine` dispatched via `_run_sync`/`run_sync_in_worker`, so a multi-await transaction context cannot share a connection without pinning a worker thread; (b) `asyncio.shield` does not deliver "the inner write completes before the outer await returns" — it detaches the shielded coroutine and lets the outer await re-raise `CancelledError` immediately; (c) the proposed `try/except/finally` block reads an unbound local in the `BaseException` path, raising `NameError` that replaces the in-flight `CancelledError`. Revision 4 resolves all three by collapsing each per-turn write (assistant row + N tool rows + corresponding `composition_states` rows + sequence-number reservations) into **one sync block dispatched through `_run_sync`**. This eliminates the cross-async-boundary race surface entirely. The `SessionsTransaction` primitive becomes a sync-only context yielded inside the `_run_sync` worker. See ADR row "Transaction primitive shape" in §3 and the rewritten §5.2 / §5.7.

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
| Redaction-summarizer failure rate | ≤ 0.1% of tool calls in a 24-hour window; alerts above | OTel counter `composer.redaction.summarizer_errors_total` |
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
| Redaction primitive (NEW in rev 4) | **R4.** **Type-driven**, with declarative escape valve. The primary mechanism is a `Sensitive[T]` `Annotated`-based marker on Pydantic argument and response model fields. The persistence layer reads field annotations and redacts marked fields unconditionally — the tool author cannot opt out. The legacy `ToolRedactionPolicy` declaration is retained ONLY for cases where the type-annotation approach is impractical (raw dict accepted, third-party model) and is gated behind an explicit `EXEMPT_FROM_TYPE_DRIVEN_REDACTION: ClassVar[bool]` flag on the tool class. The structural Level-4 leverage (Meadows hierarchy) is the type system itself; declaration discipline is no longer the dominant defence. | R3 (rev-3 design): per-tool string-keyed `ToolRedactionPolicy` with `handles_no_sensitive_data=True` opt-out and quarterly review. Vulnerable to declaration-burden normalisation (RSK-02 in rev 3). | One-way after first compose run executes against the new layer (any tool already migrated to `Sensitive[T]` cannot revert without churn). | Remove the `Sensitive[T]` annotation from every tool's argument/response models and reintroduce the rev-3 string-keyed policy. Costs auditability discontinuity. | Per-tool annotation burden (small, structural) + adequacy-guard (§4.4 expanded) + structured `handles_no_sensitive_data` schema for the legacy escape valve | RC 5.1 security review; new `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` declarations require security CODEOWNERS approval per PR |
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
    Column(                                            # NEW (rev 3) — explicit cascade
        "parent_assistant_id",
        String,
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=True,
    ),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_chat_messages_composition_state_session",
    ),
    CheckConstraint(
        "role IN ('user', 'assistant', 'system', 'tool')",
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
        # writer_principal is one of the four expected sources; future writers
        # require a schema migration that adds the new value to this CHECK.
        "writer_principal IN ('compose_loop', 'route_user_message', 'route_system_message', 'admin_tool')",
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

The CHECK constraint pins the enum at the database level; a new writer
identity requires an explicit schema migration that extends the CHECK,
which forces architectural review of any new write surface. This is the
mechanical mitigation for the "single-writer-per-session structurally
enforced" assertion in §5.7 — instead of relying on convention, the
database refuses unrecognised principal strings.

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
        "'plugin_crash_persist', 'preflight_persist', 'session_seed')",
        name="ck_composition_states_provenance",
    ),
)
```

`provenance` records which code path committed the row. Permitted values:

- `tool_call` — written by `_compose_loop` as part of the atomic per-tool
  write (assistant + tool + state rows in one transaction). This is the
  ONLY value for which the backward-direction INV-AUDIT-AHEAD invariant
  applies: every `('tool_call', version > 0)` row MUST have a corresponding
  `chat_messages` row with `role='tool'` and matching `composition_state_id`.
- `convergence_persist` — written by `_handle_convergence_error` route
  helper after a wall-clock timeout or budget exhaustion captured
  `partial_state` from a `ComposerConvergenceError`.
- `plugin_crash_persist` — written by `_handle_plugin_crash` route helper
  after a `ComposerPluginCrashError` captured `partial_state`.
- `preflight_persist` — written by `_handle_runtime_preflight_failure`
  route helper after a `ComposerRuntimePreflightError`.
- `session_seed` — initial state row written when a session is created
  with seed configuration (existing path; covered by the value set so that
  legacy rows do not violate the new CHECK after the staging DELETE/recreate
  step described in the §3 Migration ADR row).

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
- `parent_assistant_id` is non-null iff `role='tool'`, with `ON DELETE CASCADE` so deleting an assistant row removes its tool rows. (`ck_chat_messages_parent_role`.)
- `(session_id, sequence_no)` is unique — every row in a session has a unique sequence number, monotonically increasing in commit order. **Sequence numbers are ordering keys, not counts.** Gaps are permitted (e.g., when an atomic-pair transaction rolls back after the next free sequence number was reserved). The property test post-condition asserts strict monotonicity within a session, NOT density. Closes architect H-3.
- `(session_id, tool_call_id)` is unique among `role='tool'` rows. Cross-turn collisions (same `tool_call_id` reused by the LLM provider in a different turn) are rejected as a Tier-3 input-validation failure: the compose loop crashes the request rather than silently mis-correlating.
- `writer_principal` is one of the four enumerated values; the database refuses any other string at write time.
- `composition_states.provenance` is one of the five enumerated values; backward-direction INV-AUDIT-AHEAD applies only to rows with `provenance='tool_call'`.

`composition_state_id` FK behaviour — see §4.5.

### 4.2 Redaction primitives — type-driven first, declarative escape valve second

**Location.** `src/elspeth/web/composer/redaction.py` (L3, alongside the
existing `redact_source_storage_path` helper). Tools are L3; their
redaction primitives belong in the same layer.

#### 4.2.1 `Sensitive[T]` — type-driven primitive (NEW in rev 4)

```python
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, TypeVar

T = TypeVar("T")


class _SensitiveMarker:
    """Annotated metadata marker. Presence on a Pydantic field indicates
    the field's value MUST be redacted at the persistence boundary. The
    tool author has no opt-out path — redaction is mechanical, not
    declarative.

    summarizer: optional per-field replacement function. If None, the
        sentinel '<redacted>' is substituted. If present, the function
        receives the original value and returns the replacement string.
    """

    __slots__ = ("summarizer",)

    def __init__(self, summarizer: Callable[[Any], str] | None = None) -> None:
        self.summarizer = summarizer


def Sensitive(  # noqa: N802 — capitalised to read as a type alias at use sites
    *, summarizer: Callable[[Any], str] | None = None
) -> _SensitiveMarker:
    """Field-level annotation requesting redaction at the persistence
    boundary. Use as Pydantic field metadata via ``Annotated``.

    Example:
        from typing import Annotated

        class SetSourceArguments(BaseModel):
            path: Annotated[str, Sensitive(summarizer=redact_source_storage_path)]
            options: Annotated[dict, Sensitive()]
            label: str    # not sensitive, persisted verbatim
    """
    return _SensitiveMarker(summarizer=summarizer)
```

**Why this is a Level-4 (Meadows hierarchy) intervention rather than a
Level-5 / Level-6 mitigation.** The redaction policy is now expressed in
the type system. There is no "I declare this tool has no sensitive data"
flag for fields whose model uses `Sensitive[T]`; the persistence layer
reads the model's `Annotated` metadata and acts on it. The
declaration-burden-normalisation feedback loop (rev-3 RSK-02) cannot run
because there is nothing to declare. New tools that adopt `Sensitive[T]`
inherit the redaction guarantee mechanically.

**Redaction layer reads annotations recursively.** `apply_redaction_policy`
descends into nested Pydantic submodels and into `dict`/`list`-typed
`Sensitive[T]` fields, applying the marker at every level. Implementation
sketch:

```python
def _redact_pydantic_value(value: Any, model: type[BaseModel]) -> Any:
    """Walk a Pydantic model and produce a redacted dict.
    For each field:
      - If field metadata contains _SensitiveMarker, substitute sentinel/summarizer.
      - If field type is a nested BaseModel, recurse.
      - If field type is dict[str, Sensitive[V]], substitute every value.
      - If field type is list[Sensitive[V]], substitute every element.
      - Otherwise pass through verbatim.
    """
    ...
```

The recursion is bounded by Pydantic's existing model-resolution depth
limit; cycles in user-defined models would already fail Pydantic's own
validation.

#### 4.2.2 `ToolRedactionPolicy` — declarative legacy escape valve

For tools that cannot or have not yet adopted `Sensitive[T]` annotations
(e.g., raw-`dict` arguments, third-party Pydantic models the team does
not own), the rev-3 declarative policy survives as a legacy escape valve
with three strengthening changes:

1. **Structured `handles_no_sensitive_data_reason_struct`** replaces the
   free-text string. Adequacy-guard checks the structure mechanically
   rather than pattern-matching reason text against a stop-list.
2. **`known_response_keys` allowlist** is REQUIRED for any tool with
   `handles_no_sensitive_data=False`. At persistence time, every key in
   the actual tool response is checked against the allowlist; unknown
   keys are fail-closed redacted (substituted with
   `<redacted-unknown-key:{n}-bytes>`) and a counter increments. Closes
   security I-1 (response-shape drift).
3. **`EXEMPT_FROM_TYPE_DRIVEN_REDACTION: ClassVar[bool] = False`** on the
   tool class, defaulting False, must be explicitly set True for the
   legacy policy to apply. CODEOWNERS routes any commit that flips this
   to True through security review.

```python
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
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
        any prior tool's why_arguments_safe (mass-copy detection).

    why_responses_safe: same as above, for responses.

    last_reviewed_iso: ISO-8601 date when this justification was last
        reviewed. Adequacy-guard fails if the date is more than 365 days
        old at test time. The reviewer is responsible for incrementing
        this on every quarterly redaction-policy audit.
    """

    sensitive_data_locations: tuple[str, ...]
    why_arguments_safe: str
    why_responses_safe: str
    last_reviewed_iso: date

    def __post_init__(self) -> None:
        if not self.sensitive_data_locations:
            raise ValueError(
                "sensitive_data_locations is empty; declare at least one location "
                "where sensitive material related to this tool exists, OR migrate "
                "the tool's arguments/responses to use Sensitive[T] annotations."
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
    """Legacy declarative redaction policy. Use Sensitive[T] annotations
    on Pydantic field types instead when possible — the type-driven
    primitive is the preferred mechanism (§4.2.1) and is enforced
    structurally rather than declaratively.

    This policy is consulted only for tools that set
    EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True on their class. CODEOWNERS
    routes such PRs through security review.

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
        redacted and the counter
        composer.redaction.unknown_response_key_total increments. Closes
        security I-1 (response-shape drift defeating static policy).

    argument_summarizers: optional per-key replacement functions for keys
        in sensitive_argument_keys.

    handles_no_sensitive_data: explicit "this tool reviewed and asserts
        no sensitive material in arguments or responses" flag.

    handles_no_sensitive_data_reason_struct: structured justification
        REQUIRED when handles_no_sensitive_data=True. See above.

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
                "time are fail-closed redacted."
            )

        freeze_fields(
            self,
            "sensitive_argument_keys",
            "sensitive_response_keys",
            "known_response_keys",
            "argument_summarizers",
        )
```

### 4.3 Sentinel rules

- Plain sensitive key → value replaced by literal string `"<redacted>"`.
- Key with summarizer → value replaced by `summarizer(original_value)`.
  Example: `lambda b: f"<inline-blob:{len(b)}-bytes>"`.
- Existing `redact_source_storage_path` continues to handle source paths
  in `partial_state`. Unchanged by this work.

### 4.4 Adequate-redaction guard

The adequacy guard is the CI-time enforcement that every tool's arguments and
responses are either (a) annotated for type-driven redaction via
`Sensitive[T]` (§4.2.1), or (b) covered by a `ToolRedactionPolicy` legacy
escape valve with a structured justification (§4.2.2). Revision 4 expands
the rev-3 guard to address security T-4 (nested-type traversal) and T-3
(programmatic detection of policy weakening over time).

#### 4.4.1 Recursive schema introspection

Composer tools declare arguments via Pydantic `BaseModel` subclasses (see
existing tools in `src/elspeth/web/composer/tools.py`). The adequacy
test walks each tool's argument model and response model recursively.
For each field encountered:

| Field shape | Adequacy rule |
|---|---|
| `Annotated[T, Sensitive(...)]` (any `T`) | Pass. Type-driven redaction handles it. |
| `BaseModel` subclass | Recurse into the nested model's fields. |
| `dict[str, T]` where `T` is non-scalar | Treat as **opaque container**: fail-closed unless explicitly listed in `sensitive_argument_keys` / `sensitive_response_keys` OR the tool sets `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` and the policy declares the dict via `known_response_keys`. The dict could contain anything (e.g., `headers={"Authorization": "Bearer …"}`). |
| `list[T]` where `T` is `BaseModel` or non-scalar | Same as `dict`: fail-closed unless explicitly declared. |
| `Any`, `object`, untyped fields | **Fail the test.** `Any`-typed fields are inspection-resistant; tools must narrow the type or annotate with `Sensitive[T]`. |
| Discriminated union / `Union[A, B]` | Recurse into every arm. If any arm is `BaseModel`, walk it; if any arm is `str`/`bytes`/`Any`, apply the corresponding rule. |
| `str`, `bytes`, `Annotated[str, ...]` without `Sensitive` | Apply the rev-3 rule: EITHER listed in `sensitive_argument_keys` OR covered by `handles_no_sensitive_data=True` with a structured reason. |
| `int`, `float`, `bool`, `datetime`, enum members | Pass — scalars not covered by the redaction policy. |

The recursion is bounded by Pydantic's existing model-resolution depth
limit (no manual cycle detection needed).

#### 4.4.2 `EXEMPT_FROM_*` class attributes — `ClassVar`, not `getattr`

CLAUDE.md bans `getattr` for defensive attribute access (Python engineer
review Q8). The exempt flags are declared as `ClassVar[bool]` on the
shared composer tool base class, defaulting to `False`. The adequacy test
reads them by direct attribute access:

```python
class ComposerTool:
    """Shared base class for composer tools."""

    EXEMPT_FROM_ADEQUACY_CHECK: ClassVar[bool] = False
    EXEMPT_FROM_TYPE_DRIVEN_REDACTION: ClassVar[bool] = False
    # ... rest of base class ...

# Adequacy test:
for tool_class in registered_composer_tools():
    if tool_class.EXEMPT_FROM_ADEQUACY_CHECK:  # Direct read; no getattr.
        continue
    _check_tool_adequacy(tool_class)
```

`EXEMPT_FROM_ADEQUACY_CHECK=True` skips the tool entirely (e.g., a
no-argument utility tool with a void response). It must be paired with a
class-level docstring comment block explaining why exemption is
appropriate. CODEOWNERS rules require security review for any commit
that flips either flag.

#### 4.4.3 Policy-hash snapshot test (NEW in rev 4 — closes security T-3)

The adequacy guard also performs a **policy-hash snapshot test**. For
every tool with `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True`, the test
computes a deterministic SHA-256 hash of the tool's `ToolRedactionPolicy`
shape:

```python
def _policy_hash(policy: ToolRedactionPolicy) -> str:
    canon = json.dumps({
        "sensitive_argument_keys": sorted(policy.sensitive_argument_keys),
        "sensitive_response_keys": sorted(policy.sensitive_response_keys),
        "known_response_keys": sorted(policy.known_response_keys),
        "summarizer_keys": sorted(policy.argument_summarizers.keys()),
        "handles_no_sensitive_data": policy.handles_no_sensitive_data,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()
```

Hashes for every tool are committed to
`tests/unit/web/composer/redaction_policy_snapshot.json`. The CI test
fails when any hash differs from the snapshot. Policy changes therefore
require an explicit commit to the snapshot file, which CODEOWNERS routes
to security review. PR labels `policy-strengthen` or
`policy-weaken-justified` annotate the change category in the merge
log; weakening without the latter label fails CI. Closes security T-3
(programmatic detection rather than relying on quarterly human review).

#### 4.4.4 Quarterly review of structured reasons

`HandlesNoSensitiveDataReason.last_reviewed_iso` is checked at test time:
fail if older than 365 days. The quarterly review process (RC 5.1
security calendar) is now a structural requirement — CI fires red 365
days after the last review increment, forcing the review rather than
calendaring it informally. Closes the rev-3 RSK-02 erosion concern by
moving the control from process to mechanism.

#### 4.4.5 CODEOWNERS rules

The following file globs route changes to the security review team:

```
# .github/CODEOWNERS
src/elspeth/web/composer/redaction.py            @elspeth/security
src/elspeth/web/composer/tools.py                @elspeth/security  # for ToolRedactionPolicy declarations
tests/unit/web/composer/redaction_policy_snapshot.json    @elspeth/security
tests/unit/web/composer/test_redaction_policy.py @elspeth/security
```

This is the same control pattern used elsewhere in the project for
config contracts and tier-model allowlists.

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
    redact_tool_call(tc, lookup_tool_class(tc.function.name))
    for tc in assistant_message.tool_calls
)
redacted_tool_rows = tuple(
    _RedactedToolRow(
        tool_call_id=outcome.call.id,
        content=_serialize_response(outcome, lookup_tool_class(outcome.call.function.name)),
        composition_state_payload=(
            _StatePayload.from_composition_state(state, outcome.post_version)
            if outcome.post_version > outcome.pre_version
            else None
        ),
    )
    for outcome in tool_outcomes
)

# _run_sync handles ALL database writes. Cancellation of the outer task
# while this is in flight has no effect — the sync worker runs to
# completion. If the outer task is cancelled before this line is reached,
# no DB work has been done; no invariant is violated.
audit_outcome = await self._run_sync(
    self.sessions_service.persist_compose_turn,
    session_id=session_id,
    assistant_content=assistant_message.content or "",
    redacted_assistant_tool_calls=redacted_assistant_tool_calls,
    redacted_tool_rows=redacted_tool_rows,
    parent_composition_state_id=current_state_id,
    writer_principal="compose_loop",
    plugin_crash_pending=plugin_crash is not None,
)

# Step 3 — dispatch by audit outcome and any pending plugin crash.
if audit_outcome.tier1_violation:
    # Tier-1 audit invariant violation: tool succeeded but audit write
    # failed. Per CLAUDE.md primacy, raise unconditionally. The route
    # layer cannot recover from this; surface as 500 with no partial_state
    # claim (the partial state isn't durable).
    raise audit_outcome.tier1_violation_exc

if plugin_crash is not None:
    # The audit row for the crashing tool was written successfully (or the
    # audit failure was logged under primacy). Now re-raise the captured
    # ComposerPluginCrashError so _handle_plugin_crash receives it.
    raise plugin_crash

# (loop continues to next turn)
```

Where `_ToolOutcome`, `_RedactedToolRow`, `_StatePayload`, and
`_AuditOutcome` are dataclasses defined in `web/composer/service.py`
or a sibling `web/composer/_persist_payload.py` module (implementer's
choice; either location is L3 and adheres to layer rules).

#### 5.2.2 The sync write function — `SessionsService.persist_compose_turn`

```python
class SessionsService:
    def persist_compose_turn(
        self,
        *,
        session_id: str,
        assistant_content: str,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[_RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        writer_principal: str,  # "compose_loop"
        plugin_crash_pending: bool,
    ) -> _AuditOutcome:
        """Synchronous, single-transaction persistence of one compose turn.
        Called from a _run_sync worker; MUST NOT be invoked from async land.

        Atomicity contract:
        - Either every row (assistant + N tool + N state) commits, or none does.
        - INV-AUDIT-AHEAD bidirectional is delivered structurally: state rows
          and tool rows share a transaction.

        Audit-failure primacy:
        - Tool succeeded (plugin_crash_pending=False) + audit raised
          non-Integrity error => populate _AuditOutcome.tier1_violation_exc;
          caller raises. Counter:
          composer.audit.tool_row_tier1_violation_total += 1.
        - Tool failed (plugin_crash_pending=True) + audit raised
          non-Integrity error => log permitted, counter
          composer.audit.tool_row_persist_failed_during_unwind_total += 1,
          return _AuditOutcome.unwind_audit_failed=True. Caller proceeds
          to raise the ComposerPluginCrashError.
        - IntegrityError of any class => counter
          composer.audit.tool_row_integrity_violation_total += 1, raise
          (no recovery; CLAUDE.md offensive-programming).
        - OperationalError on COMMIT after successful INSERTs => same as
          non-Integrity audit failure; primacy disposition applies.
        """
        with self._engine.begin() as conn:
            # Acquire session-scoped advisory lock (PostgreSQL only; SQLite
            # has a global write lock).
            self._acquire_session_advisory_lock(conn, session_id)

            try:
                # Reserve sequence numbers for the entire batch.
                base_seq = self._reserve_sequence_range(
                    conn, session_id,
                    count=1 + len(redacted_tool_rows),  # 1 assistant + N tool
                )

                # 1) Insert assistant row.
                assistant_id = self._insert_chat_message(
                    conn,
                    session_id=session_id,
                    role="assistant",
                    content=assistant_content,
                    tool_calls=redacted_assistant_tool_calls,
                    sequence_no=base_seq,
                    writer_principal=writer_principal,
                    composition_state_id=parent_composition_state_id,
                    tool_call_id=None,
                    parent_assistant_id=None,
                )

                # 2) Insert each tool row + corresponding composition_states row.
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
                        tool_calls=None,
                        sequence_no=base_seq + offset,
                        writer_principal=writer_principal,
                        composition_state_id=state_id,
                        tool_call_id=tool_row.tool_call_id,
                        parent_assistant_id=assistant_id,
                    )

                # Transaction commits on context exit.
                return _AuditOutcome(
                    assistant_id=assistant_id,
                    tier1_violation=False,
                    tier1_violation_exc=None,
                    unwind_audit_failed=False,
                )

            except IntegrityError:
                # Tier-1 invariant violation. Counter + raise; transaction
                # rolls back automatically on context exit.
                self._telemetry.tool_row_integrity_violation_total.add(1)
                raise

            except OperationalError as audit_exc:
                # Connection drop / pool exhaustion / disk full / fsync
                # failure. Apply audit-failure primacy.
                if plugin_crash_pending:
                    self._telemetry.tool_row_persist_failed_during_unwind_total.add(1)
                    self._log.warning(
                        # Permitted under CLAUDE.md primacy: audit-system failure.
                        "audit_insert_failed_during_tool_failure_unwind",
                        session_id=session_id,
                        audit_exc_class=type(audit_exc).__name__,
                    )
                    return _AuditOutcome(
                        assistant_id=None,
                        tier1_violation=False,
                        tier1_violation_exc=None,
                        unwind_audit_failed=True,
                    )
                # Tool succeeded but audit failed → Tier-1 violation.
                self._telemetry.tool_row_tier1_violation_total.add(1)
                return _AuditOutcome(
                    assistant_id=None,
                    tier1_violation=True,
                    tier1_violation_exc=audit_exc,
                    unwind_audit_failed=False,
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
| 9 | DB INSERT succeeded, COMMIT fails (`OperationalError`) — disk full, fsync failure, connection dropped between INSERT and COMMIT, no plugin crash in flight | Counter `composer.audit.tool_row_tier1_violation_total` increments; sync worker returns `_AuditOutcome.tier1_violation=True`; async caller raises | 500 response; `partial_state_save_failed=true` carries through; Tier-1 audit invariant violation telemetry alerts. |
| 10 | DB INSERT succeeded, COMMIT fails (`OperationalError`) — plugin crash in flight | Counter `composer.audit.tool_row_persist_failed_during_unwind_total` increments; sync worker logs (permitted under primacy) and returns `_AuditOutcome.unwind_audit_failed=True`; async caller raises the captured `ComposerPluginCrashError` | Plugin crash path runs as in row 4; the audit-failure-during-unwind is visible via telemetry counter only. |
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
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[_RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        writer_principal: str,           # one of the four CHECK-permitted values
        plugin_crash_pending: bool,
    ) -> _AuditOutcome:
        """Synchronous, single-transaction persistence of one compose turn.
        Implementation in §5.2.2. Called from a _run_sync worker; do NOT
        invoke directly from async land (use the async route-helper paths
        for non-compose writers).
        """
        ...

    # ── Helpers private to persist_compose_turn ─────────────────────

    def _acquire_session_advisory_lock(self, conn: Connection, session_id: str) -> None:
        """Acquire a session-scoped advisory lock for the duration of the
        current transaction. Released automatically on COMMIT or ROLLBACK.

        SQLite: no-op. SQLite's global write lock already serialises writers
        per database file, which is sufficient because we only have one
        SQLite database per deployment.

        PostgreSQL: pg_advisory_xact_lock(hashtextextended(session_id::text, 0)).
        Note: hashtextextended (NOT hashtext) — hashtext returns int4 (32-bit)
        and would experience birthday collisions at ~65k distinct sessions per
        Python engineer Q4. hashtextextended returns int8, eliminating the
        collision risk at any foreseeable scale. The seed argument 0 is
        canonical; do not vary it across deployments or the lock space
        becomes inconsistent.
        """

    def _reserve_sequence_range(
        self, conn: Connection, session_id: str, *, count: int
    ) -> int:
        """Reserve `count` consecutive sequence numbers for `session_id`.
        Inside the same transaction, performs:
            SELECT COALESCE(MAX(sequence_no), 0) FROM chat_messages WHERE session_id = ?
        and returns max+1; the caller writes rows at max+1, max+2, ... max+count.

        The advisory lock acquired in _acquire_session_advisory_lock prevents
        cross-session collisions on PostgreSQL. The transaction's serialisable
        guarantees on SQLite handle the SQLite case.

        Note: gaps in sequence_no are permitted (transaction rollback after
        reservation leaves the next caller's MAX+1 higher than the first
        successful row's sequence_no). Sequence_no is an ordering key, not
        a count.
        """

    def _insert_chat_message(self, conn: Connection, /, **fields) -> str:
        """Single-row insert into chat_messages with the supplied fields.
        Caller must already hold the advisory lock and have reserved the
        sequence_no.
        """

    def _insert_composition_state(
        self, conn: Connection, *,
        session_id: str,
        payload: _StatePayload,
        provenance: str,                 # one of the five CHECK-permitted values
    ) -> str:
        """Single-row insert into composition_states with the supplied
        provenance discriminator. Existing rev-3 inline inserts at
        src/elspeth/web/sessions/service.py:395-418 and :828-850 are
        refactored to call this helper rather than emitting raw INSERTs.
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

The `_telemetry` module-level singleton lives in
`src/elspeth/web/composer/telemetry.py` (new), exposing the named OTel
counters introduced in §1.4:

- `composer.audit.tool_row_tier1_violation_total`
- `composer.audit.state_rolled_back_during_persist_total`
- `composer.audit.tool_row_persist_failed_during_unwind_total`
- `composer.audit.tool_row_integrity_violation_total`
- `composer.redaction.summarizer_errors_total`
- `composer.redaction.unknown_response_key_total`

Tests inject a fake counter via a constructor parameter on the composer
service for assertable behaviour. The injected fake counter is itself
asserted against by the property-test post-conditions (§8.3.2 closes
QA F-4).

#### 5.7.5 `lookup_tool_class` and redaction-layer integration

The `lookup_tool_class(tool_name: str) -> type[ComposerTool]` helper
returns the registered tool class, which in turn carries the
`Sensitive[T]`-annotated argument and response Pydantic models AND any
legacy `ToolRedactionPolicy` declared via the
`EXEMPT_FROM_TYPE_DRIVEN_REDACTION` escape valve. The redaction layer
prefers type-driven redaction; the legacy policy is consulted only when
the exempt flag is `True`.

The lookup raises `MissingToolError` (a new exception) on unregistered
names. The adequacy guard (§4.4) ensures this never fires in practice —
every registered tool either has annotated models or has the exempt
flag + legacy policy, enforced at registry build time. The crash on
missing tool is offensive programming: the case is impossible by
construction; if it ever fires, the registry itself is corrupt and the
audit trail must not proceed.

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
      Column("session_id", String, ForeignKey("sessions.id"), nullable=False),
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

- `tests/unit/web/composer/test_redaction_policy.py`
  - **Registry-presence test.** Iterate composer tool registry; assert
    every tool either uses `Sensitive[T]` annotations on its argument and
    response models OR has `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` AND
    a non-default `ToolRedactionPolicy`.
  - **Adequacy guard test (rev 4).** For every non-exempt tool, walk
    its Pydantic argument and response models recursively per §4.4.1
    table; assert no `Any`-typed fields, no opaque `dict`/`list`
    containers without explicit declaration, and every `str`/`bytes`
    field either annotated with `Sensitive[T]` or covered by the
    legacy policy's `sensitive_argument_keys` /
    `sensitive_response_keys`.
  - **Recursive descent test.** Synthetic tool with a 3-level-nested
    Pydantic model containing `Sensitive[T]` at the deepest level;
    assert the redaction layer reaches it.
  - **`EXEMPT_FROM_*` ClassVar test.** Assert direct attribute access
    works on the base class (no `getattr` defensive read); assert
    flipping the flag requires CODEOWNERS approval (verified by
    inspecting `.github/CODEOWNERS`).
  - **Structured reason validators.**
    - Orphan summarizer key → `ValueError`.
    - `handles_no_sensitive_data=True` with `None` reason struct → `ValueError`.
    - `handles_no_sensitive_data=False` with non-None reason struct → `ValueError`.
    - `HandlesNoSensitiveDataReason` empty `sensitive_data_locations` → `ValueError`.
    - `HandlesNoSensitiveDataReason` `why_arguments_safe` < 32 chars → `ValueError`.
    - `last_reviewed_iso` > 365 days old → adequacy-guard test fails red.
    - `known_response_keys` empty when `handles_no_sensitive_data=False` → `ValueError`.
  - **Policy-hash snapshot test.** Compute hash for every legacy
    policy; compare against `redaction_policy_snapshot.json`; fail on
    mismatch. Verifies T-3 mitigation.
  - **Per-policy round-trip.** Synthetic tool call with sentinel-marked
    arguments; redaction layer produces expected output; non-listed
    keys are byte-identical to input (defends against a buggy policy
    that redacts everything).
  - **Unknown response key fail-closed test (NEW).** Tool returns a
    response with a key not in `known_response_keys`; assert the
    unknown key's value is replaced with the
    `<redacted-unknown-key:...>` sentinel; counter increments.
  - **Summarizer raises.** Inject a summarizer that raises; assert the
    fallback sentinel `<redacted-summarizer-error:{exc_type}>` is
    written (exception class only — no message echo per security I-3);
    OTel counter `composer.redaction.summarizer_errors_total`
    increments; redaction never raises through the persistence boundary.

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
  - Assert `_AuditOutcome.tier1_violation=True` when no plugin crash
    pending and `OperationalError` is injected on COMMIT.
  - Assert `_AuditOutcome.unwind_audit_failed=True` when plugin crash
    pending and `OperationalError` is injected on COMMIT.

- `tests/unit/web/composer/test_audit_failure_primacy.py`
  - Tool succeeds + audit fails (non-IntegrityError on COMMIT) →
    `_AuditOutcome.tier1_violation_exc` populated; counter increments;
    caller raises.
  - Tool fails + audit fails (non-IntegrityError on COMMIT) →
    `_AuditOutcome.unwind_audit_failed=True`; counter increments
    (different counter); log permitted; caller raises the captured
    `ComposerPluginCrashError`.
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
  - **CL-PP-10a:** No plugin crash in flight. Assert `_AuditOutcome.tier1_violation=True`,
    `composer.audit.tool_row_tier1_violation_total` increments,
    caller raises, no rows visible in chat_messages (transaction rolled back).
  - **CL-PP-10b:** Plugin crash in flight. Assert
    `_AuditOutcome.unwind_audit_failed=True`,
    `composer.audit.tool_row_persist_failed_during_unwind_total` increments,
    log entry emitted (permitted under primacy), caller raises the
    captured `ComposerPluginCrashError`.

- **CL-PP-11: Concurrent multi-session writes (NEW in rev 4 — QA F-2).**
  Two compose loops on session A and session B, sharing one PostgreSQL
  connection pool. Force `hashtextextended` to produce colliding hashes
  by selecting session_ids whose hashes are known to collide (seeded
  test fixture). Assert: no deadlock; sequence_no values are independently
  monotonic per session; both transactions commit; the advisory-lock
  collision serialises but does not error.

- **CL-PP-12: Tool-call cap exceeded (NEW in rev 4 — RSK-13).** Drive
  the LLM to emit 17 tool calls in one assistant turn (cap is 16).
  Assert the loop raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")`
  BEFORE any tool execution; no DB writes for the over-cap turn; route
  helper emits the new reason code; counter
  `composer.tool_call_cap_exceeded_total` increments.

- **CL-PP-13: Unknown response key fail-closed (NEW in rev 4 — security I-1).**
  A legacy-escape-valve tool (with `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True`
  and explicit `known_response_keys`) returns a response containing a key
  not in the allowlist. Assert the unknown key's value is replaced with
  `<redacted-unknown-key:{n}-bytes>` in the persisted content;
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

### 8.6 Test path integrity — explicit composer rule

CLAUDE.md's "never bypass production code paths in tests" rule is about
not bypassing the production engine path (`ExecutionGraph.from_plugin_instances`,
`instantiate_plugins_from_config`); the composer is a different surface.
Revision 4 spells out the equivalent rule for composer tests:

**Composer integration tests MUST instantiate the full route → service →
SessionsService stack against either:**

- An in-memory SQLite database (`sqlalchemy.create_engine("sqlite:///:memory:")`
  + `metadata.create_all()`), suitable for unit-level integration that
  exercises check constraints, partial unique indexes, and CASCADE
  semantics; OR
- A testcontainer PostgreSQL database, for tests that need
  `pg_advisory_xact_lock`, `hashtextextended`, partial-unique-index
  dialect-specific behaviour, or multi-session concurrency (CL-PP-11).

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
| RSK-02 | Legacy `ToolRedactionPolicy` declarations with `handles_no_sensitive_data=True` normalize over time (Shifting the Burden). | Low (rev 4 — moved from Medium) | Medium (erosion of audit safety) | Quarterly review (mechanically enforced via `last_reviewed_iso < today - 365d` test failure) finds new declarations or stale reviews. | (Rev-4 strengthening:) Type-driven `Sensitive[T]` is the preferred primitive — declarations are no longer the dominant defence. The legacy escape valve requires `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` ClassVar AND a structured `HandlesNoSensitiveDataReason` AND CODEOWNERS routing to security review. Adequacy guard tests `last_reviewed_iso` mechanically; the calendar review is structural, not informal. | Security review at RC 5.1; PR reviewers per CODEOWNERS |
| RSK-03 | Redaction summarizer raises on pathological input. | Low | Medium | OTel counter `composer.redaction.summarizer_errors_total` exceeds 0.1% of tool calls in 24h. | Persistence wrapper catches summarizer exceptions and falls back to `<redacted-summarizer-error:{exc_type}>`. Property test asserts redaction never raises. The fallback sentinel uses ONLY the exception class name; raw exception messages are NOT included (closes security I-3 — message echo would risk leaking values). | RC 5.1 production hardening |
| RSK-04 | Per-turn sync transaction slows down the loop. | Low | Low (latency NFR §1.4) | CI sanity bound (p95 ≤ 250 ms with N ≤ 8) red. | CI sanity bound + nightly tight bench; bounded write per turn; SQLite/PostgreSQL handle small transactions well; per-turn (not per-row) atomicity reduces transaction count. | Implementing engineer |
| RSK-05 | Frontend diff blows up on very large `partial_state`. | Low | Low (UX nuisance) | Recovery panel TTI exceeds 500 ms in CI sanity test. | Diff helper iterates fields rather than diffing entire JSON; UI shows "large diff — expand" disclosure for thousands of nodes. | Frontend follow-up |
| RSK-06 | New `tool_call_id` index slows large message-table writes. | Very low | Low | Insert latency regression in benchmarks. | Composite single-column index; SQLite/PostgreSQL handle this trivially. | n/a |
| RSK-07 | Audit-ahead-of-state invariant violated during cancellation. | Very low (rev 4 — was Low; mechanism now structural) | Critical (auditability standard breach) | Property test failure; or schema-level post-condition assertion fires. | Single-sync-block design (§5.2.1) eliminates cross-await coordination; cancellation cannot tear a transaction; backward direction provable from `composition_states.provenance` discriminator. CL-PP-3, CL-PP-7, expanded `st_cancellation_arrival_time` cover the cancel paths. | Implementing engineer; verified by property test |
| RSK-08 | `chat_messages` and `audit_access_log` tables grow unboundedly without retention. | Medium (over time) | Low (pre-release) | Table size exceeds 1 GB in dev/staging. | Cascade-delete with sessions today; retention extension filed under `[elspeth-RETENTION-WEB]` (§10 OQ-1). | RC 5.1 production hardening |
| RSK-09 | Partial unique index syntax differs across DB dialects. | Low | Low | DDL fails on a target dialect. | Use SQL `CREATE UNIQUE INDEX ... WHERE ...` (SQLite 3.8.0+; PostgreSQL); SQLAlchemy DDL emit hook for both dialects. | Implementing engineer |
| RSK-10 | State-rollback race during atomic-pair commit. | **Negligible (rev 4 — structurally impossible)** | n/a | Counter retained for production observability only; never expected to fire. | Single-sync-block design means there is no atomic-pair-across-await window. The counter `composer.audit.state_rolled_back_during_persist_total` exists but the §5.2 code path cannot increment it. Kept in the schema so a future re-introduction of multi-transaction grain (per-call atomicity) has the observability point ready. | n/a |
| RSK-11 | Audit-write failure when no tool exception in flight (Tier-1 violation). | Very low | Critical | OTel counter `composer.audit.tool_row_tier1_violation_total` non-zero. | §5.2.2 sync function returns `_AuditOutcome.tier1_violation_exc`; caller raises unconditionally; CL-PP-10a asserts. | RC 5.1 SRE |
| RSK-12 | LLM provider re-uses `tool_call_id` across turns within a session. | Medium (provider-dependent) | High (silent mis-correlation absent the partial-unique index) | Partial unique index rejects insert; CL-PP-8 fires; counter `composer.audit.tool_row_integrity_violation_total` increments. | Crash on duplicate; do not silently recover. Per-provider observation: OpenRouter/OpenAI ids are message-scoped per current spec but not contractually forever. | Implementing engineer |
| RSK-13 (NEW rev 4) | LLM-driven tool-call amplification via prompt injection. | Medium (LLM-dependent) | High (storage growth, Tier-3 input attack) | Per-turn cap of 16 tool calls fires; `composer.tool_call_cap_exceeded_total` increments. | Hard cap per assistant turn (configurable); CL-PP-12 covers; `_handle_convergence_error` returns the new reason code. | Security review at RC 5.1 |
| RSK-14 (NEW rev 4) | Redaction-policy weakening lands without security review (T-3). | Low | High (silent audit-safety regression) | Policy-hash snapshot test fails on PR; no `policy-weaken-justified` PR label. | Snapshot test (§4.4.3); CODEOWNERS routes redaction.py + tools.py + snapshot file changes to security team. | Security review per PR |
| RSK-15 (NEW rev 4) | Tool response-shape drift defeats static `sensitive_response_keys` (security I-1). | Medium (over time) | High (silent leakage of new keys) | `composer.redaction.unknown_response_key_total` non-zero. | `known_response_keys` allowlist with fail-closed redaction (§4.2.2); CL-PP-13 covers; counter alerts on first occurrence in production. | Implementing engineer; tool authors when extending response shapes |
| RSK-16 (NEW rev 4) | Actor attribution missing on rows from a future writer (security R-1). | Low | Medium (audit-trail gap for unattributable rows) | A new writer attempts to insert without a registered `writer_principal` value; CHECK constraint refuses. | `writer_principal` CHECK constraint (§4.1.1); new writers require schema migration that adds the value to the CHECK enum AND coordinates with the advisory lock. | Architecture review for any new writer |
| RSK-17 (NEW rev 4) | Audit-grade transcript view exposed without access logging (security I-5). | Low | Medium (forensic gap on access patterns) | `audit_access_log` row missing for an `include_tool_rows=true` request. | Route helper writes the access-log row before returning the response; integration test asserts the row exists for every audit-grade query. | Security review at RC 5.1 |

---

## 10. Open Questions

- **OQ-1.** Filigree ticket ID for the `chat_messages` and
  `audit_access_log` retention CLI extension (referenced as
  `[elspeth-RETENTION-WEB]` in §4.6, §6.3, RSK-08). File during
  implementation; cite in the implementation PR description.
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

- **OQ-4 (NEW rev 4).** Pre-deploy migration step for staging
  (`elspeth.foundryside.dev`). The §3 Migration ADR row authorises a
  destructive `DELETE FROM chat_messages` before applying the new
  schema (because the rev-3-era schema has been generating rows that
  cannot satisfy the new NOT NULL columns without a default). Confirm
  the operator runbook for staging deployment captures this step BEFORE
  the spec is implemented; surface in the PR description and update the
  staging-deploy runbook in the same PR.

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

> **Phase 1 implementation plan is authoritative — supersession notice
> (added 2026-05-08).** The Phase 1 implementation plan
> [`docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md`](../plans/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md)
> is the governing handoff for Phase 1 code work. Where this spec and
> the plan disagree on Phase 1 mechanics, the plan wins until Task 19
> rewrites this section in place. The plan supersedes the following
> stale snippets that earlier spec drafts asserted:
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
> 7. `_AuditOutcome` has only `assistant_id` and `unwind_audit_failed`;
>    Tier-1 audit-write failures raise. Do not copy earlier
>    `_AuditOutcome` shapes that included additional fields or
>    swallowed Tier-1 failures.
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
`composition_states.provenance`, `audit_access_log` table); add the
sync persistence primitive on `SessionsService`; update existing
`add_message` callers (`routes.py:1487`, `:1883`) to pass
`writer_principal`; expand `_acquire_session_advisory_lock` to use
`hashtextextended`; refactor existing inline state-row inserts at
`web/sessions/service.py:395-418` and `:828-850` to call
`_insert_composition_state`.

**Done when.** §8.1 unit tests pass against in-memory SQLite (real
database, not metadata-only); CL-PP-11 passes against testcontainer
PostgreSQL; `enforce_tier_model.py` and `enforce_freeze_guards.py`
green; staging runbook updated for the pre-deploy DELETE step.

**Out of scope for Phase 1.** No compose-loop changes; no redaction
framework; no frontend.

### Phase 2 — Redaction framework: `Sensitive[T]`, `ToolRedactionPolicy`, adequacy guard

**Scope.** Add `Sensitive[T]` annotation primitive (§4.2.1); update
existing tool argument/response models in `web/composer/tools.py` to
use `Sensitive[T]` where applicable; add `ToolRedactionPolicy` legacy
escape valve with structured `HandlesNoSensitiveDataReason`
(§4.2.2); implement recursive adequacy guard (§4.4); add policy-hash
snapshot test (§4.4.3); add CODEOWNERS rules; declare
`EXEMPT_FROM_TYPE_DRIVEN_REDACTION` ClassVar on the composer-tool
base class.

**Done when.** §8.1 redaction-policy unit tests pass; every registered
composer tool either uses `Sensitive[T]` or has a legacy policy with
structured justification; quarterly-review structural test passes
(`last_reviewed_iso` is recent); snapshot file committed.

**Out of scope for Phase 2.** No compose-loop persistence; no frontend.

### Phase 3 — Compose-loop persistence + tool-call cap

**Scope.** Modify `_compose_loop` (§5.2.1) to (a) enforce the per-turn
tool-call cap; (b) accumulate tool outcomes; (c) dispatch
`persist_compose_turn` via `_run_sync`; (d) raise
`ComposerPluginCrashError` after audit write completes; (e) handle
`_AuditOutcome.tier1_violation` and `unwind_audit_failed` correctly.
Add the route-helper `failed_turn` field to 422/500 response bodies.
Extend `GET /api/sessions/{sid}/messages` with `include_tool_rows`
parameter and the audit-grade access-log emission (§6.3).

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
- Source report — `notes/composer-llm-eval-2026-04-28.md`.
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
- **`Sensitive[T]`** (rev 4). Type-driven redaction primitive. Used as
  Pydantic field metadata via `Annotated[T, Sensitive(summarizer=...)]`.
  The persistence layer reads field annotations and unconditionally
  redacts marked fields. The structural Level-4 (Meadows hierarchy)
  intervention that makes redaction mechanical rather than declarative.
- **`ToolRedactionPolicy`** (legacy escape valve from rev 4 onward).
  Per-tool declarative policy retained for tools that cannot use
  `Sensitive[T]` (raw-dict arguments, third-party Pydantic models).
  Requires `EXEMPT_FROM_TYPE_DRIVEN_REDACTION=True` ClassVar on the
  tool class plus security CODEOWNERS approval.
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
- **Adequacy guard** (strengthened in rev 4). The CI-time test that
  every tool's argument and response models are either annotated with
  `Sensitive[T]` for any sensitive field OR covered by a legacy
  `ToolRedactionPolicy`. The rev-4 guard recurses into nested Pydantic
  submodels and treats `dict`, `list`, `Any`, `object`-typed fields as
  fail-closed unless explicitly declared.
- **Policy-hash snapshot** (rev 4). A SHA-256 hash of every legacy
  `ToolRedactionPolicy`, committed to a JSON file under
  `tests/unit/web/composer/redaction_policy_snapshot.json`. Changes
  require an explicit commit to the snapshot, routed to security review
  via CODEOWNERS. Mechanical mitigation for policy weakening.
- **Tool-call cap** (rev 4). Per-turn hard limit on the number of tool
  calls an assistant turn may emit. Default 16. Exceeding the cap
  raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")`
  before any tool execution. Defends against prompt-injection-induced
  amplification.
- **`_AuditOutcome`** (rev 4). The dataclass returned by
  `SessionsService.persist_compose_turn`. Carries the audit-failure
  primacy disposition: `tier1_violation` (caller must raise),
  `unwind_audit_failed` (caller proceeds to raise the captured plugin
  crash), or success (no flags set, `assistant_id` populated).
