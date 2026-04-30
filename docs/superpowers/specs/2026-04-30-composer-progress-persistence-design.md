# Composer Progress Persistence — Design

**Ticket:** [elspeth-90b4542b63](filigree:elspeth-90b4542b63) — Composer progress persistence — tool-call breadcrumbs and partial drafts survive long-running failures
**Parent epic:** [elspeth-528bde62bb](filigree:elspeth-528bde62bb) — Composer LLM evaluation remediation
**Related future epic:** [elspeth-f0460a6594](filigree:elspeth-f0460a6594) — Composer async/background execution model (deferred to Future release)
**Date:** 2026-04-30
**Status:** Proposed (revision 3 — incorporates panel review findings on revision 2)
**Branch:** RC5-UX (or successor)
**Tier-artifact match:** **S-tier** change captured in a single spec; the four mini-ADRs are inlined in §3, the infrastructure additions are itemised in §5.7. (Originally labelled XS in revision 1; relabelled S after revision 2 grew to ~950 lines with 9 risk entries, 11 integration scenarios, 7 named property-test strategies, and explicit infrastructure additions.)

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
| Audit-state-rollback counter | **0 events expected**; any non-zero value in a 24-hour window alerts | OTel counter `composer.audit.state_rolled_back_during_persist_total`; SLO threshold = 0 |
| Audit-write failure rate (tool-row insert) | **0 events expected**; any non-zero value alerts | OTel counter `composer.audit.tool_row_persist_failed_total`; SLO threshold = 0 |
| Replay fidelity | Byte-identical `chat_messages.content` + `tool_call_id` + `sequence_no` ordering after redaction-sentinel substitution | Property test (§8.3) |
| Audit-ahead-of-state invariant violation rate | **0 events**; single violation = audit incident, not an SLO budget | Property test (§8.3) + post-condition assertion in `_compose_loop` |

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
| "Only final assistant text is persisted." | **Open.** The compose loop today appends each assistant turn to an in-memory `llm_messages` list (`web/composer/service.py:_compose_loop`, ~lines 673-691) but does **not** insert per-turn rows into `chat_messages`. Only the final message lands, via the route layer's call to `SessionsService.add_message(...)` (`web/sessions/service.py:283-320`). The `chat_messages.tool_calls` JSON column exists but is populated only on that final insert. |
| "No tool-result rows in chat history." | **Open.** The schema permits `role='tool'` (`web/sessions/models.py:81`) but no insert site exists. Tool *responses* are not persisted. |
| "Frontend has no recovery surface." | **Open.** The data is in DB and in failure response body, but no UX renders it as a "draft you can pick up." |
| "Tool argument redaction is unverified." | **Open.** No central redaction layer exists for `chat_messages.tool_calls` JSON content; the existing `redact_source_storage_path` helper is path-specific. |

**This design closes four open gaps**: per-turn assistant row persistence,
per-tool-call response row persistence, central redaction layer at the
write seam, and the frontend recovery surface.

The async / background-job direction is captured in the future epic and
does not constrain this work.

---

## 3. Approach Decisions (inlined mini-ADRs)

| Decision | Choice | Alternatives | Reversibility | Rollback | Cost driver | Review by |
|---|---|---|---|---|---|---|
| Tool-response persistence shape | **A1.** One `role='tool'` `chat_messages` row per tool response, correlated to its assistant turn via `tool_call_id`. | A2 (embed responses in assistant row's `tool_calls` JSON) | One-way after first compose run executes — tool rows accumulate. Pre-release the corpus is empty so revisable. | Drop the tool rows + `tool_call_id` column; revert to A2 shape. Costs one schema migration + audit-data-loss event. | Per-call audit row growth (1 + N rows per turn vs 1) | RC 5.1 readiness review |
| Frontend recovery UX | **F2.** Diff-and-confirm modal showing pipeline diff + tool transcript with explicit Apply / Discard buttons. | F1 (auto-apply) destructive of unsaved edits; F3 (reload-to-recover) clunky. | Reversible — frontend-only change, no DB shape implications. | Hide the panel; failure path falls back to existing toast/banner. | Frontend dev + accessibility cost | RC 5.1 frontend review |
| Tool argument redaction | **R3.** Each tool declares `ToolRedactionPolicy(sensitive_argument_keys, sensitive_response_keys, argument_summarizers, handles_no_sensitive_data, handles_no_sensitive_data_reason)`; persistence layer enforces. | R1 (no redaction) leaks by default; R2 (central policy) couples redaction to every tool's argument shape and rots. | One-way — once tool authors declare policies, removing the contract requires touching every tool. | Add a no-op `ToolRedactionPolicy()` to every tool and remove the registry-iteration test. Costs lost central enforcement. | Per-tool author declaration burden + adequacy-guard (§4.4) + reason-text discipline (§4.4) | RC 5.1 security review; quarterly review of `handles_no_sensitive_data=True` declarations |
| Migration | **None.** Pre-release per CLAUDE.md. New columns ship as part of the schema; no backfill. | Migration would have been required post-release. | One-way — first eval run produces rows in the new shape. | Delete the chat_messages corpus; pre-release acceptable. | No backward-compat shim cost | Once a real corpus exists |

---

## 4. Data Model

### 4.1 `chat_messages` schema change

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
    Column("tool_call_id", String, nullable=True),    # NEW
    Column("sequence_no", Integer, nullable=False),   # NEW — monotonic per session
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("composition_state_id", String, nullable=True),
    Column(                                            # NEW — explicit cascade
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
    CheckConstraint(                                                              # NEW
        "(role = 'tool') = (tool_call_id IS NOT NULL)",
        name="ck_chat_messages_tool_call_id_role",
    ),
    CheckConstraint(                                                              # NEW
        "(role = 'tool') = (parent_assistant_id IS NOT NULL)",
        name="ck_chat_messages_parent_role",
    ),
    Index(
        "ix_chat_messages_session_sequence",
        "session_id",
        "sequence_no",
        unique=True,
    ),                                                                             # NEW
    Index(
        "ix_chat_messages_session_tool_call_id",
        "session_id",
        "tool_call_id",
    ),                                                                             # NEW
)
```

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
- `(session_id, sequence_no)` is unique — every row in a session has a unique monotonic sequence number. Replay fidelity (NFR §1.4) depends on this.
- `(session_id, tool_call_id)` is unique among `role='tool'` rows. Cross-turn collisions (same `tool_call_id` reused by the LLM provider in a different turn) are rejected as a Tier-3 input-validation failure: the compose loop crashes the request rather than silently mis-correlating.

`composition_state_id` FK behaviour — see §4.5.

### 4.2 `ToolRedactionPolicy` — per-tool declared policy

**Location.** `src/elspeth/web/composer/redaction.py` (L3, alongside the
existing `redact_source_storage_path` helper). Tools are L3; their
redaction policies belong in the same layer.

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class ToolRedactionPolicy:
    """Redaction declaration owned by the tool, enforced at the persistence boundary.

    sensitive_argument_keys: keys in the tool-call argument dict whose values
        must be replaced before the tool call is persisted to
        chat_messages.tool_calls JSON.

    sensitive_response_keys: keys in the tool's response dict whose values
        must be replaced before the tool's response is persisted to
        chat_messages.content (as JSON).

    argument_summarizers: optional per-key replacement functions for keys in
        ``sensitive_argument_keys``. A summarizer key not present in
        ``sensitive_argument_keys`` is a configuration error (validator
        rejects at construction).

    handles_no_sensitive_data: explicit "this tool reviewed and asserts
        no sensitive material in arguments or responses" flag. Required to
        be True if both ``sensitive_argument_keys`` and
        ``sensitive_response_keys`` are empty AND the tool's argument
        schema contains string-typed properties.

    handles_no_sensitive_data_reason: free-text justification REQUIRED
        when ``handles_no_sensitive_data=True``. The string is reviewed
        during the quarterly redaction-policy audit (RSK-02 trigger).
        Empty / boilerplate reasons fail the adequacy-guard test.
        This is the friction point that prevents ``True`` from becoming
        a costless shortcut: the cost is the reviewer-readable
        justification, which is recorded forever.

    NOTE on freeze: ``argument_summarizers`` values are Callables. ``deep_freeze``
    passes Callables through unchanged (they are not mutable containers).
    Identity-equality of summarizer callables is the policy contract.
    """

    sensitive_argument_keys: tuple[str, ...] = ()
    sensitive_response_keys: tuple[str, ...] = ()
    argument_summarizers: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)
    handles_no_sensitive_data: bool = False
    handles_no_sensitive_data_reason: str | None = None

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

        if self.handles_no_sensitive_data and not (self.handles_no_sensitive_data_reason or "").strip():
            raise ValueError(
                "handles_no_sensitive_data=True requires a non-empty handles_no_sensitive_data_reason. "
                "The reason is reviewed during quarterly redaction-policy audits."
            )
        if not self.handles_no_sensitive_data and self.handles_no_sensitive_data_reason is not None:
            raise ValueError(
                "handles_no_sensitive_data_reason is only meaningful when handles_no_sensitive_data=True."
            )

        freeze_fields(self, "sensitive_argument_keys", "sensitive_response_keys", "argument_summarizers")
```

### 4.3 Sentinel rules

- Plain sensitive key → value replaced by literal string `"<redacted>"`.
- Key with summarizer → value replaced by `summarizer(original_value)`.
  Example: `lambda b: f"<inline-blob:{len(b)}-bytes>"`.
- Existing `redact_source_storage_path` continues to handle source paths
  in `partial_state`. Unchanged by this work.

### 4.4 Adequate-redaction guard

The naive registry test "every tool has a `ToolRedactionPolicy`" is not
sufficient — an empty policy (`()`, `()`, `{}`) trivially passes. The
adequacy guard:

**Schema introspection mechanism.** Composer tools declare arguments via
Pydantic `BaseModel` subclasses (see existing tools in
`src/elspeth/web/composer/tools.py`). The adequacy test introspects each
tool's argument model via `model.model_fields` and identifies fields
whose annotation type is `str`, `bytes`, or any `Annotated` of these.
Response schemas use the same mechanism on `ToolResponse` subclasses.

**Adequacy rule.** For every registered composer tool, the test asserts:

- If the tool's argument-model `model_fields` includes any string-typed
  property, then EITHER `sensitive_argument_keys` is non-empty OR
  `handles_no_sensitive_data=True` with a non-empty
  `handles_no_sensitive_data_reason`.
- Same rule for response schema and `sensitive_response_keys`.

**Fail-closed semantics.** If a tool has no detectable argument model
(e.g., raw dict accepted), the test FAILS rather than skipping the
tool. Every tool must have a Pydantic argument model OR an explicit
opt-out via a separate `EXEMPT_FROM_ADEQUACY_CHECK = True` class
attribute (also reviewed quarterly).

**Reason-text discipline.** The test rejects `handles_no_sensitive_data_reason`
values that match a stop-list of placeholder phrases (e.g., "n/a",
"none", "no sensitive data", "TODO"). A tool that flags
`handles_no_sensitive_data=True` must explain in concrete terms why the
arguments and responses are safe to persist verbatim. Stop-list lives
in `tests/unit/web/composer/test_redaction_policy.py` and is updated
during quarterly review.

### 4.5 `composition_state_id` FK behaviour and `IntegrityError` semantics

Under §5.2's atomic-pair transaction, the FK race the original §4.5
described (state row rolled back between tool execution and tool-row
insert) is **not reachable for the tool-row write** — the new
`composition_states` row is written inside the same transaction as the
tool row, or the tool row sets `composition_state_id=NULL`.

The remaining places `IntegrityError` can fire on a `chat_messages`
write:

| Source | Cause | Correct response |
|---|---|---|
| `fk_chat_messages_composition_state_session` on Step A (assistant row) | Externally-deleted `composition_states` row (admin tooling, race with cleanup). The assistant row references an *already-committed* state row that vanished between read and write. | Crash — this is a Tier-1 invariant violation. |
| `uq_chat_messages_tool_call_id` (partial unique) | LLM provider re-used a `tool_call_id` across turns. | Crash — Tier-3 input validation failure surfaced as a database integrity error. The compose request fails fast; the LLM is misbehaving and should not be allowed to proceed. |
| `ix_chat_messages_session_sequence` (unique) | Sequence-number reservation race. Should be unreachable per §5.7's single-writer-per-session rule. | Crash — concurrency bug. |
| `ck_chat_messages_*` (CHECK constraints) | Caller bug. | Crash — internal bug. |

**The persistence layer does NOT catch `IntegrityError`.** Any
`IntegrityError` is treated as a Tier-1 invariant violation and
propagates. CLAUDE.md offensive-programming policy: detect invalid
states and throw meaningful exceptions, do not silently recover.

(Earlier revisions of this spec proposed catching `IntegrityError` and
re-attempting with `composition_state_id=NULL`. The atomic-pair design
in §5.2 made that unreachable for the tool-row write; for the
assistant-row write the catch would silently swallow real bugs
including duplicate `tool_call_id` and CHECK violations. Removed in
revision 3.)

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

The loop in `web/composer/service.py:_compose_loop` becomes:

```python
# Step A — redact and persist the assistant turn
redacted_tool_calls = tuple(
    apply_redaction_policy(tc, lookup_redaction_policy(tc.function.name))
    for tc in assistant_message.tool_calls
)
assistant_msg_id = await sessions_service.add_message(
    session_id=session_id,
    role="assistant",
    content=assistant_message.content or "",
    tool_calls=redacted_tool_calls,
    composition_state_id=current_state_id,
    # sequence_no reserved atomically inside add_message — see §5.7
)

# Step B — execute and persist each tool turn
for tool_call in assistant_message.tool_calls:
    response_for_persistence: dict[str, Any] | ToolResponse
    pre_version = state.version
    tool_exception_in_flight: BaseException | None = None
    try:
        response = await execute_tool(tool_call, state)
        response_for_persistence = response
    except ToolArgumentError as exc:
        # Tier-3 boundary signal — the loop's existing contract continues.
        # See protocol.py:291 — ToolArgumentError is the ONLY exception class
        # the compose loop catches around execute_tool() and continues past.
        response_for_persistence = {
            "error": "ToolArgumentError",
            "message": str(exc),
        }
        # Do NOT re-raise; the loop continues to the next tool_call.
    except Exception as exc:
        response_for_persistence = {
            "error": type(exc).__name__,
            "message": str(exc),
        }
        tool_exception_in_flight = exc
        # Re-raise after recording. The audit insert in `finally` is shielded
        # from cancellation so an in-flight CancelledError cannot abandon
        # the row.
        raise
    finally:
        # response_for_persistence is guaranteed bound by this point: every
        # except arm assigns it, and the try body assigns it on success.
        # If a BaseException reaches this finally without going through any
        # except arm (e.g., asyncio.CancelledError mid-execution), the
        # `nonlocal` rebind doesn't happen and we must NOT silently fabricate
        # a row — let the BaseException propagate after the audit attempt.
        # This matches CLAUDE.md offensive-programming policy.
        try:
            redacted_response = apply_response_redaction(
                response_for_persistence,
                lookup_redaction_policy(tool_call.function.name),
            )
        except NameError:
            # response_for_persistence not bound — BaseException path.
            # Re-raise immediately; do not swallow.
            raise
        # Shielded so cancellation between tool execution and audit insert
        # cannot leave chat_messages behind composition_states. The helper
        # writes BOTH the tool row AND the new composition_states row (if
        # state advanced) in one atomic transaction — see §5.3 and §5.7.
        await asyncio.shield(
            _persist_tool_row_with_audit_failure_handling(
                sessions_service=sessions_service,
                session_id=session_id,
                tool_call_id=tool_call.id,
                parent_assistant_id=assistant_msg_id,
                content=json.dumps(redacted_response, separators=(",", ":")),
                state=state,                                  # passed in full; helper compares versions
                pre_version=pre_version,                       # so helper writes the new state row iff advanced
                tool_exception_in_flight=tool_exception_in_flight,  # signals primacy disposition
            )
        )

# _persist_tool_row_with_audit_failure_handling implementation:
async def _persist_tool_row_with_audit_failure_handling(
    *,
    sessions_service: SessionsService,
    session_id: str,
    tool_call_id: str,
    parent_assistant_id: str,
    content: str,
    state: CompositionState,
    pre_version: int,
    tool_exception_in_flight: BaseException | None,
) -> None:
    """Atomically write the tool row plus (if state advanced) the new
    composition_states row. Both go in ONE database transaction so the
    bidirectional INV-AUDIT-AHEAD invariant holds: chat_messages and
    composition_states advance together or neither advances.

    Audit-failure primacy:
    - If a tool exception is in flight (the compose loop is unwinding),
      we record the audit failure via telemetry and let the tool exception
      propagate. The tool exception is the dominant signal.
    - If no tool exception is in flight (the tool succeeded), an audit-write
      failure is a Tier-1 audit invariant violation per CLAUDE.md primacy.
      We re-raise audit_exc.
    """
    state_advanced = state.version > pre_version
    try:
        async with sessions_service.atomic_transaction() as txn:
            if state_advanced:
                composition_state_id = await txn.commit_composition_state(state)
            else:
                composition_state_id = None
            await txn.add_message(
                role="tool",
                tool_call_id=tool_call_id,
                parent_assistant_id=parent_assistant_id,
                content=content,
                composition_state_id=composition_state_id,
                # sequence_no reserved by add_message inside the same txn — §5.7
            )
    except IntegrityError:
        # Per §4.5: any IntegrityError on the chat_messages or composition_states
        # write is a Tier-1 invariant violation (FK to vanished state, duplicate
        # tool_call_id from a misbehaving LLM, CHECK constraint bug). Do NOT
        # silently recover. Propagate.
        _telemetry.composer_audit_tool_row_persist_failed_total.add(1)
        raise
    except Exception as audit_exc:
        # Audit insert failed for a non-IntegrityError reason (DB unavailable,
        # connection drop, etc.). Apply audit-failure primacy:
        _telemetry.composer_audit_tool_row_persist_failed_total.add(1)
        if tool_exception_in_flight is not None:
            # The tool already failed; the tool exception is the dominant
            # signal. Record the audit failure (counter above) and let the
            # tool exception propagate when this finally clause exits.
            log.warning(  # permitted: audit-system failure (CLAUDE.md primacy)
                "audit_insert_failed_during_tool_failure_unwind",
                tool_call_id=tool_call_id,
                tool_exc_class=type(tool_exception_in_flight).__name__,
                audit_exc_class=type(audit_exc).__name__,
            )
            return
        # Tool succeeded but audit failed: Tier-1 audit invariant violation.
        # Per CLAUDE.md primacy ("audit fires first, sync, crash-on-failure")
        # we MUST raise.
        raise
```

### 5.3 Bidirectional audit-ahead-of-state invariant (INV-AUDIT-AHEAD)

The invariant is **bidirectional**:

1. **`chat_messages` may be ahead of `composition_states`** (showing what
   was attempted) but never behind (claiming work that did not land).
2. **`composition_states` must NOT be ahead of `chat_messages`.** Every
   committed `composition_states` row that resulted from a tool call must
   have a corresponding `chat_messages` row with `role='tool'`, written
   *no later than* the `composition_states` commit.

This invariant derives directly from CLAUDE.md's auditability standard:
*"no inference — if it's not recorded, it didn't happen."* If
`composition_states` advances past `chat_messages`, the database asserts
that work happened (the state changed) without recording the evidence
(the tool row). That is the canonical fabrication failure mode CLAUDE.md
forbids.

**Mechanical enforcement:**

- `execute_tool()` does not commit `composition_states` directly; it
  mutates an in-memory `CompositionState`.
- The single transaction that writes the tool row to `chat_messages`
  ALSO writes the new `composition_states` row (when state advanced).
  The `_persist_tool_row_with_audit_failure_handling` helper takes the
  state row as an in-memory `CompositionState` and writes both
  atomically.
- `asyncio.shield` around the entire write protects against cancellation
  between the in-memory mutation and the durable commit.
- The property test (§8.3) asserts the post-condition.

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

| Failure | What persists | What the user sees |
|---|---|---|
| Tool returns successfully | assistant row + N tool rows + state mutation in one shielded transaction sequence | Normal continuation. |
| Tool raises `ToolArgumentError` | assistant row + N-1 normal tool rows + 1 error tool row (Tier-3 boundary signal); loop continues | Conversation continues; the LLM gets the error tool row as feedback. |
| Tool raises `Exception` (non-ToolArgumentError) | assistant row + tool rows up to the crash + 1 error tool row for the crashing call (shielded best-effort); exception propagates | Existing `_handle_plugin_crash` runs; 500 response with `partial_state`. Recovery panel shows the crash row. |
| `asyncio.CancelledError` during tool execution | assistant row + completed tool rows; `CancelledError` propagates *through* `finally` (no `except` arm catches `BaseException`); shielded audit insert in `finally` completes before `CancelledError` resumes propagation | `tool_calls_attempted - tool_responses_persisted` arithmetic surfaces the gap. |
| `asyncio.CancelledError` between tool execution and audit insert | shielded — the audit insert completes; `CancelledError` then resumes propagation | Invariant preserved. |
| DB write fails on tool row, no tool exception in flight | Tier-1 audit failure: helper raises | `partial_state_save_failed=true` propagates as today; 500 response. |
| DB write fails on tool row, tool exception already in flight | helper logs (permitted use), increments counter, returns | Tool exception propagates normally. Audit failure is visible via telemetry counter. |
| `IntegrityError` on any constraint | helper increments counter, raises | Crash; surfaces real bug class (LLM misbehaviour, sequence race, etc.). |

### 5.6 Why a separate transaction per tool row

Putting all tool rows in one big transaction means a late failure rolls
back early audit records. Splitting per row preserves the
audit-ahead-of-state invariant in the forward direction. The atomic pair
(tool row + composition_states row) preserves it in the backward
direction.

### 5.7 Infrastructure additions on `SessionsService`

This work requires three new methods on `SessionsService`. They are not
hypothetical — the implementing engineer adds them under this ticket.

```python
class SessionsService:
    # Existing public API — unchanged signature except for new parameters:
    async def add_message(
        self,
        *,
        session_id: str,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        tool_call_id: str | None = None,                 # NEW
        parent_assistant_id: str | None = None,          # NEW
        composition_state_id: str | None = None,
    ) -> str:
        """Insert a chat_messages row. Reserves sequence_no atomically inside
        a single transaction by (a) holding a session-scoped advisory lock,
        (b) selecting MAX(sequence_no) for the session, (c) inserting with
        MAX+1, (d) releasing the lock on commit.

        Returns the inserted row's id.

        Single-writer-per-session is structurally enforced: only the compose
        loop writes assistant/tool rows; only the route layer writes user
        rows. The route layer does not run concurrently with the compose
        loop on the same session because the compose loop is the body of
        the same HTTP request that received the user message. The advisory
        lock guards against pathological misuse (e.g., a future admin tool
        writing system rows mid-compose) and against PostgreSQL READ
        COMMITTED multi-session reservation collisions.

        Concurrency contract:
        - SQLite: relies on the global write lock; advisory-lock no-op.
        - PostgreSQL: pg_advisory_xact_lock(hashtext(session_id)) inside the
          transaction; lock released on commit.
        """
        ...

    @asynccontextmanager
    async def atomic_transaction(self) -> AsyncIterator[SessionsTransaction]:
        """Context manager that yields a transaction handle exposing
        ``commit_composition_state`` and ``add_message`` methods that share
        a single underlying transaction. On exit, commits or rolls back as
        usual.

        See §5.3 for the bidirectional INV-AUDIT-AHEAD invariant this
        primitive enforces.
        """
        ...


class SessionsTransaction:
    """Transaction-scoped handle. Lifetime is bounded by the surrounding
    ``atomic_transaction()`` context manager; do NOT retain references
    past context exit.
    """

    async def add_message(self, *, role, content, ..., composition_state_id) -> str:
        """Same signature as SessionsService.add_message except it does NOT
        open its own transaction; it uses the ambient one. Reserves
        sequence_no within the same transaction.
        """
        ...

    async def commit_composition_state(self, state: CompositionState) -> str:
        """Insert a composition_states row reflecting the current in-memory
        state. Returns the inserted row's id. Uses the ambient transaction.
        Equivalent in semantics to the existing inline state-row insert at
        web/sessions/service.py (~lines 396-411, 829-840).
        """
        ...
```

The `_telemetry` module-level singleton lives in
`src/elspeth/web/composer/telemetry.py` (new), exposing the named OTel
counters. Tests inject a fake counter via a constructor parameter on
the composer service for assertable behaviour.

The `lookup_redaction_policy(tool_name: str) -> ToolRedactionPolicy`
helper raises `MissingRedactionPolicyError` (a new exception) on
unregistered names. The adequacy guard (§4.4) ensures this never fires
in practice — every registered tool has a policy, enforced at registry
build time. The crash on missing policy is offensive programming: the
case is impossible by construction; if it ever fires, the registry
itself is corrupt and the audit trail must not proceed.

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

- Session ownership check unchanged.
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
    every tool has a `ToolRedactionPolicy` attribute or `EXEMPT_FROM_ADEQUACY_CHECK = True`.
  - **Adequacy guard test.** For every non-exempt tool, introspect its
    Pydantic argument model via `model_fields`; if any field is string-typed,
    assert either `sensitive_argument_keys` non-empty OR
    `handles_no_sensitive_data=True` with a non-stop-list reason. Symmetric
    for response schema.
  - **Reason stop-list test.** Reject `handles_no_sensitive_data_reason`
    matching the stop-list; allow concrete reasons.
  - **Construction-time validators.**
    - Orphan summarizer key → `ValueError`.
    - `handles_no_sensitive_data=True` with empty/whitespace reason → `ValueError`.
    - `handles_no_sensitive_data=False` with non-None reason → `ValueError`.
  - **Per-policy round-trip.** Synthetic tool call with sentinel-marked
    arguments; `apply_redaction_policy()` produces expected redacted
    output; `argument_summarizers` produce expected strings; **non-listed
    keys are byte-identical to input** (defends against a buggy policy
    that redacts everything).
  - **Summarizer raises.** Inject a summarizer that raises; assert the
    fallback sentinel `<redacted-summarizer-error:{exc_type}>` is written;
    OTel counter `composer.redaction.summarizer_errors_total` increments;
    redaction never raises through the persistence boundary.

- `tests/unit/web/sessions/test_chat_messages.py`
  - Assert new `tool_call_id`, `parent_assistant_id`, `sequence_no` columns
    exist.
  - Assert `(role='tool') = (tool_call_id IS NOT NULL)` check rejects.
  - Assert `(role='tool') = (parent_assistant_id IS NOT NULL)` check rejects.
  - Assert composite index `(session_id, sequence_no)` is unique.
  - Assert partial unique index `(session_id, tool_call_id) WHERE role='tool'` rejects duplicates.
  - Assert `ON DELETE CASCADE` from session removes all rows; from
    assistant row removes child tool rows (orphan prevention).

- `tests/unit/web/composer/test_compose_loop_persistence.py`
  - Drive a fake LLM emitting 3 tool calls in one turn.
  - Assert: 1 assistant row with redacted `tool_calls`; 3 `role='tool'`
    rows with matching `tool_call_id`; ordered by `sequence_no`;
    `composition_state_id` set on rows where `state.version` advanced;
    `parent_assistant_id` matches the assistant row's `id`.

- `tests/unit/web/composer/test_composer_holds_sessions_service.py`
  - Assert composer service constructor accepts and stores a
    `SessionsService` handle (the dependency that was missing today).

- `tests/unit/web/sessions/test_sessions_transaction.py`
  - Assert `atomic_transaction()` yields a `SessionsTransaction`.
  - Assert `txn.add_message(...)` reserves sequence_no inside the same txn.
  - Assert `txn.commit_composition_state(...)` commits inside the same txn.
  - Assert rollback on context-manager exception rolls both writes.

- `tests/unit/web/composer/test_audit_failure_primacy.py`
  - Tool succeeds + audit fails (non-IntegrityError) → helper raises.
  - Tool fails + audit fails (non-IntegrityError) → helper logs (permitted),
    counter increments, helper returns; tool exception propagates from finally.
  - Audit IntegrityError (any constraint) → helper raises (no recovery).

### 8.2 Backend — integration

Extend `tests/integration/pipeline/test_composer_llm_eval_characterization.py`:

- **CL-PP-1: Convergence error mid-loop.** Force budget exhaustion at
  turn N. Assert assistant rows for 0..N exist; tool rows for completed
  calls exist; response body's `failed_turn.tool_calls_attempted` and
  `tool_responses_persisted` match observed counts; `partial_state`
  matches the latest `composition_states` row.
- **CL-PP-2a: Plugin crash, audit-row-precedes-raise.** Use a
  `threading.Event` (or asyncio equivalent) inside the patched
  `execute_tool` so it awaits *after* the audit insert completes,
  forcing causal ordering. Assert the crashing tool's row exists.
- **CL-PP-2b: Plugin crash, raise-precedes-audit.** Use the same
  primitive to force the exception to fire BEFORE the persistence
  machinery runs. Assert the shielded `finally`-block insert still
  produces a row.
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
- **CL-PP-4d: DB write fails on `composition_states` commit (within atomic pair).**
  Assert the tool-row + state-row atomic write rolls back together so
  the bidirectional invariant holds.
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
`tool_raises_ToolArgumentError`, `tool_raises_Exception`,
`audit_raises_IntegrityError`, `audit_raises_OperationalError`,
`state_commit_fails`). Maps 1:1 onto the §5.5 failure-mode rows.
Coverage: every §5.5 row reached at least once per test campaign.

**`st_cancellation_arrival_time`** — value space: enum of
(`before_assistant_write`, `between_assistant_and_tool_n`,
`during_tool_execution`, `during_audit_insert`, `after_audit_insert`).
Coverage: every CL-PP-7-class race window exercised.

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
  for every session, sequence_no values are densely monotonic (no gaps,
    no duplicates, starting at 1 for first message)
  for every (session_id, tool_call_id) where role='tool': exactly one row
  for every assistant row a, every child tool row t:
    t.created_at >= a.created_at
    t.parent_assistant_id == a.id
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

This ticket closes when:

1. Backend test set above is green.
2. Frontend test set above is green.
3. RC5-UX (or successor) branch CI passes including
   `enforce_tier_model.py` and `enforce_freeze_guards.py`.
4. All scenarios listed in §8.2 (CL-PP-1, 2a, 2b, 3, 4a, 4b, 4c, 4d, 5,
   6, 7, 8 — twelve total) are present in
   `test_composer_llm_eval_characterization.py` and pass.

**This ticket does NOT validate that users can actually recover from a
failure.** That validation is owned by
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa), the final staging
replay, which becomes unblocked when this ticket lands. VER is the
contract; VAL is the user need; they are intentionally separated.

### 8.6 Test path integrity

Composer integration tests exercise the web/composer service surface;
they do not run pipelines through `ExecutionGraph.from_plugin_instances`
or `instantiate_plugins_from_config`, because the composer is the
authoring surface, not the engine. CLAUDE.md's test-path-integrity rule
("never bypass production code paths in tests") applies to **pipeline
engine integration tests**, not web tests; this design's test plan
honours that distinction.

### 8.7 Test data hygiene and fixture extension

All tests use the existing `chaos*` fixtures under `tests/`
(specifically `ChaosLLM` for composer LLM mocking). No live OpenRouter
calls in CI. **Scope note:** if `ChaosLLM` does not currently support
multi-turn LLM mocking with structured `tool_calls` JSON emissions,
extending it is part of this ticket's scope. The implementer verifies
on day 1; if extension is needed, it lands in the same PR. The
`elspeth-xdist-auto` plugin shipped inside `src/elspeth/testing/` is
separate from the project's own test suite, per CLAUDE.md.

---

## 9. Risks and Mitigations

| ID | Risk | Likelihood | Impact | Trigger | Mitigation | Owner |
|---|---|---|---|---|---|---|
| RSK-01 | Tool author ships a tool without redaction policy. | Low | High (silent leakage) | Adequacy-guard CI test fires red on PR. | Adequacy guard (§4.4); registry-iteration test enforces presence; Pydantic schema introspection drives detection; `handles_no_sensitive_data=True` requires concrete reason. | Implementing engineer per PR; reviewer sign-off |
| RSK-02 | `handles_no_sensitive_data=True` declarations normalize over time (Shifting the Burden). | Medium | Medium (erosion of audit safety) | Any new declaration of `handles_no_sensitive_data=True` after the initial set ships, OR existing reasons match the stop-list. | Stop-list of placeholder reasons (§4.4) auto-rejects boilerplate; quarterly review of all `handles_no_sensitive_data=True` declarations + reasons; new declarations surface in PR review via the registry-iteration test diff. | Security review at RC 5.1; PR reviewers |
| RSK-03 | Redaction summarizer raises on pathological input. | Low | Medium | OTel counter `composer.redaction.summarizer_errors_total` exceeds 0.1% of tool calls in 24h. | Persistence wrapper catches summarizer exceptions and falls back to `<redacted-summarizer-error:{exc_type}>`. Property test asserts redaction never raises. | RC 5.1 production hardening |
| RSK-04 | Per-row transactions slow down the loop. | Low | Low (latency NFR §1.4) | CI sanity bound (p95 ≤ 250 ms with N ≤ 8) red. | CI sanity bound + nightly tight bench; bounded write per insert; SQLite/PostgreSQL handle small transactions well. | Implementing engineer |
| RSK-05 | Frontend diff blows up on very large `partial_state`. | Low | Low (UX nuisance) | Recovery panel TTI exceeds 500 ms in CI sanity test. | Diff helper iterates fields rather than diffing entire JSON; UI shows "large diff — expand" disclosure for thousands of nodes. | Frontend follow-up |
| RSK-06 | New `tool_call_id` index slows large message-table writes. | Very low | Low | Insert latency regression in benchmarks. | Composite single-column index; SQLite/PostgreSQL handle this trivially. | n/a |
| RSK-07 | Audit-ahead-of-state invariant violated during cancellation. | Low | Critical (auditability standard breach) | Property test failure; or post-condition assertion in `_compose_loop` fires. | `asyncio.shield` around audit writes; bidirectional invariant in §5.3; CL-PP-3 + CL-PP-7 cover the cancel paths. | Implementing engineer; verified by property test |
| RSK-08 | `chat_messages` table grows unboundedly without retention. | Medium (over time) | Low (single-operator pre-release) | Table size exceeds 1 GB in dev/staging. | `chat_messages` rows cascade-delete with sessions today; explicit retention extension is filed under [elspeth-RETENTION-WEB] (§4.6, §10 OQ-1). | RC 5.1 production hardening |
| RSK-09 | Partial unique index syntax differs across DB dialects. | Low | Low | DDL fails on a target dialect. | Use SQL `CREATE UNIQUE INDEX ... WHERE ...` (SQLite 3.8.0+; PostgreSQL); SQLAlchemy DDL emit hook for both dialects. | Implementing engineer |
| RSK-10 | `composer.audit.state_rolled_back_during_persist_total` SLO breach (state-rollback race fires in production). | Very low | Critical (audit invariant violation) | Counter exceeds 0 in any 24h window. | Counter wired to alarm at threshold 0; §1.4 SLO. CL-PP-4d covers the sole reachable trigger. | RC 5.1 SRE; security review |
| RSK-11 | Audit-write failure on tool row when no tool exception in flight (Tier-1 violation). | Very low | Critical | OTel counter `composer.audit.tool_row_persist_failed_total` non-zero. | §5.2 helper raises in this case (audit primacy); §8.1 unit test asserts. | RC 5.1 SRE |
| RSK-12 | LLM provider re-uses `tool_call_id` across turns within a session. | Medium (provider-dependent) | High (silent mis-correlation) | Partial unique index rejects insert; CL-PP-8 fires. | Crash on duplicate; do not silently recover. Per-provider observation: OpenRouter/OpenAI ids are message-scoped per current spec but not contractually forever. | Implementing engineer |

---

## 10. Open Questions

- **OQ-1.** Filigree ticket ID for the `chat_messages` retention CLI
  extension (referenced as `[elspeth-RETENTION-WEB]` in §4.6 and RSK-08).
  File during implementation; cite in the implementation PR description.
- **OQ-2.** `composition_states` redaction symmetry: today the partial
  state is persisted with raw paths; the new code path uniformly
  redacts. The historical asymmetry is filed as a follow-up issue
  during implementation. Document the planned ID in the PR description.

Both questions are administrative (file a ticket, cite an ID); neither
blocks design or implementation.

---

## 11. References

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
  as of 2026-04-30, post-merge of all of the above.
- Panel reviews: revision 1 reviewed 2026-04-30 by four reviewers
  (solution architect, systems thinker, Python engineer, QA analyst);
  revision 2 reviewed by the same four reviewers; revision 3
  incorporates findings from both passes.

---

## 12. Glossary

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
  See §5.2 helper.
