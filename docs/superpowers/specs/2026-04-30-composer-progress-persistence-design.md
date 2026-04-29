# Composer Progress Persistence — Design

**Ticket:** [elspeth-90b4542b63](filigree:elspeth-90b4542b63) — Composer progress persistence — tool-call breadcrumbs and partial drafts survive long-running failures
**Parent epic:** [elspeth-528bde62bb](filigree:elspeth-528bde62bb) — Composer LLM evaluation remediation
**Related future epic:** [elspeth-f0460a6594](filigree:elspeth-f0460a6594) — Composer async/background execution model (deferred to Future release)
**Date:** 2026-04-30
**Status:** Proposed
**Branch:** RC5-UX (or successor)

---

## 1. Goals and Non-Goals

### Goal

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
   sequence the LLM saw.

### Non-Goals

- Async / background-job execution. Filed as future epic
  [elspeth-f0460a6594](filigree:elspeth-f0460a6594).
- Resume-by-replay (re-running tool calls automatically). The user resumes
  from the persisted state; they do not re-invoke the LLM with the partial
  transcript.
- Streaming progress beyond what `_active` and `composer-progress` already
  provide.
- Migration of existing `chat_messages`. Pre-release per CLAUDE.md no-legacy
  policy — direct cutover.

### In-Scope Failure Paths

The three existing route helpers in `web/sessions/routes.py`:

- `_handle_convergence_error` — wall-clock timeout, turn-budget exhaustion.
- `_handle_plugin_crash` — plugin bug surfaced through composer.
- `_handle_runtime_preflight_failure` — final-gate runtime preflight reject.

Plus any future failure path that surfaces a `ComposerServiceError` subclass
carrying `partial_state` (the contract is the exception type, not the
specific failure mode).

---

## 2. Context — What Already Exists

The original ticket text (filed 2026-04-27) was written before the
remediation work in commits 4fce0cae (RC reason codes), 1ad03ddd (in-flight
observability + cancel-race drain), and 83e6228d (redacted blob path
sentinel) landed. Several of the original concerns are already addressed:

| Original concern | Status today |
|---|---|
| "No partial state survives a failure." | **Mostly fixed.** `ComposerConvergenceError`, `ComposerPluginCrashError`, `ComposerRuntimePreflightError` each carry `partial_state: CompositionState`, captured iff `state.version > 0` (see `web/composer/protocol.py:84-288`). Three route handlers persist that partial state and return it in 422/500 responses. |
| "Wall-clock timeout has no breadcrumbs." | **Fixed.** Inner LLM-call `asyncio.wait_for` timeouts convert to `ComposerConvergenceError.capture(state)` at `web/composer/service.py:1226-1231`. |
| "No in-flight visibility." | **Fixed.** `composer.requests.inflight` UpDownCounter, `composer.request.terminal.total` Counter, `GET /api/sessions/_active` cross-session enumeration (commit 1ad03ddd). |
| "Distinct failure causes look the same." | **Fixed.** Discriminated reason codes on `ComposerProgressEvent`: `convergence_wall_clock_timeout`, `convergence_discovery_budget`, `convergence_composition_budget`, `client_cancelled`, `runtime_preflight_failed` (commit 4fce0cae). |
| "Only final assistant text is persisted." | **Partly fixed.** Per-turn assistant rows with `tool_calls` JSON metadata are written each loop iteration. The `chat_messages.tool_calls` column exists and is populated. |
| "No tool-result rows in chat history." | **Open.** The schema permits `role='tool'` (`web/sessions/models.py:82`) but no insert site exists. Tool *responses* are not persisted. |
| "Frontend has no recovery surface." | **Open.** The data is now present in DB and in failure response body, but no UX renders it as a "draft you can pick up." |
| "Tool argument redaction is unverified." | **Open.** No central redaction layer exists for `chat_messages.tool_calls` JSON content; the existing `redact_source_storage_path` helper is path-specific. |

This design closes the three remaining open gaps. The async / background-job
direction is captured in the future epic and does not constrain this work.

---

## 3. Approach Decisions

| Decision | Choice | Alternatives considered |
|---|---|---|
| Tool-response persistence shape | **A1.** One `role='tool'` `chat_messages` row per tool response, correlated to its assistant turn via `tool_call_id`. | A2 (embed responses in assistant row's `tool_calls` JSON) was simpler but not queryable per-tool, and the schema's permitted `'tool'` role is a Chekhov's gun left by the original schema author. |
| Frontend recovery UX | **F2.** Diff-and-confirm modal showing pipeline diff + tool transcript with explicit Apply / Discard buttons. | F1 (auto-apply) was destructive of unsaved manual edits; F3 (reload-to-recover) was clunky. |
| Tool argument redaction | **R3.** Each tool declares `ToolRedactionPolicy(sensitive_argument_keys, sensitive_response_keys, argument_summarizers)`; persistence layer enforces. | R1 (no redaction) leaks by default; R2 (central policy) couples the redaction layer to every tool's argument shape and rots. |
| Migration | **None.** Pre-release per CLAUDE.md. New columns ship as part of the schema; no backfill. | A migration would have been required post-release; this design stays simple. |

---

## 4. Data Model

### `chat_messages` schema change

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
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("composition_state_id", String, nullable=True),
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
    Index(                                                                         # NEW
        "ix_chat_messages_session_tool_call_id",
        "session_id",
        "tool_call_id",
    ),
)
```

**Database-enforced invariants:**

- `tool_call_id` is non-null iff `role='tool'`. A `role='tool'` row without a
  correlation ID is a bug; a non-tool row with one is also a bug.
- Composite index on `(session_id, tool_call_id)` makes the recovery
  panel's "find tool transcript for this session" query O(log n).

The `composition_state_id` FK already exists; tool rows set it to the new
state version when the tool mutated state, otherwise NULL.

### `ToolRedactionPolicy` — per-tool declared policy

```python
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class ToolRedactionPolicy:
    """Redaction declaration owned by the tool, enforced at the persistence boundary.

    sensitive_argument_keys: keys in the tool-call argument dict whose values
        must be replaced by ``<redacted>`` (or the per-key summarizer output)
        before the tool call is persisted to chat_messages.tool_calls JSON.

    sensitive_response_keys: keys in the tool's response dict whose values
        must be replaced before the tool's response is persisted to
        chat_messages.content (as JSON).

    argument_summarizers: optional per-key replacement functions. Used when
        the sentinel ``<redacted>`` would lose diagnostically useful
        information (e.g. byte-count of an inline blob).
    """

    sensitive_argument_keys: tuple[str, ...] = ()
    sensitive_response_keys: tuple[str, ...] = ()
    argument_summarizers: Mapping[str, Callable[[Any], str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        freeze_fields(self, "sensitive_argument_keys", "sensitive_response_keys", "argument_summarizers")
```

Each composer tool exposes a `REDACTION_POLICY: ToolRedactionPolicy`
attribute alongside its existing schema declaration. A test fixture
iterates the registry and asserts every tool has the attribute, even if the
policy is empty — so a future tool author cannot ship without thinking
about what is sensitive.

### Sentinel rules

- Plain sensitive key → value replaced by literal string `"<redacted>"`.
- Key with summarizer → value replaced by `summarizer(original_value)`,
  e.g. `lambda b: f"<inline-blob:{len(b)}-bytes>"`.
- Existing `redact_source_storage_path` continues to handle source paths in
  the persisted `partial_state`. It is unchanged by this work; the new
  redaction is for tool arguments / responses, a separate axis.

### Initial policy declarations

To be defined per-tool during implementation; representative examples:

| Tool | sensitive_argument_keys | sensitive_response_keys | summarizers |
|---|---|---|---|
| `wire_secret_ref` | `()` (the *name* is not sensitive; never persists the value) | `()` | none |
| `set_source` | `("path",)` (passes through `redact_source_storage_path`) | `()` | `path` → `redact_source_storage_path` |
| `create_blob` | `("content",)` | `()` | `content` → `<inline-blob:{n}-bytes>` |
| `patch_*` | depends on plugin schema; declared per-tool | `()` | as needed |

Final list assembled during implementation; the unit test enforces that
every registered tool has a declaration.

---

## 5. Persistence Boundary

### Insertion sites in the compose loop

The loop in `web/composer/service.py:_compose_loop` becomes:

```python
# Step A — redact and persist the assistant turn
redacted_tool_calls = tuple(
    apply_redaction_policy(tc, redaction_registry[tc.function.name])
    for tc in assistant_message.tool_calls
)
await save_chat_message(
    session_id=session_id,
    role="assistant",
    content=assistant_message.content or "",
    tool_calls=redacted_tool_calls,
    composition_state_id=current_state_id,
)

# Step B — execute and persist each tool turn
for tool_call in assistant_message.tool_calls:
    pre_version = state.version
    try:
        response = await execute_tool(tool_call, state)
        response_for_persistence = response
    except Exception as exc:
        # Tier-1 read-side discipline: any exception raised inside execute_tool
        # is recorded as the tool row's content, then re-raised so the existing
        # ``ComposerPluginCrashError.capture(state)`` propagation path is
        # unchanged. This is best-effort audit — the audit insert is in a
        # finally block so even an exception during persistence does not lose
        # the underlying tool failure.
        response_for_persistence = {
            "error": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        post_version = state.version
        new_state_id = (
            state_id_for_version(post_version)
            if post_version > pre_version
            else None
        )
        redacted_response = apply_response_redaction(
            response_for_persistence,
            redaction_registry[tool_call.function.name],
        )
        await save_chat_message(
            session_id=session_id,
            role="tool",
            content=json.dumps(redacted_response, separators=(",", ":")),
            tool_call_id=tool_call.id,
            composition_state_id=new_state_id,
        )
```

### Failure mode interaction

| Failure | What persists | What the user sees |
|---|---|---|
| Tool returns successfully | assistant row + N tool rows + state mutation | Normal continuation. |
| Tool raises but loop continues | assistant row + N-1 normal tool rows + 1 error tool row | Error tool row carries `{"error": "ToolError", "message": "..."}` (already redacted by the same policy). Conversation can continue. |
| Tool raises `ComposerPluginCrashError.capture(state)` | assistant row + tool rows up to the crash, plus an error tool row for the crashing call (best-effort) | 500 response with `partial_state`. Recovery panel shows the crash row. |
| Wall-clock `TimeoutError` mid-tool-execution | assistant row + tool rows up to the cancellation boundary | Cancelled tool's row missing; partial_state reflects last committed state. `tool_calls_attempted - tool_responses_persisted` arithmetic surfaces the gap. |
| DB write fails mid-loop | Whatever was committed before the failure | Existing `partial_state_save_failed` signal extends naturally. |

### Why a separate transaction per row

Putting all tool rows in one big transaction means a late failure rolls
back early audit records. Splitting per row preserves the rule that
**partial state should always be more conservative than the audit trail** —
`chat_messages` can be ahead of `composition_states` (showing what was
attempted) but never behind (claiming work that did not land).

### Cancellation semantics

The new per-row inserts run via the existing `_run_sync` helper, which
already participates in the cancel-race drain (commit 1ad03ddd). No new
cancellation surface is introduced.

---

## 6. Route Handling and Response Shape

### Existing `_handle_*` helpers — minimal change

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
        "transcript_url": "/api/sessions/{sid}/messages?since={user_message_id}",
    },
}
```

`tool_calls_attempted - tool_responses_persisted` surfaces "the LLM tried 4
tools, only 3 finished" without forcing a separate round-trip.

### Transcript fetch endpoint

The existing `GET /api/sessions/{sid}/messages` returns chat history. It
gains:

- New response field per row: `tool_call_id` (mirrors the new column).
- New query parameter: `include_tool_rows: bool = False`. Default `false`
  keeps the live chat panel's existing behavior (no tool rows interleaved
  into user/assistant flow). Recovery panel sets `include_tool_rows=true`.

### Auth and redaction reuse

- Session ownership check unchanged.
- **Redact at write, never at read.** Persisted shape is the canonical safe
  shape; route handlers do not re-redact. This eliminates the entire class
  of bug "future read path forgets to redact."

### What does NOT change

- The wall-clock 180s budget.
- The `_active` cross-session enumeration endpoint.
- OTel `composer.requests.inflight` gauge / `composer.request.terminal.total` counter.
- The `/composer-progress` endpoint.

---

## 7. Frontend Recovery Surface

### Component layout

New directory `src/elspeth/web/frontend/src/components/recovery/`:

- `RecoveryPanel.tsx` — the modal.
- `RecoveryDiff.tsx` — pipeline diff section.
- `RecoveryTranscript.tsx` — tool transcript section.

New hook `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts` —
manages open/closed state and apply/discard semantics.

### Trigger

`useComposer.sendMessage()` already has an `onError` path. When the response
body carries `partial_state` AND `failed_turn`, the hook calls
`useRecoveryPanel.open(failureResponse)` instead of (or in addition to) the
existing toast/banner.

### Visual layout

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

### Diff calculation

Client-side. Iterate `source / nodes / edges` in current editor state vs
`partial_state`; produce structured deltas. No new backend work.

### Tool transcript

Pulled from `GET /api/sessions/{sid}/messages?since={user_message_id}&include_tool_rows=true`.
Each `role='tool'` row contributes one entry; the assistant row's
redacted `tool_calls` JSON contributes the per-call argument summary.

### Apply / discard / view raw

- **Apply.** Frontend overwrites in-memory editor state with `partial_state`.
  No server round-trip — `composition_states` already has the row.
- **Discard.** Closes the panel; editor state unchanged. The DB record
  remains for audit (discard is a UI choice, not a data-deletion command).
- **View raw transcript.** Expands to a read-only view of every redacted
  tool call/response JSON.

### Accessibility

- Focus-trapped modal (existing `Modal` pattern).
- Apply requires explicit click; no auto-apply on Enter.
- Reason badge has both colour and text label (no colour-only signalling).

### What does NOT change

- Live chat panel (`components/chat/`) keeps its current default
  `include_tool_rows=false` — tool rows do not interleave into the live
  conversation by default. They appear only in the recovery panel and the
  raw-transcript view.

The frontend will get its own follow-up review pass per the user's note;
this design captures the contract and the surface, not the final visual
polish.

---

## 8. Testing Strategy

### Backend — unit

- `tests/unit/web/composer/test_redaction_policy.py`
  - Iterate composer tool registry; assert every tool has a
    `ToolRedactionPolicy` attribute (even if empty).
  - Per-policy round-trip: synthetic tool call with sentinel-marked
    arguments; `apply_redaction_policy()` produces expected redacted
    output; `argument_summarizers` produce expected strings.
- `tests/unit/web/sessions/test_chat_messages.py`
  - Assert new `tool_call_id` column exists.
  - Assert `(role='tool') = (tool_call_id IS NOT NULL)` check constraint
    rejects a violating insert.
  - Assert composite index `(session_id, tool_call_id)` is present.
- `tests/unit/web/composer/test_compose_loop_persistence.py`
  - Drive a fake LLM emitting 3 tool calls in one turn.
  - Assert: 1 assistant row with redacted `tool_calls`; 3 `role='tool'`
    rows with matching `tool_call_id`; ordered by `created_at`;
    `composition_state_id` set on rows where `state.version` advanced.

### Backend — integration

Extend `tests/integration/pipeline/test_composer_llm_eval_characterization.py`:

- **CL-PP-1: Convergence error mid-loop.** Force budget exhaustion at
  turn N. Assert assistant rows for 0..N exist; tool rows for completed
  calls exist; response body's `failed_turn.tool_calls_attempted` and
  `tool_responses_persisted` match observed counts; `partial_state`
  matches the latest `composition_states` row.
- **CL-PP-2: Plugin crash mid-tool.** Inject a tool that raises mid-execution.
  Assert partial assistant row exists; prior tool rows exist; crashing
  tool's row exists with error content (or is absent if crash beat the
  audit insert — both valid; arithmetic still tells the truth).
- **CL-PP-3: Wall-clock timeout during tool execution.** Force
  `asyncio.TimeoutError` in a tool. Assert the convergence error
  captured `partial_state`; audit trail consistent with captured state.
- **CL-PP-4: DB write fails.** Inject a `save_chat_message` failure
  mid-loop. Assert `partial_state_save_failed=true` propagates as today;
  no inconsistent state.

### Backend — property test

Hypothesis-style:

```
For any well-formed assistant turn with N tool calls,
  after the loop completes (success or failure),
    count(role='assistant' rows for this turn) == 1
    count(role='tool' rows for this turn) <= N
    count(role='tool' rows for this turn) == failed_turn.tool_responses_persisted (if failure)
    every role='tool' row has tool_call_id matching exactly one entry in the assistant.tool_calls array
```

Codifies the audit-ahead-of-state invariant; survives refactors that
change loop structure.

### Frontend

- `RecoveryPanel.test.tsx` — render with synthetic 422 response; assert
  diff section, transcript section, button states, accessibility hooks.
- `useRecoveryPanel.test.ts` — apply / discard semantics; no DB mutation
  on discard.
- `useComposer.test.ts` extension — recovery panel opens iff response
  body carries `partial_state` + `failed_turn`; otherwise existing toast
  path runs.

Full Playwright/E2E round-trip is deferred to
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa) (final staging replay).

### Verification scope

This ticket closes when:

1. Backend test set above is green.
2. Frontend test set above is green.
3. RC5-UX (or successor) branch CI passes including
   `enforce_tier_model.py`.
4. CL-PP-* scenarios are added to the characterization harness — they
   become the regression scoreboard for any future change to the compose
   loop.

The full staging replay (rerunning the original eval against
`https://elspeth.foundryside.dev`) is the responsibility of
[elspeth-599ecf69fa](filigree:elspeth-599ecf69fa), which becomes
unblocked when this ticket lands.

### Test data hygiene

All tests use the existing `chaos*` fixtures under `tests/`. Composer LLM
mocked via the existing characterization harness; no live OpenRouter calls
in CI. (The `elspeth-xdist-auto` plugin shipped inside
`src/elspeth/testing/` is separate from the project's own test suite, per
CLAUDE.md.)

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Tool author ships a new tool without declaring redaction policy. | Registry-iteration unit test fails the build until a policy is declared (even if empty). Mechanically discoverable invariant. |
| Redaction summarizer raises on pathological input (e.g. non-bytes value). | Persistence wrapper catches summarizer exceptions and falls back to `<redacted-summarizer-error:{exc_type}>`. Audit trail records the failure rather than crashing the compose loop. |
| Per-row transactions slow down the loop. | Each insert is a single bounded write. SQLite handles 20-row sequences comfortably; PostgreSQL (RC 5.1 alternate config) prefers small transactions for concurrent reads. No expected impact at current scale. |
| Frontend diff blows up on very large `partial_state`. | Diff helper iterates fields rather than diffing entire JSON; bounded by node/edge count. The UI can show a "large diff — expand" disclosure if a session has thousands of nodes. |
| New `tool_call_id` index slows large message-table writes. | Composite single-column index; SQLite/PostgreSQL handle this trivially. No expected impact. |

---

## 10. Open Questions

None blocking implementation. The frontend visual treatment is intentionally
a contract-level sketch; final polish happens in a frontend review pass after
the backend lands.

---

## 11. References

- CLAUDE.md — auditability standard, three-tier trust model, plugin
  ownership, no-legacy-code policy, frozen-dataclass deep-freeze contract.
- `engine-patterns-reference` skill — composite primary keys, schema
  contracts, secret handling, layer architecture & dependency analysis.
- `tier-model-deep-dive` skill — coercion rules, operation wrapping,
  fabrication decision test, web-server section.
- `logging-telemetry-policy` skill — audit primacy, telemetry-only
  exemptions, primacy test.
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
- Predecessor verification: this design's section 2 ("Context — what
  already exists") cites the observed state of the codebase as of
  2026-04-30, post-merge of all of the above.

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
  composer returns the response to the LLM (and now persists it as a
  `role='tool'` row).
- **Audit-ahead-of-state.** The invariant that `chat_messages` can be
  ahead of `composition_states` (showing what was attempted) but never
  behind (claiming work that did not land). The version > 0 capture rule
  for `partial_state` and the per-row insert pattern in this design both
  express it.
