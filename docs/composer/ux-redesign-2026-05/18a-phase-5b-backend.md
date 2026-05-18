# Phase 5b — Surface the LLM's interpretation (backend)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`. Implement task-by-task,
> checkbox-tracked. The overview, B2 verdict, scope, trust-tier check, and
> risks are in [18-phase-5b-surface-llm-interpretation.md](18-phase-5b-surface-llm-interpretation.md);
> read that first.

**Status header — B2 verdict (c) confirmed.** New event class →
new schema table + recorder method + composer-service wiring. See the
overview for the recon citations.

This plan covers the backend half of Phase 5b:

- New `interpretation_events_table` in the session audit DB.
- New `composition_states.provenance` enum value
  `interpretation_resolve`.
- New session and frozen-dataclass record types
  (`InterpretationEventRecord`, `InterpretationChoice`).
- New session service methods.
- New composer tool `request_interpretation_review` registered with the
  compose loop.
- New HTTP routes `/resolve` and `/opt_out`.
- Redaction-model entry for the new tool's arguments.
- Runtime hand-off: prompt-template patching at resolve time.

The frontend half is in
[18b-phase-5b-frontend.md](18b-phase-5b-frontend.md). The two MUST land
in order: 18a then 18b.

---

## Tech Stack (backend slice)

- Python 3.13, SQLAlchemy Core, pydantic v2.
- `pluggy` not touched (no new plugin types).
- Session audit DB: `web/sessions/{models,schema,service,routes,schemas,protocol}.py`.
- Composer tool surface: `web/composer/{tools,redaction,proposals,service}.py`.
- No new third-party dependencies.
- No Landscape (`core/landscape/`) changes.

## File structure (backend changes)

```text
src/elspeth/contracts/
  composer_interpretation.py                                    CREATE    (Task 1)

src/elspeth/web/sessions/
  models.py                                                     MODIFY    (Task 2)
  schema.py                                                     MODIFY    (Task 2 — validation footprint)
  protocol.py                                                   MODIFY    (Task 3)
  schemas.py                                                    MODIFY    (Task 3 — wire schemas)
  service.py                                                    MODIFY    (Task 4)
  routes.py                                                     MODIFY    (Task 6)
  _persist_payload.py                                           MODIFY    (Task 4 — if persistence carrier touched)

src/elspeth/web/composer/
  tools.py                                                      MODIFY    (Task 5)
  redaction.py                                                  MODIFY    (Task 5 — argument redaction)
  proposals.py                                                  MODIFY    (Task 5 — proposal summary)
  service.py                                                    MODIFY    (Task 5 — compose-loop hook)
  skills/pipeline_composer.md                                   MODIFY    (Task 8)

tests/unit/web/sessions/
  test_interpretation_events_table.py                           CREATE    (Task 2)
  test_interpretation_events_service.py                         CREATE    (Task 4)
  test_interpretation_events_routes.py                          CREATE    (Task 6)
  test_interpretation_opt_out_routes.py                         CREATE    (Task 7)

tests/unit/web/composer/
  test_request_interpretation_review_tool.py                    CREATE    (Task 5)
  test_request_interpretation_review_redaction.py               CREATE    (Task 5)

tests/integration/web/composer/
  test_interpretation_audit_spotcheck.py                        CREATE    (Task 9 — Landscape spot-check)
  test_interpretation_runtime_handoff.py                        CREATE    (Task 9 — prompt-template patch)
```

No Landscape schema files change. No `core/canonical.py` change.

---

## Migration runner ownership (deferred — see roadmap §D5)

Phase 5b is the third schema-addition under `project_db_migration_policy`. Each
addition wipes prior phases' user state. The structural fix — a migration
runner that preserves `user_preferences` and session history across schema
changes — is OWNED BY PHASE 9 (post-launch). Phase 5b ships under the current
delete-the-DB policy with explicit acknowledgment that users who completed the
tutorial between Phase 4 and Phase 5b will be re-tutorial'd on the Phase 5b
deploy. Operator action: communicate this to test users before each deploy.

---

## Task 0 — Empirical placeholder-convention validation gate

**This is a hard gate. All subsequent tasks depend on Task 0 passing. Do not
commit any Task 2 schema changes until Task 0 passes.**

**Ownership and prerequisites (operator-confirmed 2026-05-18):**
The implementer picking up Phase 5b runs Task 0 FIRST, before any other
Phase 5b work. The operator has confirmed that:
- LLM provider credentials are already provisioned in the project `.env`
  (the same keys the composer service uses against the staging LLM); no
  additional credential setup is required.
- Budget for ≥10 staging-LLM runs is approved; the implementer should
  not pause to ask for cost authorisation. If the gate fails and the
  architecture pivots, additional runs to validate the runtime-resolve
  fallback are also pre-authorised.
- The implementer reports the gate outcome (pass/fail + artifact path)
  to the operator before committing Task 2 schema changes. A failed gate
  surfaces to the operator immediately per the "Failure path" section
  below.

**Goal.** Confirm that the staging LLM reliably emits the
`{{interpretation:<term>}}` placeholder convention — which the entire
patch-on-resolve architecture depends on — BEFORE investing in schema,
service, and route code.

### What to run

Run the canonical hero prompt:

> create a list of 5 government web pages and use an LLM to rate how cool they are

≥10 times against the current staging LLM with the draft skill nudge embedded
in `src/elspeth/web/composer/skills/pipeline_composer.md` (the nudge text from
Task 8 may be drafted in a local branch and applied temporarily for these runs
— it does NOT need to be committed yet).

**Pass threshold:** ≥8 of 10 runs must satisfy BOTH of:

1. The LLM transform's `prompt_template` field contains a
   `{{interpretation:cool}}` placeholder (or equivalent — the term must
   literally appear in the `{{interpretation:<term>}}` form).
2. The LLM calls `request_interpretation_review` in the same composition turn
   as it stages the LLM transform.

### Failure path — architecture pivot

**If fewer than 8 of 10 runs pass:**

STOP. Do not commit Task 2 schema. The architecture MUST pivot to
**runtime-resolve-from-table**: store the resolved prompt template separately
on the interpretation event row itself (not as a patched placeholder in the
composition state), and have the runtime resolve via that table at execution
time. This pivot requires:

- A `resolved_prompt_template` column on `interpretation_events_table`
  (populated at resolve time).
- The runtime executor looks up the resolved template at run time rather than
  reading it verbatim from the composition state.
- Plan 18a must be re-drafted before any code is committed.

Surface immediately to the operator; do not silently rework.

### Artifact

Record each run in:

```
evals/composer-rgr/phase5b-task0-placeholder-validation.json
```

Schema per entry:

```json
{
  "run_index": 1,
  "timestamp_utc": "2026-05-18T12:00:00Z",
  "model_id": "anthropic/claude-opus-4-7",
  "model_version": "<provider-reported version string>",
  "prompt_template_observed": "<the prompt_template field as emitted>",
  "placeholder_emitted": true,
  "interpretation_review_called": true,
  "pass": true
}
```

The top-level object includes `"pass_count"`, `"total_runs"`, and
`"gate_passed"` (boolean, `pass_count >= 8`).

### Gate outcome

Record gate outcome in the artifact and in the PR description. If
`gate_passed = false`, the PR is blocked pending architecture pivot.

---

## Task 1 — Define the contract types

**Goal.** Add a frozen-dataclass + closed-enum contract in
`src/elspeth/contracts/composer_interpretation.py`. Following the project
convention for tier-1 read-side records: direct field access, no `.get()`,
`freeze_fields()` for any container fields. This is the type that the
service returns and the route hydrates.

**Files:**

- Create: `src/elspeth/contracts/composer_interpretation.py`.
- Create: `tests/unit/contracts/test_composer_interpretation.py`.

### Contract shape

```python
from datetime import datetime
from enum import StrEnum
from dataclasses import dataclass
from uuid import UUID

from elspeth.contracts.freeze import freeze_fields


class InterpretationChoice(StrEnum):
    PENDING = "pending"
    ACCEPTED_AS_DRAFTED = "accepted_as_drafted"
    AMENDED = "amended"
    OPTED_OUT = "opted_out"
    ABANDONED = "abandoned"  # orphan-recovery: PENDING row cleaned by Task 11 job


class InterpretationSource(StrEnum):
    """Structural source of an interpretation event row.

    Closed enum. Adding a value requires (a) amending this plan, (b) extending
    InterpretationSource here, (c) updating the closed-enum tests, and (d) a
    writer-path audit. NO SILENT EXTENSION. See models.py governance block.
    """
    USER_APPROVED = "user_approved"
    AUTO_INTERPRETED_OPT_OUT = "auto_interpreted_opt_out"
    AUTO_INTERPRETED_NO_SURFACES = "auto_interpreted_no_surfaces"


@dataclass(frozen=True, slots=True)
class InterpretationEventRecord:
    """A discrete user decision about an LLM-surfaced interpretation.

    Tier-1 read-side record: every field is required-or-explicitly-None
    per the schema. Constructors crash loudly on any anomaly.

    Two row shapes exist. Fields marked (*) are populated for user-resolved
    rows and NULL for opted_out rows (where no LLM surfacing occurred):

    Fields populated for user-resolved rows (*):
        composition_state_id -> (*) pipeline-state reference
        affected_node_id     -> (*) the LLM-transform node this binds into
        tool_call_id         -> (*) provider tool_call_id from the LLM
        user_term            -> (*) the original user-provided term ("cool")
        llm_draft            -> (*) the LLM's draft interpretation
        accepted_value       -> (*) the user-approved string (None until resolved)
        arguments_hash       -> (*) rfc8785 hash over required fields; None
                                    until resolved (populated at resolve time)

    Fields always present:
        id, session_id, choice, created_at, resolved_at, actor
        model_identifier, model_version, provider, composer_skill_hash,
        interpretation_source

    Per the auditability standard (design doc 06 §"Recording the
    interpretation"), all six of: user_term, llm_draft, accepted_value,
    created_at, actor, composition_state_id are required for user-resolved
    rows. They are intentionally NULL for opted_out rows, where the audit
    trail records the opt-out gesture itself, not a surfacing.

    Audit provenance fields (snapshot what the composer LLM was using):
        model_identifier     -> e.g., "anthropic/claude-opus-4-7"
        model_version        -> provider's reported version string
        provider             -> "anthropic", "openai", etc.
        composer_skill_hash  -> SHA-256 of pipeline_composer.md content

    interpretation_source is the structural mechanism (closed enum) that
    produced the row: USER_APPROVED, AUTO_INTERPRETED_OPT_OUT, or
    AUTO_INTERPRETED_NO_SURFACES.
    """

    id: UUID
    session_id: UUID
    composition_state_id: UUID | None  # None for opted_out rows
    affected_node_id: str | None       # None for opted_out rows
    tool_call_id: str | None           # None for opted_out rows
    user_term: str | None              # None for opted_out rows
    llm_draft: str | None              # None for opted_out rows
    accepted_value: str | None         # None until resolved (or for opted_out)
    choice: InterpretationChoice
    created_at: datetime
    resolved_at: datetime | None
    actor: str                         # user identity at resolution
    model_identifier: str              # e.g., "anthropic/claude-opus-4-7"
    model_version: str                 # provider-reported version string
    provider: str                      # "anthropic", "openai", etc.
    composer_skill_hash: str           # SHA-256 of pipeline_composer.md
    arguments_hash: str | None         # rfc8785 hash over required fields; None until resolved
    interpretation_source: InterpretationSource
```

### Rationale

Opt-out rows are recorded as `InterpretationEventRecord` instances with
`choice=OPTED_OUT` and `interpretation_source=AUTO_INTERPRETED_OPT_OUT`,
with nullable interpretation fields (`composition_state_id`,
`affected_node_id`, `tool_call_id`, `user_term`, `llm_draft`, etc.) set to
`None`. This eliminates the prior `InterpretationOptOutRecord` type and the
associated `proposal_events` routing. A single table is the single source of
truth for all interpretation-related decisions.

The `interpretation_source` field is a closed enum that records the structural
mechanism that produced the row: direct user approval, the opt-out gesture, or
the auto-interpret-no-surfaces path. It lives in `InterpretationSource`.

Audit provenance fields (`model_identifier`, `model_version`, `provider`,
`composer_skill_hash`, `arguments_hash`) snapshot what the composer LLM was
using at surfacing time so a future auditor can re-verify the prompt that
generated the draft interpretation.

`InterpretationEventRecord` has no container fields; all fields are scalars,
`StrEnum`, `datetime`, `UUID`, or `None` — `frozen=True` suffices.

### Test shape

The contract tests assert:

1. `InterpretationChoice` has five values: `pending`, `accepted_as_drafted`,
   `amended`, `opted_out`, `abandoned`.
2. `InterpretationSource` has three values: `user_approved`,
   `auto_interpreted_opt_out`, `auto_interpreted_no_surfaces`.
3. `InterpretationEventRecord` is frozen (assigning a field raises).
4. Constructing with any of the required fields missing is a `TypeError`
   (no defaults).
5. The dataclass round-trips through `dataclasses.asdict()` correctly.
6. Constructing an opted-out row (all nullable fields as `None`,
   `choice=OPTED_OUT`) succeeds.
7. An opted-out row round-trips through `dataclasses.asdict()` correctly with
   all nullable fields as `None`.

### Step 1 — RED

Write the assertions; running pytest fails because the module doesn't exist.

### Step 2 — GREEN

Implement the module exactly per the shape above. `frozen=True` suffices for
`InterpretationEventRecord` (all fields are scalars, enums, datetime, UUID,
or None).

### Step 3 — Commit

Single commit; message: `contracts: add InterpretationEventRecord +
InterpretationChoice + InterpretationSource for Phase 5b`.

---

## Task 2 — Add `interpretation_events_table` to the session audit DB

**Goal.** Add the new SQLAlchemy table to `web/sessions/models.py`,
with all six required-by-spec fields as first-class columns, a closed
enum CHECK constraint on `choice`, composite FK to `composition_states`,
and the partial unique index needed to enforce "one pending
interpretation per (session, tool_call_id)".

Also extend the closed enum on `composition_states.provenance` to add
the value `interpretation_resolve`. This is a closed-enum extension
under the "NO SILENT EXTENSION" governance posture documented at
`models.py` lines `274-289`; the spec update for this is THIS PLAN.

**Files:**

- Modify: `src/elspeth/web/sessions/models.py`.
- Modify: `src/elspeth/web/sessions/schema.py` (validation footprint —
  the schema validator inspects table presence).
- Create: `tests/unit/web/sessions/test_interpretation_events_table.py`.

### Schema definition

Append to `models.py`, after `proposal_events_table`:

```python
interpretation_events_table = Table(
    "interpretation_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    # Composite FK forces same-session ownership: an interpretation
    # event in session B cannot reference a composition state owned by
    # session A. Mirrors the pattern at composition_proposals.
    Column("composition_state_id", String, nullable=False),
    # The LLM transform's node_id within composition_states.nodes that
    # this interpretation binds into. Validated at the writer boundary
    # to exist; NOT a foreign key because nodes live inside a JSON
    # column, not a separate table.
    Column("affected_node_id", String, nullable=False),
    # The provider tool_call_id from the LLM call that surfaced this
    # interpretation. NOT a foreign key to chat_messages because the
    # tool call may still be in flight when this row is inserted.
    Column("tool_call_id", String, nullable=False),
    # Six required-by-spec fields:
    Column("user_term", Text, nullable=False),
    Column("llm_draft", Text, nullable=False),
    Column("accepted_value", Text, nullable=True),  # None until resolved
    Column("choice", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("resolved_at", DateTime(timezone=True), nullable=True),
    Column("actor", String, nullable=False),
    ForeignKeyConstraint(
        ["composition_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_interpretation_events_state_session",
    ),
    UniqueConstraint(
        "id",
        "session_id",
        name="uq_interpretation_events_id_session",
    ),
    # Audit provenance: snapshot of the composer LLM context at surfacing time.
    Column("model_identifier", String, nullable=False),   # e.g., "anthropic/claude-opus-4-7"
    Column("model_version", String, nullable=False),       # provider-reported version string
    Column("provider", String, nullable=False),            # "anthropic", "openai", etc.
    Column("composer_skill_hash", String, nullable=False), # SHA-256 of pipeline_composer.md
    Column("arguments_hash", String, nullable=True),
    # NULL until resolved. For opted_out rows, arguments_hash is NULL
    # because there is no LLM-supplied content to hash: the schema's
    # CHECK constraints (opted_out_nullability, resolved_at_status) already
    # pin the shape of opted_out rows, providing structural integrity without
    # a content hash.
    # Structural source of this row. Closed enum — see governance ceremony below.
    Column("interpretation_source", String, nullable=False),
    # Closed enum on choice. Adding a value requires (a) amending this plan,
    # (b) extending InterpretationChoice in contracts/composer_interpretation.py,
    # (c) updating the closed-enum tests, and (d) a writer-path audit.
    # NO SILENT EXTENSION. See models.py:274-289 governance block.
    CheckConstraint(
        "choice IN ('pending', 'accepted_as_drafted', 'amended', 'opted_out', 'abandoned')",
        name="ck_interpretation_events_choice",
    ),
    # Closed enum on interpretation_source. Adding a value requires the same
    # four-step ceremony as choice. NO SILENT EXTENSION.
    CheckConstraint(
        "interpretation_source IN "
        "('user_approved', 'auto_interpreted_opt_out', 'auto_interpreted_no_surfaces')",
        name="ck_interpretation_events_source",
    ),
    # Opted-out rows: composition_state_id, affected_node_id, tool_call_id,
    # user_term, and llm_draft are all NULL. The CHECK below enforces this.
    CheckConstraint(
        "(choice = 'opted_out') = "
        "(composition_state_id IS NULL AND affected_node_id IS NULL AND "
        " tool_call_id IS NULL AND user_term IS NULL AND llm_draft IS NULL)",
        name="ck_interpretation_events_opted_out_nullability",
    ),
    # If choice is anything other than 'pending', resolved_at MUST be
    # populated. For opted_out rows resolved_at records the opt-out timestamp.
    CheckConstraint(
        "(choice = 'pending') = (resolved_at IS NULL)",
        name="ck_interpretation_events_resolved_at_status",
    ),
    # accepted_value is populated only for accepted_as_drafted and amended.
    CheckConstraint(
        "(choice IN ('accepted_as_drafted', 'amended')) = "
        "(accepted_value IS NOT NULL)",
        name="ck_interpretation_events_accepted_value_status",
    ),
)

# Partial unique index: only one pending interpretation per
# (session_id, tool_call_id). After resolution (choice != 'pending'),
# the same tool_call_id is allowed to recur (which it won't, but the
# index does not need to over-constrain).
Index(
    "uq_interpretation_events_pending_tool_call",
    interpretation_events_table.c.session_id,
    interpretation_events_table.c.tool_call_id,
    unique=True,
    sqlite_where=interpretation_events_table.c.choice == "pending",
    postgresql_where=interpretation_events_table.c.choice == "pending",
)

Index(
    "ix_interpretation_events_session_created",
    interpretation_events_table.c.session_id,
    interpretation_events_table.c.created_at,
)
```

### Closed-enum extension on `composition_states.provenance`

**Deploy constraint.** This enum extension requires the DB delete (per
`project_db_migration_policy`). If the deploy environment cannot delete
the DB (i.e., production with real users), the closed-enum extension is
BLOCKED until the migration runner ships in Phase 9. Staging supports
the delete — Phase 5b can land there immediately. Production deploy
requires sequencing AFTER Phase 9's migration runner. Do not land this
on production before Phase 9.

In the same commit, extend the existing closed enum at `models.py`
line `285-287` to add the new value:

```python
CheckConstraint(
    "provenance IN ('tool_call', 'convergence_persist', 'plugin_crash_persist', "
    "'preflight_persist', 'session_seed', 'session_fork', 'interpretation_resolve')",
    name="ck_composition_states_provenance",
),
```

Update the in-code documentation block at `models.py` lines `274-289`
to record `interpretation_resolve` as the seventh value, with the
writer-path bullet pointing at the new `/resolve` route handler in
`web/sessions/routes.py`. This follows the documented governance
posture; spec amendment is THIS plan, integration test is Task 6, and
the Filigree ticket reference is tracked at PR-open time.

Also extend `CompositionStateProvenance` literal at
`web/sessions/protocol.py` so the writer-side enum stays paired with
the DB-side enum.

### Per-session opt-out: add column to `sessions_table`

The "stop asking for this session" toggle persists as a boolean column
on `sessions_table`. Append at the appropriate place in the column
list, with NOT NULL DEFAULT false:

```python
Column(
    "interpretation_review_disabled",
    Boolean,
    nullable=False,
    server_default=text("false"),
),
```

The opt-out audit event (Task 7) writes to `interpretation_events_table`
with `choice='opted_out'`, `interpretation_source='auto_interpreted_opt_out'`,
all nullable interpretation fields set to NULL, and `resolved_at` set to the
opt-out timestamp. **Do NOT extend `proposal_events.event_type` for opt-out.**
The interpretation_events_table is the single source of truth for all
interpretation-related decisions; routing opt-out through proposal_events would
split the audit trail across two tables and complicate every query that needs
to answer "did the user opt out?" The interpretation_review_disabled boolean
on sessions_table is retained for fast-path read queries by the compose-loop.

### Append-only trigger

To protect the audit integrity of resolved rows, add a SQLite
`BEFORE UPDATE` trigger that rejects mutations to settled fields. The
trigger SQL is:

```sql
CREATE TRIGGER trg_interpretation_events_immutable_resolved
BEFORE UPDATE ON interpretation_events
FOR EACH ROW
BEGIN
  -- Reject any attempt to change accepted_value, resolved_at, actor, or
  -- choice on a row that is already resolved (resolved_at IS NOT NULL).
  -- The only permitted transition is NULL → non-NULL (pending → resolved).
  SELECT CASE
    WHEN OLD.resolved_at IS NOT NULL AND (
      NEW.accepted_value IS NOT OLD.accepted_value OR
      NEW.resolved_at IS NOT OLD.resolved_at OR
      NEW.actor IS NOT OLD.actor OR
      NEW.choice IS NOT OLD.choice
    ) THEN RAISE(ABORT, 'interpretation_events: resolved rows are immutable')
  END;
END;
```

This trigger MUST be created after `CREATE TABLE interpretation_events`
in the same migration or schema bootstrap step. In SQLAlchemy, emit it
via `event.listen(metadata, 'after_create', DDL(...))` or run it in the
schema bootstrap step alongside `metadata.create_all()`.

Document the trigger in the `models.py` governance block alongside the
table definition.

### Test shape

`tests/unit/web/sessions/test_interpretation_events_table.py`:

1. Test that `metadata.create_all(engine)` succeeds against an in-memory
   SQLite engine and produces a table named `interpretation_events`
   with all the expected columns (including `model_identifier`,
   `model_version`, `provider`, `composer_skill_hash`, `arguments_hash`,
   `interpretation_source`).
2. Test that inserting a row with `choice='pending'` and `resolved_at`
   set raises `IntegrityError`.
3. Test that inserting a row with `choice='accepted_as_drafted'` and
   `accepted_value=NULL` raises `IntegrityError`.
4. Test that inserting a row with `choice='opted_out'` and all nullable
   fields (`composition_state_id`, `affected_node_id`, `tool_call_id`,
   `user_term`, `llm_draft`) set to NULL, and
   `interpretation_source='auto_interpreted_opt_out'` succeeds.
4a. Test that inserting an opted_out row with `composition_state_id`
    non-NULL raises `IntegrityError`
    (ck_interpretation_events_opted_out_nullability).
5. Test that two PENDING rows with the same `(session_id, tool_call_id)`
   tuple raises `IntegrityError` (the partial unique index).
5a. (Positive case) Test that two RESOLVED rows with the same
    `(session_id, tool_call_id)` are allowed — create two rows both
    with `resolved_at` populated and `choice` set to a non-pending
    value, assert both inserts succeed. This is the positive-case test
    for the partial index: the index ONLY prohibits the unresolved-
    duplicate case; resolved duplicates (rare but permitted) must not
    be blocked.
6. Test that two RESOLVED rows with the same `(session_id, tool_call_id)`
   are allowed (the partial unique index excludes them).
7. Test that the FK to `composition_states` is enforced for non-opted-out
   rows (insert a row with a non-existent `composition_state_id` and
   `choice != 'opted_out'`; expect IntegrityError).
8. Test that the closed enum on `composition_states.provenance` rejects
   any value not in the seven listed.
9. Test that `sessions_table` now has the new
   `interpretation_review_disabled` boolean column with default `false`.
10. **Append-only trigger tests:**
    a. After inserting a resolved row (`resolved_at` IS NOT NULL,
       `choice='accepted_as_drafted'`), attempting to UPDATE
       `accepted_value` raises `IntegrityError` (the trigger fires).
    b. After inserting a resolved row, attempting to flip `choice` from
       `accepted_as_drafted` back to `pending` raises `IntegrityError`.
    c. Updating a PENDING row's non-settled fields (e.g. `model_version`)
       does NOT raise (trigger only guards resolved rows).

### Step 1 — RED

Run the test file; all cases fail because the table doesn't exist
and the column hasn't been added.

### Step 2 — GREEN

Apply the schema edits exactly as above. Confirm RED → GREEN.

### Step 3 — Confirm schema-validation gate

`web/sessions/schema.py` calls `_validate_current_schema` which asserts
the live engine's table set matches `metadata`. The new table appears
in `metadata` automatically once added; no explicit change to
`schema.py` is needed unless the validator inspects column lists
explicitly (it does; verify by reading the function and adjusting the
expected-column lists if the validator enumerates them per-table). If
the validator enumerates columns, append the new ones with the same
nullable/index attributes.

### Step 4 — Commit

Single commit. Message: `sessions(schema): add interpretation_events_table
+ interpretation_resolve provenance + interpretation_review_disabled column
+ append-only trigger`. The commit message MUST mention all four schema
changes: the new table, the provenance enum extension, the sessions boolean
column, and the immutability trigger. NOTE: `interpretation.opted_out` is
NOT added to `proposal_events.event_type` — opt-out rows go to
`interpretation_events_table` (see opt-out pivot above).

---

## Task 3 — Define wire schemas (pydantic models)

**Goal.** Add the pydantic wire models that mirror the new contract
types and the new HTTP route bodies/responses. Source-of-truth pairing
follows the project convention: contract dataclass in `contracts/`,
pydantic wire schema in `web/sessions/schemas.py`.

**Files:**

- Modify: `src/elspeth/web/sessions/protocol.py` — extend
  `CompositionStateProvenance` literal.
- Modify: `src/elspeth/web/sessions/schemas.py` — add new pydantic
  models.
- Create: `tests/unit/web/sessions/test_interpretation_schemas.py`.

### Schemas to add

```python
# Mirrors InterpretationEventRecord. The contract is the read-side
# type; this is the wire-side type.
class InterpretationEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: UUID
    session_id: UUID
    composition_state_id: UUID | None = None  # None for opted_out rows
    affected_node_id: str | None = Field(default=None, max_length=256)
    tool_call_id: str | None = Field(default=None, max_length=256)
    user_term: str | None = Field(default=None, max_length=8192)
    llm_draft: str | None = Field(default=None, max_length=8192)
    accepted_value: str | None = Field(default=None, max_length=8192)
    choice: Literal[
        "pending", "accepted_as_drafted", "amended", "opted_out", "abandoned"
    ]
    created_at: datetime
    resolved_at: datetime | None = None
    actor: str = Field(min_length=1, max_length=256)
    interpretation_source: Literal[
        "user_approved", "auto_interpreted_opt_out", "auto_interpreted_no_surfaces"
    ]
    # Audit-provenance fields — bound to which LLM produced the draft.
    # Exposed on the wire so the audit-readiness panel and any future
    # reviewer surface can render "drafted by claude-opus-4-7 v… on
    # 2026-05-18" without a second DB round-trip. None for opted_out
    # rows (no LLM was consulted).
    model_identifier: str | None = Field(default=None, max_length=256)
    model_version: str | None = Field(default=None, max_length=128)
    provider: str | None = Field(default=None, max_length=64)
    composer_skill_hash: str | None = Field(default=None, max_length=64)  # hex SHA-256
    arguments_hash: str | None = Field(default=None, max_length=64)  # hex rfc8785-canonical hash; populated at resolve time


# Request body for POST /api/sessions/{id}/interpretations/{event_id}/resolve.
class InterpretationResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Choice is restricted to the two user-facing values; "opted_out"
    # goes through the separate /opt_out route, and "pending" is not
    # a valid resolution.
    choice: Literal["accepted_as_drafted", "amended"]
    # Only required (and only permitted) when choice == "amended".
    amended_value: str | None = Field(default=None, max_length=8192)

    @model_validator(mode="after")
    def _amended_value_consistency(self) -> "InterpretationResolveRequest":
        if self.choice == "amended" and not self.amended_value:
            raise ValueError(
                "amended_value is required when choice == 'amended'"
            )
        if self.choice == "accepted_as_drafted" and self.amended_value is not None:
            raise ValueError(
                "amended_value must be omitted when choice == 'accepted_as_drafted'"
            )
        return self

    @field_validator("amended_value", mode="after")
    @classmethod
    def _validate_amended_value_content(cls, v: str | None) -> str | None:
        """Reject template metacharacters, control chars, credential-shaped content,
        and overlength strings.

        Raises ValueError (422) with explicit per-condition messages so the
        caller receives actionable feedback.
        """
        if v is None:
            return v
        # Reject Jinja-style template metacharacters that could break
        # placeholder substitution or downstream template rendering.
        if "{{" in v or "}}" in v:
            raise ValueError(
                "accepted_value must not contain template metacharacters {{ or }}"
            )
        # Reject control characters except horizontal tab (\t). Newlines (\n)
        # are also rejected: prompt-template values are expected to be
        # single-line phrases; multi-line user input is a scope escalation.
        import re
        if re.search(r"[\x00-\x08\x0a-\x1f\x7f]", v):
            raise ValueError(
                "accepted_value must not contain control characters "
                "(newlines and non-printable characters are not permitted)"
            )
        # Soft length cap: 8 KiB total already enforced by max_length=8192;
        # single-line values should also respect a per-line cap.
        if len(v) > 1024 and "\n" not in v:
            # More than 1024 chars and still single-line is pathological.
            raise ValueError(
                "accepted_value exceeds the single-line 1024-character limit"
            )
        # Credential-shaped content prefilter (see also Task 5 chat-input boundary).
        _reject_credential_shaped_content(v)  # raises ValueError on match
        return v


# Response is the resolved event PLUS the new composition state that
# was produced by patching the affected LLM transform.
class InterpretationResolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event: InterpretationEventResponse
    new_state: CompositionStateResponse  # already exists in schemas.py


# POST /api/sessions/{id}/interpretations/opt_out has no body fields
# beyond actor (carried by the auth middleware). Response is the
# session record with the new flag set.
class InterpretationOptOutResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    session_id: UUID
    interpretation_review_disabled: bool
    opted_out_at: datetime


# GET /api/sessions/{id}/interpretations?status=pending|all
class ListInterpretationEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    events: list[InterpretationEventResponse]
```

### Step 1 — RED

`test_interpretation_schemas.py` exercises:
1. Constructing each model with all required fields succeeds.
2. Each model rejects unknown extra fields (`extra="forbid"`).
3. `InterpretationResolveRequest` rejects `choice="amended"` without
   `amended_value`.
4. `InterpretationResolveRequest` rejects `choice="accepted_as_drafted"`
   with `amended_value` present.
5. The length cap rejects 8193-char strings.
6. `InterpretationResolveRequest` rejects `amended_value` containing `{{`
   (template metachar check).
7. `InterpretationResolveRequest` rejects `amended_value` containing `\n`
   (control-char check).
8. `InterpretationResolveRequest` rejects `amended_value` containing an
   AWS access-key-shaped string (`AKIAxxx...`).
9. `InterpretationResolveRequest` rejects `amended_value` containing a
   Bearer token-shaped string.
10. `InterpretationResolveRequest` accepts a clean single-line phrase.

### Step 2 — GREEN

Implement. `_reject_credential_shaped_content` lives in a NEW shared
module `src/elspeth/web/_validation_helpers.py` (imported by both
`schemas.py` and `tools.py`). Creating a new module avoids a peer
cross-import between `web/sessions/schemas.py` and
`web/composer/tools.py`. The helper runs the credential prefilter
regexes listed in Task 5's "Credential-shape prefilter" section and
raises `ValueError` with a user-visible message ("That looks like a
credential — please re-enter without secrets") on any match. Import
with `from elspeth.web._validation_helpers import _reject_credential_shaped_content`.

### Step 3 — Commit

`sessions(schemas): wire schemas for interpretation events + resolve +
opt-out + amended_value content validators`.

---

## Task 4 — Session service methods

**Goal.** Add the writer/reader methods that drive the new table.
Method-naming and transaction-shape mirror `create_composition_proposal`
/ `list_proposal_events` from `service.py`.

**Files:**

- Modify: `src/elspeth/web/sessions/service.py`.
- Create: `tests/unit/web/sessions/test_interpretation_events_service.py`.

### Telemetry policy (standing — applies to every method in this Task)

Composition-time user decisions are audit-primary. The Landscape
`interpretation_events_table` is the source of truth; there is no ephemeral
operational signal worth emitting to telemetry for these writes. Each method
below carries the declaration:

> Telemetry: NONE — composition-time user decisions are audit-primary;
> no ephemeral operational signal required.

### Methods

```python
async def create_pending_interpretation_event(
    self,
    *,
    session_id: UUID,
    composition_state_id: UUID,
    affected_node_id: str,
    tool_call_id: str,
    user_term: str,
    llm_draft: str,
    model_identifier: str,       # e.g., "anthropic/claude-opus-4-7"
    model_version: str,          # provider-reported version string
    provider: str,               # "anthropic", "openai", etc.
    composer_skill_hash: str,    # SHA-256 of pipeline_composer.md
    created_at: datetime | None = None,
) -> InterpretationEventRecord:
    """Insert a PENDING interpretation event.

    Called from the compose-loop tool handler for
    request_interpretation_review. Acquires the session write lock for
    the duration of the insert. Validates affected_node_id exists in
    composition_states.nodes BEFORE committing the row (raises
    ValueError otherwise; the tool handler converts to ARG_ERROR).

    Telemetry: NONE — composition-time user decisions are audit-primary;
    no ephemeral operational signal required.
    """


async def resolve_interpretation_event(
    self,
    *,
    session_id: UUID,
    event_id: UUID,
    choice: InterpretationChoice,  # accepted_as_drafted or amended
    accepted_value: str,  # the resolved string (== llm_draft if accepted)
    actor: str,
    resolved_at: datetime | None = None,
) -> tuple[InterpretationEventRecord, CompositionStateRecord]:
    """Commit a resolution AND patch the affected LLM transform's
    prompt template.

    Single transaction:
        1. UPDATE interpretation_events SET choice=?, accepted_value=?,
           resolved_at=?, actor=?, arguments_hash=?
           WHERE id=? AND choice='pending'.
           (CHECK constraint enforces consistency; the WHERE on
           choice='pending' is a TOCTOU guard against double-resolve.
           arguments_hash is the rfc8785 canonical hash over all required
           fields once all final values are known.)
        2. Read the affected composition state, locate the LLM
           transform node by affected_node_id, patch the prompt
           template's interpretation placeholder with accepted_value.
           Produce a new composition_states row with provenance =
           'interpretation_resolve', version += 1.
        3. Return the resolved event + the new state.

    Acquires the session write lock. Raises ValueError if the event
    is already resolved (TOCTOU lost), if the affected node has
    disappeared from the composition state since the surfacing, or
    if the prompt-template patch fails (e.g., node is no longer an
    LLM transform).

    Telemetry: NONE — composition-time user decisions are audit-primary;
    no ephemeral operational signal required.
    """


async def list_interpretation_events(
    self,
    session_id: UUID,
    *,
    status: Literal["pending", "all"] = "all",
    composition_state_id: UUID | None = None,
) -> list[InterpretationEventRecord]:
    """Read-back of interpretation events for the session.

    Used by the audit-readiness panel (counts) and by the frontend
    on reload (rehydrate pending review affordances).

    Telemetry: NONE — composition-time user decisions are audit-primary;
    no ephemeral operational signal required.
    """


async def record_session_interpretation_opt_out(
    self,
    *,
    session_id: UUID,
    actor: str,
    opted_out_at: datetime | None = None,
) -> InterpretationEventRecord:
    """Mark the session as 'don't surface interpretations any more'.

    Writes a row to interpretation_events_table with choice='opted_out',
    interpretation_source='auto_interpreted_opt_out', all nullable
    interpretation fields (composition_state_id, affected_node_id,
    tool_call_id, user_term, llm_draft, accepted_value, arguments_hash)
    set to NULL, and resolved_at set to opted_out_at.
    Also sets sessions.interpretation_review_disabled = true.
    Single transaction.

    Does NOT write to proposal_events_table. The interpretation_events
    table is the single source of truth for all interpretation-related
    decisions.

    Telemetry: NONE — composition-time user decisions are audit-primary;
    no ephemeral operational signal required.
    """
```

### Prompt-template patch helper

The patching logic in `resolve_interpretation_event` is non-trivial.
Extract a private helper:

```python
def _patch_llm_transform_prompt(
    state: CompositionStateRecord,
    *,
    affected_node_id: str,
    user_term: str,
    accepted_value: str,
) -> Mapping[str, Any]:
    """Return a new `nodes` JSON object with the LLM transform's prompt
    template patched to embed `accepted_value` for `user_term`.

    The patch convention: the LLM transform's prompt_template field
    contains a placeholder of the form `{{interpretation:<term>}}` that
    the LLM writes when it first stages the LLM transform. This helper
    substitutes the placeholder with the user's accepted_value.

    Raises ValueError if the affected node:
      - is not present in state.nodes
      - is not an LLM transform plugin (kind != 'llm')
      - has no prompt_template field
      - has a prompt_template that does not contain the expected
        placeholder for `user_term`
      - the placeholder appears more than once in the template
      - the prefix immediately before the placeholder matches (case-insensitive)
        any of the instruction-key patterns: "system:", "role:", "instructions:"
        (these indicate the LLM placed the placeholder inside a structural
        directive rather than in the prompt body, which would produce a broken
        prompt after substitution)
    """
```

**Design note:** the placeholder convention is part of the
composer-skill contract (Task 8). The LLM is instructed to emit
`{{interpretation:<term>}}` in the prompt template *at the same time*
as it calls `request_interpretation_review`. The placeholder is the
"hole" waiting for the user's resolved value. If the LLM tries to
finalise the prompt template without leaving the hole, the
interpretation flow cannot patch it — that case raises ValueError →
ARG_ERROR back to the LLM with a recovery prompt.

An alternative convention (storing the prompt template separately on
the interpretation event row and recomposing at runtime) was
considered and rejected: it would require runtime code changes and
break the "the runtime is just executing its committed prompt
template" property. This architecture is contingent on Task 0 passing
(see Task 0 gate).

### Test shape

`tests/unit/web/sessions/test_interpretation_events_service.py`:

1. `create_pending_interpretation_event` inserts a row with all six
   required fields, returns the record. Spot-check the DB row directly.
2. `create_pending_interpretation_event` raises `RuntimeError` (or
   equivalent writer-boundary error) when `affected_node_id` references
   a node that does not exist in the parent `composition_states` row.
   The test must: (a) create a `composition_states` row with one node
   `node-A`, (b) call `create_pending_interpretation_event` with
   `affected_node_id="node-does-not-exist"`, (c) assert the call raises
   before any DB write (the `interpretation_events` table must be empty
   after the raise). This is the writer-boundary validation test required
   per project CLAUDE.md offensive-programming rules.
3. `resolve_interpretation_event` with `choice='accepted_as_drafted'`
   sets `accepted_value = llm_draft`, advances state version,
   patches the prompt template's placeholder. Spot-check the DB row.
4. `resolve_interpretation_event` with `choice='amended'` sets
   `accepted_value = amended_value`, same downstream effects.
5. `resolve_interpretation_event` raises ValueError on double-resolve
   (TOCTOU guard via WHERE choice='pending').
6. `resolve_interpretation_event` raises ValueError when the affected
   node has been removed from the composition state since surfacing.
7. `resolve_interpretation_event` raises ValueError when the LLM
   transform's prompt template doesn't contain the expected
   `{{interpretation:<term>}}` placeholder.
8. `list_interpretation_events` with `status='pending'` returns only
   pending rows.
9. `list_interpretation_events` with `status='all'` returns all rows.
10. `record_session_interpretation_opt_out` sets the boolean on
    `sessions_table` and writes a row to `interpretation_events_table`
    with `choice='opted_out'`, `interpretation_source='auto_interpreted_opt_out'`,
    all nullable interpretation fields NULL, and `resolved_at` set.
    The `proposal_events_table` row count for the session MUST NOT change
    across this call (snapshot count before, snapshot count after,
    assert equal — regression guard against the prior proposal_events
    routing; uses delta rather than `== 0` to avoid false positives when
    fixtures pre-seed unrelated proposal events).
11. `_patch_llm_transform_prompt` raises ValueError when the placeholder
    appears more than once in the template.
12. `_patch_llm_transform_prompt` raises ValueError when the prefix before
    the placeholder matches "system:" (case-insensitive position check).
13. `_patch_llm_transform_prompt` raises ValueError when the prefix matches
    "role:" (case-insensitive position check).
14. `_patch_llm_transform_prompt` succeeds with a clean template containing
    exactly one placeholder in a normal prompt body position.

### Step 1 — RED

Tests fail; methods don't exist.

### Step 2 — GREEN

Implement methods in `service.py`. Each writer uses
`_session_write_lock(conn, sid)`. Use the same `_run_sync(_sync)`
wrapper pattern as elsewhere in the file.

### Step 3 — Commit

`sessions(service): writer + reader methods for interpretation events
and opt-out (opt-out routes to interpretation_events, not proposal_events)`.

---

## Task 5 — Composer tool `request_interpretation_review`

**Goal.** Add a new LLM-callable tool that surfaces the
interpretation for user review. The tool:

1. Validates the LLM's draft and the affected_node_id.
2. Calls `create_pending_interpretation_event` to persist the pending
   row.
3. Returns a `ToolResult` whose payload tells the frontend an
   interpretation is awaiting user review.

**Files:**

- Modify: `src/elspeth/web/composer/tools.py`.
- Modify: `src/elspeth/web/composer/redaction.py`.
- Modify: `src/elspeth/web/composer/proposals.py`.
- Modify: `src/elspeth/web/composer/service.py`.
- Create: `tests/unit/web/composer/test_request_interpretation_review_tool.py`.
- Create: `tests/unit/web/composer/test_request_interpretation_review_redaction.py`.

### Tool argument model (pydantic, Tier-3 boundary)

In `tools.py`, alongside the existing `_UpsertNodeArgumentsModel`:

```python
class _RequestInterpretationReviewArgumentsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    affected_node_id: str = Field(min_length=1, max_length=256)
    user_term: str = Field(min_length=1, max_length=8192)
    llm_draft: str = Field(min_length=1, max_length=8192)
```

### Tool registration

Append to `get_tool_definitions()` (OpenAI function spec). The tool's
description text is critical — it instructs the LLM when to call. The
description is normative documentation for the LLM and is reviewed by
the composer-skill prompt (Task 8):

```python
{
    "type": "function",
    "function": {
        "name": "request_interpretation_review",
        "description": (
            "Ask the user to review your interpretation of a subjective or "
            "underspecified term they used. Call this BEFORE you finalise "
            "the prompt template for any LLM transform whose prompt depends "
            "on the term. Surface ONE term per call. The composition state "
            "MUST already contain the affected LLM transform (call upsert_node "
            "first) and its prompt_template MUST contain the placeholder "
            "{{interpretation:<term>}}. The user will see your draft and "
            "either accept it or amend it. Do not call this for concrete "
            "operators (e.g., 'rate 1-10') or for terms the user already "
            "defined in the conversation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["affected_node_id", "user_term", "llm_draft"],
            "properties": {
                "affected_node_id": {
                    "type": "string",
                    "description": "node_id of the LLM transform whose prompt template depends on this term",
                },
                "user_term": {
                    "type": "string",
                    "description": "The user-provided term, verbatim (e.g., 'cool', 'important', 'risky')",
                },
                "llm_draft": {
                    "type": "string",
                    "description": "Your draft interpretation of the term, in your own words, suitable to embed as a phrase in the prompt template",
                },
            },
        },
    },
},
```

### Pre-Task spike: dispatch shape decision

**Complete this spike before writing any handler code. The pattern set here
propagates to all subsequent session-aware tools.**

The current `ToolHandler` type alias at `tools.py:1599-1602` is:

```python
ToolHandler = Callable[
    [dict[str, Any], CompositionState, CatalogService, str | None],
    ToolResult,
]
```

This handler is synchronous and session-unaware. The new
`_handle_request_interpretation_review` handler needs:
- to be `async` (it awaits `create_pending_interpretation_event`)
- additional inputs: `session_id: UUID`, `tool_call_id: str`, and
  the service method as a callable

Three options exist:

**Option A — Extend `ToolHandler` to a richer, async signature.**
Change `ToolHandler` to `Callable[..., Awaitable[ToolResult]]` and add
session context parameters. This requires updating every existing
handler (12+ handlers at `tools.py:1691–1803`) to accept and ignore the
new parameters, and every dispatch site to `await` the result.
Blast radius: large, mechanical. Risk: a handler silently ignoring
`session_id` when it should use it becomes invisible.

**Option B — Introduce a parallel `SessionAwareToolHandler` type with
its own registry.**
Define:
```python
SessionAwareToolHandler = Callable[
    [dict[str, Any], CompositionState, CatalogService, str | None,
     UUID, str, "SessionService"],
    Awaitable[ToolResult],
]
```
Register session-aware tools in a separate dict (e.g.
`_SESSION_AWARE_TOOL_HANDLERS`). The dispatch loop checks both
registries; session-aware tools are awaited, state-pure tools are
called synchronously. New handlers opt in; existing handlers are
unaffected.
Blast radius: localised. The dispatch loop is the only shared site.

**Option C — Closure capture.**
When registering the handler, build a partial/closure that captures
`session_id`, `tool_call_id`, and the service reference. The handler
retains the existing `ToolHandler` signature. However, this does NOT
solve the async problem: a closure that captures async dependencies
and is itself synchronous still cannot `await` them without an event
loop. Option C addresses session-context plumbing only, not the async
fix required here.

**Recommendation: Option B.**
It avoids the blast radius of Option A (no changes to existing handlers
or their callers), it makes the async / session-awareness opt-in
explicit per handler, and it doesn't inherit Option C's async
incompatibility. The dispatch loop extension is a single focused
change. Document this choice with a comment at the registration site;
add a note to `tools.py:1597` explaining the two registries and the
invariant ("state-pure tools are synchronous; session-aware tools are
async and registered in `_SESSION_AWARE_TOOL_HANDLERS`").

This is the first session-aware tool in the project. Getting the
pattern right now avoids a wave of structural debt if later tools
(Phase 6, Phase 7) also need session context.

### Credential-shape prefilter (chat input boundary)

Before the tool handler validates the LLM's arguments, apply a
credential-shape prefilter to `user_term` and `llm_draft`. The same
regex set used in Task 3's `_reject_credential_shaped_content` applies
here. If either field matches:
- Return ARG_ERROR (not a tool exception — use the `_mutation_error`
  helper pattern) with the message
  "That looks like a credential — please re-enter without secrets."
- Do NOT persist the event row.

The prefilter rejects content matching:
- AWS access keys: `AKIA[0-9A-Z]{16}`
- Bearer tokens: `Bearer\s+[A-Za-z0-9._-]{20,}`
- GitHub PATs: `ghp_[A-Za-z0-9]{36}`
- Anthropic keys: `sk-ant-[A-Za-z0-9_-]{40,}`
- OpenAI keys: `sk-[A-Za-z0-9]{40,}` (including `sk-proj-...` prefix)
- JWTs: three base64url segments separated by `.` (pattern:
  `[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}`)
- SSN: `\d{3}-\d{2}-\d{4}`
- Credit-card-shaped: `\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}` — apply
  LUHN check before flagging (avoid false positives on date-like strings)

The shared regex set lives in
`src/elspeth/web/_validation_helpers.py:_reject_credential_shaped_content`
(created in Task 3 Step 2). Import with:
`from elspeth.web._validation_helpers import _reject_credential_shaped_content`.
This avoids a peer cross-import between `web/sessions/schemas.py` and
`web/composer/tools.py`.

### Per-session rate limit

Add to the handler logic (before the DB insert):

```python
await _check_interpretation_rate_limits(
    session_id=session_id,
    user_term=parsed.user_term,
    composition_state_id=state.id,
    list_events_fn=list_interpretation_events,
)
```

`_check_interpretation_rate_limits` MUST be declared `async def` because
its `list_events_fn` argument (an injected `list_interpretation_events`
service method) is itself async — it reads the session DB. Defining the
helper as sync would force a blocking `asyncio.run(...)` inside an
already-async handler, which deadlocks. Spec contract:

```python
async def _check_interpretation_rate_limits(
    *,
    session_id: UUID,
    user_term: str,
    composition_state_id: UUID,
    list_events_fn: Callable[..., Awaitable[list[InterpretationEventRecord]]],
) -> None: ...
```

Where `_check_interpretation_rate_limits` enforces:
- Max 3 surfacings of the same `(session_id, user_term)` per
  composition (same `composition_state_id` branch). On cap exceeded:
  raise `ToolArgumentError("request_interpretation_review: too many
  interpretation requests for term '{user_term}' in this session (max
  3). Use a direct interpretation in the prompt template instead.")`.
- Max 10 total `request_interpretation_review` invocations per session
  per day. On cap exceeded: raise `ToolArgumentError("Too many
  interpretation requests in this session today (max 10). The compose
  loop should fall back to auto-interpretation.")`.

On cap exceeded, the compose loop sees an ARG_ERROR and is expected
(per Task 8 skill nudge) to fall back to a non-LLM interpretation with
`interpretation_source='auto_interpreted_no_surfaces'`.

### Tool handler

In `tools.py`, alongside `_handle_upsert_node`, registered in
`_SESSION_AWARE_TOOL_HANDLERS` (see dispatch spike above):

```python
async def _handle_request_interpretation_review(
    state: CompositionState,
    arguments: object,
    *,
    session_id: UUID,
    tool_call_id: str,
    create_pending_interpretation_event: Callable[..., Awaitable[InterpretationEventRecord]],
    list_interpretation_events: Callable[..., Awaitable[list[InterpretationEventRecord]]],
) -> ToolResult:
    """Stage a pending interpretation event for user review.

    Returns a SUCCESS ToolResult whose payload signals the frontend to
    surface the review affordance. Does NOT advance composition state
    version (state changes happen at /resolve time).

    Async because it awaits create_pending_interpretation_event.
    Registered in _SESSION_AWARE_TOOL_HANDLERS (not ToolHandler registry).
    """
    parsed = _validate_mutation_arguments(
        _RequestInterpretationReviewArgumentsModel,
        arguments,
        "request_interpretation_review",
    )
    # Credential prefilter: Tier-3 boundary check before any DB write.
    _reject_credential_shaped_content(parsed.user_term)
    _reject_credential_shaped_content(parsed.llm_draft)
    # Per-session rate limits.
    await _check_interpretation_rate_limits(
        session_id=session_id,
        user_term=parsed.user_term,
        composition_state_id=state.id,
        list_events_fn=list_interpretation_events,
    )
    # Verify the affected node exists and is an LLM transform with a
    # prompt_template that contains the expected placeholder. This is
    # the Tier-3 boundary check on the LLM-provided affected_node_id.
    _assert_affected_llm_node(state, parsed.affected_node_id, parsed.user_term)
    # Persist the pending row (caller injects the service method).
    event = await create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=state.id,
        affected_node_id=parsed.affected_node_id,
        tool_call_id=tool_call_id,
        user_term=parsed.user_term,
        llm_draft=parsed.llm_draft,
        # model/provider/skill fields supplied by the caller closure from
        # the compose-loop's current LLM config snapshot
    )
    return ToolResult(
        success=True,
        updated_state=state,  # no state change yet
        data={
            "_kind": "interpretation_review_pending",
            "event_id": str(event.id),
            "affected_node_id": parsed.affected_node_id,
            "user_term": parsed.user_term,
            "llm_draft": parsed.llm_draft,
        },
        message=(
            f"Interpretation review staged for '{parsed.user_term}'. "
            f"Waiting for user acceptance/amendment before the pipeline can finalise."
        ),
    )
```

The `_assert_affected_llm_node` helper enforces:
- the node exists in `state.nodes`
- the node's plugin kind is `'llm'`
- the node has a `prompt_template` field containing the placeholder
  `{{interpretation:<user_term>}}`

If any of these fail, raise `ToolArgumentError("request_interpretation_review:
<reason>")` so the LLM sees a recoverable error and can re-stage the
tool call after fixing the prompt template.

### Redaction model

In `redaction.py`, alongside the existing per-tool argument models:

`_summarize_interpretation_term` is a NEW private helper defined in
`redaction.py` as part of this task (not an existing symbol). Following
the naming convention at `redaction.py:825` which uses American spelling
(`_summarize_inline_blob_content`), use American spelling here too:

```python
def _summarize_interpretation_term(text: str) -> str:
    """Summarizer for user_term and llm_draft fields.

    Truncates to 64 chars and appends '…' if truncated. Designed for
    term-shaped values (e.g. 'cool', 'very important to the user's goal').
    The cap prevents PII in longer term strings from persisting in the
    audit trail's tool-call column.

    Contract: MUST NOT raise on any reachable input. MUST return str.
    """
    if len(text) > 64:
        return text[:64] + "…"
    return text


class _RequestInterpretationReviewRedactionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    affected_node_id: str
    # user_term is NOT a secret — it's a word the user typed. But it
    # could carry PII (e.g., user types "rate how cool this transaction
    # involving John Doe is"); apply summariser cap to be safe.
    user_term: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
    llm_draft: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
```

Register the model alongside the per-tool redaction map.

### Proposal summary

In `proposals.py`, extend `build_tool_proposal_summary`:

```python
if tool_name == "request_interpretation_review":
    term = _string_argument(arguments, "user_term") or "term"
    return ToolProposalSummary(
        summary=f'Surface the interpretation of "{term}" for user review.',
        rationale=(
            "The term is subjective or underspecified; the user should "
            "review the LLM's draft interpretation before the prompt "
            "template is finalised."
        ),
        affects=("interpretation",),  # NOT 'graph' / 'validation' / 'yaml'
        arguments_redacted_json=redacted_arguments,
    )
```

**Note:** `affects=("interpretation",)` is a NEW value. Check
whether any frontend code switches on the closed list of `affects`
values (search `web/frontend/src/`). If so, extend the literal there
too; if not, the value flows through to the frontend untyped. (Phase
5b's frontend plan covers this.)

### Compose-loop hook

In `web/composer/service.py`, the compose loop's tool dispatch
already handles the standard set of tools generically. The new tool
is session-aware and async; it is registered in `_SESSION_AWARE_TOOL_HANDLERS`
(see dispatch spike above) rather than the existing `ToolHandler` registry.
The dispatch loop must be extended to check `_SESSION_AWARE_TOOL_HANDLERS`
and await handlers found there, passing `session_id`, `tool_call_id`, and
the injected service methods.

The compose-loop closure captures the current LLM config snapshot
(`model_identifier`, `model_version`, `provider`, `composer_skill_hash`)
and passes these as keyword arguments to
`create_pending_interpretation_event` so the audit record is complete
at insert time.

### Test shape

`test_request_interpretation_review_tool.py`:

1. Tool registered in `get_tool_definitions()` with the expected
   JSON-schema shape.
2. Calling the handler with a valid `affected_node_id` pointing at an
   LLM transform with the correct placeholder produces a SUCCESS
   ToolResult with payload `_kind='interpretation_review_pending'` and
   the pending event row exists in the DB.
3. Calling the handler when `affected_node_id` doesn't exist raises
   `ToolArgumentError` (ARG_ERROR path).
4. Calling the handler when the affected node is NOT an LLM transform
   raises `ToolArgumentError`.
5. Calling the handler when the affected node's prompt_template has
   no `{{interpretation:<term>}}` placeholder raises `ToolArgumentError`.
6. Calling the handler with `user_term` that exceeds 8192 chars raises
   `ToolArgumentError` (Pydantic validation).
7. The proposal summary builder returns the expected
   `"Surface the interpretation of …"` text.
8. **Rate-limit test:** Calling the handler 4 times with the same
   `(session_id, user_term)` pair raises `ToolArgumentError` on the
   4th call (per-term cap = 3).
9. **Rate-limit test:** Calling the handler 11 times in a session
   (with distinct `user_term` values to avoid the per-term cap) raises
   `ToolArgumentError` on the 11th call (per-session-day cap = 10).
10. **Credential prefilter:** Calling with `user_term` containing an
    AWS access key pattern raises `ToolArgumentError` before any DB
    write (table remains empty after the raise).
11. **Credential prefilter:** Calling with `llm_draft` containing a
    Bearer token pattern raises `ToolArgumentError`.

`test_request_interpretation_review_redaction.py`:

1. Redaction model accepts a valid argument dict.
2. Redaction model produces redacted output with `user_term` and
   `llm_draft` summarised (truncated to ≤64 chars + `…`).
3. Audit envelope canonicalises the redacted form (Tier-1 row should
   carry the redacted shape, not the raw user content).

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

Single commit per file group:
- `composer(tools): add request_interpretation_review handler + Pydantic args`
- `composer(redaction): redaction model for request_interpretation_review`
- `composer(proposals): proposal-summary handler for request_interpretation_review`
- `composer(service): wire request_interpretation_review into compose loop dispatch`

(Four commits if commit-per-file; ONE commit if the tool-surface is
treated as a wire-contract change per project conventions §"Convention
14: one-commit wire-contract changes". The latter is preferred for
review locality.)

---

## Task 6 — HTTP route `POST /api/sessions/{id}/interpretations/{event_id}/resolve`

**Goal.** The user-side acceptance path. Accepts the
`InterpretationResolveRequest` body, calls
`resolve_interpretation_event`, and returns the
`InterpretationResolveResponse`.

**Files:**

- Modify: `src/elspeth/web/sessions/routes.py`.
- Create: `tests/unit/web/sessions/test_interpretation_events_routes.py`.

### Route shape

> **DI pattern note:** This project uses `user: UserIdentity = Depends(get_current_user)`
> and `service = request.app.state.session_service` (not `Depends(get_session_service)`
> or `Depends(get_authenticated_actor)` — those symbols do not exist). Actor is
> extracted as `f"user:{user.user_id}"` per the convention in routes.py.
> Session ownership is verified via the local `_verify_session_ownership` wrapper
> (routes.py:1323), which returns `SessionRecord` and raises 404 on IDOR.

```python
@router.post(
    "/{session_id}/interpretations/{event_id}/resolve",
    response_model=InterpretationResolveResponse,
)
async def resolve_interpretation(
    session_id: UUID,
    event_id: UUID,
    body: InterpretationResolveRequest,
    raw_request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> InterpretationResolveResponse:
    """User-driven resolve of a pending interpretation event.

    Tier-3 boundary: request body validated by Pydantic; choice/amended_value
    consistency enforced by the model's validator. The service method
    enforces semantic constraints (event must be pending; node must still
    exist; prompt-template patch must succeed).
    """
    await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
    service: SessionServiceProtocol = raw_request.app.state.session_service
    actor = f"user:{user.user_id}"
    accepted_value = (
        body.amended_value if body.choice == "amended" else None  # filled by service from llm_draft
    )
    event, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event_id,
        choice=InterpretationChoice(body.choice),
        accepted_value=accepted_value,  # service substitutes llm_draft if None and choice == accepted_as_drafted
        actor=actor,
    )
    return InterpretationResolveResponse(
        event=_interpretation_event_response(event),
        new_state=_composition_state_response(new_state),
    )
```

### Read-back route

```python
@router.get(
    "/{session_id}/interpretations",
    response_model=ListInterpretationEventsResponse,
)
async def list_interpretations(
    session_id: UUID,
    raw_request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
    status: Literal["pending", "all"] = "all",
) -> ListInterpretationEventsResponse:
    """List interpretation events for the session.

    Used by the frontend on session reload to rehydrate pending
    review affordances, and by the audit-readiness panel for counts.
    """
    await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
    service: SessionServiceProtocol = raw_request.app.state.session_service
    events = await service.list_interpretation_events(session_id, status=status)
    return ListInterpretationEventsResponse(
        events=[_interpretation_event_response(e) for e in events],
    )
```

### Tests

`tests/unit/web/sessions/test_interpretation_events_routes.py`:

1. POST /resolve with `choice='accepted_as_drafted'` succeeds, returns
   the resolved event with `accepted_value == llm_draft`, and the
   composition state version advances by 1.
2. POST /resolve with `choice='amended'` and a valid `amended_value`
   succeeds, returns `accepted_value == amended_value`.
3. POST /resolve with `choice='amended'` and missing `amended_value`
   returns 422 (Pydantic validation).
4. POST /resolve with `choice='accepted_as_drafted'` and present
   `amended_value` returns 422.
5. POST /resolve with `choice='opted_out'` returns 422 (Literal restricts).
6. POST /resolve twice for the same event_id returns 409 / 400 on the
   second call (TOCTOU guard).
7. POST /resolve with a non-existent event_id returns 404.
8. POST /resolve with an event_id from a different session returns 404
   (cross-session isolation; the service method filters by session_id).
9. GET /interpretations?status=pending returns only pending events.
10. GET /interpretations?status=all returns all events.
11. **IDOR regression (POST /resolve):** An authenticated user whose
    session is session-B cannot POST /resolve on an event belonging to
    session-A. Returns 404, not 403.
12. **IDOR regression (GET /interpretations):** An authenticated user
    whose session is session-B cannot GET /interpretations for
    session-A's id. Returns 404, not 403.

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

`sessions(routes): add POST /interpretations/{event_id}/resolve and
GET /interpretations`.

---

## Task 7 — HTTP route `POST /api/sessions/{id}/interpretations/opt_out`

**Goal.** Record the per-session "stop asking" decision. Single
transaction: flip the boolean on `sessions` and write an
`interpretation_events` row with `choice='opted_out'`
and `interpretation_source='auto_interpreted_opt_out'`.

**Files:**

- Modify: `src/elspeth/web/sessions/routes.py`.
- Create: `tests/unit/web/sessions/test_interpretation_opt_out_routes.py`.

### Route

> **DI pattern note:** Same as Task 6. `user: UserIdentity = Depends(get_current_user)`,
> `service = request.app.state.session_service`, `actor = f"user:{user.user_id}"`.
> Add `_verify_session_ownership` as the first call.

```python
@router.post(
    "/{session_id}/interpretations/opt_out",
    response_model=InterpretationOptOutResponse,
)
async def opt_out_of_interpretations(
    session_id: UUID,
    raw_request: Request,
    user: UserIdentity = Depends(get_current_user),  # noqa: B008
) -> InterpretationOptOutResponse:
    await _verify_session_ownership(session_id, user, raw_request)  # 404 on IDOR
    service: SessionServiceProtocol = raw_request.app.state.session_service
    actor = f"user:{user.user_id}"
    record = await service.record_session_interpretation_opt_out(
        session_id=session_id,
        actor=actor,
    )
    return InterpretationOptOutResponse(
        session_id=record.session_id,
        interpretation_review_disabled=True,
        opted_out_at=record.resolved_at,  # opted_out rows use resolved_at as the event timestamp
    )
```

The service method returns an `InterpretationEventRecord` with
`choice='opted_out'`. `resolved_at` carries the opt-out timestamp.

### Tests

1. POST /opt_out flips the session column to true and writes an
   `interpretation_events` row with `choice='opted_out'` and
   `interpretation_source='auto_interpreted_opt_out'`. Verify NO
   `proposal_events` row was written (regression guard).
2. POST /opt_out is idempotent — second call still returns 200; the
   first opt-out timestamp is preserved (second call returns the
   existing record). Deterministic choice: keep FIRST opt-out timestamp.
3. After /opt_out, the compose-loop's system prompt MUST be told the
   user opted out (Task 5's compose-loop hook reads the flag at
   compose start; verify by a fixture-level integration test in Task 9).
4. **IDOR regression:** An authenticated user whose session is session-B
   cannot POST /opt_out on session-A's id. Returns 404, not 403.

### Step 1-3

`sessions(routes): add POST /interpretations/opt_out (routes to
interpretation_events, not proposal_events)`.

---

## Task 8 — Composer-skill prompt nudge

**Goal.** Teach the LLM when to call `request_interpretation_review`.
Per the project memory `feedback_no_tests_for_skill_prompts`: the skill
file is an LLM prompt, not code; do NOT add string-grep tests. Validate
empirically by re-running the canonical hero prompt.

**Files:**

- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`.

### Content to add

A new section "Surfacing your interpretation of subjective terms"
that:

1. Lists the heuristics from design doc 06 §"When the interpretation
   gets surfaced" (the four "surface / don't surface" pairs).
2. Walks through the ordering: stage the LLM transform first (with the
   `{{interpretation:<term>}}` placeholder in its prompt_template),
   THEN call `request_interpretation_review` BEFORE finalising any
   downstream node that depends on the transform.
3. Bias-toward-false-positives instruction: when in doubt, surface.
4. Tells the model that IF the session has
   `interpretation_review_disabled=true` (which the system prompt
   surfaces), it must skip the surfacing step and bake a reasonable
   interpretation directly into the prompt template — but explicitly
   tag that interpretation in a comment within the prompt template
   for auditability (`# AUTO-INTERPRETED, REVIEW SKIPPED PER USER OPT-OUT`).

### Restart the service

After editing, restart `elspeth-web.service` per the project memory
`project_composer_harness_state` (the skill is `@lru_cache`'d at import
time).

### Step 1 — Empirical validation (NOT pytest)

Re-run the canonical hero prompt:

> create a list of 5 government web pages and use an LLM to rate how cool they are

Confirm:

- The LLM stages the LLM transform with a placeholder.
- The LLM calls `request_interpretation_review(user_term='cool', llm_draft=<...>, affected_node_id=<...>)`.
- The pending interpretation event row exists in the session DB.
- The LLM does NOT also call `set_pipeline` or otherwise finalise the
  pipeline before the user resolves the interpretation.

### Step 2 — Commit

`composer(skill): teach the LLM when to call request_interpretation_review`.

---

## Task 9 — Landscape (session-audit-DB) spot-check + runtime hand-off

**Goal.** This is the **mandatory audit spot-check** per CLAUDE.md
attributability test, plus the runtime hand-off test that confirms
the resolved prompt template lands at runtime.

**Files:**

- Create: `tests/integration/web/composer/test_interpretation_audit_spotcheck.py`.
- Create: `tests/integration/web/composer/test_interpretation_runtime_handoff.py`.

### Audit spot-check (`test_interpretation_audit_spotcheck.py`)

The test:

1. Spins up a real session-audit DB (in-memory SQLite or the
   `testcontainers` Postgres lane per Phase 1C's portability work).
2. Drives the writer paths:
   a. `create_pending_interpretation_event` → ASSERT all six required
      fields are present in the DB row with the expected values, types,
      and that the FK to `composition_states` resolves.
   b. `resolve_interpretation_event(choice='accepted_as_drafted')` →
      ASSERT `accepted_value == llm_draft`, `resolved_at IS NOT NULL`,
      `choice == 'accepted_as_drafted'`, FK still resolves.
   c. `resolve_interpretation_event(choice='amended', accepted_value='custom')`
      → ASSERT `accepted_value == 'custom'`, downstream effects.
   d. `record_session_interpretation_opt_out` → ASSERT
      `sessions.interpretation_review_disabled == true` AND an
      `interpretation_events` row with `choice='opted_out'`,
      `interpretation_source='auto_interpreted_opt_out'`, and
      `actor` matches. Also ASSERT that `proposal_events_table`
      contains NO row related to this opt-out (regression guard
      against prior proposal_events routing).
3. For each row, the test opens a read-only SQLAlchemy connection and
   issues `SELECT * FROM interpretation_events WHERE id = :id`. The
   test asserts EACH of the six required-by-spec fields is present and
   correctly populated. This is the explicit CLAUDE.md attributability
   compliance step.

### Runtime hand-off (`test_interpretation_runtime_handoff.py`)

The test:

1. Builds a small pipeline whose only LLM transform has
   `prompt_template = "Rate this on the dimension of {{interpretation:cool}}."`
2. Surfaces an interpretation via the tool path; resolves with
   `accepted_value = "modern design + clear purpose + interactivity"`.
3. Reads back the new composition state; asserts the LLM transform's
   `prompt_template` is now
   `"Rate this on the dimension of modern design + clear purpose + interactivity."`.
4. Runs the pipeline against a tiny in-memory source; asserts the
   runtime Landscape's `calls` row for the LLM transform records a
   `arguments_hash` over the resolved prompt template string (NOT the
   placeholder).
5. Walks the audit chain:
   `runtime calls row` → `pipeline state JSON` →
   `interpretation_events` row → `actor` → verify the lineage is
   complete and `explain(run_id, token_id)` returns the user-approved
   interpretation as the source of the prompt-template string.

The runtime hand-off test uses the production `ExecutionGraph.from_plugin_instances()`
+ `instantiate_plugins_from_config()` path per CLAUDE.md "never bypass
production code paths in tests."

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

`composer(integration): audit spot-check + runtime hand-off for
interpretation events`.

---

## Task 10 — Activate `llm_interpretations` row in audit-readiness service

**Goal.** Phase 2A's audit-readiness service (documented in
`14a-phase-2a-backend.md`) hardcoded the `llm_interpretations` row
to `not_applicable`. Phase 5b introduces interpretation events, so
this row must now reflect real status drawn from the
`interpretation_events_table`.

**Phase 2C has shipped.** This task is unconditional. Read
`src/elspeth/web/sessions/service.py` (or the audit-readiness service)
to locate the `llm_interpretations` row emitter returning
`not_applicable`, and activate it per the SQL and row-status mapping
below.

**SQL and row-status mapping (when in scope):**

```python
async def _llm_interpretations_status(
    self,
    session_id: UUID,
    composition_state_id: UUID,
) -> ReadinessStatus:
    """Query interpretation_events and return the row status.

    Mapping:
      - No LLM transforms in composition state → not_applicable
      - LLM transforms present, no interpretation events → not_applicable
        (surfacing not yet triggered; will be `warning` after first
        request_interpretation_review call fires)
      - Any pending events → warning
      - All resolved, at least one → ok
      - Session opted out → not_applicable (with a note in the summary)
    """
    opted_out = await self._is_interpretation_opted_out(session_id)
    if opted_out:
        return "not_applicable"
    events = await self.list_interpretation_events(
        session_id,
        status="all",
        composition_state_id=composition_state_id,
    )
    if not events:
        return "not_applicable"
    if any(e.choice == InterpretationChoice.PENDING for e in events):
        return "warning"
    return "ok"
```

**Files:**

- Modify: the audit-readiness service method in
  `src/elspeth/web/sessions/service.py` (or wherever Phase 2A
  implemented the row emitter; re-read `14a-phase-2a-backend.md`
  for the exact location).
- Add test in `tests/unit/web/sessions/` asserting each of the four
  status transitions above.

**Commit message:** `sessions(audit-readiness): activate
llm_interpretations row from interpretation_events (Phase 5b.10)`.

---

## Task 11 — Orphan PENDING-row recovery

**Goal.** A compose-loop crash between `create_pending_interpretation_event`
returning and the ToolResult being returned to the frontend leaves a PENDING
row the frontend has never seen. On session reload, `refreshPending` (18b
Task 3) already surfaces it in-band. This task adds:

1. An explicit test verifying that `refreshPending` correctly rehydrates a
   PENDING event that the frontend never received in-band:
   - Fixture creates a session with a PENDING interpretation event.
   - The frontend mock is initialised with no in-band notification of the event.
   - Calling `list_interpretation_events(session_id, status='pending')` returns
     the event.
   - The frontend's `refreshPending` path is exercised and the review
     affordance re-appears.

2. **Optional cleanup job (Phase 11, operator decision):** rows in PENDING
   state older than 7 days with no resolution. Two choices:

   - **Auto-resolve with `choice='abandoned'`**: writes `resolved_at=now`,
     `choice='abandoned'`, `interpretation_source` unchanged, `accepted_value=None`.
     Requires `InterpretationChoice.ABANDONED` (already added in Task 1).
     Audit-honest: the row remains as evidence the surfacing happened and was
     abandoned.
   - **Reap (delete)**: removes the row entirely. Simpler, but destroys
     audit evidence of the surfacing. Not recommended per the auditability
     standard.

   **Recommendation: auto-resolve with `choice='abandoned'`** (see Task 1 for
   the enum extension). Document this in the Phase 11 backlog; do not implement
   the job in Phase 5b.

**Files:**

- Add test: `tests/unit/web/sessions/test_interpretation_events_service.py`
  (extend existing file with the `refreshPending` rehydration test).

**Commit message:** `sessions(interpretation): orphan-PENDING rehydration test
(Task 11)`.

---

## Task 12 — Evals regression suite

**Goal.** A CI-gatable regression eval suite that verifies the interpretation
surfacing behaviour end-to-end.

**Location:** `evals/composer-rgr/phase5b-interpretation/`

**Result format:** JSON artifact at
`evals/composer-rgr/phase5b-interpretation/results-{date}.json`
with one object per run:
```json
{
  "run_index": 1,
  "timestamp_utc": "...",
  "model_id": "...",
  "hero_prompt": "create a list of 5 government web pages and use an LLM to rate how cool they are",
  "cool_surfaced": true,
  "numeric_quantifiers_not_surfaced": true,
  "placeholder_emitted_same_turn": true,
  "opt_out_disables_surfacing": true
}
```

**Pass thresholds (all must hold for the eval to gate-pass):**

| Assertion | Threshold |
|---|---|
| Hero prompt surfaces "cool" as the interpretation term | ≥8/10 runs |
| Hero prompt does NOT surface "5" or "1-10" as interpretation terms | ≥9/10 runs |
| The LLM emits `{{interpretation:cool}}` in the same turn as staging the LLM transform | ≥8/10 runs |
| When `interpretation_review_disabled=true`, the LLM skips `request_interpretation_review` and writes the auto-interpreted value with `interpretation_source='auto_interpreted_opt_out'` | ≥9/10 runs |

**Gate:** Phase 5b's PR is blocked until this eval suite passes. The artifact
must be committed to `evals/composer-rgr/phase5b-interpretation/` with the PR.

**Commit message:** `evals: phase5b interpretation regression suite`.

---

## Failure-mode behaviour

The following table specifies what happens in each significant failure mode.
All failure modes must be covered by unit or integration tests (cited in the
relevant task above). Any failure mode without a test is an open gap.

| Failure | Source | User-visible error | Audit-row outcome |
|---|---|---|---|
| LLM provider timeout during compose-loop | Provider | "The composer LLM is slow — try again." | No interpretation event created; tool_call_id remains in flight |
| LLM returns malformed JSON for tool args | Provider | Surfaces ARG_ERROR through normal compose-loop retry | No row created |
| LLM hallucinates `affected_node_id` that doesn't exist | Provider | ARG_ERROR | No row created |
| LLM emits an `accepted_value` (or `amended_value`) containing `{{`/`}}` | User-amend | 422 with field-specific message | No row created |
| LLM omits the `{{interpretation:<term>}}` placeholder | Provider | `_patch_llm_transform_prompt` raises ValueError → ToolArgumentError → ARG_ERROR | No row; if pending row exists, it is left PENDING and cleaned by Task 11 orphan recovery |
| Network 5xx mid-resolve | Network | 500 with retry guidance | TOCTOU guard (`WHERE choice='pending'`) ensures no partial state; pending row preserved |
| Patch-helper raises (placeholder absent) | System | 422 | No row |
| Database integrity violation | System | 500 + alert | Crashes per offensive-programming discipline (CLAUDE.md) |
| Rate cap exceeded (per-term: 3) | System | ARG_ERROR: "Too many interpretation requests for term '…' (max 3)" | No row; compose loop falls back to `auto_interpreted_no_surfaces` |
| Rate cap exceeded (per-session-day: 10) | System | ARG_ERROR: "Too many interpretation requests in this session today (max 10)" | No row; compose loop falls back to `auto_interpreted_no_surfaces` |
| Credential-shaped content in `user_term` or `llm_draft` | User/LLM | 422: "That looks like a credential — please re-enter without secrets" | No row |
| `accepted_value` contains control characters | User | 422 with field-specific message | No row |
| Append-only trigger fires (attempt to modify resolved row) | System | 500 (SQLAlchemy IntegrityError) | DB row unchanged; request fails |
| Session ownership failure (IDOR attempt) | Attacker | 404 (not 403 — per IDOR contract) | No write; no audit row |

---

## Backend completion criteria

Phase 5b backend is complete when:

- [ ] **Task 0 (placeholder-convention gate)** passed; artifact at
      `evals/composer-rgr/phase5b-task0-placeholder-validation.json`
      with `gate_passed=true`. All subsequent tasks unblocked.
- [ ] Task 1 (contract types) green, committed; `InterpretationChoice`
      has five values; `InterpretationSource` has three values.
- [ ] Task 2 (schema) green, committed; the new table exists with all
      audit-provenance columns; the new provenance enum value is
      registered; the `interpretation_review_disabled` session column
      is registered; the append-only trigger is installed.
      NOTE: `proposal_events.event_type` is NOT extended — opt-out rows
      go to `interpretation_events_table`.
- [ ] Task 3 (wire schemas) green, committed; content validators on
      `amended_value` pass; credential prefilter tests pass.
- [ ] Task 4 (service methods) green, committed; all four methods
      exercised by direct-DB spot-checks; opt-out method writes to
      `interpretation_events_table` only.
- [ ] Task 5 (composer tool) green, committed; dispatch spike documented;
      `_SESSION_AWARE_TOOL_HANDLERS` registry introduced; async handler
      dispatches; rate-limit tests pass; credential prefilter tests pass;
      redaction; proposal summary.
- [ ] Task 6 (resolve route + list route) green, committed; IDOR
      regression tests pass.
- [ ] Task 7 (opt_out route) green, committed; IDOR regression test
      passes; `proposal_events_table` regression guard passes.
- [ ] Task 8 (skill prompt) committed; empirically validated against
      the canonical hero prompt; `elspeth-web.service` restarted.
- [ ] Task 9 (integration spot-check + runtime hand-off) green,
      committed; paths at `tests/integration/web/composer/`.
- [ ] Task 10 (audit-readiness `llm_interpretations` row activation)
      green, committed (unconditional — Phase 2C has shipped).
- [ ] Task 11 (orphan PENDING-row cleanup spec) documented; `refreshPending`
      test passes.
- [ ] Task 12 (evals/composer-rgr regression suite) artifact exists;
      all four pass thresholds met.
- [ ] Failure-mode table reviewed; no uncovered failure path.
- [ ] Tier-model enforcer pass (`scripts/cicd/enforce_tier_model.py
      check --root src/elspeth ...`). No new upward imports.
- [ ] Audit primacy compliance: every recording call is wrapped in the
      session write lock; no recording is bypassed on any error path.
- [ ] mypy clean on touched modules.
- [ ] ruff clean on touched modules.

Then proceed to [18b-phase-5b-frontend.md](18b-phase-5b-frontend.md).

---

## Review history

| Date | Reviewer | Verdict | Finding IDs | Notes |
|------|----------|---------|-------------|-------|
| 2026-05-15 | Review panel | CHANGES_REQUESTED | B1, C1, C2, I1, I2 | Applied in prior revision. B1: added "Migration runner ownership" section documenting cumulative DB-delete impact and Phase 9 ownership. C1: added deploy constraint paragraph to the closed-enum extension section — production deploy blocked until Phase 9. C2: expanded Task 4 test item 2 with a concrete failing-test specification for `affected_node_id` writer-boundary validation. I1: added positive-case test 5a for the partial unique index. I2: added Task 10 to activate the `llm_interpretations` audit-readiness row using `interpretation_events_table`, with conditional-on-Phase-2A guard. |
| 2026-05-18 | 7-reviewer panel | CHANGES_REQUESTED | C1, C2, C3, H1, MED-1, MED-2, MED-3, HIGH-1, HIGH-2, HIGH-3, HIGH-4, T-01, T-02, T-04, T-06, F1, F2, F3, F4, F6, F7, M3, W2 | Applied in this revision. Task 0 hard gate added. C1: DI fixes in Tasks 6+7 (no `get_authenticated_actor`/`get_session_service`). C2: n/a (clarification). C3: integration test paths corrected to `tests/integration/web/composer/`. H1: redaction helper renamed to `_summarize_interpretation_term` (American spelling, new definition). MED-1/HIGH-4: opt-out routing pivoted from `proposal_events` to `interpretation_events_table` across Tasks 1, 2, 4, 7, 9. MED-2: `InterpretationOptOutRecord` dropped; replaced by view comment on `InterpretationEventRecord`. MED-3: `async def _handle_request_interpretation_review` + dispatch spike (Option B, `_SESSION_AWARE_TOOL_HANDLERS`). HIGH-1: Task 10 conditional removed (Phase 2C has shipped). HIGH-2: telemetry NONE declarations added per Task 4 methods. HIGH-3: dispatch spike added as Pre-Task spike in Task 5. T-01: `_verify_session_ownership` added to Tasks 6+7 routes with IDOR regression tests. T-02/F2: content validators on `amended_value` (metachar, control chars, 1024-char single-line cap, credential prefilter). T-04/F5: credential-shape prefilter added to Task 3 and Task 5. T-06/F4: `model_identifier`, `model_version`, `provider`, `composer_skill_hash`, `arguments_hash` added to schema and contract. F1/F3: append-only trigger added to Task 2; `InterpretationSource` closed enum added. F6: failure-mode table section added. F7: evals regression suite spec (Task 12) added. M3: Task 11 orphan PENDING-row recovery added. `InterpretationChoice` extended with `ABANDONED`. |

---

## Backend open questions

If implementation surfaces any of the following, surface to the
operator before committing to an interpretation:

1. **Placeholder convention friction.** This question is now gated by
   Task 0. If Task 0 passes (≥8/10 runs emit the placeholder), proceed.
   If Task 0 fails, the architecture pivots to runtime-resolve-from-table
   (store the resolved prompt template separately on the interpretation
   event row; runtime resolves via that table) and 18a must be re-drafted
   before any code is committed. Surface immediately; do not silently rework.
2. **Session write-lock contention.** Under concurrent compose-loop
   + resolve-route traffic on the same session, lock contention may
   slow user-perceived latency. Mitigation if observed in staging:
   the resolve path is intentionally short (single transaction); if
   contention becomes a problem, profile before pessimising. Do not
   pre-optimise.
3. **Interaction with Phase 5a's `set_pipeline` flow.** If the LLM
   races between `set_pipeline` (which can replace the entire graph)
   and `request_interpretation_review` (which depends on a specific
   `affected_node_id`), a stale `affected_node_id` would surface as
   ARG_ERROR. The skill prompt MUST instruct the LLM to call
   `request_interpretation_review` *after* the relevant LLM transform
   is in the composition state and *before* any subsequent
   `set_pipeline` or `remove_node` that could remove it. The Task 5
   handler's `_assert_affected_llm_node` is the structural enforcement
   of this property.
