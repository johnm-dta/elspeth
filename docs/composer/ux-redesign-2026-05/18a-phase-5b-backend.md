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

tests/integration/composer/
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


@dataclass(frozen=True, slots=True)
class InterpretationEventRecord:
    """A discrete user decision about an LLM-surfaced interpretation.

    Tier-1 read-side record: every field is required-or-explicitly-None
    per the schema. Constructors crash loudly on any anomaly.

    Six required-by-spec fields (design doc 06 §"Recording the
    interpretation"):
        user_term            -> the original user-provided term ("cool")
        llm_draft            -> the LLM's draft interpretation
        accepted_value       -> the user-approved string (None until resolved)
        created_at           -> timestamp at surfacing
        actor                -> user identity at resolution
        composition_state_id -> pipeline-state reference

    Operational fields:
        id, session_id, choice, resolved_at,
        affected_node_id (the LLM-transform node this binds into),
        tool_call_id (the LLM tool-call that surfaced it).
    """

    id: UUID
    session_id: UUID
    composition_state_id: UUID  # pipeline-state reference
    affected_node_id: str       # node_id of the LLM transform
    tool_call_id: str           # provider tool_call_id from the LLM
    user_term: str              # required-by-spec
    llm_draft: str              # required-by-spec
    accepted_value: str | None  # required-by-spec (None until resolved)
    choice: InterpretationChoice
    created_at: datetime
    resolved_at: datetime | None
    actor: str                  # required-by-spec (user identity)


@dataclass(frozen=True, slots=True)
class InterpretationOptOutRecord:
    """Audit record for a user opting out of LLM interpretation surfacing.

    Tier-1 read-side record: every field is required-or-explicitly-None.
    Constructors crash loudly on any anomaly.

    Records the opt-out event separately from InterpretationEventRecord so
    that 'did the user opt out' vs 'did the user experience an interpretation
    event' answer distinctly per the auditability standard.
    """

    session_id: str
    user_id: str
    prior_choice: InterpretationChoice | None
    opted_out_at: datetime
    reason: str | None

    def __post_init__(self) -> None:
        freeze_fields(self)
```

### Rationale

Audit-trail clarity demands that "did the user opt out" vs "did the user
experience an interpretation event" answer distinctly per the auditability
standard. A dedicated dataclass aligns with audit-record clarity over
abstraction reuse.

`InterpretationOptOutRecord` carries `prior_choice` so the audit trail
records whether the opt-out replaced an in-flight pending event. All fields
are scalars, `StrEnum`, `datetime`, or `None`; `freeze_fields()` with no
arguments is supplied as belt-and-suspenders per the project deep_freeze
contract even though no container fields are present.

### Test shape

The contract tests assert:

1. `InterpretationChoice` values are the four expected strings.
2. `InterpretationEventRecord` is frozen (assigning a field raises).
3. Constructing with any of the six required-by-spec fields missing is
   a `TypeError` (no defaults).
4. The dataclass round-trips through `dataclasses.asdict()` correctly.
5. `InterpretationOptOutRecord` is frozen (assigning a field raises).
6. `InterpretationOptOutRecord` accepts `prior_choice=None` and
   `reason=None` without error (both are explicitly nullable).
7. `InterpretationOptOutRecord` round-trips through `dataclasses.asdict()`
   correctly.

### Step 1 — RED

Write the four assertions; running pytest fails because the module
doesn't exist.

### Step 2 — GREEN

Implement the module exactly per the shape above. `InterpretationEventRecord`
has no container fields; all fields are scalars, enum, datetime, UUID, or
None — `frozen=True` suffices for that record.
`InterpretationOptOutRecord` calls `freeze_fields()` with no arguments as
belt-and-suspenders per the project deep_freeze contract.

### Step 3 — Commit

Single commit; message: `contracts: add InterpretationEventRecord +
InterpretationChoice for Phase 5b`.

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
    # Closed enum. Adding a value requires (a) amending this plan, (b)
    # extending InterpretationChoice in contracts/composer_interpretation.py,
    # (c) updating the closed-enum tests, and (d) a writer-path audit.
    # NO SILENT EXTENSION.
    CheckConstraint(
        "choice IN ('pending', 'accepted_as_drafted', 'amended', 'opted_out')",
        name="ck_interpretation_events_choice",
    ),
    # If choice is anything other than 'pending', resolved_at MUST be
    # populated and accepted_value MUST be populated (unless choice is
    # 'opted_out', in which case accepted_value is NULL but resolved_at
    # is still populated to record the opt-out timestamp).
    CheckConstraint(
        "(choice = 'pending') = (resolved_at IS NULL)",
        name="ck_interpretation_events_resolved_at_status",
    ),
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

The opt-out audit event (Task 7) writes to `proposal_events_table`
with the new event_type value `interpretation.opted_out`. This DOES
extend the existing `proposal_events.event_type` CHECK constraint;
update the enum to include the new value, and document the writer-path
in the same governance block style:

```python
CheckConstraint(
    "event_type IN ('proposal.created', 'proposal.accepted', "
    "'proposal.rejected', 'trust_mode.changed', 'interpretation.opted_out')",
    name="ck_proposal_events_type",
),
```

### Test shape

`tests/unit/web/sessions/test_interpretation_events_table.py`:

1. Test that `metadata.create_all(engine)` succeeds against an in-memory
   SQLite engine and produces a table named `interpretation_events`
   with all the expected columns.
2. Test that inserting a row with `choice='pending'` and `resolved_at`
   set raises `IntegrityError`.
3. Test that inserting a row with `choice='accepted_as_drafted'` and
   `accepted_value=NULL` raises `IntegrityError`.
4. Test that inserting a row with `choice='opted_out'` and
   `accepted_value=NULL` succeeds (opt-out has no `accepted_value`).
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
7. Test that the FK to `composition_states` is enforced (insert a row
   with a non-existent `composition_state_id`; expect IntegrityError).
8. Test that the closed enum on `provenance` rejects any value not in
   the seven listed.
9. Test that `sessions_table` now has the new
   `interpretation_review_disabled` boolean column with default `false`.

### Step 1 — RED

Run the test file; all 9 cases fail because the table doesn't exist
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

Single commit. Message: `sessions(schema): add
interpretation_events_table + interpretation_resolve provenance +
interpretation.opted_out event_type + interpretation_review_disabled
column`. The commit message MUST mention all four schema edits.

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
    composition_state_id: UUID
    affected_node_id: str = Field(min_length=1, max_length=256)
    tool_call_id: str = Field(min_length=1, max_length=256)
    user_term: str = Field(min_length=1, max_length=8192)
    llm_draft: str = Field(min_length=1, max_length=8192)
    accepted_value: str | None = Field(default=None, max_length=8192)
    choice: Literal[
        "pending", "accepted_as_drafted", "amended", "opted_out"
    ]
    created_at: datetime
    resolved_at: datetime | None = None
    actor: str = Field(min_length=1, max_length=256)


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
5. The length caps reject 8193-char strings.

### Step 2 — GREEN

Implement.

### Step 3 — Commit

`sessions(schemas): wire schemas for interpretation events + resolve +
opt-out`.

---

## Task 4 — Session service methods

**Goal.** Add the writer/reader methods that drive the new table.
Method-naming and transaction-shape mirror `create_composition_proposal`
/ `list_proposal_events` from `service.py`.

**Files:**

- Modify: `src/elspeth/web/sessions/service.py`.
- Create: `tests/unit/web/sessions/test_interpretation_events_service.py`.

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
    created_at: datetime | None = None,
) -> InterpretationEventRecord:
    """Insert a PENDING interpretation event.

    Called from the compose-loop tool handler for
    request_interpretation_review. Acquires the session write lock for
    the duration of the insert. Validates affected_node_id exists in
    composition_states.nodes BEFORE committing the row (raises
    ValueError otherwise; the tool handler converts to ARG_ERROR).
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
           resolved_at=?, actor=? WHERE id=? AND choice='pending'.
           (CHECK constraint enforces consistency; the WHERE on
           choice='pending' is a TOCTOU guard against double-resolve.)
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
    """


async def record_session_interpretation_opt_out(
    self,
    *,
    session_id: UUID,
    actor: str,
    opted_out_at: datetime | None = None,
) -> InterpretationOptOutRecord:
    """Mark the session as 'don't surface interpretations any more'.

    Writes a proposal_events row with event_type='interpretation.opted_out'
    AND sets sessions.interpretation_review_disabled = true. Single
    transaction.
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
template" property.

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
    `sessions_table` and writes a `proposal_events` row with
    event_type='interpretation.opted_out'.

### Step 1 — RED

Tests fail; methods don't exist.

### Step 2 — GREEN

Implement methods in `service.py`. Each writer uses
`_session_write_lock(conn, sid)`. Use the same `_run_sync(_sync)`
wrapper pattern as elsewhere in the file.

### Step 3 — Commit

`sessions(service): writer + reader methods for interpretation events
and opt-out`.

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

### Tool handler

In `tools.py`, alongside `_handle_upsert_node`:

```python
def _handle_request_interpretation_review(
    state: CompositionState,
    arguments: object,
    *,
    session_id: UUID,
    tool_call_id: str,
    create_pending_interpretation_event: Callable[..., Awaitable[InterpretationEventRecord]],
) -> ToolResult:
    """Stage a pending interpretation event for user review.

    Returns a SUCCESS ToolResult whose payload signals the frontend to
    surface the review affordance. Does NOT advance composition state
    version (state changes happen at /resolve time).
    """
    parsed = _validate_mutation_arguments(
        _RequestInterpretationReviewArgumentsModel,
        arguments,
        "request_interpretation_review",
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

```python
class _RequestInterpretationReviewRedactionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    affected_node_id: str
    # user_term is NOT a secret — it's a word the user typed. But it
    # could carry PII (e.g., user types "rate how cool this transaction
    # involving John Doe is"); apply summariser cap to be safe.
    user_term: Annotated[str, Sensitive(summarizer=_summarise_short_string)]
    llm_draft: Annotated[str, Sensitive(summarizer=_summarise_short_string)]
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
just needs to be added to the dispatch dict alongside `_handle_upsert_node`
etc. Use the existing `dispatch_with_audit` helper — it doesn't need
modification.

The tool handler signature differs from the others (it needs
`session_id`, `tool_call_id`, and the service method as a callable).
Wire this through the compose-loop closure exactly as other
session-aware tools currently are; if no such tool exists today, the
dispatch dispatcher needs a small extension to thread these dependencies
into the handler. (Re-check current state: most existing tools are
state-pure; this is the first session-aware tool that recorders into
sessions DB directly. If it is, factor the closure cleanly rather than
sprinkling globals.)

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

`test_request_interpretation_review_redaction.py`:

1. Redaction model accepts a valid argument dict.
2. Redaction model produces redacted output with `user_term` and
   `llm_draft` summarised.
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

```python
@router.post(
    "/{session_id}/interpretations/{event_id}/resolve",
    response_model=InterpretationResolveResponse,
)
async def resolve_interpretation(
    session_id: UUID,
    event_id: UUID,
    request: InterpretationResolveRequest,
    actor: str = Depends(get_authenticated_actor),
    service: SessionServiceProtocol = Depends(get_session_service),
) -> InterpretationResolveResponse:
    """User-driven resolve of a pending interpretation event.

    Tier-3 boundary: request body validated by Pydantic; choice/amended_value
    consistency enforced by the model's validator. The service method
    enforces semantic constraints (event must be pending; node must still
    exist; prompt-template patch must succeed).
    """
    accepted_value = (
        request.amended_value if request.choice == "amended" else None  # filled by service from llm_draft
    )
    event, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event_id,
        choice=InterpretationChoice(request.choice),
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
    status: Literal["pending", "all"] = "all",
    service: SessionServiceProtocol = Depends(get_session_service),
) -> ListInterpretationEventsResponse:
    """List interpretation events for the session.

    Used by the frontend on session reload to rehydrate pending
    review affordances, and by the audit-readiness panel for counts.
    """
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

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

`sessions(routes): add POST /interpretations/{event_id}/resolve and
GET /interpretations`.

---

## Task 7 — HTTP route `POST /api/sessions/{id}/interpretations/opt_out`

**Goal.** Record the per-session "stop asking" decision. Single
transaction: flip the boolean on `sessions` and write the
`proposal_events` row with event_type='interpretation.opted_out'.

**Files:**

- Modify: `src/elspeth/web/sessions/routes.py`.
- Create: `tests/unit/web/sessions/test_interpretation_opt_out_routes.py`.

### Route

```python
@router.post(
    "/{session_id}/interpretations/opt_out",
    response_model=InterpretationOptOutResponse,
)
async def opt_out_of_interpretations(
    session_id: UUID,
    actor: str = Depends(get_authenticated_actor),
    service: SessionServiceProtocol = Depends(get_session_service),
) -> InterpretationOptOutResponse:
    record = await service.record_session_interpretation_opt_out(
        session_id=session_id,
        actor=actor,
    )
    return InterpretationOptOutResponse(
        session_id=record.session_id,
        interpretation_review_disabled=True,
        opted_out_at=record.opted_out_at,
    )
```

The service method returns a record carrying session_id and the timestamp.

### Tests

1. POST /opt_out flips the session column to true and writes the
   `proposal_events` row.
2. POST /opt_out is idempotent — second call still returns 200 and the
   timestamp updates (or stays; pick deterministic behaviour: keep
   FIRST opt-out timestamp; second call returns existing record).
3. After /opt_out, the compose-loop's system prompt MUST be told the
   user opted out (Task 5's compose-loop hook reads the flag at
   compose start; verify by a fixture-level integration test in Task 9).

### Step 1-3

`sessions(routes): add POST /interpretations/opt_out`.

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

- Create: `tests/integration/composer/test_interpretation_audit_spotcheck.py`.
- Create: `tests/integration/composer/test_interpretation_runtime_handoff.py`.

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
      `sessions.interpretation_review_disabled == true` AND a
      `proposal_events` row with `event_type='interpretation.opted_out'`
      and `actor` matches.
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

**Conditional on Phase 2A having shipped.** Check whether
`src/elspeth/web/sessions/service.py` (or the audit-readiness service)
contains a `llm_interpretations` row emitter returning
`not_applicable`. If it does, this task is in scope. If Phase 2A has
not shipped, defer this task with a follow-up ticket and note it on the
umbrella PR (matching the conditional pattern in Task 7 of plan 17).

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

## Backend completion criteria

Phase 5b backend is complete when:

- [ ] Task 1 (contract types) green, committed.
- [ ] Task 2 (schema) green, committed; the new table exists; the new
      provenance enum value is registered; the new event_type enum
      value is registered; the new session column is registered.
- [ ] Task 3 (wire schemas) green, committed.
- [ ] Task 4 (service methods) green, committed; all four methods
      exercised by direct-DB spot-checks.
- [ ] Task 5 (composer tool) green, committed; tool registered;
      handler dispatches; redaction; proposal summary.
- [ ] Task 6 (resolve route + list route) green, committed.
- [ ] Task 7 (opt_out route) green, committed.
- [ ] Task 8 (skill prompt) committed; empirically validated against
      the canonical hero prompt; `elspeth-web.service` restarted.
- [ ] Task 9 (integration spot-check + runtime hand-off) green,
      committed.
- [ ] Task 10 (audit-readiness `llm_interpretations` row activation)
      green, committed (OR deferred with tracked followup if Phase 2A
      has not shipped).
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
| 2026-05-15 | Review panel | CHANGES_REQUESTED | B1, C1, C2, I1, I2 | Applied in this revision. B1: added "Migration runner ownership" section documenting cumulative DB-delete impact and Phase 9 ownership. C1: added deploy constraint paragraph to the closed-enum extension section — production deploy blocked until Phase 9. C2: expanded Task 4 test item 2 with a concrete failing-test specification for `affected_node_id` writer-boundary validation. I1: added positive-case test 5a for the partial unique index. I2: added Task 10 to activate the `llm_interpretations` audit-readiness row using `interpretation_events_table`, with conditional-on-Phase-2A guard. |

---

## Backend open questions

If implementation surfaces any of the following, surface to the
operator before committing to an interpretation:

1. **Placeholder convention friction.** If the LLM cannot reliably
   emit `{{interpretation:<term>}}` placeholders in the prompt
   template, the patch-on-resolve approach fails. Mitigation if
   discovered: switch to "store the resolved prompt template
   separately on the interpretation event and have the runtime resolve
   via that table." This would require runtime code changes (out of
   Phase 5b's current scope); surface immediately, do not silently
   rework.
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
