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

## Worktree

**Branch:** `feat/composer-phase-5-chat-data-entry`
**Worktree path:** `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`
**Shared with:** the entire Phase 5 umbrella (17-, 18-, 18a-, 18b-). Phase 5a and Phase 5b ship as a coordinated PR; do NOT split into separate branches. This document is one of the four that will be implemented together on this single worktree. Shared with 17-, 18-, 18b- (the Phase 5a plan, Phase 5b overview, and Phase 5b frontend plan).

### Setup (one-time)

From the main checkout at `/home/john/elspeth`:

```bash
git worktree add .worktrees/composer-phase-5-chat-data-entry -b feat/composer-phase-5-chat-data-entry
cd .worktrees/composer-phase-5-chat-data-entry
uv venv --python 3.13                       # Python 3.13 to match main; mismatched versions produce ~300 spurious tier-model violations
source .venv/bin/activate
uv pip install -e ".[dev,llm]"              # editable install bound to THIS worktree's venv, not main's
```

### Operational notes

- **uv venv discipline:** every `uv pip install` invocation in this worktree MUST be preceded by `source .venv/bin/activate` OR invoked with `--python /home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/.venv/bin/python`. Without this, `uv` resolves to main's `.venv` and clobbers it. (See `feedback_uv_venv_leak`.)
- **filigree CLI:** the bare `filigree` command rejects realpath-escaping DBs from inside a worktree. Prefer the `mcp__filigree__*` tools. If you must use the CLI, run it from the git common dir: `(cd "$(git rev-parse --git-common-dir)/.." && filigree <verb>)`.
- **Subagent dispatch from this worktree:** subagents inherit parent CWD silently. Prefix every dispatch prompt with: "Your CWD is `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`; all file paths must be absolute." Use absolute paths everywhere. (See `feedback_subagents_cant_use_worktrees`.)
- **Composer-skill edits stay on main:** the `src/elspeth/web/composer/skills/pipeline_composer.md` file is read by the live `elspeth-web.service` from main, not from any worktree. Skill-prompt edits in this phase (e.g. 5a Task 8 nudge, 5b Task 8 nudge) must be applied on main and the service restarted, per `feedback_skip_worktree_for_skill_and_config_edits`. Land the rest of the work in the worktree as normal.

### Coordination during implementation

- All four plan docs ship one commit history. The order is: 17- (Phase 5a) lands first; then 18a- (Phase 5b backend); then 18b- (Phase 5b frontend). 18- (overview) carries no code changes — its amendments land alongside whichever backend doc they cross-reference.
- The two-DB deletion requirement (session DB + Landscape audit.db) is operator-visible — surface it in the PR description so the operator can run the deletion before deploy.

---

## Tech Stack (backend slice)

- Python 3.13, SQLAlchemy Core, pydantic v2.
- `pluggy` not touched (no new plugin types).
- Session audit DB: `web/sessions/{models,schema,service,routes,schemas,protocol}.py`.
- Composer tool surface: `web/composer/{tools,redaction,proposals,service}.py`.
- No new third-party dependencies.
- One Landscape (`core/landscape/`) change: a `resolved_prompt_template_hash`
  nullable `String(64)` column added to `calls_table` in
  `core/landscape/schema.py` (Task 2, §"Runtime Landscape calls table").
  No new cross-layer imports; the enforce_tier_model gate continues to pass.

## File structure (backend changes)

> All file paths below are relative to the worktree root at `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`; this is identical to main's tree but isolates working state per the project's worktree-by-default convention (`feedback_default_to_worktree`). Exception: `src/elspeth/web/composer/skills/pipeline_composer.md` (Task 8) is edited on main, not in the worktree — see the Worktree section above.

```text
src/elspeth/contracts/
  composer_interpretation.py                                    CREATE    (Task 1)
  # Adds: InterpretationEventRecord, InterpretationChoice,
  # InterpretationSource, INTERPRETATION_HASH_DOMAIN_V1 (F-12)
  # Adds: resolved_prompt_template_hash: str | None field on InterpretationEventRecord

src/elspeth/core/landscape/
  schema.py                                                     MODIFY    (Task 2 — calls_table resolved_prompt_template_hash column + index)
  # Adds: resolved_prompt_template_hash nullable String(64) column to calls_table
  # Adds: ix_calls_resolved_prompt_template_hash index

src/elspeth/web/
  validation.py                                                 MODIFY    (Task 3 — extend with shared content-check helpers; F-34)

src/elspeth/web/sessions/
  engine.py                                                     MODIFY    (Task 1.5 — WAL + busy_timeout + schema-epoch; F-9/F-10)
  models.py                                                     MODIFY    (Tasks 1.5, 2)
  # Adds: SESSION_SCHEMA_EPOCH, SESSION_DB_APPLICATION_ID (Task 1.5)
  # Adds: interpretation_events_table, skill_markdown_history_table (Task 2)
  # Adds: F-1 nullable columns, F-5c new table, F-12 hash_domain_version,
  #        F-19 runtime model columns, immutability triggers (F-4, F-23),
  #        Phase 9 migration notes (F-16), F-35 governance comment,
  #        resolved_prompt_template_hash column on interpretation_events_table
  schema.py                                                     MODIFY    (Tasks 1.5, 2 — schema-epoch check, trigger validation; F-10, F-24)
  protocol.py                                                   MODIFY    (Task 3)
  schemas.py                                                    MODIFY    (Tasks 0.5, 3 — SendMessageRequest cap + wire schemas)
  service.py                                                    MODIFY    (Task 4)
  routes.py                                                     MODIFY    (Tasks 6, 7)
  _persist_payload.py                                           MODIFY    (Task 4 — if persistence carrier touched)

src/elspeth/web/composer/
  redaction.py                                                  MODIFY    (Tasks 0.5, 5 — InlineBlobModel cap + argument redaction)
  tools.py                                                      MODIFY    (Task 5)
  proposals.py                                                  MODIFY    (Task 5 — proposal summary)
  service.py                                                    MODIFY    (Task 5 — compose-loop hook)
  skills/pipeline_composer.md                                   MODIFY    (Task 8)

src/elspeth/web/
  app.py                                                        MODIFY    (Task 0.5 — ASGI body-size middleware; F-3)

tests/unit/contracts/
  test_composer_interpretation.py                               CREATE    (Task 1)

tests/unit/web/
  (validation.py tests in existing test_validation.py or new file)        (Task 3)

tests/unit/web/sessions/
  test_interpretation_events_table.py                           CREATE    (Task 2)
  test_interpretation_events_service.py                         CREATE    (Task 4)
  test_session_engine.py                                        CREATE    (Task 1.5)
  test_interpretation_events_routes.py                          CREATE    (Task 6)
  test_interpretation_opt_out_routes.py                         CREATE    (Task 7)

tests/unit/web/composer/
  test_request_interpretation_review_tool.py                    CREATE    (Task 5)
  test_request_interpretation_review_redaction.py               CREATE    (Task 5)

tests/integration/web/composer/
  test_interpretation_audit_spotcheck.py                        CREATE    (Task 9 — Landscape spot-check)
  test_interpretation_runtime_handoff.py                        CREATE    (Task 9 — prompt-template patch)
```

One Landscape schema file changes: `core/landscape/schema.py` gains the
`resolved_prompt_template_hash` column on `calls_table` (Task 2,
§"Runtime Landscape calls table"). No `core/canonical.py` change.

---

## Migration runner ownership (deferred — see roadmap §D5)

Phase 5b is the third schema-addition under `project_db_migration_policy`. Each
addition wipes prior phases' user state. The structural fix — a migration
runner that preserves `user_preferences` and session history across schema
changes — is OWNED BY PHASE 9 (post-launch). Phase 5b ships under the current
delete-the-DB policy with explicit acknowledgment that users who completed the
tutorial between Phase 4 and Phase 5b will be re-tutorial'd on the Phase 5b
deploy.

**Phase 5b is the first phase that requires deleting BOTH the session DB AND
the Landscape audit DB.** The `resolved_prompt_template_hash` column on
`calls_table` is in `core/landscape/schema.py` (L1), which SQLAlchemy does not
auto-migrate. The operator MUST:

1. Delete the session DB (`web/sessions/*.db` or equivalent path).
2. Delete the Landscape audit DB (`audit.db` at the project root, or wherever
   the Landscape DB is configured for the deploy environment).
3. Restart the service.

Failure to delete the Landscape DB will leave `calls_table` without the
`resolved_prompt_template_hash` column; the runtime LLM-transform plugin will
fail to write the column, and the cross-DB hash-equality assertion in Task 9
will fail permanently.

Operator action: communicate the two-DB delete requirement to test users before
each Phase 5b deploy. Document the Landscape DB path in the staging deploy
runbook (`project_staging_deployment.md`).

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

## Task 0.5 — Defensive size caps (F-3)

**Goal.** Add input-size limits that prevent unbounded payload ingestion
before any interpretation-event code paths can be exercised. This task is
a prerequisite for Task 1 and has no dependencies.

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py` — add `max_length` on
  `_InlineBlobModel.content`.
- Modify: `src/elspeth/web/sessions/schemas.py` — add `max_length` on
  `SendMessageRequest.content`.
- Modify: `src/elspeth/web/app.py` — add ASGI body-size middleware.

### Changes

**1. `_InlineBlobModel.content` cap (`redaction.py`):**

```python
content: Annotated[str, Sensitive(summarizer=_summarize_inline_blob_content)] = Field(
    max_length=262144  # 256 KiB: covers ~few hundred URLs with headroom
)
```

Verify `_InlineBlobModel` at `redaction.py` around line 1021; currently no
`max_length` on `content`. The 256 KiB limit covers the canonical inline-blob
use case (a list of URLs, a short CSV fragment) with headroom for edge cases.

**2. `SendMessageRequest.content` cap (`schemas.py`):**

```python
content: str = pydantic.Field(min_length=1, max_length=65536)  # 64 KiB cap
```

The current field at `schemas.py:99` has `min_length=1` but no `max_length`.
Add `max_length=65536`. 64 KiB accommodates multi-paragraph user messages
and long paste content while preventing unbounded string allocation.

**3. ASGI body-size middleware (`app.py`):**

Add a `BaseHTTPMiddleware` subclass that rejects requests with
`Content-Length > 10 MB` with HTTP 413 before any handler reads the body:

```python
from starlette.middleware.base import BaseHTTPMiddleware

class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies exceeding 10 MB (Content-Length check only).

    Mirrors the blob-upload body-size guard at web/blobs/routes.py:171,208.
    This is a defense-in-depth measure; Pydantic field validators on
    individual fields are the primary per-field cap.
    """
    _MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB

    async def dispatch(self, request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            if int(content_length) > self._MAX_BODY_BYTES:
                from starlette.responses import Response
                return Response(
                    content='{"error": "Request body too large (max 10 MB)"}',
                    status_code=413,
                    media_type="application/json",
                )
        return await call_next(request)
```

Register in `app.py` after the existing middleware registrations:
```python
app.add_middleware(_BodySizeLimitMiddleware)
```

Verify `app.py` around lines 377-391 — currently no body-size middleware.

### Test shape

1. `test_send_message_request_content_cap`: construct
   `SendMessageRequest(content="x" * 65537)` and assert `ValidationError` is
   raised. Construct with `"x" * 65536` and assert it succeeds.
2. `test_inline_blob_model_content_cap`: construct
   `_InlineBlobModel(filename="f", mime_type="text/plain", content="x" * 262145)`
   and assert `ValidationError`. With 262144 chars, assert success.
3. `test_body_size_middleware_413`: integration test — POST any JSON body
   with `Content-Length: 11000000` header to any route; assert HTTP 413
   response.

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

`web(security): add max_length caps on content fields + ASGI 10 MB body limit`.

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

    Three row shapes exist (see InterpretationSource enum):

    user_approved rows — LLM surfaced the term; user approved or amended:
        composition_state_id -> pipeline-state reference (NOT NULL)
        affected_node_id     -> the LLM-transform node this binds into (NOT NULL)
        tool_call_id         -> provider tool_call_id from the LLM (NOT NULL)
        user_term            -> the original user-provided term, e.g. "cool" (NOT NULL)
        llm_draft            -> the LLM's draft interpretation (NOT NULL)
        accepted_value       -> the user-approved string (None until resolved)
        arguments_hash       -> rfc8785 hash over required fields; None until resolved
        model_identifier     -> e.g., "anthropic/claude-opus-4-7" (NOT NULL)
        model_version        -> provider's reported version string (NOT NULL)
        provider             -> "anthropic", "openai", etc. (NOT NULL)
        composer_skill_hash  -> SHA-256 of pipeline_composer.md content (NOT NULL)

    auto_interpreted_opt_out rows — user clicked "stop asking":
        (all nine fields above are NULL — no LLM surfacing occurred)
        choice = 'opted_out'; interpretation_source = 'auto_interpreted_opt_out'
        resolved_at records the opt-out timestamp

    auto_interpreted_no_surfaces rows — rate cap exhausted; LLM baked it in:
        composition_state_id, affected_node_id, tool_call_id, user_term,
        llm_draft are NULL (no surfacing occurred — the rejected request
        never produced a draft or a composition-state binding)
        model_identifier, model_version, provider, composer_skill_hash MUST be
        populated (the LLM was consulted; provenance is required — read from
        the compose-loop snapshot, same source as user_approved rows).
        Asymmetry: interpretation surface fields are NULL; LLM provenance
        is required. See ck_interpretation_events_no_surfaces_shape.
        choice = 'opted_out' semantics; interpretation_source = 'auto_interpreted_no_surfaces'

    Fields always present:
        id, session_id, choice, created_at, resolved_at, actor,
        interpretation_source

    Per the auditability standard (design doc 06 §"Recording the
    interpretation"), all nine of: user_term, llm_draft, accepted_value,
    created_at, actor, composition_state_id, model_identifier, model_version,
    composer_skill_hash are required for user_approved rows. They are
    intentionally NULL for auto_interpreted_opt_out rows (F-1: no LLM surfacing).

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
    # F-1: audit provenance fields are nullable — NULL for
    # auto_interpreted_opt_out rows (no LLM was consulted).
    model_identifier: str | None       # e.g., "anthropic/claude-opus-4-7"
    model_version: str | None          # provider-reported version string
    provider: str | None               # "anthropic", "openai", etc.
    composer_skill_hash: str | None    # SHA-256 of pipeline_composer.md
    arguments_hash: str | None         # rfc8785 hash over required fields; None until resolved
    hash_domain_version: str | None    # 'v1' once resolved; None for opt-out/pending
    interpretation_source: InterpretationSource
    # F-19: runtime model snapshot at resolve time (may differ from composer model).
    runtime_model_identifier_at_resolve: str | None
    runtime_model_version_at_resolve: str | None
    # Cross-DB hash anchor (Option A). NULL until resolved; NULL for
    # auto_interpreted_opt_out rows (no prompt template is patched). For
    # resolved user_approved rows, this is the SHA-256 of the resolved
    # prompt-template string, computed at resolve time using stable_hash()
    # from contracts/hashing.py. NOT part of INTERPRETATION_HASH_DOMAIN_V1.
    resolved_prompt_template_hash: str | None
```

**F-12 (hash domain versioning):** Also add to
`src/elspeth/contracts/composer_interpretation.py`:

```python
# INTERPRETATION_HASH_DOMAIN_V1: the closed set of fields used to compute
# arguments_hash for interpretation events. This constant is the source of
# truth for the v1 hash domain. The hashing function reads from this
# constant ONLY — adding a field to InterpretationEventRecord without
# adding it here leaves the new field out of the hash silently.
#
# To add a field: (1) add it here, (2) bump hash_domain_version to 'v2',
# (3) add a CI test that the new field is in the hash domain.
INTERPRETATION_HASH_DOMAIN_V1: frozenset[str] = frozenset({
    "session_id",
    "composition_state_id",
    "affected_node_id",
    "tool_call_id",
    "user_term",
    "llm_draft",
    "accepted_value",
    "actor",
    "model_identifier",
    "model_version",
    "provider",
    "composer_skill_hash",
})
```

Add a CI test (in `test_composer_interpretation.py`) that:
1. Creates an `InterpretationEventRecord` with all required fields.
2. Asserts that every field name in `INTERPRETATION_HASH_DOMAIN_V1` is
   a valid field name on `InterpretationEventRecord` (guard against typos).
3. Adds a new field to the dataclass (in the test, via monkey-patching or
   a subclass) and asserts the hashing function does NOT include the new
   field unless it is added to `INTERPRETATION_HASH_DOMAIN_V1`.

### Rationale

Opt-out rows are recorded as `InterpretationEventRecord` instances with
`choice=OPTED_OUT` and `interpretation_source=AUTO_INTERPRETED_OPT_OUT`,
with nullable interpretation fields (`composition_state_id`,
`affected_node_id`, `tool_call_id`, `user_term`, `llm_draft`,
`model_identifier`, `model_version`, `provider`, `composer_skill_hash`) set
to `None`. This eliminates the prior `InterpretationOptOutRecord` type and the
associated `proposal_events` routing. A single table is the single source of
truth for all interpretation-related decisions.

The `interpretation_source` field is a closed enum that records the structural
mechanism that produced the row: direct user approval, the opt-out gesture, or
the auto-interpret-no-surfaces path. It lives in `InterpretationSource`.

Audit provenance fields (`model_identifier`, `model_version`, `provider`,
`composer_skill_hash`, `arguments_hash`) snapshot what the composer LLM was
using at surfacing time so a future auditor can re-verify the prompt that
generated the draft interpretation. These are NULL for `auto_interpreted_opt_out`
rows (F-1: no LLM was consulted).

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

## Task 1.5 — Session DB hardening (F-9, F-10)

**Goal.** Extend `web/sessions/engine.py` with production-grade SQLite PRAGMA
settings (WAL, busy timeout, synchronous) and schema-version guards
(`SESSION_SCHEMA_EPOCH`, `application_id`, `user_version`). This is the
safety net for the operator-delete-DB policy and is tracked separately as
`elspeth-6815a49a7d` per memory `project_phase1a_panel_round_2_complete`.
Phase 5b co-lands it here because Phase 5b requires the session DB to be
production-ready before the new tables are exercised.

**Files:**

- Modify: `src/elspeth/web/sessions/engine.py`.
- Modify: `src/elspeth/web/sessions/models.py` — add `SESSION_SCHEMA_EPOCH`.

### PRAGMA settings (F-9)

Extend the `_enable_sqlite_foreign_keys` connect listener in
`create_session_engine` to mirror `core/landscape/database.py:271-275`:

```python
@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_conn, _record):
    cursor = dbapi_conn.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()
```

Extend the startup probe to also verify WAL mode:
```python
with engine.connect() as conn:
    fk = conn.execute(text("PRAGMA foreign_keys")).scalar_one()
    jm = conn.execute(text("PRAGMA journal_mode")).scalar_one()
if fk != 1:
    raise RuntimeError(...)
if jm != "wal":
    raise RuntimeError(
        f"Session engine {engine.url!r} did not enter WAL mode "
        f"(got {jm!r}). Refusing to start — production requires WAL."
    )
```

### Schema-version guards (F-10)

Add to `src/elspeth/web/sessions/models.py`:
```python
# SESSION_SCHEMA_EPOCH — schema version sentinel. Bump this constant
# whenever interpretation_events_table or any other table that requires
# a DB delete is added. The startup validator reads PRAGMA user_version
# and crashes if it does not match, giving the operator an actionable
# "delete the DB and restart" message rather than a cryptic SQLAlchemy
# error.
#
# Pattern mirrors SQLITE_SCHEMA_EPOCH in core/landscape/schema.py.
SESSION_SCHEMA_EPOCH = 2  # bumped from 1 by Phase 5b (interpretation_events_table)

# project-unique application_id (hex 0x454C5350 = "ELSP"):
SESSION_DB_APPLICATION_ID = 0x454C5350
```

Add to `initialize_session_schema` in `web/sessions/schema.py`:
```python
def _assert_schema_version(engine: Engine) -> None:
    """Crash with an actionable message if the session DB schema version
    does not match SESSION_SCHEMA_EPOCH.

    This is the mechanical enforcement of the operator-delete-DB policy.
    Without this guard, a stale DB silently fails in obscure ways when
    new tables are added. With it, the operator gets:
        SessionSchemaError: Session DB schema version mismatch ...
    and a clear instruction to delete the DB file and restart.
    """
    with engine.connect() as conn:
        app_id = conn.execute(text("PRAGMA application_id")).scalar_one()
        user_ver = conn.execute(text("PRAGMA user_version")).scalar_one()
    if app_id != 0 and app_id != SESSION_DB_APPLICATION_ID:
        raise SessionSchemaError(
            f"Session DB has unexpected application_id={app_id:#010x}. "
            f"Expected {SESSION_DB_APPLICATION_ID:#010x} (ELSP) or 0 "
            f"(new database). Delete the session DB file and restart."
        )
    if user_ver != 0 and user_ver != SESSION_SCHEMA_EPOCH:
        raise SessionSchemaError(
            f"Session DB schema version {user_ver} does not match "
            f"SESSION_SCHEMA_EPOCH={SESSION_SCHEMA_EPOCH}. "
            f"Delete the session DB file and restart."
        )
```

After `metadata.create_all()` for new DBs, set both PRAGMAs:
```python
with engine.connect() as conn:
    conn.execute(text(f"PRAGMA application_id = {SESSION_DB_APPLICATION_ID}"))
    conn.execute(text(f"PRAGMA user_version = {SESSION_SCHEMA_EPOCH}"))
    conn.commit()
```

### BEGIN IMMEDIATE note (F-25)

After F-9 lands the WAL PRAGMA, write-path service methods MUST open
transactions with `BEGIN IMMEDIATE` (not the default DEFERRED) to prevent
SQLITE_BUSY races where a WAL reader delays a writer. Document this in
a service-layer comment:

```python
# WAL + IMMEDIATE: under WAL journal mode, BEGIN DEFERRED allows
# concurrent readers but may delay the first write (SQLITE_BUSY) if
# another writer is active. BEGIN IMMEDIATE acquires the write lock
# immediately, serialising writers without blocking readers. All
# write-path methods in this class use BEGIN IMMEDIATE.
```

### Test shape

1. After `create_session_engine(url)` with a fresh in-memory DB, assert
   `PRAGMA journal_mode` returns `"wal"`.
2. Assert `PRAGMA busy_timeout` returns `5000`.
3. Assert `PRAGMA foreign_keys` returns `1`.
4. After `initialize_session_schema(engine)`, assert `PRAGMA application_id`
   returns `SESSION_DB_APPLICATION_ID` and `PRAGMA user_version` returns
   `SESSION_SCHEMA_EPOCH`.
5. Simulate a stale DB (wrong `user_version`) and assert
   `initialize_session_schema` raises `SessionSchemaError` with a message
   containing "Delete the session DB file and restart."

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

`sessions(engine): WAL + busy_timeout + synchronous PRAGMA + schema-epoch
guard (elspeth-6815a49a7d, co-landed by Phase 5b)`.

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
    # F-1: nullable=True — NULL for auto_interpreted_opt_out rows (no
    # composition state was involved); NOT NULL for user_approved rows.
    Column("composition_state_id", String, nullable=True),
    # The LLM transform's node_id within composition_states.nodes that
    # this interpretation binds into. Validated at the writer boundary
    # to exist; NOT a foreign key because nodes live inside a JSON
    # column, not a separate table.
    # F-1: nullable=True — NULL for auto_interpreted_opt_out rows.
    Column("affected_node_id", String, nullable=True),
    # The provider tool_call_id from the LLM call that surfaced this
    # interpretation. NOT a foreign key to chat_messages because the
    # tool call may still be in flight when this row is inserted.
    # F-1: nullable=True — NULL for auto_interpreted_opt_out rows.
    Column("tool_call_id", String, nullable=True),
    # Audit-mandatory fields (source-conditional: see CHECKs below):
    # NULL for auto_interpreted_opt_out rows (no LLM surfacing occurred);
    # NOT NULL for user_approved rows; NULL for auto_interpreted_no_surfaces rows.
    # F-1: all nullable=True — the CHECKs below enforce source-conditional presence.
    Column("user_term", Text, nullable=True),
    Column("llm_draft", Text, nullable=True),
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
    # F-1: nullable=True for model_identifier, model_version, provider,
    # composer_skill_hash — NULL for auto_interpreted_opt_out rows (no LLM
    # was consulted).
    Column("model_identifier", String, nullable=True),   # e.g., "anthropic/claude-opus-4-7"
    Column("model_version", String, nullable=True),       # provider-reported version string
    Column("provider", String, nullable=True),            # "anthropic", "openai", etc.
    Column("composer_skill_hash", String, nullable=True), # SHA-256 of pipeline_composer.md
    Column("arguments_hash", String, nullable=True),
    # NULL until resolved. For auto_interpreted_opt_out rows, arguments_hash
    # is NULL because there is no LLM-supplied content to hash.
    # F-12 (hash domain versioning): hash_domain_version records the field set
    # used to compute arguments_hash. The v1 field set is defined in
    # contracts/composer_interpretation.py:INTERPRETATION_HASH_DOMAIN_V1.
    # NULL for rows without a hash (opt-out, pending). NOT NULL once resolved.
    Column("hash_domain_version", String, nullable=True),  # 'v1' once resolved
    # Structural source of this row. Closed enum — see governance ceremony below.
    Column("interpretation_source", String, nullable=False),
    # F-19 (runtime model snapshot at resolve time): nullable columns populated
    # when the user resolves the event, capturing what model the affected LLM
    # transform will use at runtime (for audit drift detection).
    Column("runtime_model_identifier_at_resolve", String, nullable=True),
    Column("runtime_model_version_at_resolve", String, nullable=True),
    # Cross-DB hash anchor (Option A — see §"Hash-anchored cross-DB linkage"
    # in sibling doc 18-phase-5b-surface-llm-interpretation.md).
    # Populated by resolve_interpretation_event at the same time as accepted_value
    # is committed. NULL until resolved; NULL for auto_interpreted_opt_out rows
    # (no prompt template is patched for opt-out rows). For user_approved and
    # auto_interpreted_no_surfaces rows that have a resolved prompt template,
    # this is SHA-256 over the rfc8785 canonical JSON of the resolved
    # prompt-template string, using CANONICAL_VERSION = "sha256-rfc8785-v1"
    # (contracts/hashing.py). NOT part of INTERPRETATION_HASH_DOMAIN_V1 —
    # it covers a different input (the resolved prompt string) and serves as
    # a cross-DB anchor only.
    Column("resolved_prompt_template_hash", String(64), nullable=True),
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
    # F-1 (source-keyed nullability CHECK): rows with
    # interpretation_source = 'auto_interpreted_opt_out' have no LLM
    # context; all nine interpretation fields must be NULL.
    # Rows with interpretation_source = 'user_approved' have LLM context;
    # all nine must be NOT NULL.
    # Rows with interpretation_source = 'auto_interpreted_no_surfaces' had
    # the LLM consulted (rate-cap fallback) but no surfacing; they carry
    # LLM provenance fields but NULL for composition_state_id /
    # affected_node_id / tool_call_id / user_term / llm_draft.
    # This CHECK replaces the prior choice-keyed ck_interpretation_events_opted_out_nullability.
    CheckConstraint(
        "(interpretation_source = 'auto_interpreted_opt_out') = "
        "(composition_state_id IS NULL AND affected_node_id IS NULL AND "
        " tool_call_id IS NULL AND user_term IS NULL AND llm_draft IS NULL AND "
        " model_identifier IS NULL AND model_version IS NULL AND "
        " provider IS NULL AND composer_skill_hash IS NULL)",
        name="ck_interpretation_events_source_nullability",
    ),
    # user_approved rows must have all nine fields populated.
    CheckConstraint(
        "(interpretation_source != 'user_approved') OR "
        "(composition_state_id IS NOT NULL AND affected_node_id IS NOT NULL AND "
        " tool_call_id IS NOT NULL AND user_term IS NOT NULL AND llm_draft IS NOT NULL AND "
        " model_identifier IS NOT NULL AND model_version IS NOT NULL AND "
        " provider IS NOT NULL AND composer_skill_hash IS NOT NULL)",
        name="ck_interpretation_events_user_approved_required",
    ),
    # auto_interpreted_no_surfaces rows: the five surface fields (composition_state_id,
    # affected_node_id, tool_call_id, user_term, llm_draft) must be NULL (no surfacing
    # occurred); the four LLM provenance fields must be NOT NULL (the LLM was consulted
    # for the rate-cap fallback auto-interpretation). This is the middle shape between
    # opted_out (all nine NULL) and user_approved (all nine NOT NULL).
    CheckConstraint(
        "(interpretation_source != 'auto_interpreted_no_surfaces') OR "
        "(composition_state_id IS NULL AND affected_node_id IS NULL AND "
        " tool_call_id IS NULL AND user_term IS NULL AND llm_draft IS NULL AND "
        " model_identifier IS NOT NULL AND model_version IS NOT NULL AND "
        " provider IS NOT NULL AND composer_skill_hash IS NOT NULL)",
        name="ck_interpretation_events_no_surfaces_shape",
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
# F-26: See web/sessions/schema.py:_validate_partial_index_dialect_symmetry
# for the schema-validator gate that enforces both sqlite_where and
# postgresql_where are set consistently.
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

# F-11: index on composition_state_id for the common lookup pattern
# "all interpretation events for this composition state".
# Verify with EXPLAIN QUERY PLAN on
# SELECT * FROM interpretation_events WHERE composition_state_id = ?
# and assert SEARCH USING INDEX in the GREEN step.
Index(
    "ix_interpretation_events_composition_state",
    interpretation_events_table.c.composition_state_id,
)
```

### skill_markdown_history table (F-5c)

Append to `models.py` after `interpretation_events_table` (and its indexes):

```python
# skill_markdown_history — content-addressed archive of every distinct
# pipeline_composer.md version seen at runtime.
#
# One row per (SHA-256 hash, filename) pair. The compose loop upserts
# (INSERT OR IGNORE) on first use of a hash, capturing the exact text
# that was in memory when the LLM was prompted. This makes every
# composer_skill_hash on interpretation_events rows forensically
# traceable: an auditor can retrieve the exact skill prompt from this
# table.
#
# Storage cost is negligible — one row per distinct deploy of the skill
# markdown. Content is TEXT (not BLOB) because the skill file is UTF-8
# Markdown.
skill_markdown_history_table = Table(
    "skill_markdown_history",
    metadata,
    Column("hash", String, primary_key=True),   # SHA-256 hex, 64 chars
    Column("filename", String, nullable=False),  # e.g., "pipeline_composer.md"
    Column("content", Text, nullable=False),     # full UTF-8 Markdown content
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
)
```

### Runtime Landscape `calls` table — `resolved_prompt_template_hash` column

**Layer note.** This is an **L1 Landscape schema change** (`core/landscape/schema.py`).
The prior framing that "Phase 5b does not touch the Landscape" is superseded here.
Phase 5b now touches one L1 column addition. No new cross-layer imports are
introduced (the column is a nullable `String(64)` data column on an existing
table); the enforce_tier_model gate continues to pass. This is an additive
data-column change only, not a new module-level dependency.

**Column to add.** In `src/elspeth/core/landscape/schema.py`, inside
`calls_table`, add the following column **after** the existing `response_ref`
column (currently line ~327):

```python
# Cross-DB hash anchor for interpretation events (Option A — Phase 5b).
# Populated by the LLM-transform plugin at execution time when the
# runtime node config contains a `resolved_prompt_template_hash` sibling
# field (written by resolve_interpretation_event at compose time and
# committed into composition_states.nodes). If the sibling field is absent
# (the LLM transform is NOT downstream of an interpretation event), this
# column is NULL.
#
# When non-NULL, this hash MUST equal the corresponding
# interpretation_events.resolved_prompt_template_hash in the session audit
# DB for the same resolved prompt string. An inequality indicates tampering
# or a composition-to-execution coherence failure. Checked by the audit-
# tooling layer; a mismatch is a Tier-1 crash-on-anomaly (see failure-mode
# table below).
#
# Hash scheme: SHA-256 over rfc8785 canonical JSON of the resolved
# prompt-template string, using CANONICAL_VERSION = "sha256-rfc8785-v1"
# (contracts/hashing.py:CANONICAL_VERSION). Identical scheme used by both
# the session service (write at resolve time) and the runtime plugin
# (write at execution time), so the hashes are comparable byte-for-byte.
Column("resolved_prompt_template_hash", String(64), nullable=True),
```

**Index.** Add a supporting index in the same `schema.py` file, alongside the
existing `ix_calls_state` and `ix_calls_operation` indexes:

```python
Index("ix_calls_resolved_prompt_template_hash", calls_table.c.resolved_prompt_template_hash)
```

Rationale: the primary audit-tooling query is "given a session-side
interpretation_events.resolved_prompt_template_hash, find the matching Landscape
calls row" — a lookup by hash value. Without this index that query is a full
table scan on `calls`. With it, the query is O(log n). The index is sparse on
NULL (SQLite does not include NULL keys in B-tree indexes by default), so the
storage cost is proportional to the number of LLM-transform calls that are
downstream of an interpretation event — typically a small fraction of all calls.

**Runtime plugin read path.** The LLM-transform plugin reads its node config at
execution time. The node config is the JSON object from `composition_states.nodes`
for the affected node. If the node config contains a top-level key
`resolved_prompt_template_hash`, the plugin reads that value directly and writes
it to `calls.resolved_prompt_template_hash` without re-computation. If the key
is absent (the LLM transform was not downstream of an interpretation event), the
plugin writes NULL. The plugin MUST NOT re-compute the hash from the prompt
string at runtime — the stored value is the authority. Re-computation would
diverge if the rfc8785 library version changes between compose time and
execution time, breaking the hash-equality invariant silently.

Precise read path in plugin code:

```python
resolved_prompt_template_hash: str | None = node_config.get(
    "resolved_prompt_template_hash"
)
# Write to calls row:
#   calls_table.c.resolved_prompt_template_hash = resolved_prompt_template_hash
```

The use of `.get()` here is intentional and Tier-2 compliant: `node_config` is
post-source pipeline data (elevated trust); the key's absence is a normal
condition (not every LLM transform has an interpretation event), not a bug. A
missing key → NULL is semantically correct. This is the one permitted use of
`.get()` for this field; do not use `.get()` for Tier-1 reads.

**Schema bootstrap note.** This column is added to the Landscape `calls_table`
SQLAlchemy `Table` definition. Because the Landscape DB is governed by the same
delete-the-DB policy as the session DB (per `project_db_migration_policy`), the
operator MUST delete BOTH the session DB AND the Landscape DB (the `audit.db`
file) on Phase 5b deploy. This plan's Step 4 commit message MUST mention the
Landscape DB delete requirement explicitly. The "Migration runner ownership"
section above covers the cumulative DB-delete impact and Phase 9 ownership.

**Step 4 commit message extension.** Amend the existing commit-message template
to include the Landscape schema change:

```
sessions(schema): add interpretation_events_table + interpretation_resolve
provenance + interpretation_review_disabled column + append-only trigger +
resolved_prompt_template_hash column on calls_table (L1 Landscape).
OPERATOR: delete BOTH session DB and Landscape audit.db on deploy.
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
in the same migration or schema bootstrap step. In SQLAlchemy, use a
**table-scoped** listener with `IF NOT EXISTS` for idempotent bootstrap (F-23):

```python
from sqlalchemy import DDL, event

event.listen(
    interpretation_events_table,
    'after_create',
    DDL(
        "CREATE TRIGGER IF NOT EXISTS trg_interpretation_events_immutable_resolved "
        "BEFORE UPDATE ON interpretation_events "
        "FOR EACH ROW BEGIN "
        "  SELECT CASE "
        "    WHEN OLD.resolved_at IS NOT NULL AND ("
        "      NEW.accepted_value IS NOT OLD.accepted_value OR "
        "      NEW.resolved_at IS NOT OLD.resolved_at OR "
        "      NEW.actor IS NOT OLD.actor OR "
        "      NEW.choice IS NOT OLD.choice"
        "    ) THEN RAISE(ABORT, 'interpretation_events: resolved rows are immutable') "
        "  END; "
        "END;"
    ),
)
```

Use `table.after_create` (table-scoped), not `metadata.after_create`
(metadata-scoped). The table-scoped form fires only when this specific
table is created, not on every `metadata.create_all()` call for tables
that already exist. `IF NOT EXISTS` makes the DDL idempotent across
repeated calls.

Document the trigger in the `models.py` governance block alongside the
table definition.

### chat_messages immutability trigger (F-4)

Phase 5a elevates `chat_messages` to an audit anchor via
`created_from_message_id`. Add a BEFORE UPDATE trigger to prevent
content mutation after the fact:

```python
event.listen(
    chat_messages_table,
    'after_create',
    DDL(
        "CREATE TRIGGER IF NOT EXISTS trg_chat_messages_immutable_content "
        "BEFORE UPDATE OF content ON chat_messages "
        "BEGIN "
        "  SELECT RAISE(ABORT, 'chat_messages.content is append-only'); "
        "END;"
    ),
)
```

This trigger is in 18a- (backend) even though the table is 5a-owned
because 18a- carries all schema work for the Phase 5b umbrella PR. The
trigger prevents post-hoc content editing that would invalidate the
`created_from_message_id` lineage audit.

### Schema validator extension for triggers (F-24)

Extend `_validate_current_schema` in `web/sessions/schema.py` to query:

```python
triggers = {
    row[0]
    for row in conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='trigger'")
    )
}
expected_triggers = {"trg_interpretation_events_immutable_resolved",
                     "trg_chat_messages_immutable_content"}
if not expected_triggers.issubset(triggers):
    raise SessionSchemaError(
        f"Missing triggers: {expected_triggers - triggers}"
    )
```

This catches the case where schema bootstrap succeeds but the trigger DDL
failed silently (e.g., if the DDL event listener was removed or reordered).

### Two-source-of-truth note (F-35)

Add a governance comment alongside the `interpretation_review_disabled`
column in `models.py`:

```python
# Two-source-of-truth tension: the opted-out state is represented
# both by this boolean column (fast-path read by the compose loop)
# and by the existence of a row in interpretation_events with
# choice='opted_out' (authoritative audit record). They MUST remain
# consistent. The service's write path sets both atomically within
# a single transaction (_session_write_lock held throughout).
#
# If a future audit finds the boolean true but no opted_out row exists,
# that is a bug in the service's write path — crash on read
# (offensive programming). The boolean is a read-cache; the
# interpretation_events row is the source of truth.
#
# F-35 follow-up: consider enforcing this constraint with a trigger or
# computed column in a future schema revision. Deferred because
# SQLite does not support CHECK constraints that cross tables.
```

### Phase 9 migration notes (F-16)

Add a commented block to `models.py` near `interpretation_events_table`:

```python
# Phase 9 migration notes (F-16):
# The following four interdependent DDL objects must be recreated together
# when the schema is migrated (Phase 9's migration runner replaces the
# delete-the-DB policy for production):
#
# 1. interpretation_events_table — the main table
# 2. sessions_table.interpretation_review_disabled column
# 3. composition_states.provenance CHECK extension (add 'interpretation_resolve')
# 4. trg_interpretation_events_immutable_resolved trigger
# 5. trg_chat_messages_immutable_content trigger (F-4)
# 6. skill_markdown_history_table
#
# SQLite recreation sequence for the composition_states provenance CHECK
# extension (SQLite does not support ALTER TABLE ... ADD CONSTRAINT):
#   1. BEGIN IMMEDIATE;
#   2. CREATE TABLE composition_states_new (..., CHECK(provenance IN (..., 'interpretation_resolve')));
#   3. INSERT INTO composition_states_new SELECT * FROM composition_states;
#   4. DROP TABLE composition_states;
#   5. ALTER TABLE composition_states_new RENAME TO composition_states;
#   6. COMMIT;
# The Phase 9 migration runner MUST run this sequence atomically and
# verify row count before DROP. Reference: project_db_migration_policy.md.
```

### Test shape

`tests/unit/web/sessions/test_interpretation_events_table.py`:

1. Test that `metadata.create_all(engine)` succeeds against an in-memory
   SQLite engine and produces a table named `interpretation_events`
   with all the expected columns (including `model_identifier`,
   `model_version`, `provider`, `composer_skill_hash`, `arguments_hash`,
   `interpretation_source`, `hash_domain_version`,
   `runtime_model_identifier_at_resolve`, `runtime_model_version_at_resolve`,
   `resolved_prompt_template_hash`).
2. Test that inserting a row with `choice='pending'` and `resolved_at`
   set raises `IntegrityError`.
3. Test that inserting a row with `choice='accepted_as_drafted'` and
   `accepted_value=NULL` raises `IntegrityError`.
4. Test that inserting a row with `choice='opted_out'` and all nullable
   fields (`composition_state_id`, `affected_node_id`, `tool_call_id`,
   `user_term`, `llm_draft`, `model_identifier`, `model_version`,
   `provider`, `composer_skill_hash`) set to NULL, and
   `interpretation_source='auto_interpreted_opt_out'` succeeds.
4a. Test that inserting an opted_out row with `composition_state_id`
    non-NULL raises `IntegrityError`
    (ck_interpretation_events_source_nullability — the new source-keyed CHECK).
4b. **F-1 new test:** Test that inserting a `user_approved` row with
    `user_term` NULL raises `IntegrityError`
    (ck_interpretation_events_user_approved_required). Verify both
    directions: opted_out + non-NULL user_term → IntegrityError;
    user_approved + NULL user_term → IntegrityError.
4c. **auto_interpreted_no_surfaces shape (F-1 third CHECK):** Test that
    inserting an `auto_interpreted_no_surfaces` row with `user_term` non-NULL
    raises `IntegrityError` (ck_interpretation_events_no_surfaces_shape).
    Test that an `auto_interpreted_no_surfaces` row with `user_term` NULL and
    `model_identifier` non-NULL succeeds. Test that an
    `auto_interpreted_no_surfaces` row with `model_identifier` NULL raises
    `IntegrityError` (the provenance fields are required for this source).
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
11. **F-8 (trigger-existence via production bootstrap path):** After calling
    `metadata.create_all(engine)` via the production bootstrap path
    (i.e., call `initialize_session_schema(engine)` — do NOT create the
    table manually), assert:
    ```sql
    SELECT name FROM sqlite_master
    WHERE type='trigger' AND name='trg_interpretation_events_immutable_resolved'
    ```
    returns a non-empty result. Same assertion for
    `trg_chat_messages_immutable_content`. This is the test that the
    trigger is created by the correct SQLAlchemy event hook, not just
    that it exists in the SQL blob.
12. **F-11 (composition_state_id index):** Run `EXPLAIN QUERY PLAN
    SELECT * FROM interpretation_events WHERE composition_state_id = ?`
    after `metadata.create_all(engine)` and assert the plan contains
    "SEARCH USING INDEX" (not a full table scan). This is the GREEN-step
    verification for the new index.

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
+ append-only trigger + resolved_prompt_template_hash on calls_table (L1 Landscape).
OPERATOR: delete BOTH session DB and Landscape audit.db on deploy.`
The commit message MUST mention all five schema changes: the new table, the
provenance enum extension, the sessions boolean column, the immutability
trigger, and the `calls_table` column in L1 Landscape. The Landscape DB
delete requirement MUST appear in the commit message so it is discoverable
from `git log`. NOTE: `interpretation.opted_out` is NOT added to
`proposal_events.event_type` — opt-out rows go to `interpretation_events_table`
(see opt-out pivot above).

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
    hash_domain_version: str | None = Field(default=None, max_length=16)  # 'v1' once resolved (F-12)
    # F-19: runtime model snapshot at resolve time.
    runtime_model_identifier_at_resolve: str | None = Field(default=None, max_length=256)
    runtime_model_version_at_resolve: str | None = Field(default=None, max_length=128)
    # Cross-DB hash anchor (Option A). hex SHA-256 of the resolved prompt-template
    # string; None until resolved, None for opted_out rows. Exposed on the wire so
    # audit-tooling consumers can verify hash equality without a second DB round-trip.
    resolved_prompt_template_hash: str | None = Field(default=None, max_length=64)


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
        # NOTE: F-2 (prompt-injection bypass) — the same four checks MUST also
        # be applied to llm_draft at the tool boundary (Task 5) so that the
        # accepted_as_drafted path cannot bypass them. This validator is the
        # service-layer defense-in-depth for amended_value; the tool-boundary
        # call is the primary guard for llm_draft.
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

Implement. `_reject_credential_shaped_content` and the shared content-check
helpers (`_validate_accepted_value_content`) live in the EXISTING module
`src/elspeth/web/validation.py` (F-34: this module already exists with
`has_visible_content`, `validate_secret_name`, and related helpers; extend
it rather than creating a new `_validation_helpers.py`). Extending the
existing module avoids peer cross-imports between `web/sessions/schemas.py`
and `web/composer/tools.py`. The helpers run the credential prefilter
regexes listed in Task 5's "Credential-shape prefilter" section and raise
`ValueError` with a user-visible message ("That looks like a credential —
please re-enter without secrets") on any match. Import with:
`from elspeth.web.validation import _reject_credential_shaped_content,
_validate_accepted_value_content`.

Also add to the test suite (test item 11): `amended_value` containing a
benign prose sentence with periods (e.g., "The term 'cool' means visually
appealing.") does NOT raise from the credential prefilter (JWT false-positive
regression test — F-32 applied at the schema validation layer too).

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
    amended_value: str | None,     # None when choice == 'accepted_as_drafted' (F-14)
    actor: str,
    resolved_at: datetime | None = None,
    runtime_model_identifier: str | None = None,   # from affected LLM transform (F-19)
    runtime_model_version: str | None = None,       # from affected LLM transform (F-19)
) -> tuple[InterpretationEventRecord, CompositionStateRecord]:
    """Commit a resolution AND patch the affected LLM transform's
    prompt template.

    F-14 (business-rule split): `accepted_value` is computed internally.
    When choice == 'accepted_as_drafted', the service reads the pending
    event's `llm_draft` from the DB and uses that as `accepted_value`.
    When choice == 'amended', `amended_value` is used directly.
    The route passes only `choice` and `amended_value` — the computation
    lives here to avoid duplicating the branch across callers.

    Single transaction (F-25: write transactions use BEGIN IMMEDIATE to
    prevent SQLITE_BUSY races when WAL mode is active and two writers
    contend; mirror the pattern at other write-path service methods):
        1. SELECT the pending event by id AND session_id AND choice='pending'
           (F-7: the WHERE must include session_id to prevent cross-session
           IDOR; choice='pending' is the TOCTOU guard against double-resolve).
           Raise ValueError (→ 404 at route) if no matching row.
        2. Compute accepted_value per F-14 rule above.
        3. Validate accepted_value via _validate_accepted_value_content
           (defense-in-depth against future callers that bypass the route).
        4. Call `_patch_llm_transform_prompt` to produce the resolved
           prompt-template string (substituting the `{{interpretation:<term>}}`
           placeholder with accepted_value). This is the string that will be
           embedded in `composition_states.nodes` and executed verbatim at
           runtime. Do NOT call this helper a second time in step 5a — keep
           a local reference to the return value from this call.
        4a. Compute `resolved_prompt_template_hash` as
            `stable_hash(resolved_prompt_template_string)` using
            `contracts/hashing.py:stable_hash`. This is a SHA-256 over the
            rfc8785 canonical JSON of the resolved string, using
            `CANONICAL_VERSION = "sha256-rfc8785-v1"`. This hash is NOT part
            of `INTERPRETATION_HASH_DOMAIN_V1` — it covers a different input
            (the resolved prompt string only) and serves as a cross-DB anchor,
            not as the event's identity hash. Do not add it to the hash domain.
        5. UPDATE interpretation_events SET choice=?, accepted_value=?,
           resolved_at=?, actor=?, arguments_hash=?,
           runtime_model_identifier_at_resolve=?,
           runtime_model_version_at_resolve=?,
           resolved_prompt_template_hash=?
           WHERE id=? AND session_id=? AND choice='pending'.
           (Double WHERE guard is belt-and-suspenders; the SELECT above
           already verified the row exists, but the UPDATE's WHERE guards
           against an MVCC race window.)
        5a. Write the resolved prompt-template string (from step 4) AND the
            hash (from step 4a) into `composition_states.nodes` for the
            affected node. The node JSON gains two sibling fields alongside
            the patched `prompt_template`:
              `"prompt_template"`: resolved string (the patched result)
              `"resolved_prompt_template_hash"`: the hash from step 4a
            Produce a new composition_states row with provenance =
            'interpretation_resolve', version += 1.
            Ordering invariant: patch first (step 4), hash second (step 4a),
            write both sinks (steps 5 and 5a) in the same transaction. The
            runtime plugin reads `resolved_prompt_template_hash` from the
            nodes JSON sibling field; if absent, the column is left NULL.
            A runtime that re-computes the hash at execution time would risk
            divergence if rfc8785 normalization changes; reading the stored
            value avoids that risk.
        6. Return the resolved event + the new state.

    Acquires the session write lock (F-27). Raises ValueError if the event
    is already resolved (TOCTOU lost), if the affected node has
    disappeared from the composition state since the surfacing, or
    if the prompt-template patch fails (e.g., node is no longer an
    LLM transform).

    Trigger error note (F-28): if the immutability trigger fires (row
    is already resolved and the UPDATE attempts to overwrite settled
    fields), SQLAlchemy raises IntegrityError with the trigger's
    RAISE(ABORT, ...) message. The service-layer error classifier MUST
    match this specific message string explicitly so it is mapped to a
    409/400 response, not conflated with a generic integrity violation
    (which would emit a spurious security telemetry signal).

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

    F-27 (write-lock annotation): acquires the session write lock for the
    ENTIRE duration of the transaction — both the interpretation_events
    INSERT and the sessions boolean UPDATE must be inside one
    _session_write_lock block to ensure atomicity.

    Idempotency (F-29): if an opted_out row already exists for this
    session, return the existing record without inserting a duplicate.
    The sessions boolean remains true. First opt-out timestamp is
    authoritative.

    Writes a row to interpretation_events_table with choice='opted_out',
    interpretation_source='auto_interpreted_opt_out', all nullable
    interpretation fields (composition_state_id, affected_node_id,
    tool_call_id, user_term, llm_draft, model_identifier, model_version,
    provider, composer_skill_hash, accepted_value, arguments_hash)
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

The prefilter rejects content matching (credential-shaped — raises ARG_ERROR):
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

The following patterns emit a WARNING via operational telemetry (not a
rejection — these may be legitimate user input) and are applied to
`user_term` and `llm_draft` (F-20):
- Email addresses: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
- Phone numbers: `(\+?1?\s?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})`
- SSN-like: `\d{3}[-\s]\d{2}[-\s]\d{4}` (broader form — also covered
  by the rejection list above; the warning form fires before the reject)

Defer name/address PII detection to Phase 11 (document the deferral
explicitly in a code comment adjacent to the prefilter function in
`web/validation.py`).

Telemetry posture for PII warnings: emit as operational telemetry signal
`"interpretation_pii_candidate_detected"` with `field` (user_term/llm_draft)
and `pattern_name` (email/phone/ssn_like) but WITHOUT the field value.

The shared regex set lives in
`src/elspeth/web/validation.py` (existing module — extend with
`_reject_credential_shaped_content` and `_warn_pii_shaped_content`
helpers; do NOT create a new `_validation_helpers.py` — F-34).
The content-check helpers added for `amended_value` (Task 3) live in
the same module (`_validate_accepted_value_content`). Import with:
`from elspeth.web.validation import _reject_credential_shaped_content`.
This avoids a peer cross-import between `web/sessions/schemas.py` and
`web/composer/tools.py`.

**F-2 (prompt-injection bypass):** `_validate_accepted_value_content` MUST
also be applied to `llm_draft` at this tool boundary — BEFORE calling
`create_pending_interpretation_event`. This catches a poisoned `llm_draft`
(e.g., containing `{{system:override}}`) before it enters the audit DB
and before it reaches the `accepted_as_drafted` resolution path where it
would be embedded directly into the prompt template. Return ARG_ERROR
(not a service exception) so the LLM can re-draft. This is the preferred
rejection point because it is earlier and more visible to the LLM.

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

**Rate-limit window semantics (F-30):** The "per day" window is UTC
midnight (not a sliding 24-hour window). The daily count is the number
of `interpretation_events` rows for the session with `created_at >=
today_utc_midnight`. This is simpler to reason about for operators and
produces predictable reset behaviour. Document this choice in a comment
adjacent to the window-computation in the implementation. Tests must set
`created_at` explicitly and use a deterministic `now` fixture to avoid
clock-dependent failures.

**Rate-limit settings keys (F-31):** Move the numeric constants to
settings:
```
composer.interpretation_rate_limit_per_term = 3      # max surfacings per (session, term)
composer.interpretation_rate_limit_per_session_day = 10  # max per session per UTC day
```
Both must be read from the settings object at compose-loop initialisation;
`_check_interpretation_rate_limits` receives them as keyword arguments so
it is testable without live settings. Document the redeploy cost in a
comment: changing these limits requires a service restart (they are read
at startup, not per-request).

On cap exceeded, the compose loop sees an ARG_ERROR and is expected
(per Task 8 skill nudge) to fall back to a non-LLM interpretation with
`interpretation_source='auto_interpreted_no_surfaces'`.

**AUTO_INTERPRETED_NO_SURFACES writer (F-6):** When the rate cap is
exceeded (either limit), the compose loop MUST write an
`interpretation_events` row with:
- `interpretation_source = 'auto_interpreted_no_surfaces'`
- `choice = 'opted_out'` semantics (resolved-at-write)
- Interpretation surface fields NULL: `composition_state_id`,
  `affected_node_id`, `tool_call_id`, `user_term`, `llm_draft` (the
  rejected tool request never produced a draft or a composition-state
  binding)
- **LLM provenance fields NOT NULL** — `model_identifier`,
  `model_version`, `provider`, `composer_skill_hash` are read from the
  compose-loop snapshot at the time of the tool call, the identical
  source used for `user_approved` rows. The composer LLM that triggered
  the rate cap is fully identifiable; recording it is mandatory.
- `arguments_hash` populated (rfc8785 over the available field subset,
  with `hash_domain_version` reflecting the reduced domain)
- `resolved_at` set to now (the rate-cap event is itself a resolution)

**Rationale:** LLM provenance is required because the composer LLM that
triggered the rate cap is identifiable from the compose-loop snapshot
at the time of the tool call. The interpretation surface fields are NULL
because the rejected request never produced a draft. This matches
`ck_interpretation_events_no_surfaces_shape` (lines 853-859) and the
schema comment (lines 826-829): "the LLM was consulted (rate-cap
fallback) but no surfacing; they carry LLM provenance fields but NULL
for composition_state_id / affected_node_id / tool_call_id / user_term
/ llm_draft." Contrast with `auto_interpreted_opt_out`, where the LLM
was never consulted and all nine fields are NULL.

This makes the rate-cap event explicitly auditable: an auditor can
distinguish "user opted out" from "rate cap exhausted" via
`interpretation_source`. Add this writer call to the compose-loop hook's
rate-cap-exceeded branch, in `web/composer/service.py`. Specify
`create_pending_interpretation_event` or a dedicated
`record_auto_interpreted_no_surfaces_event` service method — prefer the
dedicated method to make the intent legible.

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

**Skill hash atomicity (F-5a):** The compose-loop closure MUST compute
`composer_skill_hash` from the **in-memory string** the LLM was prompted with,
not from a re-read of disk. The `@lru_cache` skill loader (see Task 8) MUST
return both the text and its SHA-256 hash as a frozen tuple:
```python
@lru_cache(maxsize=1)
def _load_composer_skill() -> tuple[str, str]:  # (text, sha256_hex)
    text = (Path(__file__).parent / "skills" / "pipeline_composer.md").read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()
```
The hash and text are thereby atomically consistent: the hash the DB records
is the hash of exactly the text the LLM saw. At compose-loop startup, assert:
```python
assert hashlib.sha256(skill_text.encode()).hexdigest() == skill_hash
assert hashlib.sha256(Path("...pipeline_composer.md").read_text().encode()).hexdigest() == skill_hash
```
If the on-disk file has changed since the LRU cache populated (indicating a
hot-reload partial state), crash with an operator-actionable message:
"Composer skill hash mismatch: in-memory hash differs from on-disk file. Restart
elspeth-web.service to reload."

**skill_markdown_history upsert (F-5c):** On first use of a `(skill_hash, compose_loop_init)`,
upsert into `skill_markdown_history`:
```sql
INSERT OR IGNORE INTO skill_markdown_history
  (hash, filename, content, first_seen_at)
VALUES (:hash, 'pipeline_composer.md', :text, :now)
```
This is a best-effort upsert (IGNORE on conflict); do not make it transactional
with the interpretation event row. Storage cost is negligible (one row per
distinct skill version).

**Unresolved placeholder runtime detection (F-17):** Add a sub-task in
`web/composer/service.py` (or a new helper in the executor path): before
executing any LLM transform, inspect its `prompt_template` for any
`{{interpretation:…}}` placeholder. If found:
- Raise RuntimeError: "Unresolved interpretation placeholder
  '{{interpretation:<term>}}' in LLM transform '<node_id>'. Resolve via
  /interpretations/<event_id>/resolve before running the pipeline."
- Surface as a user-actionable error (not a pipeline crash — block
  execution, return error to frontend).
- Emit operational telemetry signal `"interpretation_placeholder_unresolved_at_runtime"`
  with `node_id` and `term` (but NOT the prompt_template value).

Specify the detection helper in the plan: it should be a standalone function
`_detect_unresolved_interpretation_placeholders(nodes: dict) -> list[str]`
that returns the list of terms with unresolved placeholders. An empty list
means no placeholders; non-empty list means blocked execution. This helper
is tested independently.

**Runtime telemetry signal (F-21):** When an LLM transform's prompt template
contains an unresolved `{{interpretation:…}}` placeholder at execution time
(detected above), emit:
- Signal name: `"interpretation_placeholder_unresolved_at_runtime"`
- Posture: operational telemetry (not audit-primary; the runtime error
  is the primary record)
- Fields: `node_id`, `term` (NOT the prompt template value)
- Purpose: catches LLM under-firing post-model-upgrade without offline
  eval refresh

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
12. **F-2 (llm_draft prompt-injection at tool boundary):** Calling with
    `llm_draft` containing `{{system:override}}` raises `ToolArgumentError`
    (metacharacter check from `_validate_accepted_value_content` applied
    to llm_draft at the tool boundary). No DB write occurs.
13. **F-18 (dual-registry dispatch invariant):** Enumerate all registered
    tools: assert each tool name appears in EXACTLY one registry
    (`_TOOL_HANDLERS` XOR `_SESSION_AWARE_TOOL_HANDLERS`). Assert every
    handler in `_SESSION_AWARE_TOOL_HANDLERS` satisfies
    `asyncio.iscoroutinefunction(h) == True`. Assert every handler in
    `_TOOL_HANDLERS` satisfies `asyncio.iscoroutinefunction(h) == False`.
    This is the mechanical guard against "async tool accidentally in
    sync registry" silent failures.
14. **F-32 (JWT benign-period negative test):** Calling with `llm_draft`
    containing `"The term 'cool' means visually appealing, well-organized,
    and easy to use."` does NOT raise from the credential prefilter
    (benign periods in a prose sentence must not trigger the JWT pattern).
15. **Rate-limit test (UTC window, F-30):** The rate-limit window resets
    at UTC midnight. Test: inject a frozen `now` fixture set to
    23:59:59 UTC; create 10 events with `created_at` in the current UTC
    day. Advance `now` by 1 second past midnight; assert the per-session-day
    cap is reset (11th call succeeds in the new UTC day).
16. **Telemetry posture (F-15):** Rate-limit cap breach emits operational
    telemetry signal `"interpretation_rate_cap_exceeded"` with `cap_type`
    (per_term/per_session_day) and `session_id` (but NOT `user_term` in
    telemetry — PII risk). Assert the telemetry call is made before the
    ARG_ERROR is returned; assert no Landscape/audit row is written.

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
    # F-14 (business-rule split): the route passes only choice and amended_value.
    # The service computes accepted_value from llm_draft internally when
    # choice == 'accepted_as_drafted'. Do NOT compute accepted_value here.
    event, new_state = await service.resolve_interpretation_event(
        session_id=session_id,
        event_id=event_id,
        choice=InterpretationChoice(body.choice),
        amended_value=body.amended_value,  # None when choice == 'accepted_as_drafted'
        actor=actor,
    )
    return InterpretationResolveResponse(
        event=_interpretation_event_response(event),
        new_state=_composition_state_response(new_state),
    )
```

**Design note (F-14):** The computation "if `accepted_as_drafted`, copy `llm_draft` as the
`accepted_value`" belongs in `resolve_interpretation_event` (Task 4), not the route. The route
is a Tier-3 boundary — it validates the request shape and passes `choice` + `amended_value` to
the service. The service reads the pending event's `llm_draft` from the DB and applies the
business rule in one transaction. This prevents caller drift: a future second caller
(e.g., a CLI, an admin route) gets the same rule without duplicating the branch.

### Runtime model snapshot (F-19)

At resolve time, populate `runtime_model_identifier_at_resolve` and
`runtime_model_version_at_resolve` on the `interpretation_events_table` row
from the affected LLM transform's model config in the current composition state.
These are nullable columns (see Task 2); if the composition state's LLM transform
has no explicit model config, they remain NULL. Add to the service method call:

```python
    event, new_state = await service.resolve_interpretation_event(
        ...
        runtime_model_identifier=_extract_transform_model_identifier(new_state, event.affected_node_id),
        runtime_model_version=_extract_transform_model_version(new_state, event.affected_node_id),
    )
```

The audit-readiness panel (Task 10) emits a warning when the runtime model on the
composition state differs from `model_identifier` recorded at surfacing time
(the composer LLM may have changed between surfacing and resolve if the operator
rotated models). This is a reviewer signal, not a blocker.

### Telemetry posture (F-15)

- **User-decision audit write (resolve):** NONE (audit-primary; the
  `interpretation_events` row is the record).
- **IDOR 404 response:** operational telemetry (security signal).
- **Credential prefilter match on `llm_draft`:** operational telemetry
  (security signal indicating a potentially poisoned draft surfaced).

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
13. **Cross-session event_id IDOR (F-7):** Insert a PENDING event for
    session-B. Call POST /resolve as the owner of session-A, passing
    session-A's session_id path parameter and session-B's event_id.
    Assert the response is 404. This test verifies that
    `resolve_interpretation_event` filters on BOTH `id = :event_id AND
    session_id = :session_id`, not event_id alone.

### Step 1 — RED, Step 2 — GREEN, Step 3 — commit

`sessions(routes): add POST /interpretations/{event_id}/resolve and
GET /interpretations + cross-session IDOR test`.

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

### Opt-out idempotency contract (F-29)

`record_session_interpretation_opt_out` MUST be idempotent at the service
level. On a second call for the same session:
1. Detect the existing opted_out row (SELECT from `interpretation_events`
   WHERE `session_id = :sid AND choice = 'opted_out'` ordered by `created_at`
   LIMIT 1).
2. If found, return the existing record without inserting a new row.
   The sessions boolean remains `true` (it was already set). The first
   opt-out timestamp is authoritative.
3. If not found, proceed with the insert + boolean update as normal.

Add a test asserting `interpretation_events_table` row count is exactly 1
after two POST /opt_out calls for the same session.

### Opt-out audit tooling (F-22)

Add a §"Opt-out audit tooling" doc note below the Tests section:

> After opt-out, the LLM's auto-baked interpretations are stored in
> `composition_state.nodes` JSON with an explicit comment tag:
> `# AUTO-INTERPRETED, REVIEW SKIPPED PER USER OPT-OUT`
> (specified in the Task 8 skill nudge).
>
> API surface for session-end review (spec here, frontend in 18b-):
> `GET /api/sessions/{id}/interpretations/opt_out_summary` returns
> all `interpretation_events` rows with
> `interpretation_source='auto_interpreted_opt_out'` OR
> `interpretation_source='auto_interpreted_no_surfaces'`, ordered by
> `created_at`. The user can browse them retroactively to see what the
> LLM auto-baked during the opted-out portion of the session. This
> closes the "click opt-out once, dozens of auto-interpretations
> accumulate invisibly" audit gap. Frontend implementation is delegated
> to 18b-; route specification belongs in 18a- as it determines the
> backend contract.

### Telemetry posture (F-15)

Telemetry for this route:
- **User-decision audit write (opt-out):** NONE (audit-primary; the
  `interpretation_events` row is the record).
- **IDOR 404 response:** operational telemetry signal (security
  signal — indicates a cross-session access attempt; emit with
  `session_id` and requesting `user_id`).

### Tests

1. POST /opt_out flips the session column to true and writes an
   `interpretation_events` row with `choice='opted_out'` and
   `interpretation_source='auto_interpreted_opt_out'`. Verify NO
   `proposal_events` row was written (regression guard).
2. **Idempotency:** POST /opt_out twice for the same session — second
   call returns 200; `interpretation_events` row count for this session
   is exactly 1; the first opt-out timestamp is returned on both calls.
   Deterministic choice: keep FIRST opt-out timestamp.
3. After /opt_out, the compose-loop's system prompt MUST be told the
   user opted out (Task 5's compose-loop hook reads the flag at
   compose start; verify by a fixture-level integration test in Task 9).
4. **IDOR regression:** An authenticated user whose session is session-B
   cannot POST /opt_out on session-A's id. Returns 404, not 403.

### Step 1-3

`sessions(routes): add POST /interpretations/opt_out (routes to
interpretation_events, not proposal_events) + idempotency contract`.

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

### Skill-change governance (F-33)

Skill markdown changes constitute audit-relevant deploys. Every distinct
version of `pipeline_composer.md` hashed into `composer_skill_hash` on
interpretation events must have its content captured in
`skill_markdown_history` (see Task 2 — the table is upserted by the
compose-loop hook the first time a new hash is seen). A deploy that
rotates the skill markdown without restarting the service will produce
a hash mismatch between the in-memory LRU-cached text and the on-disk
file — detected by the Task 1.5 startup integrity check and Task 5b's
skill-text atomicity requirement. Coordinate skill-markdown changes with
service restarts so the hash stays consistent.

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
      `actor` matches. ASSERT that `composition_state_id`, `affected_node_id`,
      `tool_call_id`, `user_term`, `llm_draft`, `model_identifier`,
      `model_version`, `provider`, and `composer_skill_hash` are ALL NULL
      on the opted_out row (F-1 nullability regression guard). Also ASSERT
      that `proposal_events_table` contains NO row related to this opt-out
      (regression guard against prior proposal_events routing).
3. For each row, the test opens a read-only SQLAlchemy connection and
   issues `SELECT * FROM interpretation_events WHERE id = :id`. The
   test asserts EACH of the six required-by-spec fields is present and
   correctly populated. This is the explicit CLAUDE.md attributability
   compliance step.
4. **opted_out CHECK regression (F-1):** Attempt to INSERT an `opted_out`
   row with `user_term` non-NULL; assert `IntegrityError`
   (ck_interpretation_events_source_nullability fires). Attempt to INSERT
   a `user_approved` row with `user_term` NULL; assert `IntegrityError`.
5. **arguments_hash determinism (F-13):** Create two identical
   `interpretation_events` rows with the same field values but different
   `id` values (different timestamps to avoid PK collision), resolve both
   with identical inputs, assert `event1.arguments_hash == event2.arguments_hash`.
   This verifies rfc8785 canonicalization determinism across rows.

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

### Cross-DB hash equality check (`test_interpretation_runtime_handoff.py`)

This sub-test is the verification that the 18- overview's §"Hash-anchored
cross-DB linkage" Option A forward-references. It MUST be part of the
`test_interpretation_runtime_handoff.py` integration run (not a separate file)
so that it exercises the same real execution path used by the hand-off test.

Add the following assertion sequence immediately after the existing step 5:

6. **Session side read**: open a read-only connection to the session audit DB;
   `SELECT resolved_prompt_template_hash FROM interpretation_events WHERE id = :event_id`.
   Assert the value is non-NULL (the column was populated at resolve time).
   Assign to `session_hash`.

7. **Landscape side read**: open a read-only connection to the Landscape DB;
   `SELECT resolved_prompt_template_hash FROM calls WHERE call_id = :call_id`
   (where `call_id` is the LLM-transform call from step 4).
   Assert the value is non-NULL (the runtime plugin wrote the column at
   execution time).
   Assign to `landscape_hash`.

8. **Byte-equality assertion**: `assert session_hash == landscape_hash`. A
   failure here means the composition state was mutated between resolve time
   and execution time, or the runtime plugin received a different prompt string
   than the one recorded. The assertion message MUST include both hash values
   and the run_id so a failing CI output is immediately diagnosable.

9. **External recompute assertion**: read `composition_states.nodes` JSON for
   the affected node; extract the resolved `prompt_template` string. Compute
   `stable_hash(resolved_prompt_template_string)` using `contracts/hashing.py`
   (the same function the session service called at resolve time). Assert this
   recomputed hash equals both `session_hash` and `landscape_hash`.

   This recompute is the external audit-tooling check. It verifies that both
   DB columns are internally consistent AND that the composition state JSON
   contains the string that was actually hashed — i.e., the hash chain has no
   silent intermediate step. The production code path NEVER re-computes (it
   reads the stored hash from `composition_states.nodes`); only the test does
   this re-computation as an external verification, so divergence between the
   test's recompute and the stored values is a signal that the production
   write-path is hashing a different string than the one embedded in the
   composition state.

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
| Rate cap exceeded (per-term: 3) | System | ARG_ERROR: "Too many interpretation requests for term '…' (max 3)" | No pending row; compose loop writes a row with `interpretation_source='auto_interpreted_no_surfaces'` (see Task 5 F-6 writer) |
| Rate cap exceeded (per-session-day: 10) | System | ARG_ERROR: "Too many interpretation requests in this session today (max 10)" | No pending row; compose loop writes `auto_interpreted_no_surfaces` row |
| Credential-shaped content in `user_term` or `llm_draft` | User/LLM | 422: "That looks like a credential — please re-enter without secrets" | No row |
| `accepted_value` contains control characters | User | 422 with field-specific message | No row |
| `llm_draft` (accepted_as_drafted path) contains `{{`/`}}` | User/LLM | 422: "accepted_value must not contain template metacharacters" | No resolve committed; validated at tool boundary |
| Append-only trigger fires (attempt to modify resolved row) | System | 500 (SQLAlchemy IntegrityError) | DB row unchanged; request fails |
| Session ownership failure (IDOR attempt) | Attacker | 404 (not 403 — per IDOR contract) | No write; no audit row |
| Unresolved `{{interpretation:…}}` placeholder at runtime | System | Runtime executor raises RuntimeError; user sees error banner (see Task 5 §"Unresolved placeholder runtime detection") | No Landscape run row committed; pipeline execution blocked |
| Staged placeholder without prior `request_interpretation_review` call | System | Operational telemetry signal "interpretation placeholder unresolved at runtime" emitted (see F-21); runtime also raises | Telemetry signal recorded; no pipeline execution |
| `calls.resolved_prompt_template_hash` ≠ `interpretation_events.resolved_prompt_template_hash` for the same resolved template string | System (hash-chain integrity violation) | Surface as integrity-violation telemetry signal at audit-tooling read time; at the Landscape read layer (audit-tooling / `explain()` query) this is a Tier-1 anomaly — crash with a meaningful error per offensive-programming discipline ("hash chain broken: Landscape calls hash does not match session interpretation_events hash for run_id=…, event_id=…"). The mismatch means either the composition state was tampered with between resolution and execution, or the runtime plugin received a different prompt string than the one recorded at resolve time. | Landscape `calls` row was committed (execution completed); session `interpretation_events` row is unchanged; the audit chain is now inconsistent. No silent recovery. |

---

## Backend completion criteria

Phase 5b backend is complete when:

- [ ] **Task 0 (placeholder-convention gate)** passed; artifact at
      `evals/composer-rgr/phase5b-task0-placeholder-validation.json`
      with `gate_passed=true`. All subsequent tasks unblocked.
- [ ] **Task 0.5 (defensive size caps)** green, committed; `max_length`
      caps on `_InlineBlobModel.content`, `SendMessageRequest.content`,
      and ASGI body-size middleware in `app.py` rejecting >10 MB with 413.
- [ ] Task 1 (contract types) green, committed; `InterpretationChoice`
      has five values; `InterpretationSource` has three values;
      `model_identifier`, `model_version`, `provider`, `composer_skill_hash`
      are all `str | None` on `InterpretationEventRecord`;
      `resolved_prompt_template_hash: str | None` present on
      `InterpretationEventRecord`.
- [ ] **Task 1.5 (session DB hardening)** green, committed; WAL + busy_timeout
      + synchronous PRAGMA set in `engine.py`; startup assertion
      `journal_mode=='wal'`; `SESSION_SCHEMA_EPOCH` constant defined;
      `PRAGMA application_id` and `user_version` set and validated on open.
- [ ] Task 2 (schema) green, committed; the new table exists with all
      audit-provenance columns; the new provenance enum value is
      registered; the `interpretation_review_disabled` session column
      is registered; the append-only trigger is installed (IF NOT EXISTS,
      table-scoped listener); `chat_messages` immutability trigger
      installed; `skill_markdown_history` table added; `composition_state_id`
      index added; `hash_domain_version` column added;
      `runtime_model_identifier_at_resolve` and
      `runtime_model_version_at_resolve` nullable columns added;
      `resolved_prompt_template_hash` nullable `String(64)` column added to
      BOTH `interpretation_events_table` (session DB) AND `calls_table`
      (L1 Landscape, `core/landscape/schema.py`);
      `ix_calls_resolved_prompt_template_hash` index added to `calls_table`;
      trigger-existence test via production bootstrap path passes;
      Phase 9 migration notes block commented into `models.py`;
      Step 4 commit message mentions BOTH DB delete requirements.
      NOTE: `proposal_events.event_type` is NOT extended — opt-out rows
      go to `interpretation_events_table`.
- [ ] Task 3 (wire schemas) green, committed; content validators on
      `amended_value` pass (shared helper lives in `web/validation.py`);
      credential prefilter tests pass; JWT benign-period negative test
      passes; `_validate_accepted_value_content` applied to `llm_draft`
      at tool boundary; `resolved_prompt_template_hash` field present on
      `InterpretationEventResponse`.
- [ ] Task 4 (service methods) green, committed; all four methods
      exercised by direct-DB spot-checks; opt-out method writes to
      `interpretation_events_table` only; resolve WHERE clause includes
      `AND session_id AND choice='pending'`; `accepted_value` computed
      inside service from `llm_draft` (not from route); write-lock
      annotation on opt-out; trigger IntegrityError classifier noted;
      BEGIN IMMEDIATE noted; `resolve_interpretation_event` computes
      `resolved_prompt_template_hash` via `stable_hash()` and writes it
      atomically to both `interpretation_events_table` column and
      `composition_states.nodes` JSON sibling field in the same transaction.
- [ ] Task 5 (composer tool) green, committed; dispatch spike documented;
      `_SESSION_AWARE_TOOL_HANDLERS` registry introduced; async handler
      dispatches; rate-limit constants in settings; rate-limit window
      semantics pinned (UTC midnight); rate-limit tests pass; credential
      prefilter tests pass (including PII extensions and JWT benign-period
      negative test); redaction; proposal summary;
      `AUTO_INTERPRETED_NO_SURFACES` writer specified; unresolved-placeholder
      runtime detection sub-task specified; dual-registry invariant test
      passes; F-5a hash-atomicity from LRU cache specified; startup
      integrity check on skill text specified; skill_markdown_history
      upsert wired in compose-loop hook; opt-out audit-tooling note added.
- [ ] Task 6 (resolve route + list route) green, committed; IDOR
      regression tests pass (including cross-session event_id test);
      `accepted_value` not computed at route; runtime model snapshot
      fields populated; telemetry posture declarations present.
- [ ] Task 7 (opt_out route) green, committed; IDOR regression test
      passes; `proposal_events_table` regression guard passes;
      opt-out idempotency contract specified; session-end review route
      API surface specified.
- [ ] Task 8 (skill prompt) committed; empirically validated against
      the canonical hero prompt; `elspeth-web.service` restarted;
      skill-change governance note present.
- [ ] Task 9 (integration spot-check + runtime hand-off) green,
      committed; paths at `tests/integration/web/composer/`;
      `arguments_hash` determinism test passes; opted_out row NULLs
      asserted; new opted_out + non-NULL user_term → IntegrityError
      test passes; **cross-DB hash-equality test passes**: session
      `interpretation_events.resolved_prompt_template_hash` equals
      Landscape `calls.resolved_prompt_template_hash` byte-for-byte
      for the hero-prompt integration run (see Task 9
      §"Cross-DB hash equality check").
- [ ] **Cross-DB hash chain (`resolved_prompt_template_hash`)**: column
      present on BOTH `calls_table` (L1 Landscape, `core/landscape/schema.py`)
      AND `interpretation_events_table` (L3 session DB, `web/sessions/models.py`);
      populated identically by `resolve_interpretation_event` (session side)
      and the LLM-transform runtime plugin (Landscape side); Task 9
      hash-equality test green.
- [ ] Task 10 (audit-readiness `llm_interpretations` row activation)
      green, committed (unconditional — Phase 2C has shipped);
      audit-readiness panel warning for mismatched runtime model present.
- [ ] Task 11 (orphan PENDING-row cleanup spec) documented; `refreshPending`
      test passes.
- [ ] Task 12 (evals/composer-rgr regression suite) artifact exists;
      all four pass thresholds met.
- [ ] Failure-mode table reviewed; no uncovered failure path (including
      unresolved placeholder at runtime, F-17 detection path).
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
| 2026-05-18 | 9-reviewer panel | CHANGES_REQUESTED | F-1 through F-35 | Applied in this revision. F-1: NOT NULL × CHECK contradiction resolved — 9 columns flipped to nullable=True; opted_out CHECK rewritten as source-keyed (auto_interpreted_opt_out) rather than choice-keyed; user_approved row CHECK added; InterpretationEventRecord field types updated. F-2: prompt-injection bypass fixed — _validate_accepted_value_content shared helper added to web/validation.py; applied at tool boundary (_handle_request_interpretation_review against llm_draft) AND at service writer (resolve_interpretation_event) as defense-in-depth; accepted_as_drafted path now validated. F-3: Task 0.5 added (max_length caps + ASGI body cap). F-4: chat_messages immutability trigger added in Task 2. F-5: skill_markdown_history table and hash-atomicity spec added to Task 2 and Task 5. F-6: AUTO_INTERPRETED_NO_SURFACES writer specified in Task 5 (Option B). F-7: resolve WHERE clause explicitly includes AND session_id AND choice='pending'; IDOR test for cross-session event_id added to Task 6. F-8: trigger-existence test via production bootstrap path added to Task 2. F-9: Session DB PRAGMA discipline (WAL + busy_timeout + synchronous) added as Task 1.5. F-10: SESSION_SCHEMA_EPOCH / application_id / user_version added to Task 1.5. F-11: composition_state_id index added to Task 2. F-12: hash_domain_version column + INTERPRETATION_HASH_DOMAIN_V1 constant added to Task 2. F-13: arguments_hash determinism test added to Task 9. F-14: business-rule split (accepted_value computation) pushed to service; route now passes only choice + amended_value. F-15: telemetry posture differentiated at tool and route boundaries. F-16: Phase 9 migration notes block specified in Task 2. F-17: unresolved placeholder runtime detection sub-task added in Task 5. F-18: dual-registry dispatch invariant test added to Task 5. F-19: runtime model snapshot columns (runtime_model_identifier_at_resolve, runtime_model_version_at_resolve) added to Task 2 and Task 4. F-20: PII regex extension (email, phone, SSN-like) added to credential prefilter in Task 5. F-21: unresolved-placeholder runtime telemetry signal added as sub-task in Task 5. F-22: opt-out audit tooling note + session-end review route spec added to Task 7. F-23: IF NOT EXISTS + table-scoped DDL listener specified in Task 2. F-24: schema validator trigger registry extension noted in Task 2. F-25: BEGIN IMMEDIATE note added to Task 4. F-26: partial-index dialect symmetry note added to Task 2. F-27: write-lock annotation added to record_session_interpretation_opt_out. F-28: trigger IntegrityError classifier note added to Task 4. F-29: opt-out idempotency contract specified in Task 7. F-30: rate-limit window semantics (UTC midnight vs sliding 24h) pinned in Task 5. F-31: rate-limit constants moved to settings keys in Task 5. F-32: JWT benign-period negative test added to Task 5. F-33: skill-change governance note added to Task 8. F-34: _validate_accepted_value_content placed in web/validation.py (existing module), not a new _validation_helpers.py. F-35: opted-out two-source-of-truth note added to Task 2. |
| 2026-05-18 | Cross-file F-1 closure | APPLIED | (no finding ID) | Cross-file coordination gap closed: added `resolved_prompt_template_hash` column to both the session `interpretation_events_table` (L3 session DB) and the runtime Landscape `calls_table` (L1 Landscape DB) per the Option A commitment in sibling doc 18-. Updated `InterpretationEventRecord` (Task 1) and `InterpretationEventResponse` (Task 3) to include the field. Updated `resolve_interpretation_event` (Task 4) to compute the hash via `stable_hash()` from `contracts/hashing.py` (patch → hash → write both sinks atomically). Added Task 9 cross-DB hash-equality verification test. Added failure-mode row for hash-chain mismatch. Added completion-criteria bullet for the column pair. Updated Tech Stack "No Landscape changes" bullet and file-structure prose to reflect that Phase 5b now touches one L1 column. Added `core/landscape/schema.py MODIFY` entry to the file-structure inventory. Updated Migration runner ownership section to require deletion of both session DB and Landscape DB. Closes the dangling forward-reference from 18- §"Hash-anchored cross-DB linkage". |
| 2026-05-18 | Independent contradiction review | APPLIED | (no finding ID) | `auto_interpreted_no_surfaces` nullability contract: F-6 writer spec (Task 5) and line-409 prose aligned with `ck_interpretation_events_no_surfaces_shape` CHECK — LLM provenance (`model_identifier`, `model_version`, `provider`, `composer_skill_hash`) MUST be populated for `auto_interpreted_no_surfaces` rows; only the five interpretation surface fields (`composition_state_id`, `affected_node_id`, `tool_call_id`, `user_term`, `llm_draft`) are NULL. Two outlier sites corrected: (1) F-6 writer bullet list rewrote "all NULL" to "surface fields NULL, provenance NOT NULL" with explicit rationale paragraph citing the CHECK and schema comment; (2) line-409 prose changed "MAY be populated" to "MUST be populated" with asymmetry note. Three concordant sources (CHECK at 853-859, schema comment at 826-829, test spec at 1281-1284) were authoritative and left unchanged. |
| 2026-05-18 | Plan amendment | APPLIED | (no finding ID) | Added shared-worktree section: `feat/composer-phase-5-chat-data-entry` at `.worktrees/composer-phase-5-chat-data-entry`; Phase 5a + Phase 5b ship as coordinated PR on one branch. Added worktree-root prefix note to File structure section (with `pipeline_composer.md` carve-out). |

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

---

## Addendum (post-merge, Phase 4 hello-world tutorial residual): closed-enum extension to `tutorial_normalization`

This addendum records the eighth value added to the `composition_states.provenance` closed enum, following the same governance posture used to add `interpretation_resolve` above (CHECK + Literal paired extension + Filigree ticket + spec amendment + integration test). It is filed here because this document is the established precedent for documenting closed-enum extensions to that constraint.

### Writer path

`elspeth.web.composer.tutorial_service._normalise_current_tutorial_state_for_execution` is the sole writer. The function runs at the entry point of every live (non-cache-replay) Phase 4 hello-world tutorial run, immediately before `_run_live_tutorial`. It inspects the user's current composition state for `llm`-plugin transform nodes whose `prompt_template` references `{{ field }}` bare (without the `row.` namespace prefix) and rewrites the placeholders to `{{ row.field }}`. The rewrite is non-load-bearing in the runtime sense — the LLM transform plugin would still accept either form — but it produces canonical prompt text that round-trips identically through `stable_hash`, which the audit-row `resolved_prompt_template_hash` relies on for state-version diffing.

When `_normalise_bare_required_field_templates` reports `changed=True`, the function calls `save_composition_state(... provenance="tutorial_normalization")` with the rewritten nodes, the original source/edges/outputs/metadata, and a composer-meta header recording `tutorial_runtime_normalized=True`, `tutorial_normalization="bare_required_field_templates"`, and the originating `state_id`. Telemetry counter `composer.tutorial.runtime_normalization_total{kind="bare_required_field_templates"}` is incremented in the same code path.

### Audit semantics

`tutorial_normalization` is a SYSTEM-INITIATED writer, sibling to `convergence_persist` / `plugin_crash_persist` / `preflight_persist` / `session_seed`. The distinguishing semantics versus those neighbouring values:

- **vs. `convergence_persist`** — `convergence_persist` is reserved for the routes.py `_handle_convergence_error` writer, which persists state after a validator-failure recovery. `tutorial_normalization` runs at the START of a tutorial execution, not after a failure; it is a deliberate canonicalisation, not a recovery. Conflating the two would obscure the validator failure rate in the audit history.

- **vs. `tool_call`** — `tool_call` is user-initiated (via the LLM tool call surface). `tutorial_normalization` is system-initiated and never user-visible (the user did not request the rewrite).

- **vs. `session_seed`** — `session_seed` writes the initial empty/fresh state at session creation or fork. `tutorial_normalization` writes a non-empty rewrite of an existing state.

- **vs. `interpretation_resolve`** — Both rewrite an existing composition state mid-flow, but `interpretation_resolve` is user-initiated (the user resolves a surfaced interpretation event) and `tutorial_normalization` is system-initiated (no user gesture triggers it).

### Deploy constraint

Same as the precedent above. This enum extension requires the staging session DB delete (per `project_db_migration_policy`). `SESSION_SCHEMA_EPOCH` bumps `7 → 8`; the staging operator runs the recreation runbook (`docs/runbooks/staging-session-db-recreation.md`) after deploy. Production sequencing remains AFTER Phase 9's migration runner.

### Governance bundle

- **CHECK + Literal paired extension** — `web/sessions/models.py::ck_composition_states_provenance` and `web/sessions/protocol.py::CompositionStateProvenance`. The `test_composition_state_provenance_python_and_sql_enums_agree` test in `tests/unit/web/sessions/test_routes.py` pins them equal.
- **Integration / unit test** — `tests/unit/web/composer/test_tutorial_service.py::test_normalise_current_tutorial_state_persists_tutorial_normalization_provenance` drives the writer via a stub session-service and asserts the captured `provenance` kwarg. The DB-level CHECK accept path is covered by `tests/unit/web/sessions/test_composition_states.py::test_provenance_check_accepts_known_values`.
- **Filigree ticket** — filed at PR-open time; cited in the commit body (this is the residual close-out of the post-`ca9bc05bd` PR review's I7 finding).

## Addendum (post-compose route attribution): closed-enum extension to `post_compose`

This addendum records the ninth value added to the `composition_states.provenance` closed enum, following the same governance posture used for `interpretation_resolve` and `tutorial_normalization` (CHECK + Literal paired extension + Filigree ticket + spec amendment + integration test).

### Writer path

`post_compose` is written by the successful send-message and recompose route paths after the composer returns a newer composition state. The same value is used by the paired metadata-only persistence path that flips `guided_session.transition_consumed` after a transition prompt was consumed even when the graph version did not change.

### Audit semantics

`post_compose` is a route-level, LLM-driven state advance inside an existing session. It is distinct from:

- **`session_seed`** — initial session creation or explicit active-state reselection.
- **`tool_call`** — the atomic compose-loop tool-call writer that carries the backward-direction INV-AUDIT-AHEAD invariant.
- **`convergence_persist` / `plugin_crash_persist` / `preflight_persist`** — failure-path partial-state captures.

Fork-time blob-reference rewriting remains `session_fork` because it is part of the fork operation: it rewrites the copied state so the child session is self-contained after blob copy.

### Deploy constraint

Same as the precedent above. This enum extension requires a fresh session DB because SQLite cannot ALTER a CHECK constraint in place. `SESSION_SCHEMA_EPOCH` bumps `21 -> 22` and stale pre-release session DBs must be deleted/recreated at startup.

### Governance bundle

- **CHECK + Literal paired extension** — `web/sessions/models.py::ck_composition_states_provenance` and `web/sessions/protocol.py::CompositionStateProvenance`. The `test_composition_state_provenance_python_and_sql_enums_agree` test pins them equal.
- **Integration / unit tests** — `tests/unit/web/sessions/test_routes.py::test_send_message_post_compose_state_advance_persists_post_compose_provenance`, `tests/unit/web/sessions/test_routes.py::test_recompose_post_compose_state_advance_persists_post_compose_provenance`, and `tests/unit/web/sessions/test_fork.py::TestForkEndpoint::test_fork_blob_rewrite_persists_session_fork_provenance` drive the affected route writers and assert the persisted DB values.
- **Filigree ticket** — `elspeth-24a7fb8e54`.
