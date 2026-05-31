# Composer UX Tool Proposal Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the first executable slice of `docs/design/composer-ux-spec.md`: real Pending -> Committed -> Audited tool-call proposals, visible conversation cards, and pending overlays in Graph/Spec/YAML views.

**Architecture:** The backend becomes the source of truth for proposals: mutating composer tool calls create durable `composition_proposals` rows and append immutable `proposal_events` rows before any visible pending UI appears. Accepted proposals execute through the same `execute_tool` and composer audit helpers that the current compose loop uses, then persist the resulting composition state and mark the proposal committed in one locked session transaction. The frontend consumes proposal records from the session API and renders cards and overlays; it never fabricates pending state from chat text.

**Tech Stack:** FastAPI, SQLAlchemy Core session DB, Pydantic v2, existing composer `execute_tool` registry, React 18, Zustand, Vitest, Playwright, pytest.

---

## Scope Check

The full UX spec covers multiple independent subsystems: proposal lifecycle, ledger/status strip, density mode, audit/review tools, discoverability, data-flow view, accessibility polish, and operational telemetry. This plan deliberately implements the high-leverage Phase 0 + Phase 2 + Phase 3 slice called out in spec sections 6.2, 9.1, 16, and 20.

Planner decisions for this slice:

- Collaboration model: v1 uses the existing per-session compose lock and stale-state rejection, not CRDT or multi-cursor editing.
- Mobile depth: desktop-first with responsive survival; touch-target polish belongs in the Phase 9 plan.
- Auditor authentication: no new auth role in this slice; audit-grade data remains gated by existing `include_tool_rows`/`audit_access_log` behavior.
- Internationalisation: out of v1 for this slice; copy is centralized in components so a later i18n plan can extract it.
- Embedded vs standalone: standalone composer only.

Separate follow-up plans should cover Phase 1 (Ledger/status strip), Phase 4 (density/per-turn scaffolding), Phase 6 (audit review surface), Phase 8 (data-flow view), Phase 9 (accessibility polish), and Phase 10 (UX telemetry).

In-slice cuts from the spec, deliberately deferred from this v0 lifecycle slice:

- Graph pending overlay depth: this plan ships a visible pending count and canvas awareness, but defers per-node dashed outlines, per-edge styling, and click-to-jump from the graph pill to a Phase 2 overlay follow-up.
- Spec-view pending depth: this plan ships pending proposal rows, but defers per-affected-node row placement and struck-through old values to the same overlay follow-up.
- YAML diff depth: this plan ships a pending summary above read-only YAML, but defers line-by-line green/amber/red diff highlighting.
- Dynamic rationale: Phase 0 ships the `rationale` field and a uniform placeholder. A follow-up must populate it from LLM-emitted context or recipe metadata so the field becomes semantically useful, not only structurally present.
- Edit and Discuss actions: this plan ships Accept and Reject. Edit and Discuss require chat-input/per-turn scaffolding and belong with the Phase 4 plan.
- Copyable audit IDs: this plan displays audit IDs. Click-to-copy belongs in a polish follow-up unless it is cheap to include while implementing `ToolCallCard`.
- Atomic transaction boundary across state-save and proposal-commit: Task 6 accepts a proposal by saving the new composition state, then committing the proposal/event. A follow-up should either collapse state-save + accepted-event + status flip into a single `commit_proposal_with_state(...)` transaction, or add explicit orphan-state recovery that emits an auditable recovery event.
- Accept response symmetry: this plan keeps accept returning `CompositionProposalResponse` plus a frontend state refetch. A polish follow-up can mirror the compose response with `{ proposal, state }` to save one round trip.

## File Structure

Backend session/audit foundation:

- Modify `src/elspeth/web/sessions/models.py`: add session preference columns, `composition_proposals_table`, and `proposal_events_table`.
- Modify `src/elspeth/web/sessions/protocol.py`: add closed Literal domains and frozen record dataclasses for preferences, proposals, and proposal events.
- Modify `src/elspeth/web/sessions/schemas.py`: add strict response/request models for preferences and proposals.
- Modify `src/elspeth/web/sessions/service.py`: add service methods for preference reads/writes, proposal creation/listing, accept, and reject.
- Modify `src/elspeth/web/sessions/routes.py`: add session-scoped proposal and preference routes.
- Modify `src/elspeth/web/composer/tools.py`: expose `is_mutation_tool()` from the existing closed mutation registries.
- Create `src/elspeth/web/composer/proposals.py`: build plain-language proposal summaries, affected-component hints, and redacted display args.
- Modify `src/elspeth/web/composer/service.py`: in explicit-approve mode, create proposals for mutating tool calls instead of immediately executing them; keep auto-commit using the current execution path.

Frontend:

- Modify `src/elspeth/web/frontend/src/types/index.ts`: add proposal and preference types.
- Modify `src/elspeth/web/frontend/src/api/client.ts`: add proposal/preference API calls.
- Modify `src/elspeth/web/frontend/src/stores/sessionStore.ts`: load proposals/preferences per session and expose accept/reject actions.
- Create `src/elspeth/web/frontend/src/components/chat/ToolCallCard.tsx`: canonical read ribbon and write proposal card.
- Modify `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`: replace raw tool-call disclosure with `ToolCallCard`.
- Modify `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`: load proposal records and pass them to messages.
- Modify `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`: render pending overlay nodes/edges from proposal effects.
- Modify `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`: render pending proposal rows.
- Modify `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx`: show pending diff summary above read-only YAML.
- Modify `src/elspeth/web/frontend/src/App.css`: add proposal card and overlay styles using semantic tokens.

Tests:

- Create `tests/unit/web/sessions/test_composer_proposals.py`.
- Add route tests to `tests/unit/web/sessions/test_routes.py`.
- Add compose-loop behavior tests to `tests/unit/web/composer/test_service.py`.
- Create `src/elspeth/web/frontend/src/components/chat/ToolCallCard.test.tsx`.
- Extend `src/elspeth/web/frontend/src/components/chat/MessageBubble.test.tsx`.
- Extend `src/elspeth/web/frontend/src/stores/sessionStore.test.ts`.
- Extend `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx`.
- Extend `src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx`.
- Extend `src/elspeth/web/frontend/src/components/inspector/YamlView.test.tsx`.
- Add `src/elspeth/web/frontend/tests/e2e/composer-proposals.spec.ts`.

## Task 0: Preflight and Deployment Reset Planning

**Files:**
- Read-only verification across current source tree
- No code changes

- [x] **Step 1: Confirm the execution checkout and branch**

Preflight result: implementation moved to isolated worktree `/home/john/elspeth/.worktrees/composer-proposal-lifecycle` on branch `composer-proposal-lifecycle` at handoff commit `69a6d55d1`; `git status --short` was clean.

Run:

```bash
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git worktree list
git status --short
```

Expected: implementation happens in a dedicated checkout such as `/home/john/elspeth/.worktrees/composer-proposal-lifecycle/`, or the operator explicitly confirms that the current branch/worktree is dedicated to this work. Do not start implementation in the dirty main checkout by accident.

- [x] **Step 2: Confirm the worktree-local Python environment**

Preflight result: created and targeted the worktree-local venv explicitly after detecting the shell had `VIRTUAL_ENV=/home/john/elspeth/.venv`; confirmed `.venv/bin/python` resolves to `/home/john/elspeth/.worktrees/composer-proposal-lifecycle/.venv/bin/python`, Python 3.13.1, with `pytest-asyncio` present in `pyproject.toml`.

Run:

```bash
.venv/bin/python --version
rg -n "pytest-asyncio|asyncio_mode" pyproject.toml
```

Expected: Python 3.13.x in the worktree venv, and `pytest-asyncio` present in `pyproject.toml`. If the venv resolves outside the worktree or uses an older Python, recreate the venv before running tier-model checks so spurious policy violations do not swamp the real diff.

- [x] **Step 3: Capture policy-gate baseline before edits**

Preflight result: the literal plan command is stale because `enforce_tier_model.py check` now requires `--root`; the current repo-standard command `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` passed with `No bug-hiding patterns detected. Check passed.`

Run on the unchanged branch:

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check
```

Expected: either PASS, or a recorded baseline of pre-existing violations. Any new violation after this plan lands must be fixed or explicitly narrowed in `config/cicd/enforce_tier_model/`.

- [x] **Step 4: Verify fragile codebase assumptions before editing**

Preflight result: `compose(...)` still uses the construction-wired session service; `_require_sessions_service()`, `turn_has_mutation`, `MANIFEST`, `redact_tool_call_arguments`, `_state_from_record`, `_state_data_from_composer_state`, `run_sync_in_worker`, `_FakeComposeLLM`, `_fake_llm_response`, and LiteLLM-shaped frontend `ToolCall` are present. `fake_llm_one_set_pipeline_tool_call` is still absent and belongs to Task 5. `catalog_service`, `session_service`, and `phase3_engine` are app-state attributes in live/test code; `data_dir` and `secrets_service` are not app-state attributes in the grep result, so the accept route must reuse the existing route-local settings/secrets source. No shared frontend `isHttpConflict` helper exists; Task 7 must add or centralize one at the API/store seam.

Run:

```bash
rg -n "async def compose\(|def compose\(" src/elspeth/web/composer/service.py src/elspeth/web/composer/protocol.py
rg -n "_sessions_service|_require_sessions_service|turn_has_mutation" src/elspeth/web/composer/service.py
rg -n "MANIFEST|redact_tool_call_arguments" src/elspeth/web/composer/redaction.py
rg -n "_state_from_record|_state_data_from_composer_state|run_sync_in_worker" src/elspeth/web/sessions/routes.py
rg -n "app\.state\.(phase3_engine|catalog_service|data_dir|secrets_service|session_service)|state\.(phase3_engine|catalog_service|data_dir|secrets_service|session_service)" src/elspeth/web/app.py src/elspeth/web/sessions/routes.py tests/unit/web -g "*.py"
rg -n "class _FakeComposeLLM|def _fake_llm_response|fake_llm_one_set_pipeline_tool_call" tests/unit/web/composer/conftest.py
rg -n "export interface ToolCall|tool_calls" src/elspeth/web/frontend/src/types/index.ts
rg -n "isHttpConflict|parseResponse|status === 409|status: 409" src/elspeth/web/frontend/src
```

Expected current shape as of this plan repair:

- `ComposerServiceImpl.compose(...)` and `ComposerServiceProtocol.compose(...)` do **not** take a per-call `sessions_service` parameter; use the service wired at construction via `_sessions_service` and `_require_sessions_service()`.
- `turn_has_mutation` already exists and is consumed in the compose loop.
- `MANIFEST` and `redact_tool_call_arguments` are exported from `elspeth.web.composer.redaction`; import them at module scope unless a real circular import is proven.
- `_state_from_record`, `_state_data_from_composer_state`, and `run_sync_in_worker` are already available in `src/elspeth/web/sessions/routes.py`; route work should reuse those local helpers.
- `catalog_service`, `session_service`, and `phase3_engine` are present on `app.state` in the live app/test harness. Verify the exact state names for `data_dir` and `secrets_service` before writing the accept route; if they are not app-state attributes, use the existing route-local source for settings/secrets instead of inventing new attributes.
- `_FakeComposeLLM` and `_fake_llm_response` exist; `fake_llm_one_set_pipeline_tool_call` is introduced by Task 5.
- Frontend `ToolCall` uses LiteLLM/OpenAI shape with `id` and `function.name`/`function.arguments`.
- If no frontend HTTP-conflict helper exists, Task 7 must add one at the API/store seam and test it with the stale-proposal regression.

- [x] **Step 5: Resolve the active session DB and plan reset**

Preflight result: `WebSettings.get_session_db_url()` still resolves `session_db_url`, then `data_dir / "sessions.db"`. The isolated worktree does not carry the ignored staging env file. The live staging checkout has `/home/john/elspeth/deploy/elspeth-web.env`, but redacted inspection found neither `ELSPETH_WEB__SESSION_DB_URL` nor `ELSPETH_WEB__DATA_DIR`, so staging resolves to `/home/john/elspeth/data/sessions.db`. SQLite reset must archive/delete `sessions.db`, `sessions.db-wal`, `sessions.db-shm`, and `sessions.db-journal` together using `docs/runbooks/staging-session-db-recreation.md`; no live reset/restart has been performed.

This schema slice adds columns/tables and this project does not use Alembic migrations for the web session DB. Fresh test DBs work because `initialize_session_schema()` creates current metadata; existing dev/staging DBs must be reset before running the changed server.

Local reset:

```bash
rg -n "session_db_url|get_session_db_url|ELSPETH_WEB__SESSION_DB_URL|sessions.db" src/elspeth/web/config.py deploy docs/runbooks/staging-session-db-recreation.md
```

Resolve `WebSettings.get_session_db_url()`:

- `ELSPETH_WEB__SESSION_DB_URL`, if set.
- Else `${ELSPETH_WEB__DATA_DIR}/sessions.db`, if `ELSPETH_WEB__DATA_DIR` is set.
- Else `data/sessions.db` relative to the process working directory.

For SQLite, archive/delete the whole artifact set together: `sessions.db`, `sessions.db-wal`, `sessions.db-shm`, and `sessions.db-journal`.

Staging reset:

- Treat `https://elspeth.foundryside.dev` as requiring explicit reset orchestration too, not as local-only.
- Inspect `deploy/elspeth-web.env` directly before deployment without printing secret values. Confirm the live `ELSPETH_WEB__SESSION_DB_URL` or `ELSPETH_WEB__DATA_DIR`.
- Stop `elspeth-web.service`, archive/delete the resolved SQLite artifact set, then restart after the schema lands. Use `docs/runbooks/staging-session-db-recreation.md` as the operator runbook.
- If Codex cannot access systemd/sudo because of sandbox `no_new_privileges` or bus restrictions, report that blocker and do not claim live staging was reset or restarted.

- [x] **Step 6: Verify frontend dist policy**

Preflight result: `src/elspeth/web/frontend/dist/index.html` is ignored by `.gitignore:51` and not tracked by git, so build artifacts must not be committed.

Run:

```bash
git check-ignore -v src/elspeth/web/frontend/dist/index.html || true
git ls-files src/elspeth/web/frontend/dist/index.html
```

Expected: if `dist/` is ignored and untracked, do not commit it. Still run `npm run build` and live-check/locally inspect the generated asset names during deployment verification.

- [x] **Step 7: Use the required commit trailer**

Preflight result: all implementation commits in this worktree must include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

Every commit step in this plan must use this message shape, changing only the subject:

```bash
git commit -m "$(cat <<'EOF'
feat: short summary

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 1: Backend Proposal Schema Contracts

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Test: `tests/unit/web/sessions/test_composer_proposals.py`

- [x] **Step 1: Write the failing schema tests**

Create `tests/unit/web/sessions/test_composer_proposals.py` with:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import insert, inspect, select
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    composition_proposals_table,
    proposal_events_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


def _insert_session(conn, session_id: str) -> None:
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id="alice",
            auth_provider_type="local",
            title="Composer UX",
            trust_mode="explicit_approve",
            density_default="high",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )


def test_session_preferences_columns_exist(engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("sessions")}
    assert {"trust_mode", "density_default"} <= columns


def test_proposal_tables_exist(engine) -> None:
    table_names = set(inspect(engine).get_table_names())
    assert "composition_proposals" in table_names
    assert "proposal_events" in table_names


def test_composition_proposal_status_is_closed(engine) -> None:
    session_id = str(uuid4())
    with engine.begin() as conn:
        _insert_session(conn, session_id)
        with pytest.raises(IntegrityError, match="ck_composition_proposals_status"):
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=session_id,
                    tool_call_id="call_1",
                    tool_name="set_pipeline",
                    status="half_done",
                    summary="Add a pipeline",
                    rationale="Requested by the user",
                    affects=["graph", "validation"],
                    arguments_json={"source": {"plugin": "csv", "options": {}}},
                    arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )


def test_proposal_event_type_is_closed(engine) -> None:
    session_id = str(uuid4())
    with engine.begin() as conn:
        _insert_session(conn, session_id)
        with pytest.raises(IntegrityError, match="ck_proposal_events_type"):
            conn.execute(
                insert(proposal_events_table).values(
                    id=str(uuid4()),
                    session_id=session_id,
                    proposal_id=None,
                    event_type="proposal.maybe",
                    actor="user:alice",
                    payload={"status": "unknown"},
                    created_at=datetime.now(UTC),
                )
            )


def test_default_session_preferences_are_inserted_by_database(engine) -> None:
    session_id = str(uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                auth_provider_type="local",
                title="Defaults",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        row = conn.execute(select(sessions_table).where(sessions_table.c.id == session_id)).one()
    assert row.trust_mode == "explicit_approve"
    assert row.density_default == "high"
```

This default test intentionally uses raw `insert(sessions_table)` while `_insert_session()` passes explicit values. After implementing the columns, also inspect `SessionServiceImpl.create_session()` and add/update a focused assertion if the production session-creation path explicitly sets preferences instead of relying on server defaults. Both paths must remain valid.

- [x] **Step 2: Run the schema tests to verify they fail**

Task result: `.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py -q` failed during collection with `ImportError: cannot import name 'composition_proposals_table'`, the expected missing-schema failure.

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py -q
```

Expected: FAIL with missing `composition_proposals_table`, missing `proposal_events_table`, or missing session preference columns.

- [x] **Step 3: Add closed protocol types and records**

In `src/elspeth/web/sessions/protocol.py`, add these imports and definitions near the existing chat/session closed domains:

```python
ComposerTrustMode = Literal["explicit_approve", "auto_commit"]
ComposerDensityDefault = Literal["high", "medium", "low"]
ProposalLifecycleStatus = Literal["pending", "committed", "rejected"]
ProposalEventType = Literal[
    "proposal.created",
    "proposal.accepted",
    "proposal.rejected",
    "trust_mode.changed",
]

COMPOSER_TRUST_MODE_VALUES: frozenset[str] = frozenset(get_args(ComposerTrustMode))
COMPOSER_DENSITY_DEFAULT_VALUES: frozenset[str] = frozenset(get_args(ComposerDensityDefault))
PROPOSAL_LIFECYCLE_STATUS_VALUES: frozenset[str] = frozenset(get_args(ProposalLifecycleStatus))
PROPOSAL_EVENT_TYPE_VALUES: frozenset[str] = frozenset(get_args(ProposalEventType))
```

Add these record dataclasses after `SessionRecord`:

```python
@dataclass(frozen=True, slots=True)
class ComposerSessionPreferencesRecord:
    session_id: UUID
    trust_mode: ComposerTrustMode
    density_default: ComposerDensityDefault
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CompositionProposalRecord:
    id: UUID
    session_id: UUID
    tool_call_id: str
    tool_name: str
    status: ProposalLifecycleStatus
    summary: str
    rationale: str
    affects: Sequence[str]
    arguments_json: Mapping[str, Any]
    arguments_redacted_json: Mapping[str, Any]
    base_state_id: UUID | None
    committed_state_id: UUID | None
    audit_event_id: UUID | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        freeze_fields(self, "affects", "arguments_json", "arguments_redacted_json")


@dataclass(frozen=True, slots=True)
class ProposalEventRecord:
    id: UUID
    session_id: UUID
    proposal_id: UUID | None
    event_type: ProposalEventType
    # Actor format is canonicalized by class: composer-web:user:{user_id},
    # user:{user_id}, or system:{component}.
    actor: str
    payload: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        freeze_fields(self, "payload")
```

- [x] **Step 4: Add the SQLAlchemy tables and constraints**

In `src/elspeth/web/sessions/models.py`, no new import is required; `server_default` is a keyword argument on `Column`.

Add these columns to `sessions_table` after `title`:

```python
    Column("trust_mode", String, nullable=False, server_default="explicit_approve"),
    Column("density_default", String, nullable=False, server_default="high"),
```

Add these `sessions_table` constraints before the closing parenthesis:

```python
    CheckConstraint(
        "trust_mode IN ('explicit_approve', 'auto_commit')",
        name="ck_sessions_trust_mode",
    ),
    CheckConstraint(
        "density_default IN ('high', 'medium', 'low')",
        name="ck_sessions_density_default",
    ),
```

Add the proposal tables after `composition_states_table`:

```python
composition_proposals_table = Table(
    "composition_proposals",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("tool_call_id", String, nullable=False),
    Column("tool_name", String, nullable=False),
    Column("status", String, nullable=False),
    Column("summary", Text, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("affects", JSON, nullable=False),
    # Raw arguments are kept for replay/execution; only arguments_redacted_json
    # is safe for normal API/UI exposure.
    Column("arguments_json", JSON, nullable=False),
    Column("arguments_redacted_json", JSON, nullable=False),
    Column("base_state_id", String, nullable=True),
    Column("committed_state_id", String, nullable=True),
    Column("audit_event_id", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["base_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_composition_proposals_base_state_session",
    ),
    ForeignKeyConstraint(
        ["committed_state_id", "session_id"],
        ["composition_states.id", "composition_states.session_id"],
        name="fk_composition_proposals_committed_state_session",
    ),
    UniqueConstraint(
        "session_id",
        "tool_call_id",
        name="uq_composition_proposals_session_tool_call",
    ),
    CheckConstraint(
        "status IN ('pending', 'committed', 'rejected')",
        name="ck_composition_proposals_status",
    ),
    CheckConstraint(
        "(status = 'committed') = (committed_state_id IS NOT NULL)",
        name="ck_composition_proposals_committed_state",
    ),
)

proposal_events_table = Table(
    "proposal_events",
    metadata,
    Column("id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column(
        "proposal_id",
        String,
        ForeignKey("composition_proposals.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("event_type", String, nullable=False),
    Column("actor", String, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    CheckConstraint(
        "event_type IN ('proposal.created', 'proposal.accepted', 'proposal.rejected', 'trust_mode.changed')",
        name="ck_proposal_events_type",
    ),
)
Index(
    "ix_proposal_events_session_created",
    proposal_events_table.c.session_id,
    proposal_events_table.c.created_at,
)
```

- [x] **Step 5: Run the schema tests to verify they pass**

Task result: `.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py tests/unit/web/sessions/test_schema.py -q` passed with `14 passed`.

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py tests/unit/web/sessions/test_schema.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/models.py src/elspeth/web/sessions/protocol.py tests/unit/web/sessions/test_composer_proposals.py
git commit -m "$(cat <<'EOF'
feat: add composer proposal session schema

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 2: Session Service Proposal and Preference Operations

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Test: `tests/unit/web/sessions/test_composer_proposals.py`

- [x] **Step 1: Add failing service tests**

Append to `tests/unit/web/sessions/test_composer_proposals.py`:

```python
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def service(engine):
    return SessionServiceImpl(engine)


async def test_get_composer_preferences_returns_defaults(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    prefs = await service.get_composer_preferences(session_id)

    assert str(prefs.session_id) == str(session_id)
    assert prefs.trust_mode == "explicit_approve"
    assert prefs.density_default == "high"


async def test_update_trust_mode_writes_audit_event_before_return(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    prefs = await service.update_composer_preferences(
        session_id,
        trust_mode="auto_commit",
        density_default="medium",
        actor="user:alice",
    )

    assert prefs.trust_mode == "auto_commit"
    assert prefs.density_default == "medium"
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == ["trust_mode.changed"]
    assert events[0].payload == {
        "trust_mode": "auto_commit",
        "density_default": "medium",
    }


async def test_create_composition_proposal_writes_created_event(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))

    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline with one source and one sink.",
        rationale="Requested by the user.",
        affects=("graph", "validation"),
        arguments_json={"source": {"plugin": "csv", "options": {}}},
        arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )

    assert proposal.status == "pending"
    assert proposal.affects == ("graph", "validation")
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == ["proposal.created"]
    assert str(events[0].proposal_id) == str(proposal.id)


async def test_reject_composition_proposal_is_forward_only(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline.",
        rationale="Requested by the user.",
        affects=("graph",),
        arguments_json={"source": {"plugin": "csv", "options": {}}},
        arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )

    rejected = await service.reject_composition_proposal(
        session_id=session_id,
        proposal_id=proposal.id,
        actor="user:alice",
    )

    assert rejected.status == "rejected"
    events = await service.list_proposal_events(session_id)
    assert [event.event_type for event in events] == [
        "proposal.created",
        "proposal.rejected",
    ]
```

- [x] **Step 2: Run service tests to verify they fail**

Task result: after correcting the live `SessionServiceImpl` fixture shape, `.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py -q` failed with missing `get_composer_preferences`, `update_composer_preferences`, and `create_composition_proposal` methods.

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py -q
```

Expected: FAIL with missing service methods.

- [x] **Step 3: Extend `SessionServiceProtocol`**

In `src/elspeth/web/sessions/protocol.py`, add these methods to `SessionServiceProtocol`:

```python
    async def get_composer_preferences(
        self,
        session_id: UUID,
    ) -> ComposerSessionPreferencesRecord: ...

    async def update_composer_preferences(
        self,
        session_id: UUID,
        *,
        trust_mode: ComposerTrustMode,
        density_default: ComposerDensityDefault,
        actor: str,
    ) -> ComposerSessionPreferencesRecord: ...

    async def create_composition_proposal(
        self,
        *,
        session_id: UUID,
        tool_call_id: str,
        tool_name: str,
        summary: str,
        rationale: str,
        affects: Sequence[str],
        arguments_json: Mapping[str, Any],
        arguments_redacted_json: Mapping[str, Any],
        base_state_id: UUID | None,
        actor: str,
    ) -> CompositionProposalRecord: ...

    async def list_composition_proposals(
        self,
        session_id: UUID,
        *,
        status: ProposalLifecycleStatus | None = None,
    ) -> list[CompositionProposalRecord]: ...

    async def reject_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord: ...

    async def list_proposal_events(
        self,
        session_id: UUID,
    ) -> list[ProposalEventRecord]: ...
```

- [x] **Step 4: Implement service helpers**

In `src/elspeth/web/sessions/service.py`, import the new tables and protocol records, then add helper row converters:

```python
def _proposal_record_from_row(row: Any) -> CompositionProposalRecord:
    return CompositionProposalRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        status=row.status,
        summary=row.summary,
        rationale=row.rationale,
        affects=tuple(row.affects),
        arguments_json=row.arguments_json,
        arguments_redacted_json=row.arguments_redacted_json,
        base_state_id=UUID(row.base_state_id) if row.base_state_id else None,
        committed_state_id=UUID(row.committed_state_id) if row.committed_state_id else None,
        audit_event_id=UUID(row.audit_event_id) if row.audit_event_id else None,
        created_at=SessionServiceImpl._ensure_utc(row.created_at),
        updated_at=SessionServiceImpl._ensure_utc(row.updated_at),
    )


def _proposal_event_record_from_row(row: Any) -> ProposalEventRecord:
    return ProposalEventRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        proposal_id=UUID(row.proposal_id) if row.proposal_id else None,
        event_type=row.event_type,
        actor=row.actor,
        payload=row.payload,
        created_at=SessionServiceImpl._ensure_utc(row.created_at),
    )
```

Add the service methods to `SessionServiceImpl`:

```python
    async def get_composer_preferences(self, session_id: UUID) -> ComposerSessionPreferencesRecord:
        def _sync() -> ComposerSessionPreferencesRecord:
            with self._engine.begin() as conn:
                row = conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).one()
                return ComposerSessionPreferencesRecord(
                    session_id=UUID(row.id),
                    trust_mode=row.trust_mode,
                    density_default=row.density_default,
                    updated_at=self._ensure_utc(row.updated_at),
                )

        return await self._run_sync(_sync)

    async def update_composer_preferences(
        self,
        session_id: UUID,
        *,
        trust_mode: ComposerTrustMode,
        density_default: ComposerDensityDefault,
        actor: str,
    ) -> ComposerSessionPreferencesRecord:
        now = self._now()

        def _sync() -> ComposerSessionPreferencesRecord:
            event_id = str(uuid.uuid4())
            with self._engine.begin() as conn:
                with self._session_write_lock(conn, str(session_id)):
                    conn.execute(
                        proposal_events_table.insert().values(
                            id=event_id,
                            session_id=str(session_id),
                            proposal_id=None,
                            event_type="trust_mode.changed",
                            actor=actor,
                            payload={
                                "trust_mode": trust_mode,
                                "density_default": density_default,
                            },
                            created_at=now,
                        )
                    )
                    conn.execute(
                        sessions_table.update()
                        .where(sessions_table.c.id == str(session_id))
                        .values(
                            trust_mode=trust_mode,
                            density_default=density_default,
                            updated_at=now,
                        )
                    )
                    row = conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).one()
                    return ComposerSessionPreferencesRecord(
                        session_id=UUID(row.id),
                        trust_mode=row.trust_mode,
                        density_default=row.density_default,
                        updated_at=self._ensure_utc(row.updated_at),
                    )

        return await self._run_sync(_sync)
```

Implement `create_composition_proposal`, `list_composition_proposals`, `reject_composition_proposal`, and `list_proposal_events` with the same `_run_sync` pattern. In `create_composition_proposal`, insert the `proposal_events` row before the `composition_proposals` row in the same transaction, set `status="pending"`, and store the created event id in `audit_event_id`. In `reject_composition_proposal`, require `status == "pending"`, insert `proposal.rejected` before updating the proposal row to `rejected`, and return the updated record.

- [x] **Step 5: Run the service tests**

Task result: `.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py -q` passed with `9 passed`; schema follow-up `.venv/bin/pytest tests/unit/web/sessions/test_schema.py tests/unit/web/sessions/test_composer_proposals.py -q` passed with `18 passed`.

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_composer_proposals.py
git commit -m "$(cat <<'EOF'
feat: add composer proposal service methods

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 3: Proposal Summary and Tool Classification

**Files:**
- Modify: `src/elspeth/web/composer/tools.py`
- Create: `src/elspeth/web/composer/proposals.py`
- Test: `tests/unit/web/composer/test_proposals.py`

- [x] **Step 1: Write failing proposal summary tests**

Create `tests/unit/web/composer/test_proposals.py`:

```python
from __future__ import annotations

from elspeth.web.composer.proposals import build_tool_proposal_summary
from elspeth.web.composer.tools import is_mutation_tool


def test_is_mutation_tool_uses_closed_registries() -> None:
    assert is_mutation_tool("set_pipeline") is True
    assert is_mutation_tool("set_source_from_blob") is True
    assert is_mutation_tool("get_pipeline_state") is False
    assert is_mutation_tool("preview_pipeline") is False


def test_set_pipeline_summary_is_plain_language() -> None:
    summary = build_tool_proposal_summary(
        tool_name="set_pipeline",
        arguments={
            "source": {"plugin": "csv", "options": {}},
            "nodes": [{"id": "classify_severity", "plugin": "llm_classifier"}],
            "outputs": [{"name": "out", "plugin": "json"}],
        },
        redacted_arguments={
            "source": {"plugin": "csv", "options": {}},
            "nodes": [{"id": "classify_severity", "plugin": "llm_classifier"}],
            "outputs": [{"name": "out", "plugin": "json"}],
        },
    )

    assert summary.summary == "Replace the pipeline with csv input, 1 transform, and 1 output."
    assert summary.rationale == "Requested by the current composer turn."
    assert summary.affects == ("graph", "validation", "yaml")


def test_patch_node_options_summary_names_target_node() -> None:
    summary = build_tool_proposal_summary(
        tool_name="patch_node_options",
        arguments={"node_id": "classify_severity", "patch": {"model": "claude-haiku-4-5"}},
        redacted_arguments={"node_id": "classify_severity", "patch": {"model": "claude-haiku-4-5"}},
    )

    assert summary.summary == 'Update options for transform "classify_severity".'
    assert summary.affects == ("graph", "validation", "yaml")
```

- [x] **Step 2: Run proposal tests to verify they fail**

Task result: `.venv/bin/pytest tests/unit/web/composer/test_proposals.py -q` failed during collection with `ModuleNotFoundError: No module named 'elspeth.web.composer.proposals'`, the expected missing-module failure.

Run:

```bash
.venv/bin/pytest tests/unit/web/composer/test_proposals.py -q
```

Expected: FAIL with missing module or function.

- [x] **Step 3: Export mutation classification from the tool registry**

In `src/elspeth/web/composer/tools.py`, add near `is_discovery_tool`:

```python
def is_mutation_tool(name: str) -> bool:
    """Return True when a composer tool can mutate session state or owned artifacts."""
    return name in _MUTATION_TOOLS or name in _BLOB_MUTATION_TOOLS or name in _SECRET_MUTATION_TOOLS
```

- [x] **Step 4: Add the proposal summary module**

Create `src/elspeth/web/composer/proposals.py`:

```python
"""Plain-language summaries for pending composer tool proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class ToolProposalSummary:
    summary: str
    rationale: str
    affects: tuple[str, ...]
    arguments_redacted_json: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "affects", "arguments_redacted_json")


def _count_items(value: object) -> int:
    return len(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else 0


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def build_tool_proposal_summary(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    redacted_arguments: Mapping[str, Any],
) -> ToolProposalSummary:
    rationale = "Requested by the current composer turn."
    affects = ("graph", "validation", "yaml")

    if tool_name == "set_pipeline":
        source = arguments.get("source")
        source_plugin = "new"
        if isinstance(source, Mapping) and isinstance(source.get("plugin"), str):
            source_plugin = source["plugin"]
        node_count = _count_items(arguments.get("nodes", ()))
        output_count = _count_items(arguments.get("outputs", ()))
        return ToolProposalSummary(
            summary=f"Replace the pipeline with {source_plugin} input, {_plural(node_count, 'transform')}, and {_plural(output_count, 'output')}.",
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "set_source":
        plugin = arguments.get("plugin")
        label = plugin if isinstance(plugin, str) and plugin else "source"
        return ToolProposalSummary(
            summary=f"Set the pipeline source to {label}.",
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "patch_node_options":
        node_id = arguments.get("node_id")
        label = node_id if isinstance(node_id, str) and node_id else "selected transform"
        return ToolProposalSummary(
            summary=f'Update options for transform "{label}".',
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "patch_source_options":
        return ToolProposalSummary(
            summary="Update source options.",
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    if tool_name == "patch_output_options":
        sink_name = arguments.get("sink_name")
        label = sink_name if isinstance(sink_name, str) and sink_name else "selected output"
        return ToolProposalSummary(
            summary=f'Update options for output "{label}".',
            rationale=rationale,
            affects=affects,
            arguments_redacted_json=redacted_arguments,
        )

    return ToolProposalSummary(
        summary=f"Apply composer tool {tool_name}.",
        rationale=rationale,
        affects=affects,
        arguments_redacted_json=redacted_arguments,
    )
```

- [x] **Step 5: Run proposal tests**

Task result: `.venv/bin/pytest tests/unit/web/composer/test_proposals.py -q` passed with `3 passed`; ruff and tier-model checks also passed for the touched composer surfaces.

Run:

```bash
.venv/bin/pytest tests/unit/web/composer/test_proposals.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py src/elspeth/web/composer/proposals.py tests/unit/web/composer/test_proposals.py
git commit -m "$(cat <<'EOF'
feat: summarize composer tool proposals

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 4: Session API for Preferences and Proposals

**Files:**
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Test: `tests/unit/web/sessions/test_routes.py`

- [x] **Step 1: Add failing route tests**

Append to `tests/unit/web/sessions/test_routes.py`:

```python
def test_get_composer_preferences_returns_defaults(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Prefs"}).json()

    response = test_client.get(f"/api/sessions/{session['id']}/composer/preferences")

    assert response.status_code == 200
    assert response.json()["trust_mode"] == "explicit_approve"
    assert response.json()["density_default"] == "high"


def test_patch_composer_preferences_records_event(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Prefs"}).json()

    response = test_client.patch(
        f"/api/sessions/{session['id']}/composer/preferences",
        json={"trust_mode": "auto_commit", "density_default": "medium"},
    )

    assert response.status_code == 200
    assert response.json()["trust_mode"] == "auto_commit"
    events = test_client.get(f"/api/sessions/{session['id']}/proposal-events").json()
    assert events[-1]["event_type"] == "trust_mode.changed"


def test_list_proposals_is_session_scoped(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Proposals"}).json()

    response = test_client.get(f"/api/sessions/{session['id']}/proposals")

    assert response.status_code == 200
    assert response.json() == []


def test_send_message_response_includes_empty_proposals_array(tmp_path) -> None:
    mock_composer = _make_composer_mock(response_text="Got it!")
    app, _service = _make_app(tmp_path)
    app.state.composer_service = mock_composer
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Chat"}).json()

    response = client.post(
        f"/api/sessions/{session['id']}/messages",
        json={"content": "Hello"},
    )

    assert response.status_code == 200
    assert response.json()["proposals"] == []
```

- [x] **Step 2: Run route tests to verify they fail**

Task result: focused route tests failed with 404 responses for the new preferences/proposals endpoints and `KeyError: 'proposals'` for the send-message response, matching the missing API/contract shape.

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_routes.py::test_get_composer_preferences_returns_defaults tests/unit/web/sessions/test_routes.py::test_patch_composer_preferences_records_event tests/unit/web/sessions/test_routes.py::test_list_proposals_is_session_scoped -q
```

Expected: FAIL with 404 route responses or missing schemas.

- [x] **Step 3: Add Pydantic schemas**

In `src/elspeth/web/sessions/schemas.py`, add:

```python
class ComposerPreferencesResponse(_StrictResponse):
    session_id: str
    trust_mode: str
    density_default: str
    updated_at: datetime


class UpdateComposerPreferencesRequest(_RequestModel):
    trust_mode: str
    density_default: str


class CompositionProposalResponse(_StrictResponse):
    id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    status: str
    summary: str
    rationale: str
    affects: list[str]
    arguments_redacted_json: CompositionObject
    base_state_id: str | None = None
    committed_state_id: str | None = None
    audit_event_id: str | None = None
    created_at: datetime
    updated_at: datetime


class RejectProposalRequest(_RequestModel):
    reason: str | None = None


class ProposalEventResponse(_StrictResponse):
    id: str
    session_id: str
    proposal_id: str | None = None
    event_type: str
    actor: str
    payload: CompositionObject
    created_at: datetime
```

Also extend the existing `MessageWithStateResponse` so compose/recompose can return the current pending proposals atomically with the assistant message:

```python
class MessageWithStateResponse(_StrictResponse):
    message: ChatMessageResponse
    state: CompositionStateResponse | None
    proposals: list[CompositionProposalResponse]
```

Import `ProposalLifecycleStatus` into `routes.py` from `elspeth.web.sessions.protocol` when adding the `status` query parameter above. The route boundary must reject unknown status values with FastAPI's Literal validation instead of silently returning an empty list for `?status=garbage`.

- [x] **Step 4: Add route converters and endpoints**

In `src/elspeth/web/sessions/routes.py`, import the new schemas and add converters near `_session_response`:

```python
def _composer_preferences_response(record: ComposerSessionPreferencesRecord) -> ComposerPreferencesResponse:
    return ComposerPreferencesResponse(
        session_id=str(record.session_id),
        trust_mode=record.trust_mode,
        density_default=record.density_default,
        updated_at=record.updated_at,
    )


def _composition_proposal_response(record: CompositionProposalRecord) -> CompositionProposalResponse:
    return CompositionProposalResponse(
        id=str(record.id),
        session_id=str(record.session_id),
        tool_call_id=record.tool_call_id,
        tool_name=record.tool_name,
        status=record.status,
        summary=record.summary,
        rationale=record.rationale,
        affects=list(record.affects),
        arguments_redacted_json=dict(record.arguments_redacted_json),
        base_state_id=str(record.base_state_id) if record.base_state_id else None,
        committed_state_id=str(record.committed_state_id) if record.committed_state_id else None,
        audit_event_id=str(record.audit_event_id) if record.audit_event_id else None,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
```

Add endpoints inside `create_session_router()`:

```python
    @router.get(
        "/{session_id}/composer/preferences",
        response_model=ComposerPreferencesResponse,
    )
    async def get_composer_preferences(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> ComposerPreferencesResponse:
        session = await _verify_session_ownership(session_id, user, request)
        prefs = await request.app.state.session_service.get_composer_preferences(session.id)
        return _composer_preferences_response(prefs)

    @router.patch(
        "/{session_id}/composer/preferences",
        response_model=ComposerPreferencesResponse,
    )
    async def update_composer_preferences(
        session_id: UUID,
        body: UpdateComposerPreferencesRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> ComposerPreferencesResponse:
        session = await _verify_session_ownership(session_id, user, request)
        prefs = await request.app.state.session_service.update_composer_preferences(
            session.id,
            trust_mode=body.trust_mode,
            density_default=body.density_default,
            actor=f"user:{user.user_id}",
        )
        return _composer_preferences_response(prefs)

    @router.get(
        "/{session_id}/proposals",
        response_model=list[CompositionProposalResponse],
    )
    async def list_composition_proposals(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
        status: ProposalLifecycleStatus | None = Query(None),
    ) -> list[CompositionProposalResponse]:
        session = await _verify_session_ownership(session_id, user, request)
        proposals = await request.app.state.session_service.list_composition_proposals(session.id, status=status)
        return [_composition_proposal_response(proposal) for proposal in proposals]

    @router.post(
        "/{session_id}/proposals/{proposal_id}/reject",
        response_model=CompositionProposalResponse,
    )
    async def reject_composition_proposal(
        session_id: UUID,
        proposal_id: UUID,
        body: RejectProposalRequest,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> CompositionProposalResponse:
        session = await _verify_session_ownership(session_id, user, request)
        proposal = await request.app.state.session_service.reject_composition_proposal(
            session_id=session.id,
            proposal_id=proposal_id,
            actor=f"user:{user.user_id}",
        )
        _ = body
        return _composition_proposal_response(proposal)
```

- [x] **Step 5: Return proposals from compose/recompose responses**

Add a helper near `_composition_proposal_response` so both compose response paths share the same server-side shape:

```python
async def _pending_proposal_responses(
    service: SessionServiceProtocol,
    session_id: UUID,
) -> list[CompositionProposalResponse]:
    proposals = await service.list_composition_proposals(session_id, status="pending")
    return [_composition_proposal_response(proposal) for proposal in proposals]
```

Update both existing `MessageWithStateResponse(...)` construction sites in `send_message` and `recompose`. As of the preflight pass, they are the response blocks after the comments "Build the response object FIRST" and "See send_message return-flow comment". The new code shape is:

```python
                proposals = await _pending_proposal_responses(service, session.id)
                response = MessageWithStateResponse(
                    message=_message_response(assistant_msg),
                    state=state_response,
                    proposals=proposals,
                )
```

Do not only update frontend API types. This is a backend contract change: every live `POST /api/sessions/{session_id}/messages` and recompose response must include `proposals`, even when the array is empty. The route test in Step 1 pins the empty-array shape before Task 5 creates real proposal rows.

- [x] **Step 6: Run route tests**

Task result: focused route contract command passed with `4 passed`; `ruff check`, `mypy` for `schemas.py`/`routes.py`, and tier-model check passed after refreshing stale route allowlist fingerprints caused by line shifts.

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_routes.py::test_get_composer_preferences_returns_defaults tests/unit/web/sessions/test_routes.py::test_patch_composer_preferences_records_event tests/unit/web/sessions/test_routes.py::test_list_proposals_is_session_scoped -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_routes.py
git commit -m "$(cat <<'EOF'
feat: expose composer proposal APIs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 5: Explicit-Approve Compose Loop Proposal Creation

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `tests/unit/web/composer/conftest.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Test: `tests/unit/web/composer/test_service.py`
- Test: `tests/unit/web/sessions/test_routes.py`

- [x] **Step 1: Add the set_pipeline fake LLM fixture**

In `tests/unit/web/composer/conftest.py`, add:

```python
@pytest.fixture
def fake_llm_one_set_pipeline_tool_call() -> _FakeComposeLLM:
    return _FakeComposeLLM(
        (
            _fake_llm_response(
                tool_calls=(
                    {
                        "id": "call_set_pipeline",
                        "name": "set_pipeline",
                        "arguments": {
                            "source": {
                                "plugin": "csv",
                                "options": {"path": "data/input.csv"},
                            },
                            "nodes": [],
                            "outputs": [
                                {
                                    "name": "out",
                                    "plugin": "json",
                                    "options": {"path": "data/output.json"},
                                }
                            ],
                            "metadata": {"name": "proposal-test"},
                        },
                    },
                )
            ),
            _fake_llm_response(content="Done."),
        )
    )
```

- [x] **Step 2: Write the failing compose-loop test**

Add a test in `tests/unit/web/composer/test_service.py` using the existing fake LLM fixtures:

```python
async def test_explicit_approve_mutating_tool_creates_pending_proposal_without_state_mutation(
    composer_service,
    session_service,
    fake_llm_one_set_pipeline_tool_call,
) -> None:
    session = await session_service.create_session(
        user_id="alice",
        title="Explicit approve",
        auth_provider_type="local",
    )
    await session_service.update_composer_preferences(
        session.id,
        trust_mode="explicit_approve",
        density_default="high",
        actor="user:alice",
    )
    composer_service._llm = fake_llm_one_set_pipeline_tool_call

    result = await composer_service.compose(
        "Build a csv to json pipeline",
        messages=[],
        state=CompositionState(),
        user_id="alice",
        session_id=str(session.id),
    )

    assert result.updated_state is None
    proposals = await session_service.list_composition_proposals(session.id, status="pending")
    assert len(proposals) == 1
    assert proposals[0].tool_call_id == "call_set_pipeline"
    assert proposals[0].tool_name == "set_pipeline"
    assert proposals[0].status == "pending"
```

- [x] **Step 3: Run the compose-loop test to verify it fails**

Run:

```bash
.venv/bin/pytest tests/unit/web/composer/test_service.py::test_explicit_approve_mutating_tool_creates_pending_proposal_without_state_mutation -q
```

Expected: FAIL because mutating tools still execute immediately.

- [x] **Step 4: Add proposal creation to the composer service path**

In the compose loop section that iterates `assistant_message.tool_calls`, get the wired session service once with `sessions_service = self._require_sessions_service()`, initialize `proposals_this_turn = 0` at the turn scope, and after decoding/required-path validation but before `execute_tool`, add this branch:

```python
                preferences = await sessions_service.get_composer_preferences(UUID(session_id))
                if preferences.trust_mode == "explicit_approve" and is_mutation_tool(tool_name):
                    if proposals_this_turn >= _MAX_PENDING_PROPOSALS_PER_TURN:
                        raise ComposerServiceError(
                            f"Composer produced more than {_MAX_PENDING_PROPOSALS_PER_TURN} pending proposals in one turn"
                        )

                    redacted_args = (
                        redact_tool_call_arguments(tool_name, arguments, telemetry=self._redaction_telemetry)
                        if tool_name in MANIFEST
                        else arguments
                    )
                    proposal_summary = build_tool_proposal_summary(
                        tool_name=tool_name,
                        arguments=arguments,
                        redacted_arguments=redacted_args,
                    )
                    await sessions_service.create_composition_proposal(
                        session_id=UUID(session_id),
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        summary=proposal_summary.summary,
                        rationale=proposal_summary.rationale,
                        affects=proposal_summary.affects,
                        arguments_json=arguments,
                        arguments_redacted_json=proposal_summary.arguments_redacted_json,
                        base_state_id=UUID(current_state_id) if current_state_id else None,
                        actor=actor,
                    )
                    proposals_this_turn += 1
                    proposal_payload = {
                        "status": "PENDING_USER_DECISION",
                        "proposal_tool_call_id": tool_call.id,
                        "summary": proposal_summary.summary,
                    }
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(proposal_payload),
                        }
                    )
                    turn_has_mutation = True
                    continue
```

Make sure `is_mutation_tool` is imported from `elspeth.web.composer.tools`, `build_tool_proposal_summary` is imported from `elspeth.web.composer.proposals`, `MANIFEST`/`redact_tool_call_arguments` are imported from `elspeth.web.composer.redaction`, and `UUID` is imported from `uuid`. Add `_MAX_PENDING_PROPOSALS_PER_TURN = 10` near the other compose-loop constants, and initialize/increment `proposals_this_turn` at the correct loop scope so a runaway LLM cannot create an unbounded wall of cards.

Use the actor convention from `ProposalEventRecord`: proposals created from the composer loop use `actor=f"composer-web:user:{user_id}"`; direct human actions use `actor=f"user:{user_id}"`; background repairs use `actor="system:<component>"`.

- [x] **Step 5: Preserve auto-commit behavior**

Add a second test beside the first:

```python
async def test_auto_commit_mutating_tool_preserves_existing_state_mutation_path(
    composer_service,
    session_service,
    fake_llm_one_set_pipeline_tool_call,
) -> None:
    session = await session_service.create_session(
        user_id="alice",
        title="Auto commit",
        auth_provider_type="local",
    )
    await session_service.update_composer_preferences(
        session.id,
        trust_mode="auto_commit",
        density_default="high",
        actor="user:alice",
    )
    composer_service._llm = fake_llm_one_set_pipeline_tool_call

    result = await composer_service.compose(
        "Build a csv to json pipeline",
        messages=[],
        state=CompositionState(),
        user_id="alice",
        session_id=str(session.id),
    )

    assert result.updated_state is not None
    proposals = await session_service.list_composition_proposals(session.id)
    assert proposals == []
```

- [x] **Step 6: Run compose-loop tests**

Run:

```bash
.venv/bin/pytest tests/unit/web/composer/test_service.py::test_explicit_approve_mutating_tool_creates_pending_proposal_without_state_mutation tests/unit/web/composer/test_service.py::test_auto_commit_mutating_tool_preserves_existing_state_mutation_path -q
```

Expected: PASS.

- [x] **Step 7: Add route regression for atomic proposal response**

Add to `tests/unit/web/sessions/test_routes.py`:

```python
def test_send_message_response_includes_pending_proposals_created_during_compose(tmp_path) -> None:
    app, service = _make_app(tmp_path)
    mock_composer = AsyncMock()

    async def _compose_with_pending_proposal(*args: object, **kwargs: object) -> ComposerResult:
        session_id = UUID(str(kwargs["session_id"]))
        await service.create_composition_proposal(
            session_id=session_id,
            tool_call_id="call_set_pipeline",
            tool_name="set_pipeline",
            summary="Replace the pipeline.",
            rationale="Requested by the current composer turn.",
            affects=("graph", "yaml"),
            arguments_json={"source": {"plugin": "csv", "options": {}}},
            arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
            base_state_id=None,
            actor="composer-web:alice",
        )
        return ComposerResult(message="Needs approval.", state=_EMPTY_STATE)

    mock_composer.compose = AsyncMock(side_effect=_compose_with_pending_proposal)
    app.state.composer_service = mock_composer
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Atomic proposals"}).json()

    response = client.post(
        f"/api/sessions/{session['id']}/messages",
        json={"content": "Build a csv pipeline"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["content"] == "Needs approval."
    assert body["proposals"][0]["tool_call_id"] == "call_set_pipeline"
    assert body["proposals"][0]["status"] == "pending"
```

If `UUID` is not already imported in `test_routes.py`, add `from uuid import UUID`. This test is deliberately route-level: it proves the live backend response includes proposals created during the compose request, so Task 7's mocked frontend test cannot drift away from the server contract.

- [x] **Step 8: Run route regression**

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_routes.py::test_send_message_response_includes_pending_proposals_created_during_compose -q
```

Expected: PASS.

- [x] **Step 9: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/conftest.py tests/unit/web/composer/test_service.py tests/unit/web/sessions/test_routes.py
git commit -m "$(cat <<'EOF'
feat: create pending proposals for explicit approval

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 6: Accept Proposal Execution Path

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Test: `tests/unit/web/sessions/test_composer_proposals.py`
- Test: `tests/unit/web/sessions/test_routes.py`

- [x] **Step 1: Write failing accept tests**

Add to `tests/unit/web/sessions/test_composer_proposals.py`:

```python
async def test_accept_composition_proposal_requires_pending_status(service) -> None:
    session_id = uuid4()
    with service._engine.begin() as conn:
        _insert_session(conn, str(session_id))
    proposal = await service.create_composition_proposal(
        session_id=session_id,
        tool_call_id="call_set_pipeline",
        tool_name="set_pipeline",
        summary="Replace the pipeline.",
        rationale="Requested by the user.",
        affects=("graph",),
        arguments_json={"source": {"plugin": "csv", "options": {}}},
        arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
        base_state_id=None,
        actor="composer-web:user-alice",
    )
    await service.reject_composition_proposal(
        session_id=session_id,
        proposal_id=proposal.id,
        actor="user:alice",
    )

    with pytest.raises(ValueError, match="pending"):
        await service.mark_composition_proposal_committed(
            session_id=session_id,
            proposal_id=proposal.id,
            committed_state_id=uuid4(),
            actor="user:alice",
        )
```

Add to `tests/unit/web/sessions/test_routes.py`:

```python
def test_accept_unknown_proposal_returns_404(test_client) -> None:
    session = test_client.post("/api/sessions", json={"title": "Accept"}).json()
    response = test_client.post(
        f"/api/sessions/{session['id']}/proposals/00000000-0000-0000-0000-000000000000/accept"
    )
    assert response.status_code == 404
```

- [x] **Step 2: Run accept tests to verify they fail**

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py::test_accept_composition_proposal_requires_pending_status tests/unit/web/sessions/test_routes.py::test_accept_unknown_proposal_returns_404 -q
```

Expected: FAIL with missing `mark_composition_proposal_committed` or missing route.

- [x] **Step 3: Add committed-state marker method**

In `SessionServiceProtocol`, add:

```python
    async def mark_composition_proposal_committed(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        committed_state_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord: ...
```

In `SessionServiceImpl`, implement:

```python
    async def mark_composition_proposal_committed(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        committed_state_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord:
        now = self._now()

        def _sync() -> CompositionProposalRecord:
            with self._engine.begin() as conn:
                with self._session_write_lock(conn, str(session_id)):
                    row = conn.execute(
                        select(composition_proposals_table)
                        .where(composition_proposals_table.c.id == str(proposal_id))
                        .where(composition_proposals_table.c.session_id == str(session_id))
                    ).one_or_none()
                    if row is None:
                        raise KeyError(str(proposal_id))
                    if row.status != "pending":
                        raise ValueError(f"Proposal {proposal_id} must be pending to commit; got {row.status!r}")
                    event_id = str(uuid.uuid4())
                    conn.execute(
                        proposal_events_table.insert().values(
                            id=event_id,
                            session_id=str(session_id),
                            proposal_id=str(proposal_id),
                            event_type="proposal.accepted",
                            actor=actor,
                            payload={"committed_state_id": str(committed_state_id)},
                            created_at=now,
                        )
                    )
                    conn.execute(
                        composition_proposals_table.update()
                        .where(composition_proposals_table.c.id == str(proposal_id))
                        .values(
                            status="committed",
                            committed_state_id=str(committed_state_id),
                            audit_event_id=event_id,
                            updated_at=now,
                        )
                    )
                    updated = conn.execute(
                        select(composition_proposals_table).where(composition_proposals_table.c.id == str(proposal_id))
                    ).one()
                    return _proposal_record_from_row(updated)

        return await self._run_sync(_sync)
```

- [x] **Step 4: Add accept route skeleton**

Add the route:

```python
    @router.post(
        "/{session_id}/proposals/{proposal_id}/accept",
        response_model=CompositionProposalResponse,
    )
    async def accept_composition_proposal(
        session_id: UUID,
        proposal_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),
    ) -> CompositionProposalResponse:
        session = await _verify_session_ownership(session_id, user, request)
        service = request.app.state.session_service
        proposals = await service.list_composition_proposals(session.id)
        proposal = next((item for item in proposals if item.id == proposal_id), None)
        if proposal is None:
            raise HTTPException(status_code=404, detail="Proposal not found")
        raise HTTPException(
            status_code=501,
            detail="Proposal execution wiring is implemented in the next step of this task.",
        )
```

- [x] **Step 5: Wire execution through existing composer tool handler**

Replace the 501 branch with execution code that:

1. Fetches the current `CompositionStateRecord`.
2. Converts it to `CompositionState` using existing `_state_from_record`.
3. Resolves `catalog_service`, `phase3_engine`, data directory, and secret service from the same app-state/settings seams used by the existing route helpers. Do not assume `request.app.state.data_dir` or `request.app.state.secrets_service` exist if Task 0 proved otherwise.
4. Calls `execute_tool(proposal.tool_name, dict(proposal.arguments_json), state, catalog, data_dir=..., session_engine=..., session_id=str(session.id), secret_service=..., user_id=user.user_id)`.
5. If the result did not mutate state, returns 409 with a clear `detail`.
6. Persists the returned state with `save_composition_state(..., provenance="tool_call")`.
7. Calls `mark_composition_proposal_committed`.

The route code shape:

```python
        current_record = await service.get_current_state(session.id)
        if proposal.base_state_id is not None and (
            current_record is None or current_record.id != proposal.base_state_id
        ):
            raise HTTPException(
                status_code=409,
                detail="The session state changed after this proposal was created. Ask ELSPETH to rebase the proposal.",
            )
        current_state = _state_from_record(current_record) if current_record else CompositionState()
        result = await run_sync_in_worker(
            execute_tool,
            proposal.tool_name,
            dict(proposal.arguments_json),
            current_state,
            request.app.state.catalog_service,
            data_dir=resolved_data_dir,
            session_engine=request.app.state.phase3_engine,
            session_id=str(session.id),
            secret_service=resolved_secret_service,
            user_id=user.user_id,
        )
        if result.updated_state.version == current_state.version:
            raise HTTPException(
                status_code=409,
                detail="Accepted proposal did not change composition state.",
            )
        state_data, _validation = await _state_data_from_composer_state(
            result.updated_state,
            repair_turns_used=None,
            # pass the same settings/user/service arguments required by the
            # current helper signature in routes.py
        )
        state_record = await service.save_composition_state(
            session.id,
            state_data,
            provenance="tool_call",
        )
        committed = await service.mark_composition_proposal_committed(
            session_id=session.id,
            proposal_id=proposal.id,
            committed_state_id=state_record.id,
            actor=f"user:{user.user_id}",
        )
        return _composition_proposal_response(committed)
```

- [x] **Step 6: Run accept tests**

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py tests/unit/web/sessions/test_routes.py::test_accept_unknown_proposal_returns_404 -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/routes.py tests/unit/web/sessions/test_composer_proposals.py tests/unit/web/sessions/test_routes.py
git commit -m "$(cat <<'EOF'
feat: accept and audit composer proposals

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 7: Frontend API, Store, and Types

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Test: `src/elspeth/web/frontend/src/stores/sessionStore.test.ts`

- [x] **Step 1: Add failing store tests**

Add to `src/elspeth/web/frontend/src/stores/sessionStore.test.ts`:

```typescript
it("loads proposals when selecting a session", async () => {
  vi.spyOn(api, "fetchSessions").mockResolvedValue([]);
  vi.spyOn(api, "fetchMessages").mockResolvedValue([]);
  vi.spyOn(api, "fetchCompositionState").mockResolvedValue(null);
  vi.spyOn(api, "fetchCompositionProposals").mockResolvedValue([
    {
      id: "proposal-1",
      session_id: "session-1",
      tool_call_id: "call-1",
      tool_name: "set_pipeline",
      status: "pending",
      summary: "Replace the pipeline.",
      rationale: "Requested by the current composer turn.",
      affects: ["graph", "validation", "yaml"],
      arguments_redacted_json: {},
      base_state_id: null,
      committed_state_id: null,
      audit_event_id: "event-1",
      created_at: "2026-05-14T00:00:00Z",
      updated_at: "2026-05-14T00:00:00Z",
    },
  ]);
  vi.spyOn(api, "fetchComposerPreferences").mockResolvedValue({
    session_id: "session-1",
    trust_mode: "explicit_approve",
    density_default: "high",
    updated_at: "2026-05-14T00:00:00Z",
  });
  vi.spyOn(api, "getGuided").mockRejectedValue(new Error("guided unavailable"));

  await useSessionStore.getState().selectSession("session-1");

  expect(useSessionStore.getState().compositionProposals).toHaveLength(1);
  expect(useSessionStore.getState().composerPreferences?.trust_mode).toBe("explicit_approve");
});

it("appends proposals returned by sendMessage without waiting for a session reload", async () => {
  vi.spyOn(api, "sendMessage").mockResolvedValue({
    message: {
      id: "assistant-1",
      session_id: "session-1",
      role: "assistant",
      content: "I need approval.",
      tool_calls: [],
      tool_call_id: null,
      parent_assistant_id: null,
      created_at: "2026-05-14T00:00:00Z",
      composition_state_id: null,
    },
    state: null,
    proposals: [
      {
        id: "proposal-1",
        session_id: "session-1",
        tool_call_id: "call-1",
        tool_name: "set_pipeline",
        status: "pending",
        summary: "Replace the pipeline.",
        rationale: "Requested by the current composer turn.",
        affects: ["graph"],
        arguments_redacted_json: {},
        base_state_id: null,
        committed_state_id: null,
        audit_event_id: "event-1",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      },
    ],
  });
  useSessionStore.setState({ activeSessionId: "session-1", messages: [] });

  await useSessionStore.getState().sendMessage("build it");

  expect(useSessionStore.getState().compositionProposals).toHaveLength(1);
});

it("marks stale proposals after accept returns a stale-state conflict", async () => {
  vi.spyOn(api, "acceptCompositionProposal").mockRejectedValue(
    Object.assign(new Error("stale"), { status: 409 }),
  );
  vi.spyOn(api, "fetchCompositionProposals").mockResolvedValue([]);
  useSessionStore.setState({ activeSessionId: "session-1" });

  await useSessionStore.getState().acceptProposal("proposal-1");

  expect(useSessionStore.getState().staleProposalIds).toContain("proposal-1");
});
```

- [x] **Step 2: Run the store test to verify it fails**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/stores/sessionStore.test.ts
```

Expected: FAIL with missing API/store fields.

Observed: FAIL with `compositionProposals` undefined and missing `acceptProposal`; initial run was blocked by missing frontend `node_modules`, then `npm ci` installed the lockfile dependencies.

- [x] **Step 3: Add frontend types**

In `src/elspeth/web/frontend/src/types/index.ts`, add:

```typescript
export type ComposerTrustMode = "explicit_approve" | "auto_commit";
export type ComposerDensityDefault = "high" | "medium" | "low";
export type ProposalLifecycleStatus = "pending" | "committed" | "rejected";

export interface ComposerPreferences {
  session_id: string;
  trust_mode: ComposerTrustMode;
  density_default: ComposerDensityDefault;
  updated_at: string;
}

export interface CompositionProposal {
  id: string;
  session_id: string;
  tool_call_id: string;
  tool_name: string;
  status: ProposalLifecycleStatus;
  summary: string;
  rationale: string;
  affects: string[];
  arguments_redacted_json: Record<string, unknown>;
  base_state_id: string | null;
  committed_state_id: string | null;
  audit_event_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface MessageWithStateResponse {
  message: ChatMessage;
  state: CompositionState | null;
  proposals: CompositionProposal[];
}
```

Re-export them from `src/elspeth/web/frontend/src/types/api.ts`.

- [x] **Step 4: Add API client methods**

In `src/elspeth/web/frontend/src/api/client.ts`, add:

```typescript
export async function fetchComposerPreferences(sessionId: string): Promise<ComposerPreferences> {
  const response = await fetch(`/api/sessions/${sessionId}/composer/preferences`, {
    headers: authHeaders(),
  });
  return parseResponse<ComposerPreferences>(response);
}

export async function updateComposerPreferences(
  sessionId: string,
  body: Pick<ComposerPreferences, "trust_mode" | "density_default">,
): Promise<ComposerPreferences> {
  const response = await fetch(`/api/sessions/${sessionId}/composer/preferences`, {
    method: "PATCH",
    headers: authHeaders("application/json"),
    body: JSON.stringify(body),
  });
  return parseResponse<ComposerPreferences>(response);
}

export async function fetchCompositionProposals(sessionId: string): Promise<CompositionProposal[]> {
  const response = await fetch(`/api/sessions/${sessionId}/proposals`, {
    headers: authHeaders(),
  });
  return parseResponse<CompositionProposal[]>(response);
}

export async function acceptCompositionProposal(
  sessionId: string,
  proposalId: string,
): Promise<CompositionProposal> {
  const response = await fetch(`/api/sessions/${sessionId}/proposals/${proposalId}/accept`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({}),
  });
  return parseResponse<CompositionProposal>(response);
}

export async function rejectCompositionProposal(
  sessionId: string,
  proposalId: string,
): Promise<CompositionProposal> {
  const response = await fetch(`/api/sessions/${sessionId}/proposals/${proposalId}/reject`, {
    method: "POST",
    headers: authHeaders("application/json"),
    body: JSON.stringify({}),
  });
  return parseResponse<CompositionProposal>(response);
}
```

Also update existing message-composition API methods:

```typescript
export async function sendMessage(...): Promise<MessageWithStateResponse> {
  // preserve existing request construction
  return parseResponse<MessageWithStateResponse>(response);
}

export async function recomposeMessage(...): Promise<MessageWithStateResponse> {
  // preserve existing request construction
  return parseResponse<MessageWithStateResponse>(response);
}
```

- [x] **Step 5: Add store state and actions**

In `src/elspeth/web/frontend/src/stores/sessionStore.ts`, add fields:

```typescript
  compositionProposals: CompositionProposal[];
  composerPreferences: ComposerPreferences | null;
  staleProposalIds: string[];
  proposalActionPendingIds: string[];
  loadCompositionProposals: (sessionId?: string) => Promise<void>;
  acceptProposal: (proposalId: string) => Promise<void>;
  rejectProposal: (proposalId: string) => Promise<void>;
```

Add initial state:

```typescript
  compositionProposals: [] as CompositionProposal[],
  composerPreferences: null as ComposerPreferences | null,
  staleProposalIds: [] as string[],
  proposalActionPendingIds: [] as string[],
```

Update `api.sendMessage` and `api.recomposeMessage` to return `MessageWithStateResponse`. In `sendMessage`/`retryMessage`, merge `result.proposals` into `compositionProposals` in the same state update as the assistant message so the pending card appears immediately after compose returns.

In `selectSession`, fetch proposals and preferences beside messages/state, and clear stale/action flags for the new session:

```typescript
      const [messages, compositionState, proposals, preferences] = await Promise.all([
        api.fetchMessages(id),
        api.fetchCompositionState(id),
        api.fetchCompositionProposals(id),
        api.fetchComposerPreferences(id),
      ]);
      set({
        messages,
        compositionState,
        compositionProposals: proposals,
        composerPreferences: preferences,
        staleProposalIds: [],
        proposalActionPendingIds: [],
      });
```

Add actions:

```typescript
  async loadCompositionProposals(sessionId?: string) {
    const targetSessionId = sessionId ?? get().activeSessionId;
    if (!targetSessionId) return;
    const proposals = await api.fetchCompositionProposals(targetSessionId);
    if (get().activeSessionId === targetSessionId) {
      set({ compositionProposals: proposals });
    }
  },

  async acceptProposal(proposalId: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) {
      throw new Error("acceptProposal called without active session");
    }
    set((state) => ({
      proposalActionPendingIds: [...state.proposalActionPendingIds, proposalId],
      staleProposalIds: state.staleProposalIds.filter((id) => id !== proposalId),
    }));
    try {
      const proposal = await api.acceptCompositionProposal(activeSessionId, proposalId);
      const compositionState = await api.fetchCompositionState(activeSessionId);
      set((state) => ({
        compositionProposals: state.compositionProposals.map((item) =>
          item.id === proposal.id ? proposal : item,
        ),
        compositionState,
      }));
    } catch (error) {
      if (isHttpConflict(error)) {
        const proposals = await api.fetchCompositionProposals(activeSessionId);
        set((state) => ({
          compositionProposals: proposals,
          staleProposalIds: [...new Set([...state.staleProposalIds, proposalId])],
        }));
        return;
      }
      throw error;
    } finally {
      set((state) => ({
        proposalActionPendingIds: state.proposalActionPendingIds.filter((id) => id !== proposalId),
      }));
    }
  },

  async rejectProposal(proposalId: string) {
    const { activeSessionId } = get();
    if (!activeSessionId) {
      throw new Error("rejectProposal called without active session");
    }
    set((state) => ({
      proposalActionPendingIds: [...state.proposalActionPendingIds, proposalId],
    }));
    try {
      const proposal = await api.rejectCompositionProposal(activeSessionId, proposalId);
      set((state) => ({
        compositionProposals: state.compositionProposals.map((item) =>
          item.id === proposal.id ? proposal : item,
        ),
      }));
    } finally {
      set((state) => ({
        proposalActionPendingIds: state.proposalActionPendingIds.filter((id) => id !== proposalId),
      }));
    }
  },
```

Use the repository's existing API-error shape when implementing `isHttpConflict(error)`; if no helper exists, add one in the API/store test seam and pin it with the stale-state regression above. The UX contract is: 409 from accept does not become an unhandled fetch error; the store refreshes proposals and records the clicked proposal ID as stale for rendering.

- [x] **Step 6: Run store tests**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/stores/sessionStore.test.ts
```

Expected: PASS.

Observed: PASS (`21 passed`). Additional verification: `npm run typecheck` PASS after updating the App-level compose mock to include `proposals: []`; `npm run lint` PASS with five pre-existing warnings in unrelated components.

- [x] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/types/index.ts src/elspeth/web/frontend/src/types/api.ts src/elspeth/web/frontend/src/api/client.ts src/elspeth/web/frontend/src/stores/sessionStore.ts src/elspeth/web/frontend/src/stores/sessionStore.test.ts
git commit -m "$(cat <<'EOF'
feat: load composer proposals in frontend store

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 8: Conversation Tool Call Cards

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/ToolCallCard.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/ToolCallCard.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/MessageBubble.test.tsx`
- Modify: `src/elspeth/web/frontend/src/App.css`

- [x] **Step 1: Write failing ToolCallCard tests**

Create `src/elspeth/web/frontend/src/components/chat/ToolCallCard.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ToolCallCard } from "./ToolCallCard";
import type { CompositionProposal, ToolCall } from "@/types/api";

const toolCall: ToolCall = {
  id: "call-1",
  type: "function",
  function: { name: "set_pipeline", arguments: "{\"source\":{\"plugin\":\"csv\"}}" },
};

const proposal: CompositionProposal = {
  id: "proposal-1",
  session_id: "session-1",
  tool_call_id: "call-1",
  tool_name: "set_pipeline",
  status: "pending",
  summary: "Replace the pipeline with csv input, 1 transform, and 1 output.",
  rationale: "Requested by the current composer turn.",
  affects: ["graph", "validation", "yaml"],
  arguments_redacted_json: { source: { plugin: "csv" } },
  base_state_id: null,
  committed_state_id: null,
  audit_event_id: "event-1",
  created_at: "2026-05-14T00:00:00Z",
  updated_at: "2026-05-14T00:00:00Z",
};

describe("ToolCallCard", () => {
  it("renders pending write proposals with balanced accept and reject actions", () => {
    render(
      <ToolCallCard
        toolCall={toolCall}
        proposal={proposal}
        onAccept={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText("Proposed: set_pipeline")).toBeInTheDocument();
    expect(screen.getByText(proposal.summary)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: `Accept proposal: ${proposal.summary}` }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: `Reject proposal: ${proposal.summary}` }),
    ).toBeInTheDocument();
  });

  it("renders read-only tools as ribbons", () => {
    render(
      <ToolCallCard
        toolCall={{
          id: "read-1",
          type: "function",
          function: { name: "get_pipeline_state", arguments: "{}" },
        }}
        proposal={null}
        onAccept={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText("Looked up: get_pipeline_state")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Accept proposal/ })).not.toBeInTheDocument();
  });

  it("calls accept and reject handlers", async () => {
    const user = userEvent.setup();
    const onAccept = vi.fn();
    const onReject = vi.fn();
    render(
      <ToolCallCard
        toolCall={toolCall}
        proposal={proposal}
        onAccept={onAccept}
        onReject={onReject}
      />,
    );

    await user.click(screen.getByRole("button", { name: `Accept proposal: ${proposal.summary}` }));
    await user.click(screen.getByRole("button", { name: `Reject proposal: ${proposal.summary}` }));

    expect(onAccept).toHaveBeenCalledWith("proposal-1");
    expect(onReject).toHaveBeenCalledWith("proposal-1");
  });

  it("renders stale proposals without actionable accept or reject buttons", () => {
    render(
      <ToolCallCard
        toolCall={toolCall}
        proposal={proposal}
        isStale={true}
        isBusy={false}
        onAccept={vi.fn()}
        onReject={vi.fn()}
      />,
    );

    expect(screen.getByText(/Stale proposal/)).toBeInTheDocument();
    expect(screen.getByText(/Ask the composer to rebase or revise this proposal/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Accept proposal/ })).not.toBeInTheDocument();
  });
});
```

- [x] **Step 2: Run ToolCallCard tests to verify they fail**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/components/chat/ToolCallCard.test.tsx
```

Expected: FAIL because `ToolCallCard.tsx` does not exist.

Observed: FAIL because `ToolCallCard.tsx` did not exist; MessageBubble integration test also failed because raw tool-call rendering did not surface proposals.

- [x] **Step 3: Implement ToolCallCard**

Create `src/elspeth/web/frontend/src/components/chat/ToolCallCard.tsx`:

```typescript
import type { CompositionProposal, ToolCall } from "@/types/api";

interface ToolCallCardProps {
  toolCall: ToolCall;
  proposal: CompositionProposal | null;
  isStale?: boolean;
  isBusy?: boolean;
  onAccept: (proposalId: string) => void;
  onReject: (proposalId: string) => void;
}

export function ToolCallCard({
  toolCall,
  proposal,
  isStale = false,
  isBusy = false,
  onAccept,
  onReject,
}: ToolCallCardProps) {
  if (!proposal) {
    return (
      <div className="tool-call-ribbon">
        <span aria-hidden="true">?</span>
        <span>Looked up: {toolCall.function.name}</span>
      </div>
    );
  }

  const isPending = proposal.status === "pending";
  const isCommitted = proposal.status === "committed";
  const isRejected = proposal.status === "rejected";

  return (
    <article className={`tool-call-card tool-call-card--${proposal.status}`}>
      <header className="tool-call-card-header">
        <strong>
          {isPending
            ? `Proposed: ${proposal.tool_name}`
            : isCommitted
              ? `Applied: ${proposal.tool_name}`
              : `Rejected: ${proposal.tool_name}`}
        </strong>
        {proposal.audit_event_id && (
          <code className="tool-call-audit-id">audit {proposal.audit_event_id.slice(0, 8)}</code>
        )}
      </header>
      <p className="tool-call-summary">{proposal.summary}</p>
      <p className="tool-call-rationale">
        <strong>Why:</strong> {proposal.rationale}
      </p>
      <p className="tool-call-affects">
        <strong>Affects:</strong> {proposal.affects.join(", ")}
      </p>
      <details className="tool-call-details">
        <summary>View arguments (JSON)</summary>
        <pre>{JSON.stringify(proposal.arguments_redacted_json, null, 2)}</pre>
      </details>
      {isStale && (
        <p className="tool-call-stale">
          Stale proposal. Ask the composer to rebase or revise this proposal.
        </p>
      )}
      {isPending && !isStale && (
        <div className="tool-call-actions">
          <button
            type="button"
            onClick={() => onAccept(proposal.id)}
            aria-label={`Accept proposal: ${proposal.summary}`}
            disabled={isBusy || isStale}
          >
            Accept
          </button>
          <button
            type="button"
            onClick={() => onReject(proposal.id)}
            aria-label={`Reject proposal: ${proposal.summary}`}
            disabled={isBusy || isStale}
          >
            Reject
          </button>
        </div>
      )}
    </article>
  );
}
```

- [x] **Step 4: Wire MessageBubble to proposal records**

Change `MessageBubbleProps`:

```typescript
  proposalsByToolCallId?: Map<string, CompositionProposal>;
  staleProposalIds?: string[];
  proposalActionPendingIds?: string[];
  onAcceptProposal?: (proposalId: string) => void;
  onRejectProposal?: (proposalId: string) => void;
```

Replace the existing raw tool call `<ul className="message-tools-list">` rendering with:

```typescript
              <div className="message-tools-list">
                {message.tool_calls.map((tc, i) => (
                  <ToolCallCard
                    key={tc.id ?? i}
                    toolCall={tc}
                    proposal={tc.id ? proposalsByToolCallId?.get(tc.id) ?? null : null}
                    isStale={
                      tc.id
                        ? staleProposalIds?.includes(proposalsByToolCallId?.get(tc.id)?.id ?? "") ?? false
                        : false
                    }
                    isBusy={
                      tc.id
                        ? proposalActionPendingIds?.includes(proposalsByToolCallId?.get(tc.id)?.id ?? "") ?? false
                        : false
                    }
                    onAccept={onAcceptProposal ?? (() => undefined)}
                    onReject={onRejectProposal ?? (() => undefined)}
                  />
                ))}
              </div>
```

- [x] **Step 5: Add CSS**

In `src/elspeth/web/frontend/src/App.css`, add:

```css
.tool-call-ribbon {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-top: 8px;
  padding: 6px 8px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  font-size: 0.875rem;
}

.tool-call-card {
  margin-top: 10px;
  padding: 10px;
  border: 1px solid var(--color-border-strong);
  border-left: 4px solid var(--color-warning);
  background: var(--color-surface-elevated);
  border-radius: 6px;
}

.tool-call-card--committed {
  border-left-color: var(--color-success);
  opacity: 0.88;
}

.tool-call-card--rejected {
  border-left-color: var(--color-danger);
  opacity: 0.78;
}

.tool-call-card-header,
.tool-call-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  justify-content: space-between;
}

.tool-call-summary,
.tool-call-rationale,
.tool-call-affects {
  margin: 8px 0 0;
}

.tool-call-actions button {
  min-height: 36px;
}

.tool-call-stale {
  margin: 8px 0 0;
  color: var(--color-warning-text);
  font-weight: 600;
}
```

- [x] **Step 6: Run frontend tests**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/components/chat/ToolCallCard.test.tsx src/components/chat/MessageBubble.test.tsx
```

Expected: PASS.

Observed: PASS (`ToolCallCard.test.tsx` and `MessageBubble.test.tsx`: `14 passed`; `ChatPanel.test.tsx`: `21 passed`). Additional verification: `npm run typecheck` PASS; `npm run lint` PASS with five pre-existing warnings in unrelated components.

- [x] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ToolCallCard.tsx src/elspeth/web/frontend/src/components/chat/ToolCallCard.test.tsx src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx src/elspeth/web/frontend/src/components/chat/MessageBubble.test.tsx src/elspeth/web/frontend/src/App.css
git commit -m "$(cat <<'EOF'
feat: render composer tool proposal cards

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 9: Pending Overlay in Canvas Views

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx`
- Modify: `src/elspeth/web/frontend/src/App.css`
- Test: `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx`
- Test: `src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx`
- Test: `src/elspeth/web/frontend/src/components/inspector/YamlView.test.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`

- [x] **Step 1: Add failing overlay tests**

Add to `GraphView.test.tsx`:

```typescript
it("renders a pending proposal pill when proposal affects graph", () => {
  useSessionStore.setState({
    compositionState: makeState({
      nodes: [makeNode({ id: "classify", node_type: "transform", plugin: "llm_transform" })],
    }),
    compositionProposals: [
      {
        id: "proposal-1",
        session_id: "session-1",
        tool_call_id: "call-1",
        tool_name: "set_pipeline",
        status: "pending",
        summary: "Replace the pipeline.",
        rationale: "Requested by the current composer turn.",
        affects: ["graph", "validation", "yaml"],
        arguments_redacted_json: {},
        base_state_id: null,
        committed_state_id: null,
        audit_event_id: "event-1",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      },
    ],
  });

  render(<GraphView />);

  expect(screen.getByText("pending #1")).toBeInTheDocument();
});
```

Add equivalent assertions in `SpecView.test.tsx` for `Pending proposal` and `YamlView.test.tsx` for `Pending YAML change`.

Add a `ChatPanel.test.tsx` assertion that a message with `tool_calls[0].id === "call-1"` renders the matching proposal from `compositionProposals[0].tool_call_id === "call-1"` through `MessageBubble`/`ToolCallCard`, and that a stale proposal ID in `staleProposalIds` renders the stale badge. This locks the wiring path instead of only testing `MessageBubble` in isolation.

- [x] **Step 2: Run overlay tests to verify they fail**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/components/inspector/GraphView.test.tsx src/components/inspector/SpecView.test.tsx src/components/inspector/YamlView.test.tsx
```

Expected: FAIL because pending proposal overlays do not render.

Observed: FAIL in GraphView, SpecView, and YamlView overlay assertions; ChatPanel pass-through already passed because Task 8 wired the message bubble props.

- [x] **Step 3: Pass proposals into freeform message bubbles**

In `ChatPanel.tsx`, select proposals and actions:

```typescript
  const compositionProposals = useSessionStore((s) => s.compositionProposals);
  const staleProposalIds = useSessionStore((s) => s.staleProposalIds);
  const proposalActionPendingIds = useSessionStore((s) => s.proposalActionPendingIds);
  const acceptProposal = useSessionStore((s) => s.acceptProposal);
  const rejectProposal = useSessionStore((s) => s.rejectProposal);
  const proposalsByToolCallId = useMemo(
    () => new Map(compositionProposals.map((proposal) => [proposal.tool_call_id, proposal])),
    [compositionProposals],
  );
```

Pass these props to `MessageBubble`:

```typescript
              proposalsByToolCallId={proposalsByToolCallId}
              staleProposalIds={staleProposalIds}
              proposalActionPendingIds={proposalActionPendingIds}
              onAcceptProposal={(proposalId) => void acceptProposal(proposalId)}
              onRejectProposal={(proposalId) => void rejectProposal(proposalId)}
```

- [x] **Step 4: Add GraphView pending pill**

In `GraphView.tsx`, read pending proposals:

```typescript
  const pendingProposalCount = useSessionStore(
    (s) => s.compositionProposals.filter((proposal) => proposal.status === "pending" && proposal.affects.includes("graph")).length,
  );
```

Inside the graph container above `<ReactFlow>`, render:

```typescript
      {pendingProposalCount > 0 && (
        <div
          role="status"
          className="pending-overlay-pill"
          aria-label={`${pendingProposalCount} pending graph proposal${pendingProposalCount === 1 ? "" : "s"}`}
        >
          pending #{pendingProposalCount}
        </div>
      )}
```

- [x] **Step 5: Add SpecView pending rows**

In `SpecView.tsx`, read pending proposals:

```typescript
  const pendingProposals = useSessionStore((s) =>
    s.compositionProposals.filter((proposal) => proposal.status === "pending" && proposal.affects.includes("graph")),
  );
```

Render after validation banners:

```typescript
      {pendingProposals.map((proposal, index) => (
        <div key={proposal.id} className="spec-pending-proposal" role="note">
          <strong>Pending proposal #{index + 1}</strong>
          <span>{proposal.summary}</span>
        </div>
      ))}
```

- [x] **Step 6: Add YamlView pending summary**

In `YamlView.tsx`, read pending proposals:

```typescript
  const pendingYamlProposals = useSessionStore((s) =>
    s.compositionProposals.filter((proposal) => proposal.status === "pending" && proposal.affects.includes("yaml")),
  );
```

Render above the toolbar:

```typescript
      {pendingYamlProposals.length > 0 && (
        <div className="yaml-pending-summary" role="note">
          Pending YAML change: {pendingYamlProposals[0].summary}
        </div>
      )}
```

- [x] **Step 7: Add overlay CSS**

In `App.css`, add:

```css
.pending-overlay-pill {
  position: absolute;
  z-index: 5;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  border: 1px dashed var(--color-warning);
  background: var(--color-surface-elevated);
  color: var(--color-text);
  border-radius: 6px;
}

.spec-pending-proposal,
.yaml-pending-summary {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
  padding: 8px 10px;
  border-left: 4px dashed var(--color-warning);
  background: color-mix(in srgb, var(--color-warning) 12%, var(--color-surface));
}
```

- [x] **Step 8: Run overlay tests**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/components/inspector/GraphView.test.tsx src/components/inspector/SpecView.test.tsx src/components/inspector/YamlView.test.tsx src/components/chat/ChatPanel.test.tsx
```

Expected: PASS.

Observed: PASS (`GraphView.test.tsx`, `SpecView.test.tsx`, `YamlView.test.tsx`, and `ChatPanel.test.tsx`: `49 passed`). Additional verification: `npm run typecheck` PASS; `npm run lint` PASS with five pre-existing warnings.

- [x] **Step 9: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx src/elspeth/web/frontend/src/components/inspector/GraphView.tsx src/elspeth/web/frontend/src/components/inspector/SpecView.tsx src/elspeth/web/frontend/src/components/inspector/YamlView.tsx src/elspeth/web/frontend/src/App.css src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx src/elspeth/web/frontend/src/components/inspector/YamlView.test.tsx
git commit -m "$(cat <<'EOF'
feat: show pending composer proposals on canvas

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

## Task 10: End-to-End Proposal Workflow and Verification

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/composer-proposals.spec.ts`
- Modify: `src/elspeth/web/frontend/tests/e2e/page-objects/composer-page.ts`

- [x] **Step 1: Add the Playwright workflow test**

Create `src/elspeth/web/frontend/tests/e2e/composer-proposals.spec.ts`:

```typescript
import { expect, test } from "@playwright/test";
import { ComposerPage } from "./page-objects/composer-page";

test("explicit approve tool call is visible before commit", async ({ page }) => {
  const composer = new ComposerPage(page);
  await composer.goto();
  await composer.createSession("Proposal workflow");
  await composer.sendMessage("Build a simple csv to json pipeline");

  await expect(page.getByText(/Proposed: set_pipeline/)).toBeVisible();
  await expect(page.getByRole("button", { name: /Accept proposal:/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Reject proposal:/ })).toBeVisible();
  await expect(page.getByText(/pending #/)).toBeVisible();

  await page.getByRole("button", { name: /Accept proposal:/ }).click();

  await expect(page.getByText(/Applied: set_pipeline/)).toBeVisible();
  await expect(page.getByText(/audit /)).toBeVisible();
});
```

This E2E must run against a deterministic composer provider, not a live non-deterministic LLM. Use the existing ChaosLLM/test-provider seam in the E2E harness if present; otherwise extend `tests/e2e/page-objects/composer-page.ts` in this task so the test input deterministically emits a `set_pipeline` tool call. Do not merge this as a real-provider Playwright test.

Observed: implemented a deterministic route-fixture Playwright workflow that exercises session creation, chat send, pending proposal display, Graph pending overlay, accept, committed proposal display, and audit ID display without calling a live LLM. First E2E run failed because proposal cards were hidden behind the collapsed tool-call disclosure; `MessageBubble` now auto-expands proposal-bearing tool calls, and the regression is covered by `MessageBubble.test.tsx`.

- [x] **Step 2: Run focused backend and frontend tests**

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py tests/unit/web/sessions/test_routes.py tests/unit/web/composer/test_proposals.py tests/unit/web/composer/test_service.py -q
cd src/elspeth/web/frontend
npm run test -- src/components/chat/ToolCallCard.test.tsx src/components/chat/MessageBubble.test.tsx src/components/chat/ChatPanel.test.tsx src/components/inspector/GraphView.test.tsx src/components/inspector/SpecView.test.tsx src/components/inspector/YamlView.test.tsx src/stores/sessionStore.test.ts
```

Expected: PASS.

Observed: PASS.

- Backend focused batch: `291 passed`.
- Frontend focused batch: `84 passed`.
- Additional frontend focused smoke: `ToolCallCard.test.tsx` and `MessageBubble.test.tsx`, `14 passed`.
- `npm run typecheck` PASS.

- [x] **Step 3: Run frontend build**

Run:

```bash
cd src/elspeth/web/frontend
npm run build
```

Expected: PASS and refreshed `src/elspeth/web/frontend/dist/index.html`.

Observed: PASS. `npm run build` refreshed ignored `dist/` assets; per Task 0, build artifacts remain uncommitted.

- [x] **Step 4: Run E2E proposal test**

Run:

```bash
cd src/elspeth/web/frontend
npm run test:e2e -- tests/e2e/composer-proposals.spec.ts
```

Expected: PASS.

Observed: PASS after the proposal-card auto-expansion fix. The Playwright web server emitted the known virtualenv path warning from the frontend harness, but the test passed.

- [x] **Step 5: Run policy gates**

Run:

```bash
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: PASS. If `enforce_tier_model.py` reports new intentional imports, add a narrowly scoped entry to the correct file under `config/cicd/enforce_tier_model/` and rerun the command.

Observed: PASS.

- `scripts.check_contracts`: all contract checks passed.
- `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`: no bug-hiding patterns detected.
- `npm run lint`: PASS with the same five pre-existing React Hook warnings in `CatalogDrawer.tsx`, `ChatInput.tsx`, and `GraphView.tsx`.

- [x] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-05-14-composer-ux-tool-proposal-lifecycle.md src/elspeth/web/frontend/tests/e2e/composer-proposals.spec.ts src/elspeth/web/frontend/tests/e2e/page-objects/composer-page.ts src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx src/elspeth/web/frontend/src/components/chat/MessageBubble.test.tsx tests/unit/web/composer/test_service.py tests/unit/web/sessions/test_routes.py
git commit -m "$(cat <<'EOF'
test: cover composer proposal workflow

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Observed: committed with the required co-author trailer.

## Final Verification

Run:

```bash
.venv/bin/pytest tests/unit/web/sessions/test_composer_proposals.py tests/unit/web/sessions/test_routes.py tests/unit/web/composer/test_proposals.py tests/unit/web/composer/test_service.py -q
cd src/elspeth/web/frontend
npm run test
npm run build
```

Expected: all commands PASS.

If backend Python files changed outside the focused test surface, also run:

```bash
.venv/bin/pytest tests/unit/web tests/integration/web -q
```

Expected: PASS or only pre-existing failures documented with exact failing test names.

## Self-Review

Spec coverage:

- Section 6.2 tool call rendering: Tasks 7-9 render pending, committed, rejected, read-only, summary, rationale, affects, args, and audit id.
- Section 9.1 Pending -> Committed -> Audited lifecycle: Tasks 1-6 create real proposal records and immutable proposal events before state mutation.
- Section 9.2 trust mode: Tasks 1, 2, 4, and 5 add durable `trust_mode` and use it to choose explicit approve versus auto-commit.
- Section 16 Phase 0: Tasks 1-6 provide schema, preferences, proposal state, and audit events.
- Section 16 Phase 2: Task 9 adds pending overlays to Graph, Spec, YAML, and Chat.
- Section 16 Phase 3: Task 8 makes tool calls first-class conversation messages.

Future plans:

- Ledger/status strip, density mode replacement for `ExitToFreeformButton`, audit/review explorer, data-flow view, mobile polish, and operational telemetry each need their own executable plan.

In-slice cuts from spec:

- Full graph overlay with per-node dashed outlines, per-edge styling, and click-to-jump is deferred.
- Spec-view struck-through old values and per-affected-node row placement are deferred.
- YAML line-by-line diff highlighting is deferred.
- Dynamic, context-rich rationale is deferred; this slice creates the field and placeholder only.
- Edit and Discuss buttons are deferred to the Phase 4 per-turn/chat-input plan.
- Click-to-copy audit IDs are deferred unless `ToolCallCard` implementation can include it without widening the task.

Placeholder scan:

- No banned placeholder instructions remain.

Type consistency:

- Backend statuses use `pending | committed | rejected`.
- Frontend `ProposalLifecycleStatus` mirrors the backend Literal.
- Event types use `proposal.created | proposal.accepted | proposal.rejected | trust_mode.changed`.
- Trust modes use `explicit_approve | auto_commit`.
- Density defaults use `high | medium | low`.
