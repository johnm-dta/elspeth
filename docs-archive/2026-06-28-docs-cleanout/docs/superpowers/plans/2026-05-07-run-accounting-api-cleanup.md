# Run Accounting API Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misleading web `rows_*` API contract with explicit source-row, emitted-token, terminal-token, routing, and integrity accounting so valid fan-out workflows inspect correctly.

**Architecture:** Keep the sessions database columns as internal cached engine counters for now, but stop exposing them as the public web contract. Add a Landscape-derived run-accounting projection for terminal runs, use it in REST/session/WebSocket payloads, and make status validation depend on token closure and terminal outcome classes rather than source-row conservation.

**Tech Stack:** Python 3.12/3.13, FastAPI, Pydantic v2 strict response models, SQLAlchemy Core, Landscape audit tables, React/TypeScript/Vitest.

---

## Design Boundaries

This is prerelease software. Do not preserve the old `rows_processed`, `rows_succeeded`, `rows_failed`, `rows_routed_success`, `rows_routed_failure`, or `rows_quarantined` wire fields for compatibility. They are misleading because they mix source-row and materialized-token units.

The public API should expose these units instead:

```json
{
  "accounting": {
    "source": {
      "rows_processed": 1
    },
    "tokens": {
      "emitted": 9324,
      "terminal": 9324,
      "succeeded": 9323,
      "failed": 0,
      "structural": 1,
      "pending": 0
    },
    "routing": {
      "routed_success": 0,
      "routed_failure": 0,
      "quarantined": 0,
      "discarded": 0
    },
    "integrity": {
      "closure": "closed",
      "missing_terminal_outcomes": 0,
      "duplicate_terminal_outcomes": 0
    }
  }
}
```

For terminal operator-completion statuses (`completed`, `completed_with_failures`, `empty`), `accounting` is required and must be Landscape-derived. For `pending`, `running`, `cancelled`, and exception-origin `failed`, `accounting` may be `null` when the run did not reach a readable Landscape record.

## File Structure

- `src/elspeth/web/execution/schemas.py`: Owns the new public response/event models and Pydantic validation rules.
- `src/elspeth/web/execution/accounting.py`: New Landscape read model that derives run accounting from `rows`, `tokens`, and `token_outcomes`.
- `src/elspeth/web/execution/service.py`: Maps session records and engine progress events into the new wire payloads.
- `src/elspeth/web/execution/routes.py`: Loads accounting for status/results/diagnostics and stops converting internal validation failures into fake 404s.
- `src/elspeth/web/sessions/schemas.py`: Updates session run-list response shape.
- `src/elspeth/web/sessions/routes.py`: Batch-loads accounting for session run lists.
- `src/elspeth/web/frontend/src/types/index.ts`: Mirrors the new API/event contracts.
- `src/elspeth/web/frontend/src/stores/executionStore.ts`: Tracks renamed progress counters and accounting objects.
- `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`: Displays source-row and token accounting explicitly.
- `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx`: Displays live source/token progress explicitly.
- Tests under `tests/unit/web/execution/`, `tests/integration/web/`, and frontend Vitest files.

## Task 1: Define The New Backend Accounting Schemas

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py`
- Create: `tests/unit/web/execution/test_run_accounting_schemas.py`

- [ ] **Step 1: Write the failing schema tests**

Create `tests/unit/web/execution/test_run_accounting_schemas.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from elspeth.web.execution.schemas import (
    CompletedData,
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    RunResultsResponse,
    RunStatusResponse,
)


def _fanout_accounting() -> RunAccounting:
    return RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(
            emitted=9324,
            terminal=9324,
            succeeded=9323,
            failed=0,
            structural=1,
            pending=0,
        ),
        routing=RunAccountingRouting(
            routed_success=0,
            routed_failure=0,
            quarantined=0,
            discarded=0,
        ),
        integrity=RunAccountingIntegrity(
            closure="closed",
            missing_terminal_outcomes=0,
            duplicate_terminal_outcomes=0,
        ),
    )


def test_run_status_accepts_one_source_row_many_terminal_tokens() -> None:
    response = RunStatusResponse(
        run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        status="completed",
        started_at=datetime(2026, 5, 6, 14, 30, tzinfo=UTC),
        finished_at=datetime(2026, 5, 6, 14, 31, tzinfo=UTC),
        accounting=_fanout_accounting(),
        error=None,
        landscape_run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        discard_summary=None,
    )

    assert response.accounting is not None
    assert response.accounting.source.rows_processed == 1
    assert response.accounting.tokens.succeeded == 9323
    assert response.accounting.tokens.structural == 1


def test_run_results_accepts_one_source_row_many_terminal_tokens() -> None:
    response = RunResultsResponse(
        run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        status="completed",
        accounting=_fanout_accounting(),
        landscape_run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        error=None,
        discard_summary=None,
    )

    assert response.accounting.tokens.emitted == 9324
    assert response.accounting.tokens.terminal == 9324


def test_completed_event_carries_accounting_instead_of_mixed_rows() -> None:
    event = CompletedData(
        status="completed",
        accounting=_fanout_accounting(),
        landscape_run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
    )

    assert event.accounting.tokens.succeeded == 9323


def test_closed_accounting_requires_all_emitted_tokens_terminal() -> None:
    with pytest.raises(ValidationError, match="closed accounting requires pending == 0"):
        RunAccounting(
            source=RunAccountingSource(rows_processed=1),
            tokens=RunAccountingTokens(
                emitted=3,
                terminal=2,
                succeeded=2,
                failed=0,
                structural=0,
                pending=1,
            ),
            routing=RunAccountingRouting(
                routed_success=0,
                routed_failure=0,
                quarantined=0,
                discarded=0,
            ),
            integrity=RunAccountingIntegrity(
                closure="closed",
                missing_terminal_outcomes=1,
                duplicate_terminal_outcomes=0,
            ),
        )


def test_completed_status_requires_closed_accounting() -> None:
    accounting = _fanout_accounting().model_copy(
        update={
            "integrity": RunAccountingIntegrity(
                closure="open",
                missing_terminal_outcomes=1,
                duplicate_terminal_outcomes=0,
            ),
            "tokens": RunAccountingTokens(
                emitted=9324,
                terminal=9323,
                succeeded=9323,
                failed=0,
                structural=0,
                pending=1,
            ),
        }
    )

    with pytest.raises(ValidationError, match="status='completed' requires closed token accounting"):
        RunStatusResponse(
            run_id="run-1",
            status="completed",
            started_at=datetime(2026, 5, 6, 14, 30, tzinfo=UTC),
            finished_at=datetime(2026, 5, 6, 14, 31, tzinfo=UTC),
            accounting=accounting,
            error=None,
            landscape_run_id="run-1",
            discard_summary=None,
        )
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/test_run_accounting_schemas.py
```

Expected: fail during import with `ImportError` for `RunAccounting`.

- [ ] **Step 3: Add accounting models and replace row-count validation**

In `src/elspeth/web/execution/schemas.py`, replace `_validate_row_decomposition` and `_check_status_row_count_invariant` with accounting-aware models and validation:

```python
class RunAccountingSource(_StrictResponse):
    """Source-ingestion counts for a run."""

    rows_processed: int = Field(ge=0)


class RunAccountingTokens(_StrictResponse):
    """Pipeline-token accounting for emitted materialized work."""

    emitted: int = Field(ge=0)
    terminal: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    structural: int = Field(ge=0)
    pending: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_token_balance(self) -> Self:
        if self.terminal != self.succeeded + self.failed + self.structural:
            raise ValueError(
                "tokens.terminal must equal tokens.succeeded + tokens.failed + tokens.structural "
                f"(got terminal={self.terminal}, succeeded={self.succeeded}, "
                f"failed={self.failed}, structural={self.structural})"
            )
        if self.emitted != self.terminal + self.pending:
            raise ValueError(
                "tokens.emitted must equal tokens.terminal + tokens.pending "
                f"(got emitted={self.emitted}, terminal={self.terminal}, pending={self.pending})"
            )
        return self


class RunAccountingRouting(_StrictResponse):
    """Routing/disposition subset counts for terminal tokens."""

    routed_success: int = Field(ge=0)
    routed_failure: int = Field(ge=0)
    quarantined: int = Field(ge=0)
    discarded: int = Field(ge=0)


class RunAccountingIntegrity(_StrictResponse):
    """Closure integrity of the Landscape token ledger."""

    closure: Literal["closed", "open", "unknown"]
    missing_terminal_outcomes: int = Field(ge=0)
    duplicate_terminal_outcomes: int = Field(ge=0)


class RunAccounting(_StrictResponse):
    """Explicit run accounting split by unit of account."""

    source: RunAccountingSource
    tokens: RunAccountingTokens
    routing: RunAccountingRouting
    integrity: RunAccountingIntegrity

    @model_validator(mode="after")
    def _check_integrity_contract(self) -> Self:
        if self.routing.routed_success > self.tokens.succeeded:
            raise ValueError(
                "routing.routed_success must be a subset of tokens.succeeded "
                f"(got routed_success={self.routing.routed_success}, succeeded={self.tokens.succeeded})"
            )
        if self.routing.routed_failure > self.tokens.failed:
            raise ValueError(
                "routing.routed_failure must be a subset of tokens.failed "
                f"(got routed_failure={self.routing.routed_failure}, failed={self.tokens.failed})"
            )
        if self.routing.quarantined > self.tokens.failed:
            raise ValueError(
                "routing.quarantined must be a subset of tokens.failed "
                f"(got quarantined={self.routing.quarantined}, failed={self.tokens.failed})"
            )
        if self.routing.discarded > self.tokens.failed:
            raise ValueError(
                "routing.discarded must be a subset of tokens.failed "
                f"(got discarded={self.routing.discarded}, failed={self.tokens.failed})"
            )
        if self.integrity.closure == "closed":
            if self.tokens.pending != 0:
                raise ValueError(f"closed accounting requires pending == 0, got {self.tokens.pending}")
            if self.integrity.missing_terminal_outcomes != 0:
                raise ValueError(
                    "closed accounting requires missing_terminal_outcomes == 0, "
                    f"got {self.integrity.missing_terminal_outcomes}"
                )
            if self.integrity.duplicate_terminal_outcomes != 0:
                raise ValueError(
                    "closed accounting requires duplicate_terminal_outcomes == 0, "
                    f"got {self.integrity.duplicate_terminal_outcomes}"
                )
        return self


def _check_status_accounting_invariant(status: str, accounting: RunAccounting | None) -> None:
    """Validate status taxonomy against explicit token accounting."""
    if status in {"running", "pending", "cancelled"}:
        return

    if status in OPERATOR_COMPLETION_RUN_STATUS_VALUES and accounting is None:
        raise ValueError(f"status={status!r} requires Landscape-derived accounting")

    if status == "completed":
        assert accounting is not None
        if accounting.integrity.closure != "closed":
            raise ValueError("status='completed' requires closed token accounting")
        if accounting.tokens.succeeded <= 0:
            raise ValueError("status='completed' requires tokens.succeeded > 0")
        if accounting.tokens.failed != 0:
            raise ValueError("status='completed' requires tokens.failed == 0")
        return

    if status == "completed_with_failures":
        assert accounting is not None
        if accounting.integrity.closure != "closed":
            raise ValueError("status='completed_with_failures' requires closed token accounting")
        if accounting.tokens.succeeded <= 0:
            raise ValueError("status='completed_with_failures' requires tokens.succeeded > 0")
        if accounting.tokens.failed <= 0:
            raise ValueError("status='completed_with_failures' requires tokens.failed > 0")
        return

    if status == "failed":
        return

    if status == "empty":
        assert accounting is not None
        if accounting.source.rows_processed != 0:
            raise ValueError(
                "status='empty' requires accounting.source.rows_processed == 0, "
                f"got {accounting.source.rows_processed}"
            )
        if accounting.tokens.emitted != 0:
            raise ValueError(f"status='empty' requires accounting.tokens.emitted == 0, got {accounting.tokens.emitted}")
        return

    raise ValueError(f"Unknown status {status!r}")
```

Update `CompletedData`, `RunStatusResponse`, and `RunResultsResponse` to carry `accounting` instead of `rows_*` fields:

```python
class CompletedData(_StrictResponse):
    status: Literal["completed", "completed_with_failures", "empty"]
    accounting: RunAccounting
    landscape_run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_status_consistency(self) -> Self:
        _check_status_accounting_invariant(self.status, self.accounting)
        return self


class RunStatusResponse(_StrictResponse):
    """REST response for run status queries."""

    run_id: str
    status: SessionRunStatus
    started_at: datetime | None
    finished_at: datetime | None
    accounting: RunAccounting | None = None
    error: str | None
    landscape_run_id: str | None
    discard_summary: DiscardSummary | None = None

    @model_validator(mode="after")
    def _check_status_contract(self) -> Self:
        if self.status in RUN_STATUS_TERMINAL_VALUES:
            _require_terminal_run_fields(
                self.status,
                finished_at=self.finished_at,
                finished_at_required=True,
                error=self.error,
                landscape_run_id=self.landscape_run_id,
            )
        _check_status_accounting_invariant(self.status, self.accounting)
        return self


class RunResultsResponse(_StrictResponse):
    """REST response for terminal run results."""

    run_id: str
    status: TerminalSessionRunStatus
    accounting: RunAccounting
    landscape_run_id: str | None
    error: str | None
    discard_summary: DiscardSummary | None = None

    @model_validator(mode="after")
    def _check_status_contract(self) -> Self:
        _require_terminal_run_fields(
            self.status,
            error=self.error,
            landscape_run_id=self.landscape_run_id,
        )
        _check_status_accounting_invariant(self.status, self.accounting)
        return self
```

- [ ] **Step 4: Run the schema tests and verify they pass**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/test_run_accounting_schemas.py
```

Expected: pass.

- [ ] **Step 5: Commit the schema contract**

```bash
git add src/elspeth/web/execution/schemas.py tests/unit/web/execution/test_run_accounting_schemas.py
git commit -m "refactor(web): define explicit run accounting schema"
```

## Task 2: Derive Run Accounting From Landscape

**Files:**
- Create: `src/elspeth/web/execution/accounting.py`
- Create: `tests/unit/web/execution/test_run_accounting_projection.py`

- [ ] **Step 1: Write the failing Landscape projection tests**

Create `tests/unit/web/execution/test_run_accounting_projection.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import create_engine

from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.schema import metadata, rows_table, token_outcomes_table, tokens_table
from elspeth.web.execution.accounting import load_run_accounting_from_db


def _insert_row(conn, *, run_id: str, row_id: str, row_index: int) -> None:
    conn.execute(
        rows_table.insert().values(
            row_id=row_id,
            run_id=run_id,
            source_node_id="source",
            row_index=row_index,
            source_data_hash=f"hash-{row_index}",
            source_data_ref=None,
            created_at=datetime(2026, 5, 6, tzinfo=UTC),
        )
    )


def _insert_token(conn, *, run_id: str, token_id: str, row_id: str, step: int) -> None:
    conn.execute(
        tokens_table.insert().values(
            token_id=token_id,
            row_id=row_id,
            run_id=run_id,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            branch_name=None,
            step_in_pipeline=step,
            created_at=datetime(2026, 5, 6, tzinfo=UTC),
        )
    )


def _insert_completed_outcome(
    conn,
    *,
    run_id: str,
    token_id: str,
    outcome: TerminalOutcome,
    path: TerminalPath,
    sink_name: str | None = None,
) -> None:
    conn.execute(
        token_outcomes_table.insert().values(
            outcome_id=f"outcome-{token_id}-{path.value}",
            run_id=run_id,
            token_id=token_id,
            outcome=outcome.value,
            path=path.value,
            completed=1,
            recorded_at=datetime(2026, 5, 6, tzinfo=UTC),
            sink_name=sink_name,
            batch_id=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            error_hash=None,
            context_json=None,
            expected_branches_json=None,
        )
    )


def test_one_source_row_many_tokens_is_closed_accounting() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)

    with engine.begin() as conn:
        _insert_row(conn, run_id="run-1", row_id="row-1", row_index=0)
        _insert_token(conn, run_id="run-1", token_id="parent", row_id="row-1", step=0)
        _insert_completed_outcome(
            conn,
            run_id="run-1",
            token_id="parent",
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.EXPAND_PARENT,
        )
        for i in range(3):
            token_id = f"child-{i}"
            _insert_token(conn, run_id="run-1", token_id=token_id, row_id="row-1", step=1)
            _insert_completed_outcome(
                conn,
                run_id="run-1",
                token_id=token_id,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            )

    accounting = load_run_accounting_from_db(engine, ("run-1",))["run-1"]

    assert accounting.source.rows_processed == 1
    assert accounting.tokens.emitted == 4
    assert accounting.tokens.terminal == 4
    assert accounting.tokens.succeeded == 3
    assert accounting.tokens.failed == 0
    assert accounting.tokens.structural == 1
    assert accounting.tokens.pending == 0
    assert accounting.integrity.closure == "closed"


def test_missing_terminal_outcome_marks_accounting_open() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)

    with engine.begin() as conn:
        _insert_row(conn, run_id="run-2", row_id="row-1", row_index=0)
        _insert_token(conn, run_id="run-2", token_id="token-1", row_id="row-1", step=0)

    accounting = load_run_accounting_from_db(engine, ("run-2",))["run-2"]

    assert accounting.tokens.emitted == 1
    assert accounting.tokens.terminal == 0
    assert accounting.tokens.pending == 1
    assert accounting.integrity.closure == "open"
    assert accounting.integrity.missing_terminal_outcomes == 1
```

- [ ] **Step 2: Run the projection tests and verify they fail**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/test_run_accounting_projection.py
```

Expected: fail during import with `ModuleNotFoundError` for `elspeth.web.execution.accounting`.

- [ ] **Step 3: Add the Landscape accounting projection**

Create `src/elspeth/web/execution/accounting.py`:

```python
"""Landscape-derived run accounting for the web execution API."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import Engine, func, select

from elspeth.contracts.audit import DISCARD_SINK_NAME
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, token_outcomes_table, tokens_table
from elspeth.web.config import WebSettings
from elspeth.web.execution.discard_summary import _sqlite_database_file_missing, _unique_run_ids
from elspeth.web.execution.schemas import (
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
)


def load_run_accounting_for_settings(
    settings: WebSettings,
    landscape_run_ids: Iterable[str | None],
) -> dict[str, RunAccounting]:
    """Load run accounting from the configured Landscape database."""
    run_ids = _unique_run_ids(landscape_run_ids)
    if not run_ids:
        return {}

    landscape_url = settings.get_landscape_url()
    if _sqlite_database_file_missing(landscape_url):
        return {}

    with LandscapeDB.from_url(
        landscape_url,
        passphrase=settings.landscape_passphrase,
        create_tables=False,
    ) as db:
        return load_run_accounting_from_db(db.engine, run_ids)


def load_run_accounting_from_db(
    engine: Engine,
    landscape_run_ids: Iterable[str],
) -> dict[str, RunAccounting]:
    """Derive source/token/routing/integrity accounting from Landscape."""
    run_ids = _unique_run_ids(landscape_run_ids)
    if not run_ids:
        return {}

    source_rows = {run_id: 0 for run_id in run_ids}
    emitted_tokens = {run_id: 0 for run_id in run_ids}
    terminal_tokens = {run_id: 0 for run_id in run_ids}
    succeeded_tokens = {run_id: 0 for run_id in run_ids}
    failed_tokens = {run_id: 0 for run_id in run_ids}
    structural_tokens = {run_id: 0 for run_id in run_ids}
    routed_success = {run_id: 0 for run_id in run_ids}
    routed_failure = {run_id: 0 for run_id in run_ids}
    quarantined = {run_id: 0 for run_id in run_ids}
    discarded = {run_id: 0 for run_id in run_ids}
    completed_distinct = {run_id: 0 for run_id in run_ids}
    duplicate_terminal_outcomes = {run_id: 0 for run_id in run_ids}

    with engine.connect() as conn:
        for run_id, count in conn.execute(
            select(rows_table.c.run_id, func.count().label("count"))
            .where(rows_table.c.run_id.in_(run_ids))
            .group_by(rows_table.c.run_id)
        ):
            source_rows[run_id] = int(count)

        for run_id, count in conn.execute(
            select(tokens_table.c.run_id, func.count().label("count"))
            .where(tokens_table.c.run_id.in_(run_ids))
            .group_by(tokens_table.c.run_id)
        ):
            emitted_tokens[run_id] = int(count)

        for run_id, outcome, path, sink_name, count in conn.execute(
            select(
                token_outcomes_table.c.run_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.sink_name,
                func.count().label("count"),
            )
            .where(token_outcomes_table.c.run_id.in_(run_ids))
            .where(token_outcomes_table.c.completed == 1)
            .group_by(
                token_outcomes_table.c.run_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.sink_name,
            )
        ):
            value = int(count)
            terminal_tokens[run_id] += value
            if outcome == TerminalOutcome.SUCCESS.value:
                succeeded_tokens[run_id] += value
            elif outcome == TerminalOutcome.FAILURE.value:
                failed_tokens[run_id] += value
            elif outcome == TerminalOutcome.TRANSIENT.value:
                structural_tokens[run_id] += value

            if outcome == TerminalOutcome.SUCCESS.value and path == TerminalPath.GATE_ROUTED.value:
                routed_success[run_id] += value
            if outcome == TerminalOutcome.FAILURE.value and path == TerminalPath.ON_ERROR_ROUTED.value:
                routed_failure[run_id] += value
            if outcome == TerminalOutcome.FAILURE.value and path == TerminalPath.QUARANTINED_AT_SOURCE.value:
                quarantined[run_id] += value
            if outcome == TerminalOutcome.FAILURE.value and (
                path == TerminalPath.SINK_DISCARDED.value or sink_name == DISCARD_SINK_NAME
            ):
                discarded[run_id] += value

        completed_by_token = (
            select(
                token_outcomes_table.c.run_id.label("run_id"),
                token_outcomes_table.c.token_id.label("token_id"),
                func.count().label("completed_count"),
            )
            .where(token_outcomes_table.c.run_id.in_(run_ids))
            .where(token_outcomes_table.c.completed == 1)
            .group_by(token_outcomes_table.c.run_id, token_outcomes_table.c.token_id)
            .subquery()
        )

        for run_id, count in conn.execute(
            select(completed_by_token.c.run_id, func.count().label("count")).group_by(completed_by_token.c.run_id)
        ):
            completed_distinct[run_id] = int(count)

        for run_id, count in conn.execute(
            select(completed_by_token.c.run_id, func.count().label("count"))
            .where(completed_by_token.c.completed_count > 1)
            .group_by(completed_by_token.c.run_id)
        ):
            duplicate_terminal_outcomes[run_id] = int(count)

    accounting: dict[str, RunAccounting] = {}
    for run_id in run_ids:
        missing = max(0, emitted_tokens[run_id] - completed_distinct[run_id])
        pending = missing
        closure = "closed" if missing == 0 and duplicate_terminal_outcomes[run_id] == 0 else "open"
        accounting[run_id] = RunAccounting(
            source=RunAccountingSource(rows_processed=source_rows[run_id]),
            tokens=RunAccountingTokens(
                emitted=emitted_tokens[run_id],
                terminal=terminal_tokens[run_id],
                succeeded=succeeded_tokens[run_id],
                failed=failed_tokens[run_id],
                structural=structural_tokens[run_id],
                pending=pending,
            ),
            routing=RunAccountingRouting(
                routed_success=routed_success[run_id],
                routed_failure=routed_failure[run_id],
                quarantined=quarantined[run_id],
                discarded=discarded[run_id],
            ),
            integrity=RunAccountingIntegrity(
                closure=closure,
                missing_terminal_outcomes=missing,
                duplicate_terminal_outcomes=duplicate_terminal_outcomes[run_id],
            ),
        )
    return accounting
```

- [ ] **Step 4: Run the projection tests and verify they pass**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/test_run_accounting_projection.py
```

Expected: pass.

- [ ] **Step 5: Commit the accounting projection**

```bash
git add src/elspeth/web/execution/accounting.py tests/unit/web/execution/test_run_accounting_projection.py
git commit -m "feat(web): derive run accounting from Landscape"
```

## Task 3: Wire Accounting Into Status, Results, And Diagnostics Routes

**Files:**
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/execution/routes.py`
- Test: existing execution route/service tests under `tests/unit/web/execution/` and `tests/integration/web/`

- [ ] **Step 1: Add a failing route test for valid fan-out diagnostics**

Find the current route test file with:

```bash
rg -n "get_run_diagnostics|/api/runs/.*/diagnostics|RunStatusResponse" tests/unit tests/integration
```

In the matching execution route test file, add this test. If the project already has a fixture factory for route apps, use that factory and keep the assertions unchanged:

```python
async def test_run_diagnostics_accepts_fanout_accounting(client, session_service, execution_service, monkeypatch):
    run_id = UUID("a2a7354a-5732-475b-a4ac-ed166a9e0f25")
    session_id = UUID("a95eb527-fc07-4169-bb44-9366b0d84d1f")
    user_id = "user-1"

    session_service.add_session(id=session_id, user_id=user_id, auth_provider_type="local")
    session_service.add_run(
        id=run_id,
        session_id=session_id,
        status="completed",
        rows_processed=1,
        rows_succeeded=9323,
        rows_failed=0,
        rows_routed_success=0,
        rows_routed_failure=0,
        rows_quarantined=0,
        landscape_run_id=str(run_id),
    )

    accounting = RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(emitted=9324, terminal=9324, succeeded=9323, failed=0, structural=1, pending=0),
        routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
        integrity=RunAccountingIntegrity(closure="closed", missing_terminal_outcomes=0, duplicate_terminal_outcomes=0),
    )

    monkeypatch.setattr(
        "elspeth.web.execution.routes.load_run_accounting_for_settings",
        lambda settings, run_ids: {str(run_id): accounting},
    )
    monkeypatch.setattr(
        "elspeth.web.execution.routes.load_run_diagnostics_for_settings",
        lambda *args, **kwargs: RunDiagnosticsResponse(
            run_id=str(run_id),
            landscape_run_id=str(run_id),
            run_status="completed",
            summary=RunDiagnosticSummary(
                token_count=9324,
                preview_limit=50,
                preview_truncated=True,
                state_counts={},
                operation_counts={},
                latest_activity_at=None,
            ),
            tokens=[],
            operations=[],
            artifacts=[],
        ),
    )

    response = await client.get(f"/api/runs/{run_id}/diagnostics")

    assert response.status_code == 200
    assert response.json()["summary"]["token_count"] == 9324
```

- [ ] **Step 2: Run the route test and verify it fails**

Run the test path found in Step 1:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/<route-test-file>.py::test_run_diagnostics_accepts_fanout_accounting
```

Expected: fail because route/service status construction does not supply `accounting`.

- [ ] **Step 3: Update service status construction to accept accounting**

In `src/elspeth/web/execution/service.py`, change `get_status` to accept optional accounting supplied by routes:

```python
async def get_status(self, run_id: UUID, *, accounting: RunAccounting | None = None) -> RunStatusResponse:
    """Return current run status. AC #17: delegates to SessionService."""
    run = await self._session_service.get_run(run_id)
    return RunStatusResponse(
        run_id=str(run.id),
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        accounting=accounting,
        error=run.error,
        landscape_run_id=run.landscape_run_id,
    )
```

Add `RunAccounting` to the imports from `elspeth.web.execution.schemas`.

- [ ] **Step 4: Update route status loading and stop fake 404 masking**

In `src/elspeth/web/execution/routes.py`, import:

```python
from pydantic import ValidationError

from elspeth.web.execution.accounting import load_run_accounting_for_settings
```

Add this helper near `_run_not_found_http`:

```python
def _run_integrity_http(exc: ValidationError) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={
            "code": "run_integrity_error",
            "message": "Run status failed internal accounting validation.",
            "validation_errors": exc.errors(include_url=False),
        },
    )
```

In `/api/runs/{run_id}` and `/api/runs/{run_id}/diagnostics`, load accounting after `_verify_run_ownership`:

```python
run_record = await request.app.state.session_service.get_run(run_id)
accounting_by_run_id = await run_sync_in_worker(
    load_run_accounting_for_settings,
    request.app.state.settings,
    (run_record.landscape_run_id,),
)
accounting = accounting_by_run_id.get(run_record.landscape_run_id or "")
try:
    status = await service.get_status(run_id, accounting=accounting)
except ValidationError as exc:
    raise _run_integrity_http(exc) from exc
```

Remove the broad `except ValueError: raise _run_not_found_http()` around `service.get_status()`. `_verify_run_ownership()` remains the only IDOR-safe missing/unauthorized check for these routes.

- [ ] **Step 5: Update `/api/runs/{run_id}/results` construction**

Where `RunResultsResponse` is built, pass `accounting=status.accounting` and fail closed when a terminal result has no accounting:

```python
if status.accounting is None:
    raise HTTPException(
        status_code=500,
        detail={
            "code": "run_integrity_error",
            "message": "Terminal run has no Landscape-derived accounting.",
        },
    )

return RunResultsResponse(
    run_id=status.run_id,
    status=status.status,
    accounting=status.accounting,
    landscape_run_id=status.landscape_run_id,
    error=status.error,
    discard_summary=status.discard_summary,
)
```

- [ ] **Step 6: Run the route test and verify it passes**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/<route-test-file>.py::test_run_diagnostics_accepts_fanout_accounting
```

Expected: pass.

- [ ] **Step 7: Add and run the fake-404 regression**

In the same route test file, add:

```python
async def test_internal_status_validation_error_is_not_fake_404(client, session_service, execution_service, monkeypatch):
    run_id = UUID("11111111-1111-4111-8111-111111111111")
    session_id = UUID("22222222-2222-4222-8222-222222222222")

    session_service.add_session(id=session_id, user_id="user-1", auth_provider_type="local")
    session_service.add_run(
        id=run_id,
        session_id=session_id,
        status="completed",
        rows_processed=1,
        rows_succeeded=1,
        rows_failed=0,
        rows_routed_success=0,
        rows_routed_failure=0,
        rows_quarantined=0,
        landscape_run_id=str(run_id),
    )

    accounting = RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(emitted=1, terminal=0, succeeded=0, failed=0, structural=0, pending=1),
        routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
        integrity=RunAccountingIntegrity(closure="open", missing_terminal_outcomes=1, duplicate_terminal_outcomes=0),
    )
    monkeypatch.setattr(
        "elspeth.web.execution.routes.load_run_accounting_for_settings",
        lambda settings, run_ids: {str(run_id): accounting},
    )

    response = await client.get(f"/api/runs/{run_id}")

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "run_integrity_error"
```

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/<route-test-file>.py::test_internal_status_validation_error_is_not_fake_404
```

Expected: pass after Step 4.

- [ ] **Step 8: Commit route/service integration**

```bash
git add src/elspeth/web/execution/service.py src/elspeth/web/execution/routes.py tests/unit/web/execution/<route-test-file>.py
git commit -m "fix(web): validate run status with explicit accounting"
```

## Task 4: Update Session Run Lists To Use Accounting

**Files:**
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Test: session route tests under `tests/unit/web/sessions/` or `tests/integration/web/`

- [ ] **Step 1: Write the failing session run-list test**

Find the session route test file:

```bash
rg -n "list_runs|get.*runs|/api/sessions/.*/runs" tests/unit tests/integration
```

Add:

```python
async def test_session_run_list_returns_accounting_for_fanout_run(client, session_service, monkeypatch):
    session_id = UUID("a95eb527-fc07-4169-bb44-9366b0d84d1f")
    run_id = UUID("a2a7354a-5732-475b-a4ac-ed166a9e0f25")

    session_service.add_session(id=session_id, user_id="user-1", auth_provider_type="local")
    session_service.add_state(id=UUID("33333333-3333-4333-8333-333333333333"), session_id=session_id, version=1)
    session_service.add_run(
        id=run_id,
        session_id=session_id,
        state_id=UUID("33333333-3333-4333-8333-333333333333"),
        status="completed",
        rows_processed=1,
        rows_succeeded=9323,
        rows_failed=0,
        rows_routed_success=0,
        rows_routed_failure=0,
        rows_quarantined=0,
        landscape_run_id=str(run_id),
    )

    accounting = RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(emitted=9324, terminal=9324, succeeded=9323, failed=0, structural=1, pending=0),
        routing=RunAccountingRouting(routed_success=0, routed_failure=0, quarantined=0, discarded=0),
        integrity=RunAccountingIntegrity(closure="closed", missing_terminal_outcomes=0, duplicate_terminal_outcomes=0),
    )
    monkeypatch.setattr(
        "elspeth.web.sessions.routes.load_run_accounting_for_settings",
        lambda settings, run_ids: {str(run_id): accounting},
    )

    response = await client.get(f"/api/sessions/{session_id}/runs")

    assert response.status_code == 200
    run = response.json()[0]
    assert "rows_processed" not in run
    assert run["accounting"]["source"]["rows_processed"] == 1
    assert run["accounting"]["tokens"]["succeeded"] == 9323
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/sessions/<session-route-test-file>.py::test_session_run_list_returns_accounting_for_fanout_run
```

Expected: fail because `RunResponse` still serializes legacy row fields and no `accounting`.

- [ ] **Step 3: Update session run schema**

In `src/elspeth/web/sessions/schemas.py`, import `RunAccounting` and replace `RunResponse` with:

```python
class RunResponse(_StrictResponse):
    """Response for GET /api/sessions/{id}/runs."""

    id: str
    session_id: str
    status: SessionRunStatus
    accounting: RunAccounting | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    composition_version: int
    discard_summary: DiscardSummary | None = None
```

- [ ] **Step 4: Batch-load accounting in the session route**

In `src/elspeth/web/sessions/routes.py`, import `load_run_accounting_for_settings` and load accounting alongside discard summaries:

```python
terminal_landscape_run_ids = tuple(
    run.landscape_run_id for run in runs if run.landscape_run_id is not None and run.status in TERMINAL_RUN_STATUS_VALUES
)
accounting_by_run_id = {}
if terminal_landscape_run_ids:
    accounting_by_run_id = await run_sync_in_worker(
        load_run_accounting_for_settings,
        request.app.state.settings,
        terminal_landscape_run_ids,
    )
```

When building `RunResponse`, pass:

```python
accounting = None
if run.landscape_run_id is not None:
    accounting = accounting_by_run_id.get(run.landscape_run_id)

responses.append(
    RunResponse(
        id=str(run.id),
        session_id=str(run.session_id),
        status=run.status,
        accounting=accounting,
        error=run.error,
        started_at=run.started_at,
        finished_at=run.finished_at,
        composition_version=version,
        discard_summary=discard_summary,
    )
)
```

- [ ] **Step 5: Run the session test and verify it passes**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/sessions/<session-route-test-file>.py::test_session_run_list_returns_accounting_for_fanout_run
```

Expected: pass.

- [ ] **Step 6: Commit session list accounting**

```bash
git add src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes.py tests/unit/web/sessions/<session-route-test-file>.py
git commit -m "refactor(web): expose accounting in session run lists"
```

## Task 5: Update WebSocket Run Events

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/execution/protocol.py` if its method signature requires adjustment
- Test: execution service/progress tests under `tests/unit/web/execution/`

- [ ] **Step 1: Write the failing event schema test**

Append to `tests/unit/web/execution/test_run_accounting_schemas.py`:

```python
from elspeth.web.execution.schemas import ProgressData


def test_progress_event_uses_explicit_source_and_token_names() -> None:
    progress = ProgressData(
        source_rows_processed=1,
        tokens_succeeded=9323,
        tokens_failed=0,
        tokens_quarantined=0,
        tokens_routed_success=0,
        tokens_routed_failure=0,
    )

    payload = progress.model_dump()

    assert payload["source_rows_processed"] == 1
    assert payload["tokens_succeeded"] == 9323
    assert "rows_processed" not in payload
```

- [ ] **Step 2: Run the event schema test and verify it fails**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/test_run_accounting_schemas.py::test_progress_event_uses_explicit_source_and_token_names
```

Expected: fail because `ProgressData` still requires legacy `rows_*` names.

- [ ] **Step 3: Rename progress and cancellation event fields**

In `src/elspeth/web/execution/schemas.py`, replace `ProgressData` counter fields with:

```python
class ProgressData(_StrictResponse):
    """Payload for progress events with explicit units."""

    source_rows_processed: int = Field(ge=0)
    tokens_succeeded: int = Field(ge=0)
    tokens_failed: int = Field(ge=0)
    tokens_quarantined: int = Field(ge=0)
    tokens_routed_success: int = Field(ge=0)
    tokens_routed_failure: int = Field(ge=0)
```

Replace `CancelledData` with:

```python
class CancelledData(_StrictResponse):
    """Payload for cancelled events with best-known progress counters."""

    status: Literal["cancelled"] = "cancelled"
    source_rows_processed: int = Field(ge=0)
    tokens_succeeded: int = Field(ge=0)
    tokens_failed: int = Field(ge=0)
    tokens_quarantined: int = Field(ge=0)
    tokens_routed_success: int = Field(ge=0)
    tokens_routed_failure: int = Field(ge=0)
```

- [ ] **Step 4: Update event producers**

In `src/elspeth/web/execution/service.py`, update all `ProgressData(...)` and `CancelledData(...)` construction:

```python
ProgressData(
    source_rows_processed=progress.rows_processed,
    tokens_succeeded=progress.rows_succeeded,
    tokens_failed=progress.rows_failed,
    tokens_quarantined=progress.rows_quarantined,
    tokens_routed_success=progress.rows_routed_success,
    tokens_routed_failure=progress.rows_routed_failure,
)
```

For terminal completed events, pass `accounting=result_accounting` instead of row counters. If the completed event is emitted before the route has loaded accounting, load it in `_run_pipeline` immediately after the engine returns `RunResult` and before broadcasting:

```python
accounting_by_run_id = await run_sync_in_worker(
    load_run_accounting_for_settings,
    self._settings,
    (result.landscape_run_id,),
)
accounting = accounting_by_run_id.get(result.landscape_run_id)
if accounting is None:
    raise RuntimeError(f"Landscape accounting missing for completed run {result.landscape_run_id}")
```

Then:

```python
CompletedData(
    status=result.status.value,
    accounting=accounting,
    landscape_run_id=result.landscape_run_id,
)
```

- [ ] **Step 5: Run backend event tests**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution/test_run_accounting_schemas.py
PYTHONPATH=src uv run pytest -q tests/unit/web/execution
```

Expected: pass after updating affected event tests.

- [ ] **Step 6: Commit WebSocket event contract**

```bash
git add src/elspeth/web/execution/schemas.py src/elspeth/web/execution/service.py tests/unit/web/execution
git commit -m "refactor(web): rename run event counters by unit"
```

## Task 6: Update Frontend Types And Store State

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/stores/executionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/executionStore.test.ts`

- [ ] **Step 1: Write the failing store test**

In `src/elspeth/web/frontend/src/stores/executionStore.test.ts`, add:

```typescript
it("applies progress events with explicit source and token counters", () => {
  const event: RunEvent = {
    run_id: "run-1",
    timestamp: "2026-05-06T14:38:00Z",
    event_type: "progress",
    data: {
      source_rows_processed: 1,
      tokens_succeeded: 9323,
      tokens_failed: 0,
      tokens_quarantined: 0,
      tokens_routed_success: 0,
      tokens_routed_failure: 0,
    },
  };

  useExecutionStore.getState().applyRunEventForTest(event);

  expect(useExecutionStore.getState().progress?.source_rows_processed).toBe(1);
  expect(useExecutionStore.getState().progress?.tokens_succeeded).toBe(9323);
});
```

If the store does not expose `applyRunEventForTest`, add this export in the test-only section already used by the file:

```typescript
export const __testing = { applyRunEvent };
```

Then call `__testing.applyRunEvent(...)` from the test instead of adding a store method.

- [ ] **Step 2: Run the store test and verify it fails**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- executionStore.test.ts
```

Expected: fail because frontend types/store still use `rows_*`.

- [ ] **Step 3: Replace frontend run/accounting types**

In `src/elspeth/web/frontend/src/types/index.ts`, replace the `Run` and progress event counter interfaces with:

```typescript
export interface RunAccountingSource {
  rows_processed: number;
}

export interface RunAccountingTokens {
  emitted: number;
  terminal: number;
  succeeded: number;
  failed: number;
  structural: number;
  pending: number;
}

export interface RunAccountingRouting {
  routed_success: number;
  routed_failure: number;
  quarantined: number;
  discarded: number;
}

export interface RunAccountingIntegrity {
  closure: "closed" | "open" | "unknown";
  missing_terminal_outcomes: number;
  duplicate_terminal_outcomes: number;
}

export interface RunAccounting {
  source: RunAccountingSource;
  tokens: RunAccountingTokens;
  routing: RunAccountingRouting;
  integrity: RunAccountingIntegrity;
}

export interface Run {
  id: string;
  session_id: string;
  status: RunStatus;
  accounting: RunAccounting | null;
  error: string | null;
  started_at: string;
  finished_at: string | null;
  composition_version: number;
  discard_summary?: DiscardSummary | null;
}

export interface RunEventProgress {
  source_rows_processed: number;
  tokens_succeeded: number;
  tokens_failed: number;
  tokens_quarantined: number;
  tokens_routed_success: number;
  tokens_routed_failure: number;
}

export interface RunEventCompleted {
  status: "completed" | "completed_with_failures" | "empty";
  accounting: RunAccounting;
  landscape_run_id: string;
}

export interface RunEventCancelled {
  status: "cancelled";
  source_rows_processed: number;
  tokens_succeeded: number;
  tokens_failed: number;
  tokens_quarantined: number;
  tokens_routed_success: number;
  tokens_routed_failure: number;
}

export interface RunProgress {
  source_rows_processed: number;
  tokens_succeeded: number;
  tokens_failed: number;
  tokens_quarantined: number;
  tokens_routed_success: number;
  tokens_routed_failure: number;
  recent_errors: RunEventError[];
  status: RunStatus;
}
```

- [ ] **Step 4: Update the execution store reducer**

In `src/elspeth/web/frontend/src/stores/executionStore.ts`, replace legacy counter extraction with:

```typescript
const sourceRowsProcessed =
  "source_rows_processed" in data
    ? (data as RunEventProgress | RunEventCancelled).source_rows_processed
    : (state.progress?.source_rows_processed ?? 0);
const tokensSucceeded =
  "tokens_succeeded" in data ? (data as RunEventProgress | RunEventCancelled).tokens_succeeded : (state.progress?.tokens_succeeded ?? 0);
const tokensFailed =
  "tokens_failed" in data ? (data as RunEventProgress | RunEventCancelled).tokens_failed : (state.progress?.tokens_failed ?? 0);
const tokensQuarantined =
  "tokens_quarantined" in data
    ? (data as RunEventProgress | RunEventCancelled).tokens_quarantined
    : (state.progress?.tokens_quarantined ?? 0);
const tokensRoutedSuccess =
  "tokens_routed_success" in data
    ? (data as RunEventProgress | RunEventCancelled).tokens_routed_success
    : (state.progress?.tokens_routed_success ?? 0);
const tokensRoutedFailure =
  "tokens_routed_failure" in data
    ? (data as RunEventProgress | RunEventCancelled).tokens_routed_failure
    : (state.progress?.tokens_routed_failure ?? 0);
```

Set `progress` with:

```typescript
progress: {
  source_rows_processed: sourceRowsProcessed,
  tokens_succeeded: tokensSucceeded,
  tokens_failed: tokensFailed,
  tokens_quarantined: tokensQuarantined,
  tokens_routed_success: tokensRoutedSuccess,
  tokens_routed_failure: tokensRoutedFailure,
  recent_errors,
  status,
}
```

When applying completed events to `runs[]`, update the matching run with `accounting: (data as RunEventCompleted).accounting`.

- [ ] **Step 5: Run frontend store tests**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- executionStore.test.ts
```

Expected: pass after updating existing assertions from `rows_*` to the new names.

- [ ] **Step 6: Commit frontend store/types**

```bash
git add src/elspeth/web/frontend/src/types/index.ts src/elspeth/web/frontend/src/stores/executionStore.ts src/elspeth/web/frontend/src/stores/executionStore.test.ts
git commit -m "refactor(frontend): track run accounting counters"
```

## Task 7: Update Frontend Run Displays

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx`
- Test: relevant frontend Vitest files

- [ ] **Step 1: Write the failing RunsView display test**

In `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx`, update the default run factory to use accounting:

```typescript
const accounting = {
  source: { rows_processed: 1 },
  tokens: {
    emitted: 9324,
    terminal: 9324,
    succeeded: 9323,
    failed: 0,
    structural: 1,
    pending: 0,
  },
  routing: {
    routed_success: 0,
    routed_failure: 0,
    quarantined: 0,
    discarded: 0,
  },
  integrity: {
    closure: "closed" as const,
    missing_terminal_outcomes: 0,
    duplicate_terminal_outcomes: 0,
  },
};
```

Add:

```typescript
it("renders source rows and terminal token counts separately", () => {
  render(<RunsView runs={[makeRun({ accounting })]} />);

  expect(screen.getByText("1 source row")).toBeInTheDocument();
  expect(screen.getByText("9,323 succeeded tokens")).toBeInTheDocument();
  expect(screen.getByText("1 structural token")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the RunsView test and verify it fails**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- RunsView.test.tsx
```

Expected: fail because the component still renders `run.rows_processed` and `run.rows_succeeded`.

- [ ] **Step 3: Update RunsView display copy**

In `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`, add helpers near the top:

```typescript
function pluralize(count: number, singular: string, plural: string): string {
  return `${count.toLocaleString()} ${count === 1 ? singular : plural}`;
}

function formatRunAccounting(run: Run): string[] {
  if (!run.accounting) {
    return ["Accounting unavailable"];
  }
  const parts = [
    pluralize(run.accounting.source.rows_processed, "source row", "source rows"),
    pluralize(run.accounting.tokens.succeeded, "succeeded token", "succeeded tokens"),
  ];
  if (run.accounting.tokens.failed > 0) {
    parts.push(pluralize(run.accounting.tokens.failed, "failed token", "failed tokens"));
  }
  if (run.accounting.tokens.structural > 0) {
    parts.push(pluralize(run.accounting.tokens.structural, "structural token", "structural tokens"));
  }
  if (run.accounting.tokens.pending > 0) {
    parts.push(pluralize(run.accounting.tokens.pending, "pending token", "pending tokens"));
  }
  return parts;
}
```

Replace the run-card row-count display with:

```tsx
{formatRunAccounting(run).map((part) => (
  <span key={part}>{part}</span>
))}
```

- [ ] **Step 4: Update ProgressView display copy**

In `src/elspeth/web/frontend/src/components/execution/ProgressView.tsx`, replace `progress.rows_processed`, `progress.rows_succeeded`, and related field reads with:

```tsx
{progress.source_rows_processed.toLocaleString()}
{progress.tokens_succeeded.toLocaleString()}
{progress.tokens_failed.toLocaleString()}
{progress.tokens_quarantined.toLocaleString()}
```

Use labels:

```tsx
<span>Source rows</span>
<span>Succeeded tokens</span>
<span>Failed tokens</span>
<span>Quarantined tokens</span>
```

- [ ] **Step 5: Run frontend component tests**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- RunsView.test.tsx executionStore.test.ts ProgressView
```

Expected: pass after updating existing assertions and fixture objects.

- [ ] **Step 6: Commit frontend display cleanup**

```bash
git add src/elspeth/web/frontend/src/components/inspector/RunsView.tsx src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx src/elspeth/web/frontend/src/components/execution/ProgressView.tsx
git commit -m "refactor(frontend): display source and token accounting"
```

## Task 8: Verification And Staging Build

**Files:**
- Modify only files touched by previous tasks if verification finds failures.

- [ ] **Step 1: Run targeted backend tests**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web/execution tests/unit/web/sessions
```

Expected: pass.

- [ ] **Step 2: Run targeted frontend tests**

Run:

```bash
cd src/elspeth/web/frontend
npm run test
```

Expected: pass.

- [ ] **Step 3: Run frontend build**

Run:

```bash
cd src/elspeth/web/frontend
npm run build
```

Expected: build completes and refreshes `src/elspeth/web/frontend/dist/`.

- [ ] **Step 4: Run broader Python validation**

Run:

```bash
PYTHONPATH=src uv run pytest -q tests/unit/web tests/integration/web
```

Expected: pass. If an existing unrelated test-harness hang appears, record the exact test and rerun the targeted slice that excludes only the hung fixture path.

- [ ] **Step 5: Live-check staging after deploy/restart**

If backend Python files changed, restart `elspeth-web.service` using the approved host path for this project. Then run:

```bash
curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
```

Expected:

```json
{"status":"ok"}
```

Then inspect the known fan-out run from the browser or API and verify:

```text
/api/runs/a2a7354a-5732-475b-a4ac-ed166a9e0f25 returns 200
/api/runs/a2a7354a-5732-475b-a4ac-ed166a9e0f25/diagnostics returns 200
Run card shows 1 source row and 9,323 succeeded tokens
Diagnostics panel loads token_count around 9,324
```

- [ ] **Step 6: Commit verification fixes**

If verification required code changes:

```bash
git add src/elspeth tests
git commit -m "test(web): verify run accounting API cleanup"
```

If no code changes were needed after Task 7, do not create an empty commit.

## Self-Review

- Spec coverage: The plan covers explicit source/token/routing/integrity accounting, token-closure validation, fan-out inspection, fake-404 masking, session run lists, WebSocket events, frontend types, and frontend displays.
- Placeholder scan: No placeholder markers, deferred implementation markers, or unspecified validation instructions remain.
- Type consistency: Backend `RunAccounting*` names match frontend `RunAccounting*` names. Backend progress fields match frontend event fields: `source_rows_processed`, `tokens_succeeded`, `tokens_failed`, `tokens_quarantined`, `tokens_routed_success`, `tokens_routed_failure`.
- Scope check: This is one coherent web execution accounting contract change. It spans backend and frontend, but the slices are not independent enough to split because every surface exposes the same API model.
