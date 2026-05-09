# Composer Progress Persistence — Phase 1B: Compose-Turn Primitive and Audit Semantics

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this schedule task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Traceability:** Task numbers are preserved from the original Phase 1 plan so review findings can cite the same task IDs across Schedule 1A/1B/1C. Do not renumber tasks inside a schedule.

**Goal:** Add the dormant compose-turn persistence primitive after Schedule 1A has made the schema and current writers safe.

**Risk controlled:** atomic compose-turn persistence, assistant/tool transcript consistency, audit primacy, unwind behavior, and async cancellation/idempotency before any production route depends on the primitive.

**Architecture:** This schedule may introduce `_persist_payload.py`, sessions telemetry, `persist_compose_turn`, and `persist_compose_turn_async`. It must not introduce PostgreSQL/testcontainer CI infrastructure, compose-loop integration, route-visible behavior, redaction primitives, or frontend work.

**Review focus:** no partial writes, exact tool-call ID matching, stale-current-state rejection, cancellation/retry semantics, failure privacy, and INV-AUDIT-AHEAD proof that does not depend on Docker.

---

## Schedule 1B Scope

**Included tasks:** Task 5, Task 6, Task 11, Task 12, Task 13, and Task 15.

**Explicit exclusions:** Schedule 1A schema/current-writer work, Task 16, Task 17, Task 19, Task 20, and all later composer/frontend phases.

**Must land after Schedule 1A and before Schedule 1C.** Schedule 1C assumes this schedule's primitive, audit, and cancellation contracts are already reviewed and merged.

---

## Schedule 1B Preflight: Contract Gate

- [ ] **Step 1: Confirm Schedule 1A is merged**

Verify the branch contains the Schedule 1A schema/current-writer compatibility changes. Do not re-implement Schedule 1A work in this PR.

- [ ] **Step 2: Define cancellation/idempotency before coding**

Record the chosen retry-after-cancel behavior in this PR before
`persist_compose_turn_async` is implemented. The shielded worker
bridge can continue after caller cancellation, so the contract must
be explicit and testable. The contract is documented and tested in
**Task 11 Step 3d** (Q-F2 fix); the chosen contract is "commit-wins":
once the worker is dispatched, callers MUST NOT retry on
``CancelledError``. See Step 3d for the full rationale, the binding
caller contract for Phase 3, and the
``test_persist_compose_turn_async_caller_cancellation_commits_anyway``
regression that pins the behaviour.

- [ ] **Step 3: Define transcript validation before coding**

Before any insert, `persist_compose_turn` must prove assistant
`tool_calls` and redacted tool rows have the same unique tool-call ID
set. Missing, extra, mismatched, and duplicate IDs must fail before
database writes. The guard, the typed exception
(``ToolCallIDMismatchError``), and the four named regression tests
are implemented in **Task 11 Step 3c** (Q-F1 fix); the helper is
``_validate_tool_call_id_set_equality`` and is called pre-lock,
pre-transaction.

---

## Task 5: Telemetry counters module + wire into `SessionServiceImpl` constructor

**Why this task includes the constructor wiring.** Tasks 7–13 reference
`self._telemetry` and `self._log` on `SessionServiceImpl`. The current
constructor (`service.py:265`) takes only `(engine, data_dir=None)` —
neither attribute exists. Extending the constructor independently of
the telemetry module would create a circular dependency (constructor
needs `_SessionsTelemetry` type; telemetry module exists only after this
task ships); deferring the extension to Task 7 would mean every
intervening task references attributes that don't yet exist. Bundling
both concerns in one atomic task is the only consistent shape.

The production wiring at `app.py:391` is updated in the same commit,
per CLAUDE.md no-legacy / single-commit-hard-cut policy: every caller
that constructs `SessionServiceImpl` is updated atomically with the
signature change.

**Current call-site reality.** As of the post-1A worktree snapshot,
`rg -n "SessionServiceImpl\\(" src tests -g '*.py'` finds 17
`SessionServiceImpl(` call sites across `src` and `tests` (the count
grew from 14 in the original Phase 1 plan-review pass because Schedule
1A added new persist-compose-turn unit tests and additional regression
fixtures), including `tests/unit/web/blobs/test_routes.py` outside the
sessions test tree.
Task 5 is not complete until the implementer runs
`rg -n "SessionServiceImpl\\(" src tests -g '*.py'`, updates every
existing construction site, and pastes the before/after inventory into
the PR body. Narrowly updating `app.py` plus the new constructor test
false-greens this task.

**Files:**
- Create: `src/elspeth/web/sessions/telemetry.py`
- Create: `tests/unit/web/sessions/test_telemetry.py`
- Modify: `src/elspeth/web/sessions/service.py` (extend `__init__`)
- Modify: `src/elspeth/web/app.py` (update production wiring at line 391)
- Modify: every current `SessionServiceImpl(...)` construction site found by `rg -n "SessionServiceImpl\\(" src tests -g '*.py'`, including the existing sessions tests and `tests/unit/web/blobs/test_routes.py`.
- Test: `tests/unit/web/sessions/test_service_construction.py` (create — verifies the new constructor signature)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/sessions/test_telemetry.py`:

```python
"""Tests for the named OTel counters introduced in spec §1.4 / §5.7.4."""
from __future__ import annotations

from elspeth.web.sessions.telemetry import (
    _FakeCounter,
    build_sessions_telemetry,
    observed_value,
)

# Spec §1.4 NFR table — production OTel metric strings for Phase 1.
# Verified end-to-end by ``test_production_meter_registers_named_metrics``
# below.
#
# Phase 1 ships ONLY the four audit-primacy counters that
# persist_compose_turn writes. The remaining four spec-§1.4 counters
# (``summarizer_errors_total``, ``unknown_response_key_total``,
# ``tool_call_cap_exceeded_total``, ``audit_grade_view_total``) are
# Phase 2/3 territory:
#
#   - summarizer_errors_total + unknown_response_key_total → Phase 2
#     (redaction primitives produce them).
#   - tool_call_cap_exceeded_total + audit_grade_view_total → Phase 3
#     (compose-loop cap and audit-grade view emit them).
#
# Adding them to Phase 1 would ship eight metric names of which four
# never increment — operationally indistinguishable from broken
# counters. Each Phase introduces its own counters when the code
# paths that emit them ship. Closes synthesised review finding M14
# / SA-4 (phase scope leak).
EXPECTED_METRIC_NAMES = {
    "composer.audit.tool_row_tier1_violation_total",
    "composer.audit.state_rolled_back_during_persist_total",
    "composer.audit.tool_row_persist_failed_during_unwind_total",
    "composer.audit.tool_row_integrity_violation_total",
}


def test_telemetry_field_names_match_spec_exactly():
    """Use ``set ==`` (not ``issubset``) so an accidental rename — say
    ``tier1_violation_total`` losing its ``tool_row_`` prefix — fails
    the test rather than passing under ``issubset``. Closes synthesised
    review finding L10 / Q-F-13."""
    telem = build_sessions_telemetry()
    expected_fields = {
        "tool_row_tier1_violation_total",
        "state_rolled_back_during_persist_total",
        "tool_row_persist_failed_during_unwind_total",
        "tool_row_integrity_violation_total",
    }
    actual = set(telem.__dataclass_fields__)
    assert actual == expected_fields, (
        f"field-name mismatch — added: {actual - expected_fields}; "
        f"removed: {expected_fields - actual}"
    )


def test_counter_increments_visible_via_observed_value_helper():
    """Test path: build_sessions_telemetry() with no meter returns
    fake counters; the ``observed_value`` helper extracts cumulative
    sum after type-narrowing to ``_FakeCounter``. Production code
    never reads ``observed_value`` — it only writes via ``add()`` —
    so the helper makes the test-only inspection explicit at the
    call site."""
    telem = build_sessions_telemetry()
    starting = observed_value(telem.tool_row_tier1_violation_total)
    telem.tool_row_tier1_violation_total.add(1)
    assert observed_value(telem.tool_row_tier1_violation_total) == starting + 1


def test_counter_records_attributes_dict():
    """Real OTel ``Counter.add`` accepts ``attributes`` as the second
    positional/keyword argument. Production code at composer/service.py
    and routes.py uses this for structured emission (e.g.,
    ``add(1, {"outcome": "failure"})``). The fake must mirror the
    signature so tests with attributed metrics do not raise
    ``TypeError`` against a fake-narrow ``add(amount)`` signature.
    Closes synthesised review finding H6."""
    telem = build_sessions_telemetry()
    telem.tool_row_tier1_violation_total.add(
        1, {"reason": "commit_failure", "session_id": "s_test"}
    )
    fake = telem.tool_row_tier1_violation_total
    assert isinstance(fake, _FakeCounter)
    assert fake.calls == [
        (1, {"reason": "commit_failure", "session_id": "s_test"}, None),
    ]


def test_production_meter_registers_named_metrics():
    """Closes synthesised review finding F-10 / L7. Verifies that the
    four Phase-1 ``meter.create_counter(...)`` strings in
    ``build_sessions_telemetry`` match spec §1.4 exactly. Without this
    test, a typo (e.g. ``tool_row_tier1_violations_total`` with a
    spurious ``s``) would pass the field-name check (which inspects
    Python attribute names, not OTel metric names) and silently break
    production observability."""

    class _RecordingMeter:
        """Captures the names passed to ``create_counter`` so the
        test can assert them as a set."""

        def __init__(self) -> None:
            self.registered: dict[str, _FakeCounter] = {}

        def create_counter(self, name: str) -> _FakeCounter:
            counter = _FakeCounter()
            self.registered[name] = counter
            return counter

    meter = _RecordingMeter()
    build_sessions_telemetry(meter=meter)
    assert set(meter.registered.keys()) == EXPECTED_METRIC_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_telemetry.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the telemetry module**

Create `src/elspeth/web/sessions/telemetry.py`:

```python
"""OTel counters for SessionServiceImpl and the compose loop.

Production code uses the real OTel meter from
``opentelemetry.metrics.get_meter``; tests use
``build_sessions_telemetry()`` with no meter, which returns
``_FakeCounter`` instances. The ``_Counter`` and ``_Meter`` Protocols
match the OTel API exactly (``add(amount, attributes=None,
context=None)`` and ``create_counter(name, ...)``), so production
wiring type-checks without ``# type: ignore`` and the real meter
satisfies the structural contract.

The fake counter records every ``add`` call as ``(amount,
attributes, context)`` tuples. Tests inspect via the ``observed_value(counter)``
helper (cumulative sum) or directly through the ``calls`` attribute
after type-narrowing with ``isinstance(counter, _FakeCounter)``.
``observed_value`` is intentionally NOT on the ``_Counter`` Protocol
— production OTel counters do not expose observation, and adding it
would force a structural lie.

**Ownership-vs-metric namespace.** The container type is named
``_SessionsTelemetry`` and the module lives under
``web/sessions/telemetry.py`` because ``SessionServiceImpl`` owns
the persistence counters. The OTel metric strings remain
``composer.audit.*`` because operators consume them as part of the
composer-progress surface, but metric naming does not justify an
import from ``web/sessions/service.py`` up into ``web/composer``.
Composer code may import the sessions-owned container or receive it
from app wiring; sessions code must not import composer-owned modules.
Phase 2 (redaction counters) and Phase 3 (compose-loop and
audit-grade counters) may extend this container only if ownership
still belongs to the sessions persistence surface; otherwise those
phases add composer-owned telemetry separately.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from opentelemetry.context import Context


_AttributeValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | Sequence[str]
    | Sequence[bool]
    | Sequence[int]
    | Sequence[float]
)
_Attributes: TypeAlias = Mapping[str, _AttributeValue]


class _Counter(Protocol):
    """Subset of ``opentelemetry.metrics.Counter`` that production
    code uses.

    The real OTel signature is ``add(amount, attributes=None,
    context=None)``. Keep this Protocol as broad as the SDK surface that
    callers may legally use: ``amount`` may be ``int`` or ``float``;
    attributes may include every OTel scalar/sequence value type; and
    ``context`` is accepted even though Phase 1 callers omit it. This
    avoids a fake-narrow structural type that passes local tests but
    rejects a real Counter-compatible call shape.
    """

    def add(
        self,
        amount: int | float,
        attributes: _Attributes | None = None,
        context: Context | None = None,
    ) -> None: ...


class _Meter(Protocol):
    """Subset of ``opentelemetry.metrics.Meter`` that
    ``build_sessions_telemetry`` uses for production wiring."""

    def create_counter(self, name: str) -> _Counter: ...


class _FakeCounter:
    """Test-only counter that records every ``add`` call.

    Implements the ``_Counter`` Protocol structurally and adds the
    test inspection surface (``calls`` and the ``observed_value``
    helper). Production code MUST NOT depend on this class — it is
    re-exported only so the telemetry test module and the
    audit-failure-primacy tests can construct it directly.
    """

    def __init__(self) -> None:
        self.calls: list[
            tuple[int | float, dict[str, _AttributeValue] | None, Context | None]
        ] = []

    def add(
        self,
        amount: int | float,
        attributes: _Attributes | None = None,
        context: Context | None = None,
    ) -> None:
        # Defensive copy of the attributes mapping so later mutation
        # of the caller's dict cannot rewrite recorded history.
        recorded_attrs = dict(attributes) if attributes is not None else None
        self.calls.append((amount, recorded_attrs, context))


def observed_value(counter: _Counter) -> int | float:
    """Return the cumulative ``add`` total for a fake counter.

    Test-only helper. Raises ``TypeError`` if ``counter`` is not a
    ``_FakeCounter`` — production OTel counters do not expose
    observation, so a misuse (test running against a
    ``build_sessions_telemetry`` that was wired with a real meter)
    fails loudly rather than producing a confusing attribute error.
    """
    if not isinstance(counter, _FakeCounter):
        raise TypeError(
            f"observed_value: expected _FakeCounter, got "
            f"{type(counter).__name__}. Tests must call "
            f"build_sessions_telemetry() without a meter argument so "
            f"the container is populated with fake counters."
        )
    return sum(amount for amount, _attrs, _context in counter.calls)


@dataclass(frozen=True, slots=True)
class _SessionsTelemetry:
    """Container for the named counters introduced by composer progress
    persistence. All counters default to fakes so tests can assert without
    wiring the real OTel SDK; production wiring replaces them at startup.
    """

    # Phase 1 audit-primacy counters only. Phase 2 (redaction) adds
    # ``summarizer_errors_total`` and ``unknown_response_key_total``;
    # Phase 3 (compose loop + audit-grade view) adds
    # ``tool_call_cap_exceeded_total`` and ``audit_grade_view_total``.
    # Each phase extends this dataclass when its emitter ships,
    # which keeps "registered" and "exercised" in lock-step
    # operationally.
    tool_row_tier1_violation_total: _Counter
    state_rolled_back_during_persist_total: _Counter
    tool_row_persist_failed_during_unwind_total: _Counter
    tool_row_integrity_violation_total: _Counter


def build_sessions_telemetry(
    *, meter: _Meter | None = None
) -> _SessionsTelemetry:
    """Build a telemetry container.

    With ``meter=None`` (the default) returns ``_FakeCounter``
    instances; tests use this path. Production callers pass an OTel
    ``Meter`` (typed structurally as ``_Meter`` so we don't import
    ``opentelemetry.metrics`` here unnecessarily — the structural
    Protocol is satisfied by the real meter at runtime).
    """

    if meter is None:
        return _SessionsTelemetry(
            tool_row_tier1_violation_total=_FakeCounter(),
            state_rolled_back_during_persist_total=_FakeCounter(),
            tool_row_persist_failed_during_unwind_total=_FakeCounter(),
            tool_row_integrity_violation_total=_FakeCounter(),
        )

    # Production wiring against the real OTel meter. The ``_Meter``
    # Protocol satisfies mypy without ``# type: ignore`` decorations
    # — the real OTel ``Meter.create_counter`` matches the structural
    # contract, and the returned counter satisfies ``_Counter``.
    return _SessionsTelemetry(
        tool_row_tier1_violation_total=meter.create_counter(
            "composer.audit.tool_row_tier1_violation_total"
        ),
        state_rolled_back_during_persist_total=meter.create_counter(
            "composer.audit.state_rolled_back_during_persist_total"
        ),
        tool_row_persist_failed_during_unwind_total=meter.create_counter(
            "composer.audit.tool_row_persist_failed_during_unwind_total"
        ),
        tool_row_integrity_violation_total=meter.create_counter(
            "composer.audit.tool_row_integrity_violation_total"
        ),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_telemetry.py -v
```
Expected: PASS.

- [ ] **Step 5: Write the failing constructor-signature test**

Create `tests/unit/web/sessions/test_service_construction.py`:

```python
"""Tests pinning the SessionServiceImpl constructor signature.

Phase 1 extends the constructor with required ``telemetry`` and ``log``
arguments so that ``persist_compose_turn`` and the audit-failure
disposition path can emit OTel counters and (only when the audit
system itself fails) log diagnostics. The signature is part of the
service's public contract; this test prevents accidental drift.
"""
from __future__ import annotations

from sqlalchemy.pool import StaticPool

import pytest
import structlog

from elspeth.web.sessions.telemetry import build_sessions_telemetry
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl


@pytest.fixture
def engine():
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


def test_constructor_accepts_telemetry_and_log(engine, tmp_path):
    telem = build_sessions_telemetry()
    log = structlog.get_logger("test")
    service = SessionServiceImpl(
        engine,
        data_dir=tmp_path,
        telemetry=telem,
        log=log,
    )
    assert service._telemetry is telem
    assert service._log is log


def test_constructor_rejects_missing_telemetry(engine, tmp_path):
    with pytest.raises(TypeError, match="telemetry"):
        SessionServiceImpl(engine, data_dir=tmp_path)  # type: ignore[call-arg]


def test_constructor_rejects_missing_log(engine, tmp_path):
    telem = build_sessions_telemetry()
    with pytest.raises(TypeError, match="log"):
        SessionServiceImpl(engine, data_dir=tmp_path, telemetry=telem)  # type: ignore[call-arg]
```

- [ ] **Step 6: Run the constructor test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_service_construction.py -v
```
Expected: FAIL — constructor does not yet accept `telemetry` or `log`.

- [ ] **Step 7: Extend `SessionServiceImpl.__init__`**

In `src/elspeth/web/sessions/service.py`, replace the existing
constructor (around line 265) with:

```python
from elspeth.web.sessions.telemetry import _SessionsTelemetry  # add to imports

class SessionServiceImpl:
    """Concrete session service backed by SQLAlchemy Core.

    All public methods are async. Database I/O runs through _run_sync() in a
    bounded worker thread so the async event loop is never blocked.
    """

    def __init__(
        self,
        engine: Engine,
        data_dir: Path | None = None,
        *,
        telemetry: _SessionsTelemetry,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        self._engine = engine
        self._data_dir = data_dir
        self._telemetry = telemetry
        self._log = log
```

Notes:
- `telemetry` and `log` are keyword-only (after the `*,` separator) so
  positional callers cannot accidentally swap them.
- `data_dir` keeps its default for backwards compatibility within this
  PR's call-site sweep — production passes it positionally (per
  app.py:391 today). After the sweep, every call site supplies
  `telemetry` and `log` explicitly.
- The `_SessionsTelemetry` import uses the underscore-prefixed type:
  the type itself is module-private but importing it here is
  intentional because `service.py` needs to type-check against the
  concrete container shape, not a generic `object`.
- **Test-only access convention.** The Phase 1 unit and integration
  tests access ``service._engine`` directly to set up fixture state
  (insert sessions, run schema-introspection queries) that the
  public ``SessionServiceImpl`` API does not expose. The underscore
  is conventional Python for "internal," not "do not access from
  tests" — pytest-style fixtures and tests have legitimate reasons
  to reach past the public surface. If a future Phase introduces a
  third consumer that ALSO needs raw engine access (a maintenance
  CLI, an admin tool), promote the field to a typed
  ``engine_for_test_or_admin: Engine`` property. Until then,
  ``service._engine`` is the documented test pattern and need not
  be re-exposed. Closes synthesised review finding SA-12 / M12.

You will also need to add `import structlog` near the top of the
module if it is not already imported. **As of the Phase 1 plan-review
verification, `structlog` is NOT imported by
`src/elspeth/web/sessions/service.py` — this Step MUST add the import**
(alongside the standard-library imports at the top of the file). Confirm
via `grep -n '^import structlog' src/elspeth/web/sessions/service.py`
before editing.

- [ ] **Step 8: Update production wiring at `app.py:391`**

Replace the existing line at `src/elspeth/web/app.py:391`:

```python
# BEFORE
session_service = SessionServiceImpl(session_engine, data_dir=settings.data_dir)
```

with:

```python
# AFTER
from opentelemetry import metrics  # add to top-of-module imports — NOT currently imported by app.py per plan-review verification; required for the metrics.get_meter(...) call below
from elspeth.web.sessions.telemetry import build_sessions_telemetry  # add to imports — also new

# ...

session_service = SessionServiceImpl(
    session_engine,
    data_dir=settings.data_dir,
    telemetry=build_sessions_telemetry(meter=metrics.get_meter("elspeth.web.composer")),
    log=structlog.get_logger("sessions"),
)
```

Notes:
- `metrics.get_meter(...)` returns a real OTel meter when the runtime
  has a `MeterProvider` configured, and a no-op meter otherwise.
  Production code never uses fake counters — fakes are a test affordance
  only (per `_FakeCounter` docstring in telemetry.py).
- `structlog.get_logger("sessions")` matches the established
  pattern in `app.py` for naming subsystem loggers.

- [ ] **Step 9: Sweep every existing constructor call site**

Run the inventory command before editing:

```bash
rg -n "SessionServiceImpl\\(" src tests -g '*.py'
```

Expected before this task on the post-1A snapshot: 17 call sites
(verified by `rg -n "SessionServiceImpl\\(" src tests -g '*.py'` from
the worktree root), spread across:

- `src/elspeth/web/app.py` — 1 site (line 391, production wiring)
- `tests/unit/web/blobs/test_routes.py` — 2 sites
- `tests/unit/web/sessions/test_datetime_timezone.py` — 1 site
- `tests/unit/web/sessions/test_fork.py` — 3 sites
- `tests/unit/web/sessions/test_persist_compose_turn.py` — 3 sites
  (Schedule 1A's persist-compose-turn unit tests; not present in the
  pre-1A snapshot)
- `tests/unit/web/sessions/test_routes.py` — 4 sites
- `tests/unit/web/sessions/test_service.py` — 3 sites

Update every construction to pass `telemetry=build_sessions_telemetry()`
and `log=structlog.get_logger(...)` in tests, or the production OTel
meter/logger in `app.py`. Re-run the inventory and paste both counts
into the PR description. Any remaining `SessionServiceImpl(engine)`
shape is a blocker.

- [ ] **Step 10: Run all affected tests + mypy**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_telemetry.py tests/unit/web/sessions/test_service_construction.py tests/unit/web/sessions/test_service.py tests/unit/web/sessions/test_routes.py tests/unit/web/sessions/test_fork.py tests/unit/web/sessions/test_datetime_timezone.py tests/unit/web/blobs/test_routes.py -v
.venv/bin/python -m mypy src/elspeth/web/sessions/service.py src/elspeth/web/app.py src/elspeth/web/sessions/telemetry.py
```
Expected: all tests PASS; mypy clean.

- [ ] **Step 11: Commit**

```bash
git add src/elspeth/web/sessions/telemetry.py \
        tests/unit/web/sessions/test_telemetry.py \
        tests/unit/web/sessions/test_service_construction.py \
        tests/unit/web/sessions/test_service.py \
        tests/unit/web/sessions/test_routes.py \
        tests/unit/web/sessions/test_fork.py \
        tests/unit/web/sessions/test_datetime_timezone.py \
        tests/unit/web/blobs/test_routes.py \
        src/elspeth/web/sessions/service.py \
        src/elspeth/web/app.py
git commit -m "feat(sessions): add SessionsTelemetry counter container and wire into SessionServiceImpl (composer-progress-persistence phase 1)

- Adds web/sessions/telemetry.py with the _SessionsTelemetry container
  and build_sessions_telemetry() factory.
- Extends SessionServiceImpl.__init__ to accept required telemetry and
  log keyword arguments; production wiring at app.py:391 updated in the
  same commit (no-legacy single-cut policy).
- Production wiring uses opentelemetry.metrics.get_meter(...) for the
  real OTel meter and structlog.get_logger('sessions') for the log.
"
```

---
## Task 6: Persist-payload dataclasses

**Files:**
- Create: `src/elspeth/web/sessions/_persist_payload.py`
- Create: `tests/unit/web/sessions/test_persist_payload.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/web/sessions/test_persist_payload.py`:

```python
"""Tests for the persist-payload dataclasses (spec §5.2.1)."""
from __future__ import annotations

import pytest

from elspeth.web.sessions._persist_payload import (
    _AuditOutcome,
    _RedactedToolRow,
    _StatePayload,
    _ToolOutcome,
)


def test_audit_outcome_success_shape():
    """Success: assistant_id set, no unwind failure."""
    outcome = _AuditOutcome(
        assistant_id="abc",
        unwind_audit_failed=False,
    )
    assert outcome.assistant_id == "abc"
    assert outcome.unwind_audit_failed is False


def test_audit_outcome_unwind_failure_shape():
    """Tool failed AND audit unwind failed: assistant_id=None,
    flag set. Caller will raise the captured plugin crash."""
    outcome = _AuditOutcome(
        assistant_id=None,
        unwind_audit_failed=True,
    )
    assert outcome.assistant_id is None
    assert outcome.unwind_audit_failed is True


def test_audit_outcome_rejects_ambiguous_shape():
    """assistant_id=set + unwind_audit_failed=True is contradictory:
    the unwind path runs only when the tool already failed, so no
    assistant message could have been produced. The dataclass rejects
    the combination at construction time."""
    with pytest.raises(ValueError, match="incompatible"):
        _AuditOutcome(
            assistant_id="abc",          # produced by a successful path
            unwind_audit_failed=True,    # claimed by an unwind path
        )


def test_audit_outcome_no_tier1_violation_field():
    """Sanity: the tier1_violation flag-return path was deleted in
    Stage 4 of the plan revision; persist_compose_turn now raises
    AuditIntegrityError directly. Closes finding H1."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(_AuditOutcome)}
    assert fields == {"assistant_id", "unwind_audit_failed"}


def test_redacted_tool_row_with_state_advance():
    from elspeth.web.sessions.protocol import CompositionStateData

    row = _RedactedToolRow(
        tool_call_id="tc_1",
        content='{"ok": true}',
        composition_state_payload=_StatePayload(
            # B1 (Phase 1 plan-review synthesis): no ``version=``.
            # Version is allocated inside _session_write_lock by
            # ``_insert_composition_state`` (Task 10), not supplied by
            # the caller. Removing the field at the dataclass level
            # forecloses the dual-allocator race that fabricated
            # Tier-1 violations on contention loss.
            data=CompositionStateData(
                source={"kind": "test"},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={},
                is_valid=True,
                validation_errors=None,
            ),
            derived_from_state_id="prev_state_id",
        ),
    )
    assert row.composition_state_payload is not None
    assert row.composition_state_payload.data.is_valid is True
    assert row.composition_state_payload.derived_from_state_id == "prev_state_id"


def test_state_payload_has_no_version_field():
    """B1 (Phase 1 plan-review synthesis): ``_StatePayload`` MUST NOT
    carry a caller-supplied ``version`` field. Version is allocated
    inside _session_write_lock by
    ``_insert_composition_state`` via
    ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
    WHERE session_id = :sid`` (Task 10).

    Pre-B1 the field existed; the compose loop in Phase 3 read
    ``MAX(version)`` outside the lock and dispatched a precomputed
    version into the locked helper. Two concurrent allocators could
    both compute ``MAX+1``; the loser's INSERT triggered
    ``uq_composition_state_version`` → ``IntegrityError`` → the
    locked-path handler incremented ``tool_row_integrity_violation_total``
    on what was structurally a contention loss, fabricating a Tier-1
    audit-integrity violation. SLO threshold is 0; the alert fires on a
    non-event.

    The fix is structural — version simply isn't a payload field — so
    no new caller can reintroduce the race by accident. This test pins
    the contract so a refactor that re-adds the field fails fast."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(_StatePayload)}
    assert "version" not in fields, (
        "B1 regression: _StatePayload must not carry a caller-supplied "
        "version field — version is allocated by "
        "_insert_composition_state under _session_write_lock"
    )
    assert fields == {"data", "derived_from_state_id"}, (
        f"unexpected _StatePayload fields: {fields}"
    )


def test_tool_outcome_state_unchanged_when_pre_eq_post():
    outcome = _ToolOutcome(
        call=type("FakeCall", (), {"id": "x", "function": type("F", (), {"name": "n"})()})(),
        response={"ok": True},
        error_class=None,
        error_message=None,
        pre_version=3,
        post_version=3,
    )
    assert outcome.post_version == outcome.pre_version


def test_tool_outcome_freezes_dict_response():
    """``_ToolOutcome.__post_init__`` must call ``freeze_fields`` on
    ``call`` and ``response`` so that a dict response cannot be mutated
    through the dataclass reference. Without the guard, ``frozen=True``
    is a lie about deep immutability — the attribute cannot be
    reassigned but the dict it points to remains fully mutable.
    Closes synthesised review finding H5."""
    from types import MappingProxyType

    response_dict = {"ok": True, "nested": {"k": "v"}}
    outcome = _ToolOutcome(
        call={"id": "tc_x", "function": {"name": "set_source"}},
        response=response_dict,
        error_class=None,
        error_message=None,
        pre_version=1,
        post_version=2,
    )

    # Both call and response must be deeply frozen — the outer mapping
    # is a MappingProxyType (rejects __setitem__) and nested dicts are
    # also frozen.
    assert isinstance(outcome.response, MappingProxyType)
    assert isinstance(outcome.call, MappingProxyType)
    with pytest.raises(TypeError):
        outcome.response["ok"] = False  # type: ignore[index]
    # Mutation via the original reference must not be visible through
    # the dataclass reference (deep_freeze copies inputs to immutable
    # equivalents; the original dict and the proxy are distinct).
    response_dict["ok"] = False
    assert outcome.response["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_payload.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the dataclasses**

Create `src/elspeth/web/sessions/_persist_payload.py`:

```python
"""Dataclasses passed across the async/sync boundary in
``SessionServiceImpl.persist_compose_turn`` (spec §5.2.1).

These types have no async behaviour; they are pure data containers that
the compose loop populates in async land and then hands to the sync
worker via ``_run_sync``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.sessions.protocol import CompositionStateData


@dataclass(frozen=True, slots=True)
class _StatePayload:
    """Snapshot of a CompositionState ready for insertion.

    Composes the existing :class:`CompositionStateData` input DTO (which
    carries the per-column state contents and is already
    ``freeze_fields``-protected by its own ``__post_init__``) with a
    ``derived_from_state_id`` that records lineage from the pre-call
    state.

    **B1 (Phase 1 plan-review synthesis): no ``version`` field.** Earlier
    drafts of this dataclass carried a caller-supplied ``version: int``.
    That contract was unsafe: in Phase 3 the compose loop reads
    ``MAX(version)`` outside the session write lock and then
    dispatches into the ``_session_write_lock``-protected ``_insert_composition_state``
    helper. Two concurrent allocators can both compute ``MAX+1`` before
    either acquires the lock; the loser's INSERT then hits
    ``uq_composition_state_version`` and the locked path's
    ``IntegrityError`` handler classifies it as a Tier-1 audit-integrity
    violation — fabricating a Tier-1 violation from a benign contention
    loss. SLO threshold for ``tool_row_integrity_violation_total`` is 0,
    so the alert fires on a non-event. Under ELSPETH's auditability
    standard this is evidence-tampering-class harm: the audit trail
    asserts a violation that did not occur.

    The fix is structural: the version is no longer a payload field at
    all. ``_insert_composition_state`` allocates it under the held
    session write lock via ``SELECT COALESCE(MAX(version), 0) + 1 FROM
    composition_states WHERE session_id = :sid`` (see Task 10). With
    version off the payload, the dual-allocator race becomes
    structurally impossible: every caller must be inside the lock context to invoke
    the helper, and the SELECT-MAX-then-INSERT sequence is atomic
    against every other writer for that session.

    The contract is fixed in Phase 1 even though the call shape only
    manifests in Phase 3 — making the helper impossible to misuse from
    Phase 3 onward is cheaper than catching the misuse later.

    Why this shape rather than a single JSON blob:

    The ``composition_states`` table has eight content columns
    (``source/nodes/edges/outputs/metadata_/is_valid/validation_errors/derived_from_state_id``).
    The plan's earlier ``payload_json: str`` design was a hallucination —
    no ``payload`` column exists, and the existing
    ``save_composition_state`` insert at ``service.py:1005-1020``
    (function body starts at 948) writes
    each column individually via a method-local ``_enveloped(...)`` helper
    and ``deep_thaw(...)`` patterns. Task 10 extracts that rule to the
    shared ``_enveloped_state_column(...)`` helper. ``_StatePayload``
    mirrors that real schema by reusing :class:`CompositionStateData` rather than
    duplicating its fields and freeze-guard machinery.

    ``derived_from_state_id`` is ``str | None`` rather than ``str``
    because the existing inline inserts at ``service.py:1005-1020``
    (initial state) currently set it to ``None``. The compose-loop
    caller in Phase 3 will always supply a non-None value (every
    tool-call-driven state advance has a predecessor).

    Note on freeze-fields. ``derived_from_state_id`` is a scalar (or
    ``None``); ``data`` is a frozen ``CompositionStateData`` with its own
    ``freeze_fields`` discipline. No ``__post_init__`` is required on
    ``_StatePayload`` itself — ``frozen=True`` is sufficient because
    every remaining field is either scalar or an already-frozen
    dataclass. (Removing ``version`` did not change this analysis;
    ``version: int`` was scalar too.)
    """

    data: CompositionStateData
    derived_from_state_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    """Result of one tool call within a compose turn.

    The ``call`` and ``response`` fields are typed ``Any`` because the
    compose loop populates them with framework-specific objects (LiteLLM
    ToolCall, Pydantic response models, plain dicts, etc.) that this
    module deliberately does not couple to. At runtime these values are
    typically dicts or ``Mapping`` types, so the ``frozen=True``
    declaration alone is a lie about immutability — the dataclass
    attribute cannot be reassigned, but the dict it points to remains
    fully mutable through the reference.

    CLAUDE.md's ``freeze_fields`` contract is unconditional for frozen
    dataclasses with container/Any fields: ``__post_init__`` must call
    ``freeze_fields`` on every such field. ``deep_freeze`` (which
    ``freeze_fields`` invokes per field) is identity-preserving for
    values that are already frozen, so the cost of running it on
    already-immutable inputs (e.g. an integer-only ``call``, which
    won't happen in practice but is contractually possible) is zero.
    """

    call: Any                       # ToolCall — typed in protocol module
    response: Any                   # tool response object or None on error
    error_class: str | None
    error_message: str | None
    pre_version: int
    post_version: int

    def __post_init__(self) -> None:
        freeze_fields(self, "call", "response")


@dataclass(frozen=True, slots=True)
class _RedactedToolRow:
    """One persisted tool row, with redactions already applied."""

    tool_call_id: str
    content: str                                       # JSON-serialised redacted response
    composition_state_payload: _StatePayload | None    # set iff state advanced


@dataclass(frozen=True, slots=True)
class _AuditOutcome:
    """Disposition returned by SessionServiceImpl.persist_compose_turn (§5.2.2).

    Two outcome shapes:

    - **Success.** ``assistant_id`` is set, ``unwind_audit_failed=False``.
      Caller continues with the new assistant message id.
    - **Tool failed AND audit unwind failed.** ``assistant_id=None``,
      ``unwind_audit_failed=True``. Caller raises the captured plugin
      crash; the audit failure is recorded by ``persist_compose_turn``
      via counter increment + ``slog.warning`` (permitted under
      CLAUDE.md primacy because the audit system itself failed).

    There is NO tier-1-violation outcome shape. When the audit
    database fails AND no plugin crash is in flight,
    ``persist_compose_turn`` raises
    :class:`elspeth.contracts.errors.AuditIntegrityError` directly,
    chained from the underlying ``OperationalError`` via ``raise ...
    from audit_exc``. The exception is registered in
    ``TIER_1_ERRORS`` (via the ``@tier_1_error`` decoration on
    ``AuditIntegrityError``) so ``except Exception:`` blocks cannot
    silently swallow it. The caller has no opportunity to ignore the
    failure — this is the Tier-1 crash doctrine ("Bad data in the
    audit trail = crash immediately") encoded structurally rather
    than asked nicely.

    Why ``unwind_audit_failed`` stays a flag-return (not a raise):
    when a tool plugin has crashed in flight, the caller already has
    a captured plugin-crash exception to raise. Surfacing a separate
    audit exception would mask the original tool failure. The flag
    tells the caller "your raise should ALSO record this audit
    failure," and the counter + slog inside ``persist_compose_turn``
    have already done so.

    Closes synthesised review finding H1 (audit primacy via
    return-flag instead of raised exception violates Tier-1 doctrine).
    """

    assistant_id: str | None
    unwind_audit_failed: bool

    def __post_init__(self) -> None:
        # Success and unwind-failure are the only two valid shapes.
        # Reject any combination that would make the outcome ambiguous.
        if self.assistant_id is not None and self.unwind_audit_failed:
            raise ValueError(
                "_AuditOutcome: unwind_audit_failed=True is incompatible with "
                "assistant_id being set; the unwind path cannot have produced "
                "an assistant id"
            )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_payload.py -v
```
Expected: PASS.

- [ ] **Step 5: Refactor `_insert_composition_state` to accept `payload: _StatePayload`**

**Why this refactor lands in Schedule 1B and not in Schedule 1A.**
Schedule 1A introduced ``_insert_composition_state`` at
``service.py:534`` with signature ``(conn, *, session_id,
state: CompositionStateData, derived_from_state_id, provenance,
created_at=None, state_id=None)`` — i.e. ``state`` and
``derived_from_state_id`` as separate keyword arguments. Schedule 1B
introduces ``_StatePayload``, which by construction (Steps 1-4 above)
bundles exactly those two fields (``data: CompositionStateData`` and
``derived_from_state_id: str | None``) into one frozen dataclass.

Keeping the helper signature unchanged would force every Schedule 1B
caller (Tasks 11, 12, 13, 15) to unpack the payload at the call site
— ``service._insert_composition_state(conn, session_id=...,
state=row.composition_state_payload.data,
derived_from_state_id=row.composition_state_payload.derived_from_state_id,
...)``. That repeated unpacking is a structural smell that contradicts
the B1 rationale: the payload object is the unit of state-advance,
and a helper that takes the payload directly is the only signature
that prevents future callers from passing inconsistent
``(data, derived_from_state_id)`` pairs by mistake. The refactor
coheres with the same plan-review synthesis (B1) that removed
``version`` from ``_StatePayload``: both fixes reshape the helper
contract to make misuse structurally impossible rather than asking
callers to be careful.

The refactor lands in 1B (not 1A) because ``_StatePayload`` is a 1B
deliverable — Schedule 1A could not have used the new shape, and
landing the refactor in 1A would have required defining the dataclass
out-of-order. Now that Step 1-4 above has just defined ``_StatePayload``
in this PR, the helper can be migrated to take it as a unit in the
same commit.

**Refactor target.** Replace the current signature at
``src/elspeth/web/sessions/service.py:534`` with:

```python
from elspeth.web.sessions._persist_payload import _StatePayload  # add to imports

def _insert_composition_state(
    self,
    conn: Connection,
    *,
    session_id: str,
    payload: _StatePayload,
    provenance: str,
    created_at: datetime | None = None,
    state_id: str | None = None,
) -> str:
    """Single-row insert into composition_states with per-session
    version allocation under _session_write_lock.

    PRECONDITION: caller MUST be inside
    ``with self._session_write_lock(conn, session_id):`` in the same
    transaction. (See full precondition contract in the existing
    docstring — reproduced unchanged across the refactor; only the
    parameter shape changes.)
    """
    # Extract the bundled fields once so the rest of the body can refer
    # to ``state`` and ``derived_from_state_id`` exactly as it did
    # pre-refactor. This avoids touching the version-allocation
    # arithmetic, the per-column INSERT, or the IntegrityError handler
    # — all of which Schedule 1A reviewed and merged.
    state = payload.data
    derived_from_state_id = payload.derived_from_state_id
    # ... existing body unchanged ...
```

**Caller sweep.** Run, then update every site:

```bash
rg -n "_insert_composition_state\\(" src tests -g '*.py'
```

Expected before this step (post-1A snapshot, verified by `grep -rn
'_insert_composition_state(' src/ tests/`): 9 sites — 1 producer
(``service.py:534`` definition), 1 internal caller
(``service.py:1871``, in the ``_save_composition_states_for_revert``
unwind path or equivalent — verify shape and update kwargs), and 7
test sites in ``tests/unit/web/sessions/test_persist_compose_turn.py``
(lines 388, 432, 466, 507, 519, 540, 561 — re-confirm via the rg
output above; the file was added by Schedule 1A).

For every test site, replace:

```python
# BEFORE — Schedule 1A signature
service._insert_composition_state(
    conn,
    session_id=...,
    state=CompositionStateData(...),
    derived_from_state_id=...,
    provenance=...,
)
```

with:

```python
# AFTER — 1B signature
service._insert_composition_state(
    conn,
    session_id=...,
    payload=_StatePayload(
        data=CompositionStateData(...),
        derived_from_state_id=...,
    ),
    provenance=...,
)
```

For the in-service caller at ``service.py:1871``, mirror the same
shape — read the surrounding context to check whether the caller
already has a ``_StatePayload`` in hand (e.g. from a tool row) or
needs to construct one from local ``state``/``derived_from_state_id``
locals.

No-legacy policy applies: the old ``state=``/``derived_from_state_id=``
parameter shape is removed from the helper in this same commit. There
is no migration period — every site is updated atomically per CLAUDE.md
"No Legacy Code Policy" / single-cut hard-policy.

- [ ] **Step 6: Run all `_insert_composition_state` tests + mypy**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v
.venv/bin/python -m mypy src/elspeth/web/sessions/service.py src/elspeth/web/sessions/_persist_payload.py
```
Expected: every existing 1A persist-compose-turn test PASSES (the
behaviour is unchanged; only the parameter shape moved); mypy clean.
If any test was relying on the old kwarg names being available, it
must be updated in this same commit, not later — failing to do so is
a no-legacy violation.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/sessions/_persist_payload.py \
        tests/unit/web/sessions/test_persist_payload.py \
        src/elspeth/web/sessions/service.py \
        tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add persist-payload dataclasses + refactor _insert_composition_state to take payload (composer-progress-persistence phase 1B)

- Adds web/sessions/_persist_payload.py with _StatePayload, _ToolOutcome,
  _RedactedToolRow, _AuditOutcome (B1 fix: no caller-supplied version).
- Refactors _insert_composition_state to take payload: _StatePayload
  instead of (state, derived_from_state_id) separately. Coheres with B1:
  the payload object is the unit of state-advance and a helper that
  takes it as a unit prevents future callers from passing inconsistent
  (data, derived_from_state_id) pairs.
- Updates the in-service caller and all Schedule-1A persist-compose-turn
  tests in the same commit (no-legacy single-cut policy).
"
```

---
## Task 11: `persist_compose_turn` happy path

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_persist_compose_turn_happy_path(service):
    from elspeth.web.sessions._persist_payload import (
        _RedactedToolRow, _StatePayload,
    )
    from elspeth.web.sessions.protocol import CompositionStateData
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s6")

    outcome = service.persist_compose_turn(
        session_id="s6",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_1", "function": {"name": "set_source"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                tool_call_id="tc_1",
                content='{"ok": true}',
                # B1 (Phase 1 plan-review synthesis): no ``version=``.
                # ``_insert_composition_state`` allocates it under the
                # held session write lock; the assertion below pins the
                # allocated value to 1 (first state in this session).
                composition_state_payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # On the success path, _AuditOutcome carries the new
    # assistant_id and unwind_audit_failed=False. The old
    # tier1_violation field was removed in Stage 4 of the plan
    # revision (Tier-1 failures now raise AuditIntegrityError
    # directly — see Task 13).
    assert outcome.unwind_audit_failed is False
    assert outcome.assistant_id is not None

    with service._engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT role, sequence_no, tool_call_id "
            "FROM chat_messages WHERE session_id='s6' ORDER BY sequence_no"
        )).fetchall()
        assert [r.role for r in rows] == ["assistant", "tool"]
        assert rows[0].sequence_no == 1
        assert rows[1].sequence_no == 2
        assert rows[1].tool_call_id == "tc_1"

        states = conn.execute(text(
            "SELECT version, provenance FROM composition_states WHERE session_id='s6'"
        )).fetchall()
        assert len(states) == 1
        assert states[0].version == 1
        assert states[0].provenance == "tool_call"


def test_persist_compose_turn_zero_tool_rows(service):
    """W10a (Phase 1 plan-review synthesis): a turn with
    ``redacted_tool_rows=()`` and ``redacted_assistant_tool_calls=()``
    is a valid and reachable shape — the assistant produced text but
    chose not to call any tools. Spec §5.2 explicitly allows this. The
    primitive MUST commit cleanly: the assistant row is persisted, no
    tool rows are inserted, and no ``composition_states`` rows are
    created (because the empty tool-row tuple has no
    ``composition_state_payload`` to write).

    The zero-row case is not exercised by ``happy_path`` (which always
    includes one ``_RedactedToolRow``), so without this regression the
    next caller migrating an assistant-only call site (Phase 3) would
    discover an off-by-one or empty-tuple bug at integration time
    rather than at the primitive's own unit boundary.
    """
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_zero")
    outcome = service.persist_compose_turn(
        session_id="s_zero",
        assistant_content="text only",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False
    with service._engine.begin() as conn:
        roles = [
            r.role for r in conn.execute(text(
                "SELECT role FROM chat_messages WHERE session_id='s_zero'"
            )).fetchall()
        ]
        assert roles == ["assistant"]
        states = conn.execute(text(
            "SELECT id FROM composition_states WHERE session_id='s_zero'"
        )).fetchall()
        assert states == []


def test_persist_compose_turn_persists_raw_content(service):
    """B2 (Phase 1 plan-review synthesis): ``persist_compose_turn`` must
    plumb the optional ``raw_content`` argument to the assistant row
    verbatim. ``raw_content`` is the audit-attribution column that
    captures the original LLM output BEFORE preflight redaction
    rewrote ``content``. Routes ``src/elspeth/web/sessions/routes.py``
    lines 2151 and 2601 (formerly 1749/2152 in earlier plan revisions,
    and 1542/1945 pre-rev-4) already pass
    ``raw_content=result.raw_assistant_content`` to
    ``add_message``; Phase 3 migrates those call sites to
    ``persist_compose_turn``, so the primitive must accept and persist
    the column today (Phase 1) — not later (Phase 3, which is
    explicitly "loop only, no new primitives").

    Pre-B2 ``persist_compose_turn`` hardcoded ``raw_content=None`` at
    the assistant-row insert; this test would have failed by reading
    ``None`` from ``chat_messages.raw_content`` instead of the supplied
    string. Tool rows still get ``raw_content=None`` regardless —
    redaction-attribution applies only to LLM-authored content.
    """
    from elspeth.web.sessions._persist_payload import _RedactedToolRow

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s6_raw")

    outcome = service.persist_compose_turn(
        session_id="s6_raw",
        assistant_content="ok (redacted)",
        raw_content="original LLM output before preflight redaction",
        redacted_assistant_tool_calls=({"id": "tc_1", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(tool_call_id="tc_1", content="{}", composition_state_payload=None),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False

    with service._engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT role, content, raw_content FROM chat_messages "
            "WHERE session_id='s6_raw' ORDER BY sequence_no"
        )).fetchall()
        # Assistant row carries both visible content (post-redaction)
        # and raw_content (pre-redaction); tool row has raw_content=None.
        assert rows[0].role == "assistant"
        assert rows[0].content == "ok (redacted)"
        assert rows[0].raw_content == "original LLM output before preflight redaction"
        assert rows[1].role == "tool"
        assert rows[1].raw_content is None


def test_persist_compose_turn_rejects_cross_session_parent_state(service):
    """B5 (Phase 1 plan-review synthesis): when ``parent_composition_state_id``
    is supplied and points to a state that belongs to a DIFFERENT session,
    the call MUST raise ``RuntimeError`` with the precise diagnostic
    produced by ``_assert_state_in_session`` — not a generic FK error.

    The composite FK ``fk_chat_messages_composition_state_session`` would
    eventually catch the mismatch at INSERT time, but the ELSPETH
    offensive-programming policy requires a named pre-check at the
    service boundary so the diagnostic identifies the caller, the state,
    and the session mismatch (CLAUDE.md "Defensive Programming:
    Forbidden" — "Proactively detect invalid states and throw meaningful
    exceptions").

    This test would have failed pre-B5 by either crashing with an opaque
    ``IntegrityError`` (if the FK fired) or by silently inserting a row
    that the FK validated only via column equality, producing an
    audit-trail that lies about which session authored the row.
    """
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    # Set up two sessions; insert a composition_state into session A.
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_A")
        _make_session(conn, session_id="s_B")
        with service._session_write_lock(conn, "s_A"):
            state_a_id = service._insert_composition_state(
                conn,
                session_id="s_A",
                # B1: no ``version=`` — helper allocates under the lock.
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    # Now try to persist a turn on session B that references session A's state.
    # The guard MUST fire BEFORE the FK does and produce the precise
    # _assert_state_in_session diagnostic.
    with pytest.raises(
        RuntimeError,
        match=r"persist_compose_turn: composition_state_id=.*belongs to session "
              r"'s_A', not 's_B'.*cross-session reference is a contract violation",
    ):
        service.persist_compose_turn(
            session_id="s_B",
            assistant_content="should fail",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=state_a_id,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    # Post-condition: no chat_messages rows were inserted on s_B.
    # The guard fires INSIDE the transaction, so the engine.begin() context
    # rolls back any partial work; verify the row count is zero.
    with service._engine.begin() as conn:
        b_count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_B'"
        )).scalar()
        assert b_count == 0, (
            f"persist_compose_turn rolled back incorrectly; s_B has "
            f"{b_count} chat rows after a guard-rejected call"
        )


def test_persist_compose_turn_accepts_valid_same_session_parent_state(service):
    """B5 happy path: when ``parent_composition_state_id`` references a
    state that belongs to the SAME session, the guard passes silently and
    the assistant row is correctly stamped with that
    ``composition_state_id``. Closes a coverage gap noted in the quality
    reviewer's W-3 (no test exercised valid non-None parent state)."""
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_C")
        with service._session_write_lock(conn, "s_C"):
            state_c_id = service._insert_composition_state(
                conn,
                session_id="s_C",
                # B1: no ``version=`` — helper allocates under the lock.
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    outcome = service.persist_compose_turn(
        session_id="s_C",
        assistant_content="ok",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=state_c_id,
        expected_current_state_id=state_c_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    assert outcome.unwind_audit_failed is False
    assert outcome.assistant_id is not None

    with service._engine.begin() as conn:
        assistant_row = conn.execute(text(
            "SELECT composition_state_id FROM chat_messages "
            "WHERE session_id='s_C' AND role='assistant'"
        )).fetchone()
        assert assistant_row is not None
        assert assistant_row.composition_state_id == state_c_id


def test_persist_compose_turn_rejects_stale_expected_current_state(service):
    """A compose turn may not persist if the session's current state
    changed while the LLM call was in flight.

    The session-write lock serializes the DB mutation, but the intent
    check is what prevents a compose based on state A from becoming the
    newest state after a concurrent revert/save already made state B
    current. Closes the compose-vs-revert race identified in plan review.
    """
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.protocol import StaleComposeStateError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_stale")
        with service._session_write_lock(conn, "s_stale"):
            stale_state_id = service._insert_composition_state(
                conn,
                session_id="s_stale",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )
            current_state_id = service._insert_composition_state(
                conn,
                session_id="s_stale",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=stale_state_id,
                ),
                provenance="session_seed",
            )

    with pytest.raises(
        StaleComposeStateError,
        match=r"current composition state changed.*expected=.*actual=",
    ):
        service.persist_compose_turn(
            session_id="s_stale",
            assistant_content="stale",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=stale_state_id,
            expected_current_state_id=stale_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    with service._engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT role FROM chat_messages WHERE session_id='s_stale'"
        )).fetchall()
        latest = conn.execute(text(
            "SELECT id FROM composition_states WHERE session_id='s_stale' "
            "ORDER BY version DESC LIMIT 1"
        )).scalar_one()
    assert rows == []
    assert latest == current_state_id


def test_persist_compose_turn_accepts_matching_expected_current_state(service):
    from elspeth.web.sessions._persist_payload import _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_current_ok")
        with service._session_write_lock(conn, "s_current_ok"):
            current_state_id = service._insert_composition_state(
                conn,
                session_id="s_current_ok",
                payload=_StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
                provenance="session_seed",
            )

    outcome = service.persist_compose_turn(
        session_id="s_current_ok",
        assistant_content="ok",
        redacted_assistant_tool_calls=(),
        redacted_tool_rows=(),
        parent_composition_state_id=None,
        expected_current_state_id=current_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False


@pytest.mark.asyncio
async def test_persist_compose_turn_refuses_async_invocation(service):
    """Calling ``persist_compose_turn`` directly from a coroutine
    must raise RuntimeError. Production callers (Phase 3 compose
    loop) use ``await service.persist_compose_turn_async(...)``, which
    dispatches to a worker thread; the body's
    synchronous SQLAlchemy transaction would otherwise block the
    event loop.

    Closes synthesised review finding SA-7 / M1 (async-loop guard
    is convention-only without a runtime check)."""
    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_async_guard")

    with pytest.raises(RuntimeError, match="must be dispatched via"):
        # Calling the SYNC method from inside an async test function
        # — there IS a running loop in the calling thread, which is
        # exactly the misuse the guard exists to detect.
        service.persist_compose_turn(
            session_id="s_async_guard",
            assistant_content="",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )


@pytest.mark.asyncio
async def test_persist_compose_turn_async_protocol_dispatch_succeeds_from_async(service):
    """Companion to the async-guard test: production callers use the
    protocol-public async dispatcher, not the concrete sync primitive
    and not a concrete ``_run_sync`` bridge. The dispatcher runs the sync
    primitive in a worker thread (no running loop in that thread), so the
    guard passes while routes keep depending on SessionServiceProtocol."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_run_sync")

    outcome = await service.persist_compose_turn_async(
        session_id="s_run_sync",
        assistant_content="ok",
        redacted_assistant_tool_calls=(
            {"id": "tc_run_sync", "function": {"name": "f"}},
        ),
        redacted_tool_rows=(_RedactedToolRow("tc_run_sync", "{}", None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    assert outcome.assistant_id is not None
    assert outcome.unwind_audit_failed is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "happy_path or zero_tool_rows or persists_raw_content or rejects_cross_session_parent_state or accepts_valid_same_session_parent_state or rejects_stale_expected_current_state or accepts_matching_expected_current_state or refuses_async_invocation or async_protocol_dispatch_succeeds_from_async"
```
Expected: FAIL — `persist_compose_turn` does not yet exist (or, after Step 3, a partial implementation is missing the `raw_content` plumbing that
`test_persist_compose_turn_persists_raw_content` asserts).

- [ ] **Step 3: Implement `persist_compose_turn` (success path only)**

**Architectural note (A-F1, plan-review synthesis): place
``StaleComposeStateError`` in ``protocol.py``, not ``service.py``.**
The exception is part of the public contract that
``persist_compose_turn_async`` raises through
``SessionServiceProtocol`` (Step 3b adds the protocol method).
Defining the exception in the concrete service implementation would
force every Phase 3 caller — and any future replacement service
implementation — to import a concrete-class symbol just to catch a
contract-level error. Mirroring ``AuditIntegrityError``'s placement
in ``elspeth.contracts.errors`` (a leaf module that protocol clients
already import without taking a service dependency), the stale-state
error belongs alongside the Protocol definition. This keeps the
protocol surface — including its raises-set — importable by callers
that depend only on the abstraction.

First, in ``src/elspeth/web/sessions/protocol.py`` add the exception
class near the top of the module (after the imports, before the
Protocol class):

```python
class StaleComposeStateError(RuntimeError):
    """Compose result was based on a no-longer-current composition state.

    Raised by ``SessionServiceProtocol.persist_compose_turn_async`` (and
    its concrete implementation ``SessionServiceImpl.persist_compose_turn``)
    when the session's current composition state changed between the LLM
    call and the persist attempt. Defined here on the protocol module so
    Phase 3 callers can catch the error without importing the concrete
    service class — the symbol is part of the public contract, not an
    implementation detail.
    """
```

Then, in `src/elspeth/web/sessions/service.py`, add the method (full body in spec §5.2.2; success-path implementation here, error-path tasks later). Import the exception from the protocol module:

```python
from elspeth.web.sessions.protocol import StaleComposeStateError  # add to imports

def persist_compose_turn(
    self,
    *,
    session_id: str,
    assistant_content: str,
    raw_content: str | None = None,
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
    redacted_tool_rows: tuple[_RedactedToolRow, ...],
    parent_composition_state_id: str | None,
    expected_current_state_id: str | None,
    writer_principal: str,
    plugin_crash_pending: bool,
) -> _AuditOutcome:
    """Synchronous, single-transaction persistence of one compose turn.

    Spec §5.2.2. Concrete sync primitive. Production async callers MUST
    invoke ``await self.persist_compose_turn_async(...)`` through
    ``SessionServiceProtocol``; that dispatcher uses ``_run_sync`` under
    the hood. Calling this sync primitive directly from async land would
    block the event loop because the body opens a synchronous SQLAlchemy
    transaction.

    The guard below uses ``asyncio.get_running_loop()`` to detect
    misuse: if there is a running loop in the calling thread, we are
    in async land and MUST refuse. ``RuntimeError`` is the canonical
    "you called the wrong API" signal — the call site is a bug, not
    a recoverable user error. Closes synthesised review finding
    SA-7 / M1.
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread — we are in a worker thread
        # or pure sync test context. Proceed.
        pass
    else:
        raise RuntimeError(
            "persist_compose_turn must be dispatched via "
            "await self.persist_compose_turn_async(...) — "
            "calling it directly from a coroutine blocks the event "
            "loop on synchronous DB I/O."
        )

    now = self._now()
    with self._engine.begin() as conn:
        with self._session_write_lock(conn, session_id):
            # B5 (Phase 1 plan-review synthesis): if a parent composition
            # state is supplied, it MUST belong to this session. The composite
            # FK ``fk_chat_messages_composition_state_session`` would catch a
            # cross-session reference at INSERT time, but only with a generic
            # FK error AFTER the row attempt. The offensive-programming policy
            # (CLAUDE.md "Defensive Programming: Forbidden") requires a precise
            # named pre-check that produces a diagnostic identifying the caller,
            # the state, and the mismatched session. ``_assert_state_in_session``
            # is the canonical guard (already called by ``add_message``); using
            # it here brings ``persist_compose_turn`` to the same offensive
            # standard. Mirrors the §14.6 fork-slice parent-assistant guard.
            if parent_composition_state_id is not None:
                _assert_state_in_session(
                    conn,
                    state_id=parent_composition_state_id,
                    expected_session_id=session_id,
                    caller="persist_compose_turn",
                )

            # ``composition_states_table`` is imported directly at
            # ``service.py:27-29`` (``from elspeth.web.sessions.models
            # import composition_states_table``); the bare module-level
            # symbol is the canonical reference shape used elsewhere in
            # this file (see ``service.py:617`` and ``service.py:1005``).
            # Do NOT prefix with ``models.`` — there is no ``models``
            # module alias bound in this scope, and a stray prefix would
            # raise ``NameError`` at import time.
            current_state_id = conn.execute(
                select(composition_states_table.c.id)
                .where(composition_states_table.c.session_id == session_id)
                .order_by(composition_states_table.c.version.desc())
                .limit(1)
            ).scalar_one_or_none()
            if current_state_id != expected_current_state_id:
                raise StaleComposeStateError(
                    "persist_compose_turn: current composition state changed "
                    f"for session_id={session_id!r}; "
                    f"expected={expected_current_state_id!r}, "
                    f"actual={current_state_id!r}. Refusing to persist a "
                    "compose result based on a stale state."
                )

            base_seq = self._reserve_sequence_range(
                conn, session_id, count=1 + len(redacted_tool_rows)
            )

            assistant_id = self._insert_chat_message(
                conn,
                session_id=session_id,
                role="assistant",
                content=assistant_content,
                # B2 (Phase 1 plan-review synthesis): ``raw_content`` is the
                # audit-attribution column for assistant messages whose
                # ``content`` was rewritten by runtime preflight redaction.
                # Pre-B2 the parameter was hardcoded ``None`` here with a
                # "Phase 3 plumbs through then" comment, but Phase 3 is
                # described in its own plan as "loop only, no new
                # primitives" — Phase 3 wires the call site, it does not
                # extend the signature. Routes 2151 and 2601 in
                # ``src/elspeth/web/sessions/routes.py`` already pass
                # ``raw_content=result.raw_assistant_content`` to
                # ``add_message`` today; persist_compose_turn must accept
                # the same column from Phase 3 onward, so the parameter is
                # introduced in Phase 1 with a default of ``None`` (which
                # preserves existing Phase 1 happy-path test behaviour
                # unchanged) and Phase 3 supplies the LLM-response value.
                raw_content=raw_content,
                # ``deep_thaw`` recursively converts any MappingProxyType /
                # tuple inputs to JSON-serializable dict / list forms; this
                # is the same pattern ``save_composition_state`` uses for
                # frozen ``validation_errors`` (service.py:1015). Plain
                # ``list(...)`` would leave MappingProxyType inner values
                # unchanged, which json.dumps then rejects with TypeError.
                # Closes synthesised review finding P-L-4 / L18.
                tool_calls=deep_thaw(redacted_assistant_tool_calls) if redacted_assistant_tool_calls else None,
                sequence_no=base_seq,
                writer_principal=writer_principal,
                composition_state_id=parent_composition_state_id,
                tool_call_id=None,
                parent_assistant_id=None,
                created_at=now,
            )

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
                    created_at=now,
                )

        return _AuditOutcome(
            assistant_id=assistant_id,
            unwind_audit_failed=False,
        )
```

- [ ] **Step 3b: Expose only the async dispatcher on `SessionServiceProtocol`**

In `src/elspeth/web/sessions/protocol.py`, add the protocol method in
the same task that introduces `SessionServiceImpl.persist_compose_turn`.
This keeps the overview's public contract true for Phase 1 and prevents
Phase 3 from reaching into the concrete service type or its `_run_sync`
bridge. The sync `SessionServiceImpl.persist_compose_turn` remains
concrete-only and guarded against direct async-loop use.

Avoid a runtime circular import: Task 6's `_persist_payload.py` imports
`CompositionStateData` from `protocol.py`, so `protocol.py` must import
the payload dataclasses only for static typing.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.web.sessions._persist_payload import _AuditOutcome, _RedactedToolRow


class SessionServiceProtocol(Protocol):
    # ... existing methods ...

    async def persist_compose_turn_async(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[_RedactedToolRow, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: str,
        plugin_crash_pending: bool,
    ) -> _AuditOutcome: ...
```

In `SessionServiceImpl`, add the matching async dispatcher with the same
signature:

```python
async def persist_compose_turn_async(
    self,
    *,
    session_id: str,
    assistant_content: str,
    raw_content: str | None = None,
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
    redacted_tool_rows: tuple[_RedactedToolRow, ...],
    parent_composition_state_id: str | None,
    expected_current_state_id: str | None,
    writer_principal: str,
    plugin_crash_pending: bool,
) -> _AuditOutcome:
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

If `from __future__ import annotations` is ever removed from this file,
the annotation strategy must be revisited before landing the change.

**Protocol-isolation discipline note (A-F1).**
``StaleComposeStateError`` is defined on
``src/elspeth/web/sessions/protocol.py`` (see Step 3 above), not on
``service.py``. Phase 3 callers and any future replacement service
implementation can therefore catch the error using
``from elspeth.web.sessions.protocol import StaleComposeStateError``
without taking a dependency on the concrete service class. This
mirrors ``AuditIntegrityError``'s placement in
``elspeth.contracts.errors``: protocol-level error shapes belong on
the abstraction, not the implementation.

- [ ] **Step 3c: Validate tool-call ID set equality before any DB write (Q-F1)**

Before the ``self._engine.begin()`` block — and therefore before any
INSERT, before ``_session_write_lock`` acquisition, and before the
``_assert_state_in_session`` guard — ``persist_compose_turn`` MUST
prove that the assistant row's ``tool_calls`` list and the
``redacted_tool_rows`` tuple agree on a single, duplicate-free set
of tool-call IDs. Mismatches must fail with a precise typed exception
that names the violating IDs.

**Why this guard exists.** The transcript invariant
"every assistant ``tool_calls`` ID appears exactly once in the tool
rows for the same turn" is part of the audit contract: a transcript
where the assistant claimed to call tool ``tc_X`` but no tool row
was ever persisted (or vice versa) is a lie the audit trail cannot
distinguish from a successful exchange. The composite FK
``fk_chat_messages_composition_state_session`` and the partial unique
``uq_chat_messages_tool_call_id`` would catch a duplicate ID at
INSERT time, but only with a generic constraint error AFTER partial
work — and missing/extra IDs are not caught by any DB-level check
(an assistant with ``tool_calls=[{"id": "tc_X"}]`` and zero tool
rows is structurally legal at the row level, but is a contract
violation at the turn level).

The guard also runs BEFORE ``_session_write_lock`` because acquiring
the per-session lock to validate caller-supplied data would needlessly
serialise contention. The validation is a function of the call's
own arguments — no DB read required — so it is safe (and correct) to
run pre-lock.

**Where to place the helper.** Define it as a module-level private
function in ``src/elspeth/web/sessions/service.py`` (alongside the
existing ``_assert_state_in_session`` helper at ``service.py:88``)
so it is testable in isolation and the diagnostic strings live with
the rest of the service-boundary guards. Do NOT put it on
``_persist_payload.py`` — the payload module is intentionally pure
data containers, no validation behaviour.

```python
class ToolCallIDMismatchError(RuntimeError):
    """Assistant ``tool_calls`` and persisted tool rows disagreed on
    the set of tool-call IDs for one compose turn.

    Carries the four mutually-exclusive failure axes (missing, extra,
    duplicate-in-assistant, duplicate-in-rows) so the diagnostic
    string identifies WHICH violation fired without forcing the
    caller to re-derive it.

    Defined here on the service module because the error shape is
    internal to ``persist_compose_turn``; callers either catch it as
    a contract violation or do not catch it at all (the more typical
    shape — a contract violation indicates a bug in the compose loop,
    not a recoverable user error).
    """

    def __init__(
        self,
        *,
        missing: frozenset[str],
        extra: frozenset[str],
        duplicates_in_assistant: frozenset[str],
        duplicates_in_rows: frozenset[str],
    ) -> None:
        self.missing = missing
        self.extra = extra
        self.duplicates_in_assistant = duplicates_in_assistant
        self.duplicates_in_rows = duplicates_in_rows
        super().__init__(
            "persist_compose_turn: assistant tool_calls and tool rows "
            "disagree on the tool-call ID set "
            f"(missing={sorted(missing)!r}, extra={sorted(extra)!r}, "
            f"duplicates_in_assistant={sorted(duplicates_in_assistant)!r}, "
            f"duplicates_in_rows={sorted(duplicates_in_rows)!r}). "
            "Refusing to persist a turn that would leave the audit "
            "trail with an asymmetric assistant/tool transcript."
        )


def _validate_tool_call_id_set_equality(
    *,
    redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
    redacted_tool_rows: tuple[_RedactedToolRow, ...],
) -> None:
    """Raise ``ToolCallIDMismatchError`` if the assistant's
    ``tool_calls`` IDs and the tool rows' ``tool_call_id`` values are
    not the same unique set.

    Four failure axes — any of them raises:

    - ``missing``: an assistant ``tool_calls`` ID with no
      corresponding ``_RedactedToolRow``.
    - ``extra``: a ``_RedactedToolRow.tool_call_id`` not claimed by
      any assistant ``tool_calls`` entry.
    - ``duplicates_in_assistant``: the same ID appears twice in
      ``redacted_assistant_tool_calls``.
    - ``duplicates_in_rows``: the same ID appears twice in
      ``redacted_tool_rows``.

    All four are reported simultaneously (rather than short-circuiting
    on the first one) so the caller sees the full picture in one
    diagnostic — debugging asymmetric transcripts is exponentially
    easier when every violation is named at once.

    The empty-empty case (assistant has zero tool calls, zero tool
    rows) is valid and returns silently — see W10a /
    ``test_persist_compose_turn_zero_tool_rows``.
    """
    assistant_ids: list[str] = [
        # The ``id`` key is contractually present on every entry — the
        # OpenAI / LiteLLM tool-call shape requires it. If it isn't,
        # that's an upstream framework bug, not data we should defend
        # against (CLAUDE.md offensive programming).
        tc["id"]
        for tc in redacted_assistant_tool_calls
    ]
    row_ids: list[str] = [row.tool_call_id for row in redacted_tool_rows]

    assistant_set = set(assistant_ids)
    row_set = set(row_ids)
    missing = frozenset(assistant_set - row_set)
    extra = frozenset(row_set - assistant_set)
    duplicates_in_assistant = frozenset(
        i for i in assistant_set if assistant_ids.count(i) > 1
    )
    duplicates_in_rows = frozenset(
        i for i in row_set if row_ids.count(i) > 1
    )

    if missing or extra or duplicates_in_assistant or duplicates_in_rows:
        raise ToolCallIDMismatchError(
            missing=missing,
            extra=extra,
            duplicates_in_assistant=duplicates_in_assistant,
            duplicates_in_rows=duplicates_in_rows,
        )
```

In ``persist_compose_turn``, call the guard FIRST — before
``_now()``, before ``_engine.begin()``:

```python
def persist_compose_turn(self, *, ...):
    # ... async-loop guard ...
    _validate_tool_call_id_set_equality(
        redacted_assistant_tool_calls=redacted_assistant_tool_calls,
        redacted_tool_rows=redacted_tool_rows,
    )
    now = self._now()
    with self._engine.begin() as conn:
        # ... rest of body ...
```

Add four named regression tests to
``tests/unit/web/sessions/test_persist_compose_turn.py``. Each test
verifies the guard fires BEFORE any DB write — the post-condition is
``SELECT COUNT(*) FROM chat_messages WHERE session_id = ...`` returns
0 — and that the diagnostic identifies the right axis:

```python
def test_persist_compose_turn_rejects_missing_tool_row(service):
    """Assistant claimed tc_X but no tool row was supplied. Guard
    fires pre-DB; ``missing`` axis named in the diagnostic."""
    from elspeth.web.sessions.service import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_missing")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"missing=\['tc_X'\].*extra=\[\]",
    ):
        service.persist_compose_turn(
            session_id="s_missing",
            assistant_content="ok",
            redacted_assistant_tool_calls=(
                {"id": "tc_X", "function": {"name": "f"}},
            ),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_missing'"
        )).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_extra_tool_row(service):
    """Tool row supplied for tc_Y, but assistant did not claim it.
    Guard fires pre-DB; ``extra`` axis named in the diagnostic."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.service import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_extra")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"missing=\[\].*extra=\['tc_Y'\]",
    ):
        service.persist_compose_turn(
            session_id="s_extra",
            assistant_content="ok",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(_RedactedToolRow("tc_Y", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_extra'"
        )).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_mismatched_tool_call_ids(service):
    """Assistant claimed tc_A; the tool row was for tc_B. Both axes
    fire — ``missing=['tc_A']`` AND ``extra=['tc_B']`` — and the
    diagnostic names both."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.service import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_mismatch")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"missing=\['tc_A'\].*extra=\['tc_B'\]",
    ):
        service.persist_compose_turn(
            session_id="s_mismatch",
            assistant_content="ok",
            redacted_assistant_tool_calls=(
                {"id": "tc_A", "function": {"name": "f"}},
            ),
            redacted_tool_rows=(_RedactedToolRow("tc_B", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_mismatch'"
        )).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_duplicate_tool_call_id_in_assistant(service):
    """Same ID appears twice in ``redacted_assistant_tool_calls`` —
    structurally malformed transcript. Guard fires pre-DB."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.service import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_dup_assist")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"duplicates_in_assistant=\['tc_D'\]",
    ):
        service.persist_compose_turn(
            session_id="s_dup_assist",
            assistant_content="ok",
            redacted_assistant_tool_calls=(
                {"id": "tc_D", "function": {"name": "f"}},
                {"id": "tc_D", "function": {"name": "g"}},
            ),
            redacted_tool_rows=(_RedactedToolRow("tc_D", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_dup_assist'"
        )).scalar()
        assert count == 0


def test_persist_compose_turn_rejects_duplicate_tool_call_id_in_rows(service):
    """Same ID appears twice in ``redacted_tool_rows`` — would
    eventually fire ``uq_chat_messages_tool_call_id`` at INSERT time,
    but the named guard catches it pre-DB so the diagnostic identifies
    the duplicate and the audit-integrity counter does not move."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.service import ToolCallIDMismatchError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_dup_rows")
    with pytest.raises(
        ToolCallIDMismatchError,
        match=r"duplicates_in_rows=\['tc_E'\]",
    ):
        service.persist_compose_turn(
            session_id="s_dup_rows",
            assistant_content="ok",
            redacted_assistant_tool_calls=(
                {"id": "tc_E", "function": {"name": "f"}},
            ),
            redacted_tool_rows=(
                _RedactedToolRow("tc_E", "{}", None),
                _RedactedToolRow("tc_E", "{}", None),
            ),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        count = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_dup_rows'"
        )).scalar()
        assert count == 0
```

Closes synthesised review finding Q-F1 and makes the Preflight Step 3
contract testable rather than aspirational. The guard placement
(pre-lock, pre-transaction) is what makes Done-When items 1 and 2
implementable: missing/extra/mismatched/duplicate IDs fail before
any database write.

- [ ] **Step 3d: Document the cancellation/idempotency contract for `persist_compose_turn_async` (Q-F2)**

Per Preflight Step 2 ("Define cancellation/idempotency before
coding"), record the chosen retry-after-cancel behaviour explicitly
in this PR before ``persist_compose_turn_async`` is exercised by any
caller. The dispatcher uses ``self._run_sync(self.persist_compose_turn,
...)``, which (per the existing ``SessionServiceImpl._run_sync``
contract) bridges to a worker thread — that thread is shielded from
the caller's task cancellation, so a caller ``cancel()`` in async
land does NOT interrupt the in-flight DB transaction.

**Decision: commit-wins.** When the caller is cancelled mid-flight
through ``persist_compose_turn_async``, the underlying worker
continues to run the synchronous ``persist_compose_turn`` body to
completion. One of two terminal states results:

1. **The transaction commits.** The assistant row, tool rows, and
   composition_states rows are durably persisted. The caller observes
   ``CancelledError`` and never sees the ``_AuditOutcome`` — but the
   audit trail records that the turn was persisted. **Callers MUST
   NOT retry on ``CancelledError`` once the worker is dispatched.**
   Retrying would attempt to insert the same assistant + tool rows a
   second time and fire ``uq_chat_messages_tool_call_id``, fabricating
   a ``tool_row_integrity_violation_total`` increment — which under
   ELSPETH's auditability standard is evidence-tampering-class harm
   (SLO threshold = 0; the alert fires on a non-event). The compose
   loop in Phase 3 is the only caller of
   ``persist_compose_turn_async``, so the no-retry policy is enforced
   in one location.
2. **The transaction rolls back atomically.** A DB-level error
   (``IntegrityError``, ``OperationalError``, the new
   ``ToolCallIDMismatchError`` raised pre-DB) causes the
   ``engine.begin()`` block to roll back. No rows are persisted.
   Retry-on-``CancelledError`` is still forbidden in this branch
   because the caller cannot distinguish branch 1 (committed) from
   branch 2 (rolled back) without reading back the database — and
   reading back is itself a race against the next compose turn.

**Why "commit-wins" rather than "rollback on cancel".** ELSPETH's
audit doctrine treats Tier-1 audit failures as crash-immediately
events (``AuditIntegrityError`` is registered in ``TIER_1_ERRORS``).
A "rollback on cancel" semantics would require the worker to
cooperate with caller cancellation — i.e. propagate
``CancelledError`` into the synchronous transaction context — which
is only achievable by raising mid-transaction. Mid-transaction
cancellation of an in-flight INSERT is not deterministically safe
across SQLite and PostgreSQL: SQLite's transaction model would not
roll back a partially-committed group of statements, and PostgreSQL
would emit a connection-state error that the existing
``OperationalError`` handler would classify as a Tier-1 audit
failure. Both behaviours fabricate Tier-1 alerts on a benign
caller-cancel pattern. "Commit-wins" sidesteps this by guaranteeing
the transaction reaches a definite terminal state regardless of
caller fate.

**Caller contract (binding).** The Phase 3 compose loop is the only
caller of ``persist_compose_turn_async``. The contract it must
honour, recorded here so Phase 3 can pin against it:

- On ``CancelledError`` from ``await
  service.persist_compose_turn_async(...)``: do not retry. The
  worker may have committed the turn; retrying risks a duplicate
  tool-call-ID INSERT that fires a fabricated Tier-1 counter
  increment. Re-raise the ``CancelledError`` and let the caller's
  cancellation propagate.
- On ``ToolCallIDMismatchError``, ``StaleComposeStateError``,
  ``IntegrityError``, ``AuditIntegrityError``: do not retry. These
  are caller-contract / Tier-1 errors and retry would hide the bug
  or fabricate audit-integrity violations.
- The ``unwind_audit_failed=True`` outcome path is the only flag-
  return shape that Phase 3 may handle locally — and even there the
  caller raises the captured plugin-crash exception (see Task 13).

**Testable proof.** Add the following to
``tests/unit/web/sessions/test_persist_compose_turn.py``:

```python
@pytest.mark.asyncio
async def test_persist_compose_turn_async_caller_cancellation_commits_anyway(service):
    """Q-F2 contract: when the caller is cancelled mid-flight through
    ``persist_compose_turn_async``, the underlying worker continues to
    completion. The post-cancel DB state must contain the persisted
    rows (commit-wins), and the integrity counter MUST NOT have moved
    on a benign cancel-and-retry pattern.

    The shielded ``_run_sync`` worker bridge does not propagate
    ``CancelledError`` into the synchronous transaction. The test
    proves this by:

    1. Awaiting ``persist_compose_turn_async`` inside an inner task.
    2. Cancelling the inner task immediately after awaiting it.
    3. Observing ``CancelledError`` on the caller side.
    4. Asserting the assistant row IS in the database (commit-wins).
    5. Asserting ``tool_row_integrity_violation_total`` did NOT move.

    The cancellation injection uses ``asyncio.shield`` /
    ``inner_task.cancel()`` — no monkeypatching of ``_run_sync`` is
    required. Closes synthesised review finding Q-F2.
    """
    import asyncio
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_cancel")

    starting = observed_value(
        service._telemetry.tool_row_integrity_violation_total
    )

    async def _do_persist() -> None:
        await service.persist_compose_turn_async(
            session_id="s_cancel",
            assistant_content="commit-wins",
            redacted_assistant_tool_calls=(
                {"id": "tc_c1", "function": {"name": "f"}},
            ),
            redacted_tool_rows=(_RedactedToolRow("tc_c1", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    inner = asyncio.create_task(_do_persist())
    # Yield control to let the dispatcher hand off to the worker
    # thread before we cancel. The worker is shielded; the cancel
    # only affects the awaiter.
    # 50ms — single asyncio.sleep(0) is racy on slow CI; this validates
    # the worker has actually started before cancel arrives.
    await asyncio.sleep(0.05)
    inner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inner

    # Wait until the shielded worker has finished. Polling is fine
    # because the worker bridge has no public completion signal —
    # the test only asserts the committed terminal state.
    for _ in range(200):
        with service._engine.begin() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id='s_cancel'"
            )).scalar()
        if count == 2:  # assistant + tool
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail(
            "shielded worker did not commit within 2s; commit-wins "
            "contract is not honoured by the current _run_sync bridge."
        )

    # Counter MUST NOT have moved — there was no IntegrityError, no
    # benign "fabricated Tier-1" event from the cancel path.
    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting
    ), (
        "Q-F2 regression: caller cancellation produced a "
        "tool_row_integrity_violation_total increment. Under the "
        "commit-wins contract, a clean cancel must not fabricate "
        "Tier-1 alerts (SLO threshold = 0)."
    )
```

The test's commit-wins polling loop is intentional: the shielded
worker's completion is observable only through the database state.
``asyncio.shield`` would prevent cancellation from reaching the
worker thread, but ``_run_sync`` already provides equivalent
shielding by dispatching to the bounded worker pool — a future
refactor that drops the shield must update this test along with the
caller contract above.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v -k "happy_path or zero_tool_rows or persists_raw_content or rejects_cross_session_parent_state or accepts_valid_same_session_parent_state or rejects_stale_expected_current_state or accepts_matching_expected_current_state or refuses_async_invocation or async_protocol_dispatch_succeeds_from_async or rejects_missing_tool_row or rejects_extra_tool_row or rejects_mismatched_tool_call_ids or rejects_duplicate_tool_call_id_in_assistant or rejects_duplicate_tool_call_id_in_rows or async_caller_cancellation_commits_anyway"
.venv/bin/python -m mypy src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py
```
Expected: PASS for all fifteen Task-11 tests:
- `test_persist_compose_turn_happy_path`
- `test_persist_compose_turn_zero_tool_rows` (W10a fix — pins the
  assistant-only call shape with empty `redacted_tool_rows` and empty
  `redacted_assistant_tool_calls`)
- `test_persist_compose_turn_persists_raw_content` (B2 fix — the
  primitive accepts the optional `raw_content` argument and persists
  it on the assistant row)
- `test_persist_compose_turn_rejects_cross_session_parent_state`
- `test_persist_compose_turn_accepts_valid_same_session_parent_state`
- `test_persist_compose_turn_rejects_stale_expected_current_state`
- `test_persist_compose_turn_accepts_matching_expected_current_state`
- `test_persist_compose_turn_refuses_async_invocation`
- `test_persist_compose_turn_async_protocol_dispatch_succeeds_from_async`
- `test_persist_compose_turn_rejects_missing_tool_row` (Q-F1 fix —
  Step 3c transcript validation guard)
- `test_persist_compose_turn_rejects_extra_tool_row` (Q-F1 fix)
- `test_persist_compose_turn_rejects_mismatched_tool_call_ids` (Q-F1
  fix — both ``missing`` and ``extra`` axes fire simultaneously)
- `test_persist_compose_turn_rejects_duplicate_tool_call_id_in_assistant` (Q-F1 fix)
- `test_persist_compose_turn_rejects_duplicate_tool_call_id_in_rows` (Q-F1 fix)
- `test_persist_compose_turn_async_caller_cancellation_commits_anyway`
  (Q-F2 fix — Step 3d commit-wins contract)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): add persist_compose_turn happy path with transcript validation guard and commit-wins async contract (composer-progress-persistence phase 1B)

- Adds the synchronous persist_compose_turn primitive (success path,
  spec §5.2.2) and its async dispatcher persist_compose_turn_async on
  SessionServiceProtocol.
- StaleComposeStateError lives on protocol.py, not service.py (A-F1 —
  protocol-level error shape belongs on the abstraction).
- Adds the pre-DB transcript validation guard
  _validate_tool_call_id_set_equality and ToolCallIDMismatchError
  (Q-F1 — Preflight Step 3 made testable; missing/extra/mismatched/
  duplicate IDs fail before any DB write).
- Documents and tests the commit-wins cancellation contract for
  persist_compose_turn_async (Q-F2 — Preflight Step 2 made binding
  for Phase 3).
"
```

---
## Task 12: `persist_compose_turn` IntegrityError disposition

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_persist_compose_turn_integrity_error_propagates(service):
    """Duplicate tool_call_id within one session triggers IntegrityError;
    counter increments; helper re-raises (no recovery — spec §4.5)."""
    from sqlalchemy.exc import IntegrityError
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s7")

    # First turn: creates tool_call_id='dup'
    service.persist_compose_turn(
        session_id="s7",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "dup", "function": {"name": "x"}},),
        redacted_tool_rows=(_RedactedToolRow("dup", "{}", None),),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="s7",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "dup", "function": {"name": "x"}},),
            redacted_tool_rows=(_RedactedToolRow("dup", "{}", None),),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    assert observed_value(service._telemetry.tool_row_integrity_violation_total) == starting + 1


# Spec §4.5 enumerates multiple constraint sources that all flow
# through the same IntegrityError handler in persist_compose_turn.
# The test above covers the partial-unique-tool_call_id source; the
# parametrised test below covers the other reachable source via
# persist_compose_turn's parameter surface.
#
# Sources that are not reachable through public parameters are NOT in
# this matrix because they cannot occur via the entry point:
#
# - role enum violation — persist_compose_turn hardcodes
#   'assistant'/'tool'.
# - **uq_composition_state_version — closed by B1 (Phase 1
#   plan-review synthesis).** ``_StatePayload`` no longer carries a
#   caller-supplied ``version``; ``_insert_composition_state``
#   allocates it under _session_write_lock via
#   ``SELECT COALESCE(MAX(version), 0) + 1 WHERE session_id = :sid``.
#   The constraint is structurally unreachable from
#   ``persist_compose_turn`` — every concurrent allocator serialises
#   on the same lock and observes the previous allocator's
#   committed MAX. The pre-B1 draft of this test asserted the
#   counter SHOULD increment when version=1 was supplied twice;
#   that test codified the fabrication vector B1 closes (every
#   increment was structurally a contention loss masquerading as a
#   Tier-1 audit-integrity violation, with SLO threshold = 0). The
#   replacement test
#   ``test_persist_compose_turn_state_versions_do_not_collide``
#   below pins the post-B1 contract: serial successful persists
#   allocate contiguous versions and the counter MUST remain at its
#   starting value.
#
# ``nonexistent_parent_composition_state`` is deliberately NOT in this
# matrix. Task 11 added the offensive `_assert_state_in_session` guard,
# so a missing parent state is rejected before any INSERT attempts and
# raises RuntimeError with a precise service-boundary diagnostic. Treating
# that as an IntegrityError would make the test pass for the wrong path
# and would double-count a caller contract violation as a Tier-1 DB
# integrity event. The separate RuntimeError regression below pins this.
#
# Closes synthesised review finding M6 / Q-F-04.


@pytest.mark.parametrize(
    "scenario_name, setup_kwargs, expected_match",
    [
        pytest.param(
            "unknown_writer_principal",
            {"writer_principal": "rogue_caller"},
            r"ck_chat_messages_writer_principal",
            id="ck_chat_messages_writer_principal",
        ),
    ],
)
def test_persist_compose_turn_integrity_error_matrix(
    service, scenario_name, setup_kwargs, expected_match,
):
    """Each scenario triggers a distinct §4.5 source via
    persist_compose_turn's parameter surface; all flow through the
    same handler (counter increment + raise). The test asserts both
    the counter increments AND the constraint name appears in the
    raised exception message — without the ``match=``, a wrong
    constraint firing first would false-green this test."""
    from sqlalchemy.exc import IntegrityError
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id=f"s_{scenario_name}")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    base_kwargs = {
        "session_id": f"s_{scenario_name}",
        "assistant_content": "",
        "redacted_assistant_tool_calls": (
            {"id": f"{scenario_name}_tc", "function": {"name": "f"}},
        ),
        "redacted_tool_rows": (
            _RedactedToolRow(f"{scenario_name}_tc", "{}", None),
        ),
        "parent_composition_state_id": None,
        "expected_current_state_id": None,
        "writer_principal": "compose_loop",
        "plugin_crash_pending": False,
    }
    base_kwargs.update(setup_kwargs)

    with pytest.raises(IntegrityError, match=expected_match):
        service.persist_compose_turn(**base_kwargs)

    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting + 1
    ), f"counter must increment for {scenario_name}"


def test_persist_compose_turn_rejects_missing_parent_state_before_insert(service):
    """A nonexistent parent composition state is a caller contract error,
    not an IntegrityError-source matrix case.

    Task 11's `_assert_state_in_session` guard must reject the missing
    state before the assistant row INSERT. The audit-integrity counter
    must not move because no DB constraint fired and no Tier-1 audit
    corruption was observed.
    """
    from elspeth.web.sessions._persist_payload import _RedactedToolRow
    from elspeth.web.sessions.telemetry import observed_value

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_missing_parent")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    with pytest.raises(
        RuntimeError,
        match=(
            r"persist_compose_turn: composition_state_id='doesnotexist' "
            r"does not exist"
        ),
    ):
        service.persist_compose_turn(
            session_id="s_missing_parent",
            assistant_content="",
            redacted_assistant_tool_calls=(
                {"id": "missing_parent_tc", "function": {"name": "f"}},
            ),
            redacted_tool_rows=(
                _RedactedToolRow("missing_parent_tc", "{}", None),
            ),
            parent_composition_state_id="doesnotexist",
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )

    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting
    )


def test_persist_compose_turn_state_versions_do_not_collide(service):
    """B1 (Phase 1 plan-review synthesis): the earlier draft of this
    test was named ``test_persist_compose_turn_duplicate_state_version_propagates``
    and asserted that supplying ``_StatePayload(version=1)`` on two
    successive turns triggered ``uq_composition_state_version`` AND
    incremented ``tool_row_integrity_violation_total``. **That test
    codified the fabrication vector B1 closes** — every IntegrityError
    increment on this constraint is structurally a contention loss
    masquerading as a Tier-1 audit-integrity violation, and the SLO
    threshold for the counter is 0.

    Post-B1 the test is impossible to write: ``_StatePayload`` has no
    ``version`` field, and ``_insert_composition_state`` allocates
    versions under _session_write_lock via
    ``SELECT COALESCE(MAX(version), 0) + 1 WHERE session_id = :sid``.
    Two successive ``persist_compose_turn`` calls for the same session
    each receive a contiguous version (1, then 2); neither raises.
    The counter MUST remain at its starting value.

    This replacement test pins the post-B1 behaviour: serial successful
    persists allocate contiguous versions and never increment the
    integrity counter for the version-collision constraint. The
    concurrent same-state compose contract is exercised on PostgreSQL by
    Task 16's stale-rejection regression."""
    from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData
    from elspeth.web.sessions.telemetry import observed_value
    from sqlalchemy import text

    with service._engine.begin() as conn:
        _make_session(conn, session_id="s_ver")

    starting = observed_value(service._telemetry.tool_row_integrity_violation_total)

    service.persist_compose_turn(
        session_id="s_ver",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_v1", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_v1", "{}",
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    with service._engine.begin() as conn:
        first_state_id = conn.execute(text(
            "SELECT id FROM composition_states "
            "WHERE session_id='s_ver' ORDER BY version DESC LIMIT 1"
        )).scalar_one()

    # Second turn — pre-B1 this would have collided on
    # uq_composition_state_version because the test supplied
    # version=1 on both turns. Post-B1 the helper allocates
    # version=2 (COALESCE(MAX,0)+1 = 2), so the call succeeds.
    service.persist_compose_turn(
        session_id="s_ver",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_v2", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_v2", "{}",
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=first_state_id,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # Counter MUST NOT have moved — there is no version collision
    # because the helper allocated 1 then 2 under the lock.
    assert (
        observed_value(service._telemetry.tool_row_integrity_violation_total)
        == starting
    ), (
        "B1 regression: tool_row_integrity_violation_total incremented "
        "on serial state-version allocation. SLO threshold for this "
        "counter is 0; any increment here is a fabricated Tier-1 alert "
        "and evidence-tampering-class harm under the audit doctrine."
    )

    # And the two states have contiguous versions starting at 1.
    with service._engine.begin() as conn:
        versions = [
            r.version for r in conn.execute(text(
                "SELECT version FROM composition_states "
                "WHERE session_id='s_ver' ORDER BY version"
            ))
        ]
    assert versions == [1, 2], (
        f"B1 regression: per-session version allocation broken; got {versions}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py::test_persist_compose_turn_integrity_error_propagates -v
```
Expected: FAIL — counter does not increment because the helper does not catch IntegrityError yet.

- [ ] **Step 3: Wrap the body with IntegrityError catch**

Update `persist_compose_turn` in `src/elspeth/web/sessions/service.py` so the entire `with self._engine.begin() as conn:` body is inside a `try`:

```python
def persist_compose_turn(self, ...) -> _AuditOutcome:
    try:
        with self._engine.begin() as conn:
            # ... existing body ...
    except IntegrityError:
        self._telemetry.tool_row_integrity_violation_total.add(1)
        raise
```

(Add `from sqlalchemy.exc import IntegrityError`.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py -v
```
Expected: PASS for IntegrityError disposition AND happy path still passes.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
git commit -m "feat(sessions): persist_compose_turn IntegrityError disposition (composer-progress-persistence phase 1)"
```

---
## Task 13: `persist_compose_turn` OperationalError + audit-failure primacy

**Files:**
- Modify: `src/elspeth/web/sessions/service.py`
- Test: `tests/unit/web/sessions/test_persist_compose_turn.py` (extend) and `tests/unit/web/composer/test_audit_failure_primacy.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/web/composer/test_audit_failure_primacy.py`:

```python
"""Audit-failure primacy disposition (spec §5.2.2 / §5.5 rows 9-10).

Failure injection patches SQLAlchemy's dialect-level ``do_commit``
hook for one COMMIT attempt. This:

1. Complies with spec §8.6 (no mocking of ``persist_compose_turn``'s
   private helpers — ``_acquire_session_advisory_lock``,
   ``_reserve_sequence_range``, ``_insert_chat_message``, and
   ``_insert_composition_state`` exist to be exercised, not mocked).
2. Simulates the **dominant** production trigger named in spec §4.5 —
   COMMIT-time failure (disk full, fsync failure, network partition
   between the last INSERT and COMMIT) — rather than the INSERT-time
   failure the earlier plan draft simulated by mocking
   ``_insert_chat_message``.
3. Exercises the production code's actual ``try: with engine.begin():
   ... except OperationalError: ...`` path end to end. The wrapped
   COMMIT failure surfaces from ``engine.begin().__exit__`` and is
   caught by the outer ``except`` clause in ``persist_compose_turn``.
4. Avoids assigning to ``sqlite3.Connection.commit``. That method is
   read-only on CPython's sqlite3 connection object, so a test that
   patches it fails during setup for the wrong reason.

The earlier draft used ``patch.object(service, "_insert_chat_message",
side_effect=OperationalError(...))``. That violates spec §8.6 (helpers
are mocked) and tests the wrong failure point (INSERT-time).
"""
from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# Shared ``engine`` fixture and ``_make_session`` helper come from
# ``tests/unit/web/conftest.py`` — the parent-package conftest that
# both the sessions suite and this composer suite share. pytest
# auto-loads it for every test under ``tests/unit/web/...``, so the
# ``engine`` fixture is visible here without further wiring; the
# ``_make_session`` helper is imported explicitly via its absolute
# path (a bare ``from .conftest`` would resolve to
# ``tests/unit/web/composer/conftest.py``, which does not exist —
# synthesised review B5).
from tests.unit.web.conftest import _make_session as _make_session_in_conn


@pytest.fixture
def service(engine, tmp_path):
    return SessionServiceImpl(
        engine, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


def _make_session(service, session_id):
    """Open a transaction on the service's engine and call the conftest
    helper. Wraps the connection-level helper so audit-primacy tests
    can express setup tersely."""
    with service._engine.begin() as conn:
        _make_session_in_conn(conn, session_id=session_id)


@contextlib.contextmanager
def _force_commit_failure(engine: Engine) -> Iterator[None]:
    """Inject an ``OperationalError`` on the next COMMIT.

    Patches ``engine.dialect.do_commit`` for one call. SQLAlchemy calls
    this hook from ``engine.begin().__exit__``; raising a
    ``sqlite3.OperationalError`` here is wrapped by SQLAlchemy as
    ``sqlalchemy.exc.OperationalError`` and reaches
    ``persist_compose_turn``'s outer OperationalError handler. The
    original hook is restored in ``finally`` so cleanup paths (e.g.
    test teardown) can commit normally.

    SQLite-only — the test suite for audit-failure primacy runs
    against the in-memory SQLite engine. The CL-PP-11 testcontainer
    Postgres test exercises a different scenario (advisory-lock
    contention) and does not require commit-failure injection.
    """
    original_do_commit = engine.dialect.do_commit
    fired = False

    def _fail_once(dbapi_conn: object) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise sqlite3.OperationalError(
                "simulated COMMIT failure (test injection)"
            )
        original_do_commit(dbapi_conn)

    engine.dialect.do_commit = _fail_once
    try:
        yield
    finally:
        engine.dialect.do_commit = original_do_commit


def test_audit_fail_no_plugin_crash_raises_audit_integrity_error(service):
    """Tool succeeded (plugin_crash_pending=False), audit COMMIT failed:
    ``persist_compose_turn`` must increment the Tier-1 counter AND
    raise :class:`AuditIntegrityError` chained from the original
    ``OperationalError``. Returning a flag would be a doctrine
    violation — the caller could ignore the flag and proceed with
    corrupted audit state. Closes synthesised review finding H1."""
    from elspeth.contracts.errors import AuditIntegrityError, TIER_1_ERRORS
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p1")
    starting = observed_value(service._telemetry.tool_row_tier1_violation_total)

    with _force_commit_failure(service._engine):
        with pytest.raises(AuditIntegrityError) as exc_info:
            service.persist_compose_turn(
                session_id="p1",
                assistant_content="hi",
                redacted_assistant_tool_calls=(),
                redacted_tool_rows=(),
                parent_composition_state_id=None,
                expected_current_state_id=None,
                writer_principal="compose_loop",
                plugin_crash_pending=False,
            )

    # The original OperationalError is preserved as the chained cause.
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, OperationalError)

    # Counter increments before the raise — telemetry-after-audit per
    # CLAUDE.md primacy.
    assert (
        observed_value(service._telemetry.tool_row_tier1_violation_total)
        == starting + 1
    )

    # The exception must be in TIER_1_ERRORS so ``except Exception:``
    # blocks cannot silently swallow it.
    assert isinstance(exc_info.value, TIER_1_ERRORS)


def test_audit_fail_during_plugin_crash_records_unwind_failure(service):
    """Tool failed (plugin_crash_pending=True) AND audit COMMIT failed:
    ``persist_compose_turn`` must increment the unwind-audit-failure
    counter and RETURN an outcome with ``unwind_audit_failed=True``.
    The unwind path returns rather than raises because the caller
    already has a captured plugin-crash exception to raise; surfacing
    a separate audit exception here would mask the original tool
    failure. The audit failure is recorded via counter + slog
    (permitted under CLAUDE.md primacy because the audit system
    itself failed)."""
    from elspeth.web.sessions.telemetry import observed_value

    _make_session(service, "p2")
    starting = observed_value(
        service._telemetry.tool_row_persist_failed_during_unwind_total
    )

    with _force_commit_failure(service._engine):
        outcome = service.persist_compose_turn(
            session_id="p2",
            assistant_content="hi",
            redacted_assistant_tool_calls=(),
            redacted_tool_rows=(),
            parent_composition_state_id=None,
            expected_current_state_id=None,
            writer_principal="compose_loop",
            plugin_crash_pending=True,
        )

    assert outcome.assistant_id is None
    assert outcome.unwind_audit_failed is True
    assert (
        observed_value(
            service._telemetry.tool_row_persist_failed_during_unwind_total
        )
        == starting + 1
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py -v
```
Expected: FAIL.

- [ ] **Step 3: Add OperationalError handler in `persist_compose_turn`**

In `src/elspeth/web/sessions/service.py`, expand the try/except to:

```python
def persist_compose_turn(self, ...) -> _AuditOutcome:
    try:
        with self._engine.begin() as conn:
            # ... existing body ...
    except IntegrityError:
        self._telemetry.tool_row_integrity_violation_total.add(1)
        raise
    except OperationalError as audit_exc:
        if plugin_crash_pending:
            # Tool plugin already crashed; audit unwind also failed.
            # Caller will raise the captured plugin crash — surfacing
            # a separate audit exception here would mask the original
            # tool failure. Record the audit failure via counter +
            # slog (permitted under CLAUDE.md primacy because the
            # audit system itself failed) and return the
            # unwind-failure outcome.
            self._telemetry.tool_row_persist_failed_during_unwind_total.add(1)
            self._log.warning(
                "audit_insert_failed_during_tool_failure_unwind",
                session_id=session_id,
                audit_exc_class=type(audit_exc).__name__,
            )
            return _AuditOutcome(
                assistant_id=None,
                unwind_audit_failed=True,
            )
        # Tier-1 violation: tool succeeded, audit failed. CRASH per
        # CLAUDE.md doctrine. AuditIntegrityError is registered in
        # TIER_1_ERRORS (via @tier_1_error on its declaration in
        # contracts/errors.py) so ``except Exception:`` blocks
        # cannot silently swallow it. The caller has no opportunity
        # to ignore the failure — flag-return would be a doctrine
        # violation per synthesised review finding H1.
        self._telemetry.tool_row_tier1_violation_total.add(1)
        raise AuditIntegrityError(
            f"persist_compose_turn: audit insert failed for "
            f"session_id={session_id!r} with tool succeeded — "
            f"Tier-1 audit corruption (no recovery)"
        ) from audit_exc
```

(Add `from sqlalchemy.exc import OperationalError` and confirm `from elspeth.contracts.errors import AuditIntegrityError` is imported — it already is at the top of `service.py`.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py tests/unit/web/sessions/test_persist_compose_turn.py -v
```
Expected: PASS for primacy tests AND no regression on the previous tests.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/service.py tests/unit/web/composer/test_audit_failure_primacy.py
git commit -m "feat(sessions): persist_compose_turn OperationalError + audit-failure primacy (composer-progress-persistence phase 1)"
```

---
## Task 15: Schema-level INV-AUDIT-AHEAD backward-direction integration test

**Files:**
- Create: `tests/integration/web/test_inv_audit_ahead_backward.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/web/test_inv_audit_ahead_backward.py`:

```python
"""Spec §4.1.2 / §1.4 NFR: state-ahead-of-audit is impossible at the
schema level. After any persist_compose_turn call, the SQL predicate
below must return zero rows."""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions._persist_payload import _RedactedToolRow, _StatePayload
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.telemetry import build_sessions_telemetry
import structlog

# ``_make_session`` lives in ``tests/integration/web/conftest.py`` — a
# duplicate of the unit-test conftest helper. Importing the helper
# here keeps the per-test session-insert site uniform with the rest
# of the suite.
from .conftest import _make_session


@pytest.fixture
def service(tmp_path):
    """Service with an in-memory SQLite engine. The test runs
    end-to-end against the real production code paths
    (``create_session_engine`` + ``initialize_session_schema``);
    integration here means "exercises persist_compose_turn against
    a real SQLite engine," not "uses Docker" — see the conftest
    docstring for why integration and unit conftests are separate."""
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return SessionServiceImpl(
        eng, data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger(),
    )


_BACKWARD_PREDICATE = """
SELECT cs.id
  FROM composition_states cs
  LEFT JOIN chat_messages cm
    ON cm.composition_state_id = cs.id AND cm.role = 'tool'
 WHERE cs.provenance = 'tool_call' AND cs.version > 0
   AND cm.id IS NULL
"""


def test_backward_direction_holds_after_successful_persist(service):
    with service._engine.begin() as conn:
        _make_session(conn, session_id="b1")
    service.persist_compose_turn(
        session_id="b1",
        assistant_content="ok",
        redacted_assistant_tool_calls=({"id": "tc_a", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_a",
                '{"r": 1}',
                # B1 (Phase 1 plan-review synthesis): no ``version=``;
                # ``_insert_composition_state`` allocates under the lock.
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    with service._engine.begin() as conn:
        violations = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
    assert violations == [], f"backward-direction violation rows: {violations}"


def test_backward_direction_holds_after_integrity_error_rollback(service):
    """After a failed persist_compose_turn (rolled back transaction), no
    composition_states row should be visible."""
    from sqlalchemy.exc import IntegrityError

    with service._engine.begin() as conn:
        _make_session(conn, session_id="b2")
    # First successful turn.
    service.persist_compose_turn(
        session_id="b2",
        assistant_content="",
        redacted_assistant_tool_calls=({"id": "tc_x", "function": {"name": "f"}},),
        redacted_tool_rows=(
            _RedactedToolRow(
                "tc_x",
                "{}",
                # B1: no ``version=``; helper allocates under the lock.
                _StatePayload(
                    data=CompositionStateData(),
                    derived_from_state_id=None,
                ),
            ),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )
    with service._engine.begin() as conn:
        first_state_id = conn.execute(text(
            "SELECT id FROM composition_states "
            "WHERE session_id='b2' ORDER BY version DESC LIMIT 1"
        )).scalar_one()
    # Second turn deliberately reuses tc_x to trigger the partial
    # unique index ``uq_chat_messages_tool_call_id`` (added in Task 2).
    with pytest.raises(
        IntegrityError,
        match=(
            r"(UNIQUE.*chat_messages.*session_id.*tool_call_id"
            r"|uq_chat_messages_tool_call_id)"
        ),
    ):
        service.persist_compose_turn(
            session_id="b2",
            assistant_content="",
            redacted_assistant_tool_calls=({"id": "tc_x", "function": {"name": "f"}},),
            redacted_tool_rows=(
                _RedactedToolRow(
                    "tc_x",
                    "{}",
                    # B1: no ``version=``.
                    _StatePayload(
                        data=CompositionStateData(),
                        derived_from_state_id=None,
                    ),
                ),
            ),
            parent_composition_state_id=None,
            expected_current_state_id=first_state_id,
            writer_principal="compose_loop",
            plugin_crash_pending=False,
        )
    with service._engine.begin() as conn:
        violations = conn.execute(text(_BACKWARD_PREDICATE)).fetchall()
    assert violations == []
    # And exactly one tool_call provenance row from the first (successful) turn.
    state_count = conn.execute(text(
        "SELECT COUNT(*) AS c FROM composition_states "
        "WHERE session_id='b2' AND provenance='tool_call'"
    )).scalar()
    assert state_count == 1


def test_get_messages_orders_assistant_before_tool_rows_within_one_turn(service):
    """B2 (Phase 1 plan-review synthesis): a single ``persist_compose_turn``
    stamps every row in the turn with one shared ``created_at`` = ``now``;
    on fast SQLite the rows share a microsecond, so ``get_messages``'s
    pre-B2 ``ORDER BY created_at`` returned them nondeterministically.
    Post-B2 ``get_messages`` orders by ``sequence_no`` (allocated under
    the session write lock = monotonic and unique), so the
    intra-turn order is the order in which the writer appended rows
    (assistant first, tool rows in plan order). This test would have
    failed on the pre-B2 codebase.
    """
    from uuid import UUID

    # B2 (Phase 1 plan-review synthesis): pre-B2 this test bound
    # ``sid="ord1"`` and ``sid_uuid=UUID("00000000-...-001")``, then
    # inserted the session under ``sid`` and queried under
    # ``sid_uuid`` — two different sessions. The fix derives one
    # canonical UUID and uses its string form for ``_make_session`` /
    # ``persist_compose_turn`` and the UUID form for ``get_messages``.
    sid_uuid = UUID("00000000-0000-0000-0000-000000000001")
    sid = str(sid_uuid)
    with service._engine.begin() as conn:
        _make_session(conn, session_id=sid)
    # B2: ``assistant_id`` and ``assistant_raw_content`` were stale
    # kwargs from a prior plan draft that ``persist_compose_turn`` does
    # not declare. ``assistant_id`` is helper-generated (the test never
    # referenced the supplied value); ``assistant_raw_content`` was a
    # typo for the new ``raw_content`` parameter and is left at its
    # default ``None`` here because the test's narrative does not
    # exercise the redaction-attribution path.
    service.persist_compose_turn(
        session_id=sid,
        assistant_content="ok",
        redacted_assistant_tool_calls=(
            {"id": "tc_a", "function": {"name": "f"}},
            {"id": "tc_b", "function": {"name": "g"}},
            {"id": "tc_c", "function": {"name": "h"}},
        ),
        # B1 (Phase 1 plan-review synthesis): no ``version=`` kwargs.
        # ``_insert_composition_state`` allocates per-session contiguous
        # versions (1, 2, 3) under _session_write_lock.
        redacted_tool_rows=(
            _RedactedToolRow("tc_a", "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
            _RedactedToolRow("tc_b", "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
            _RedactedToolRow("tc_c", "{}", _StatePayload(data=CompositionStateData(), derived_from_state_id=None)),
        ),
        parent_composition_state_id=None,
        expected_current_state_id=None,
        writer_principal="compose_loop",
        plugin_crash_pending=False,
    )

    # ``get_messages`` is async (returns ChatMessageRecord objects); the
    # post-B2 ORDER BY sequence_no clause guarantees a stable order.
    import asyncio
    msgs = asyncio.run(service.get_messages(sid_uuid))
    roles = [m.role for m in msgs]
    # Exactly four rows from this turn: 1 assistant + 3 tool. No fork
    # rows, no system messages — _make_session left the chat empty.
    assert roles == ["assistant", "tool", "tool", "tool"], (
        f"intra-turn ordering broken: expected assistant before all tool rows, "
        f"got {roles!r} — see plan §14.7 (B2 fix). The pre-B2 ORDER BY created_at "
        f"would produce a nondeterministic permutation of these four roles."
    )
    # And the tool_call_id sequence is preserved (a→b→c, the order
    # ``redacted_tool_rows`` was supplied in).
    tool_ids = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert tool_ids == ["tc_a", "tc_b", "tc_c"], (
        f"intra-tool ordering broken: {tool_ids!r}"
    )
```

- [ ] **Step 2: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/integration/web/test_inv_audit_ahead_backward.py -v
```
Expected: PASS — including the new
`test_get_messages_orders_assistant_before_tool_rows_within_one_turn`.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/web/test_inv_audit_ahead_backward.py
git commit -m "test(integration): schema-level INV-AUDIT-AHEAD backward-direction + intra-turn ordering (composer-progress-persistence phase 1)"
```

---

---

## Schedule 1B Done When

1. [ ] `persist_compose_turn` cannot persist assistant/tool transcript mismatches. (Implemented by Task 11 Step 3c — `_validate_tool_call_id_set_equality` guard and `ToolCallIDMismatchError`. Pinned by `test_persist_compose_turn_rejects_missing_tool_row`, `..._rejects_extra_tool_row`, `..._rejects_mismatched_tool_call_ids`.)
2. [ ] Missing, extra, mismatched, and duplicate tool-call IDs fail before any database write. (Same guard as item 1; the post-condition `SELECT COUNT(*) FROM chat_messages = 0` after each rejection proves the pre-DB placement. Duplicates pinned by `..._rejects_duplicate_tool_call_id_in_assistant` and `..._rejects_duplicate_tool_call_id_in_rows`.)
3. [ ] IntegrityError and broader persistence-failure tests preserve primary failure semantics and privacy. (Tasks 12 + 13 — IntegrityError handler with counter increment, OperationalError handler with audit-primacy disposition.)
4. [ ] `persist_compose_turn_async` has tested cancellation/retry semantics. (Task 11 Step 3d — commit-wins contract documented; `test_persist_compose_turn_async_caller_cancellation_commits_anyway` proves the worker commits despite caller cancel and the integrity counter does not move.)
5. [ ] INV-AUDIT-AHEAD backward-direction proof passes without relying on PostgreSQL/testcontainer infrastructure. (Task 15 — SQL predicate against in-memory SQLite engine.)
6. [ ] Run `pytest tests/unit/web/sessions/test_static_direct_writers.py -v` and confirm it passes after all 1B tasks land.
      Reason: 1B introduces new direct-write paths (`_insert_chat_message`, `_insert_composition_state` callers via `persist_compose_turn`).
      The static-writer scanner is the mechanical guard that they're allowlisted with past-tense purpose strings explaining why each direct write is necessary
      instead of going through the normal session-write façade.
7. [ ] Pre-merge gate (run all four from worktree root, all must pass before opening or merging the PR):
      ```bash
      .venv/bin/python -m pytest tests/unit/web tests/integration/web && \
      .venv/bin/python -m mypy src/ && \
      .venv/bin/python -m ruff check src/ tests/ && \
      .venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
      ```
      The `&&` chain is intentional: a failure in any earlier command short-circuits and surfaces immediately rather than masking later failures.
8. [ ] No compose-loop, route-visible, redaction, frontend, or PostgreSQL CI work is included.
9. [ ] A follow-up review confirms Schedule 1B no longer blocks Schedule 1C.
