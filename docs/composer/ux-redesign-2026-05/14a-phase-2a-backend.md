# Phase 2A — Backend: audit-readiness aggregation endpoint

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Land the backend for Phase 2 — a new `audit_readiness` package that composes existing validation, catalog, secrets, and retention signals into a per-session snapshot endpoint plus a separate narrative-Explain endpoint. No new audit-trail work, no new validation logic.

**Architecture:** Service-then-routes (mirrors Phase 1A). The package `src/elspeth/web/audit_readiness/` exposes `ReadinessService` taking the existing `ExecutionService`, `SessionServiceProtocol`, `WebSecretService`, and `WebSettings` as injected dependencies. Two routes: `GET /api/sessions/{sid}/audit-readiness` (with `Cache-Control: no-store`) returns the snapshot; `GET /api/sessions/{sid}/audit-readiness/explain` returns narrative.

**Tech Stack:** FastAPI, Pydantic v2, pytest.

**Sibling plan:** [14b-phase-2b-frontend.md](14b-phase-2b-frontend.md) — frontend panel, Explain view, Validate-button removal.

**Design reference:** [07-audit-readiness-panel.md](07-audit-readiness-panel.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**
- New package `src/elspeth/web/audit_readiness/` with `__init__.py`, `models.py`, `trust.py`, `service.py`, `explain.py`, `routes.py`.
- Pydantic models `AuditReadinessSnapshot`, `ReadinessRow`, `AuditReadinessExplain` inheriting `_StrictResponse` from `web/execution/schemas.py`.
- A closed allowlist of "external-boundary" plugin names in `trust.py` (see §"Plugin-trust derivation").
- `ReadinessService` composing existing checks (see §"Six rows"). `__init__` accepts a `state_from_record` collaborator (default: real converter) for test injection.
- `ValidationError` gains `error_code: str | None`; `ValidationCheck` gains `affected_nodes: tuple[str, ...]` — used by the secrets and provenance rows respectively.
- Narrative `build_narrative()` in `explain.py` — deterministic prose, no LLM call.
- Two GET routes (`Cache-Control: no-store`) wired into `src/elspeth/web/app.py`.
- `src/elspeth/web/sessions/ownership.py` (L3 peer) — shared `verify_session_ownership`; both `execution/routes.py` and `audit_readiness/routes.py` import from there.
- New field `WebSettings.payload_store_retention_days: int = Field(default=90, ge=1)` to back the Retention row.

**Out of scope (Phase 2B or later):**
- Frontend store, panel component, Explain view (Phase 2B).
- Removal of the standalone Validate button (Phase 2B).
- A per-user retention preference to compare against (would need new schema).
- LLM-interpretations row content (Phase 5b, gated by roadmap question B2).
- Auto-rerun on every composition change (Phase 2B fires on `compositionState.version`).
- Repair-hints click-target (Phase 2B; `component_ids` are already in the payload).

## Pre-existing test baseline (reviewers — read before TDD)

`tests/integration/web/composer/guided/*` has **17 pre-existing failures** on the RC5.2 baseline (independently verified at HEAD `fb73a4f76`, not caused by Phase 1A or 1B work). Failure pattern: `assert None is not None` on `get_current_state()` returns plus 200-vs-400/409 status mismatches in `step_chat` and `respond` endpoints — looks like a single guided-session fixture or state-bootstrap regression, not 17 independent bugs.

**Tracked as:** `elspeth-b1e04bea6d` (P2 bug, "17 pre-existing guided composer test failures on RC5.2") — owned by the per-step-chat track.

**Implications for Phase 2A reviewers:**

- Phase 2A's red→green discipline runs the full `tests/integration/web/` suite. Those 17 failures will appear in your output and are **not** caused by Phase 2A work.
- Phase 2A's own file surface is `src/elspeth/web/audit_readiness/` (new package) + `tests/integration/web/audit_readiness/` and `tests/unit/web/audit_readiness/` (new test directories). No overlap with the guided-session test surface.
- If you see **18+** failures, the 18th is on Phase 2A. If you see the same 17, the baseline is unchanged.
- If `elspeth-b1e04bea6d` ships before Phase 2A, this section becomes stale — delete it.

## Trust tier check (per CLAUDE.md)

- **Inbound:** `session_id` (path UUID parsed by FastAPI) + auth via `get_current_user`. No payload body.
- **Reads:** `CompositionState` via `state_from_record(record)` — **Tier 1**, direct typed access; no `.get()` defensive lookups.
- **Reads:** `CatalogService`, `WebSecretService.list_refs()`, `WebSettings` — **Tier 1**.
- **Composes:** the existing `ValidationResult` — **Tier 1** (we produced it). Direct attribute access.
- **Output:** strict Pydantic (`strict=True`, `extra="forbid"`) — drift crashes at construction.

## Plugin-trust derivation (load-bearing — DO NOT defer)

No `trust_tier` attribute exists on plugin classes; CLAUDE.md treats trust as per-data-flow. The audit panel collapses this onto per-component classification:

- Sources are uniformly `BOUNDARY` (their job is to cross Tier-3).
- Transforms are `BOUNDARY` only if on the closed list `EXTERNAL_BOUNDARY_TRANSFORMS`. Internal otherwise.
- Sinks are `BOUNDARY` only if on the closed list `EXTERNAL_BOUNDARY_SINKS`. Internal otherwise.

Closed lists are short by design. Adding a plugin that crosses Tier-3 requires updating `trust.py` in the same commit. The subset-of-catalog test in `test_trust.py` fails the build when an entry doesn't resolve to a registered plugin. **No fallback heuristic.**

## Retention row (load-bearing — DO NOT defer)

`payload_store.retention_days` is system-level (CLI default 90 in `cli.py:1356,1446`). No per-user surface exists. Phase 2A behaviour:

- Row `value` reports `WebSettings.payload_store_retention_days`.
- Row `status` is `not_applicable` (no user requirement to compare against).
- Row `summary` is `f"System retention: {days} days"`.

Phase 2A **does not** invent a phantom user-retention preference. A future phase that adds a per-composition retention surface flips this row to `ok`/`warning`.

## LLM-interpretations row (load-bearing — DO NOT defer)

Roadmap gates this behind Phase 5b (question B2). Phase 2A behaviour:

- Row always emitted with `status = not_applicable`.
- The "shown only if pipeline has LLM transforms" condition lives in **Phase 2B** as a frontend rendering rule. Backend is unconditional so Phase 5b flips one place.

## Six rows — projection mapping

| Row | Reads | Status logic |
|---|---|---|
| `validation` | `ValidationResult.is_valid`, `.errors` | `ok` if `is_valid`; else `error`. |
| `plugin_trust` | `CompositionState` + `trust.classify_plugin()` | `error` on unknown plugin name; else `ok` (boundary plugins listed in detail). |
| `provenance` | `ValidationResult.checks` filtered by `name == "identity_node_advisory"`; node ids from `check.affected_nodes` (structured field, not prose parse) | `warning` if any advisory; else `ok`. |
| `retention` | `WebSettings.payload_store_retention_days` | Always `not_applicable` in Phase 2A. |
| `llm_interpretations` | `CompositionState.nodes` (for the summary text only) | Always `not_applicable` in Phase 2A. |
| `secrets` | `ValidationResult.errors` filtered by `error_code` (structured discriminant, not substring match); `WebSecretService.list_refs()` | `error` on missing/fabricated refs; `ok` if refs and they resolve; `not_applicable` if no refs in the composition. |

## File structure

**New:**
- `src/elspeth/web/audit_readiness/{__init__.py, models.py, trust.py, service.py, explain.py, routes.py}`
- `tests/unit/web/audit_readiness/{__init__.py, test_models.py, test_trust.py, test_service.py, test_explain.py}`
- `tests/integration/web/test_audit_readiness_routes.py`

**Modified:**
- `src/elspeth/web/config.py` — add `payload_store_retention_days` to `WebSettings`.
- `src/elspeth/web/app.py` — wire `ReadinessService` to `app.state.readiness_service` and `include_router(create_audit_readiness_router())`.
- `src/elspeth/web/execution/routes.py` — import `verify_session_ownership` from `sessions/ownership.py`; drop local copy in the same commit.
- `src/elspeth/web/execution/schemas.py` (or equivalent) — add `error_code: str | None` to `ValidationError`; add `affected_nodes: tuple[str, ...]` to `ValidationCheck`. **No defaults on these fields — per CLAUDE.md No-Legacy policy, all call sites are updated in the same commit.** See Task 1 Step 3 for the co-update list.

**New (additional):**
- `src/elspeth/web/sessions/ownership.py` — shared `verify_session_ownership` (L3 peer).

---

## Task 1: Pydantic response models + Settings field

**Files:** `web/audit_readiness/__init__.py`, `web/audit_readiness/models.py`, `web/config.py`, `tests/unit/web/audit_readiness/__init__.py`, `tests/unit/web/audit_readiness/test_models.py`.

- [ ] **Step 1: Write the failing test**

Empty `tests/unit/web/audit_readiness/__init__.py` (one-line comment).

Create `tests/unit/web/audit_readiness/test_models.py`:

```python
"""Tests for audit-readiness Pydantic response models."""

from typing import get_args

import pytest
from pydantic import ValidationError

from elspeth.web.audit_readiness.models import (
    AuditReadinessExplain,
    AuditReadinessSnapshot,
    ReadinessRow,
    ReadinessRowId,
    ReadinessStatus,
)


def _row(row_id, status="ok"):
    return ReadinessRow(
        id=row_id, label=row_id, status=status, summary="x",
        detail=None, component_ids=(),
    )


def test_row_constructs_with_minimal_fields():
    row = _row("validation")
    assert row.status == "ok"


def test_row_rejects_unknown_id():
    with pytest.raises(ValidationError):
        ReadinessRow(id="kiosk", label="x", status="ok", summary="x",
                     detail=None, component_ids=())  # type: ignore[arg-type]


def test_row_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ReadinessRow(id="validation", label="x", status="purple",  # type: ignore[arg-type]
                     summary="x", detail=None, component_ids=())


def test_row_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ReadinessRow(id="validation", label="x", status="ok",
                     summary="x", detail=None, component_ids=(),
                     sneaky="oops")  # type: ignore[call-arg]


def test_snapshot_emits_six_canonical_rows():
    rows = tuple(_row(r) for r in (
        "validation", "plugin_trust", "provenance",
        "retention", "llm_interpretations", "secrets",
    ))
    snap = AuditReadinessSnapshot(
        session_id="11111111-1111-1111-1111-111111111111",
        composition_version=1, rows=rows,
    )
    assert {row.id for row in snap.rows} == set(get_args(ReadinessRowId))


def test_snapshot_rejects_duplicate_rows():
    rows = (_row("validation"), _row("validation"))
    with pytest.raises(ValidationError, match="duplicate"):
        AuditReadinessSnapshot(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1, rows=rows,
        )


def test_snapshot_rejects_missing_rows():
    rows = (_row("validation"),)
    with pytest.raises(ValidationError, match="missing"):
        AuditReadinessSnapshot(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1, rows=rows,
        )


def test_explain_constructs():
    ex = AuditReadinessExplain(
        session_id="11111111-1111-1111-1111-111111111111",
        composition_version=1,
        narrative="When you run this pipeline, ELSPETH will record:\n- foo",
    )
    assert "ELSPETH" in ex.narrative


def test_explain_rejects_empty_narrative():
    with pytest.raises(ValidationError):
        AuditReadinessExplain(
            session_id="11111111-1111-1111-1111-111111111111",
            composition_version=1, narrative="",
        )


def test_status_literal_closed_set():
    assert set(get_args(ReadinessStatus)) == {
        "ok", "warning", "error", "not_applicable",
    }
```

Run: `.venv/bin/python -m pytest tests/unit/web/audit_readiness/test_models.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 2: Implement**

`src/elspeth/web/audit_readiness/__init__.py`:

```python
"""Audit-readiness package — composition-time presentation of audit signals.

Read-only: no audit-trail writes happen here. Composes existing checks
(validation, catalog, secrets, retention) into a single panel snapshot.

Layer: L3 (application).
"""
```

`src/elspeth/web/audit_readiness/models.py`:

```python
"""Pydantic models for the audit-readiness endpoints."""

from __future__ import annotations

from typing import Literal, Self, get_args

from pydantic import Field, model_validator

from elspeth.web.execution.schemas import _StrictResponse

# Maps 1:1 to panel rows per docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md.
# Adding a row requires updating ReadinessService and Phase 2B's renderer.
ReadinessRowId = Literal[
    "validation",
    "plugin_trust",
    "provenance",
    "retention",
    "llm_interpretations",
    "secrets",
]

# Panel glyphs: ok→✓, warning→⚠, error→✗, not_applicable→—.
ReadinessStatus = Literal["ok", "warning", "error", "not_applicable"]

_EXPECTED_ROW_IDS: frozenset[str] = frozenset(get_args(ReadinessRowId))


class ReadinessRow(_StrictResponse):
    """One row in the audit-readiness panel."""

    id: ReadinessRowId
    label: str = Field(min_length=1)
    status: ReadinessStatus
    summary: str = Field(min_length=1)
    detail: str | None
    # component_ids let the frontend render jump-to-where links (Phase 2B).
    # Empty when the row is system-scoped (retention) or all-green.
    component_ids: tuple[str, ...]


class AuditReadinessSnapshot(_StrictResponse):
    """Aggregated payload for the audit-readiness panel."""

    session_id: str = Field(min_length=1)
    composition_version: int = Field(ge=1)
    rows: tuple[ReadinessRow, ...]

    @model_validator(mode="after")
    def _check_row_completeness(self) -> Self:
        ids = [row.id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate row ids in snapshot: {ids}")
        missing = _EXPECTED_ROW_IDS - set(ids)
        if missing:
            raise ValueError(f"snapshot missing required rows: {sorted(missing)}")
        extra = set(ids) - _EXPECTED_ROW_IDS
        if extra:
            raise ValueError(f"snapshot has unexpected rows: {sorted(extra)}")
        return self


class AuditReadinessExplain(_StrictResponse):
    """Narrative form for the Explain detail view."""

    session_id: str = Field(min_length=1)
    composition_version: int = Field(ge=1)
    narrative: str = Field(min_length=1)
```

In `src/elspeth/web/config.py`, after the existing `payload_store_path` field (around line 99), add:

```python
    payload_store_retention_days: int = Field(
        default=90,
        ge=1,
        description=(
            "Payload retention in days surfaced by the audit-readiness "
            "panel. Mirrors the CLI default (cli.py:1356). The panel "
            "row is informational only in Phase 2A — there is no "
            "user-stated requirement to compare against yet."
        ),
    )
```

Run tests → PASS.

- [ ] **Step 2b: Co-update schema fields + all call sites (No-Legacy — same commit)**

Add to `src/elspeth/web/execution/schemas.py`:

```python
class ValidationCheck(_StrictResponse):
    """Individual check result from dry-run validation."""

    name: str
    passed: bool
    detail: str
    # Structured field: node ids affected by this check (e.g. identity-node
    # advisories). Populated by the producer (validation.py) in the same
    # commit that adds this field — no compat-shim default.
    affected_nodes: tuple[str, ...]


class ValidationError(_StrictResponse):
    """Error with per-component attribution."""

    component_id: str | None
    component_type: str | None
    message: str
    suggestion: str | None
    # Structured discriminant for semantic error routing (e.g. "missing_secret_ref").
    # Populated at every construction site — no compat-shim default.
    error_code: str | None
```

In `src/elspeth/web/execution/validation.py`, locate the `_CHECK_IDENTITY_NODE_ADVISORY`
site (grep for `name=_CHECK_IDENTITY_NODE_ADVISORY`) and update the `ValidationCheck`
constructor to pass `affected_nodes`. (14a-1: line number removed — use the constant name
`_CHECK_IDENTITY_NODE_ADVISORY` as the stable locator.)

```python
checks.append(
    ValidationCheck(
        name=_CHECK_IDENTITY_NODE_ADVISORY,
        passed=True,
        detail=(...),
        affected_nodes=(identity_finding.node_id,),  # co-update: C1
    )
)
```

**Audit all other `ValidationCheck(...)` and `ValidationError(...)` construction
sites** and add the new required fields explicitly:

```bash
grep -rn "ValidationCheck(" src/ tests/
grep -rn "ValidationError(" src/ tests/
```

For `ValidationCheck` sites: pass `affected_nodes=()` unless the check is an
identity advisory (in which case pass the actual node ids). For `ValidationError`
sites: pass `error_code=None` unless the error has a semantic code (secret
errors get their code, other errors get `None`). These are not defaults — they
are explicit, auditor-visible values.

**CLAUDE.md No-Legacy requirement**: every call site is updated in this single
commit. Do not defer any call site to a later task.

- [ ] **Step 3: Commit**

The No-Legacy grep audit in Step 2b will surface call sites beyond the
models and schemas files. The full authoritative list of files requiring
co-update is:

```
# Sources
src/elspeth/web/audit_readiness/__init__.py
src/elspeth/web/audit_readiness/models.py
src/elspeth/web/config.py
src/elspeth/web/execution/schemas.py
src/elspeth/web/execution/validation.py        # identity_node_advisory site — pass affected_nodes=(identity_finding.node_id,)
src/elspeth/web/validation.py                  # identity_node_advisory site — load-bearing for provenance row
src/elspeth/web/execution/service.py           # ValidationCheck/ValidationError construction sites (14a-1: cite _CHECK_IDENTITY_NODE_ADVISORY, not line numbers)
src/elspeth/web/composer/service.py            # ValidationCheck/ValidationError construction sites

# Tests
tests/unit/web/audit_readiness/__init__.py
tests/unit/web/audit_readiness/test_models.py
tests/unit/web/execution/test_schemas.py       # 9+ sites
tests/unit/web/execution/test_routes.py        # 2 sites
tests/unit/web/composer/test_service.py        # 5+ sites
tests/unit/web/composer/test_tools.py          # ValidationCheck at line 7246, ValidationError at line 7253
tests/unit/web/sessions/test_routes.py         # 4 sites
tests/unit/web/composer/test_route_integration.py  # 1 site
```

> **Attention bias warning:** reviewing `validation.py` will make the
> `ValidationCheck`/`ValidationError` sites in `composer/service.py` and
> `execution/service.py` easy to overlook — both are in the list above and
> must be updated in this commit. (14a-1: line numbers removed; grep the files
> for `ValidationCheck(` and `ValidationError(` to locate the sites.)

Run mypy **before** `git add` — it is the only mechanical discriminant for
missing no-default required fields that tests may not exercise:

```bash
mypy src/elspeth/web/audit_readiness/ \
     src/elspeth/web/execution/ \
     src/elspeth/web/composer/ \
     src/elspeth/web/config.py \
     src/elspeth/web/validation.py \
     tests/unit/web/audit_readiness/ \
     tests/unit/web/execution/ \
     tests/unit/web/composer/ \
     tests/unit/web/sessions/
ruff check src/elspeth/web/audit_readiness/ src/elspeth/web/execution/schemas.py src/elspeth/web/config.py
ruff format --check src/elspeth/web/audit_readiness/ src/elspeth/web/execution/schemas.py src/elspeth/web/config.py
```

Stage files using the actual edit footprint after the grep-and-update pass:

```bash
git add $(git diff --name-only)
```

Verify the staged set matches the authoritative list above before committing.

```bash
git commit -m "feat(web): add audit-readiness response models + schema co-update (Phase 2A.1)"
```

### Schema rollout coordination (R2-W5)

`error_code` and `affected_nodes` are added without Python defaults. Every
call site must be updated in the same commit (CLAUDE.md No-Legacy).
This creates a deployment ordering constraint:

- **Backend (Phase 2A) MUST ship before frontend (Phase 2B) widens types**,
  OR both phases co-ship in a single deploy. Do not ship Phase 2B ahead of
  Phase 2A.
- **Operator must delete the sessions DB on staging** before deploying Phase
  2A. Project DB migration policy is: no Alembic, no migration scripts —
  the operator deletes the old DB. See project memory
  `project_db_migration_policy.md`.
- **Why a wipe is required:** `guided/audit.py:249` persists
  `dict(validation_result)`. Old persisted records lack `error_code` and
  `affected_nodes` fields; reading them back produces `KeyError` or
  deserialization failures. The wipe removes all stale records.
- **Cross-reference:** see `14-phase-2-audit-readiness-panel.md` (umbrella
  plan) for the unified rollout note shared across phases.

## Task 2: Plugin-trust classifier (closed allowlists)

**Files:** `web/audit_readiness/trust.py`, `tests/unit/web/audit_readiness/test_trust.py`.

The closed allowlists are the load-bearing rule. Plugin renames without an allowlist update fail the subset-of-catalog test.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for plugin-trust classification."""

from __future__ import annotations

import pytest

from elspeth.web.audit_readiness.trust import (
    EXTERNAL_BOUNDARY_SINKS,
    EXTERNAL_BOUNDARY_TRANSFORMS,
    PluginTrust,
    classify_plugin,
)


def test_source_kind_always_boundary():
    assert classify_plugin("source", "csv") is PluginTrust.BOUNDARY
    assert classify_plugin("source", "json") is PluginTrust.BOUNDARY


def test_external_call_transforms_are_boundary():
    for name in EXTERNAL_BOUNDARY_TRANSFORMS:
        assert classify_plugin("transform", name) is PluginTrust.BOUNDARY


def test_internal_transforms_are_internal():
    assert classify_plugin("transform", "passthrough") is PluginTrust.INTERNAL


def test_external_sinks_are_boundary():
    for name in EXTERNAL_BOUNDARY_SINKS:
        assert classify_plugin("sink", name) is PluginTrust.BOUNDARY


def test_internal_sinks_are_internal():
    assert classify_plugin("sink", "csv") is PluginTrust.INTERNAL


def test_unknown_plugin_kind_raises():
    with pytest.raises(ValueError, match="unknown plugin kind"):
        classify_plugin("gate", "anything")  # type: ignore[arg-type]


def test_external_boundary_transforms_subset_of_catalog():
    """Allowlist drift guard: every entry must resolve via the live catalog."""
    from elspeth.plugins.infrastructure.manager import PluginManager

    pm = PluginManager()
    pm.register_builtin_plugins()
    transform_names = {cls.name for cls in pm.get_transforms()}
    missing = EXTERNAL_BOUNDARY_TRANSFORMS - transform_names
    assert not missing, (
        f"EXTERNAL_BOUNDARY_TRANSFORMS has unregistered plugins: "
        f"{sorted(missing)}. Either register the plugin or drop the entry."
    )


def test_external_boundary_sinks_subset_of_catalog():
    from elspeth.plugins.infrastructure.manager import PluginManager

    pm = PluginManager()
    pm.register_builtin_plugins()
    sink_names = {cls.name for cls in pm.get_sinks()}
    missing = EXTERNAL_BOUNDARY_SINKS - sink_names
    assert not missing, (
        f"EXTERNAL_BOUNDARY_SINKS has unregistered plugins: "
        f"{sorted(missing)}. Either register the plugin or drop the entry."
    )


def test_every_external_call_plugin_is_on_allowlist_or_explicitly_excepted():
    """Every Determinism.EXTERNAL_CALL plugin must be on an allowlist or EXTERNAL_CALL_EXCEPTIONS.

    This test FAILS when a new external-call plugin is added without being
    categorised. Keep EXTERNAL_CALL_EXCEPTIONS empty; populate only with
    an explicit written justification.
    """
    from elspeth.plugins.infrastructure.manager import PluginManager
    from elspeth.contracts.enums import Determinism  # Determinism lives in contracts/enums.py (StrEnum, EXTERNAL_CALL = "external_call")

    EXTERNAL_CALL_EXCEPTIONS: frozenset[str] = frozenset()  # noqa: N806
    pm = PluginManager()
    pm.register_builtin_plugins()
    external_call_plugins = {
        cls.name for cls in list(pm.get_transforms()) + list(pm.get_sinks())
        if cls.determinism is Determinism.EXTERNAL_CALL  # offensive: attribute must exist per plugin-ownership doctrine
    }
    covered = EXTERNAL_BOUNDARY_TRANSFORMS | EXTERNAL_BOUNDARY_SINKS | EXTERNAL_CALL_EXCEPTIONS
    uncategorised = external_call_plugins - covered
    assert not uncategorised, (
        f"External-call plugins not categorised for audit-readiness: "
        f"{sorted(uncategorised)}. Add to an allowlist or EXTERNAL_CALL_EXCEPTIONS."
    )
```

Run → FAIL (ModuleNotFoundError).

- [ ] **Step 2: Preflight — confirm plugin names in the live catalog BEFORE implementing**

**Run these greps and record their output. Populate the allowlists only with
names that appear in the grep output.** Do not invent names — the subset-of-catalog
tests fail on first run if an entry does not resolve to a registered plugin.

```bash
# Step 2a: List all transform plugin names registered via hookimpl
grep -rn "@hookimpl.*register\|def register" src/elspeth/plugins/transforms/ | grep -v __pycache__

# Step 2b: List every transform and sink .name attribute
grep -rn 'name = "\|name: ClassVar.*=' src/elspeth/plugins/transforms --include="*.py" | grep -v __pycache__
grep -rn 'name = "\|name: ClassVar.*=' src/elspeth/plugins/sinks --include="*.py" | grep -v __pycache__

# Step 2c: Identify transforms with Determinism.EXTERNAL_CALL (these MUST be on allowlist or exceptions)
grep -rn "Determinism\." src/elspeth/plugins/transforms --include="*.py" | grep -v __pycache__ | sort
grep -rn "Determinism\." src/elspeth/plugins/sinks --include="*.py" | grep -v __pycache__ | sort

# Step 2d: Confirm external HTTP / API / LLM call points (corroboration)
grep -rn "external_call\|api_call\|llm_call\|http\|requests" src/elspeth/plugins/transforms/ | grep -v test_ | grep -v __pycache__ | grep -i "class\|determinism" | head -30
```

**Expected results at RC5.2 HEAD** (verified 2026-05-16 — update if catalog changes):

`EXTERNAL_CALL` transforms: `web_scrape` (web_scrape.py:339), `rag_retrieval`
(rag/transform.py:55), `azure_content_safety` (azure/content_safety.py:122),
`azure_prompt_shield` (azure/prompt_shield.py:96). The `llm` transform is
`NON_DETERMINISTIC` (not `EXTERNAL_CALL`) and is manually curated onto the
allowlist because it crosses an LLM API boundary.

`EXTERNAL_CALL` sinks: `dataverse` (sinks/dataverse.py:224).

Phantom names that must NOT appear in the allowlist (they do not exist in the
catalog): `web_fetch`, `web_scrape_extraction`, `dataverse_query`.

- [ ] **Step 3: Implement**

`src/elspeth/web/audit_readiness/trust.py`:

```python
"""Plugin-trust classification for the audit-readiness panel.

CLAUDE.md treats trust as a per-data-flow doctrine: sources cross Tier-3
(external input), transforms that make external calls (HTTP, LLM, blob
store) cross Tier-3 at the call boundary, everything else is Tier-2.

This module collapses that into a per-component classification:

  - BOUNDARY: crosses Tier-3. Sources are uniformly BOUNDARY; transforms
    and sinks are BOUNDARY only when on the closed allowlists below.
  - INTERNAL: operates only on pipeline data.

The allowlists are closed by design. Adding a plugin that crosses Tier-3
requires updating this file in the same commit. The subset-of-catalog
tests fail the build when an entry doesn't resolve to a registered
plugin — a rename without an update fails CI rather than silently
breaking the panel.

Two-tier rationale:
  - EXTERNAL_CALL determinism (automated): web_scrape, rag_retrieval,
    azure_content_safety, azure_prompt_shield are Determinism.EXTERNAL_CALL.
    The completeness test (test_every_external_call_plugin_is_on_allowlist_or_explicitly_excepted)
    catches new EXTERNAL_CALL plugins added without allowlist update.
  - Manual curation (LLM-class): "llm" is Determinism.NON_DETERMINISTIC
    but crosses an LLM API boundary and must be visible to auditors as
    BOUNDARY. Manually curated; not caught by the completeness test.
    Document any future additions to this tier with an explicit comment.

Layer: L3 (application).

Phase 7 deletion commitment (required verbatim — do not paraphrase):
When Phase 7 adds `data_trust_tier: ClassVar` to plugin base classes,
delete this module entirely and replace `classify_plugin()` callers with
direct attribute lookup. CLAUDE.md No-Legacy requires same-commit
replacement.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

PluginKind = Literal["source", "transform", "sink"]


class PluginTrust(StrEnum):
    BOUNDARY = "boundary"
    INTERNAL = "internal"


# Transform plugins that cross an external boundary (Tier-3).
# Two categories — see module docstring two-tier rationale:
#   (a) Determinism.EXTERNAL_CALL: web_scrape, rag_retrieval,
#       azure_content_safety, azure_prompt_shield (verified via grep).
#   (b) Manual curation (NON_DETERMINISTIC + LLM boundary): llm.
# Adding a new entry:
#   1. Add the plugin's `.name` value here (verified via grep, not guessed).
#   2. Confirm the plugin module documents the external surface.
#   3. The subset-of-catalog test in test_trust.py validates the name is real.
EXTERNAL_BOUNDARY_TRANSFORMS: frozenset[str] = frozenset({
    "llm",                   # plugins/transforms/llm/transform.py — NON_DETERMINISTIC, manually curated (LLM API boundary)
    "web_scrape",            # plugins/transforms/web_scrape.py — Determinism.EXTERNAL_CALL
    "rag_retrieval",         # plugins/transforms/rag/transform.py — Determinism.EXTERNAL_CALL
    "azure_content_safety",  # plugins/transforms/azure/content_safety.py — Determinism.EXTERNAL_CALL
    "azure_prompt_shield",   # plugins/transforms/azure/prompt_shield.py — Determinism.EXTERNAL_CALL
})

# Sink plugins that write to external systems (Determinism.EXTERNAL_CALL).
# Adding a new entry: follow the same 3-step process as transforms above.
EXTERNAL_BOUNDARY_SINKS: frozenset[str] = frozenset({
    "dataverse",             # plugins/sinks/dataverse.py — Determinism.EXTERNAL_CALL
})


def classify_plugin(kind: PluginKind, name: str) -> PluginTrust:
    """Classify a plugin by kind + name.

    Raises:
        ValueError: when ``kind`` is not one of the three known values.
            Tier-1 invariant — the aggregator dispatches kinds taken from
            the typed CompositionState.
    """
    if kind == "source":
        return PluginTrust.BOUNDARY
    if kind == "transform":
        return (
            PluginTrust.BOUNDARY
            if name in EXTERNAL_BOUNDARY_TRANSFORMS
            else PluginTrust.INTERNAL
        )
    if kind == "sink":
        return (
            PluginTrust.BOUNDARY
            if name in EXTERNAL_BOUNDARY_SINKS
            else PluginTrust.INTERNAL
        )
    raise ValueError(f"unknown plugin kind: {kind!r}")
```

Run tests → PASS. If the subset-of-catalog tests fail, drop the offending entries from the allowlists and re-run the Step 2 greps to find the correct names.

> **R2-W8 — Phase 7 deletion commitment:** The "Phase 7 deletion commitment" paragraph in the docstring above is **required verbatim — do not paraphrase**. A filigree dependency must link Phase 7A backend → this deletion before Phase 7 implementation begins; file it when Phase 7 is scoped.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/audit_readiness/trust.py tests/unit/web/audit_readiness/test_trust.py
mypy src/elspeth/web/audit_readiness/trust.py
ruff check src/elspeth/web/audit_readiness/trust.py
ruff format --check src/elspeth/web/audit_readiness/trust.py
git commit -m "feat(web): add closed-list plugin-trust classifier (Phase 2A.2)"
```

## Task 3: ReadinessService — compose existing signals

**Files:** `web/audit_readiness/service.py`, `tests/unit/web/audit_readiness/test_service.py`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for ReadinessService."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from elspeth.web.audit_readiness.service import ReadinessService
from elspeth.web.composer.state import (
    CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec,
)
from elspeth.web.execution.schemas import (
    ValidationCheck, ValidationError, ValidationResult,
)

# ── Test factories ────────────────────────────────────────────────────────────
# Co-located here; if this conftest grows, extract to
# tests/integration/web/audit_readiness/conftest.py.
#
# NodeSpec has 13 required fields + 3 defaulted (trigger, output_mode,
# expected_output_count). OutputSpec has 4 required fields (name, plugin,
# options, on_write_failure — all required, no defaults).
# These factories cover ALL required kwargs so tests never TypeError at
# construction time (review B1, B2).
#
# Cross-reference: the identical factories must be used in test_explain.py
# and in any other test module that constructs NodeSpec/OutputSpec inline.

def make_node_spec(
    nid: str,
    plugin: str | None,
    *,
    input: str = "src_out",
    on_success: str | None = "out",
    node_type: str = "transform",
) -> "NodeSpec":
    """Factory for NodeSpec covering all 13 required fields.

    Required-but-structural fields (on_error, condition, routes, fork_to,
    branches, policy, merge) are passed as None — they are required kwargs but
    are None for standard transform nodes.
    """
    return NodeSpec(
        id=nid,
        node_type=node_type,
        plugin=plugin,
        input=input,
        on_success=on_success,
        on_error=None,
        options={},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def make_output_spec(name: str, plugin: str) -> "OutputSpec":
    """Factory for OutputSpec covering all 4 required fields.

    on_write_failure defaults to "discard" — the canonical safe choice.
    """
    return OutputSpec(name=name, plugin=plugin, options={}, on_write_failure="discard")


def _state(*, source_plugin="csv", transforms=(), sinks=(("out", "csv"),)):
    src = (
        SourceSpec(plugin=source_plugin, on_success="src_out",
                   options={}, on_validation_failure="quarantine")
        if source_plugin is not None else None
    )
    nodes = tuple(
        make_node_spec(
            nid, plg,
            input="src_out" if i == 0 else f"t{i - 1}_out",
            on_success=f"t{i}_out",
        )
        for i, (nid, plg) in enumerate(transforms)
    )
    outputs = tuple(make_output_spec(n, p) for n, p in sinks)
    return CompositionState(
        source=src, nodes=nodes, edges=(), outputs=outputs,
        metadata=PipelineMetadata(name="t", description=""), version=1,
    )


def _make_service(state, validation_result, inventory=()):
    exec_svc = MagicMock()
    exec_svc.validate = AsyncMock(return_value=validation_result)
    sess_svc = MagicMock()
    record = MagicMock()
    sess_svc.get_current_state = AsyncMock(return_value=record)
    # Use scoped_secret_resolver mock (list_refs(user_id) only — no auth_provider_type).
    # Matches app.py:470 precedent and the _SecretServiceLike Protocol (fix C4).
    scoped_resolver = MagicMock()
    scoped_resolver.list_refs = MagicMock(return_value=list(inventory))
    settings = MagicMock()
    settings.payload_store_retention_days = 90
    return ReadinessService(
        execution_service=exec_svc, session_service=sess_svc,
        secret_service=scoped_resolver, settings=settings,
        state_from_record=lambda _record: state,
    )


def _row(snap, row_id):
    matches = [r for r in snap.rows if r.id == row_id]
    if not matches:
        raise AssertionError(f"row {row_id!r} not in snapshot")
    return matches[0]


_OK = ValidationResult(is_valid=True, checks=[], errors=[], semantic_contracts=[])


def test_validation_row_ok_when_no_errors():
    svc = _make_service(_state(transforms=(("t", "passthrough"),)), _OK)
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    assert _row(snap, "validation").status == "ok"


def test_validation_row_error_lists_component_ids():
    result = ValidationResult(
        is_valid=False, checks=[],
        errors=[ValidationError(component_id="out", component_type="sink",
                                message="boom", suggestion=None, error_code=None)],
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    row = _row(snap, "validation")
    assert row.status == "error"
    assert row.component_ids == ("out",)


def test_plugin_trust_row_ok_when_all_internal():
    svc = _make_service(
        _state(transforms=(("t", "passthrough"),), sinks=(("out", "csv"),)), _OK,
    )
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    assert _row(snap, "plugin_trust").status == "ok"


def test_provenance_warning_on_identity_advisory():
    result = ValidationResult(
        is_valid=True,
        checks=[ValidationCheck(
            name="identity_node_advisory", passed=True,
            detail="Node 'pass' is an identity-shaped passthrough between 'source' and sink 'out'.",
            affected_nodes=("pass",),  # structured field; no prose parse needed
        )],
        errors=[], semantic_contracts=[],
    )
    svc = _make_service(_state(transforms=(("pass", "passthrough"),)), result)
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    row = _row(snap, "provenance")
    assert row.status == "warning"
    assert "pass" in (row.detail or "")
    assert "pass" in row.component_ids


def test_retention_row_reports_system_value():
    svc = _make_service(_state(), _OK)
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    row = _row(snap, "retention")
    assert row.status == "not_applicable"
    assert "90" in row.summary


def test_llm_interpretations_always_not_applicable_in_phase_2a():
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK)
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    assert _row(snap, "llm_interpretations").status == "not_applicable"


def test_secrets_not_applicable_when_no_refs():
    svc = _make_service(_state(), _OK, inventory=())
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    assert _row(snap, "secrets").status == "not_applicable"


def test_secrets_error_on_missing_refs():
    result = ValidationResult(
        is_valid=False,
        checks=[ValidationCheck(name="secret_refs", passed=False,
                                detail="Missing secret references: openai_key",
                                affected_nodes=())],  # no node attribution for secret check
        errors=[ValidationError(
            component_id=None, component_type=None,
            message="Cannot resolve secret references: openai_key",
            suggestion="Add via Secrets panel.",
            error_code="missing_secret_ref",  # structured discriminant
        )],
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(svc.compute_snapshot(
        session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
    ))
    assert _row(snap, "secrets").status == "error"


def test_snapshot_raises_when_no_state():
    exec_svc = MagicMock()
    exec_svc.validate = AsyncMock(return_value=_OK)
    sess_svc = MagicMock()
    sess_svc.get_current_state = AsyncMock(return_value=None)
    scoped_resolver = MagicMock()
    scoped_resolver.list_refs = MagicMock(return_value=[])
    settings = MagicMock(payload_store_retention_days=90)
    svc = ReadinessService(
        execution_service=exec_svc, session_service=sess_svc,
        secret_service=scoped_resolver, settings=settings,
    )
    with pytest.raises(LookupError, match="no composition state"):
        asyncio.run(svc.compute_snapshot(
            session_id="11111111-1111-1111-1111-111111111111", user_id="alice",
        ))
```

Run → FAIL (ModuleNotFoundError).

- [ ] **Step 2: Implement**

`src/elspeth/web/audit_readiness/service.py`:

```python
"""ReadinessService — pure aggregation of audit signals into a snapshot.
No new validation logic. Layer: L3 (application).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.audit_readiness.models import (
    AuditReadinessSnapshot, ReadinessRow,
)
from elspeth.web.audit_readiness.trust import (
    EXTERNAL_BOUNDARY_SINKS, EXTERNAL_BOUNDARY_TRANSFORMS,
    PluginTrust, classify_plugin,
)
from elspeth.web.composer.state import CompositionState
from elspeth.web.execution.schemas import (
    ValidationCheck, ValidationError, ValidationResult,
)
from elspeth.web.sessions.converters import state_from_record as _default_state_from_record

# Mirror of validation.py's private constant — duplicated to keep the
# dependency unidirectional (audit_readiness depends on the result shape,
# not on validation's internal naming).
_CHECK_IDENTITY_NODE_ADVISORY = "identity_node_advisory"


class _ExecutionServiceLike(Protocol):
    # session_id is UUID to match ExecutionServiceImpl.validate (web/execution/service.py:638);
    # the route layer constructs it from the FastAPI path-param parser.
    # Amendment 14a-2 (backend-006): tightened from str to UUID in landed code.
    async def validate(self, session_id: UUID, *, user_id: str | None = None) -> ValidationResult: ...


class _SessionServiceLike(Protocol):
    # session_id is UUID to match SessionServiceImpl.get_current_state (web/sessions/service.py:1884).
    # Amendment 14a-2 (backend-006): tightened from untyped to UUID in landed code.
    async def get_current_state(self, session_id: UUID) -> Any: ...


class _SecretServiceLike(Protocol):
    # Matches ScopedSecretResolver.list_refs (service.py:312) — no auth_provider_type.
    # The scoped resolver already has auth_provider_type baked in at construction
    # (app.py:470: ScopedSecretResolver(service, settings.auth_provider)).
    # Inject app.state.scoped_secret_resolver, not app.state.secret_service.
    def list_refs(self, user_id: str) -> list[SecretInventoryItem]: ...  # sync; caller uses run_sync_in_worker


class _SettingsLike(Protocol):
    payload_store_retention_days: int


class ReadinessService:
    """Compose audit-readiness signals into a snapshot for the panel."""

    def __init__(
        self, *,
        execution_service: _ExecutionServiceLike,
        session_service: _SessionServiceLike,
        secret_service: _SecretServiceLike,
        settings: _SettingsLike,
        state_from_record: Callable[..., CompositionState] | None = None,
    ) -> None:
        self._execution_service = execution_service
        self._session_service = session_service
        self._secret_service = secret_service
        self._settings = settings
        self._state_from_record: Callable[..., CompositionState] = (
            state_from_record if state_from_record is not None
            else _default_state_from_record
        )

    async def compute_snapshot(
        self, *, session_id: UUID, user_id: str,  # Amendment 14a-2: str→UUID (backend-006)
    ) -> AuditReadinessSnapshot:
        """Return the six-row snapshot.

        Raises:
            LookupError: when the session has no composition state. The
                route layer translates this into a 404.
        """
        record = await self._session_service.get_current_state(session_id)
        if record is None:
            raise LookupError(f"no composition state for session {session_id!r}")
        state: CompositionState = self._state_from_record(record)
        validation = await self._execution_service.validate(session_id, user_id=user_id)
        inventory = await run_sync_in_worker(
            self._secret_service.list_refs,
            user_id,  # scoped_secret_resolver.list_refs takes user_id only
        )
        rows: tuple[ReadinessRow, ...] = (
            _build_validation_row(validation),
            _build_plugin_trust_row(state),
            _build_provenance_row(validation),
            _build_retention_row(self._settings.payload_store_retention_days),
            _build_llm_interpretations_row(state),
            _build_secrets_row(validation, inventory),
        )
        # Amendment 14a-2 (backend-006): session_id is UUID; model's Field(min_length=1)
        # accepts the canonical 36-char UUID string representation.
        return AuditReadinessSnapshot(
            session_id=str(session_id),
            composition_version=state.version,
            rows=rows,
        )


# ── Row projections ───────────────────────────────────────────────


def _build_validation_row(result: ValidationResult) -> ReadinessRow:
    if result.is_valid:
        return ReadinessRow(
            id="validation", label="Validation", status="ok",
            summary="All checks pass", detail=None, component_ids=(),
        )
    component_ids = tuple(sorted({
        err.component_id for err in result.errors if err.component_id is not None
    }))
    summary = (
        f"{len(result.errors)} errors — see details"
        if len(result.errors) != 1 else "1 error — see details"
    )
    detail = "\n".join(
        f"[{err.component_type or 'unknown'}] {err.component_id or 'unknown'}: {err.message}"
        for err in result.errors
    )
    return ReadinessRow(
        id="validation", label="Validation", status="error",
        summary=summary, detail=detail, component_ids=component_ids,
    )


def _build_plugin_trust_row(state: CompositionState) -> ReadinessRow:
    """Classify every plugin in the composition (boundary vs internal).
    Panel uses "Plugin trust" vocabulary; tier numbers belong in the Explain view.
    """
    boundary: list[tuple[str, str, str]] = []
    unknown: list[tuple[str, str]] = []

    def _record(kind: str, component_id: str, name: str | None) -> None:
        if name is None:
            unknown.append((kind, component_id))
            return
        trust = classify_plugin(kind, name)  # type: ignore[arg-type]
        if trust is PluginTrust.BOUNDARY:
            boundary.append((kind, component_id, name))

    if state.source is not None:
        _record("source", "source", state.source.plugin)
    for node in state.nodes:
        if node.node_type == "transform":
            _record("transform", node.id, node.plugin)
    for output in state.outputs:
        _record("sink", output.name, output.plugin)

    if unknown:
        ids = tuple(sorted({cid for _, cid in unknown}))
        return ReadinessRow(
            id="plugin_trust", label="Plugin trust", status="error",
            summary="Unknown plugin in composition",
            detail=("The composition references plugin names not in the "
                    "registered catalog. Validation will block execution."),
            component_ids=ids,
        )

    if not boundary:
        return ReadinessRow(
            id="plugin_trust", label="Plugin trust", status="ok",
            summary="All plugins operate on pipeline data",
            detail=None, component_ids=(),
        )

    # Boundary plugins are recorded as ok with the boundaries named in
    # detail. The "sensitivity-vs-tier mismatch" warning case needs a
    # user-stated sensitivity surface that does not yet exist (roadmap §G2).
    detail = "\n".join(
        f"- [{kind}] {cid} ({name}) — crosses an external boundary"
        for kind, cid, name in boundary
    )
    return ReadinessRow(
        id="plugin_trust", label="Plugin trust", status="ok",
        summary=f"{len(boundary)} external-boundary plugin(s) recorded",
        detail=detail,
        component_ids=tuple(cid for _, cid, _ in boundary),
    )


def _build_provenance_row(result: ValidationResult) -> ReadinessRow:
    """Project identity_node_advisory checks into the provenance row.

    Node ids come from check.affected_nodes (structured tuple added by
    Finding 2). No prose parsing of the detail field.
    """
    advisories = [
        c for c in result.checks if c.name == _CHECK_IDENTITY_NODE_ADVISORY
    ]
    if not advisories:
        return ReadinessRow(
            id="provenance", label="Provenance", status="ok",
            summary="All paths record provenance",
            detail=None, component_ids=(),
        )
    component_ids = tuple(
        node_id
        for c in advisories
        for node_id in c.affected_nodes  # structured field; no prose parse
    )
    return ReadinessRow(
        id="provenance", label="Provenance", status="warning",
        summary=f"{len(advisories)} identity passthrough(s) — provenance gap",
        detail="\n".join(c.detail for c in advisories),
        component_ids=component_ids,
    )


def _build_retention_row(retention_days: int) -> ReadinessRow:
    """System-configured; no user requirement to compare against."""
    return ReadinessRow(
        id="retention", label="Retention", status="not_applicable",
        summary=f"System retention: {retention_days} days",
        detail=("Per-composition retention configuration is not yet "
                "exposed; configured retention applies to all payloads."),
        component_ids=(),
    )


def _build_llm_interpretations_row(state: CompositionState) -> ReadinessRow:
    """Always not_applicable in Phase 2A; Phase 5b implements the real signal."""
    has_llm = any(
        n.node_type == "transform" and n.plugin == "llm" for n in state.nodes
    )
    summary = (
        "Interpretation surface not yet available (Phase 5b)"
        if has_llm else "No LLM transforms in this composition"
    )
    return ReadinessRow(
        id="llm_interpretations", label="LLM interpretations",
        status="not_applicable", summary=summary,
        detail=None, component_ids=(),
    )


_SECRET_ERROR_CODES: frozenset[str] = frozenset({
    "missing_secret_ref", "fabricated_secret_ref", "disallowed_secret_ref",
})


def _build_secrets_row(
    validation: ValidationResult, inventory: list[SecretInventoryItem],
) -> ReadinessRow:
    """error/ok/not_applicable per secret ref resolution.
    Keyed on ValidationError.error_code, not message substring.
    """
    secret_errors = [
        err for err in validation.errors
        if err.error_code in _SECRET_ERROR_CODES
    ]
    if secret_errors:
        return ReadinessRow(
            id="secrets", label="Secrets", status="error",
            summary="Secret references unresolved",
            detail="\n".join(err.message for err in secret_errors),
            component_ids=tuple(
                err.component_id for err in secret_errors if err.component_id
            ),
        )
    secret_check = next(
        (c for c in validation.checks if c.name == "secret_refs"), None,
    )
    if secret_check is None and not inventory:
        return ReadinessRow(
            id="secrets", label="Secrets", status="not_applicable",
            summary="No secret references in this composition",
            detail=None, component_ids=(),
        )
    return ReadinessRow(
        id="secrets", label="Secrets", status="ok",
        summary="All secret references resolve",
        detail=(f"{len(inventory)} secret(s) in your inventory"
                if inventory else "Composition references no secrets"),
        component_ids=(),
    )
```

Run tests → PASS.

- [ ] **Step 2b: Integration test — validate_pipeline path + secret resolver (Fix C1, C4)**

Add these integration tests to `tests/integration/web/test_audit_readiness_routes.py`
(same file as Task 5 — the route integration tests cover sessions with state,
so these guard tests colocate naturally there rather than in a separate file).

**Prerequisite:** Task 5 Step 2.0 scaffolds the integration fixtures
(`audit_readiness_test_client`, `audit_readiness_client_with_state`, etc.) in
`tests/integration/web/conftest.py`. If running tasks in order, complete Task 5
Step 2.0 before writing these tests. The existing `composer_test_client` fixture
in that conftest is NOT sufficient — it sets `execution_service=None` and
`scoped_secret_resolver=None`, both of which are required by the audit-readiness
routes. Do not use `composer_test_client` directly for these tests.

```python
"""C1/C4 guard tests — colocated in test_audit_readiness_routes.py."""


def test_provenance_row_component_ids_populated_via_real_validate_pipeline(
    audit_readiness_client_with_state,
):
    """C1 guard: affected_nodes wired in validation.py must propagate to component_ids.

    If the `_CHECK_IDENTITY_NODE_ADVISORY` ValidationCheck site in validation.py
    does not pass affected_nodes, this assertion will fail even if the unit
    test passes (because the unit test supplies affected_nodes manually).
    """
    client, session_id = audit_readiness_client_with_state
    # The session fixture must include a passthrough node (triggers the advisory).
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    rows = {r["id"]: r for r in response.json()["rows"]}
    # If provenance status is "warning", component_ids must be non-empty.
    if rows["provenance"]["status"] == "warning":
        assert rows["provenance"]["component_ids"], (
            "provenance row status is 'warning' but component_ids is empty — "
            "the _CHECK_IDENTITY_NODE_ADVISORY site in validation.py must pass "
            "affected_nodes=(node_id,) to ValidationCheck"
        )


def test_secrets_row_uses_scoped_resolver(audit_readiness_client_with_state):
    """C4 guard: ReadinessService must call list_refs via scoped_secret_resolver.

    If wired to the raw secret_service (which requires auth_provider_type),
    this will TypeError at runtime rather than in unit tests.
    """
    client, session_id = audit_readiness_client_with_state
    # A session whose composition references a secret ref exercises the resolver.
    # Wire the secret ref into the session fixture's CompositionState, or add
    # a separate fixture variant with a secret ref pre-populated.
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    rows = {r["id"]: r for r in response.json()["rows"]}
    # A session with a secret ref that resolves → ok; unresolved → error.
    # Either is acceptable; what must NOT happen is a 500 from a TypeError.
    assert rows["secrets"]["status"] in ("ok", "error", "not_applicable")
```

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/audit_readiness/service.py \
        tests/unit/web/audit_readiness/test_service.py \
        tests/integration/web/audit_readiness/test_readiness_service_integration.py
mypy src/elspeth/web/audit_readiness/service.py
ruff check src/elspeth/web/audit_readiness/service.py
ruff format --check src/elspeth/web/audit_readiness/service.py
git commit -m "feat(web): add ReadinessService composing audit signals (Phase 2A.3)"
```

## Task 4: Explain narrative builder

**Files:** `web/audit_readiness/explain.py`, `tests/unit/web/audit_readiness/test_explain.py`.

Deterministic stringification of CompositionState — no LLM call.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the Explain narrative builder."""

from __future__ import annotations

from elspeth.web.audit_readiness.explain import build_narrative
from elspeth.web.composer.state import (
    CompositionState, PipelineMetadata, SourceSpec,
)

# Import shared factories (co-located in test_service.py or extracted to
# tests/unit/web/audit_readiness/conftest.py if this module grows).
# These factories cover ALL required NodeSpec/OutputSpec kwargs (review B1, B2).
from tests.unit.web.audit_readiness.test_service import make_node_spec, make_output_spec


def _state(*, source_plugin="csv", transforms=(), sinks=(("out", "csv"),)):
    src = (
        SourceSpec(plugin=source_plugin, on_success="src_out",
                   options={}, on_validation_failure="quarantine")
        if source_plugin is not None else None
    )
    nodes = tuple(
        make_node_spec(
            nid, plg,
            input="src_out" if i == 0 else f"t{i - 1}_out",
            on_success=f"t{i}_out",
        )
        for i, (nid, plg) in enumerate(transforms)
    )
    outputs = tuple(make_output_spec(n, p) for n, p in sinks)
    return CompositionState(
        source=src, nodes=nodes, edges=(), outputs=outputs,
        metadata=PipelineMetadata(name="t", description=""), version=1,
    )


def test_opens_with_recorded_promise():
    text = build_narrative(_state(), retention_days=90)
    assert text.startswith("When you run this pipeline, ELSPETH will record:")


def test_names_source_plugin():
    text = build_narrative(_state(source_plugin="csv"), retention_days=90)
    assert "csv" in text.lower()


def test_walks_transforms_in_order():
    text = build_narrative(
        _state(transforms=(("t1", "passthrough"), ("t2", "llm"))),
        retention_days=90,
    )
    assert text.index("t1") < text.index("t2")


def test_calls_out_llm_recording_details():
    text = build_narrative(_state(transforms=(("judge", "llm"),)), retention_days=90)
    assert "prompt" in text.lower()
    assert "response" in text.lower() or "model" in text.lower()


def test_includes_each_sink():
    text = build_narrative(
        _state(sinks=(("primary", "csv"), ("backup", "json"))), retention_days=90,
    )
    assert "primary" in text and "backup" in text


def test_mentions_retention():
    text = build_narrative(_state(), retention_days=42)
    assert "42" in text


def test_is_deterministic():
    s = _state(transforms=(("t", "passthrough"),))
    assert build_narrative(s, retention_days=90) == build_narrative(s, retention_days=90)


def test_no_source_explains_incomplete():
    text = build_narrative(_state(source_plugin=None), retention_days=90)
    assert "no source" in text.lower() or "incomplete" in text.lower()


def test_closes_with_evidence_promise():
    text = build_narrative(_state(), retention_days=90)
    assert "evidence" in text.lower() or "answer" in text.lower()
```

Run → FAIL (ModuleNotFoundError).

- [ ] **Step 2: Implement**

`src/elspeth/web/audit_readiness/explain.py`:

```python
"""Narrative builder for the audit-readiness Explain view.

Generates deterministic prose describing what ELSPETH will record when
the composition runs. No LLM call; same composition + retention → same text.

Layer: L3 (application).
"""

from __future__ import annotations

from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec


def build_narrative(state: CompositionState, *, retention_days: int) -> str:
    lines: list[str] = [
        "When you run this pipeline, ELSPETH will record:", "",
    ]

    if state.source is None:
        lines.append(
            "- No source configured yet — the composition is incomplete. "
            "Once you add a source, this view will describe what it records."
        )
    else:
        lines.append(_describe_source(state.source.plugin))

    for node in state.nodes:
        if node.node_type == "transform":
            lines.append(_describe_transform(node))

    for output in state.outputs:
        lines.append(_describe_output(output))

    lines.extend([
        "",
        f"Retention: {retention_days} days by default. This applies to "
        "stored payloads; row-level hashes are retained indefinitely.",
        "",
        "Run metadata: when, who (you), and which plugin versions were "
        "in use at run time.",
        "",
        "This evidence is sufficient to answer questions about any output "
        "row of this pipeline, including which plugin produced it and "
        "from what input.",
    ])
    return "\n".join(lines)


def _describe_source(plugin: str | None) -> str:
    if plugin is None:
        return ("- Source — plugin not yet selected. Once chosen, each row "
                "read from the source will be hashed and recorded.")
    if plugin == "csv":
        return ("- Source data — each row from the CSV input. SHA-256 hash "
                "recorded for the source file and for each row.")
    if plugin == "json":
        return ("- Source data — each record from the JSON input. SHA-256 "
                "hash recorded for the source file and for each record.")
    if plugin == "dataverse":
        return ("- Source data — each record returned by the Dataverse "
                "query, with query parameters and result hashes recorded.")
    return (f"- Source data — each row from the {plugin} source. "
            f"Row-level hash recorded for every record.")


def _describe_transform(node: NodeSpec) -> str:
    name = node.id
    plugin = node.plugin or "unknown"
    if plugin == "llm":
        return (f"- {name} (LLM transform) — for each row: the full prompt "
                f"(with your accepted definitions), the full response, the "
                f"model and version, and the timestamp. Recorded in the "
                f"audit database.")
    if plugin == "passthrough":
        return (f"- {name} (passthrough) — copies the row unchanged. The "
                f"audit trail records the hop; no new fields are written.")
    if plugin == "web_scrape":
        return (f"- {name} (web scrape) — for each URL: HTTP status, "
                f"response time, and response body hash. Bytes are stored "
                f"under the payload retention policy.")
    if plugin == "rag_retrieval":
        return (f"- {name} (RAG retrieval) — for each query: the retrieval "
                f"request and top-k result hashes recorded. External call to "
                f"the configured vector store.")
    if plugin in ("azure_content_safety", "azure_prompt_shield"):
        return (f"- {name} ({plugin} — Azure safety) — for each row: the "
                f"safety analysis request and verdict recorded. External call "
                f"to Azure AI Services.")
    return (f"- {name} ({plugin} transform) — input row hash, output row "
            f"hash, and per-row outcome recorded.")


def _describe_output(output: OutputSpec) -> str:
    plugin = output.plugin or "unknown"
    if plugin in ("csv", "json"):
        return (f"- {output.name} ({plugin} file) — written to your session "
                f"storage. SHA-256-hashed; chain-of-custody recorded with "
                f"the run id and timestamp.")
    if plugin == "azure_blob":
        return (f"- {output.name} (Azure Blob) — uploaded to the configured "
                f"container. Content hash and remote path recorded; the "
                f"local emit is hashed before transit.")
    return (f"- {output.name} ({plugin} sink) — write outcome and output "
            f"hash recorded for each row.")
```

Run tests → PASS.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/audit_readiness/explain.py tests/unit/web/audit_readiness/test_explain.py
mypy src/elspeth/web/audit_readiness/explain.py
ruff check src/elspeth/web/audit_readiness/explain.py
ruff format --check src/elspeth/web/audit_readiness/explain.py
git commit -m "feat(web): add deterministic Explain narrative builder (Phase 2A.4)"
```

## Task 5: REST routes — snapshot + explain

**Files:** `web/audit_readiness/routes.py`, `web/app.py`, `tests/integration/web/test_audit_readiness_routes.py`.

- [ ] **Step 1: Confirm the app-composition site**

The router wiring lives in `src/elspeth/web/app.py` around line 386 (`catalog_router` is included there). The attributes the new routes need:
- `app.state.execution_service` (confirmed in `execution/routes.py:80`).
- `app.state.session_service`.
- `app.state.secret_service` — confirm via `grep -n "secret_service\b" src/elspeth/web/app.py`. If the attribute is named differently (e.g., `secret_resolver`), use the actual name; do not rename.
- `app.state.settings`.

- [ ] **Step 2.0 (prerequisite): Verify or scaffold integration fixtures**

Before writing the route tests, grep for the actual fixtures in the integration
conftest:

```bash
grep -rn "^def " tests/integration/web/conftest.py
```

The integration conftest (`tests/integration/web/conftest.py`) currently
provides `composer_test_client` (a minimal app with `session_service` wired;
`execution_service`, `composer_service`, and `scoped_secret_resolver` are all
set to `None`). The audit-readiness routes need all four on `app.state`.

If the following fixtures do **not** exist in the integration conftest, scaffold
them there before writing the route tests. **Do not adapt-rename phantom names
— scaffold properly.** The model is `tests/integration/web/conftest.py`'s
existing `composer_test_client` fixture.

Fixtures needed and their contracts:

| Fixture | Contract |
|---------|----------|
| `audit_readiness_test_client` | Full app with `execution_service`, `session_service`, `scoped_secret_resolver`, `settings` on `app.state`; auth bypassed to a test `UserIdentity`. |
| `audit_readiness_client_with_state` | `audit_readiness_test_client` + a session with a valid `CompositionState` persisted for the test user. Returns `(client, session_id)`. |
| `audit_readiness_client_without_state` | `audit_readiness_test_client` + a session with no composition state. Returns `(client, session_id)`. |
| `audit_readiness_client_anonymous` | `audit_readiness_test_client` with auth override raising `HTTPException(401)`. |
| `audit_readiness_other_user_session_id` | A session_id owned by a different user than the test `UserIdentity`. Used for IDOR tests. |

For IDOR test shape, model on `tests/integration/web/test_preferences_routes.py`
which uses a local `client_anonymous` fixture.

- [ ] **Step 2: Write the failing route test**

Read `tests/integration/web/conftest.py` first to confirm which fixtures are
available, then adapt the stubs below to the real fixture names from Step 2.0.

```python
"""Integration tests for /api/sessions/{sid}/audit-readiness routes."""

# Adapt fixture names to whatever Step 2.0 scaffolded in
# tests/integration/web/conftest.py.  The names below use the recommended
# scaffold names from Task 5 Step 2.0.


def test_snapshot_returns_six_canonical_rows(audit_readiness_client_with_state):
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == str(session_id)
    assert body["composition_version"] >= 1
    assert {row["id"] for row in body["rows"]} == {
        "validation", "plugin_trust", "provenance",
        "retention", "llm_interpretations", "secrets",
    }


def test_snapshot_404_when_no_state(audit_readiness_client_without_state):
    client, session_id = audit_readiness_client_without_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 404


def test_snapshot_404_on_cross_user_access(audit_readiness_test_client, audit_readiness_other_user_session_id):
    response = audit_readiness_test_client.get(
        f"/api/sessions/{audit_readiness_other_user_session_id}/audit-readiness"
    )
    assert response.status_code == 404


def test_snapshot_requires_auth(audit_readiness_client_anonymous):
    import uuid
    any_session_id = uuid.uuid4()
    response = audit_readiness_client_anonymous.get(
        f"/api/sessions/{any_session_id}/audit-readiness"
    )
    assert response.status_code == 401


def test_explain_returns_narrative(audit_readiness_client_with_state):
    client, session_id = audit_readiness_client_with_state
    response = client.get(
        f"/api/sessions/{session_id}/audit-readiness/explain"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["narrative"].startswith(
        "When you run this pipeline, ELSPETH will record:"
    )


def test_explain_404_when_no_state(audit_readiness_client_without_state):
    client, session_id = audit_readiness_client_without_state
    response = client.get(
        f"/api/sessions/{session_id}/audit-readiness/explain"
    )
    assert response.status_code == 404


def test_explain_requires_auth(audit_readiness_client_anonymous):
    import uuid
    any_session_id = uuid.uuid4()
    response = audit_readiness_client_anonymous.get(
        f"/api/sessions/{any_session_id}/audit-readiness/explain"
    )
    assert response.status_code == 401


def test_snapshot_includes_no_store_cache_header(audit_readiness_client_with_state):
    client, session_id = audit_readiness_client_with_state
    response = client.get(f"/api/sessions/{session_id}/audit-readiness")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"


def test_rejects_malformed_session_id(audit_readiness_test_client):
    response = audit_readiness_test_client.get("/api/sessions/not-a-uuid/audit-readiness")
    assert response.status_code == 422
```

Run → FAIL.

Run → FAIL.

- [ ] **Step 3a: Create `sessions/ownership.py`**

Create `src/elspeth/web/sessions/ownership.py` (L3 peer, no circular dependency risk) with a single `async def verify_session_ownership(session_id: UUID, user: UserIdentity, request: Request) -> None` that reads `session_service` and `settings` from `request.app.state`, calls `session_service.get_session(session_id)` (ValueError → 404), and raises HTTP 404 on user_id or auth_provider mismatch. Returns 404 in all cases (not 403) to avoid leaking session existence (IDOR). The implementation is identical to the `_verify_session_ownership` function currently in `execution/routes.py`.

In the same commit, update `execution/routes.py` to import `verify_session_ownership` from `sessions/ownership.py` and delete its local copy.

- [ ] **Step 3: Implement the routes**

`src/elspeth/web/audit_readiness/routes.py`:

```python
"""FastAPI router for the audit-readiness endpoints.

GET /api/sessions/{sid}/audit-readiness         → AuditReadinessSnapshot
GET /api/sessions/{sid}/audit-readiness/explain → AuditReadinessExplain

Both GET, Cache-Control: no-store, auth-required; missing state → 404.
Layer: L3 (application).
"""

from __future__ import annotations
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from elspeth.web.audit_readiness.explain import build_narrative
from elspeth.web.audit_readiness.models import AuditReadinessExplain, AuditReadinessSnapshot
from elspeth.web.audit_readiness.service import ReadinessService
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.config import WebSettings
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.ownership import verify_session_ownership
from elspeth.web.sessions.protocol import SessionServiceProtocol

_NO_STORE = "no-store"


def create_audit_readiness_router() -> APIRouter:
    router = APIRouter(tags=["audit-readiness"])

    @router.get(
        "/api/sessions/{session_id}/audit-readiness",
        response_model=AuditReadinessSnapshot,
    )
    async def snapshot(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> JSONResponse:
        await verify_session_ownership(session_id, user, request)
        service: ReadinessService = request.app.state.readiness_service
        try:
            result = await service.compute_snapshot(
                session_id=str(session_id), user_id=user.user_id,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return JSONResponse(
            content=result.model_dump(),
            headers={"Cache-Control": _NO_STORE},
        )

    @router.get(
        "/api/sessions/{session_id}/audit-readiness/explain",
        response_model=AuditReadinessExplain,
    )
    async def explain(
        session_id: UUID,
        request: Request,
        user: UserIdentity = Depends(get_current_user),  # noqa: B008
    ) -> JSONResponse:
        await verify_session_ownership(session_id, user, request)
        # Accepted double-read of session record (deferred optimization; tune if profiling shows need).
        session_service: SessionServiceProtocol = request.app.state.session_service
        settings: WebSettings = request.app.state.settings
        record = await session_service.get_current_state(session_id)
        if record is None:
            raise HTTPException(
                status_code=404, detail="No composition state for this session",
            )
        state = state_from_record(record)
        result = AuditReadinessExplain(
            session_id=str(session_id),
            composition_version=state.version,
            narrative=build_narrative(
                state, retention_days=settings.payload_store_retention_days,
            ),
        )
        return JSONResponse(
            content=result.model_dump(),
            headers={"Cache-Control": _NO_STORE},
        )

    return router
```

- [ ] **Step 4: Wire the service + router into the app**

In `src/elspeth/web/app.py`, alongside the existing `catalog_router` inclusion (line 386), add:

```python
from elspeth.web.audit_readiness.routes import create_audit_readiness_router
from elspeth.web.audit_readiness.service import ReadinessService

# After execution_service, session_service, scoped_secret_resolver are on app.state
# (app.py:470: scoped_secret_resolver = ScopedSecretResolver(secret_service, settings.auth_provider)):
app.state.readiness_service = ReadinessService(
    execution_service=app.state.execution_service,
    session_service=app.state.session_service,
    secret_service=app.state.scoped_secret_resolver,  # matches ExecutionService (line 232) precedent
    settings=app.state.settings,
)

app.include_router(create_audit_readiness_router())
```

**Do NOT use `app.state.secret_service`** — that is the raw `WebSecretService`
which requires `auth_provider_type`. Use `app.state.scoped_secret_resolver`
(set at app.py:470) which already has `auth_provider` baked in. This matches
the precedent set by `ExecutionService`, `BlobService`, and `ComposerService`
(all injected at app.py:232, 480 etc.).

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_readiness_routes.py -v
.venv/bin/python -m pytest tests/integration/web/ -v
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: all three PASS. The layer-import check should accept the new package; `audit_readiness` is L3 and only imports L0/L1/L2/L3 (`composer/state.py`, `execution/schemas.py`, `sessions/converters.py`, `sessions/protocol.py` are all L3 — peer imports are fine).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/audit_readiness/routes.py \
        src/elspeth/web/app.py \
        tests/integration/web/test_audit_readiness_routes.py
mypy src/elspeth/web/audit_readiness/ src/elspeth/web/sessions/ownership.py
ruff check src/elspeth/web/audit_readiness/ src/elspeth/web/sessions/ownership.py
ruff format --check src/elspeth/web/audit_readiness/ src/elspeth/web/sessions/ownership.py
git commit -m "$(cat <<'EOF'
feat(web): expose audit-readiness snapshot + explain routes (Phase 2A.5)

GET /api/sessions/{sid}/audit-readiness         → six-row panel snapshot.
GET /api/sessions/{sid}/audit-readiness/explain → narrative prose.
Both routes respond with Cache-Control: no-store.
Session ownership verified via sessions/ownership.py (shared with execution/routes.py).
Missing composition state returns 404.
ReadinessService wired with app.state.scoped_secret_resolver (matches precedent).

See docs/composer/ux-redesign-2026-05/14a-phase-2a-backend.md.
EOF
)"
```

---

## What Phase 2A leaves the backend in

- New `audit_readiness/` package with models, closed-list trust classifier, aggregator service, narrative builder, and two REST routes.
- The aggregator composes existing validation / catalog / secrets / retention signals — no new validation logic.
- Narrative is deterministic and system-generated.
- `WebSettings.payload_store_retention_days` (default 90) backs the Retention row.
- No frontend changes; Phase 2B picks up.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Plugin-trust allowlist drifts as plugins are renamed | Subset-of-catalog test fails the build. Allowlists are short by design. |
| `identity_node_advisory` affected_nodes field not wired in validation.py | No longer a risk: Task 1 Step 2b co-updates the `_CHECK_IDENTITY_NODE_ADVISORY` site in `validation.py` in the same commit as the schema change. The integration test in Task 3 Step 2b asserts `component_ids` is non-empty when `status == "warning"`. |
| LLM-interpretations row stays not_applicable until Phase 5b | By design. Phase 2B hides the row when no LLM transforms are present. |
| Retention row never moves off not_applicable | By design (load-bearing decision). Future phase flips it when a per-composition retention surface exists. |
| Aggregator calls `validate_pipeline()` on every panel refresh; validation is expensive | Phase 2B debounces on `compositionState.version`. Cache by `(session_id, composition_version)` only if profiling shows the need — premature caching is forbidden. |

## Review history

**2026-05-15** — Panel findings applied: `state_from_record` collaborator replaces `_state_override`/`getattr` (BLOCKER); structured `error_code`/`affected_nodes` replaces substring matching (CRITICAL); `_verify_session_ownership` extracted to `sessions/ownership.py` (IMPORTANT); POST→GET with `Cache-Control: no-store` (IMPORTANT); `Determinism.EXTERNAL_CALL` completeness test added (IMPORTANT); `list_refs` wrapped in `run_sync_in_worker` (IMPORTANT); deferred-optimization note on `explain` double-read (SUGGESTION).

### 2026-05-16 — 4-reviewer panel verdict CHANGES_REQUESTED → fixes applied

Reviewers: reality, architecture, quality, systems (full report:
`14-phase-2-audit-readiness-panel.review.json`).

Fixes applied to 14a in this revision:
1. NodeSpec/OutputSpec factories with full required-kwarg coverage (review B1, B2)
2. EXTERNAL_BOUNDARY_TRANSFORMS allowlist replaced with verified plugin names; missing EXTERNAL_CALL transforms added (review B4)
3. affected_nodes / error_code producer co-update with schema in same commit (convergence C1)
4. scoped_secret_resolver wiring (matches app.py:470 precedent); _SecretServiceLike Protocol drops auth_provider_type (convergence C4)
5. mypy + ruff in every Task commit checklist (quality REC)
6. No-Legacy enforcement — no compat-shim defaults (review B5 + CLAUDE.md)
7. Preflight greps prescribed before populating allowlist (review B4)

Strategic adjudications baked in:
- secret service: scoped_secret_resolver (matches established precedent)
- No compat-shim defaults (CLAUDE.md No-Legacy)
- Phase 2C Task 8 scope-shrink handled separately in 14c review fix-up

### 2026-05-17 — Post-landing audit reconciliation (elspeth-a615f8c418)

Source findings: backend-005, backend-006.

**backend-005 (14a-1):** Stale line citations replaced with self-locating constant and
symbol references throughout the plan. Previous citations (`execution/service.py:657,664`,
`composer/service.py:1071,1073–1079`, `validation.py:1248`) have drifted from the landed
code; future edits should cite `_CHECK_IDENTITY_NODE_ADVISORY` and surrounding function
names rather than line numbers. Note: the integration test docstring in
`tests/integration/web/test_audit_readiness_routes.py` (the `test_provenance_row_component_ids_populated_via_real_validate_pipeline` function) still references `validation.py:1248` — the programmer agent's commit should update that docstring in the same pass.

**backend-006 (14a-2):** `compute_snapshot` signature was tightened from `session_id: str`
(prescribed) to `session_id: UUID` (landed) for type safety end-to-end. The
`_ExecutionServiceLike` and `_SessionServiceLike` Protocols are correspondingly UUID-typed.
Stringification (`session_id=str(session_id)`) happens at the `AuditReadinessSnapshot`
construction boundary. This is an authorised deviation — see amendment note added to Task 3.

## Memory references

- `project_composer_personas` — informs the panel-vocabulary mapping.
- `feedback_no_calendar_shipping_commitments` — no calendar commitments.
- `feedback_default_is_fix_not_ticket` — observations only for genuinely out-of-scope items.
