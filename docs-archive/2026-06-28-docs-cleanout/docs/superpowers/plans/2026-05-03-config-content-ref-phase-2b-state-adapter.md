# Phase 2b — Canonical `composition_state` Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a single canonical adapter for "view a `composition_states` row as the YAML-form pipeline config dict it generates." Today the only path from a `CompositionStateRecord` (DB form) to the YAML-form dict is `state_from_record(record)` (typed domain object) → `generate_yaml(state)` (string) → `yaml.safe_load(...)` (round-trip back to dict). That round-trip is wasteful, but worse: any walker that traverses a `composition_state` and assumes YAML-form keys (`transforms`, `gates`, `aggregations`, `coalesce`, `outputs.<sink>`) faces a silent-miss footgun if it is handed the raw DB row directly. Phase 2b extracts the dict-build logic from `generate_yaml` into a public function `generate_pipeline_dict(state)` so DB-form callers go through `state_from_record` then `generate_pipeline_dict` with no YAML round-trip and no second flattener. This closes architectural gap [`elspeth-be405bac87`](filigree:elspeth-be405bac87) before P3's walker extension lands.

**Architecture:** One refactor in `src/elspeth/web/composer/yaml_generator.py` (extract `generate_pipeline_dict(state) -> dict`; `generate_yaml` becomes a one-line wrapper). One round-trip identity property test that pins the contract `yaml.safe_load(generate_yaml(state)) == generate_pipeline_dict(state)` AND `state == state_from_record(record_for(state))`. One YAML-shape snapshot test pinning the exact emitted dict for a state covering every node kind. One call-site migration at `src/elspeth/web/blobs/service.py:467-513` (`delete_blob`'s pre-link active-run guard) that proves the adapter works end-to-end against the existing source-only walker BEFORE P3 extends that walker. The walker name `_source_references_blob` and its YAML-form input shape are unchanged; only the input is sourced through the canonical adapter instead of by selecting `composition_states.source` directly. Phase 2b ships zero behaviour change for any pipeline that does not exercise the migrated call site.

**Tech Stack:** Python 3.13, Hypothesis (round-trip property test), PyYAML (existing dependency), pytest. No new dependencies.

---

## Pre-phase verification

- [ ] **Step 1: Confirm P2 has merged**

```bash
git log --oneline --all -- src/elspeth/contracts/blobs_inline.py src/elspeth/core/blobs_inline.py | head -5
```

Expected: at least one commit landing the L0/L1 modules from P2. P2b refactors a module that does not depend on P2's contracts, but the phase ordering pins P2b after P2 so the merge train stays linear.

- [ ] **Step 2: Confirm `generate_yaml` lives where this plan cites**

```bash
grep -n "^def generate_yaml" src/elspeth/web/composer/yaml_generator.py
```

Expected: `42:def generate_yaml(state: CompositionState) -> str:`. If the line number drifted (refactor since this plan was authored), update the citations in Tasks 2 and 3 in-place.

- [ ] **Step 3: Confirm the DB-form bridge function exists under the cited name**

```bash
grep -n "^def state_from_record" src/elspeth/web/sessions/converters.py
```

Expected: `19:def state_from_record(record: CompositionStateRecord) -> CompositionState:`. If the name has changed, fix every citation in this plan AND the P3 dependency-note replacement (Edit 4 of the parent ticket) before continuing.

- [ ] **Step 4: Confirm the layer enforcer baseline is clean**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
echo "Exit: $?"
```

Expected: exit 0. P2b touches only L3 (composer + sessions + blobs); the layer enforcer is the gate.

---

## Task 1: Round-trip identity property test (RED)

**Files:**
- Test: `tests/unit/web/composer/test_yaml_generator_round_trip.py`

This task pins the contract Phase 2b establishes: the YAML-form dict is the same dict whether you go through `generate_yaml` + `yaml.safe_load` or directly through `generate_pipeline_dict`, and a `CompositionState` round-trips losslessly through `state_from_record(record_for(state))`. The test fails RED at this step because `generate_pipeline_dict` does not yet exist.

- [ ] **Step 1: Write the failing property test**

```python
# tests/unit/web/composer/test_yaml_generator_round_trip.py
"""Round-trip identity for the canonical composition_state adapter.

Pins two contracts:

1. yaml.safe_load(generate_yaml(state)) == generate_pipeline_dict(state)
   — the YAML wrapper is a thin string serialization; the dict it
   serializes is the source of truth.

2. CompositionState round-trips losslessly through the DB representation:
   state == state_from_record(record_for(state)).

Together these pin the canonical adapter introduced by Phase 2b: any
walker that consumes a CompositionStateRecord goes through
state_from_record → generate_pipeline_dict and obtains the same dict
shape that yaml_generator emits.
"""

from __future__ import annotations

import yaml
from hypothesis import given, settings

from elspeth.web.composer.yaml_generator import (
    generate_pipeline_dict,
    generate_yaml,
)
from elspeth.web.sessions.converters import state_from_record

# Reuse the existing CompositionState strategy from the composer test
# suite if one is published; otherwise scope the property to a small
# library of hand-built states covering every node kind (source-only,
# source+transforms, source+gate, source+aggregation, source+coalesce,
# source+sinks, full-DAG).  See conftest.py:_state_examples.
from tests.unit.web.composer.conftest import composition_state_strategy
from tests.unit.web.composer.conftest import record_for  # state -> CompositionStateRecord


@given(state=composition_state_strategy())
@settings(max_examples=100, deadline=None)
def test_yaml_dump_matches_generate_pipeline_dict(state) -> None:
    """The YAML string deserializes to exactly generate_pipeline_dict(state)."""
    dict_form = generate_pipeline_dict(state)
    yaml_form = yaml.safe_load(generate_yaml(state))
    assert yaml_form == dict_form


@given(state=composition_state_strategy())
@settings(max_examples=100, deadline=None)
def test_state_round_trips_through_record(state) -> None:
    """state == state_from_record(record_for(state)) — DB representation is lossless."""
    assert state == state_from_record(record_for(state))
```

If `composition_state_strategy` / `record_for` do not yet exist in `tests/unit/web/composer/conftest.py`, add them as part of this step. They are reusable test fixtures, not implementation code — the strategy emits `CompositionState` instances covering every node kind; `record_for` builds a `CompositionStateRecord` whose raw-dict fields match the state.

- [ ] **Step 2: Run test to verify it fails RED**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator_round_trip.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_pipeline_dict' from 'elspeth.web.composer.yaml_generator'`.

- [ ] **Step 3: Commit RED**

```bash
git add tests/unit/web/composer/test_yaml_generator_round_trip.py tests/unit/web/composer/conftest.py
git commit -m "test(composer): pin generate_pipeline_dict + state_from_record round-trip identity (RED)

Property tests for the canonical composition_state adapter:

  yaml.safe_load(generate_yaml(state)) == generate_pipeline_dict(state)
  state == state_from_record(record_for(state))

generate_pipeline_dict does not exist yet — the test fails RED at
import time.  Task 2 extracts the public function from generate_yaml.

Refs: elspeth-be405bac87, elspeth-fdebcaa79a"
```

---

## Task 2: Extract `generate_pipeline_dict(state) -> dict`

**Files:**
- Modify: `src/elspeth/web/composer/yaml_generator.py:42-184`

Move the dict-build logic out of `generate_yaml` into a new public function `generate_pipeline_dict(state: CompositionState) -> dict`. `generate_yaml` becomes a one-line wrapper that calls `yaml.dump(generate_pipeline_dict(state), default_flow_style=False, sort_keys=False)`. All `yaml.dump` flags and the `sort_keys=False` insertion-order guarantee are preserved verbatim.

- [ ] **Step 1: Refactor**

```python
# src/elspeth/web/composer/yaml_generator.py — replace generate_yaml

def generate_pipeline_dict(state: CompositionState) -> dict[str, Any]:
    """Convert a CompositionState to the YAML-form pipeline config dict.

    The returned dict is the canonical "view a composition_state as a
    pipeline config" shape: keys are ``source``, ``transforms``,
    ``gates``, ``aggregations``, ``coalesce``, ``sinks`` (the natural
    pipeline flow order; insertion-ordered).  This is the dict
    ``generate_yaml`` serializes; consumers that need a dict (lifecycle
    walkers, validation walkers, audit-trail diffs) call this directly
    and skip the YAML round-trip.

    Pure function. Same CompositionState always produces the same dict.

    Args:
        state: The pipeline composition state to convert.

    Returns:
        Dict matching the shape ELSPETH's load_settings() parser
        consumes (after yaml.safe_load).
    """
    # Unwrap frozen containers to plain Python types (R4).
    state_dict = state.to_dict()

    doc: dict[str, Any] = {}

    # ... (existing source/transforms/gates/aggregations/coalesce/sinks
    #      block from the current generate_yaml body, UNCHANGED) ...

    return doc


def generate_yaml(state: CompositionState) -> str:
    """Convert a CompositionState to ELSPETH pipeline YAML.

    Thin wrapper around ``generate_pipeline_dict``.  The output is
    deterministic: same state produces byte-identical YAML.

    Args:
        state: The pipeline composition state to serialize.

    Returns:
        YAML string representing the pipeline configuration.
    """
    # sort_keys=False preserves insertion order: source → transforms →
    # gates → aggregations → coalesce → sinks (the natural pipeline flow).
    return yaml.dump(
        generate_pipeline_dict(state),
        default_flow_style=False,
        sort_keys=False,
    )
```

The body of `generate_pipeline_dict` is the *exact* current body of `generate_yaml` from line 60 (`state_dict = state.to_dict()`) through line 178 (the sinks block) — moved verbatim. Do **not** alter any field-emission rule, conditional-key check, or `_strip_web_metadata` call. The behavioural contract `yaml.safe_load(generate_yaml(state))` produced before this refactor must equal `generate_pipeline_dict(state)` after this refactor.

- [ ] **Step 2: Run the round-trip test to verify it passes GREEN**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator_round_trip.py -v`
Expected: PASS for both properties.

- [ ] **Step 3: Run the full composer + execution test suite**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/web/execution/ tests/integration/web/ -v`
Expected: PASS (no regression — every existing call site of `generate_yaml` is unaffected because the wrapper preserves the exact previous output).

- [ ] **Step 4: Commit GREEN**

```bash
git add src/elspeth/web/composer/yaml_generator.py
git commit -m "feat(composer): extract generate_pipeline_dict from generate_yaml

The dict-build logic that yaml_generator.generate_yaml has carried
since inception is now a public function generate_pipeline_dict(state).
generate_yaml is a one-line wrapper: yaml.dump(generate_pipeline_dict
(state), default_flow_style=False, sort_keys=False).

This is the canonical 'view a composition_state as a pipeline config'
adapter.  DB-form callers go through state_from_record(record) ->
generate_pipeline_dict(state) and skip the YAML round-trip.  Lifecycle
walkers, validation walkers, and audit-trail diffs that need the
YAML-form dict shape consume this function directly, eliminating the
DB-form / YAML-form silent-miss footgun (elspeth-be405bac87).

No behaviour change at any existing call site: every yaml.dump flag is
preserved; the dict shape is byte-identical to what generate_yaml
emitted before this refactor (pinned by the round-trip property test
landed in the previous commit).

Refs: elspeth-be405bac87, elspeth-fdebcaa79a"
```

---

## Task 3: Add an explicit YAML-shape snapshot pin

**Files:**
- Test: `tests/unit/web/composer/test_yaml_generator_shape_snapshot.py`

The round-trip test from Task 1 pins that the YAML and dict forms agree, but says nothing about *what* shape `generate_pipeline_dict` emits. A separate snapshot test pins the exact dict for a single representative state covering every node kind. This isolates "the YAML shape changed" failures to one test rather than scattering them across the lifecycle/validation walkers downstream of the adapter.

- [ ] **Step 1: Write the snapshot test**

```python
# tests/unit/web/composer/test_yaml_generator_shape_snapshot.py
"""Snapshot pin for generate_pipeline_dict's emitted shape.

A single representative CompositionState covering source, transforms,
gates, aggregations, coalesce, and sinks.  The expected dict is
checked in alongside the test.  When the YAML shape changes, this
test breaks (and only this test); downstream walkers that consume
generate_pipeline_dict's output are unaffected by the change in
shape itself — they break only if their expectations of the shape
diverge from what's pinned here.
"""

from __future__ import annotations

from elspeth.web.composer.yaml_generator import generate_pipeline_dict
from tests.unit.web.composer.conftest import full_dag_state  # fixture: every node kind


def test_generate_pipeline_dict_emits_expected_shape(full_dag_state) -> None:
    expected = {
        "source": {
            "plugin": "csv",
            "on_success": "continue",
            "options": {
                "path": "data.csv",
                "on_validation_failure": "fail",
            },
        },
        "transforms": [
            {
                "name": "classify",
                "plugin": "llm",
                "input": "source",
                "on_success": "continue",
                "on_error": "fail",
                "options": {"system_prompt": "..."},
            },
        ],
        "gates": [
            {
                "name": "route",
                "input": "classify",
                "condition": "...",
                "routes": ["a", "b"],
            },
        ],
        "aggregations": [
            {
                "name": "summarize",
                "plugin": "count",
                "input": "route",
                "on_success": "continue",
                "on_error": "fail",
            },
        ],
        "coalesce": [
            {
                "name": "merge",
                "branches": ["a", "b"],
                "policy": "first",
                "merge": "...",
            },
        ],
        "sinks": {
            "out": {"plugin": "csv", "on_write_failure": "fail", "options": {"path": "out.csv"}},
        },
    }
    assert generate_pipeline_dict(full_dag_state) == expected
```

The exact `expected` dict will be slightly different in practice — copy the actual output of `generate_pipeline_dict(full_dag_state)` once, eyeball it for correctness, and pin that. The point is the snapshot, not the specific values.

- [ ] **Step 2: Run the snapshot test**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator_shape_snapshot.py -v`
Expected: PASS (the snapshot was lifted from the GREEN function output).

- [ ] **Step 3: Commit the snapshot**

```bash
git add tests/unit/web/composer/test_yaml_generator_shape_snapshot.py tests/unit/web/composer/conftest.py
git commit -m "test(composer): pin generate_pipeline_dict YAML shape snapshot

Single representative state covering every node kind; expected dict
checked in.  When the YAML shape changes, this test breaks (and only
this test); downstream walkers that consume generate_pipeline_dict's
output break only if their expectations of the shape diverge from
what's pinned here.  Isolates 'the YAML shape changed' failures to
one location.

Refs: elspeth-be405bac87, elspeth-fdebcaa79a"
```

---

## Task 4: Migrate `delete_blob`'s pre-link active-run guard to the adapter

**Files:**
- Modify: `src/elspeth/web/blobs/service.py:467-513`

The `delete_blob` pre-link active-run guard at `service.py:467-513` currently joins `runs` to `composition_states` and selects only `composition_states.source`, then passes that source dict to `_source_references_blob(active_run.source, blob_id_str, row.storage_path)`. The walker walks `source.options` for `blob_ref`/`path`/`file` matches.

This task changes the SELECT to fetch the full `CompositionStateRecord` shape (every column the record carries), then routes through the canonical adapter:

```python
state_record = ...  # built from the fetched columns
state = state_from_record(state_record)
config_dict = generate_pipeline_dict(state)
if _source_references_blob(config_dict, blob_id_str, row.storage_path):
    raise BlobActiveRunError(blob_id_str, run_id=active_run.id)
```

Because P3's walker extension is *not yet* landed, the existing source-only walker `_source_references_blob` runs against the YAML-form dict. The walker's existing logic walks the `source` key of its input — and `generate_pipeline_dict(state)["source"]` is the same dict the previous SELECT obtained directly. The migration is shape-preserving: every blob deletion that the previous code path blocked is still blocked; every deletion it permitted is still permitted.

The walker name and signature stay unchanged. P3 will widen the walker (rename to `_composition_references_blob`, walk transforms/gates/aggregations/coalesce/outputs) — this Phase 2b migration is what makes that widening trivially correct, because the walker's input shape is now canonically YAML-form regardless of which DB row it came from.

- [ ] **Step 1: Update the SELECT and call site**

```python
# src/elspeth/web/blobs/service.py — replace lines 492-502

# 2. Pre-link window: select the full composition_states row,
#    convert to CompositionStateRecord, then route through the
#    canonical YAML-form adapter (generate_pipeline_dict).
#    See docs/superpowers/plans/2026-05-03-config-content-ref-phase-2b-state-adapter.md.
active_run_row = conn.execute(
    select(
        runs_table.c.id.label("run_id"),
        composition_states_table.c.id.label("state_id"),
        composition_states_table.c.version,
        composition_states_table.c.source,
        composition_states_table.c.nodes,
        composition_states_table.c.edges,
        composition_states_table.c.outputs,
        composition_states_table.c.metadata_,
    )
    .join(
        composition_states_table,
        runs_table.c.state_id == composition_states_table.c.id,
    )
    .where(runs_table.c.session_id == row.session_id)
    .where(runs_table.c.status.in_(["pending", "running"]))
).first()

if active_run_row is not None:
    state_record = CompositionStateRecord(
        id=active_run_row.state_id,
        version=active_run_row.version,
        source=active_run_row.source,
        nodes=active_run_row.nodes,
        edges=active_run_row.edges,
        outputs=active_run_row.outputs,
        metadata_=active_run_row.metadata_,
        # remaining audit fields populated per CompositionStateRecord contract
    )
    config_dict = generate_pipeline_dict(state_from_record(state_record))
    if _source_references_blob(config_dict, blob_id_str, row.storage_path):
        raise BlobActiveRunError(blob_id_str, run_id=active_run_row.run_id)
```

The exact `CompositionStateRecord` field list must match the dataclass definition at `src/elspeth/web/sessions/protocol.py:162` — verify and update the SELECT to fetch every required column. If `CompositionStateRecord` carries fields that are not relevant to YAML emission (e.g. `created_at`), populate them with the row's value or a defensible default (the `state_from_record → generate_pipeline_dict` path does not read them).

Imports to add at the top of `service.py`:

```python
from elspeth.web.composer.yaml_generator import generate_pipeline_dict
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.protocol import CompositionStateRecord
```

- [ ] **Step 2: Run the existing active-run guard tests**

Run: `.venv/bin/python -m pytest tests/unit/web/blobs/test_service.py -v -k "active_run"`
Expected: PASS — every existing test passes unchanged. The migration is shape-preserving: the walker sees the same `source` dict it saw before, and the four existing pinning tests (`test_delete_blob_rejects_when_active_run_linked`, `test_delete_blob_rejects_when_active_run_exists_without_link`, `test_delete_blob_allows_when_active_run_uses_different_source`, `test_delete_blob_rejects_when_active_run_path_matches_storage`) cover the source-bound surface that the walker still consumes.

- [ ] **Step 3: Layer enforcer must stay clean**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
echo "Exit: $?"
```

Expected: exit 0. `web/blobs/service.py` (L3) imports from `web/composer/yaml_generator` (L3) and `web/sessions/converters` (L3) — same-layer, allowed.

- [ ] **Step 4: Commit the migration**

```bash
git add src/elspeth/web/blobs/service.py
git commit -m "refactor(blobs): route delete_blob pre-link guard through generate_pipeline_dict

The pre-link active-run guard at delete_blob now selects the full
composition_states row, converts to CompositionStateRecord, and routes
through state_from_record -> generate_pipeline_dict before invoking
_source_references_blob.  The walker sees the same YAML-form dict it
saw before — only the input plumbing changes.

Shape-preserving migration: every active-run guard test passes
unchanged.  This proves the canonical adapter works against the
existing source-only walker BEFORE P3 widens the walker to consume
transforms/gates/aggregations/coalesce/outputs (which the same
adapter trivially provides).

Refs: elspeth-be405bac87, elspeth-fdebcaa79a"
```

---

## Task 5: Bug-verification protocol at the migration site

**Files:**
- (no new files; manual revert + observe + re-apply)

The agreement-suite docstring's bug-verification protocol requires that each load-bearing fix site is exercised by a revert-and-observe step at the same PR boundary. For Phase 2b the load-bearing fix site is the call-site migration in Task 4. The intent: prove that the migration is shape-preserving by reverting it and confirming the existing source-only test surface still passes (because the walker is correct against `composition_state.source` regardless of which input shape it came from).

- [ ] **Step 1: Manually revert Task 4's call-site change**

Restore the previous `select(runs_table.c.id, composition_states_table.c.source)` → `_source_references_blob(active_run.source, ...)` block. Do NOT commit the revert — keep it as an uncommitted local change.

- [ ] **Step 2: Run the existing source-data lifecycle tests**

Run: `.venv/bin/python -m pytest tests/unit/web/blobs/test_service.py -v -k "active_run"`
Expected: PASS — the four existing tests (`test_delete_blob_rejects_when_active_run_linked`, `test_delete_blob_rejects_when_active_run_exists_without_link`, `test_delete_blob_allows_when_active_run_uses_different_source`, `test_delete_blob_rejects_when_active_run_path_matches_storage`) pass against the reverted code. They pass because the source-only walker has always been correct against `composition_state.source`; the Phase 2b migration changed the input plumbing, not the walker semantics.

The bug-verification confirms the migration is shape-preserving: nothing the walker formerly blocked has become unblocked, and nothing it formerly permitted has become blocked.

- [ ] **Step 3: Re-apply Task 4's change**

Restore the migrated code path. Run the active-run tests one more time to confirm: `.venv/bin/python -m pytest tests/unit/web/blobs/test_service.py -v -k "active_run"`. Expected: PASS.

- [ ] **Step 4: Commit the bug-verification documentation**

Add a short inline comment to `delete_blob` referencing this protocol so future readers can see why the migration is safe:

```python
# Pre-link window guard.  Bug-verification (P2b): manually reverting this
# block to the previous select(composition_states.source) form leaves the
# four active-run pinning tests in test_service.py passing — the walker
# is correct against composition_state.source regardless of input
# plumbing.  This migration changed the input shape (raw row -> canonical
# YAML-form via generate_pipeline_dict), not the walker semantics.  P3
# widens the walker to traverse the full YAML-form dict; the adapter
# established here is what makes that widening trivially correct.
```

Then:

```bash
git add src/elspeth/web/blobs/service.py
git commit -m "docs(blobs): document P2b bug-verification at delete_blob pre-link guard

Inline comment recording the shape-preserving nature of the P2b
migration: the four existing active-run tests pass against both the
pre-migration code (select source only) and the post-migration code
(select full row + route through generate_pipeline_dict).  P3 widens
the walker; the adapter established in P2b makes that widening
trivially correct.

Refs: elspeth-be405bac87, elspeth-fdebcaa79a"
```

---

## Task 6: Open the P2b PR + close `elspeth-be405bac87`

- [ ] **Step 1: Push and open**

```bash
git push -u origin <branch-name>
gh pr create --title "refactor(composer,blobs): canonical composition_state adapter (generate_pipeline_dict)" --body "$(cat <<'EOF'
## Summary

- Extracts the dict-build logic from \`yaml_generator.generate_yaml\` into a public function \`generate_pipeline_dict(state) -> dict\`. \`generate_yaml\` becomes a one-line \`yaml.dump\` wrapper.
- Pins the round-trip identity contract: \`yaml.safe_load(generate_yaml(state)) == generate_pipeline_dict(state)\` AND \`state == state_from_record(record_for(state))\`. Property test (Hypothesis) + explicit YAML-shape snapshot.
- Migrates \`delete_blob\`'s pre-link active-run guard at \`web/blobs/service.py:467-513\` to route through \`state_from_record → generate_pipeline_dict\` instead of selecting \`composition_states.source\` directly. Shape-preserving: every existing active-run pinning test passes unchanged.
- Bug-verification protocol applied at the migration site: manual revert leaves the four existing tests passing — proof that the migration changes input plumbing, not walker semantics.

Closes \`elspeth-be405bac87\` (architectural gap: no canonical DB-form ↔ YAML-form adapter for \`composition_states\`).

P3's walker extension (Task 5b in \`2026-05-03-config-content-ref-phase-3-runtime-preflight.md\`) consumes \`generate_pipeline_dict(state_from_record(record))\` as its input, eliminating the per-implementer DB-vs-YAML-shape caveat that P3 previously carried.

## Test plan

- [ ] \`pytest tests/unit/web/composer/test_yaml_generator_round_trip.py tests/unit/web/composer/test_yaml_generator_shape_snapshot.py -v\`
- [ ] \`pytest tests/unit/web/blobs/test_service.py -v -k active_run\` (existing tests pass unchanged)
- [ ] \`pytest tests/unit/web/composer/ tests/unit/web/execution/ tests/integration/web/ -v\` (no regression at any \`generate_yaml\` call site)
- [ ] \`enforce_tier_model.py check\` exits 0
- [ ] Manual: revert the \`delete_blob\` call-site migration and confirm the four \`active_run\` pinning tests still pass (bug-verification — shape-preserving migration)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: After merge, close the architectural-gap ticket**

```bash
filigree close elspeth-be405bac87 --reason="canonical composition_state adapter shipped in P2b: generate_pipeline_dict(state) is the single source of truth for the YAML-form dict; DB-form callers route through state_from_record. P3 walker extension consumes the adapter, retiring the DB-vs-YAML-shape caveat."
```

---

## Done conditions

P2b is done when:

1. `generate_pipeline_dict(state) -> dict` exists in `src/elspeth/web/composer/yaml_generator.py` and is the function `generate_yaml` wraps.
2. The round-trip property test in `tests/unit/web/composer/test_yaml_generator_round_trip.py` passes (both properties).
3. The YAML-shape snapshot test in `tests/unit/web/composer/test_yaml_generator_shape_snapshot.py` passes.
4. `delete_blob`'s pre-link active-run guard at `src/elspeth/web/blobs/service.py:467-513` consumes `generate_pipeline_dict(state_from_record(record))` instead of `composition_states.source` directly.
5. The four existing active-run pinning tests in `tests/unit/web/blobs/test_service.py` pass against both the migrated and the (manually) un-migrated code paths (shape-preserving).
6. `enforce_tier_model.py check` exits 0.
7. `enforce_freeze_guards.py` is clean.
8. P3's Task 5b caveat block is replaced by the one-line dependency note (executed as Edit 4 of the umbrella ticket that introduced this plan).
9. PR is merged.
10. `elspeth-be405bac87` is closed with the closure message above.

Move to `2026-05-03-config-content-ref-phase-3-runtime-preflight.md` only after P2b is merged.
