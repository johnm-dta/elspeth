# Composer Progress Persistence — Phase 2: Redaction Framework (rev 5 rewrite, rev-4 + rev-5 fix sets applied)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review history (read before implementing).** This plan has been through four review passes. Each pass surfaced findings that were folded into the plan body. The findings, their closure mechanism, and (where relevant) the closure mechanism that itself was later superseded, are tracked in Appendices A–D. **The current load-bearing closure for the property test (Task 19) is the rev-4 invariant — not the rev-3 mechanism.** Read Appendix D before editing the property test.
>
> Reviews to date:
> - rev-1 → `…-redaction.review.json` (4 BLOCKERs, closed by full rewrite committed in `7338e4e2`)
> - rev-2 → `…-redaction.review-rev2.json` (3 BLOCKERs A/B/C, 5 MAJOR groups, closed by `b73abd8a`)
> - rev-3 → `…-redaction.review-rev3.json` (1 BLOCKER B1, 2 MAJORs M1/M2; closure landed but rev-3 B1 closure was later determined insufficient — see Appendix C)
> - rev-4 → `…-redaction.review-rev4.json` (1 BLOCKER, 2 MAJORs; closures applied — see Appendix D)
> - rev-5 → `…-redaction.review-rev5.json` (no BLOCKERs across all four reviewers — first review pass to break the property-test convergence pattern; warnings folded — see Appendix E)

**Goal.** Introduce a manifest-keyed redaction primitive (`MANIFEST: dict[str, ToolRedaction]`) that mirrors the project's existing `_TOOL_REQUIRED_PATHS: dict[str, ...]` precedent at `src/elspeth/web/composer/service.py:702`, alongside the function-pointer dispatch dicts at `src/elspeth/web/composer/tools.py:5250–5314`. Promote ~6–8 sensitive-touching composer tools to type-driven manifest entries with `Sensitive[T]`-annotated Pydantic argument models and `Model.model_validate` dispatch validation; promoted handlers catch `pydantic.ValidationError` and re-raise as `ToolArgumentError` (per `tools.py:2668–2801` pattern), which is caught at `service.py:2480` and routes to ARG_ERROR. Cover the remaining ~29–31 tools with declarative manifest entries. Enforce coverage and weakening with a single shared traversal iterator consumed by both the CI-time adequacy guard and the runtime walker, plus a content-keyed policy-hash snapshot and a direction-aware CI-enforced PR-label gate. Close all prior BLOCKERs and warnings, plus the rev-2 BLOCKERs and MAJORs from `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-2-redaction.review-rev2.json`, plus the rev-3 and rev-4 findings tracked in Appendices C and D.

**Architecture.** Pure redaction-layer work. Phase 1 schema is in place; this phase does not modify the database, the compose loop's transactional structure, or the frontend. The redaction layer is L3 (alongside the composer tools). The promotion wave touches handler dispatch within `tools.py` (still L3) — no upward layer hops. Tier-model and freeze-guard CI gates apply unchanged.

**Tech Stack.** Python 3.13, Pydantic 2.x (`Annotated`, `model_fields`, `model_validate`), pytest, hashlib for the snapshot, GitHub Actions for the label-gate.

**Spec sections.** Header rev-5 architectural pivot; §4.2 (manifest, Sensitive[T], declarative shape, RedactionTelemetry, shared traversal iterator, walker boundary); §4.3 (sentinel rules); §4.4 (four-assertion adequacy guard); §4.7 (initial policy declarations — superseded by the per-tool tasks below); §9 RSK-03 (summarizer crash discipline); §11 Phase 2 scope and done-when; §12.2 reviewer-finding traceability.

**This plan supersedes** the rev-4-derived 13-task plan that returned `CHANGES_REQUESTED` with four BLOCKERs. The previous plan's tasks are deleted, not refactored — the precondition gap (BLOCKER B1) is structural, not localised.

---

## Preflight — gate state required before Task 1

Before starting Task 1, verify the worktree is in a clean Phase 2 starting state:

- [ ] **Working directory.** `/home/john/elspeth/.worktrees/composer-progress-1a` (the umbrella branch worktree). Branch `feat/composer-progress-persistence-1a`. HEAD `f5115fd5` or later (Phase 1A + 1B + 1C landed; pre-Phase-2 hygiene complete).
- [ ] **venv.** `.venv/bin/python` is Python 3.13 (memory: `project_tier_model_python_version` — version skew triggers ~300 spurious tier-model false positives). Confirm with `.venv/bin/python --version`.
- [ ] **Editable install rebound.** `uv pip install --python .venv/bin/python -e ".[dev]"` if the worktree has not been rebound since fork (memory: `feedback_uv_venv_leak`).
- [ ] **Pre-Phase-2 gate green.** Run, in this order, and confirm all pass before any code change:
  - `.venv/bin/python -m pytest tests/unit -q` (all pass — verified collection count on 2026-05-12 in the `composer-progress-1a` worktree was 14,948; verify zero failures, not an exact match on the count, since the suite grows steadily as Phase 2 tests land)
  - `.venv/bin/python -m pytest tests/integration -q -m "not testcontainer"` (all pass — verified collection count on 2026-05-12 was 785; same guidance as above, verify zero failures rather than an exact count)
  - `.venv/bin/python -m mypy src/` (clean, 381 files)
  - `.venv/bin/python -m ruff check src/` (clean)
  - `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` (no findings beyond allowlisted)
  - `.venv/bin/python scripts/cicd/enforce_freeze_guards.py` (clean)
- [ ] **Spec rev-5 landed.** `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` Status line reads "revision 5"; §4.2, §4.4, §11 Phase 2, §12.2 reflect the manifest architecture (not the rev-4 class-hierarchy assumption). The plan-review JSON at `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json` is no longer load-bearing — this rewrite supersedes it.

If any of the above is red, **stop and surface to operator.** Do not begin Task 1.

**TDD discipline applies to every task in this plan.** Tasks 1–4 spell out the explicit step sequence: Step 1 — write the failing test; Step 2 — run, verify it fails; Step 3 — implement; Step 4 — run, verify it passes; Step 5 — gate; Step 6 — commit. Tasks 5–8 use compressed notation ("Standard TDD task; test cases: [list]"). The compression is readability shorthand, not an exemption — when executing those tasks, expand to the same six-step sequence. Security-sensitive code paths (summarizer-error discipline, fail-closed sentinels, `AuditIntegrityError` chaining) particularly require test-before-code sequencing.

---

## File Structure

### Files to create

- `src/elspeth/web/composer/redaction.py` — extended (the file already exists with `redact_source_storage_path`; Phase 2 adds `_SensitiveMarker`, `Sensitive`, `ToolRedaction`, `MANIFEST`, `HandlesNoSensitiveDataReason`, `ToolRedactionPolicy`, `walk_model_schema`, `TraversalNode`, `redact_tool_call_arguments`, `redact_tool_call_response`).
- `src/elspeth/web/composer/redaction_telemetry.py` — `RedactionTelemetry` Protocol + `NoopRedactionTelemetry` test implementation + `OtelRedactionTelemetry` production implementation (wraps the project's structured-counter helper).
- `tests/unit/web/composer/test_walk_model_schema.py` — shared traversal iterator coverage.
- `tests/unit/web/composer/test_tool_redaction_dataclass.py` — manifest-entry construction-error coverage.
- `tests/unit/web/composer/test_redaction_telemetry.py` — Protocol + Noop coverage.
- `tests/unit/web/composer/test_redact_set_source.py` — tracer-bullet end-to-end test.
- `tests/unit/web/composer/test_handles_no_sensitive_data_reason.py` — structured-reason validators.
- `tests/unit/web/composer/test_tool_redaction_policy.py` — declarative policy validators.
- `tests/unit/web/composer/test_redact_tool_call_response.py` — response walker (fixed sentinel, telemetry, summarizer crash discipline).
- `tests/unit/web/composer/test_redact_tool_call_arguments.py` — full-shape argument walker beyond the tracer-bullet path.
- `tests/unit/web/composer/test_adequacy_guard.py` — the four assertions of §4.4.
- `tests/unit/web/composer/redaction_policy_snapshot.json` — the committed snapshot covering all manifest entries.
- `tests/unit/web/composer/test_promote_*.py` — one per promoted tool, asserting dispatch validation + redaction + ARG_ERROR routing: handler catches `pydantic.ValidationError`, re-raises as `ToolArgumentError` (with `__cause__` being the `ValidationError`); `ToolArgumentError` caught at `service.py:2480` routes to ARG_ERROR.
- `tests/unit/web/composer/test_compose_loop_unknown_tool_name.py` — pins existing routing: `tools.py:5481` `_failure_result` fall-through → compose loop records and continues.
- `tests/unit/web/composer/test_redaction_completeness_property.py` — Hypothesis property test: no raw `Sensitive[T]` field value appears in `json.dumps(redact_tool_call_arguments(…))` (rev-2 BLOCKER_A MAJOR-4).
- `tests/unit/web/composer/test_walker_guard_parity.py` — behavioural parity: `walk_model_schema(M, with_values=False)` == `walk_model_schema(M, with_values=True)` path-sets for each manifest entry model (rev-2 M_walker_guard_parity).
- `tests/unit/web/composer/test_label_gate_direction.py` — direction-misclassification tests: (a) weakening + `policy-weaken-justified` → pass; (b) weakening + `policy-strengthen` → fail; (c) strengthening + `policy-strengthen` → pass; (d) strengthening + `policy-weaken-justified` → fail.
- `.github/workflows/composer-redaction-gate.yml` — direction-aware label-gate CI step (rev-2 BLOCKER_B).
- `scripts/cicd/bootstrap_redaction_snapshot.py` — relocated from `scripts/composer/` (that directory does not exist; rev-2 m_script_dir_missing; `scripts/cicd/` is the existing CI helper location alongside `enforce_tier_model.py`).

### Files to modify

- `src/elspeth/web/composer/tools.py` — promoted tools (Tasks 4, 13, 14, 15) gain Pydantic argument-model `model_validate` at dispatch and read typed attributes. Other tools' handlers are unchanged. The six dispatch dicts at lines 5250–5314 are unchanged in shape (only their handler bodies for promoted tools).
- `src/elspeth/web/composer/service.py` — Task 17 may add a regression test reference; no compose-loop logic changes (Phase 3 owns the loop).

### Files NOT touched in Phase 2

- `src/elspeth/web/sessions/` — Phase 1 owns the schema.
- `src/elspeth/web/composer/service.py` compose-loop body — Phase 3 wires the loop.
- `src/elspeth/web/frontend/` — Phase 4.
- `.github/CODEOWNERS` — **not created.** The `@elspeth/security` team cannot exist on `johnm-dta/elspeth` (personal-account repo). Spec rev-5 §4.4.5 promotes the label-gate to primary control. (This is a rev-5 deviation from rev-4; see plan-review W9 / M10 / spec §12.2.)
- `src/elspeth/contracts/composer_audit.py` — the `ComposerToolInvocation.arguments_canonical` field retains raw LLM-supplied arguments per spec §4.2.8 posture (a) intentional raw. Phase 3 MUST NOT redact this surface; the `arguments_hash` Tier-1 integrity invariant depends on it remaining unmodified at `begin_dispatch` / `begin_dispatch_or_arg_error` (`service.py:1930`).

---

## Task 1: Shared traversal iterator `walk_model_schema`

**Why this task is first.** Both the adequacy guard (§4.4.2) and the runtime walker (§4.2.6) consume this iterator. Building it first with comprehensive container-shape test coverage forecloses the rev-4 BLOCKER 2 pattern (walker omits container types the spec promised, guard cannot detect). The test set IS the contract; subsequent tasks consume the shipped iterator unchanged.

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_walk_model_schema.py`

**Steps:**

- [ ] **Step 1: Write the failing test.**

  Create `tests/unit/web/composer/test_walk_model_schema.py`:

  ```python
  """Tests for the shared traversal iterator (spec §4.2.5).

  Covers every container shape the iterator must descend into. Both the
  adequacy guard and the runtime walker consume this iterator; gaps here
  silently allow gaps in either consumer. See plan-review B2 in
  docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json.
  """

  from __future__ import annotations

  from typing import Annotated, Any

  import pytest
  from pydantic import BaseModel

  from elspeth.web.composer.redaction import (
      Sensitive,
      TraversalNode,
      _SensitiveMarker,
      walk_model_schema,
  )


  class _FlatModel(BaseModel):
      ok: str
      secret: Annotated[str, Sensitive()]


  class _NestedModel(BaseModel):
      header: str
      payload: _FlatModel


  class _ListModel(BaseModel):
      items: list[_FlatModel]


  class _DictModel(BaseModel):
      lookup: dict[str, _FlatModel]


  class _TupleModel(BaseModel):
      pair: tuple[_FlatModel, ...]


  class _OptionalModel(BaseModel):
      maybe: _FlatModel | None = None


  class _UnionModel(BaseModel):
      either: _FlatModel | _NestedModel


  class _MarkerLastModel(BaseModel):
      tail: Annotated[str, "irrelevant", Sensitive()]


  def _paths(nodes: list[TraversalNode]) -> list[str]:
      return [n.path for n in nodes]


  def _has_marker(nodes: list[TraversalNode], path: str) -> bool:
      for n in nodes:
          if n.path == path and any(isinstance(m, _SensitiveMarker) for m in n.metadata):
              return True
      return False


  def test_walks_flat_model() -> None:
      nodes = list(walk_model_schema(_FlatModel))
      assert _paths(nodes) == ["ok", "secret"]
      assert not _has_marker(nodes, "ok")
      assert _has_marker(nodes, "secret")


  def test_descends_into_nested_basemodel() -> None:
      nodes = list(walk_model_schema(_NestedModel))
      assert "header" in _paths(nodes)
      assert "payload.ok" in _paths(nodes)
      assert "payload.secret" in _paths(nodes)
      assert _has_marker(nodes, "payload.secret")


  def test_descends_into_list_of_basemodel() -> None:
      nodes = list(walk_model_schema(_ListModel))
      assert "items[*].ok" in _paths(nodes)
      assert "items[*].secret" in _paths(nodes)
      assert _has_marker(nodes, "items[*].secret")


  def test_descends_into_dict_of_basemodel() -> None:
      nodes = list(walk_model_schema(_DictModel))
      assert "lookup{*}.ok" in _paths(nodes)
      assert "lookup{*}.secret" in _paths(nodes)
      assert _has_marker(nodes, "lookup{*}.secret")


  def test_descends_into_tuple_of_basemodel() -> None:
      nodes = list(walk_model_schema(_TupleModel))
      assert "pair[*].ok" in _paths(nodes)
      assert "pair[*].secret" in _paths(nodes)
      assert _has_marker(nodes, "pair[*].secret")


  def test_descends_into_optional_basemodel_arm() -> None:
      nodes = list(walk_model_schema(_OptionalModel))
      assert "maybe.ok" in _paths(nodes)
      assert "maybe.secret" in _paths(nodes)
      assert _has_marker(nodes, "maybe.secret")


  def test_descends_into_every_union_arm() -> None:
      nodes = list(walk_model_schema(_UnionModel))
      paths = _paths(nodes)
      # Both arms walked
      assert "either.ok" in paths
      assert "either.secret" in paths
      assert "either.payload.secret" in paths


  def test_marker_position_in_annotated_does_not_matter() -> None:
      """Sensitive() may appear anywhere in the Annotated tuple, not just args[0]."""
      nodes = list(walk_model_schema(_MarkerLastModel))
      assert _has_marker(nodes, "tail")


  def test_three_level_nesting_descends_fully() -> None:
      class L1(BaseModel):
          inner: _NestedModel

      class L0(BaseModel):
          outer: L1

      nodes = list(walk_model_schema(L0))
      assert "outer.inner.payload.secret" in _paths(nodes)
      assert _has_marker(nodes, "outer.inner.payload.secret")


  def test_any_field_raises_at_walk_time() -> None:
      """Any-typed fields are inspection-resistant; the iterator surfaces them so
      the adequacy guard can fail with a precise error message."""

      class M(BaseModel):
          junk: Any

      nodes = list(walk_model_schema(M))
      junk_node = next(n for n in nodes if n.path == "junk")
      assert junk_node.field_type is Any


  def test_value_provider_extracts_from_root_dict_when_with_values() -> None:
      """walker mode: each node's value_provider returns the value at the
      node's path given the root dict."""
      nodes = list(walk_model_schema(_NestedModel, with_values=True))
      root = {"header": "h", "payload": {"ok": "v", "secret": "S"}}
      ok_node = next(n for n in nodes if n.path == "payload.ok")
      assert ok_node.value_provider is not None
      assert ok_node.value_provider(root) == "v"
      secret_node = next(n for n in nodes if n.path == "payload.secret")
      assert secret_node.value_provider is not None
      assert secret_node.value_provider(root) == "S"


  def test_value_provider_handles_list_index_descent() -> None:
      """For list/dict/tuple element descents the provider returns a sequence
      of (key_or_index, value) pairs; the walker iterates."""
      nodes = list(walk_model_schema(_ListModel, with_values=True))
      root = {"items": [{"ok": "a", "secret": "X"}, {"ok": "b", "secret": "Y"}]}
      secret_node = next(n for n in nodes if n.path == "items[*].secret")
      assert secret_node.value_provider is not None
      values = list(secret_node.value_provider(root))
      assert values == [(0, "X"), (1, "Y")]


  def test_value_provider_handles_dict_key_descent() -> None:
      nodes = list(walk_model_schema(_DictModel, with_values=True))
      root = {"lookup": {"first": {"ok": "a", "secret": "X"}, "second": {"ok": "b", "secret": "Y"}}}
      secret_node = next(n for n in nodes if n.path == "lookup{*}.secret")
      assert secret_node.value_provider is not None
      values = sorted(secret_node.value_provider(root))
      assert values == [("first", "X"), ("second", "Y")]


  def test_with_values_false_yields_no_value_provider() -> None:
      """Adequacy-guard mode: walker does not need value extraction."""
      nodes = list(walk_model_schema(_FlatModel))
      assert all(n.value_provider is None for n in nodes)
  ```

  **Additional walker container-shape tests (rev-2 M_adequacy_mechanical_enforcement — 4 coverage gaps).** Add these four test functions to `test_walk_model_schema.py` before Step 2:

  ```python
  def test_walk_duplicate_sensitive_markers() -> None:
      """Duplicate _SensitiveMarker in one field's Annotated tuple raises ValueError
      (spec §4.2.5 promises this; rev-2 M_adequacy quality MAJOR-1 gap A)."""
      from typing import Annotated
      from pydantic import BaseModel
      from elspeth.web.composer.redaction import Sensitive, walk_model_schema

      class _DuplicateModel(BaseModel):
          bad: Annotated[str, Sensitive(), Sensitive()]

      with pytest.raises(ValueError, match="bad"):
          list(walk_model_schema(_DuplicateModel))


  def test_walk_list_of_list_of_basemodel() -> None:
      """Iterator descends through two list levels (rev-2 quality MAJOR-1 gap B)."""
      from typing import Annotated
      from pydantic import BaseModel
      from elspeth.web.composer.redaction import Sensitive, walk_model_schema

      class _Inner(BaseModel):
          secret: Annotated[str, Sensitive()]

      class _Outer(BaseModel):
          matrix: list[list[_Inner]]

      nodes = list(walk_model_schema(_Outer))
      paths = {n.path for n in nodes}
      assert "matrix[*][*].secret" in paths
      assert any(any(isinstance(m, _SensitiveMarker) for m in n.metadata)
                 for n in nodes if n.path == "matrix[*][*].secret")


  def test_walk_field_plus_annotated_combined() -> None:
      """Sensitive() marker is detected regardless of FieldInfo position in Annotated
      metadata tuple (rev-2 quality MAJOR-1 gap C)."""
      from typing import Annotated
      from pydantic import BaseModel, Field
      from elspeth.web.composer.redaction import Sensitive, walk_model_schema

      class _FieldAnnotatedModel(BaseModel):
          combined: Annotated[str, Field(description="desc"), Sensitive()]

      nodes = list(walk_model_schema(_FieldAnnotatedModel))
      target = next((n for n in nodes if n.path == "combined"), None)
      assert target is not None
      assert any(isinstance(m, _SensitiveMarker) for m in target.metadata)


  def test_walk_optional_annotated_scalar_arm() -> None:
      """Optional[Annotated[str, Sensitive()]] scalar-arm coverage (rev-3 W8a / rev-4 W8a).

      The earlier _OptionalModel test covers Optional[<BaseModel>]; this test
      covers the scalar arm Optional[Annotated[<scalar>, Sensitive()]] which
      goes through a different unwrap path (Optional[X] → Union[X, None] →
      Annotated[str, Sensitive()] → str). The marker must survive the Optional
      unwrap; if it doesn't, every Optional[Annotated[scalar, Sensitive()]]
      field is silently treated as non-sensitive at the runtime walker.
      """
      from typing import Annotated, Optional
      from pydantic import BaseModel
      from elspeth.web.composer.redaction import Sensitive, walk_model_schema

      class _OptionalScalarModel(BaseModel):
          maybe_secret: Optional[Annotated[str, Sensitive()]] = None

      nodes = list(walk_model_schema(_OptionalScalarModel))
      target = next((n for n in nodes if n.path == "maybe_secret"), None)
      assert target is not None, (
          "Optional[Annotated[str, Sensitive()]] field did not appear in walk output. "
          "Walker must descend into the non-None Union arm and yield a node for "
          "the wrapped scalar."
      )
      assert any(isinstance(m, _SensitiveMarker) for m in target.metadata), (
          "Optional unwrap dropped the _SensitiveMarker. The marker MUST survive "
          "Optional[Annotated[T, Sensitive()]] unwrapping; otherwise every "
          "Optional sensitive scalar in the manifest is silently non-sensitive."
      )


  def test_walk_three_arm_union() -> None:
      """Iterator descends into BaseModel arm of a 3-arm Union; skips non-BaseModel
      arms cleanly without spurious yields (rev-2 quality MAJOR-1 gap D)."""
      from typing import Union, Annotated
      from pydantic import BaseModel
      from elspeth.web.composer.redaction import Sensitive, walk_model_schema

      class _InnerWithSecret(BaseModel):
          secret: Annotated[str, Sensitive()]

      class _UnionOuter(BaseModel):
          field: Union[str, _InnerWithSecret, bool]

      nodes = list(walk_model_schema(_UnionOuter))
      paths = {n.path for n in nodes}
      # Must find the nested secret via the BaseModel arm.
      assert "field.secret" in paths
      # Must NOT find spurious scalar-arm yields.
      assert not any(p in paths for p in ("field[str]", "field[bool]"))
  ```

  **Reviewer note:** if Pydantic 2.x's introspection capability cannot represent dict-key descent cleanly, the test fails and the iterator implementation must use a custom walk. Do not work around this with a half-measure.

- [ ] **Step 2: Run the test to verify it fails.**

  ```bash
  .venv/bin/python -m pytest tests/unit/web/composer/test_walk_model_schema.py -v
  ```
  Expected: every test fails with `ImportError` on `walk_model_schema`, `TraversalNode`, `_SensitiveMarker`, `Sensitive`.

- [ ] **Step 3: Add the iterator and Sensitive primitive to redaction.py.**

  Append to `src/elspeth/web/composer/redaction.py`:

  ```python
  # ... existing content (redact_source_storage_path, REDACTED_BLOB_SOURCE_PATH) ...

  from collections.abc import Callable, Iterator
  from dataclasses import dataclass
  from types import UnionType
  from typing import Annotated, Any, TypeVar, Union, get_args, get_origin
  from pydantic import BaseModel

  T = TypeVar("T")


  class _SensitiveMarker:
      """Annotated metadata marker (spec §4.2.2)."""
      __slots__ = ("summarizer",)

      def __init__(self, summarizer: Callable[[Any], str] | None = None) -> None:
          self.summarizer = summarizer


  def Sensitive(  # noqa: N802 — capitalised to read as a type alias at use sites
      *, summarizer: Callable[[Any], str] | None = None
  ) -> _SensitiveMarker:
      """Field-level annotation requesting redaction (§4.2.2)."""
      return _SensitiveMarker(summarizer=summarizer)


  @dataclass(frozen=True, slots=True)
  class TraversalNode:
      """One field encountered while walking a model schema (§4.2.5).

      Freeze-guard elision (rev-3 A2 / W3):
        `metadata` is typed as `tuple[Any, ...]` to admit `_SensitiveMarker`
        instances in the metadata position. _SensitiveMarker is a regular
        (non-frozen) class — it holds a `summarizer` callable that we never
        mutate after construction. The freeze-guard CI tool
        (scripts/cicd/enforce_freeze_guards.py) only flags forbidden patterns
        in __post_init__; this dataclass intentionally has no __post_init__
        and no `freeze_fields()` call. The design assumption is:
          1. _SensitiveMarker instances are constructed once (at Annotated[...]
             definition time, module load) and never mutated;
          2. TraversalNode is produced inside walk_model_schema and discarded
             after iteration — there is no long-lived reference path that
             would expose mutation of a marker;
          3. All other metadata entries are either pydantic.FieldInfo (built-in
             immutable for our usage) or scalar/None.
        If a future change introduces stateful metadata objects, ADD a
        freeze_fields() call here AND a deep_freeze() of `metadata` — do not
        rely on the type signature alone, which `tuple[Any, ...]` does not
        constrain.
      """
      path: str
      field_type: type
      metadata: tuple[Any, ...]
      value_provider: Callable[[dict[str, Any]], Any] | None = None


  def walk_model_schema(
      model: type[BaseModel],
      *,
      with_values: bool = False,
      _path_prefix: str = "",
      _value_path: tuple[str | int, ...] = (),
  ) -> Iterator[TraversalNode]:
      """Yield TraversalNode per field; descend per §4.2.5 rules."""
      # ... full implementation ...
  ```

  Implementation details (the agent may decide test names + helper internals; the iterator's externally observable behaviour is fixed by the test set above):

  - Use `model.model_fields` to enumerate.
  - For each field, unwrap `Annotated[T, *, m1, *, m2]` → metadata tuple is the full args tail; `field_type` is `args[0]`.
  - Use `get_origin(field_type)` to detect container; `get_args(field_type)` to extract element type.
  - Containers descended into: `list`, `dict`, `tuple`, `Optional`/`Union` (Python 3.10+ `X | None` syntax via `types.UnionType`).
  - For `dict[str, X]`: key type must be `str` (assert at walk time); element descent yields path with `{*}`.
  - For `list[X]` / `tuple[X, ...]`: element descent yields path with `[*]`.
  - For `Optional[X]` / `Union[*, X, *]`: descend into every non-`None` arm; emit one node per arm with the same path.
  - `Any`/`object` typed fields: yield a node with `field_type=Any`; do not descend.
  - Cycle handling: rely on Pydantic's model-resolution; the iterator does not maintain a visited set.
  - `value_provider` when `with_values=True`: a closure over `_value_path` that walks the root dict; for container descents the provider returns an iterator over `(key_or_index, value)` pairs.
  - **Sensitive short-circuit (rev-5 architecture A1 — forward-guard).** When `walk_model_schema` yields a node carrying `_SensitiveMarker` in its metadata, it MUST NOT descend further into that node's container or sub-model. The redactor is going to replace the whole sub-tree with the summarizer output (or fixed sentinel); any inner Sensitive markers below that point are unreachable in the redacted view, and yielding them would cause the property test's `value_provider(redacted_args)` call at the inner path to attempt iteration of the (now string) summarizer output, raising a confusing `TypeError`. No current MANIFEST entry uses nested `Annotated[..., Sensitive()]`-inside-`Annotated[container, Sensitive()]`, so this is latent — but a future promotion that uses `Annotated[dict[str, InnerModel], Sensitive()]` where `InnerModel` has `Sensitive` fields would trip the bug without this discipline. Add a test in `test_walk_model_schema.py`:

  ```python
  def test_walk_short_circuits_under_sensitive_container() -> None:
      """A Sensitive marker on a container suppresses descent into inner Sensitive markers.

      Closes rev-5 architecture A1. The redactor replaces the whole container's
      value with the summarizer output; inner markers below that point have no
      reachable values in the redacted view. The walker therefore must NOT yield
      them, or the property test's value_provider(redacted) call at the inner
      path will attempt to iterate the summarizer's string output and raise
      TypeError.
      """
      from typing import Annotated
      from pydantic import BaseModel
      from elspeth.web.composer.redaction import Sensitive, walk_model_schema, _SensitiveMarker

      class _InnerWithSecret(BaseModel):
          inner_secret: Annotated[str, Sensitive()]

      class _OuterSensitiveContainer(BaseModel):
          outer: Annotated[list[_InnerWithSecret], Sensitive(summarizer=lambda v: f"<list:{len(v)}>")]

      nodes = list(walk_model_schema(_OuterSensitiveContainer))
      paths = {n.path for n in nodes}
      # The outer container is yielded with its marker.
      assert "outer" in paths
      outer = next(n for n in nodes if n.path == "outer")
      assert any(isinstance(m, _SensitiveMarker) for m in outer.metadata)
      # The inner secret is NOT yielded — the outer Sensitive short-circuits descent.
      assert "outer[*].inner_secret" not in paths, (
          "Walker descended past a Sensitive-marked container into an inner "
          "Sensitive node. This is forbidden because the redactor replaces the "
          "whole container with the summarizer output; the inner path has no "
          "reachable value in the redacted view and yielding it would cause "
          "the property test to raise TypeError on value_provider(redacted)."
      )
  ```

- [ ] **Step 4: Run tests to verify they pass.**

  ```bash
  .venv/bin/python -m pytest tests/unit/web/composer/test_walk_model_schema.py -v
  ```

- [ ] **Step 5: Run the project gate.**

  ```bash
  .venv/bin/python -m mypy src/elspeth/web/composer/redaction.py
  .venv/bin/python -m ruff check src/elspeth/web/composer/redaction.py
  ```

- [ ] **Step 6: Commit.**

  Commit message: `feat(composer/redaction): shared traversal iterator (spec §4.2.5)`.

  Include in the commit body: "Closes plan-review B2 by establishing a single iterator that both the adequacy guard (§4.4) and the runtime walker (§4.2.6) consume."

**Reviewer assignments:**

- Implementer: agent.
- Spec reviewer (post-implementation, before next task): re-read §4.2.5 and confirm the iterator's externally observable behaviour matches.
- Code-quality reviewer: confirm no `hasattr` / `.get(` / `getattr(_, _, default)` defensive patterns; direct typed-attribute access throughout.

---

## Task 2: `ToolRedaction` manifest-entry dataclass

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_tool_redaction_dataclass.py`

**Steps:**

- [ ] **Step 1: Write the failing test.**

  ```python
  """Tests for the ToolRedaction manifest-entry dataclass (spec §4.2.1).

  Construction errors are precondition contracts; any future bug that
  introduces both-shapes-set or neither-shape-set should fail at import
  time, not at walk time.
  """

  from __future__ import annotations

  import pytest
  from pydantic import BaseModel

  from elspeth.web.composer.redaction import (
      HandlesNoSensitiveDataReason,
      ToolRedaction,
      ToolRedactionPolicy,
  )


  class _DummyModel(BaseModel):
      x: str


  def _ok_reason() -> HandlesNoSensitiveDataReason:
      return HandlesNoSensitiveDataReason(
          sensitive_data_locations=("nowhere — see test",),
          why_arguments_safe="A" * 32,
          why_responses_safe="B" * 32,
      )


  def _ok_policy_no_sensitive() -> ToolRedactionPolicy:
      return ToolRedactionPolicy(
          handles_no_sensitive_data=True,
          handles_no_sensitive_data_reason_struct=_ok_reason(),
      )


  def test_type_driven_entry_is_constructable() -> None:
      entry = ToolRedaction(argument_model=_DummyModel)
      assert entry.argument_model is _DummyModel
      assert entry.policy is None


  def test_declarative_entry_is_constructable() -> None:
      policy = _ok_policy_no_sensitive()
      entry = ToolRedaction(policy=policy)
      assert entry.policy is policy
      assert entry.argument_model is None


  def test_both_shapes_set_raises_value_error() -> None:
      with pytest.raises(ValueError, match="both argument_model and policy"):
          ToolRedaction(argument_model=_DummyModel, policy=_ok_policy_no_sensitive())


  def test_neither_shape_set_raises_value_error() -> None:
      with pytest.raises(ValueError, match="neither argument_model nor policy"):
          ToolRedaction()


  def test_response_model_without_argument_model_raises() -> None:
      with pytest.raises(ValueError, match="response_model requires argument_model"):
          ToolRedaction(response_model=_DummyModel, policy=_ok_policy_no_sensitive())
  ```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Add `ToolRedaction` to `redaction.py`.** Use the spec §4.2.1 sketch verbatim.

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Run full gate slice.**

- [ ] **Step 6: Commit.** `feat(composer/redaction): ToolRedaction manifest-entry dataclass (spec §4.2.1)`. Note in body: "MANIFEST is empty until subsequent tasks populate it. Adequacy guard (Task 9) will be red until Task 16 finishes."

---

## Task 3: `RedactionTelemetry` Protocol + `NoopRedactionTelemetry`

**Files:**

- Create: `src/elspeth/web/composer/redaction_telemetry.py`
- Create: `tests/unit/web/composer/test_redaction_telemetry.py`

**Steps:**

- [ ] **Step 1: Write the failing test.**

  ```python
  """Tests for RedactionTelemetry Protocol and NoopRedactionTelemetry impl.

  Closes plan-review W4 (telemetry duck-typing). The walker accepts a
  typed Protocol instance, never None.

  Rev-2 M_telemetry_implementation: the OtelRedactionTelemetry impl uses
  module-level create_counter() objects + .add() calls per the project's
  established pattern at service.py:135, 148, 824, 868, 1172, 1182.
  There is NO _increment_counter helper — that function does not exist.
  """
  from __future__ import annotations

  from unittest.mock import MagicMock, call

  from elspeth.web.composer.redaction_telemetry import (
      NoopRedactionTelemetry,
      RedactionTelemetry,
  )


  def test_noop_implements_protocol() -> None:
      noop: RedactionTelemetry = NoopRedactionTelemetry()
      noop.unknown_response_key_redacted(tool_name="t")
      noop.manifest_dispatch(tool_name="t", shape="declarative")
      noop.summarizer_error(tool_name="t")


  def test_noop_records_for_assertion_in_tests() -> None:
      noop = NoopRedactionTelemetry()
      noop.unknown_response_key_redacted(tool_name="set_source")
      noop.manifest_dispatch(tool_name="set_source", shape="type_driven")
      noop.summarizer_error(tool_name="set_source")
      assert noop.unknown_response_key_calls == [{"tool_name": "set_source"}]
      assert noop.manifest_dispatch_calls == [{"tool_name": "set_source", "shape": "type_driven"}]
      assert noop.summarizer_error_calls == [{"tool_name": "set_source"}]


  def test_otel_telemetry_emits_via_module_level_counters(monkeypatch) -> None:
      """Production impl uses module-level counter objects + .add() calls.

      Rev-2 M_telemetry_implementation: patch the counter objects themselves,
      not a nonexistent _increment_counter helper. The established OTel pattern
      in this project (service.py:135, 148, 824, 868, 1172, 1182) is:
          _FOO_COUNTER = metrics.get_meter(__name__).create_counter(...)
          _FOO_COUNTER.add(1, {"label_key": value})
      """
      import elspeth.web.composer.redaction_telemetry as rt_mod
      from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry

      mock_unknown = MagicMock()
      mock_dispatch = MagicMock()
      mock_summarizer = MagicMock()

      monkeypatch.setattr(rt_mod, "_UNKNOWN_RESPONSE_KEY_COUNTER", mock_unknown)
      monkeypatch.setattr(rt_mod, "_MANIFEST_DISPATCH_COUNTER", mock_dispatch)
      monkeypatch.setattr(rt_mod, "_SUMMARIZER_ERROR_COUNTER", mock_summarizer)

      tel = OtelRedactionTelemetry()
      tel.unknown_response_key_redacted(tool_name="set_source")
      tel.manifest_dispatch(tool_name="set_source", shape="type_driven")
      tel.summarizer_error(tool_name="set_source")

      mock_unknown.add.assert_called_once_with(1, {"tool_name": "set_source"})
      mock_dispatch.add.assert_called_once_with(1, {"tool_name": "set_source", "shape": "type_driven"})
      mock_summarizer.add.assert_called_once_with(1, {"tool_name": "set_source"})
  ```

- [ ] **Step 2: Run, expect fail.**

- [ ] **Step 3: Add the module.**

  ```python
  """OTel surface for the redaction walker (spec §4.2.4).

  Rev-2 M_telemetry_implementation: uses module-level create_counter() objects
  and .add() calls per the project's established pattern (service.py:135, 148,
  824, 868, 1172, 1182). No _increment_counter helper — that does not exist.
  """
  from __future__ import annotations

  from typing import Protocol

  from opentelemetry import metrics

  _meter = metrics.get_meter(__name__)

  _UNKNOWN_RESPONSE_KEY_COUNTER = _meter.create_counter(
      "composer.redaction.unknown_response_key_redacted",
      description="Count of unknown response keys substituted with the fixed sentinel.",
  )

  _MANIFEST_DISPATCH_COUNTER = _meter.create_counter(
      "composer.redaction.manifest_dispatch",
      description="Count of tool calls dispatched through the redaction manifest.",
  )

  _SUMMARIZER_ERROR_COUNTER = _meter.create_counter(
      "composer.redaction.summarizer_errors_total",
      description="Count of summarizer failures (exception OR non-string return) immediately before AuditIntegrityError raise.",
  )


  class RedactionTelemetry(Protocol):
      def unknown_response_key_redacted(self, *, tool_name: str) -> None: ...
      def manifest_dispatch(self, *, tool_name: str, shape: str) -> None: ...
      def summarizer_error(self, *, tool_name: str) -> None:
          """Incremented immediately before AuditIntegrityError raise on
          summarizer exception or non-str return. Wired in Task 7's walker code
          path. (Rev-2 M_telemetry_implementation / M.8)"""
          ...


  class NoopRedactionTelemetry:
      """In-memory test impl. Records every call; assertable."""
      def __init__(self) -> None:
          self.unknown_response_key_calls: list[dict[str, str]] = []
          self.manifest_dispatch_calls: list[dict[str, str]] = []
          self.summarizer_error_calls: list[dict[str, str]] = []

      def unknown_response_key_redacted(self, *, tool_name: str) -> None:
          self.unknown_response_key_calls.append({"tool_name": tool_name})

      def manifest_dispatch(self, *, tool_name: str, shape: str) -> None:
          self.manifest_dispatch_calls.append({"tool_name": tool_name, "shape": shape})

      def summarizer_error(self, *, tool_name: str) -> None:
          self.summarizer_error_calls.append({"tool_name": tool_name})


  class OtelRedactionTelemetry:
      """Production impl. Emits via module-level OTel counter objects."""

      def unknown_response_key_redacted(self, *, tool_name: str) -> None:
          _UNKNOWN_RESPONSE_KEY_COUNTER.add(1, {"tool_name": tool_name})

      def manifest_dispatch(self, *, tool_name: str, shape: str) -> None:
          _MANIFEST_DISPATCH_COUNTER.add(1, {"tool_name": tool_name, "shape": shape})

      def summarizer_error(self, *, tool_name: str) -> None:
          _SUMMARIZER_ERROR_COUNTER.add(1, {"tool_name": tool_name})
  ```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Gate slice + tier-model check (the new module is L3; verify no upward imports).**

- [ ] **Step 6: Commit.** `feat(composer/redaction): typed RedactionTelemetry Protocol (spec §4.2.4, closes W4)`.

---

## Task 4 (TRACER BULLET): `set_source` end-to-end promotion

**Why this task is fourth.** Foundational pieces (iterator + manifest dataclass + telemetry) are in place; one tool is promoted end-to-end before bulk migration so integration assumptions about the LLM dispatch path are validated against a real handler before bulk-migration work locks them in. If `model_validate` at the dispatch boundary surfaces an unexpected behaviour (e.g. interaction with the `_TOOL_REQUIRED_PATHS` schema-validation step at `service.py:2023`), this task surfaces it; subsequent waves do not.

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py` (add `redact_tool_call_arguments` minimal impl + first MANIFEST entry; subsequent tasks generalise the impl)
- Modify: `src/elspeth/web/composer/tools.py` (promote `_execute_set_source`)
- Create: `tests/unit/web/composer/test_redact_set_source.py`

**Steps:**

- [ ] **Step 1: Read the existing handler.** Open `src/elspeth/web/composer/tools.py` at `_execute_set_source` (line ~2299). Inspect every `arguments[KEY]` and `arguments.get(KEY)` site. Record the argument schema (paths, types, defaults). Cross-reference `_TOOL_REQUIRED_PATHS["set_source"]` at `src/elspeth/web/composer/service.py` to confirm which paths are required by the existing schema check; the Pydantic model must be at least as strict (else the schema check becomes load-bearing in a way the model doesn't capture). The Pydantic model SHOULD be exactly as strict — duplication is a divergence risk.

- [ ] **Step 2: Write the failing test.**

  ```python
  """Tracer-bullet: set_source end-to-end through manifest + walker (spec §11)."""
  from __future__ import annotations

  from typing import Annotated, Any

  import pytest
  from pydantic import BaseModel, ValidationError

  from elspeth.web.composer.redaction import (
      MANIFEST,
      REDACTED_BLOB_SOURCE_PATH,
      Sensitive,
      SetSourceArgumentsModel,
      ToolRedaction,
      redact_source_storage_path,
      redact_tool_call_arguments,
  )
  from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


  def test_set_source_manifest_entry_is_type_driven() -> None:
      entry = MANIFEST["set_source"]
      assert entry.argument_model is SetSourceArgumentsModel
      assert entry.policy is None


  def test_set_source_argument_model_validates_real_llm_shape() -> None:
      llm_args = {
          "plugin": "csv",
          "options": {"path": "/tmp/data.csv", "header": True},
          "on_success": "continue",
      }
      validated = SetSourceArgumentsModel.model_validate(llm_args)
      assert validated.plugin == "csv"
      assert validated.options == {"path": "/tmp/data.csv", "header": True}
      assert validated.on_success == "continue"


  def test_set_source_argument_model_rejects_missing_required() -> None:
      with pytest.raises(ValidationError):
          SetSourceArgumentsModel.model_validate({})


  def test_set_source_argument_model_rejects_wrong_type() -> None:
      with pytest.raises(ValidationError):
          SetSourceArgumentsModel.model_validate({"plugin": 42, "options": {}})


  def test_redact_substitutes_options_via_summarizer() -> None:
      tel = NoopRedactionTelemetry()
      args = {
          "plugin": "csv",
          "options": {"path": "/internal/blob/path.csv", "blob_ref": "abc"},
      }
      redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
      assert redacted["plugin"] == "csv"
      # options.path went through redact_source_storage_path
      assert redacted["options"]["path"] == REDACTED_BLOB_SOURCE_PATH
      assert tel.manifest_dispatch_calls == [
          {"tool_name": "set_source", "shape": "type_driven"}
      ]


  def test_redact_passes_through_when_no_blob_ref() -> None:
      """redact_source_storage_path is the summarizer; absence of blob_ref
      means options is preserved (existing behaviour at redaction.py:15-42)."""
      tel = NoopRedactionTelemetry()
      args = {"plugin": "csv", "options": {"path": "/tmp/data.csv"}}
      redacted = redact_tool_call_arguments("set_source", args, telemetry=tel)
      assert redacted["options"]["path"] == "/tmp/data.csv"
  ```

- [ ] **Step 3: Implement the model and the minimal walker.**

  In `src/elspeth/web/composer/redaction.py`:

  ```python
  from pydantic import BaseModel

  class SetSourceArgumentsModel(BaseModel):
      """Redaction-bearing argument model for the set_source tool.

      Mirrors the JSON schema currently consumed by _execute_set_source
      (tools.py:~2299) and the required-paths check at service.py:702
      _TOOL_REQUIRED_PATHS["set_source"]. The annotation on `options`
      drives mechanical redaction at the persistence boundary; the model
      itself is also used at the dispatch boundary by Model.model_validate
      so the LLM-supplied dict is validated before _execute_set_source
      reads any field.
      """
      plugin: str
      options: Annotated[dict[str, Any], Sensitive(summarizer=_summarize_set_source_options)]
      on_success: str | None = None
      on_validation_failure: str | None = None
      label: str | None = None
      blob_id: str | None = None
      inline_blob: dict[str, Any] | None = None
      model_config = ConfigDict(extra="forbid")  # rev-2 M.1 — prevents argument_canonical / walker discrepancy
      # ... any additional fields required by the existing schema ...


  def _summarize_set_source_options(options: dict[str, Any]) -> str:
      """Summarizer for set_source.options. Wraps the existing
      redact_source_storage_path helper. Contract: must not raise on any
      reachable input value; must return str (spec §4.2.6 / §9 RSK-03).
      default=str ensures spec §9 RSK-03 ('summarizer must not raise on any
      reachable input value') holds for Pydantic-coerced non-JSON-primitive
      values (datetime, bytes, UUID, etc.) that can flow through Any-typed
      fields after Pydantic coercion."""
      # The existing helper returns dict[str, Any]; we must return str.
      # Wrap by re-encoding to a sentinel-marked JSON. Decision documented
      # in the commit body; the simplest correct shape is to call the
      # existing helper for path redaction and re-stringify.
      redacted = redact_source_storage_path({"source": {"options": options}})
      return json.dumps(redacted["source"]["options"], sort_keys=True, separators=(",", ":"), default=str)


  def redact_tool_call_arguments(
      tool_name: str,
      arguments: dict[str, Any],
      *,
      telemetry: RedactionTelemetry,
  ) -> dict[str, Any]:
      """Minimal tracer-bullet impl. Generalised in Task 8."""
      entry = MANIFEST[tool_name]  # KeyError → AuditIntegrityError in full impl
      if entry.argument_model is not None:
          telemetry.manifest_dispatch(tool_name=tool_name, shape="type_driven")
          # Validate. Callers (promoted handlers) MUST catch pydantic.ValidationError
          # and re-raise as ToolArgumentError — the compose loop catches
          # ToolArgumentError at service.py:2480 and routes to ARG_ERROR.
          validated = entry.argument_model.model_validate(arguments)
          # Walk the model's fields, substituting Sensitive-marked ones.
          return _redact_via_schema(validated, entry.argument_model)
      # ... declarative branch added in Task 8 ...
      raise NotImplementedError("declarative branch — see Task 8")


  from types import MappingProxyType

  MANIFEST: Mapping[str, ToolRedaction] = MappingProxyType({
      "set_source": ToolRedaction(argument_model=SetSourceArgumentsModel),
  })
  ```

  Per spec §4.2.1 the manifest is `Mapping[str, ToolRedaction]` wrapped in `MappingProxyType` — module-level immutability matches the project's freeze-guard discipline (CLAUDE.md "Frozen Dataclass Immutability"; mutable module-level dicts have caused import-cycle accidents elsewhere in the project). Subsequent task waves extend the manifest by **building a new dict, then replacing the module-level binding** — never by mutating the proxy view.

  **Important (W5 / spec §9 RSK-03):** the summarizer wrapper above MUST return `str`. The walker asserts the return type and raises `AuditIntegrityError` on any non-`str` return. Test the summarizer returns `str` for every reachable input — this is part of the property test (Task 19 gate). Specific test case required: the summarizer must accept a Pydantic-coerced `datetime` value in the options dict and return a `str` without raising `TypeError` (closes rev-3 A7; `json.dumps` raises `TypeError` on `datetime` without `default=str`).

- [ ] **Step 3a: Add the rev-3 A7 datetime-coercion test case.**

  Add a test function to `tests/unit/web/composer/test_redact_set_source.py` (the test file from Step 2) that mechanically pins the `default=str` fix. Without this test, the `default=str` argument can silently regress and the spec §9 RSK-03 "summarizer must not raise" contract is only protected by Task 19's property test (which uses Hypothesis-generated values and may not happen to generate a `datetime`).

  ```python
  def test_summarize_set_source_options_accepts_coerced_datetime() -> None:
      """Pin rev-3 A7: spec §9 RSK-03 requires the summarizer must not raise
      on any reachable input value. Pydantic 2.x coerces some string inputs to
      `datetime` when the underlying field allows it; `json.dumps` raises
      `TypeError` on `datetime` unless `default=str` is supplied. This test
      pins the `default=str` argument so a future refactor that removes it
      fails loudly here rather than silently violating RSK-03."""
      from datetime import datetime, timezone
      from elspeth.web.composer.redaction import _summarize_set_source_options

      options = {"since": datetime(2026, 1, 1, tzinfo=timezone.utc), "key": "v"}
      result = _summarize_set_source_options(options)
      assert isinstance(result, str)
  ```

- [ ] **Step 4: Promote `_execute_set_source` in `tools.py`.**

  At the top of `_execute_set_source`:

  ```python
  from elspeth.web.composer.redaction import SetSourceArgumentsModel

  def _execute_set_source(
      arguments: dict[str, Any],
      state: CompositionState,
      catalog: CatalogService,
      data_dir: str | None = None,
  ) -> ToolResult:
      # Validate the LLM-supplied arguments against the redaction-bearing
      # model. MUST catch pydantic.ValidationError and re-raise as
      # ToolArgumentError so the compose loop's ToolArgumentError handler at
      # service.py:2480 routes to ARG_ERROR. A bare ValidationError escaping
      # here hits service.py:2564 (catch-all → ComposerPluginCrashError →
      # HTTP 500), which is the wrong disposition for Tier-3 input.
      # Pattern from tools.py:2668, 2761, 2767, 2773, 2787, 2801.
      try:
          validated = SetSourceArgumentsModel.model_validate(arguments)
      except pydantic.ValidationError as exc:
          raise ToolArgumentError(
              argument="set_source arguments",
              expected="schema-conformant dict",
              actual_type=type(exc).__name__,
          ) from exc
      # Read typed attributes; the previous arguments["plugin"] etc. are
      # replaced with validated.plugin etc.
      plugin = validated.plugin
      options = validated.options
      # ... rest of the handler ...
  ```

  Also: **REMOVE** `_TOOL_REQUIRED_PATHS["set_source"]` **unconditionally** in the same commit that promotes the handler. (Note: rev-5 reality check confirmed there is no `_TOOL_OPTIONAL_PATHS` companion in `service.py` — an earlier draft of this plan speculatively referenced one. Only the required-paths dict exists; remove the `set_source` entry from it.) The Pydantic model with `model_config = ConfigDict(extra="forbid")` and typed required fields is strictly more capable than the dotted-path check at `service.py:2023`: it covers every required-path assertion, PLUS type rejection, PLUS extra-key rejection, PLUS Sensitive[T] declaration. Keeping both is forbidden by CLAUDE.md "No Legacy Code Policy" — two checks that do similar things at different layers always drift. Rev-3 N7 and rev-4 M1 both flagged the previously-conditional language as a deferral; this revision makes the removal mandatory.

  **MissingRequiredPaths test-pin update (rev-4 M2 — must be in the same commit).** Removing the `_TOOL_REQUIRED_PATHS["set_source"]` entry means the audit `error_class` emitted at `service.py:2039` (`error_class="MissingRequiredPaths"`) no longer fires for `set_source` — invalid `set_source` arguments now route through `ToolArgumentError` → ARG_ERROR with `error_class` derived from the `pydantic.ValidationError` class name. Any test that pinned the literal string `"MissingRequiredPaths"` on a `set_source` failure scenario will break with a stale assertion. Run:

  ```bash
  grep -Rn '"MissingRequiredPaths"\|MissingRequiredPaths' tests/ src/elspeth/web/ --include='*.py'
  ```

  Inspect every match. For each match in `tests/` whose test scenario invokes the `set_source` handler (directly or via compose-loop) with invalid arguments:

  1. Read the test's intent. The pre-rev-4 expectation was the `_TOOL_REQUIRED_PATHS` path; the post-rev-4 expectation is the `ValidationError` → `ToolArgumentError` path. The audit record will carry `error_class="ValidationError"` (or whatever `type(exc).__name__` returns for the actual `pydantic.ValidationError`) and `error_message` will be the class name, not the dotted-path enumeration.
  2. Update the assertion to match the new error_class string. Where the old test asserted `"MissingRequiredPaths" in result.message`, the new test asserts `"ValidationError" in result.message` AND that the missing-field information appears in the audit `error_payload` (Pydantic ValidationError messages include field paths). Cross-check by reading the actual recorded message from a working test run before committing.
  3. If a match exercises a NON-promoted tool (its `_TOOL_REQUIRED_PATHS` entry remains because Task 13/14/15 has not yet promoted it), **leave it untouched**. Document in the commit message which matches were left for later tasks.

  **Confirmed pins (rev-4 reality + reviewers).** The following sites are known to require update — verify each is addressed before commit:

  - `tests/unit/web/composer/test_service.py:367` — asserts `"MissingRequiredPaths" in result.message` for a `set_pipeline` failure (note: `set_pipeline`, not `set_source`; this site updates in Task 14 along with the `set_pipeline` promotion, NOT here in Task 4. Leave it alone in this commit; document why in the commit body so the next agent does not "fix" it prematurely).
  - Any test in `tests/unit/web/composer/test_service.py`, `test_compose_loop_*.py`, or `test_promote_set_source.py` that drives an invalid `set_source` payload through the dispatch path. The Task 5 ARG_ERROR regression test (added below in Step 5) is the new positive pin for this code path.

  **Memory cross-reference:** `feedback_locked_in_buggy_expectations` — the wave of failures here is the bug landing visibly, not a regression. Update the tests to assert the new (correct) error class rather than rolling back the unconditional removal.

- [ ] **Step 5: Add a regression test that ValidationError routes through ARG_ERROR.**

  `tests/unit/web/composer/test_promote_set_source.py`:

  ```python
  """ARG_ERROR routing for set_source ToolArgumentError (spec §11 done-when).

  Rev-2 BLOCKER_A: promoted handlers MUST catch pydantic.ValidationError and
  re-raise as ToolArgumentError. The compose loop's ToolArgumentError handler
  at service.py:2480 routes to ARG_ERROR. A bare ValidationError escaping the
  handler hits service.py:2564 (→ ComposerPluginCrashError → HTTP 500) — wrong
  disposition for Tier-3 input.
  """
  from __future__ import annotations

  import pydantic
  import pytest

  from elspeth.web.composer.protocol import ToolArgumentError
  from elspeth.web.composer.tools import _execute_set_source
  # ... fixtures: build a CompositionState, a CatalogService, a data_dir ...


  def test_invalid_arguments_raise_tool_argument_error(state, catalog, tmp_path) -> None:
      """The handler catches ValidationError and re-raises as ToolArgumentError.
      The compose loop's handler at service.py:2480 catches ToolArgumentError
      and routes to ARG_ERROR (out-of-scope for this unit test)."""
      with pytest.raises(ToolArgumentError) as exc_info:
          _execute_set_source({}, state, catalog, str(tmp_path))
      # __cause__ must be the pydantic ValidationError (from exc discipline)
      assert isinstance(exc_info.value.__cause__, pydantic.ValidationError)


  def test_valid_arguments_dispatch_normally(state, catalog, tmp_path) -> None:
      """Sanity: the typed-attribute access path is exercised."""
      result = _execute_set_source(
          {"plugin": "csv", "options": {"path": str(tmp_path / "x.csv")}},
          state, catalog, str(tmp_path),
      )
      assert result.success
  ```

- [ ] **Step 6: Run all redaction tests + `_execute_set_source` regression suite.**

  ```bash
  .venv/bin/python -m pytest tests/unit/web/composer/test_redact_set_source.py tests/unit/web/composer/test_promote_set_source.py -v
  ```

- [ ] **Step 7: Run the existing set_source integration tests.**

  ```bash
  .venv/bin/python -m pytest tests/ -k "set_source" -v
  ```

  Expected: all existing tests pass (the typed-attribute change is internal; behaviour is unchanged for valid inputs). If any test fails because it was passing invalid arguments (e.g. missing required key) and the model now rejects, update the test to use valid arguments — that's the locked-in-buggy-expectations pattern (memory: `feedback_locked_in_buggy_expectations`).

- [ ] **Step 8 (NEW — rev-2 BLOCKER_A serialization boundary test): Pin the serialization boundary.**

  Add to `test_redact_set_source.py` (or a dedicated `test_redact_set_source_serialization.py`):

  ```python
  import json
  from elspeth.web.composer.redaction import redact_tool_call_arguments
  from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

  _CANARY = "CANARY-SENSITIVE-PATH-DO-NOT-LEAK"

  def test_serialization_boundary_canary_not_in_json_output() -> None:
      """Pins that the sensitive value does NOT appear in json.dumps output.

      This is the cross-boundary integration test (rev-2 BLOCKER_A quality
      MAJOR-2). Phase 3 will pass the result of redact_tool_call_arguments
      through json.dumps before writing to chat_messages.tool_calls; this
      test verifies the canary never survives that serialization.
      """
      args = {"plugin": "csv", "options": {"path": _CANARY}}
      result = redact_tool_call_arguments("set_source", args, telemetry=NoopRedactionTelemetry())
      serialized = json.dumps(result, sort_keys=True)
      assert _CANARY not in serialized, (
          f"Sensitive canary value appeared in serialized output. "
          f"Redaction did not remove it from the persistence path. "
          f"Serialized: {serialized!r}"
      )
      # The key must still be present — redaction replaces the value, not the key.
      assert "options" in serialized
  ```

- [ ] **Step 9: Run the project gate.**

- [ ] **Step 10: Commit.**

  ```
  feat(composer/redaction): tracer-bullet — promote set_source to type-driven manifest entry

  • Adds SetSourceArgumentsModel with extra="forbid" + Sensitive[T] annotation on options.
  • _execute_set_source now validates via Model.model_validate at the
    dispatch boundary; handler catches pydantic.ValidationError and re-raises
    as ToolArgumentError (pattern at tools.py:2668-2801); ToolArgumentError
    caught at service.py:2480 routes to ARG_ERROR.
  • Adds redact_tool_call_arguments minimal impl (type-driven branch
    only); generalised in subsequent tasks.
  • First MANIFEST entry. Adequacy guard registry-equality assertion
    will remain red until Task 16 completes.
  • Adds serialization boundary canary test pinning Phase 3 integration
    contract (rev-2 BLOCKER_A).

  Validates the integration end-to-end on one path before the bulk
  Sensitive[T] promotion wave (Tasks 13-15) and declarative manifest
  population (Task 16).
  ```

**Reviewer assignments:** implementer + spec reviewer (confirm §11 done-when bullet 3 is satisfied for set_source) + code-quality reviewer (no defensive patterns, typed-attribute access throughout the rewritten handler).

---

## Task 5: `HandlesNoSensitiveDataReason` structured dataclass

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_handles_no_sensitive_data_reason.py`

Standard TDD task. Test cases (must all be present before implementation):

- Empty `sensitive_data_locations` raises `ValueError` with explicit guidance.
- `why_arguments_safe` shorter than 32 chars raises with explicit guidance.
- `why_responses_safe` shorter than 32 chars raises with explicit guidance.
- Whitespace-only `why_arguments_safe` (e.g. `"   " * 11`) is rejected (`.strip()` semantics).
- Frozen container check: `sensitive_data_locations` is a `tuple[str, ...]` after `__post_init__`, immutable.
- Identity-preserving idempotency: a second `__post_init__` call on the same instance is a no-op (the `freeze_fields` contract).

Implementation copies §4.2.3 sketch verbatim; **no `last_reviewed_iso` field** (closes W2). Mass-copy uniqueness is enforced cross-entry by the adequacy guard (Task 11), not by `__post_init__`.

Commit: `feat(composer/redaction): HandlesNoSensitiveDataReason structured dataclass (§4.2.3, closes W2 calendar synchronicity)`.

---

## Task 6: `ToolRedactionPolicy` declarative dataclass

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_tool_redaction_policy.py`

TDD task. Test cases per spec §4.2.3:

- Orphan summarizer (key in `argument_summarizers` but not in `sensitive_argument_keys`) raises.
- `handles_no_sensitive_data=True` without `handles_no_sensitive_data_reason_struct` raises.
- `handles_no_sensitive_data=False` with a non-`None` `handles_no_sensitive_data_reason_struct` raises.
- `handles_no_sensitive_data=False` without `known_response_keys` raises.
- A construction with `handles_no_sensitive_data=False`, `sensitive_argument_keys=("path",)`, `argument_summarizers={"path": _redact}`, `known_response_keys=("status",)` succeeds.
- Frozen container: every named tuple/Mapping field is deeply frozen.

Commit: `feat(composer/redaction): ToolRedactionPolicy declarative manifest-entry shape (§4.2.3)`.

---

## Task 7: `redact_tool_call_response` walker

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_redact_tool_call_response.py`

**Why this comes before the full `redact_tool_call_arguments`.** The response walker exercises the unknown-key fail-closed sentinel path (closes W6), the summarizer-crash path (closes M2 / W5), and the non-`str`-return path (closes W5). These are the rev-5 boundary disciplines that diverge most sharply from rev-4; getting them right in isolation before the bigger argument walker is built keeps the failure modes legible.

TDD task. Test cases:

- Response with all keys in `known_response_keys` and none in `sensitive_response_keys`: passthrough.
- Response with key in `sensitive_response_keys` (no summarizer): substituted with `<redacted>`.
- Response with key in `sensitive_response_keys` and summarizer present: substituted with summarizer output.
- Response with unknown key (not in any declared set): substituted with the **fixed sentinel `<redacted-unknown-response-key>`** using `==` (string equality, not regex or prefix match — the sentinel is security-relevant; closes W6); `telemetry.unknown_response_key_redacted(tool_name=...)` called once per unknown key.
- Response shape that satisfies a type-driven entry's `response_model`: walked via `walk_model_schema(response_model, with_values=True)`; `Sensitive`-annotated fields substituted.
- Summarizer raises → `AuditIntegrityError` chained from the underlying exception. **Immediately before `raise AuditIntegrityError(...)`, call `telemetry.summarizer_error(tool_name=tool_name)` so the counter fires before the request dies (rev-2 M.8).** The test MUST assert `telemetry.summarizer_error_calls` was called once in this path; use `NoopRedactionTelemetry` to capture it. The test must also assert the new exception is registered in `TIER_1_ERRORS` (spec §9 RSK-03 / §4.5).
- Summarizer returns non-`str` (returns `dict`, `int`, `None`, etc.) → `AuditIntegrityError` with a typed message. **Same `telemetry.summarizer_error(tool_name=tool_name)` call immediately before `raise` (rev-2 M.8).** Test asserts `telemetry.summarizer_error_calls` was called.
- Manifest entry missing for `tool_name` → `AuditIntegrityError`.
- **Empty `known_response_keys` (rev-3 W8c / rev-4 W8c).** Declarative entry with `known_response_keys=()` and a non-empty response payload: every key in the response is unknown, so every value is replaced with the fixed sentinel `<redacted-unknown-response-key>` and `telemetry.unknown_response_key_redacted(tool_name=...)` is called once per key. This pins fail-closed behaviour for the degenerate case where the schema author has declared no known shape — the walker must NOT skip the response or pass it through; an empty known set is a maximally-strict policy, not a no-op.
- **Walker atomicity contract (rev-3 W8b / rev-4 W8b).** When the walker raises mid-walk (summarizer raise, non-`str` summarizer return, unknown manifest entry), the partial output dict that was being built MUST NOT be returned to the caller. The caller (compose-loop in Phase 3) will see only the raised `AuditIntegrityError` — no half-redacted dict reaches `chat_messages` or `tool_calls`. Test shape: a fixture response with three keys (one normal, one whose summarizer raises, one normal-but-after-the-raise) drives the walker; assert `AuditIntegrityError` propagates AND that no partially-built dict can be observed from the call site (the function does not return a value on the raise path; the test verifies this by ensuring the assignment target retains its pre-call value via `result = sentinel; with pytest.raises(...): result = redact_tool_call_response(...); assert result is sentinel`).

**Implementation note (rev-2 M.8):** The summarizer-error wiring must precede EVERY `raise AuditIntegrityError(...)` site reached via a summarizer failure (exception OR non-str return). Pattern:

```python
# summarizer raised or returned non-str
telemetry.summarizer_error(tool_name=tool_name)  # counter BEFORE raise
raise AuditIntegrityError(f"Summarizer for {tool_name!r} ...") from exc
```

The same pattern applies in Task 8 (`redact_tool_call_arguments` full implementation) which also has summarizer code paths.

Commit: `feat(composer/redaction): redact_tool_call_response with fixed-sentinel, crash discipline, summarizer_error counter (§4.2.4, §4.2.6, closes M2 M3 W5 W6, M.8)`.

---

## Task 8: `redact_tool_call_arguments` full implementation

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Create: `tests/unit/web/composer/test_redact_tool_call_arguments.py`

Generalises the tracer-bullet (Task 4) impl to handle:

- Type-driven entries: walks via `walk_model_schema(argument_model, with_values=True)`; substitutes `Sensitive`-marked nodes.
- Declarative entries: walks `arguments` by `policy.sensitive_argument_keys`; missing keys are no-ops; present keys are summarized or sentinel-substituted.
- Manifest entry missing → `AuditIntegrityError` (registry-consistency violation; the registry-equality adequacy assertion would normally catch this at CI, but the runtime walker must still crash if it is reached).
- Summarizer raises → `AuditIntegrityError` (same discipline as Task 7; call `telemetry.summarizer_error(tool_name=tool_name)` immediately before `raise` per rev-2 M.8).
- Summarizer returns non-`str` → `AuditIntegrityError` (same `telemetry.summarizer_error` wiring).
- Telemetry: `manifest_dispatch(tool_name=..., shape=...)` called once per invocation.

TDD test set covers every cell of the §4.2.6 disposition table. The Task 4 tracer-bullet tests stay green throughout (they exercise the type-driven branch on `set_source`). Test assertions for summarizer-error paths must verify `telemetry.summarizer_error_calls` was called (using `NoopRedactionTelemetry`).

Commit: `feat(composer/redaction): redact_tool_call_arguments full impl with summarizer_error counter (§4.2.6, rev-2 M.8)`.

---

## Task 9: Adequacy guard — assertion 1 (registry-manifest set equality)

**Files:**

- Create: `tests/unit/web/composer/test_adequacy_guard.py`
- Create: `tests/unit/web/composer/_adequacy_helpers.py` (collects registry names)

**Steps:**

- [ ] Implement `_adequacy_helpers.collect_registry_names()` per spec §4.4.1 — union the six dispatch dicts at `tools.py:5250–5314` plus the three inline-handled tool names (`preview_pipeline`, `diff_pipeline`, `set_pipeline` — already in `_DISCOVERY_TOOLS`/`_MUTATION_TOOLS` but special-cased in `execute_tool` at `tools.py:5411` onward) plus the advisor-escape-hatch name (`request_advisor_hint`, intercepted at `service.py:2070`).
- [ ] Test: `MANIFEST.keys() == collect_registry_names()` at the time of writing this task. Expected: red until Task 16 completes (the manifest currently has only `set_source` from Task 4). The redness is the contract — adding a tool to a dispatch dict without a manifest entry must fail CI.
- [ ] **DO NOT** add manifest entries to make the test green at this task. The manifest is populated by Tasks 13-16; the assertion's job is to fail until they do.

Commit: `test(composer/redaction): adequacy guard registry-manifest set equality (§4.4.1, closes B1)`.

---

## Task 10: Adequacy guard — assertion 2 (per-entry shape walk) + assertion 5 (`extra="forbid"`)

**Files:**

- Modify: `tests/unit/web/composer/test_adequacy_guard.py`

Implements the per-entry shape walk per spec §4.4.2:

- For each manifest entry: walk `argument_model` (and `response_model`) via the shared iterator; assert each `TraversalNode` either has a `_SensitiveMarker` or is a scalar/non-redaction-eligible field.
- For declarative entries: assert `policy` is internally consistent: `sensitive_argument_keys ⊆ known_response_keys ∪ argument_summarizers.keys()` and no orphan summarizers. **DO NOT perform AST inspection of handler source** (rev-2 M_adequacy_mechanical_enforcement M.3). AST inspection is implementation coupling — any handler using `args = arguments; args['x']`, destructuring, or helper delegation evades the scan. Tools requiring mechanical key-coverage guarantees MUST be promoted to type-driven Pydantic argument models with `extra="forbid"`.
- **Sixth assertion — `sensitive_response_keys ⊆ known_response_keys` (rev-5 quality W7-PARTIAL).** For every declarative entry, assert `set(policy.sensitive_response_keys) <= set(policy.known_response_keys)`. This closes the dual-typo silent-pass that the Task 16-final-a runtime smoke test alone cannot catch: when an entry declares `sensitive_response_keys=("contents",)` (misspelled) AND `known_response_keys` happens to contain the correctly-spelled `"content"` on the passthrough list, the smoke test passes (the typo'd key, when injected, is redacted as expected; the real `"content"` key is never injected by the smoke fixture) but production responses with the real key flow through as known-passthrough. This adequacy assertion at construction-time catches the mismatch without needing any runtime payload. Mechanically:

  ```python
  for tool_name, entry in MANIFEST.items():
      if entry.policy is None:
          continue
      policy = entry.policy
      if policy.handles_no_sensitive_data:
          continue
      orphans = set(policy.sensitive_response_keys) - set(policy.known_response_keys)
      assert not orphans, (
          f"Tool {tool_name!r}: sensitive_response_keys contains keys not in "
          f"known_response_keys: {sorted(orphans)!r}. Either the sensitive key "
          f"is misspelled (compare against the actual handler response shape) "
          f"or it must be added to known_response_keys. A 'sensitive' key that "
          f"is not in the known set is a typo silently producing the wrong "
          f"redaction path."
      )
  ```

  The same discipline does NOT apply to `sensitive_argument_keys` because there is no `known_argument_keys` analogue (argument validation is structurally upstream of redaction for type-driven entries, and for declarative entries the handler's argument-key set is implicit in the dispatch contract).
- **Fifth assertion — `extra="forbid"` on type-driven entries (rev-2 M.2):** For every type-driven manifest entry, assert `entry.argument_model.model_config.get("extra") == "forbid"`. This prevents the `arguments_canonical` / walker discrepancy described in spec §4.4.2.
- **Walker completeness floor-check (rev-3 M1):** Assert that `walk_model_schema` emits at least one node whose path root matches every top-level field in `model_fields`. This catches the class of bug the shared-iterator pattern does NOT protect against: both the guard and the walker share the same blind spot when the iterator silently drops a field due to an unhandled annotation form — so disagreement between guard and walker is impossible, but shared omissions are not. Add the following test adjacent to the five adequacy assertions:

  ```python
  def test_walker_emits_node_for_every_top_level_field() -> None:
      """Floor check: walk_model_schema must yield at least one node whose path
      root matches every top-level field in model_fields. Catches the failure
      mode where the iterator silently drops a field due to an unhandled
      annotation form — a class of bug the shared iterator pattern does NOT
      protect against (it forecloses guard/walker disagreement, not shared
      blind spots). Closes rev-3 M1.
      """
      from elspeth.web.composer.redaction import MANIFEST, walk_model_schema

      def _root(path: str) -> str:
          # Strip container-descent suffixes to get the top-level field name.
          return path.split(".")[0].split("[")[0].split("{")[0]

      for tool_name, entry in MANIFEST.items():
          if entry.argument_model is None:
              continue
          expected = set(entry.argument_model.model_fields.keys())
          actual = {_root(n.path) for n in walk_model_schema(entry.argument_model)}
          assert expected <= actual, (
              f"Iterator dropped fields for {tool_name}: missing={expected - actual!r}; "
              f"this means walk_model_schema does not understand a field annotation "
              f"on {entry.argument_model.__name__}. Adequacy guard cannot detect this "
              f"because the runtime walker shares the same blind spot."
          )
  ```

Commit: `test(composer/redaction): adequacy guard per-entry shape walk + extra=forbid assertion + walker completeness floor-check (§4.4.2, closes B2, M.2, M.3, rev-3 M1)`.

---

## Task 11: Adequacy guard — assertion 3 (mass-copy uniqueness)

**Files:**

- Modify: `tests/unit/web/composer/test_adequacy_guard.py`

Implements the mass-copy uniqueness check per spec §4.4.4. Tests:

- Two manifest entries sharing exact-match `why_arguments_safe` text fail the assertion.
- Two entries with `why_arguments_safe` differing only by trailing whitespace also fail (the assertion normalises whitespace before comparing).
- Identical-text `sensitive_data_locations` is allowed (some tools genuinely share a location, e.g. "server-side secret resolver").

Commit: `test(composer/redaction): adequacy guard mass-copy uniqueness (§4.4.4, closes W7)`.

---

## Task 12: Adequacy guard — assertion 4 (policy-hash snapshot)

**Files:**

- Modify: `tests/unit/web/composer/test_adequacy_guard.py`
- Create: `tests/unit/web/composer/redaction_policy_snapshot.json`
- Create: `scripts/cicd/bootstrap_redaction_snapshot.py` (**NOT** `scripts/composer/` — that directory does not exist; rev-2 m_script_dir_missing fix; `scripts/cicd/` is the existing CI helper location alongside `enforce_tier_model.py`)

**Steps:**

- [ ] Implement `_entry_hash(name, entry)` per spec §4.4.3 with the broadened coverage (every entry, not only declarative). Include `sensitive_path_count` as a field in each snapshot entry (for the direction-aware label gate in Task 18 to use without needing a live Python run).
- [ ] Implement `scripts/cicd/bootstrap_redaction_snapshot.py`: reads `MANIFEST`, computes the snapshot (including `sensitive_path_count`), writes the JSON file. Idempotent. No `mkdir -p scripts/composer/` step required.
- [ ] The snapshot file at this task is *empty / not yet committed*. The test asserts `{name: _entry_hash(name, e) for name, e in MANIFEST.items()} == json.load(open(snapshot_path))`. Bootstrap is run at the end of Task 16 once all manifest entries exist; a pre-merge run regenerates the snapshot to capture the final state.
- [ ] Test: confirm a removed `Sensitive()` annotation flips the hash for a type-driven entry.
- [ ] Test: confirm a renamed key in `sensitive_argument_keys` flips the hash for a declarative entry.
- [ ] **Hash-semantics tests (rev-2 BLOCKER_C):**
  - Test: replacing a summarizer with a new function object flips the hash.
  - Test (documents known false-negative): closing over a module-level variable, computing hash, mutating the variable WITHOUT replacing the summarizer, recomputing hash — assert the hash is UNCHANGED. Test docstring MUST say: "This is a known false-negative class; see spec §4.2.3 / §4.4.3. In-place mutation of closure-captured state does not flip the hash."

Commit: `test(composer/redaction): adequacy guard policy-hash snapshot + hash-semantics tests (§4.4.3, closes M9 W12, BLOCKER_C)`.

---

## Task 13 (Sensitive[T] Wave 2): promote `create_blob`, `update_blob`, `set_source_from_blob`

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py` (add three argument models + manifest entries)
- Modify: `src/elspeth/web/composer/tools.py` (promote three handlers)
- Create: `tests/unit/web/composer/test_promote_create_blob.py`
- Create: `tests/unit/web/composer/test_promote_update_blob.py`
- Create: `tests/unit/web/composer/test_promote_set_source_from_blob.py`

Per-tool sub-task shape (repeated three times):

- [ ] Read the handler. Document every `arguments["key"]` and `arguments.get("key")` site.
- [ ] Define the Pydantic argument model, mirroring the existing required-paths schema. Set `model_config = ConfigDict(extra="forbid")` (rev-2 M.1). Annotate sensitive fields with `Sensitive[T]`. For `create_blob` and `update_blob`, the `content` field is `Annotated[str, Sensitive(summarizer=lambda b: f"<inline-blob:{len(b)}-bytes>")]`.
- [ ] Add the manifest entry: `MANIFEST["create_blob"] = ToolRedaction(argument_model=CreateBlobArgumentsModel)` etc.
- [ ] Promote the handler: validate with `try: validated = Model.model_validate(arguments) except pydantic.ValidationError as exc: raise ToolArgumentError(...) from exc` (pattern from Task 4 and `tools.py:2668–2801`). Replace `arguments["k"]` with `validated.k`.
- [ ] **REMOVE** `_TOOL_REQUIRED_PATHS[tool]` **unconditionally** in the same commit that promotes the handler. (No `_TOOL_OPTIONAL_PATHS` exists in the codebase — rev-5 reality check correction.) Rationale identical to Task 4 (Pydantic model + `extra="forbid"` is strictly more capable than dotted-path checks; CLAUDE.md "No Legacy Code Policy" forbids the two-checks-at-different-layers pattern). Rev-3 N7 / rev-4 M1.
- [ ] **MissingRequiredPaths test-pin update for this tool (rev-4 M2 — same commit).** Run `grep -Rn '"MissingRequiredPaths"\|MissingRequiredPaths' tests/ src/elspeth/web/ --include='*.py'`. For every match whose test scenario invokes THIS tool's handler with invalid arguments, update the assertion from `"MissingRequiredPaths"` to the new ARG_ERROR / `ValidationError`-class error_class string. Leave matches for tools not promoted yet in this Wave. Discipline detail (read once, in Task 4): see Task 4 Step 4 "MissingRequiredPaths test-pin update" subsection — that section is the canonical procedure; this bullet is a per-tool reminder, not a duplicate procedure.
- [ ] TDD per-tool test: assert `ToolArgumentError` raised on invalid inputs (with `pydantic.ValidationError` as `__cause__`); redaction substitutes Sensitive fields correctly. (Pattern from Task 4 — do not duplicate the rationale here; cite Task 4.)
- [ ] **Blob-tool summarizer type-variability step (rev-2 M_blob_summarizer_type_variability M.10):** Verify the summarizer for each blob tool's sensitive field handles every value type the Pydantic model admits. For `create_blob.content` (typed `str`): test the summarizer against `None` (verify `extra="forbid"` rejects it at model_validate; do NOT test the summarizer directly on `None`), empty string, ASCII string, non-ASCII string. The lambda `lambda b: f"<inline-blob:{len(b)}-bytes>"` is only safe for `str` — since `extra="forbid"` and Pydantic validates `content: str`, `None` or other types will be rejected before the summarizer runs; document this reasoning in the test.
- [ ] Existing integration tests for the tool pass.

Commit per tool (three commits): `feat(composer/redaction): promote {tool} to type-driven manifest entry`. Reviewer note in body: "Sensitive[T] promotion wave 2/4; extra=forbid; ToolArgumentError wrap pattern per Task 4."

---

## Task 14 (Sensitive[T] Wave 3): promote `set_pipeline`, `apply_pipeline_recipe`

Same shape as Task 13 (including `model_config = ConfigDict(extra="forbid")`, the `try/except ValidationError → ToolArgumentError` wrap pattern, `ToolArgumentError`-not-`ValidationError` test assertions, the **unconditional** `_TOOL_REQUIRED_PATHS[tool]` removal per Task 4 M1, and the per-tool `MissingRequiredPaths` test-pin update per Task 4 M2) but for two tools that have more complex argument schemas. `set_pipeline` is special-cased in `execute_tool()` at `tools.py:5436` (it can own `source.inline_blob`); the promotion must preserve that branch.

**Special caution for `set_pipeline`:** it has an extended dispatch signature (with `session_engine` / `session_id` kwargs). The Pydantic model only validates the LLM-supplied `arguments` dict; the kwargs are wired by the dispatcher at `execute_tool()` and are not user-facing. Document this in the model's docstring.

**Confirmed `MissingRequiredPaths` pin to update with the `set_pipeline` promotion (rev-4 M2).** `tests/unit/web/composer/test_service.py:367` asserts `"MissingRequiredPaths" in result.message` for a failed `set_pipeline` call where the pipeline source plugin is missing. This pin is Task 14's responsibility because it exercises `set_pipeline`. When the `set_pipeline` promotion lands:

1. The `_TOOL_REQUIRED_PATHS["set_pipeline"]` entry is removed.
2. The invalid-pipeline scenario now routes through `pydantic.ValidationError` → `ToolArgumentError` → ARG_ERROR with `error_class=type(exc).__name__` (i.e. `"ValidationError"`).
3. Update `tests/unit/web/composer/test_service.py:367` so the assertion reads:
   ```python
   assert "ValidationError" in result.message
   assert "source.plugin" in result.message  # Pydantic error path still mentions the missing field
   ```
   Run the test against a real `pydantic.ValidationError` for the empty-pipeline scenario first; copy the exact `error_class` and confirm `source.plugin` appears in the Pydantic message before committing. Memory: `feedback_locked_in_buggy_expectations`.
4. Same procedure for `apply_pipeline_recipe` if it has any pinned-string tests.

Two commits, one per tool. Each commit body MUST cite which `MissingRequiredPaths` matches were updated and confirm the grep is now clean for the tool being promoted.

---

## Task 15 (Sensitive[T] Wave 4): inspect `patch_*_options`

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py` (potentially)
- Modify: `src/elspeth/web/composer/tools.py` (potentially)

**Investigation step.** Read `_handle_patch_source_options` (`tools.py`), `_handle_patch_node_options`, `_handle_patch_output_options`. Determine whether their `options` dicts can carry secret-shaped values (e.g. credentials, paths). Three outcomes:

1. The option dict is plugin-defined and unconstrained at composition time → promote to a Pydantic argument model where `options` is `Annotated[dict[str, Any], Sensitive(summarizer=...)]`. Use `model_config = ConfigDict(extra="forbid")` and the `try/except ValidationError → ToolArgumentError` wrap pattern from Task 4.
2. The option dict is purely structural (e.g. `{"name": "x"}`) → declarative manifest entry suffices; defer to Task 16.
3. Mixed — some keys are sensitive, some aren't → declarative entry with explicit `sensitive_argument_keys` is the cleanest representation.

Surface the finding (with citations to the handler source) before promoting; if outcome 2 or 3, this task is a no-op (entry created in Task 16). If outcome 1, this task promotes one to three tools.

---

## Task 16: Declarative manifest entries for remaining tools

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py` (populate `MANIFEST` for the remaining ~29-31 tools)
- Run: `scripts/cicd/bootstrap_redaction_snapshot.py` to regenerate snapshot

**Subdivision per plan-review W10.** This task is split into seven sub-tasks, one per registry, plus an inline-tools sub-task. Each sub-task is a separate commit so reviewers can read each registry's entries in isolation.

The full enumeration (37 registry tools plus `request_advisor_hint` = 38 MANIFEST entries total; minus those promoted in Tasks 4, 13, 14, 15):

- [ ] **16a — `_DISCOVERY_TOOLS` (12 entries).**
  - `list_sources`, `list_transforms`, `list_sinks`, `get_plugin_schema`, `get_expression_grammar`, `explain_validation_error`, `get_plugin_assistance`, `list_models`, `list_recipes`, `get_pipeline_state`, `preview_pipeline`, `diff_pipeline`.
  - All discovery tools are read-only, return cached metadata, and emit no LLM-user data. Expected shape: `policy=ToolRedactionPolicy(handles_no_sensitive_data=True, handles_no_sensitive_data_reason_struct=...)` for every entry. Each `HandlesNoSensitiveDataReason` must be **distinct** (mass-copy uniqueness asserted by Task 11) — write 12 distinct reasons.
  - Commit: `feat(composer/redaction): declarative manifest entries for _DISCOVERY_TOOLS (12 entries)`.

- [ ] **16b — `_MUTATION_TOOLS` remaining (after `set_source`, `set_pipeline`, possibly `patch_*_options` promoted): ~7-10 entries.**
  - `upsert_node`, `upsert_edge`, `remove_node`, `remove_edge`, `set_metadata`, `set_output`, `remove_output`, `clear_source`, plus any `patch_*_options` not promoted in Task 15.
  - Mostly structural arguments (graph IDs, node names, plugin keys). Most expected shape: `handles_no_sensitive_data=True`.
  - Commit.

- [ ] **16c — `_BLOB_DISCOVERY_TOOLS` (4 entries).**
  - `list_blobs`, `get_blob_metadata`, `get_blob_content`, `inspect_source`.
  - `get_blob_content` is special: it returns blob content. The response shape includes the actual content, which may itself be sensitive. Decision: declarative entry with `sensitive_response_keys=("content",)` and a summarizer that replaces with `<blob-content:{len}-bytes>`. **CAUTION:** confirm with a unit test that the response shape always includes a `content` key and no other content-bearing keys.
  - Commit.

- [ ] **16d — `_BLOB_MUTATION_TOOLS` remaining (after `create_blob`, `update_blob`, `set_source_from_blob` promoted, possibly `apply_pipeline_recipe` promoted): up to 1 entry.**
  - `delete_blob` is the only one that should remain: declarative, `handles_no_sensitive_data=True` (it takes a blob ID, no sensitive payload).
  - Commit.

- [ ] **16e — `_SECRET_DISCOVERY_TOOLS` (2 entries).**
  - `list_secret_refs`, `validate_secret_ref`.
  - These return **secret names** (Tier-3 inventory metadata) but never secret **values**. Declarative: `handles_no_sensitive_data=True`. The `HandlesNoSensitiveDataReason` for each must explicitly cite "secret values are resolved server-side and never traverse this surface" with the exact path to the resolver.
  - Commit.

- [ ] **16f — `_SECRET_MUTATION_TOOLS` (1 entry).**
  - `wire_secret_ref`. Same discipline as 16e. The reason text mirrors spec §4.7's `wire_secret_ref` example.
  - Commit.

- [ ] **16g — Inline tools (`request_advisor_hint`, the advisor escape-hatch).**
  - The advisor tool calls a frontier LLM with a hint string supplied by the LLM. The hint string is potentially sensitive (the LLM may include task context that leaked from the prompt). Decision: declarative entry with `sensitive_argument_keys=("hint",)` and a summarizer `lambda h: f"<advisor-hint:{len(h)}-chars>"`. Response is the advisor's reply, which is general-purpose guidance — declarative `known_response_keys=("guidance",)` (or similar; confirm against the actual response shape at `service.py:2070-onward`).
  - Commit.

- [ ] **16-final-a: per-entry runtime smoke test (rev-3 W7 / rev-4 W7).** The adequacy guard verifies structural consistency of declarative entries; it does NOT verify that a runtime call to `redact_tool_call_arguments(tool_name, args, ...)` for each declarative entry actually executes the redaction path without raising. A typo in `sensitive_response_keys=("contents",)` for `get_blob_content` (with the actual response key spelled `"content"`) passes every adequacy assertion but produces a runtime no-op redaction at the response surface. Create `tests/unit/web/composer/test_declarative_manifest_runtime_smoke.py`:

  ```python
  """Per-entry runtime smoke test for declarative manifest entries.

  Closes rev-3 W7 / rev-4 W7. Each declarative entry is exercised through
  redact_tool_call_arguments AND redact_tool_call_response with a minimal
  representative payload, asserting:
    1. The redaction call returns without raising.
    2. For entries with sensitive_argument_keys: the named keys are replaced
       with the summarizer output (or fixed sentinel) when present in the
       payload; absent keys are no-ops (not added).
    3. For entries with sensitive_response_keys: the named keys are replaced
       in the response when present; unknown keys are replaced with the
       fixed sentinel <redacted-unknown-response-key>.
    4. For handles_no_sensitive_data=True entries: redaction is a passthrough
       (output == input as JSON; no keys added or removed).

  This is the second line of defense behind the structural adequacy guard
  (Tasks 9-12). A misspelled key in a declarative entry passes structural
  checks but fails this runtime smoke immediately.
  """
  from __future__ import annotations

  import pytest

  from elspeth.web.composer.redaction import (
      MANIFEST,
      redact_tool_call_arguments,
      redact_tool_call_response,
  )
  from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


  _DECLARATIVE_ENTRIES = [
      (name, entry) for name, entry in MANIFEST.items()
      if entry.policy is not None
  ]


  @pytest.mark.parametrize(("tool_name", "entry"), _DECLARATIVE_ENTRIES)
  def test_declarative_entry_argument_redaction_runtime(
      tool_name: str, entry: object
  ) -> None:
      """Each declarative entry's argument redaction path executes without raising
      and replaces every declared sensitive key when present in the payload."""
      policy = entry.policy
      telemetry = NoopRedactionTelemetry()

      if policy.handles_no_sensitive_data:
          # Passthrough invariant: arguments come out as-equal-to-input for any
          # representative shape (we use {} since handles_no_sensitive_data
          # entries should not need to inspect the payload).
          out = redact_tool_call_arguments(tool_name, {}, telemetry=telemetry)
          assert out == {}, (
              f"Tool {tool_name!r} declared handles_no_sensitive_data=True but "
              f"redact_tool_call_arguments mutated {{}} → {out!r}. Passthrough "
              f"contract violated."
          )
          return

      # For entries with sensitive_argument_keys, build a representative payload
      # containing every declared key with a non-empty string sentinel value,
      # then assert each key's value is replaced.
      raw_value = "RUNTIME-SMOKE-SENTINEL"
      payload = {key: raw_value for key in policy.sensitive_argument_keys}
      out = redact_tool_call_arguments(tool_name, payload, telemetry=telemetry)
      for key in policy.sensitive_argument_keys:
          assert key in out, (
              f"Tool {tool_name!r}: sensitive_argument_key {key!r} disappeared "
              f"from the redacted output. Redaction must preserve key structure."
          )
          assert out[key] != raw_value, (
              f"Tool {tool_name!r}: sensitive_argument_key {key!r} was NOT "
              f"replaced. raw={raw_value!r}, redacted={out[key]!r}. Common cause: "
              f"key name in MANIFEST does not match the actual argument key the "
              f"handler reads (typo in sensitive_argument_keys tuple)."
          )


  @pytest.mark.parametrize(("tool_name", "entry"), _DECLARATIVE_ENTRIES)
  def test_declarative_entry_response_redaction_runtime(
      tool_name: str, entry: object
  ) -> None:
      """Each declarative entry's response redaction path replaces every declared
      sensitive_response_key when present, and substitutes the fixed sentinel
      for any unknown key not in known_response_keys."""
      policy = entry.policy
      telemetry = NoopRedactionTelemetry()

      if policy.handles_no_sensitive_data:
          # No response-side declaration to exercise.
          return

      raw_value = "RUNTIME-SMOKE-SENTINEL"
      payload = {key: raw_value for key in policy.sensitive_response_keys}
      out = redact_tool_call_response(tool_name, payload, telemetry=telemetry)
      for key in policy.sensitive_response_keys:
          assert key in out
          assert out[key] != raw_value, (
              f"Tool {tool_name!r}: sensitive_response_key {key!r} was NOT "
              f"replaced. Common cause: key name in MANIFEST does not match the "
              f"actual response key the handler emits (typo in "
              f"sensitive_response_keys tuple)."
          )

      # Unknown-key fail-closed: a key not in known_response_keys triggers the
      # fixed sentinel. Pick a key name unlikely to collide.
      unknown_payload = {"__runtime_smoke_unknown_key__": raw_value}
      unknown_out = redact_tool_call_response(
          tool_name, unknown_payload, telemetry=telemetry
      )
      assert unknown_out["__runtime_smoke_unknown_key__"] == "<redacted-unknown-response-key>", (
          f"Tool {tool_name!r}: unknown-key fail-closed sentinel did not fire. "
          f"redact_tool_call_response must replace any key not in "
          f"known_response_keys with the fixed sentinel."
      )
  ```

  Commit: `test(composer/redaction): per-entry runtime smoke for declarative manifest (closes rev-3 W7 / rev-4 W7)`.

- [ ] **16-final-b: regenerate snapshot.**
  - Run `scripts/cicd/bootstrap_redaction_snapshot.py`.
  - Verify `tests/unit/web/composer/test_adequacy_guard.py` is now fully green (all four assertions pass).
  - Verify `tests/unit/web/composer/test_declarative_manifest_runtime_smoke.py` from 16-final-a is fully green.
  - Commit the snapshot file: `chore(composer/redaction): bootstrap redaction policy hash snapshot`.

After Task 16 the manifest covers every tool name in the dispatch registries; the four adequacy assertions all pass and the per-entry runtime smoke is green.

---

## Task 17: Compose-loop unknown-tool-name routing test

**Files:**

- Create: `tests/unit/web/composer/test_compose_loop_unknown_tool_name.py`

**Why this task.** Plan-review M7 / W3. The previous plan introduced a `MissingToolError` crash for LLM-hallucinated tool names. Spec rev-5 §4.2.6 / §5.7.5 makes this an explicit boundary discipline: an LLM-supplied unknown tool name is Tier-3 input and must route through `_failure_result` + continue, not crash. The dispatcher's fall-through at `tools.py:5481` already handles this. The compose loop records the failure and continues. The JSON-decode / non-dict pre-dispatch gate at `service.py:1838–1881` is a separate, earlier site for malformed arguments — NOT the site for unknown tool names.

**Steps:**

- [ ] **Read the existing fall-through.** `tools.py:5481` is the dispatcher's final `return _failure_result(state, f"Unknown tool: {tool_name}")` — the dispatcher returns a failure `ToolResult`, it does not raise. Read the compose-loop's handling of this `ToolResult` (after `service.py:2480`'s `ToolArgumentError` catch block) to find where the failure `ToolResult` flows. Document the actual error_class and audit-status the recorder receives in this path.
- [ ] **Construct the test scenario.** A compose-loop scenario where the LLM emits a tool call with `function.name = "this_tool_does_not_exist"`. Drive it through the actual `execute_tool` + recorder path (no production-bypass shortcut per spec §8.6 / CLAUDE.md test-path-integrity).
- [ ] **Assert the observed behaviour with escalation rule (rev-2 m_task17_pinning_risk).** The dispatcher returns a failure `ToolResult` (no exception propagates). Verify the actual audit status produced for an unknown tool name. **If the actual status is `ARG_ERROR` (or the compose-loop constant for this condition), pin it.** If the actual status differs from `ARG_ERROR`, **report to operator — do not pin a non-`ARG_ERROR` status without escalation.** The LLM receives a `role=tool` message with the failure payload and the loop continues.
- [ ] **Do not introduce a `MissingToolError` exception class.** The dispatcher's existing failure-`ToolResult` path is the correct shape; the test pins it. The previous plan's `MissingToolError` design treated Tier-3 input as a Tier-1 crash condition (closed by plan-review M7 / W3 and spec §4.2.6 / §5.7.5).
- [ ] **Phase 3 call-order precondition pin (rev-3 M2).** Add the following test to `tests/unit/web/composer/test_compose_loop_unknown_tool_name.py`. This pin documents Phase 3's required call ordering mechanically: `redact_tool_call_arguments` must NOT be called before MANIFEST membership is confirmed. CLAUDE.md principle: "if a constraint isn't mechanically enforced, assume the next session won't know about it." An LLM-hallucinated tool name that bypasses the MANIFEST check and reaches the redaction layer would convert a graceful Tier-3 quarantine into a Tier-1 `AuditIntegrityError` crash.

  ```python
  def test_redact_tool_call_arguments_raises_for_unknown_tool() -> None:
      """Phase 3 contract pin: redact_tool_call_arguments must NOT be called for
      a tool name that is not in MANIFEST. The compose loop's existing
      unknown-tool check (tools.py:5481 -> _failure_result with
      'Unknown tool: {name}') MUST fire BEFORE the redaction layer.

      If Phase 3 inverts this ordering (redact-then-check), an LLM-hallucinated
      tool name will be silently converted from a graceful Tier-3 quarantine
      into a Tier-1 AuditIntegrityError crash. This test asserts that
      redact_tool_call_arguments fails loudly when called out of order, so
      Phase 3's call site is mechanically constrained. Closes rev-3 M2.
      """
      import pytest
      from elspeth.contracts.errors import AuditIntegrityError
      from elspeth.web.composer.redaction import redact_tool_call_arguments
      from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry

      with pytest.raises(AuditIntegrityError) as excinfo:
          redact_tool_call_arguments(
              tool_name="nonexistent_tool_name_for_call_order_pin",
              arguments={},
              telemetry=NoopRedactionTelemetry(),
          )
      # Error message must cite the missing tool name so Phase 3 implementers
      # see the contract violation in stack traces.
      assert "nonexistent_tool_name_for_call_order_pin" in str(excinfo.value)
  ```

Commit: `test(composer): pin compose-loop unknown-tool-name routing via tools.py:5481 + Phase 3 call-order precondition (closes M7 W3, rev-3 M2)`.

---

## Task 18: Label-gate CI workflow

**Files:**

- Create: `.github/workflows/composer-redaction-gate.yml`

**Workflow shape (rev-2 BLOCKER_B direction-aware rewrite):**

```yaml
name: composer-redaction-gate
on:
  pull_request:
    paths:
      - "src/elspeth/web/composer/redaction.py"
      - "src/elspeth/web/composer/redaction_telemetry.py"
      - "src/elspeth/web/composer/tools.py"
      - "tests/unit/web/composer/redaction_policy_snapshot.json"
      - "tests/unit/web/composer/test_adequacy_guard.py"
      - "tests/unit/web/composer/test_walk_model_schema.py"

jobs:
  redaction-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Compute snapshot diff vs main
        id: diff
        run: |
          git fetch origin main
          if git diff --quiet origin/main HEAD -- tests/unit/web/composer/redaction_policy_snapshot.json; then
            echo "snapshot_changed=false" >> "$GITHUB_OUTPUT"
            echo "direction=none" >> "$GITHUB_OUTPUT"
          else
            echo "snapshot_changed=true" >> "$GITHUB_OUTPUT"
            # Direction-aware check: sum sensitive_path_count across changed entries
            # The snapshot includes a sensitive_path_count field per entry (Task 12).
            # Compare PR head vs main to detect weakening (count decreased).
            python3 - <<'PYEOF'
import json, os, subprocess, sys

base_json = subprocess.check_output(["git", "show", "origin/main:tests/unit/web/composer/redaction_policy_snapshot.json"])
head_json = open("tests/unit/web/composer/redaction_policy_snapshot.json").read()

base = json.loads(base_json)
head = json.loads(head_json)

changed_entries = {k for k in set(base) | set(head) if base.get(k) != head.get(k)}
base_total = sum(base.get(k, {}).get("sensitive_path_count", 0) for k in changed_entries)
head_total = sum(head.get(k, {}).get("sensitive_path_count", 0) for k in changed_entries)

if head_total < base_total:
    direction = "weaken"
else:
    direction = "strengthen"

print(f"Changed entries: {sorted(changed_entries)}")
print(f"Base sensitive_path_count (changed entries): {base_total}")
print(f"Head sensitive_path_count (changed entries): {head_total}")
print(f"Direction: {direction}")

with open(f"{os.environ['GITHUB_OUTPUT']}", 'a') as f:
    f.write(f"direction={direction}\n")
PYEOF
          fi
      - name: Assert correct direction label on snapshot change
        if: steps.diff.outputs.snapshot_changed == 'true'
        env:
          PR_LABELS: ${{ toJson(github.event.pull_request.labels.*.name) }}
          PR_BODY: ${{ github.event.pull_request.body }}
          DIRECTION: ${{ steps.diff.outputs.direction }}
        run: |
          if [ "$DIRECTION" = "weaken" ]; then
            # Weakening: ONLY policy-weaken-justified is valid
            echo "$PR_LABELS" | jq -e 'any(. == "policy-weaken-justified")' \
              || (echo "::error::snapshot diff shows coverage reduction (weakening); PR must carry policy-weaken-justified label, not policy-strengthen"; exit 1)
            echo "$PR_BODY" | grep -F "Redaction policy weakening rationale" \
              || (echo "::error::policy-weaken-justified label requires a 'Redaction policy weakening rationale' section in the PR body"; exit 1)
            echo "$PR_LABELS" | jq -e 'any(. == "policy-strengthen")' \
              && (echo "::error::snapshot diff shows weakening; policy-strengthen is incorrect for this change"; exit 1) || true
          else
            # Strengthening or neutral: ONLY policy-strengthen is valid
            echo "$PR_LABELS" | jq -e 'any(. == "policy-strengthen")' \
              || (echo "::error::redaction snapshot changed; PR must carry policy-strengthen label"; exit 1)
            echo "$PR_LABELS" | jq -e 'any(. == "policy-weaken-justified")' \
              && (echo "::error::snapshot diff shows no coverage reduction; do not use policy-weaken-justified for this change"; exit 1) || true
          fi
```

**Caveats and tests:**

- [ ] The workflow runs only when one of the redaction-relevant paths changes; this is a soft scoping (a PR that changes the manifest implicitly changes the snapshot, so the workflow runs).
- [ ] No CODEOWNERS file is created.
- [ ] Add a `tests/unit/web/composer/test_label_gate_yaml.py` that parses the YAML and asserts the workflow is direction-aware (checks for `direction` output and separate weaken/strengthen branches).
- [ ] Add `tests/unit/web/composer/test_label_gate_direction.py` covering all four combinations (rev-2 BLOCKER_B):
  - (a) weakening diff + `policy-weaken-justified` → pass
  - (b) weakening diff + `policy-strengthen` → fail with direction-mismatch message
  - (c) strengthening diff + `policy-strengthen` → pass
  - (d) strengthening diff + `policy-weaken-justified` → fail with "no coverage reduction" message
- [ ] Document the workflow in `docs/guides/redaction-policy-changes.md` (NEW file): when to use which label, what the rationale section must contain, escalation path, single-owner governance note.

Commit: `ci(composer/redaction): label-gate workflow (§4.4.5, closes B4 W9 M10)`.

---

## Task 19: Final gate run

**Steps:**

- [ ] Run the full project gate from preflight, in the same order:

  ```bash
  .venv/bin/python -m pytest tests/unit -q
  .venv/bin/python -m pytest tests/integration -q -m "not testcontainer"
  .venv/bin/python -m mypy src/
  .venv/bin/python -m ruff check src/
  .venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
  .venv/bin/python scripts/cicd/enforce_freeze_guards.py
  ```

- [ ] Adequacy guard sanity bound (closes M4): time the guard's runtime; assert it completes in under 5 seconds for the current 38-entry MANIFEST (37 registry tools plus `request_advisor_hint`). The bound is order-of-magnitude, not a tight budget — flake-source guidance per spec §1.4.

- [ ] **Hypothesis completeness property test (rev-2 BLOCKER_A MAJOR-4; rev-4 B1 rewrite).** Create `tests/unit/web/composer/test_redaction_completeness_property.py`.

  **History of this test — read before implementing.** This test has been rewritten twice in response to convergent reviewer findings about silent-pass modes. Each previous shape (rev-2 string-in-serialized-output, rev-3 value_provider-with-isinstance-filter) admitted a different class of silent skip that converged three reviewers at the same line range. The current shape (rev-4) is **type-agnostic and has no silent-skip surface** — the only legitimate skip is `raw_value is None`, and every other state that would have been silently skipped in earlier versions is now an explicit `assert` failure with a precise message.

  **Invariant (rev-4, type-agnostic).** For every Sensitive-annotated path reachable through `walk_model_schema(model, with_values=True)`, the value extracted at that path from the *redacted* output must NOT equal the value extracted from the *raw* input. The comparison is `!=` at the path, regardless of `T`. This sidesteps the substring problem entirely (no `"a" in "abc"` false-positives) and admits no `isinstance(...)` filter that could quietly exempt a `T` the manifest later starts using.

  **Why this invariant and not "raw value not in serialized output".** The prior invariant required string-shaped values; any Sensitive[T] where T is `dict`, `bytes`, `int`, `bool`, `None`, or even empty-string was silently exempted by an `isinstance(raw_value, str) and raw_value` filter at the comparison site. The Task 15 spec explicitly plans `Annotated[dict[str, Any], Sensitive(summarizer=...)]` for `patch_*_options`, and Task 8 plans the same for blob `content` summarizers — those are the most common Sensitive shapes, and they were the ones the filter silently exempted. The path-aware `!=` invariant has no such filter.

  **What the test must NOT do (forbidden patterns).** Three patterns turned the previous shapes into silent-pass tests; do not reintroduce them under any circumstance:

  1. `if isinstance(raw_value, str) and raw_value:` (or any `isinstance` gate before `assert`). This silently exempts every non-string `T`. Replace with type-agnostic `!=` on the extracted value.
  2. `if node.value_provider is None: continue` for a Sensitive-marked node. If `with_values=True` and the node carries `_SensitiveMarker` but `value_provider is None`, that is a **walker bug** and the test MUST fail loudly, not skip. The only `value_provider is None` state allowed is when `with_values=False`, which is not the mode this test uses.
  3. `except (TypeError, ValueError): raw_values = [extracted]` (defensive fallback that degrades container-descent failures to scalar treatment). If `value_provider(args)` for a `[*]`/`{*}` path does not yield an iterable of `(key, value)` pairs, that is a walker contract violation; let the exception propagate so the test fails loudly.

  **Skip rules (mechanically narrow).** The only allowed silent skip is `raw_value is None` at a scalar path. Reason: a `None` raw means there is no sensitive data at this path for this Hypothesis example — there is nothing to redact, and asserting `redacted != None` would either tautologically fail (if redaction passes None through, which is correct) or pass spuriously (if redaction over-aggressively substitutes None with a sentinel, which is a different bug not under test here). Empty strings, empty containers (handled via key-set check), and `False` are NOT skipped — those values still must be replaced if the field is declared Sensitive, and the test catches a redactor that lazily passes them through.

  ```python
  """Hypothesis property test (rev-4): redaction replaces every Sensitive[T] field
  value at every reachable path, regardless of T's type.

  Closes rev-2 BLOCKER_A (quality MAJOR-4), rev-3 B1 (ad-hoc path.split skip),
  and rev-4 B1 (isinstance(str)/None-skip/except-fallback silent-pass family).

  Invariant: for each Sensitive-annotated path in the model, the value extracted
  at that path from the redacted output MUST NOT equal the value extracted from
  the raw input. The comparison is type-agnostic `!=` at the path; there is no
  `isinstance` filter and no defensive `except` fallback. If `value_provider` is
  None for a Sensitive-marked node when `with_values=True`, that is a walker
  bug and the test FAILS — it does not silently skip.

  Rev-history banner (do not delete — institutional memory):
    rev-2 design: "raw_value not in json.dumps(redacted)" — failed because
      ad-hoc path.split(".") loop couldn't navigate container indices, silently
      skipping every Sensitive[T] inside a list or dict.
    rev-3 design: value_provider-based extraction + "raw_value not in serialized"
      with isinstance(str) gate — failed because the gate silently exempted
      every non-str T (dict, bytes, int, bool, None, empty-str) including the
      Task 15 planned Annotated[dict[str, Any], Sensitive(...)] shape.
    rev-4 design (this file): value_provider-based extraction + path-aware `!=`
      between raw and redacted views — type-agnostic, no silent-skip surface.

  settings(max_examples=50, deadline=None) for stable CI execution.
  """
  from __future__ import annotations

  import pytest
  from hypothesis import event, given, settings
  from hypothesis import strategies as st

  from elspeth.web.composer.redaction import (
      MANIFEST,
      _SensitiveMarker,
      redact_tool_call_arguments,
      walk_model_schema,
  )
  from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


  _CONTAINER_PATH_MARKERS = ("[*]", "{*}")


  def _is_container_descent_path(path: str) -> bool:
      """A node's path is container-descent iff it carries a [*] or {*} segment.

      walk_model_schema documents these markers as the canonical container
      indicators; relying on the path shape (not on get_origin runtime checks)
      keeps the test decoupled from Pydantic introspection internals.
      """
      return any(marker in path for marker in _CONTAINER_PATH_MARKERS)


  @pytest.mark.parametrize(
      "tool_name",
      [name for name, entry in MANIFEST.items() if entry.argument_model is not None],
  )
  def test_redaction_replaces_every_sensitive_value(tool_name: str) -> None:
      """For each type-driven manifest entry: every Sensitive path's extracted
      value differs between raw and redacted views.

      This is the load-bearing security claim. A buggy redactor that passes
      through a Sensitive value of any type (str, dict, bytes, int, bool,
      empty-str, ...) at any path (scalar, list-item, dict-value, nested)
      causes this test to fail with a precise path + value diagnostic.
      """
      entry = MANIFEST[tool_name]
      model = entry.argument_model
      assert model is not None  # parametrize filter guarantees this; assert for mypy

      # Discover Sensitive paths once per tool (walker is deterministic).
      # value_provider MUST be non-None for every node when with_values=True;
      # we assert that mechanically so any walker regression that produces a
      # None value_provider for a Sensitive node fails this test loudly rather
      # than skipping silently. This is the rev-4 hardening against the
      # "value_provider is None: continue" silent-pass mode from rev-3.
      sensitive_nodes = []
      for node in walk_model_schema(model, with_values=True):
          if not any(isinstance(m, _SensitiveMarker) for m in node.metadata):
              continue
          assert node.value_provider is not None, (
              f"walk_model_schema(model={model.__name__!r}, with_values=True) "
              f"yielded a Sensitive-marked node at path {node.path!r} with "
              f"value_provider=None. This is a walker bug — when with_values=True "
              f"every node MUST carry a value_provider closure. The test fails "
              f"loudly rather than silently skipping (rev-4 B1 hardening)."
          )
          sensitive_nodes.append(node)
      assert sensitive_nodes, (
          f"Tool {tool_name!r}'s argument_model {model.__name__!r} is registered "
          f"as a type-driven manifest entry but walk_model_schema found no "
          f"Sensitive-marked paths. Either the manifest entry should be "
          f"declarative (no Sensitive[T] fields) or walk_model_schema is "
          f"failing to detect the markers. The test fails loudly rather than "
          f"silently passing on a no-op iteration (rev-4 B1 hardening)."
      )

      @given(st.from_type(model))  # type: ignore[arg-type]
      @settings(max_examples=50, deadline=None)
      def check(payload: object) -> None:
          raw_args = payload.model_dump()
          redacted_args = redact_tool_call_arguments(
              tool_name, raw_args, telemetry=NoopRedactionTelemetry()
          )

          for node in sensitive_nodes:
              # Extract values at this path from BOTH views. value_provider is
              # path-aware: for scalar paths it returns the value directly; for
              # container-descent paths ([*] or {*}) it returns an iterable of
              # (key_or_index, value) pairs. We let any contract violation
              # (e.g. value_provider returns a non-iterable for a container
              # path) propagate as an exception — this is a walker bug and the
              # test must surface it, not absorb it via try/except.
              raw_extracted = node.value_provider(raw_args)
              redacted_extracted = node.value_provider(redacted_args)

              if _is_container_descent_path(node.path):
                  # Container descent: compare key-by-key.
                  raw_pairs = dict(raw_extracted)
                  redacted_pairs = dict(redacted_extracted)
                  # The redactor must NOT alter the container's key structure;
                  # only values at Sensitive paths are substituted.
                  assert raw_pairs.keys() == redacted_pairs.keys(), (
                      f"Redaction altered the key set of the container at path "
                      f"{node.path!r} for tool {tool_name!r}. "
                      f"Raw keys: {sorted(raw_pairs)!r}; "
                      f"redacted keys: {sorted(redacted_pairs)!r}. "
                      f"Redaction must preserve container shape — only values "
                      f"at Sensitive paths are substituted, never keys."
                  )
                  # An empty container yields no comparisons for this example.
                  # Hypothesis will generate non-empty containers in other
                  # examples (max_examples=50). The `event()` call below makes
                  # the empty-container ratio observable in Hypothesis's
                  # summary output, so a parametrize entry whose strategy
                  # produces ONLY empty containers across all 50 examples
                  # surfaces as a high event count rather than passing as a
                  # silent zero-assertion run (rev-5 systems W-empty-container).
                  # Currently no MANIFEST entry uses sub-element Sensitive
                  # markers (all markers are on the field itself), so this
                  # branch is unreachable today; the instrumentation is
                  # forward-safety against future entries.
                  if not raw_pairs:
                      event(f"empty_container:{tool_name}:{node.path}")
                  for key, raw_value in raw_pairs.items():
                      if raw_value is None:
                          # Skip rule (the ONLY allowed skip): no sensitive
                          # data at this path for this example. Asserting
                          # redacted != None would either tautologically fail
                          # for a correct redactor (passes None through) or
                          # pass spuriously for an over-aggressive one
                          # (substitutes None with a sentinel). Out of scope.
                          continue
                      redacted_value = redacted_pairs[key]
                      assert redacted_value != raw_value, (
                          f"Sensitive value at container path {node.path!r}"
                          f"[{key!r}] for tool {tool_name!r} was NOT redacted. "
                          f"raw={raw_value!r}, redacted={redacted_value!r}. "
                          f"redact_tool_call_arguments must replace the value "
                          f"at every Sensitive path with the summarizer output "
                          f"or the fixed sentinel."
                      )
              else:
                  # Scalar descent.
                  if raw_extracted is None:
                      continue  # only allowed skip; see container branch
                  assert redacted_extracted != raw_extracted, (
                      f"Sensitive value at scalar path {node.path!r} for tool "
                      f"{tool_name!r} was NOT redacted. "
                      f"raw={raw_extracted!r}, redacted={redacted_extracted!r}. "
                      f"redact_tool_call_arguments must replace the value at "
                      f"every Sensitive path with the summarizer output or the "
                      f"fixed sentinel."
                  )

      check()
  ```

  **Implementer cross-check before commit (mandatory).** Before commit, re-read the test body against this checklist. Each item maps to a known silent-pass mode from a prior review pass:

  - No `isinstance(...)` filter appears before any `assert` statement. (rev-4 B1: isinstance(str) silently exempted non-str T.)
  - No `if node.value_provider is None: continue` appears. The `assert node.value_provider is not None` runs once during node discovery and fails the whole test if the walker regresses. (rev-4 B1 secondary mode.)
  - No `except (TypeError, ValueError):` or any other broad `except` wraps the value-extraction or comparison code. Walker contract violations propagate. (rev-4 B1 tertiary mode.)
  - The only `continue` is the explicit `raw_value is None` skip, with a comment explaining exactly why this is the only allowed skip.
  - The discovery loop emits an empty `sensitive_nodes` list → the outer `assert sensitive_nodes` fails loudly with a precise message, so a misclassified type-driven entry with no Sensitive fields cannot produce a green vacuous-pass test.
  - For container-descent paths, the key-set equality assertion runs *before* the per-key value comparison, so a redactor that drops keys is caught even when the values would have matched.

  **Reviewer note for the implementer's reviewer.** The test must be inspected for the six items above. A code-quality reviewer signing off without re-checking each item is a process gap (this is the third review pass that has converged on a silent-pass mode in this file). Cite each item in the commit message body.

  **Collection-time verification step (rev-5 architecture A2).** `st.from_type(model)` for Pydantic 2.x models with `extra="forbid"` and `Annotated[dict[str, Any], Sensitive(...)]` fields may raise `ResolutionFailed` at Hypothesis collection time for some model shapes. A collection failure on a single parametrize entry can disable the property test for that tool WITHOUT producing an obvious failure signal in the pytest run (the entry is reported as an error, but the rest of the file passes and the operator may miss the error line in a long log). Before the commit that lands the property test, run:

  ```bash
  .venv/bin/python -m pytest tests/unit/web/composer/test_redaction_completeness_property.py --collect-only -q
  ```

  Confirm every `MANIFEST` type-driven entry produces a parametrize entry that collects without error. If any entry fails collection with `hypothesis.errors.ResolutionFailed` or `hypothesis.errors.InvalidArgument`, the model needs a custom Hypothesis strategy (register it via `hypothesis.strategies.register_type_strategy(Model, custom_strategy)` in a `conftest.py` adjacent to the test file). Fail the gate before commit, not after.

- [ ] **Walker-guard parity test (rev-2 M_walker_guard_parity).** Create `tests/unit/web/composer/test_walker_guard_parity.py`:

  ```python
  """Behavioural parity: walk_model_schema(M, with_values=False) == walk_model_schema(M, with_values=True)
  path-sets for each manifest entry's argument model.

  Pins the structural coupling claim: walker and guard cannot diverge in path
  enumeration or marker detection because they share one iterator. Closes
  rev-2 M_walker_guard_parity (quality MAJOR-3 + systems MINOR-1).
  """
  from __future__ import annotations

  import pytest

  from elspeth.web.composer.redaction import MANIFEST, TraversalNode, _SensitiveMarker, walk_model_schema


  @pytest.mark.parametrize(
      "tool_name",
      [name for name, entry in MANIFEST.items() if entry.argument_model is not None],
  )
  def test_walker_guard_path_sets_are_identical(tool_name: str) -> None:
      entry = MANIFEST[tool_name]
      model = entry.argument_model

      guard_nodes = list(walk_model_schema(model, with_values=False))
      walker_nodes = list(walk_model_schema(model, with_values=True))

      def has_sensitive(node: TraversalNode) -> bool:
          return any(isinstance(m, _SensitiveMarker) for m in node.metadata)

      guard_paths = {(n.path, has_sensitive(n)) for n in guard_nodes}
      walker_paths = {(n.path, has_sensitive(n)) for n in walker_nodes}

      assert guard_paths == walker_paths, (
          f"Walker and guard path-sets diverged for tool {tool_name!r}.\n"
          f"Only in guard: {guard_paths - walker_paths}\n"
          f"Only in walker: {walker_paths - guard_paths}"
      )
  ```

- [ ] Property test for the summarizer contract (spec §9 RSK-03): for every committed summarizer, generate ~10000 representative inputs (Hypothesis strategies match the field's declared type) and assert (i) the summarizer returns `str` for every input; (ii) the summarizer does not raise. Failure is a system bug; the test is a regression net.

- [ ] Telemetry sanity check: with the production `OtelRedactionTelemetry`, run a representative compose-loop integration test and assert the named counters fire (no silent counter-name typos).

- [ ] Document gate state in a final commit: `chore(composer/redaction): Phase 2 gate green`. The commit body lists every gate command and its output summary (counts of tests passed, mypy clean, ruff clean, etc.).

---

## Task 20: Surface to operator for PR-open decision

**Steps:**

- [ ] Summarise the Phase 2 work in a brief surface-to-operator note: branch state, total commits, gate state, list of promoted tools and declarative entries, link to spec rev-5 and §12.2 traceability.
- [ ] **Do NOT run `gh pr create`.** Per Phase 1B/1C convention and operator memory, PR-open is operator-confirmed. The plan ends here.
- [ ] If the operator approves PR-open: create the PR with a description that includes (1) summary, (2) test plan, (3) "Redaction policy weakening rationale" section if any committed change weakens the snapshot relative to main (none expected for the Phase 2 greenfield, but the section header must be present so the label-gate CI step does not block the bootstrap commit), (4) reference to spec §12.2 traceability for each closed BLOCKER and warning.

---

## Summary

Phase 2 ships the manifest-keyed redaction primitive, promotes a wave of ~6–8 sensitive-touching tools to type-driven Pydantic argument models with dispatch validation, declares declarative manifest entries for the remaining ~29–31 tools, and enforces coverage and weakening with a single shared traversal iterator consumed by both the CI-time adequacy guard and the runtime walker, plus a content-keyed policy-hash snapshot and a CI-enforced PR-label gate.

**What this plan deliberately does NOT do:**

- It does not modify the compose loop's transactional structure (Phase 3).
- It does not modify the database schema (Phase 1).
- It does not create a `.github/CODEOWNERS` file (the team that rev-4 routed to cannot exist on this repo; label-gate is the rev-5 control).
- It does not promote non-sensitive-touching tools to Pydantic argument models (the operator's "remove loose dicts is always ongoing" direction continues outside Phase 2).
- It does not run `gh pr create` (operator-confirmed per Phase 1B/1C convention).

## Spec

- `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` — revision 5.

## Depends on

- Phase 1A + 1B + 1C have landed on the umbrella branch `feat/composer-progress-persistence-1a` (HEAD `f5115fd5` or later).
- The pre-Phase-2 hygiene pass is complete (gate green per preflight).

## Out of scope (later phases)

- Phase 3: compose-loop persistence.
- Phase 4: frontend recovery panel.
- Loose-dict elimination for the ~29–31 tools that take declarative manifest entries in Phase 2.
- Transition to an organisation-owned repo (would unlock CODEOWNERS team routing as defense-in-depth atop the label-gate; filed as a follow-up).

**Phase 3 preconditions established in Phase 2 (rev-2 BLOCKER_A A.6):**

- Phase 3 MUST call `redact_tool_call_arguments` only AFTER MANIFEST membership is confirmed for the dispatched tool name. `redact_tool_call_arguments` raises `AuditIntegrityError` on missing tool names — reserved for system-internal registry violations, NOT for Tier-3 LLM hallucinations. Unknown LLM-supplied tool names must route through `tools.py:5481` (`_failure_result`) BEFORE redaction is attempted.
- Phase 3 MUST NOT thread redacted arguments through `begin_dispatch` / `begin_dispatch_or_arg_error` at `service.py:1930`. `arguments_canonical` retains raw LLM arguments per spec §4.2.8 posture (a). Redacted views land in `chat_messages` only.
- Phase 3 must verify Tier-1 access controls on the MCP JSONL sidecar and `BufferingRecorder` surfaces before wiring.
- **Independent integration check at the persistence boundary (rev-5 systems W-phase3-independent-check).** Phase 3's plan MUST include at least one end-to-end test that drives a promoted tool OTHER than `set_source` (Task 6 in Phase 2 already covers `set_source`) through the full compose-loop → redaction → `chat_messages` write path and asserts the persisted row does NOT contain the raw Sensitive value. Phase 2's Hypothesis property test is structurally strong (rev-5 quality verdict: no residual silent-pass modes) but it tests the redactor in isolation; an independent integration check at the persistence boundary protects against future regressions that decouple "redactor produces correct output" from "compose loop persists the redacted output" — exactly the kind of failure-mode independence that rev-3/4/5 reviewer convergence taught us to engineer. Suggested target: `create_blob` or `set_pipeline` end-to-end through to `chat_messages.tool_calls` JSON, with a canary value in the Sensitive field, asserting the canary does not appear in the persisted row.

## Test plan

- All §8.1 redaction-policy unit tests pass; all eleven container-shape tests in `test_walk_model_schema.py` pass (original six, four rev-2 additions, plus the Optional[Annotated[scalar, Sensitive()]] arm test added in rev-4 W8a); tracer-bullet test (`test_redact_set_source.py`) including serialization boundary canary test and Pydantic-coerced datetime summarizer test passes; per-tool `ToolArgumentError`-routing tests (with `pydantic.ValidationError` as `__cause__`) pass for all promoted tools; adequacy-guard five core assertions pass (including `extra="forbid"` check) plus walker completeness floor-check (rev-3 M1); policy-hash snapshot equals committed file; hash-semantics tests (replacement flips hash; closure-state mutation does not) pass; compose-loop unknown-tool-name routing test pins `tools.py:5481` fall-through behaviour plus Phase 3 call-order precondition pin (rev-3 M2); direction-aware label-gate workflow passes direction-misclassification tests; Hypothesis property test passes using the rev-4 type-agnostic path-aware `!=` invariant (closes rev-3 B1 path-traversal defect AND rev-4 B1 isinstance/None-skip/except-fallback silent-pass family — no `isinstance` filter, no `value_provider is None` skip, no defensive `except` fallback); walker-guard parity test passes; per-entry runtime smoke test (`test_declarative_manifest_runtime_smoke.py`) is green for every declarative manifest entry (rev-3 W7 / rev-4 W7); response walker empty-`known_response_keys` test and atomicity test pass (rev-3 W8b/W8c / rev-4 W8b/W8c).

## Phase 2 Done When

- [ ] `tests/unit/web/composer/test_adequacy_guard.py` passes all five assertions (registry-manifest set equality; per-entry shape walk; mass-copy uniqueness; policy-hash snapshot equality; `extra="forbid"` on type-driven entries).
- [ ] Every promoted tool's dispatch validates via `Model.model_validate(arguments)`; promoted handlers catch `pydantic.ValidationError` and re-raise as `ToolArgumentError` (per `tools.py:2668–2801` pattern); `ToolArgumentError` is caught at `service.py:2480` and routes to `ARG_ERROR`.
- [ ] Every manifest entry is either type-driven (`argument_model` set) or declarative (`policy` set), never both, never neither.
- [ ] Every type-driven argument model sets `model_config = ConfigDict(extra="forbid")`.
- [ ] `redaction_policy_snapshot.json` covers all manifest entries (type-driven + declarative) and is committed.
- [ ] `composer-redaction-gate.yml` workflow is direction-aware: computes `sensitive_path_count` per entry, rejects `policy-strengthen` on a weakening diff and `policy-weaken-justified` on a strengthening diff.
- [ ] `composer.redaction.summarizer_errors_total` SLO threshold is asserted to be 0 in production telemetry config (RSK-03).
- [ ] `composer.redaction.unknown_response_key_total` and `composer.redaction.manifest_dispatch_total` counters fire in a representative integration scenario.
- [ ] Compose-loop unknown-tool-name routing test confirms LLM-supplied unknown tool names route via `tools.py:5481` `_failure_result` fall-through; the loop records and continues.
- [ ] **Integration scenario pinned end-to-end (rev-2 BLOCKER_A).** At least one test exercises `redact_tool_call_arguments("set_source", args_with_canary, telemetry=NoopRedactionTelemetry())` → `json.dumps` → asserts (a) canary absent, (b) key present, (c) redacted value matches summarizer output.
- [ ] **Hypothesis completeness property test passes with the rev-4 invariant (rev-2 BLOCKER_A / rev-3 B1 / rev-4 B1).** `test_redaction_completeness_property.py` with `settings(max_examples=50, deadline=None)`. The test MUST be inspected against the six-item implementer cross-check listed in Task 19 (no `isinstance` filter, no `value_provider is None` skip, no defensive `except` fallback, only `raw_value is None` skip, `assert sensitive_nodes` empty-list guard, key-set equality assertion for container descent). A code-quality reviewer signing off without re-checking each of those six items is a process gap.
- [ ] **Walker-guard parity test passes (rev-2 M_walker_guard_parity).** `test_walker_guard_parity.py` asserts identical path-sets between `with_values=False` and `with_values=True` modes for each manifest entry's argument model.
- [ ] **Walker completeness floor-check passes (rev-3 M1).** `test_walker_emits_node_for_every_top_level_field` in `test_adequacy_guard.py` asserts that every `model_fields` key appears as a path root for each type-driven manifest entry's argument model.
- [ ] **Phase 3 call-order pin passes (rev-3 M2).** `test_redact_tool_call_arguments_raises_for_unknown_tool` in `test_compose_loop_unknown_tool_name.py` asserts `AuditIntegrityError` when `redact_tool_call_arguments` is called with a tool name absent from MANIFEST.
- [ ] **Per-entry runtime smoke for declarative manifest entries passes (rev-3 W7 / rev-4 W7).** `test_declarative_manifest_runtime_smoke.py` exercises every `policy is not None` entry through `redact_tool_call_arguments` and `redact_tool_call_response`, catching typos in `sensitive_argument_keys`/`sensitive_response_keys`/`known_response_keys` that pass structural adequacy.
- [ ] **Optional[Annotated[scalar, Sensitive()]] walker test passes (rev-3 W8a / rev-4 W8a).** `test_walk_optional_annotated_scalar_arm` in `test_walk_model_schema.py` asserts the `_SensitiveMarker` survives the Optional unwrap on the scalar arm.
- [ ] **Response walker atomicity + empty-known-response-keys tests pass (rev-3 W8b/c / rev-4 W8b/c).** Empty-`known_response_keys` test pins fail-closed behaviour (every key → fixed sentinel + telemetry counter); atomicity test pins that mid-walk raise does not return a partial dict.
- [ ] **`_TOOL_REQUIRED_PATHS` unconditional removal complete for all promoted tools (rev-3 N7 / rev-4 M1).** Grep `grep -Rn '_TOOL_REQUIRED_PATHS\[' src/elspeth/web/composer/service.py` shows no remaining entries for promoted tools (`set_source`, `create_blob`, `update_blob`, `set_source_from_blob`, `set_pipeline`, `apply_pipeline_recipe`, and any `patch_*_options` promoted in Task 15).
- [ ] **`"MissingRequiredPaths"` test-pin sweep complete (rev-4 M2).** Grep `grep -Rn '"MissingRequiredPaths"\|MissingRequiredPaths' tests/ src/elspeth/web/ --include='*.py'` shows: (a) no matches in `tests/` exercise a promoted tool's failure path with the old assertion; (b) any remaining matches reference NON-promoted tools whose `_TOOL_REQUIRED_PATHS` entries remain (valid carry-overs).
- [ ] **`sensitive_response_keys ⊆ known_response_keys` adequacy assertion passes (rev-5 quality W7 hardening).** Task 10's sixth assertion proves no declarative entry has a misspelled `sensitive_response_keys` whose actual key is on the `known_response_keys` passthrough — closes the dual-typo silent-pass that the Task 16-final-a runtime smoke alone could not catch.
- [ ] **Sensitive short-circuit walker test passes (rev-5 architecture A1).** `test_walk_short_circuits_under_sensitive_container` in `test_walk_model_schema.py` asserts the walker does NOT descend past a Sensitive-marked container into inner Sensitive markers — forward-guard for future entries using nested Sensitive composition.
- [ ] **Property test collection-time check (rev-5 architecture A2).** `pytest tests/unit/web/composer/test_redaction_completeness_property.py --collect-only -q` reports zero collection errors. A `hypothesis.errors.ResolutionFailed` on any parametrize entry is a blocking issue that must be resolved with a custom strategy before commit, not after.
- [ ] No `.github/CODEOWNERS` file is created in this PR.
- [ ] Project gate is green (pytest unit + integration; mypy; ruff; tier-model; freeze-guard).
- [ ] Operator has been surfaced the deliverables and the PR-open decision is awaiting confirmation.

---

**Appendix A — plan-review finding closure (cross-reference to spec §12.2)**

| Finding | Closed by |
|---|---|
| **B1** Class-based `ComposerTool` hierarchy fictional | Tasks 2, 9, 10 (manifest dataclass + registry-equality + per-entry walk based on actual function-pointer dispatch); preflight references the actual lines 5250–5314 |
| **B2** Walker omits container-type recursion | Task 1 (shared iterator with all six container shapes covered); Tasks 7, 8, 10 (both walker and guard consume the same iterator) |
| **B3** Spec false BaseModel-declaration claim | Spec rev-5 (already committed before plan); plan preflight requires rev-5 |
| **B4** Unconditional `gh pr create` | Task 20 (operator-confirmed; plan ends without PR open) |
| **W1** ComposerTool name collision | No `ComposerTool` name introduced; Task 2's manifest type is `ToolRedaction` |
| **W2** Calendar synchronicity | Task 5 (no `last_reviewed_iso`); content-keyed snapshot |
| **W3** MissingToolError on Tier-3 hallucination | Task 17 (pins `tools.py:5481` `_failure_result` fall-through; no MissingToolError introduced; escalation rule: report to operator if actual status is not ARG_ERROR) |
| **W4** Untyped telemetry | Task 3 (typed Protocol; no `None` default) |
| **W5** Non-str summarizer return | Task 7 (assertion in walker; AuditIntegrityError) |
| **W6** Length-disclosure sentinel | Task 7 (fixed sentinel `<redacted-unknown-response-key>`) |
| **W7** Mass-copy uniqueness absent | Task 11 (assertion 3) |
| **W8** Sensitive[T] vs declarative precedence | Task 2 (construction-time `ValueError` for both shapes set) |
| **W9** Non-existent CODEOWNERS team | Task 18 (label-gate; no CODEOWNERS file) |
| **W10** Task 9 under-scoped | Task 16 split into 7 sub-tasks by registry |
| **W11** Default-form hasattr/getattr | All tasks reference spec §4.2 / §4.4 sketches that use direct typed-attribute access; code-quality reviewer assignment per task |
| **W12** Snapshot covers only legacy | Task 12 (broadened to all manifest entries) |
| **M1** test_redaction_policy.py is create | Task 6 file lists `Create` |
| **M2** sorted() on non-string keys | Task 12 (`_entry_hash` sorts only `str`-typed key sets; manifest keys are tool names of `dict[str, ToolRedaction]`) |
| **M3** Double JSON parse | Task 8 (walker accepts already-decoded `dict[str, Any]`; compose loop decodes once at `service.py:1838`) |
| **M4** Adequacy guard CI scaling | Task 19 (sanity bound: < 5s for 37 registry tools plus `request_advisor_hint` — 38 MANIFEST entries total) |

**Appendix B — rev-2 plan-review finding closure (cross-reference to spec §12.3)**

| Finding ID | Closed by |
|---|---|
| **BLOCKER_A** ValidationError routing + arguments_canonical + serialization boundary | Task 4 (ToolArgumentError wrap pattern + serialization boundary canary test); Tasks 13–15 (same pattern, citing Task 4); Task 19 (Hypothesis completeness property test + walker-guard parity test); Files NOT touched (composer_audit.py note); spec §4.2.6, §4.2.8, §11 Phase 3 preconditions |
| **BLOCKER_B** Label-gate direction not enforced | Task 18 (direction-aware workflow with sensitive_path_count); test_label_gate_direction.py (4 combination tests) |
| **BLOCKER_C** Summarizer purity / false-negative class | Task 12 (hash-semantics tests: replacement flips hash; closure-state mutation does not — with documented false-negative docstring); spec §4.2.3, §4.4.3 |
| **M_adequacy_mechanical_enforcement** (M.1–M.4) | M.1: Tasks 4, 13, 14 (extra=forbid on all promoted models); M.2: Task 10 (fifth adequacy assertion for extra=forbid); M.3: Task 10 (AST inspection removed; declarative consistency check only); M.4: Task 1 (four additional walker container-shape tests) |
| **M_telemetry_implementation** (M.6–M.8) | Task 3 (module-level create_counter() + .add() pattern; OtelRedactionTelemetry rewired; summarizer_error counter wired in Protocol and walker) |
| **M_walker_guard_parity** (M.5) | Task 19 (test_walker_guard_parity.py) |
| **M_governance_single_owner** (M.9) | docs/guides/redaction-policy-changes.md (new file with single-owner note) |
| **M_blob_summarizer_type_variability** (M.10) | Task 13 (explicit type-variability verification step) |
| **m_script_dir_missing** | Task 12 (script relocated to scripts/cicd/bootstrap_redaction_snapshot.py) |
| **m_citation_hygiene** | Tasks 4, 17, 18 commit comments; spec §4.2.6, §4.4.2, §5.7.5, §8.1, §11, §12.2 |

**Appendix C — rev-3 plan-review finding closure**

| Finding ID | Closed by |
|---|---|
| **rev-3 B1** Property test path-traversal defect | Task 19 `value_provider`-based path extraction (replaces ad-hoc `path.split(".")` loop; container-descent nodes yield `(key, value)` pairs correctly) — **subsequently SUPERSEDED by rev-4 B1 rewrite (see Appendix D); the rev-3 fix admitted an `isinstance(str)` silent-pass for non-str `T` that the rev-4 type-agnostic invariant eliminates.** |
| **rev-3 M1** Iterator completeness floor-check missing | Task 10 walker-completeness floor-check (`test_walker_emits_node_for_every_top_level_field`; asserts every `model_fields` key appears as a walk root) |
| **rev-3 M2** Phase 3 call-order precondition not mechanically enforced | Task 17 call-order precondition pinning test (`test_redact_tool_call_arguments_raises_for_unknown_tool`; pins that `redact_tool_call_arguments` raises `AuditIntegrityError` for unknown tool names, mechanically constraining Phase 3's call site ordering) |
| **rev-3 W1/W2** Preflight stale test counts (1565 unit, ≥6 integration) | Preflight refreshed with verified 2026-05-12 counts: 14,948 unit, 785 integration; guidance now reads "verify zero failures, not exact count" |
| **rev-3 W6** Summarizer `json.dumps` raise risk on Pydantic-coerced types | Task 4 Step 3a Pydantic-coerced datetime test; `_summarize_set_source_options` uses `default=str` |
| **rev-3 W7** No per-entry runtime smoke for declarative entries | Task 16-final-a `test_declarative_manifest_runtime_smoke.py` — parametrised over every `policy is not None` entry, exercises both argument and response redaction paths |
| **rev-3 W8a** Optional[Annotated[scalar, Sensitive()]] arm uncovered | Task 1 `test_walk_optional_annotated_scalar_arm` — asserts `_SensitiveMarker` survives Optional unwrap on scalar arm |
| **rev-3 W8b** Walker atomicity contract unspecified | Task 7 test case "Walker atomicity contract" — mid-walk raise propagates AND does not return partial dict |
| **rev-3 W8c** `redact_tool_call_response` with empty `known_response_keys` | Task 7 test case "Empty `known_response_keys`" — every key → fixed sentinel + telemetry counter |
| **rev-3 A2** TraversalNode freeze-elision rationale inline | Task 1 `TraversalNode` docstring expanded with explicit freeze-elision design assumption and the conditions under which it must be revisited |
| **rev-3 N3 / A4** `test_service.py:367` MissingRequiredPaths string pin | Task 14 `set_pipeline` promotion includes explicit update of `test_service.py:367` from `"MissingRequiredPaths"` to `"ValidationError"` assertion (this pin is `set_pipeline`-specific, not `set_source`, so it updates in Task 14 not Task 4) |
| **rev-3 N7 / A3** `_TOOL_REQUIRED_PATHS` removal conditional in plan | Tasks 4, 13, 14 now state "REMOVE unconditionally" — see rev-4 M1 in Appendix D |

**Appendix D — rev-4 plan-review finding closure**

| Finding ID | Closed by |
|---|---|
| **rev-4 B1** Property test `isinstance(str)` / `value_provider is None` skip / `except` fallback silent-pass family | Task 19 property test rewritten with the type-agnostic path-aware `!=` invariant. The test compares `value_provider(raw_args)` vs `value_provider(redacted_args)` at each Sensitive node, with NO `isinstance` filter, NO `value_provider is None` skip (becomes `assert` instead), and NO defensive `except` fallback. The only allowed skip is `raw_value is None` (no sensitive data at this example's path). Discovery loop asserts `sensitive_nodes` is non-empty to prevent vacuous-pass on a misclassified type-driven entry. A six-item implementer cross-check is added to the task body; the Done-When item requires inspection against that checklist. |
| **rev-4 M1** `_TOOL_REQUIRED_PATHS` removal made unconditional in plan | Tasks 4, 13, 14 rewritten: removal is mandatory (not conditional on a "redundancy check"); rationale cites CLAUDE.md "No Legacy Code Policy" and the strict-strengthening of `extra="forbid"` over dotted-path checks. |
| **rev-4 M2** `"MissingRequiredPaths"` test-pin grep-and-update step missing | Task 4 Step 4 grows a "MissingRequiredPaths test-pin update" subsection with the canonical grep command, per-match decision rules, and the confirmed pin at `test_service.py:367` flagged as Task 14's responsibility (it pins `set_pipeline`, not `set_source`). Tasks 13 and 14 cite Task 4 as the canonical procedure and add per-tool reminders. Done-When adds an explicit grep-clean criterion. |
| **rev-4 W7** Per-entry runtime smoke for declarative entries (rev-3 W7 carry-forward) | New Task 16-final-a as documented above. |
| **rev-4 W8a/b/c** Optional scalar arm / atomicity / empty known_response_keys (rev-3 W8 carry-forwards) | Task 1 (W8a), Task 7 (W8b/c) as documented above. |
| **rev-4 W1/W2** Preflight stale 1565 figure still in prose (rev-3 carry-forward) | Preflight refreshed with verified 2026-05-12 counts (14,948 unit, 785 integration). |
| **rev-4 A2** TraversalNode freeze-elision inline comment (rev-3 A2 carry-forward) | Task 1 `TraversalNode` docstring expanded as documented above. |
| **rev-4 R-W3** JSON-decode gate line citation drift (1836-1870 → 1838-1881) | Task 17 citation refreshed to `service.py:1838–1881` (verified via grep against current `service.py`). |
| **rev-4 S5 / N6** Bulk editorial PR clearing the label-gate's single check | Documented limitation. The rev-5 spec §4.4.5 promoted the PR-label gate from secondary to primary control because CODEOWNERS cannot exist on a personal-account repo. A bulk-editorial PR that flips 12+ hashes simultaneously clears one label gate. Mitigation deferred to a future phase (post-Phase-2): a per-entry hash-diff count threshold in the label-gate workflow, OR an out-of-band reviewer escalation rule. Not blocking Phase 2 because the single-owner posture is unchanged from rev-3, and the operator is the only person able to author such a PR. |
| **rev-4 S4 / N5** `m_dynamic_tool_name` not closed in Appendix B | Spec rev-5 §4.2.2 covers the static-name discipline of MANIFEST entries (tool names are literal `str` keys at module load; no dynamic construction). Adding the closure entry here for visibility; no further code change needed. |

**Appendix E — rev-5 plan-review finding closure**

Rev-5 was the first review pass with **no BLOCKERs** across all four reviewers. The 3-way convergence pattern that drove rev-3 and rev-4 was broken: Quality and Systems both confirmed the rev-4 property test rewrite is a "genuine structural inversion" (type-agnostic Python `__ne__` has no comparison-site filter that could be silently re-inserted, unlike the prior string-substring + isinstance(str) shape). All four reviewers returned APPROVED_WITH_WARNINGS.

The warnings folded into this plan revision:

| Finding ID | Closed by |
|---|---|
| **rev-5 quality W7-PARTIAL** Dual-typo: `sensitive_response_keys=("contents",)` (misspelled) AND correctly-spelled `"content"` in `known_response_keys` passes the runtime smoke. | Task 10 sixth adequacy assertion: `set(policy.sensitive_response_keys) <= set(policy.known_response_keys)` per declarative entry, construction-time, no payload required. |
| **rev-5 reality W-OPT-PATHS** `_TOOL_OPTIONAL_PATHS` companion symbol referenced but does not exist. | Removed speculative reference from Tasks 4 and 13; added correction note. |
| **rev-5 architecture A1** Nested Sensitive-in-Sensitive container descent is unspecified and would cause confusing `TypeError` in the property test. | Task 1 walker contract: Sensitive marker on a container short-circuits descent; new `test_walk_short_circuits_under_sensitive_container` pins the behaviour. Latent until a future MANIFEST entry uses sub-element Sensitive composition. |
| **rev-5 architecture A2** Hypothesis `ResolutionFailed` at collection time disables a parametrize entry silently in pytest output. | Task 19 collection-time verification step: `pytest test_redaction_completeness_property.py --collect-only -q` must report zero errors before the property-test commit. Done-When adds the explicit criterion. |
| **rev-5 systems W-empty-container** Empty raw container produces a zero-assertion run; currently unreachable (no MANIFEST entry uses sub-element Sensitive) but latent. | Hypothesis `event()` instrumentation added inside the container-descent branch — empty-container ratio is observable in the test's summary output, so a parametrize entry whose strategy produces only empty containers across all 50 examples surfaces as a high event count rather than passing silently. |
| **rev-5 systems W-phase3-independent-check** Phase 3 inherits the property test's correctness claim without an independent integration check at the persistence boundary. | Added explicit Phase 3 precondition: Phase 3's plan MUST include an end-to-end test driving a promoted tool other than `set_source` through compose-loop → redaction → `chat_messages` write, asserting the canary value is absent from the persisted row. |

Warnings NOT folded into this revision (deferred with rationale):

| Finding ID | Disposition |
|---|---|
| **rev-5 architecture A3** Done-When list is 22 flat items; could be grouped by concern. | Stylistic; no mechanical effect. Deferred — re-evaluate if a future revision adds further items. |
| **rev-5 systems W-done-when-manual-enforcement** Done-When checklist is enforced via human checkboxes, not CI. | Out of scope for plan-doc edits; a `test_phase2_done_when.py` meta-test that imports each named test file would mechanically enforce a subset, but creating that test belongs to a separate "process meta-test" workstream. |
| **rev-5 systems W-bulk-editorial-pr-saturation** Carry-forward from rev-4. | Documented limitation; operator is sole PR author, risk bounded in practice. |
| **rev-5 reality minor R-W-test-count-presentation** Cited 14,948 unit / 785 integration are selection counts. | Counts are accurate for the cited commands; the preflight prose already says "verify zero failures, not exact count." |
