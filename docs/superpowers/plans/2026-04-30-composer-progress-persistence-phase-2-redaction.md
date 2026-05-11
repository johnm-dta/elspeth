# Composer Progress Persistence — Phase 2: Redaction Framework (rev 5 rewrite)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Introduce a manifest-keyed redaction primitive (`MANIFEST: dict[str, ToolRedaction]`) that mirrors the project's existing `_TOOL_REQUIRED_PATHS: dict[str, ...]` precedent at `src/elspeth/web/composer/service.py:702`, alongside the function-pointer dispatch dicts at `src/elspeth/web/composer/tools.py:5250–5314`. Promote ~6–8 sensitive-touching composer tools to type-driven manifest entries with `Sensitive[T]`-annotated Pydantic argument models and `Model.model_validate` dispatch validation; promoted handlers catch `pydantic.ValidationError` and re-raise as `ToolArgumentError` (per `tools.py:2668–2801` pattern), which is caught at `service.py:2480` and routes to ARG_ERROR. Cover the remaining ~29–31 tools with declarative manifest entries. Enforce coverage and weakening with a single shared traversal iterator consumed by both the CI-time adequacy guard and the runtime walker, plus a content-keyed policy-hash snapshot and a direction-aware CI-enforced PR-label gate. Close all prior BLOCKERs and warnings, plus the rev-2 BLOCKERs and MAJORs from `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.review-rev2.json`.

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
  - `.venv/bin/python -m pytest tests/unit -q` (1565 collected, all pass)
  - `.venv/bin/python -m pytest tests/integration -q -m "not testcontainer"` (≥6 collected, all pass)
  - `.venv/bin/python -m mypy src/` (clean, 381 files)
  - `.venv/bin/python -m ruff check src/` (clean)
  - `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` (no findings beyond allowlisted)
  - `.venv/bin/python scripts/cicd/enforce_freeze_guards.py` (clean)
- [ ] **Spec rev-5 landed.** `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` Status line reads "revision 5"; §4.2, §4.4, §11 Phase 2, §12.2 reflect the manifest architecture (not the rev-4 class-hierarchy assumption). The plan-review JSON at `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json` is no longer load-bearing — this rewrite supersedes it.

If any of the above is red, **stop and surface to operator.** Do not begin Task 1.

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
  docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json.
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
      """One field encountered while walking a model schema (§4.2.5)."""
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
      reachable input value; must return str (spec §4.2.6 / §9 RSK-03)."""
      # The existing helper returns dict[str, Any]; we must return str.
      # Wrap by re-encoding to a sentinel-marked JSON. Decision documented
      # in the commit body; the simplest correct shape is to call the
      # existing helper for path redaction and re-stringify.
      redacted = redact_source_storage_path({"source": {"options": options}})
      return json.dumps(redacted["source"]["options"], sort_keys=True, separators=(",", ":"))


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

  **Important (W5 / spec §9 RSK-03):** the summarizer wrapper above MUST return `str`. The walker asserts the return type and raises `AuditIntegrityError` on any non-`str` return. Test the summarizer returns `str` for every reachable input — this is part of the property test (Task 19 gate).

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

  Also: confirm the compose-loop's existing `_TOOL_REQUIRED_PATHS["set_source"]` check at `service.py:2023` is now redundant (the Pydantic model is strictly more capable). If so, remove the entry from `_TOOL_REQUIRED_PATHS` to avoid divergence; if there is a meaningful difference (e.g. dotted-path checks the model can't express), document why both are kept.

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
- **Fifth assertion — `extra="forbid"` on type-driven entries (rev-2 M.2):** For every type-driven manifest entry, assert `entry.argument_model.model_config.get("extra") == "forbid"`. This prevents the `arguments_canonical` / walker discrepancy described in spec §4.4.2.

Commit: `test(composer/redaction): adequacy guard per-entry shape walk + extra=forbid assertion (§4.4.2, closes B2, M.2, M.3)`.

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
- [ ] If `_TOOL_REQUIRED_PATHS[tool]` is now redundant, remove it; otherwise document why both are kept.
- [ ] TDD per-tool test: assert `ToolArgumentError` raised on invalid inputs (with `pydantic.ValidationError` as `__cause__`); redaction substitutes Sensitive fields correctly. (Pattern from Task 4 — do not duplicate the rationale here; cite Task 4.)
- [ ] **Blob-tool summarizer type-variability step (rev-2 M_blob_summarizer_type_variability M.10):** Verify the summarizer for each blob tool's sensitive field handles every value type the Pydantic model admits. For `create_blob.content` (typed `str`): test the summarizer against `None` (verify `extra="forbid"` rejects it at model_validate; do NOT test the summarizer directly on `None`), empty string, ASCII string, non-ASCII string. The lambda `lambda b: f"<inline-blob:{len(b)}-bytes>"` is only safe for `str` — since `extra="forbid"` and Pydantic validates `content: str`, `None` or other types will be rejected before the summarizer runs; document this reasoning in the test.
- [ ] Existing integration tests for the tool pass.

Commit per tool (three commits): `feat(composer/redaction): promote {tool} to type-driven manifest entry`. Reviewer note in body: "Sensitive[T] promotion wave 2/4; extra=forbid; ToolArgumentError wrap pattern per Task 4."

---

## Task 14 (Sensitive[T] Wave 3): promote `set_pipeline`, `apply_pipeline_recipe`

Same shape as Task 13 (including `model_config = ConfigDict(extra="forbid")`, the `try/except ValidationError → ToolArgumentError` wrap pattern, and `ToolArgumentError`-not-`ValidationError` test assertions) but for two tools that have more complex argument schemas. `set_pipeline` is special-cased in `execute_tool()` at `tools.py:5436` (it can own `source.inline_blob`); the promotion must preserve that branch.

**Special caution for `set_pipeline`:** it has an extended dispatch signature (with `session_engine` / `session_id` kwargs). The Pydantic model only validates the LLM-supplied `arguments` dict; the kwargs are wired by the dispatcher at `execute_tool()` and are not user-facing. Document this in the model's docstring.

Two commits, one per tool.

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

The full enumeration (37 tools total minus those promoted in Tasks 4, 13, 14, 15):

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

- [ ] **16-final: regenerate snapshot.**
  - Run `scripts/cicd/bootstrap_redaction_snapshot.py`.
  - Verify `tests/unit/web/composer/test_adequacy_guard.py` is now fully green (all four assertions pass).
  - Commit the snapshot file: `chore(composer/redaction): bootstrap redaction policy hash snapshot`.

After Task 16 the manifest covers every tool name in the dispatch registries; the four adequacy assertions all pass.

---

## Task 17: Compose-loop unknown-tool-name routing test

**Files:**

- Create: `tests/unit/web/composer/test_compose_loop_unknown_tool_name.py`

**Why this task.** Plan-review M7 / W3. The previous plan introduced a `MissingToolError` crash for LLM-hallucinated tool names. Spec rev-5 §4.2.6 / §5.7.5 makes this an explicit boundary discipline: an LLM-supplied unknown tool name is Tier-3 input and must route through `_failure_result` + continue, not crash. The dispatcher's fall-through at `tools.py:5481` already handles this. The compose loop records the failure and continues. The JSON-decode / non-dict pre-dispatch gate at `service.py:1836–1870` is a separate, earlier site for malformed arguments — NOT the site for unknown tool names.

**Steps:**

- [ ] **Read the existing fall-through.** `tools.py:5481` is the dispatcher's final `return _failure_result(state, f"Unknown tool: {tool_name}")` — the dispatcher returns a failure `ToolResult`, it does not raise. Read the compose-loop's handling of this `ToolResult` (after `service.py:2480`'s `ToolArgumentError` catch block) to find where the failure `ToolResult` flows. Document the actual error_class and audit-status the recorder receives in this path.
- [ ] **Construct the test scenario.** A compose-loop scenario where the LLM emits a tool call with `function.name = "this_tool_does_not_exist"`. Drive it through the actual `execute_tool` + recorder path (no production-bypass shortcut per spec §8.6 / CLAUDE.md test-path-integrity).
- [ ] **Assert the observed behaviour with escalation rule (rev-2 m_task17_pinning_risk).** The dispatcher returns a failure `ToolResult` (no exception propagates). Verify the actual audit status produced for an unknown tool name. **If the actual status is `ARG_ERROR` (or the compose-loop constant for this condition), pin it.** If the actual status differs from `ARG_ERROR`, **report to operator — do not pin a non-`ARG_ERROR` status without escalation.** The LLM receives a `role=tool` message with the failure payload and the loop continues.
- [ ] **Do not introduce a `MissingToolError` exception class.** The dispatcher's existing failure-`ToolResult` path is the correct shape; the test pins it. The previous plan's `MissingToolError` design treated Tier-3 input as a Tier-1 crash condition (closed by plan-review M7 / W3 and spec §4.2.6 / §5.7.5).

Commit: `test(composer): pin compose-loop unknown-tool-name routing via tools.py:5481 (closes M7 W3)`.

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

- [ ] Adequacy guard sanity bound (closes M4): time the guard's runtime; assert it completes in under 5 seconds for the current 37-tool set. The bound is order-of-magnitude, not a tight budget — flake-source guidance per spec §1.4.

- [ ] **Hypothesis completeness property test (rev-2 BLOCKER_A MAJOR-4).** Create `tests/unit/web/composer/test_redaction_completeness_property.py`:

  ```python
  """Hypothesis property test: no Sensitive[T] field value reaches json.dumps output.

  This is the load-bearing test that proves no Sensitive[T] field value reaches
  serialization unchanged. Closes rev-2 BLOCKER_A (quality MAJOR-4).
  settings(max_examples=50, deadline=None) for stable CI execution.
  """
  from __future__ import annotations

  import json

  import hypothesis
  import hypothesis.strategies as st
  import pytest
  from hypothesis import given, settings

  from elspeth.web.composer.redaction import MANIFEST, walk_model_schema, _SensitiveMarker
  from elspeth.web.composer.redaction import redact_tool_call_arguments
  from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry


  @pytest.mark.parametrize(
      "tool_name",
      [name for name, entry in MANIFEST.items() if entry.argument_model is not None],
  )
  def test_no_sensitive_value_in_json_output(tool_name: str) -> None:
      """For each type-driven manifest entry, verify no raw Sensitive field value
      appears in the serialized output of redact_tool_call_arguments."""
      entry = MANIFEST[tool_name]
      model = entry.argument_model

      @given(st.from_type(model))  # type: ignore[arg-type]
      @settings(max_examples=50, deadline=None)
      def check(payload: object) -> None:
          args = payload.model_dump()
          result = redact_tool_call_arguments(tool_name, args, telemetry=NoopRedactionTelemetry())
          serialized = json.dumps(result, sort_keys=True)

          # For each Sensitive-annotated field, the raw value must NOT appear.
          sensitive_nodes = [
              n for n in walk_model_schema(model, with_values=False)
              if any(isinstance(m, _SensitiveMarker) for m in n.metadata)
          ]
          for node in sensitive_nodes:
              raw_value = args
              for part in node.path.split("."):
                  if isinstance(raw_value, dict):
                      raw_value = raw_value.get(part)
                  else:
                      break
              if isinstance(raw_value, str) and raw_value:
                  assert raw_value not in serialized, (
                      f"Sensitive value at path {node.path!r} appeared in serialized output "
                      f"for tool {tool_name!r}. Redaction failed to cover this path."
                  )

      check()
  ```

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

## Test plan

- All §8.1 redaction-policy unit tests pass; all ten container-shape tests in `test_walk_model_schema.py` pass (original six plus four rev-2 additions); tracer-bullet test (`test_redact_set_source.py`) including serialization boundary canary test passes; per-tool `ToolArgumentError`-routing tests (with `pydantic.ValidationError` as `__cause__`) pass for all promoted tools; adequacy-guard five assertions pass (including `extra="forbid"` check); policy-hash snapshot equals committed file; hash-semantics tests (replacement flips hash; closure-state mutation does not) pass; compose-loop unknown-tool-name routing test pins `tools.py:5481` fall-through behaviour; direction-aware label-gate workflow passes direction-misclassification tests; Hypothesis completeness property test passes; walker-guard parity test passes.

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
- [ ] **Hypothesis completeness property test passes (rev-2 BLOCKER_A).** `test_redaction_completeness_property.py` with `settings(max_examples=50, deadline=None)`.
- [ ] **Walker-guard parity test passes (rev-2 M_walker_guard_parity).** `test_walker_guard_parity.py` asserts identical path-sets between `with_values=False` and `with_values=True` modes for each manifest entry's argument model.
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
| **M4** Adequacy guard CI scaling | Task 19 (sanity bound: < 5s for 37 tools) |

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
