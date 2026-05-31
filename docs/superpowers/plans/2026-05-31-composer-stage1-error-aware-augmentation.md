# Stage 1 — Error-Aware Failure Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the composer from re-dumping a plugin's full ~5 KB schema on every option-shape validation rejection once the model has already received that schema this session; replace the repeat dump with the verbatim error plus an explicit, copy-pasteable tool-call pointer so convergence is preserved.

**Architecture:** A session-scoped "schemas delivered this session" set already exists on the composer service (`_schemas_loaded_by_session`, populated by `_mark_plugin_schema_loaded`, read via `_schemas_loaded_for_session`). We post-process the dispatched `ToolResult` in the tool-batch dispatch helper: for any plugin whose schema the model already has, strip it from `plugin_schemas` and instead set a new `schema_pull_directive` field carrying explicit `get_plugin_schema` / `get_plugin_assistance` invocations. Newly-delivered schemas are marked into the same tracker so the *next* rejection is suppressed. `execute_tool` and `_augment_with_plugin_schemas` are left untouched — all session-aware policy lives where the service and the tracker are already in scope.

**Tech Stack:** Python 3.13, pytest, frozen dataclasses (`elspeth.contracts.freeze`), the composer tool dispatch (`src/elspeth/web/composer/`).

**Spec:** `docs/superpowers/specs/2026-05-31-composer-reset-debrief-and-rootcause-fixes-design.md` (Stage 1; decisions D1, D2).

---

## Background facts (verified, with citations)

- The augment fires only on option-shape rejections: `_augment_with_plugin_schemas` (`tools/_dispatch.py:329-359`) → `build_plugin_schemas_for_failure` (`tools/_common.py:821-862`), which scans `result.validation.errors` for `_INVALID_OPTIONS_PLUGIN_RE` (`"Invalid options for (source|transform|sink) '<plugin>'"`, `_common.py:816-818`) and inlines `PluginSchemaInfo.model_dump()` keyed `"<kind>/<plugin>"`.
- `ToolResult` (`tools/_common.py`): `success: bool`, `validation: ValidationSummary`, `plugin_schemas: Mapping[str, Mapping[str, Any]] | None = None`. `to_dict()` emits `plugin_schemas` only when non-empty. `__post_init__` calls `freeze_fields(self, "plugin_schemas")` when non-None.
- The tracker (the key enabler):
  - store: `self._schemas_loaded_by_session: dict[str, set[tuple[str, str]]]` (`service.py:984`) — tuple is `(plugin_type, plugin_name)` where `plugin_type ∈ {source, transform, sink}`.
  - accessor: `_schemas_loaded_for_session(session_id) -> frozenset[tuple[str, str]]` (`service.py:2566`).
  - writer: `_mark_plugin_schema_loaded(session_id, plugin_type, plugin_name)` (`service.py:2581`); no-op when `session_id is None`.
  - existing write site: `tool_batch.py:1342` marks on every successful `get_plugin_schema` (args `arguments["plugin_type"]`, `arguments["name"]`).
- `get_plugin_schema` tool args: `name`, `plugin_type` (`tool_batch.py:1344`). `get_plugin_assistance` tool args: `plugin_type` (source/transform/sink), `plugin_name`, optional `issue_code` (`tools/generation.py:470-528`, `redaction.py:1968-1972`).
- At `tool_batch.py:1342` the following are in scope: `result`, `arguments`, `tool_name`, `session_id`, `ctx.service`.

## Why this design (and not the alternatives)

- **Not** threading the loaded-set into `execute_tool`: `execute_tool` is a pure sync dispatch with no service handle; the tracker lives on the service, reached in `tool_batch`. Post-processing there is the smallest, lowest-coupling change.
- **D2 is load-bearing.** The augment exists because agents almost never converged without context-specific guidance (`_dispatch.py:344`, composer session `47cfbb5e`). Suppression must therefore *never* drop the lifeline — it converts a *push* into an explicit *pull* (`schema_pull_directive`). A regression test asserts the directive is present, and a code comment cites this rationale so a future edit does not silently delete it.
- **Keep the first dump full** (operator decision): the first rejection for a not-yet-delivered plugin behaves exactly as today.

## File structure

- `src/elspeth/web/composer/tools/_common.py` — add `schema_pull_directive: str | None` to `ToolResult` + `to_dict` emission; add `build_schema_pull_directive()` helper and a `matched_invalid_option_plugins()` helper that returns the `(kind, plugin)` pairs the augment matched.
- `src/elspeth/web/composer/tool_batch.py` — add the delivery-aware suppression block immediately after the existing `get_plugin_schema` marking block (~`:1347`).
- `tests/unit/web/composer/test_failure_augmentation.py` — new test module (mirrors existing composer dispatch unit tests).

---

## Task 1: Add `schema_pull_directive` to `ToolResult`

**Files:**
- Modify: `src/elspeth/web/composer/tools/_common.py` (the `ToolResult` dataclass + `to_dict`)
- Test: `tests/unit/web/composer/test_failure_augmentation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/web/composer/test_failure_augmentation.py
from dataclasses import replace

from elspeth.web.composer.tools._common import ToolResult
from elspeth.web.composer.state import CompositionState, ValidationSummary


def _empty_result() -> ToolResult:
    state = CompositionState.empty()
    return ToolResult(
        success=False,
        updated_state=state,
        validation=ValidationSummary(is_valid=False, errors=(), warnings=(), suggestions=()),
        affected_nodes=(),
    )


def test_to_dict_emits_schema_pull_directive_only_when_set():
    base = _empty_result()
    assert "schema_pull_directive" not in base.to_dict()

    with_directive = replace(base, schema_pull_directive="call get_plugin_schema(name='csv', plugin_type='source')")
    payload = with_directive.to_dict()
    assert payload["schema_pull_directive"] == "call get_plugin_schema(name='csv', plugin_type='source')"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py::test_to_dict_emits_schema_pull_directive_only_when_set -v`
Expected: FAIL — `TypeError: ToolResult.__init__() got an unexpected keyword argument` is not raised (replace would fail) → actually FAIL with `dataclasses.replace` raising `TypeError: ... unexpected keyword 'schema_pull_directive'`.

- [ ] **Step 3: Add the field and to_dict emission**

In `ToolResult` (after `plugin_schemas`):

```python
    plugin_schemas: Mapping[str, Mapping[str, Any]] | None = None
    schema_pull_directive: str | None = None
```

`schema_pull_directive` is a scalar `str | None`; it needs no `freeze_fields` entry.

In `to_dict()`, alongside the existing `plugin_schemas` conditional emission, add:

```python
        if self.schema_pull_directive is not None:
            result["schema_pull_directive"] = self.schema_pull_directive
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py::test_to_dict_emits_schema_pull_directive_only_when_set -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tools/_common.py tests/unit/web/composer/test_failure_augmentation.py
git commit -m "feat(composer): add schema_pull_directive field to ToolResult"
```

---

## Task 2: Helpers — matched plugins and the pull directive

**Files:**
- Modify: `src/elspeth/web/composer/tools/_common.py`
- Test: `tests/unit/web/composer/test_failure_augmentation.py`

- [ ] **Step 1: Write the failing test**

```python
from elspeth.web.composer.tools._common import (
    build_schema_pull_directive,
    matched_invalid_option_plugins,
)
from elspeth.web.composer.state import ValidationEntry, ValidationSummary


def _summary(*messages: str) -> ValidationSummary:
    return ValidationSummary(
        is_valid=False,
        errors=tuple(ValidationEntry(component="rejected_mutation", message=m, severity="high") for m in messages),
        warnings=(),
        suggestions=(),
    )


def test_matched_invalid_option_plugins_extracts_kind_plugin_pairs():
    summary = _summary(
        "Invalid options for source 'csv': schema: ...",
        "Invalid options for transform 'web_scrape': http: ...",
        "Transform 'x' on_error 'y' references unknown sink.",  # no match
    )
    assert matched_invalid_option_plugins(summary) == {("source", "csv"), ("transform", "web_scrape")}


def test_build_schema_pull_directive_names_exact_tool_calls():
    directive = build_schema_pull_directive([("source", "csv")])
    assert "get_plugin_schema(name='csv', plugin_type='source')" in directive
    assert "get_plugin_assistance(plugin_type='source', plugin_name='csv'" in directive
    # Names every suppressed plugin
    multi = build_schema_pull_directive([("source", "csv"), ("transform", "web_scrape")])
    assert "'csv'" in multi and "'web_scrape'" in multi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py -k "matched or build_schema" -v`
Expected: FAIL with `ImportError: cannot import name 'build_schema_pull_directive'`.

- [ ] **Step 3: Implement the helpers**

In `tools/_common.py`, near `build_plugin_schemas_for_failure` (reuse the existing `_INVALID_OPTIONS_PLUGIN_RE`):

```python
def matched_invalid_option_plugins(
    validation: ValidationSummary,
) -> set[tuple[str, str]]:
    """Return every (kind, plugin) pair named by an option-shape rejection.

    Mirrors the scan in ``build_plugin_schemas_for_failure`` but returns
    only the identities, so the dispatch layer can decide per-plugin
    whether to inline the schema or emit a pull directive.
    """
    matched: set[tuple[str, str]] = set()
    for entry in validation.errors:
        for match in _INVALID_OPTIONS_PLUGIN_RE.finditer(entry.message):
            matched.add((match.group(1), match.group(2)))
    return matched


def build_schema_pull_directive(pairs: Sequence[tuple[str, str]]) -> str:
    """Build the explicit pull directive for schemas already delivered.

    D2 (spec 2026-05-31): suppressing the re-dump must NEVER drop the
    convergence lifeline — it converts a push into an explicit pull. The
    augment exists because agents almost never converged without
    context-specific guidance (see ``_dispatch._augment_with_plugin_schemas``
    docstring, composer session 47cfbb5e). Do not weaken this text to a
    bare error without re-reading that rationale.
    """
    lines = [
        "You already received the full schema for "
        f"{kind} '{plugin}' earlier this session. The validation error above "
        "is actionable as-is. For the full schema again call "
        f"get_plugin_schema(name='{plugin}', plugin_type='{kind}'); for "
        "targeted repair guidance call "
        f"get_plugin_assistance(plugin_type='{kind}', plugin_name='{plugin}', "
        "issue_code=<the code from the error above>)."
        for (kind, plugin) in pairs
    ]
    return "\n".join(lines)
```

Ensure `Sequence` is imported (it is used elsewhere in the module; add to the `collections.abc` import if absent).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py -k "matched or build_schema" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tools/_common.py tests/unit/web/composer/test_failure_augmentation.py
git commit -m "feat(composer): add matched-plugin + pull-directive helpers (D2)"
```

---

## Task 3: Delivery-aware suppression in the dispatch helper

**Files:**
- Modify: `src/elspeth/web/composer/tool_batch.py` (immediately after the `get_plugin_schema` marking block, ~`:1347`)
- Test: `tests/unit/web/composer/test_failure_augmentation.py`

The logic: when a dispatched result carries `plugin_schemas`, split the matched plugins into *already-delivered* (in `_schemas_loaded_for_session`) and *new*. Strip already-delivered schemas; if any were stripped, set `schema_pull_directive`. Then mark *all* matched plugins as delivered so the next rejection is suppressed.

- [ ] **Step 1: Write the failing test** (drives the suppression behaviour through the real service tracker)

```python
import pytest

# Helper builds a ToolResult that already carries a full plugin_schemas dump,
# simulating execute_tool's output for an option-shape rejection on `csv`.
def _rejected_with_csv_schema():
    from elspeth.web.composer.state import CompositionState, ValidationEntry, ValidationSummary
    from elspeth.web.composer.tools._common import ToolResult
    summary = ValidationSummary(
        is_valid=False,
        errors=(ValidationEntry(
            component="rejected_mutation",
            message="Invalid options for source 'csv': schema: Observed schemas cannot have explicit field definitions.",
            severity="high"),),
        warnings=(), suggestions=())
    return ToolResult(
        success=False,
        updated_state=CompositionState.empty(),
        validation=summary,
        affected_nodes=(),
        plugin_schemas={"source/csv": {"name": "csv", "plugin_type": "source", "big": "x" * 5000}},
    )


def test_suppresses_repeat_dump_and_emits_directive(monkeypatch):
    from elspeth.web.composer.tool_batch import apply_delivery_aware_schema_policy

    delivered: set[tuple[str, str]] = set()

    class _SvcStub:
        def _schemas_loaded_for_session(self, sid):
            return frozenset(delivered)
        def _mark_plugin_schema_loaded(self, sid, ptype, pname):
            delivered.add((ptype, pname))

    svc = _SvcStub()

    # First rejection: csv not yet delivered -> full schema kept, no directive, csv now marked.
    r1 = apply_delivery_aware_schema_policy(_rejected_with_csv_schema(), service=svc, session_id="s1")
    assert r1.plugin_schemas == {"source/csv": {"name": "csv", "plugin_type": "source", "big": "x" * 5000}}
    assert r1.schema_pull_directive is None
    assert ("source", "csv") in delivered

    # Second rejection: csv already delivered -> schema stripped, directive set.
    r2 = apply_delivery_aware_schema_policy(_rejected_with_csv_schema(), service=svc, session_id="s1")
    assert r2.plugin_schemas is None
    assert "get_plugin_schema(name='csv', plugin_type='source')" in r2.schema_pull_directive


def test_noop_when_no_plugin_schemas():
    from elspeth.web.composer.tool_batch import apply_delivery_aware_schema_policy
    from elspeth.web.composer.state import CompositionState, ValidationSummary
    from elspeth.web.composer.tools._common import ToolResult

    class _SvcStub:
        def _schemas_loaded_for_session(self, sid): return frozenset()
        def _mark_plugin_schema_loaded(self, sid, ptype, pname): raise AssertionError("must not mark")

    result = ToolResult(success=True, updated_state=CompositionState.empty(),
                        validation=ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=()),
                        affected_nodes=())
    out = apply_delivery_aware_schema_policy(result, service=_SvcStub(), session_id="s1")
    assert out is result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py -k "suppress or noop_when_no" -v`
Expected: FAIL with `ImportError: cannot import name 'apply_delivery_aware_schema_policy'`.

- [ ] **Step 3: Implement `apply_delivery_aware_schema_policy` and wire it in**

Add the function near the top of `tool_batch.py` (module scope, so it is unit-testable without constructing the full dispatch context):

```python
def apply_delivery_aware_schema_policy(
    result: ToolResult,
    *,
    service: Any,
    session_id: str | None,
) -> ToolResult:
    """Strip already-delivered plugin schemas and substitute a pull directive.

    D1/D2 (spec 2026-05-31): the full schema dump is helpful exactly once
    per plugin per session — the cheap composer model re-processes it on
    every subsequent turn otherwise (the 40K-token / ~2-minute stall in
    session 2e6d5e3e). On a repeat option-shape rejection for a plugin
    whose schema was already delivered, we suppress the re-dump and emit an
    explicit pull directive instead. Newly-delivered schemas are recorded
    in the same session tracker as get_plugin_schema, so the NEXT rejection
    is suppressed. Never drop the lifeline — only change its form.
    """
    if result.plugin_schemas is None:
        return result
    already = service._schemas_loaded_for_session(session_id)
    suppressed: list[tuple[str, str]] = []
    kept: dict[str, Mapping[str, Any]] = {}
    for key, payload in result.plugin_schemas.items():
        kind, _, plugin = key.partition("/")
        if (kind, plugin) in already:
            suppressed.append((kind, plugin))
        else:
            kept[key] = payload
    # Mark every matched plugin (kept + suppressed) as delivered for next turn.
    for key in result.plugin_schemas:
        kind, _, plugin = key.partition("/")
        service._mark_plugin_schema_loaded(session_id, kind, plugin)
    if not suppressed:
        return result
    directive = build_schema_pull_directive(suppressed)
    return replace(
        result,
        plugin_schemas=kept or None,
        schema_pull_directive=directive,
    )
```

Imports at top of `tool_batch.py`: ensure `from dataclasses import replace`, `from collections.abc import Mapping`, `from elspeth.web.composer.tools._common import ToolResult, build_schema_pull_directive` are present (add any missing).

Wire it in at the dispatch site. Immediately after the existing `get_plugin_schema` marking block (the `if tool_name == "get_plugin_schema" and result.success:` block ending ~`:1347`), add:

```python
        # Delivery-aware schema policy: suppress repeat full-schema dumps
        # for plugins already delivered this session; substitute an explicit
        # pull directive (D1/D2, spec 2026-05-31). Applies to option-shape
        # mutation rejections whose ToolResult carries plugin_schemas.
        result = apply_delivery_aware_schema_policy(
            result, service=ctx.service, session_id=session_id
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tool_batch.py tests/unit/web/composer/test_failure_augmentation.py
git commit -m "feat(composer): suppress repeat plugin-schema dumps, emit pull directive (D1/D2)"
```

---

## Task 4: Regression test reproducing the real session shape

**Files:**
- Test: `tests/unit/web/composer/test_failure_augmentation.py`

This pins the exact failure mode from session `2e6d5e3e`: the *second* `csv` option-shape rejection in a session must not re-ship the 5 KB schema, but must carry the directive and the verbatim error.

- [ ] **Step 1: Write the test**

```python
def test_real_session_2e6d5e3e_second_csv_rejection_is_lean(monkeypatch):
    """Regression: session 2e6d5e3e stalled ~2min re-processing a re-dumped
    csv schema. The second rejection must be lean (directive, no schema)."""
    from elspeth.web.composer.tool_batch import apply_delivery_aware_schema_policy

    delivered: set[tuple[str, str]] = set()

    class _Svc:
        def _schemas_loaded_for_session(self, sid): return frozenset(delivered)
        def _mark_plugin_schema_loaded(self, sid, ptype, pname): delivered.add((ptype, pname))

    svc = _Svc()
    first = apply_delivery_aware_schema_policy(_rejected_with_csv_schema(), service=svc, session_id="2e6d5e3e")
    second = apply_delivery_aware_schema_policy(_rejected_with_csv_schema(), service=svc, session_id="2e6d5e3e")

    # First keeps the (large) schema; second strips it.
    assert first.plugin_schemas is not None
    assert second.plugin_schemas is None
    # Lifeline preserved (D2): directive names the exact tool calls.
    assert "get_plugin_schema" in second.schema_pull_directive
    assert "get_plugin_assistance" in second.schema_pull_directive
    # Verbatim error still present (unchanged).
    assert "Observed schemas cannot have explicit field definitions" in second.validation.errors[0].message
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_failure_augmentation.py::test_real_session_2e6d5e3e_second_csv_rejection_is_lean -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_failure_augmentation.py
git commit -m "test(composer): regression for session 2e6d5e3e schema re-dump stall"
```

---

## Task 5: Full verification gate

- [ ] **Step 1: Targeted suite**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -q`
Expected: PASS (no regressions in existing dispatch/augmentation tests)

- [ ] **Step 2: Type + lint (canonical local gate)**

Run:
```bash
.venv/bin/python -m mypy src/elspeth/web/composer/tool_batch.py src/elspeth/web/composer/tools/_common.py
.venv/bin/python -m ruff check src/elspeth/web/composer/
```
Expected: clean

- [ ] **Step 3: Plugin source-file-hash gate (only if a plugin file changed)**

This stage touches no plugin files, so the `source_file_hash` gate (PH3) should be untouched. Confirm:
Run: `git diff --name-only HEAD~5 -- src/elspeth/plugins/ | head`
Expected: empty. If non-empty, refresh hashes per `project_plugin_hash_gate_ci_only`.

- [ ] **Step 4: Final commit (if any gate fixups)**

```bash
git add -A && git commit -m "chore(composer): stage-1 gate fixups"
```

---

## Self-review checklist (completed by plan author)

- **Spec coverage:** Stage 1 of the spec (error-aware augmentation, D1 no-re-seed, D2 explicit pull-directive, "keep first dump full / suppress repeats") → Tasks 1-4. ✓
- **Placeholder scan:** no TBD/TODO; every code step shows real code. `issue_code=<...>` inside the directive is intentional model-facing instruction text, not a plan placeholder. ✓
- **Type consistency:** `apply_delivery_aware_schema_policy` / `build_schema_pull_directive` / `matched_invalid_option_plugins` / `schema_pull_directive` used identically across tasks; tracker tuple shape `(plugin_type, plugin_name)` = `(kind, plugin)` consistent with `_common.py` key format `"<kind>/<plugin>"`. ✓
- **Open confirm for implementer:** verify the exact line of the `result = handler(...)`/`execute_tool` assignment that feeds `tool_batch.py:1342` so the new policy call sits on the live `result` variable in that scope (the marking block at `:1342` confirms `result`, `arguments`, `session_id`, `ctx.service` are in scope there).

## Risk

- **R1 (convergence regression):** mitigated by D2 directive + the code comment citing `47cfbb5e` + Task 4 asserting the directive is present.
- Tracker is in-memory per service instance and `session_id`-keyed; `None` session (unsaved) → no suppression (first-dump behaviour always), which is correct.
