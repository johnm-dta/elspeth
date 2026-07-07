# CLI/TUI Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modernize the 0.7-era CLI and auditor TUI so `elspeth explain` is a real interactive lineage explorer, plugin catalog CLI output is useful for humans and automation, and docs/tests match the implemented behavior.

**Architecture:** Keep the first upgrade vertical and testable: build a real Textual interaction shell, then replace the flat lineage presenter with a graph-backed TUI view model fed by existing Landscape repositories and `core.landscape.lineage.explain()`. For CLI work, avoid another large `cli.py` branch by moving plugin catalog behavior behind a small CLI service/formatter module that can later host more command extraction.

**Tech Stack:** Python 3.13, Typer, Textual 7.5, Pydantic response models, Landscape SQLite repositories, pytest, ruff, mypy, Wardline.

**Prerequisites:**
- Worktree: `/home/john/elspeth/.worktrees/cli-tui-upgrade`.
- Branch: `feature/elspeth-82c3914f95-cli-tui-upgrade`.
- Filigree feature: `elspeth-82c3914f95` in `building`.
- Worktree-local environment: `uv sync --extra dev`.

---

## Current Baseline

- `src/elspeth/tui/explain_app.py` now mounts a Textual `Tree` for loaded lineage and updates the detail panel on selection.
- Regression test added in `tests/unit/tui/test_explain_app.py`.
- Verified baseline:
  - `uv run pytest tests/unit/tui tests/unit/cli/test_explain_tui.py tests/unit/cli/test_explain_command.py -q` -> 66 passed.
  - `uv run ruff check src/elspeth/tui/explain_app.py tests/unit/tui/test_explain_app.py` -> passed.
  - `uv run mypy src/elspeth/tui/explain_app.py tests/unit/tui/test_explain_app.py` -> passed.
  - `uv run elspeth health --json` -> healthy.
  - `wardline scan . --fail-on ERROR` -> exit 0, but taint gate reported inert because zero trust boundaries were recognized.

---

## Task 1: Initial Interactive TUI Shell Baseline

**Files:**
- Modify: `src/elspeth/tui/explain_app.py`
- Test: `tests/unit/tui/test_explain_app.py`

**Status:** Implemented in the current worktree, not committed. This is a partial shell baseline, not an independently shippable checkpoint; Task 2 must land before the TUI shell is commit-ready because refresh state changes can still leave the mounted widget type stale.

**Definition of Done:**
- [x] Loaded TUI uses `textual.widgets.Tree`, not static lineage text.
- [x] Selecting a tree node calls `ExplainScreen.on_tree_select()`.
- [x] Mounted detail panel updates after selection.
- [x] Refresh updates mounted tree/detail widgets.
- [x] Help text names actual navigation behavior.
- [x] Focused TUI tests, ruff, and targeted mypy pass.

---

## Task 2: Stabilize Selection Payloads and Refresh State Transitions

**Files:**
- Modify: `src/elspeth/tui/types.py`
- Modify: `src/elspeth/tui/explain_app.py`
- Modify: `src/elspeth/tui/widgets/lineage_tree.py`
- Modify: `src/elspeth/tui/screens/explain_screen.py`
- Test: `tests/unit/tui/test_explain_app.py`
- Test: `tests/unit/cli/test_explain_tui.py`

**Why this comes before the graph rewrite:** The first shell slice stores only `node_id` / `node_type` in the Textual tree payload. Token leaves currently carry `node_id=token_id` in the legacy `LineageTree`, so the app can still treat a token selection as a node lookup. Fix the selection model before adding more selectable object kinds.

**Step 1: Write failing tests**

Add tests that prove:
- Real keyboard navigation focuses or clicks `#lineage-tree`, then `pilot.press("down", "enter")` selects a node and updates detail, not only direct `Tree.NodeSelected` message posting.
- Selecting a token leaf does not call `data_flow.get_node(token_id, run_id)`.
- Selecting the run/root row and empty/status rows does not call a node lookup.
- Refresh from loaded -> failed clears or replaces stale lineage.
- Refresh from failed -> loaded mounts/selects the real tree.
- Empty/no-node loaded state shows a nonselectable empty message, not a synthetic `Source: (unknown)`, and clears detail.

**Expected RED:** direct message-posting test passes, but keyboard path, token selection, and loaded/failed state replacement are missing or weak.

**Step 2: Add a discriminated selection payload**

Add a payload type such as:

```python
class TreeSelection(TypedDict, total=False):
    kind: Required[Literal["run", "node", "token", "edge", "outcome"]]
    run_id: Required[str]
    node_id: str
    node_type: str
    token_id: str
    row_id: str
    sink: str
    state_id: str
```

Rules:
- Node rows use `kind="node"` with `node_id`.
- Token rows use `kind="token"` with `token_id`, `row_id`, and optional `sink`.
- Run/root rows use `kind="run"`.
- Status/empty/error rows use no selection payload.
- `ExplainScreen.on_tree_select()` should accept the selection object or a narrow wrapper method should translate it before querying.
- `TreeNodeDict` should carry `selection: TreeSelection | None`, and the Textual `Tree.data` value should preserve that exact object rather than reconstructing identity from `node_id` / `node_type`.

**Step 3: Mount a stable state container**

Avoid changing the type of `#lineage-tree` between `Static` and `Tree` across refreshes. Use a stable content container or consistently remount/replace children so `action_refresh()` can handle loaded/failed/uninitialized transitions without stale displays.

**Definition of Done:**
- [x] Keyboard path is covered by tests.
- [x] Token/run/root selection cannot be mistaken for a node lookup.
- [x] Refresh handles loaded/failed/uninitialized transitions.
- [x] Zero recorded nodes render an honest empty/status row, not a fake unknown source.
- [x] Existing direct selection regression still passes.

**Verify:**

```bash
uv run pytest tests/unit/tui/test_explain_app.py tests/unit/cli/test_explain_tui.py -q
```

---

## Task 3: Replace Flat Lineage Data With a Graph-Backed View Model

**Files:**
- Create: `src/elspeth/tui/lineage_view.py`
- Modify: `src/elspeth/tui/types.py`
- Modify: `src/elspeth/tui/screens/explain_screen.py`
- Modify: `src/elspeth/tui/widgets/lineage_tree.py`
- Test: `tests/unit/tui/test_lineage_tree.py`
- Test: `tests/unit/cli/test_explain_tui.py`

**Step 1: Write failing tests**

Add tests that prove the current bug:

```python
def test_lineage_tree_renders_multiple_sources_without_dropping_second_source() -> None:
    view = build_lineage_view_model(
        run_id="run-1",
        nodes=[
            node("src-a", "csv", "source"),
            node("src-b", "json", "source"),
            node("merge", "coalesce", "coalesce"),
            node("sink", "json", "sink"),
        ],
        edges=[
            edge("src-a", "merge", "a"),
            edge("src-b", "merge", "b"),
            edge("merge", "sink", "default"),
        ],
        focused_lineage=None,
    )

    labels = [item.label for item in view.items]

    assert "Source: csv" in labels
    assert "Source: json" in labels
    assert "Coalesce: coalesce" in labels
    assert any("a" in label for label in labels)
    assert any("b" in label for label in labels)
```

Also add a fork/coalesce test that would fail against the old linear chain.

**Expected RED:** assertions fail because `ExplainScreen._load_pipeline_structure()` stores only `source_nodes[0]` and `LineageTree` renders transforms as a single linear chain.

**Step 2: Implement graph view model**

Create a small presenter module with frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class TuiLineageItem:
    label: str
    selection: TreeSelection | None
    depth: int
    has_children: bool
    expanded: bool = True
    node_id: str | None = None
    node_type: str = ""
    token_id: str | None = None
    edge_label: str | None = None
```

Build it from:
- `factory.data_flow.get_nodes(run_id)`
- `factory.data_flow.get_edges(run_id)`
- optional focused `LineageResult` from `core.landscape.lineage.explain()`
- token family reads where needed: `query.get_all_token_parents_for_run()`, `query.get_node_states_for_tokens()`, and token outcomes.

Rules:
- Root is always `Run: <run_id>`.
- Show every source under the root.
- Traverse graph edges by `from_node_id -> to_node_id`, ordered deterministically by edge label then destination.
- Render branch labels when edge labels exist.
- Avoid infinite recursion by showing a repeated-node marker when a DAG join is encountered through a second path.
- If a focused token/row exists, annotate nodes in the token path and attach terminal token/outcome rows at the terminal node.
- For coalesce/fork ancestry, start with immediate parent/child token display from existing lineage results. Add lazy expansion or bounded token-family preload only after the base graph renderer is correct.
- Every selectable view item must carry the Task 2 `TreeSelection` object. Graph, repeated-node, token, edge, and outcome rows must not fall back to ad hoc `node_id` dispatch.

**Step 3: Wire view model into `ExplainScreen` and `LineageTree`**

`ExplainScreen._load_pipeline_structure()` should stop constructing the old singular-source `LineageData` as the primary model. It should construct the graph view once, then `LineageTree.get_tree_nodes()` should return `TreeNodeDict` derived from the view model.

**Definition of Done:**
- [x] Multiple source nodes are preserved.
- [x] Fork/coalesce/branch labels are visible enough for auditors to understand the topology.
- [x] Unsorted edge input renders in stable order.
- [x] Diamond DAG joins terminate and render with a clear repeated-node marker or another explicitly tested non-duplicating representation.
- [x] Old linear-pipeline tests still pass.
- [x] New DAG tests fail before implementation and pass after implementation.

**Verify:**

```bash
uv run pytest tests/unit/tui/test_lineage_tree.py tests/unit/cli/test_explain_tui.py -q
```

---

## Task 4: Scope Node Details to the Selected Token/Row and Avoid Per-Selection Full-Run Scans

**Files:**
- Modify: `src/elspeth/tui/screens/explain_screen.py`
- Optional modify: `src/elspeth/core/landscape/query_repository.py`
- Test: `tests/unit/cli/test_explain_tui.py`

**Step 1: Write failing tests**

Add tests proving:
- With `token_id` or `row_id` selected, selecting a node shows that focused token's `NodeState`, not the latest state for another token.
- Repeated selection does not call `get_all_node_states_for_run()` once per selection.

**Expected RED:** current `_load_node_state()` calls `factory.query.get_all_node_states_for_run(run_id)` inside every selection and picks `max(...)` across all tokens for a node.

**Step 2: Implement scoped detail cache**

Prefer a screen-local cache to avoid widening repository API unless tests show a query helper is cleaner:

Task 3 defines the final loaded-state shape. Task 4 should extend that same state with detail caches instead of reintroducing `LineageData` as the primary model:

```python
@dataclass(frozen=True, slots=True)
class LoadedState:
    db: LandscapeDB
    run_id: str
    lineage_view: TuiLineageView
    tree: LineageTree
    focused_state_by_node_id: Mapping[str, NodeState]
    latest_state_by_node_id: Mapping[str, NodeState]
```

Rules:
- If a focused `LineageResult` exists, populate `focused_state_by_node_id` from `result.node_states`.
- Otherwise populate `latest_state_by_node_id` once from `get_all_node_states_for_run()`.
- `on_tree_select()` first uses the focused map, then the latest map.
- Detail rendering must have explicit variants for run summary, node state, token/row outcome, edge/branch, and empty/error states.
- Token tree leaves show a token/row summary or outcome detail, not a misleading "node not found" blank and not `get_node(token_id, run_id)`.
- The detail pane should be scrollable/focusable before long JSON/error context is expanded in the UI.

**Definition of Done:**
- [x] Focused token/row detail uses the focused token state.
- [x] Selection no longer performs full-run state scans.
- [x] Token leaf selection does not produce a misleading "node not found" blank.
- [x] Run, edge, outcome, empty, and error selections have explicit detail behavior.

**Verify:**

```bash
uv run pytest tests/unit/cli/test_explain_tui.py tests/unit/tui/test_explain_app.py -q
```

---

## Task 5: Add CLI Plugin Catalog JSON and Inspect Surfaces

**Files:**
- Create: `src/elspeth/cli_plugins.py`
- Modify: `src/elspeth/cli.py`
- Test: `tests/unit/cli/test_plugins_command.py`
- Test: `tests/integration/cli/test_cli.py`

**Step 1: Write failing tests**

Add tests for:

```bash
elspeth plugins list --format json
elspeth plugins list --type source --format json
elspeth plugins inspect source csv --format json
elspeth plugins inspect transform passthrough
```

Expected JSON shape for list:

```json
{
  "source": [{"name": "csv", "description": "...", "plugin_type": "source"}],
  "transform": [],
  "sink": []
}
```

Expected JSON shape for inspect should reuse catalog/schema data where feasible:

```json
{
  "name": "csv",
  "plugin_type": "source",
  "description": "...",
  "config_fields": [],
  "json_schema": {},
  "knob_schema": {}
}
```

This inspect payload is a CLI-specific JSON DTO: join `PluginSchemaInfo` with the matching `PluginSummary.config_fields`, then serialize to plain JSON-ready dictionaries using `model_dump(mode="json")`. Do not return Pydantic model instances directly to `json.dumps()`.

**Expected RED:** current `plugins list` has no `--format`, and `plugins inspect` does not exist.

**Step 2: Extract plugin catalog helpers**

Create `src/elspeth/cli_plugins.py` with functions that are independent of Typer:

```python
def build_catalog_service() -> CatalogServiceImpl:
    return CatalogServiceImpl(get_shared_plugin_manager())

def list_plugins_payload(plugin_type: PluginKind | None) -> dict[str, list[dict[str, object]]]:
    ...

def inspect_plugin_payload(plugin_type: PluginKind, name: str) -> dict[str, object]:
    ...
```

`src/elspeth/cli.py` should keep Typer command definitions but delegate business logic to this module. Retire the duplicate CLI-only catalog registry as the behavior moves behind the `CatalogServiceImpl` adapter; replace old helper-contract tests with adapter/formatter tests so there is one catalog truth.

**Step 3: Preserve text behavior**

Existing `plugins list` text output should stay compatible so current tests and docs keep working. `--format json` must write only JSON to stdout.

Do not refactor or change `run`, `resume`, or `join` streaming command internals in this slice. Those commands have JSON/stdout purity and exit-code contracts that need their own focused upgrade pass.

JSON purity tests must parse `json.loads(result.stdout)`, assert stdout starts with `{`, assert no human headings or banners precede the JSON object, and assert successful commands write nothing to stderr. Add one subprocess smoke with `capture_output=True` to verify real stdout/stderr separation outside Typer's in-process runner.

**Definition of Done:**
- [x] Existing text tests still pass.
- [x] JSON list output is valid JSON with no banner contamination.
- [x] Inspect command exists for text and JSON.
- [x] Invalid plugin type/name exits non-zero with a useful message.
- [x] JSON success paths are parseable from stdout and do not write success noise to stderr.

**Verify:**

```bash
uv run pytest tests/unit/cli/test_plugins_command.py tests/integration/cli/test_cli.py::TestCLIIntegration::test_plugins_list_shows_all_types -q
uv run elspeth plugins list --format json
uv run elspeth plugins inspect source csv --format json
```

---

## Task 6: Document Structured Output and TUI Reality

**Files:**
- Modify: `README.md`
- Modify: `docs/guides/user-manual.md`
- Modify: `docs/guides/your-first-pipeline.md`
- Modify: `docs/guides/docker.md`
- Modify: `docs/product/roadmap.md`

**Step 1: Update docs after implementation**

Do not document promised behavior before it exists. Once Tasks 2-4 pass:
- Update `explain` TUI docs to describe the actual interactive tree/detail workflow.
- Replace stale language that says the old TUI already exposes every routing decision.
- Add `plugins list --format json` and `plugins inspect` examples.
- Fix stale roadmap tracker `elspeth-5bb9dfc9fa` to the live feature `elspeth-82c3914f95` or remove the stale tracker reference.
- Align README explain/plugin examples with the implemented TUI and plugin CLI behavior, or explicitly remove overclaims.

**Definition of Done:**
- [ ] Docs match current CLI help/output.
- [ ] Docker docs do not imply TUI works in non-interactive containers unless `-it` is used.
- [ ] Roadmap references a live tracker ID.

**Verify:**

```bash
uv run elspeth explain --help
uv run elspeth plugins --help
uv run elspeth plugins list --help
uv run elspeth plugins inspect --help
uv run pytest tests/unit/docs -q
! rg "elspeth-5bb9dfc9fa" docs/product/roadmap.md README.md docs/guides
```

---

## Task 7: Specialist Review and Cleanup

**Required review lenses:**
- UX specialist: keyboard flow, affordance clarity, help text, dense terminal layout, no misleading capability claims.
- CLI/API reviewer: stdout/stderr JSON purity, command help, backward compatibility.
- Architecture reviewer: blast radius, module boundaries, no avoidable `cli.py` growth.
- Quality/security reviewer: tests, audit-data display risks, Wardline output, docs truthfulness.

**Review process:**
1. Spawn read-only plan-review agents before Task 2 resumes and incorporate accepted plan fixes. Reviewers must inspect `/home/john/elspeth/.worktrees/cli-tui-upgrade` live; Loomweave is not the symbol authority unless it is refreshed against this exact checkout.
2. Spawn read-only implementation-review agents after implementation.
3. Fix every accepted finding.
4. Re-run focused and broad gates.

**Final verification commands:**

```bash
uv run pytest tests/unit/tui tests/unit/cli/test_explain_tui.py tests/unit/cli/test_explain_command.py tests/unit/cli/test_plugins_command.py -q
uv run pytest tests/integration/cli/test_cli.py::TestCLIIntegration::test_plugins_list_shows_all_types -q
uv run ruff check src/elspeth/tui src/elspeth/cli.py src/elspeth/cli_plugins.py tests/unit/tui tests/unit/cli/test_plugins_command.py
uv run mypy src/elspeth/tui src/elspeth/cli.py src/elspeth/cli_plugins.py tests/unit/tui/test_explain_app.py tests/unit/cli/test_plugins_command.py
uv run elspeth health --json
wardline scan . --fail-on ERROR
```

If Wardline exits 0 but reports an inert gate, record that exact warning in the closeout. Do not claim the taint gate proved trust-boundary cleanliness when it recognized zero trust boundaries.

**CI parity gate:**

```bash
uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
uv run mypy src/ elspeth-lints/src/
uv run python scripts/check_contracts.py
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.options_metadata --root .
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.component_type,plugin_contract.plugin_hashes --root src/elspeth
PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
PYTHONPATH=elspeth-lints/src uv run python scripts/cicd/parity_harness.py --manifest config/cicd/lint_migration_status.yaml --root .
```

Keep `wardline scan . --fail-on ERROR`, but pair it with the trust-boundary lint commands above while Wardline reports zero recognized trust boundaries. After Wardline, check `git status --short` for generated `findings.jsonl` artifacts and do not commit scan output unless it is intentionally part of a separate evidence artifact.

---

## Task 8: Commit the Completed Worktree

**Files:**
- All files changed by the upgrade after review cleanup.
- Include `docs/plans/2026-07-07-cli-tui-upgrade.md`; this implementation plan is intentionally part of the upgrade branch history.

**Commit command:**

```bash
git add -A
git commit -m "feat(cli,tui): modernize explain TUI and plugin catalog CLI"
```

**Definition of Done:**
- [ ] All review findings accepted by the main agent are fixed.
- [ ] Final verification commands pass or any residual warnings are explicitly recorded.
- [ ] Checkpoint commits exist for TUI shell stabilization, graph/detail behavior, plugin catalog CLI, and final docs/review cleanup unless a later finding requires squashing before handoff.
- [ ] Worktree is committed on `feature/elspeth-82c3914f95-cli-tui-upgrade`.
- [ ] Main release branch is not merged or modified beyond Filigree metadata.
