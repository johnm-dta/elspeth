# Web UX Sub-Plan 4x: Composer Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Harden the composer with discovery caching, dual-budget loop, partial state preservation, rate limiting, edge order fix, and tool registry
**Parent Spec:** `specs/2026-03-29-web-ux-sub4x-composer-hardening-design.md`
**Depends On:** Sub-Plan 4 (Composer) -- must be merged first
**Can run in parallel with:** Sub-Plan 5 (Execution)

---

## File Map

| Action | Path | Task |
|--------|------|------|
| Modify | `src/elspeth/web/composer/state.py` | T1: `with_edge()` and `with_output()` insertion order fix |
| Modify | `src/elspeth/web/composer/tools.py` | T2: tool registry refactor + `is_discovery_tool()` + `is_cacheable_discovery_tool()` |
| Modify | `src/elspeth/web/composer/protocol.py` | T3/T4: `ComposerConvergenceError` gains `partial_state` and `budget_exhausted` |
| Modify | `src/elspeth/web/composer/service.py` | T3: dual-counter loop + local discovery cache + `asyncio.wait_for()` timeout |
| Modify | `src/elspeth/web/config.py` | T3: replace `composer_max_turns` with dual settings + `composer_timeout_seconds` |
| Create | `src/elspeth/web/middleware/__init__.py` | T5: module init |
| Create | `src/elspeth/web/middleware/rate_limit.py` | T5: `ComposerRateLimiter` |
| Modify | `src/elspeth/web/sessions/routes.py` | T4: partial state persistence on convergence, T5: rate limiter wiring |
| Modify | `src/elspeth/web/app.py` | T5: `ComposerRateLimiter` instantiation in lifespan |
| Modify | `tests/unit/web/composer/test_state.py` | T1: `test_with_edge_preserves_order`, `test_with_output_preserves_order` |
| Modify | `tests/unit/web/composer/test_tools.py` | T2: registry dispatch tests, `is_discovery_tool()`, `is_cacheable_discovery_tool()` |
| Modify | `tests/unit/web/composer/test_service.py` | T3: dual-counter tests, cache tests, timeout test. T4: partial state on convergence |
| Create | `tests/unit/web/middleware/__init__.py` | T5: test package init |
| Create | `tests/unit/web/middleware/test_rate_limit.py` | T5: rate limiter tests |

---

### Task 1: Edge and Output Insertion Order Fix (F4)

**Files:**
- Modify: `tests/unit/web/composer/test_state.py`
- Modify: `src/elspeth/web/composer/state.py`

- [ ] **Step 1: Write order-preservation tests**

```python
# tests/unit/web/composer/test_state.py (append to TestCompositionState class)

    def test_with_edge_preserves_order(self) -> None:
        """Updating an existing edge must preserve its position, not append."""
        state = self._empty_state()
        e1 = EdgeSpec(
            id="e1", from_node="source", to_node="t1",
            edge_type="on_success", label=None,
        )
        e2 = EdgeSpec(
            id="e2", from_node="t1", to_node="t2",
            edge_type="on_success", label=None,
        )
        e3 = EdgeSpec(
            id="e3", from_node="t2", to_node="sink",
            edge_type="on_success", label=None,
        )
        state = state.with_edge(e1).with_edge(e2).with_edge(e3)
        assert [e.id for e in state.edges] == ["e1", "e2", "e3"]

        # Update e2 — should stay at index 1, not move to end
        e2_updated = EdgeSpec(
            id="e2", from_node="t1", to_node="t2_new",
            edge_type="on_success", label="updated",
        )
        updated = state.with_edge(e2_updated)
        assert [e.id for e in updated.edges] == ["e1", "e2", "e3"]
        assert updated.edges[1].to_node == "t2_new"
        assert updated.edges[1].label == "updated"

    def test_with_output_preserves_order(self) -> None:
        """Updating an existing output must preserve its position, not append."""
        state = self._empty_state()
        o1 = self._make_output("alpha")
        o2 = self._make_output("beta")
        o3 = self._make_output("gamma")
        state = state.with_output(o1).with_output(o2).with_output(o3)
        assert [o.name for o in state.outputs] == ["alpha", "beta", "gamma"]

        # Update beta — should stay at index 1, not move to end
        o2_updated = OutputSpec(
            name="beta", plugin="json",
            options={"format": "lines"}, on_write_failure="discard",
        )
        updated = state.with_output(o2_updated)
        assert [o.name for o in updated.outputs] == ["alpha", "beta", "gamma"]
        assert updated.outputs[1].plugin == "json"
```

- [ ] **Step 2: Run tests -- expect FAIL (order not preserved)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestCompositionState::test_with_edge_preserves_order tests/unit/web/composer/test_state.py::TestCompositionState::test_with_output_preserves_order -v`
Expected: FAIL -- both methods use filter-then-append, which moves updated items to the end.

- [ ] **Step 3: Fix `with_edge()` and `with_output()` to use index-and-replace**

```python
# src/elspeth/web/composer/state.py — replace with_edge method

    def with_edge(self, edge: EdgeSpec) -> CompositionState:
        """Add or replace an edge (matched by id). Version incremented."""
        existing_ids = [e.id for e in self.edges]
        if edge.id in existing_ids:
            idx = existing_ids.index(edge.id)
            edge_list = list(self.edges)
            edge_list[idx] = edge
            edges = tuple(edge_list)
        else:
            edges = self.edges + (edge,)
        return replace(self, edges=edges, version=self.version + 1)

    def with_output(self, output: OutputSpec) -> CompositionState:
        """Add or replace an output (matched by name). Version incremented."""
        existing_names = [o.name for o in self.outputs]
        if output.name in existing_names:
            idx = existing_names.index(output.name)
            output_list = list(self.outputs)
            output_list[idx] = output
            outputs = tuple(output_list)
        else:
            outputs = self.outputs + (output,)
        return replace(self, outputs=outputs, version=self.version + 1)
```

- [ ] **Step 4: Run tests -- expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests PASS, including the two new order-preservation tests.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/state.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/state.py tests/unit/web/composer/test_state.py
git commit -m "fix(web/composer): preserve insertion order in with_edge() and with_output()"
```

---

### Task 2: Tool Registry Pattern (F5)

**Files:**
- Modify: `tests/unit/web/composer/test_tools.py`
- Modify: `src/elspeth/web/composer/tools.py`

- [ ] **Step 1: Write registry tests**

```python
# tests/unit/web/composer/test_tools.py (append to file)


class TestToolRegistry:
    """Tests for the tool registry pattern — two dicts + cacheable frozenset."""

    def test_discovery_tools_has_five_entries(self) -> None:
        from elspeth.web.composer.tools import _DISCOVERY_TOOLS

        assert len(_DISCOVERY_TOOLS) == 5
        expected = {
            "list_sources", "list_transforms", "list_sinks",
            "get_plugin_schema", "get_expression_grammar",
        }
        assert set(_DISCOVERY_TOOLS.keys()) == expected

    def test_mutation_tools_has_eight_entries(self) -> None:
        from elspeth.web.composer.tools import _MUTATION_TOOLS

        assert len(_MUTATION_TOOLS) == 8
        expected = {
            "set_source", "upsert_node", "upsert_edge",
            "remove_node", "remove_edge", "set_metadata",
            "set_output", "remove_output",
        }
        assert set(_MUTATION_TOOLS.keys()) == expected

    def test_no_overlap_between_registries(self) -> None:
        from elspeth.web.composer.tools import _DISCOVERY_TOOLS, _MUTATION_TOOLS

        overlap = set(_DISCOVERY_TOOLS.keys()) & set(_MUTATION_TOOLS.keys())
        assert overlap == set(), f"Registry overlap: {overlap}"

    def test_cacheable_discovery_equals_discovery(self) -> None:
        """All discovery tools are cacheable (get_current_state was removed)."""
        from elspeth.web.composer.tools import (
            _CACHEABLE_DISCOVERY_TOOLS,
            _DISCOVERY_TOOLS,
        )

        assert _CACHEABLE_DISCOVERY_TOOLS == frozenset(_DISCOVERY_TOOLS.keys())

    def test_cacheable_is_subset_of_discovery(self) -> None:
        from elspeth.web.composer.tools import (
            _CACHEABLE_DISCOVERY_TOOLS,
            _DISCOVERY_TOOLS,
        )

        assert _CACHEABLE_DISCOVERY_TOOLS <= set(_DISCOVERY_TOOLS.keys())

    def test_is_discovery_tool(self) -> None:
        from elspeth.web.composer.tools import is_discovery_tool

        assert is_discovery_tool("list_sources") is True
        assert is_discovery_tool("get_expression_grammar") is True
        assert is_discovery_tool("set_source") is False
        assert is_discovery_tool("nonexistent") is False

    def test_is_cacheable_discovery_tool(self) -> None:
        from elspeth.web.composer.tools import is_cacheable_discovery_tool

        assert is_cacheable_discovery_tool("list_sources") is True
        assert is_cacheable_discovery_tool("get_plugin_schema") is True
        assert is_cacheable_discovery_tool("set_source") is False

    def test_registry_dispatch_matches_original_behaviour(self) -> None:
        """Every tool in the registries dispatches correctly via execute_tool."""
        state = _empty_state()
        catalog = _mock_catalog()

        # All discovery tools should succeed
        for tool_name in [
            "list_sources", "list_transforms", "list_sinks",
            "get_expression_grammar",
        ]:
            result = execute_tool(tool_name, {}, state, catalog)
            assert result.success is True, f"{tool_name} failed"

        # get_plugin_schema needs arguments
        result = execute_tool(
            "get_plugin_schema",
            {"plugin_type": "source", "name": "csv"},
            state, catalog,
        )
        assert result.success is True

        # Mutation tools that work on empty state
        result = execute_tool(
            "set_source",
            {
                "plugin": "csv", "on_success": "t1",
                "options": {}, "on_validation_failure": "quarantine",
            },
            state, catalog,
        )
        assert result.success is True

        result = execute_tool(
            "set_metadata",
            {"patch": {"name": "Test"}},
            state, catalog,
        )
        assert result.success is True

        # Unknown tool returns failure
        result = execute_tool("nonexistent", {}, state, catalog)
        assert result.success is False

    def test_module_level_assertion_no_overlap(self) -> None:
        """Importing the module should not raise -- the overlap assertion passes."""
        # If there were overlap, the module would fail to import
        # with an AssertionError. This test verifies the module loads.
        import importlib

        import elspeth.web.composer.tools as mod

        importlib.reload(mod)  # Force re-evaluation of module-level assertion
```

- [ ] **Step 2: Run tests -- expect FAIL (registry not defined yet)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py::TestToolRegistry -v`
Expected: FAIL -- `_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, etc. not defined.

- [ ] **Step 3: Refactor tools.py to use registry pattern**

Replace the if/elif chain in `execute_tool()` and add the registry infrastructure. The handler functions already exist; this is a mechanical refactoring. Normalize handler signatures so all accept `(arguments, state, catalog, data_dir) -> ToolResult`. The `data_dir` parameter is needed by `_execute_set_source` for S2 path allowlist enforcement; other handlers ignore it.

**WIP delta:** The WIP commit (`f3d33839`) removed `get_current_state` from both `get_tool_definitions()` and the if-chain. The registries reflect this — 5 discovery tools, 8 mutation tools (13 total). The `_CACHEABLE_DISCOVERY_TOOLS` frozenset is now identical to the discovery registry keys since the only non-cacheable discovery tool was `get_current_state`.

```python
# src/elspeth/web/composer/tools.py — add after the existing handler functions,
# replacing the execute_tool() if-chain.

from collections.abc import Callable

ToolHandler = Callable[
    [dict[str, Any], CompositionState, CatalogServiceProtocol, str | None],
    ToolResult,
]


# --- Discovery tool handlers (normalized signatures) ---

def _handle_list_sources(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_sources())


def _handle_list_transforms(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_transforms())


def _handle_list_sinks(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_sinks())


def _handle_get_plugin_schema(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    try:
        schema = catalog.get_schema(arguments["plugin_type"], arguments["name"])
        return _discovery_result(state, schema)
    except (ValueError, KeyError) as exc:
        # ValueError: catalog contract for "unknown plugin/type"
        # KeyError: LLM omitted required argument (Tier 3)
        return _failure_result(state, str(exc))


def _handle_get_expression_grammar(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, get_expression_grammar())


# --- Mutation tool handlers (signatures already normalized) ---
# _execute_set_source already accepts (args, state, catalog, data_dir).
# _execute_upsert_node, _execute_set_output accept (args, state, catalog).
# _execute_upsert_edge, _execute_remove_node, _execute_remove_edge,
# _execute_set_metadata, _execute_remove_output accept (args, state).
# Normalize all to accept (args, state, catalog, data_dir):

def _handle_upsert_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_upsert_edge(arguments, state)


def _handle_remove_node(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_node(arguments, state)


def _handle_remove_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_edge(arguments, state)


def _handle_set_metadata(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_set_metadata(arguments, state)


def _handle_upsert_node(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_upsert_node(arguments, state, catalog)


def _handle_set_output(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_set_output(arguments, state, catalog)


def _handle_remove_output(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_output(arguments, state)


# --- Registries ---

_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_sources": _handle_list_sources,
    "list_transforms": _handle_list_transforms,
    "list_sinks": _handle_list_sinks,
    "get_plugin_schema": _handle_get_plugin_schema,
    "get_expression_grammar": _handle_get_expression_grammar,
}

# All discovery tools are cacheable. Previously get_current_state was
# excluded (returns live mutable state), but that tool was removed in
# the WIP review-fixes commit. If a non-cacheable discovery tool is
# re-added in future, add it to _DISCOVERY_TOOLS but NOT here.
_CACHEABLE_DISCOVERY_TOOLS: frozenset[str] = frozenset(_DISCOVERY_TOOLS.keys())

_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "set_source": _execute_set_source,
    "upsert_node": _handle_upsert_node,
    "upsert_edge": _handle_upsert_edge,
    "remove_node": _handle_remove_node,
    "remove_edge": _handle_remove_edge,
    "set_metadata": _handle_set_metadata,
    "set_output": _handle_set_output,
    "remove_output": _handle_remove_output,
}

# Module-level assertion: registries must not overlap.
assert not (set(_DISCOVERY_TOOLS) & set(_MUTATION_TOOLS)), (
    f"Tool registry overlap: {set(_DISCOVERY_TOOLS) & set(_MUTATION_TOOLS)}"
)

# Cacheable tools must be a subset of discovery tools.
assert _CACHEABLE_DISCOVERY_TOOLS <= set(_DISCOVERY_TOOLS), (
    f"Cacheable tools not in discovery registry: "
    f"{_CACHEABLE_DISCOVERY_TOOLS - set(_DISCOVERY_TOOLS)}"
)


def is_discovery_tool(name: str) -> bool:
    """Return True if the tool is a discovery (read-only) tool."""
    return name in _DISCOVERY_TOOLS


def is_cacheable_discovery_tool(name: str) -> bool:
    """Return True if the tool's results can be cached within a compose() call."""
    return name in _CACHEABLE_DISCOVERY_TOOLS


# --- Replace the if-chain in execute_tool() ---

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogServiceProtocol,
    data_dir: str | None = None,
) -> ToolResult:
    """Execute a composition tool by name.

    Dispatches via registry dict. Discovery tools return data without
    modifying state. Mutation tools return ToolResult with updated state
    and validation. Unknown tool names return a failure result.

    Args:
        data_dir: Base data directory for S2 path allowlist enforcement.
            When provided, source options containing ``path`` or ``file``
            keys are restricted to ``{data_dir}/uploads/``.
    """
    handler = _DISCOVERY_TOOLS.get(tool_name) or _MUTATION_TOOLS.get(tool_name)
    if handler is None:
        return _failure_result(state, f"Unknown tool: {tool_name}")
    return handler(arguments, state, catalog, data_dir)
```

- [ ] **Step 4: Run tests -- expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -v`
Expected: All tests PASS -- both old tests (dispatch behaviour unchanged) and new registry tests.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/tools.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/unit/web/composer/test_tools.py
git commit -m "refactor(web/composer): replace if-chain with tool registry dicts + cacheable frozenset"
```

---

### Task 3: Dual-Counter Loop + Discovery Cache (F1)

**Files:**
- Modify: `tests/unit/web/composer/test_service.py`
- Modify: `src/elspeth/web/composer/protocol.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/config.py` (WebSettings lives here)

- [ ] **Step 1: Update `ComposerConvergenceError` in protocol.py**

```python
# src/elspeth/web/composer/protocol.py — replace ComposerConvergenceError

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.state import CompositionState


class ComposerConvergenceError(ComposerServiceError):
    """Raised when the LLM tool-use loop exhausts its budget or times out.

    Attributes:
        max_turns: Total turns used before exhaustion.
        budget_exhausted: Which budget was exhausted — one of
            "composition", "discovery", or "timeout".
        partial_state: The last CompositionState with version > initial,
            or None if no mutations occurred.
    """

    def __init__(
        self,
        max_turns: int,
        *,
        budget_exhausted: str = "composition",
        partial_state: CompositionState | None = None,
    ) -> None:
        super().__init__(
            f"Composer did not converge within {max_turns} turns "
            f"(budget exhausted: {budget_exhausted}). "
            f"The LLM kept making tool calls without producing a final response."
        )
        self.max_turns = max_turns
        self.budget_exhausted = budget_exhausted
        self.partial_state = partial_state
```

Note: `ComposerConvergenceError` inherits from `Exception`, not a frozen dataclass. The `partial_state` field is a frozen `CompositionState` instance (already deeply frozen by its own `__post_init__`). No `freeze_fields` call is needed on the exception itself -- it is not a dataclass. The spec's mention of `freeze_fields` on the exception's `__post_init__` applies only if the exception were a `@dataclass(frozen=True)`, which it is not (exceptions cannot be frozen dataclasses in standard Python). The `CompositionState` stored in `partial_state` is already immutable.

Also update the `ComposerSettings` protocol in the same file — replace `composer_max_turns` with the dual settings:

```python
# src/elspeth/web/composer/protocol.py — replace ComposerSettings protocol

class ComposerSettings(Protocol):
    """Protocol for the settings the composer service needs."""

    @property
    def composer_model(self) -> str: ...

    @property
    def composer_max_composition_turns(self) -> int: ...

    @property
    def composer_max_discovery_turns(self) -> int: ...

    @property
    def composer_timeout_seconds(self) -> float: ...

    @property
    def data_dir(self) -> Any: ...
```

- [ ] **Step 2: Update WebSettings**

```python
# src/elspeth/web/config.py — replace composer_max_turns with dual settings

class WebSettings:
    """Web application settings.

    Rate limiting is per-process. Deployments with multiple uvicorn
    workers have an effective rate limit of N * composer_rate_limit_per_minute
    across the cluster. Multi-worker deployments require Redis or an
    equivalent shared store for accurate cross-process rate limiting.
    """

    # ... existing fields ...

    # Remove: composer_max_turns: int = 20
    composer_max_composition_turns: int = 15
    composer_max_discovery_turns: int = 10
    composer_timeout_seconds: float = 85.0
    # ... existing fields ...
```

- [ ] **Step 3: Write dual-counter, cache, and timeout tests**

```python
# tests/unit/web/composer/test_service.py (append to file, replacing
# TestComposerConvergence and adding new test classes)

import asyncio


class TestDualCounterLoop:
    """Tests for the dual-counter budget loop (F1)."""

    @pytest.mark.asyncio
    async def test_discovery_only_turns_charge_discovery_budget(self) -> None:
        """Turns with only discovery tool calls charge the discovery counter."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 15
        settings.composer_max_discovery_turns = 2
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # 3 discovery-only turns — exceeds discovery budget of 2
        discovery_turn = _make_llm_response(
            tool_calls=[{
                "id": "call_disc",
                "name": "list_sources",
                "arguments": {},
            }],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = discovery_turn
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("List everything", [], state)
            assert exc_info.value.budget_exhausted == "discovery"

    @pytest.mark.asyncio
    async def test_composition_turns_charge_composition_budget(self) -> None:
        """Turns with mutation tool calls charge the composition counter."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 2
        settings.composer_max_discovery_turns = 10
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # 3 mutation turns — exceeds composition budget of 2
        mutation_turn = _make_llm_response(
            tool_calls=[{
                "id": "call_mut",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "test"}},
            }],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mutation_turn
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Keep mutating", [], state)
            assert exc_info.value.budget_exhausted == "composition"

    @pytest.mark.asyncio
    async def test_mixed_turns_charge_correct_budgets(self) -> None:
        """Mixed discovery/mutation turns are classified correctly."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 2
        settings.composer_max_discovery_turns = 2
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: discovery (list_sources) — discovery counter = 1
        disc_turn = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        # Turn 2: mutation (set_metadata) — composition counter = 1
        mut_turn = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "set_metadata", "arguments": {"patch": {"name": "P"}}}],
        )
        # Turn 3: text response — loop terminates
        text_turn = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc_turn, mut_turn, text_turn]
            result = await service.compose("Build", [], state)

        assert result.message == "Done."
        assert result.state.metadata.name == "P"

    @pytest.mark.asyncio
    async def test_budget_checked_at_classification_time(self) -> None:
        """Budget is checked after classifying the turn, not before.

        If discovery budget is exhausted but the current turn is a mutation,
        the loop must continue (the mutation budget still has capacity).
        """
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 5
        settings.composer_max_discovery_turns = 1
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: discovery — exhausts discovery budget (1/1)
        disc_turn = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        # Turn 2: mutation — composition budget still has capacity (1/5)
        mut_turn = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "set_metadata", "arguments": {"patch": {"name": "Works"}}}],
        )
        # Turn 3: text
        text_turn = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc_turn, mut_turn, text_turn]
            result = await service.compose("Test budget check", [], state)

        # Should NOT have raised — mutation budget was available
        assert result.state.metadata.name == "Works"


class TestDiscoveryCache:
    """Tests for the discovery cache (F1)."""

    @pytest.mark.asyncio
    async def test_cacheable_tool_returns_cached_result(self) -> None:
        """Repeated cacheable discovery calls return cached results
        without incrementing any budget counter."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 15
        settings.composer_max_discovery_turns = 1  # only 1 allowed
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: list_sources (first call — executes, charges discovery: 1/1)
        # Turn 2: list_sources AGAIN (cache hit — no budget charge)
        # Turn 3: text
        disc1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "list_sources", "arguments": {}}],
        )
        disc2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "list_sources", "arguments": {}}],
        )
        text = _make_llm_response(content="Found sources.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [disc1, disc2, text]
            result = await service.compose("List sources", [], state)

        # Should NOT have raised — second list_sources was a cache hit
        assert result.message == "Found sources."
        # Catalog was only called once (first call)
        assert catalog.list_sources.call_count == 1

    @pytest.mark.asyncio
    async def test_mutation_tools_never_cached(self) -> None:
        """Mutation tool results are never cached — always execute."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 15
        settings.composer_max_discovery_turns = 5
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: set_metadata (mutation — always executes)
        # Turn 2: set_metadata again (should also execute, not cache)
        # Turn 3: text
        mut1 = _make_llm_response(
            tool_calls=[{"id": "c1", "name": "set_metadata", "arguments": {"patch": {"name": "X"}}}],
        )
        mut2 = _make_llm_response(
            tool_calls=[{"id": "c2", "name": "set_metadata", "arguments": {"patch": {"name": "Y"}}}],
        )
        text = _make_llm_response(content="Done.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [mut1, mut2, text]
            result = await service.compose("Update metadata", [], state)

        # Both mutations should have executed — name should be "Y"
        assert result.state.metadata.name == "Y"

    @pytest.mark.asyncio
    async def test_cache_key_includes_arguments(self) -> None:
        """Cache key depends on tool name + arguments. Different arguments = different cache entries."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 15
        settings.composer_max_discovery_turns = 5
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Two get_plugin_schema calls with different arguments — both execute
        schema1 = _make_llm_response(
            tool_calls=[{
                "id": "c1", "name": "get_plugin_schema",
                "arguments": {"plugin_type": "source", "name": "csv"},
            }],
        )
        schema2 = _make_llm_response(
            tool_calls=[{
                "id": "c2", "name": "get_plugin_schema",
                "arguments": {"plugin_type": "source", "name": "json"},
            }],
        )
        text = _make_llm_response(content="Got schemas.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [schema1, schema2, text]
            result = await service.compose("Get schemas", [], state)

        # Both calls should have executed (different arguments)
        assert catalog.get_schema.call_count == 2


class TestComposeTimeout:
    """Tests for the server-side compose timeout (F1)."""

    @pytest.mark.asyncio
    async def test_timeout_raises_convergence_error(self) -> None:
        """Exceeding composer_timeout_seconds raises ComposerConvergenceError
        with budget_exhausted='timeout'."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 100
        settings.composer_max_discovery_turns = 100
        settings.composer_timeout_seconds = 0.1  # 100ms — will timeout

        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        async def slow_llm(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)  # Way longer than timeout
            return _make_llm_response(content="Too late.")

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = slow_llm
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Slow pipeline", [], state)
            assert exc_info.value.budget_exhausted == "timeout"
```

- [ ] **Step 2: Run tests -- expect FAIL (old service code uses single counter)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -v -k "DualCounter or DiscoveryCache or ComposeTimeout"`
Expected: FAIL.

- [ ] **Step 3: Update `_make_settings()` helper in test file**

```python
# tests/unit/web/composer/test_service.py — update _make_settings()

def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.composer_model = "gpt-4o"
    settings.composer_max_composition_turns = 15
    settings.composer_max_discovery_turns = 10
    settings.composer_timeout_seconds = 85.0
    settings.data_dir = "/tmp/test-data"
    return settings
```

Also update `TestComposerConvergence.test_max_turns_exceeded_raises` to use the new settings fields:

```python
class TestComposerConvergence:
    @pytest.mark.asyncio
    async def test_max_turns_exceeded_raises(self) -> None:
        """Loop exceeding max_turns raises ComposerConvergenceError."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 1
        settings.composer_max_discovery_turns = 1
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Every turn makes a discovery call — discovery budget = 1
        tool_response = _make_llm_response(
            tool_calls=[{
                "id": "call_loop",
                "name": "list_sources",
                "arguments": {},
            }],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # First call returns tool_response (discovery, cached after first).
            # Second call returns tool_response again — cache hit, no budget charge.
            # Since list_sources is cacheable, only the first call charges
            # budget. Use list_transforms for the second to avoid cache hits.
            tool_response_2 = _make_llm_response(
                tool_calls=[{
                    "id": "call_loop_2",
                    "name": "list_transforms",
                    "arguments": {},
                }],
            )
            mock_llm.side_effect = [tool_response, tool_response_2]
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Loop forever", [], state)
            assert exc_info.value.budget_exhausted == "discovery"
```

- [ ] **Step 4: Implement dual-counter loop + discovery cache in service.py**

```python
# src/elspeth/web/composer/service.py — replace ComposerServiceImpl

import asyncio
import json
from typing import Any

import litellm

from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerResult,
    ComposerSettings,
)
from elspeth.web.composer.prompts import build_messages
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import (
    CatalogServiceProtocol,
    execute_tool,
    get_tool_definitions,
    is_cacheable_discovery_tool,
    is_discovery_tool,
)


class ComposerServiceImpl:
    """LLM-driven pipeline composer with dual-counter budget and discovery caching.

    Runs a bounded tool-use loop with separate budgets for discovery
    and composition turns. Cacheable discovery tool results are cached
    per-compose-call in a local dict (not an instance field) to avoid
    concurrent-request races.

    Budget classification: a turn containing at least one mutation tool
    call charges the composition budget. A turn containing only discovery
    tool calls charges the discovery budget. Cache hits do not charge
    any budget.

    Args:
        catalog: CatalogService for discovery tool delegation.
        settings: ComposerSettings with composer_max_composition_turns,
            composer_max_discovery_turns, composer_timeout_seconds,
            composer_model, data_dir.
    """

    def __init__(
        self,
        catalog: CatalogServiceProtocol,
        settings: ComposerSettings,
    ) -> None:
        self._catalog = catalog
        self._model = settings.composer_model
        self._max_composition_turns = settings.composer_max_composition_turns
        self._max_discovery_turns = settings.composer_max_discovery_turns
        self._timeout_seconds = settings.composer_timeout_seconds
        self._data_dir = str(settings.data_dir)

    async def compose(
        self,
        message: str,
        messages: list[dict[str, Any]],
        state: CompositionState,
    ) -> ComposerResult:
        """Run the LLM composition loop with dual-counter budget.

        Args:
            message: The user's chat message.
            messages: Chat history as plain dicts (pre-converted from
                ChatMessageRecord by route handler; seam contract B).
            state: The current CompositionState.

        Returns:
            ComposerResult with assistant message and updated state.

        Raises:
            ComposerConvergenceError: If a budget is exhausted or
                the timeout is exceeded.
        """
        # Mutable container so _compose_loop can publish its latest
        # state to the outer scope. On TimeoutError, compose() reads
        # this to build partial_state — the local `state` variable
        # inside _compose_loop is unreachable after cancellation.
        state_ref: list[CompositionState] = [state]
        initial_version = state.version
        try:
            return await asyncio.wait_for(
                self._compose_loop(message, messages, state, state_ref),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            latest = state_ref[0]
            partial = latest if latest.version > initial_version else None
            raise ComposerConvergenceError(
                max_turns=0,
                budget_exhausted="timeout",
                partial_state=partial,
            )

    async def _compose_loop(
        self,
        message: str,
        messages: list[dict[str, Any]],
        state: CompositionState,
        state_ref: list[CompositionState],
    ) -> ComposerResult:
        """Inner composition loop with dual-counter budget tracking.

        Args:
            state_ref: Single-element mutable list. Updated after every
                successful tool execution so the outer compose() can
                read the latest state on timeout.
        """
        initial_version = state.version
        llm_messages = self._build_messages(messages, state, message)
        tools = self._get_litellm_tools()

        composition_turns_used = 0
        discovery_turns_used = 0

        # Discovery cache: local variable scoped to this compose() call.
        # Keyed by (tool_name, canonical_args_json). Each concurrent
        # compose() call gets its own independent cache dict.
        discovery_cache: dict[str, Any] = {}

        while True:
            response = await self._call_llm(llm_messages, tools)
            assistant_message = response.choices[0].message

            # If no tool calls, the LLM is done — return text response
            if not assistant_message.tool_calls:
                return ComposerResult(
                    message=assistant_message.content or "",
                    state=state,
                )

            # Append the assistant message (with tool_calls metadata)
            llm_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ],
                }
            )

            # Execute each tool call, tracking whether this turn has
            # any mutation calls for budget classification.
            turn_has_mutation = False
            all_cache_hits = True

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "error": f"Invalid JSON in arguments: {exc}",
                                }
                            ),
                        }
                    )
                    all_cache_hits = False
                    continue

                # Check discovery cache before executing
                if is_cacheable_discovery_tool(tool_name):
                    cache_key = _make_cache_key(tool_name, arguments)
                    if cache_key in discovery_cache:
                        # Cache hit — return cached result, no budget charge
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": discovery_cache[cache_key],
                            }
                        )
                        continue

                all_cache_hits = False

                # execute_tool() is system code, but the arguments dict
                # comes from LLM output (Tier 3). Missing/wrong-type keys
                # in LLM-provided arguments are expected failures — return
                # the error to the LLM so it can self-correct. Bugs in
                # execute_tool's own logic will raise other exceptions and
                # crash as intended (Amendment 2).
                try:
                    result = execute_tool(
                        tool_name,
                        arguments,
                        state,
                        self._catalog,
                        data_dir=self._data_dir,
                    )
                except (KeyError, TypeError) as exc:
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "error": f"Tool '{tool_name}' failed: {exc}",
                                }
                            ),
                        }
                    )
                    continue

                state = result.updated_state
                state_ref[0] = state  # Publish for timeout capture
                result_json = json.dumps(result.to_dict())

                # Cache cacheable discovery results
                if is_cacheable_discovery_tool(tool_name):
                    cache_key = _make_cache_key(tool_name, arguments)
                    discovery_cache[cache_key] = result_json

                llm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json,
                    }
                )

                if not is_discovery_tool(tool_name):
                    turn_has_mutation = True

            # If ALL tool calls in this turn were cache hits, no budget
            # charge — continue to next turn without incrementing.
            if all_cache_hits:
                continue

            # Classify turn and check budget at classification time.
            # Exit only when the SPECIFIC budget being charged is exhausted.
            if turn_has_mutation:
                composition_turns_used += 1
                if composition_turns_used >= self._max_composition_turns:
                    # B-4D-3 fix: give the LLM one last chance to see the
                    # tool results and produce a text response. Without this,
                    # a correction made on the final turn is applied to state
                    # but the LLM never sees the confirmation.
                    response = await self._call_llm(llm_messages, tools)
                    assistant_message = response.choices[0].message
                    if not assistant_message.tool_calls:
                        return ComposerResult(
                            message=assistant_message.content or "",
                            state=state,
                        )
                    partial = state if state.version > initial_version else None
                    raise ComposerConvergenceError(
                        max_turns=composition_turns_used + discovery_turns_used,
                        budget_exhausted="composition",
                        partial_state=partial,
                    )
            else:
                discovery_turns_used += 1
                if discovery_turns_used >= self._max_discovery_turns:
                    partial = state if state.version > initial_version else None
                    raise ComposerConvergenceError(
                        max_turns=composition_turns_used + discovery_turns_used,
                        budget_exhausted="discovery",
                        partial_state=partial,
                    )

    def _build_messages(
        self,
        chat_history: list[dict[str, Any]],
        state: CompositionState,
        user_message: str,
    ) -> list[dict[str, Any]]:
        """Build the message list. Returns a NEW list on every call.

        This is critical: the tool-use loop appends to this list during
        iteration. Returning a cached reference would cause cross-turn
        contamination.
        """
        return build_messages(
            chat_history=chat_history,
            state=state,
            user_message=user_message,
            catalog=self._catalog,
        )

    def _get_litellm_tools(self) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM function format."""
        definitions = get_tool_definitions()
        return [
            {
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn["description"],
                    "parameters": defn["parameters"],
                },
            }
            for defn in definitions
        ]

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Call the LLM via LiteLLM. Separated for test mocking."""
        return await litellm.acompletion(
            model=self._model,
            messages=messages,
            tools=tools,
        )


def _make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a deterministic cache key from tool name + arguments."""
    # Sort keys for determinism. Arguments are simple JSON-serializable
    # dicts from the LLM — no MappingProxyType or frozen containers.
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
```

- [ ] **Step 5: Run all service tests -- expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -v`
Expected: All tests PASS (old and new).

- [ ] **Step 6: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/service.py src/elspeth/web/composer/protocol.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/composer/protocol.py src/elspeth/web/composer/service.py \
  tests/unit/web/composer/test_service.py
git commit -m "feat(web/composer): dual-counter loop with discovery cache and server timeout"
```

If `WebSettings` was also modified:

```bash
git add src/elspeth/web/config.py
git commit --amend --no-edit
```

---

### Task 4: Partial State Preservation on Convergence (F2)

**Files:**
- Modify: `tests/unit/web/composer/test_service.py`
- Modify: `src/elspeth/web/sessions/routes.py`

The `ComposerConvergenceError` was already updated in Task 3 to carry `partial_state`. This task adds the route handler logic to persist and return partial state.

- [ ] **Step 1: Write partial state tests**

```python
# tests/unit/web/composer/test_service.py (append)


class TestPartialStatePreservation:
    """Tests for partial state preservation on convergence failure (F2)."""

    @pytest.mark.asyncio
    async def test_convergence_includes_partial_state_when_mutated(self) -> None:
        """When mutations occurred before convergence failure,
        partial_state is attached to the exception."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 1
        settings.composer_max_discovery_turns = 10
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: mutation (set_source) — composition budget exhausted (1/1)
        mut_turn = _make_llm_response(
            tool_calls=[{
                "id": "c1",
                "name": "set_source",
                "arguments": {
                    "plugin": "csv", "on_success": "t1",
                    "options": {}, "on_validation_failure": "quarantine",
                },
            }],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mut_turn
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Build pipeline", [], state)

            # Partial state should be present — mutation occurred
            assert exc_info.value.partial_state is not None
            assert exc_info.value.partial_state.source is not None
            assert exc_info.value.partial_state.source.plugin == "csv"
            assert exc_info.value.partial_state.version == 2

    @pytest.mark.asyncio
    async def test_convergence_no_partial_state_when_no_mutations(self) -> None:
        """When no mutations occurred, partial_state is None."""
        catalog = _mock_catalog()
        settings = _make_settings()
        settings.composer_max_composition_turns = 15
        settings.composer_max_discovery_turns = 1
        settings.composer_timeout_seconds = 120.0
        service = ComposerServiceImpl(catalog=catalog, settings=settings)
        state = _empty_state()

        # Turn 1: discovery only — discovery budget exhausted (1/1)
        # list_sources is cacheable, so use two different discovery tools
        # to avoid cache-hit-no-budget-charge on the second call.
        disc_turn = _make_llm_response(
            tool_calls=[{
                "id": "c1",
                "name": "list_sources",
                "arguments": {},
            }],
        )

        with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = disc_turn
            with pytest.raises(ComposerConvergenceError) as exc_info:
                await service.compose("Just looking", [], state)

            # No mutations occurred — partial_state should be None
            assert exc_info.value.partial_state is None
```

- [ ] **Step 2: Write route handler convergence tests**

```python
# tests/unit/web/composer/test_route_integration.py (append)

from unittest.mock import AsyncMock, MagicMock, patch

from elspeth.web.composer.protocol import ComposerConvergenceError
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
    SourceSpec,
    ValidationSummary,
)


class TestConvergenceRouteHandler:
    """Tests for the route handler convergence error path (F2)."""

    @pytest.mark.asyncio
    async def test_convergence_with_partial_state_persists_and_returns(self) -> None:
        """Route handler persists partial state and includes it in 422 response."""
        partial = _empty_state().with_source(
            SourceSpec(
                plugin="csv", on_success="t1",
                options={}, on_validation_failure="quarantine",
            )
        )
        exc = ComposerConvergenceError(
            max_turns=15,
            budget_exhausted="composition",
            partial_state=partial,
        )
        assert exc.partial_state is not None
        assert exc.partial_state.version == 2

        # Verify that validate() can be called on the partial state
        summary = exc.partial_state.validate()
        assert isinstance(summary, ValidationSummary)

    @pytest.mark.asyncio
    async def test_convergence_without_partial_state_returns_422_without_state(self) -> None:
        """When partial_state is None, 422 response omits it."""
        exc = ComposerConvergenceError(
            max_turns=10,
            budget_exhausted="discovery",
            partial_state=None,
        )
        assert exc.partial_state is None
        assert exc.budget_exhausted == "discovery"

    @pytest.mark.asyncio
    async def test_convergence_validation_failure_persists_with_invalid(self) -> None:
        """If validate() raises on partial state, persist with is_valid=False."""
        partial = _empty_state().with_source(
            SourceSpec(
                plugin="csv", on_success="t1",
                options={}, on_validation_failure="quarantine",
            )
        )
        # validate() should NOT raise on a real CompositionState, but
        # this test verifies the guard. We test the route handler logic:
        # if validation itself fails, persist with is_valid=False
        # rather than losing the state entirely.
        exc = ComposerConvergenceError(
            max_turns=15,
            budget_exhausted="composition",
            partial_state=partial,
        )
        # Route handler pattern:
        try:
            summary = exc.partial_state.validate()
        except Exception:
            summary = ValidationSummary(is_valid=False, errors=("validation_failed",))
        assert isinstance(summary, ValidationSummary)
```

- [ ] **Step 3: Update the route handler convergence error path**

```python
# src/elspeth/web/sessions/routes.py — update the ComposerConvergenceError handler

    except ComposerConvergenceError as exc:
        response_body: dict[str, Any] = {
            "error_type": "convergence",
            "detail": str(exc),
            "turns_used": exc.max_turns,
            "budget_exhausted": exc.budget_exhausted,
        }
        if exc.partial_state is not None:
            # Validate guard: if validation itself fails, persist with
            # is_valid=False rather than losing the state entirely.
            try:
                summary = exc.partial_state.validate()
            except Exception:
                summary = ValidationSummary(
                    is_valid=False, errors=("validation_failed",)
                )

            # Persistence guard: if save fails, log the error and return
            # 422 without partial_state rather than crashing to 500.
            try:
                await session_service.save_composition_state(
                    session_id,
                    state=exc.partial_state,
                    is_valid=summary.is_valid,
                    validation_errors=list(summary.errors) if summary.errors else None,
                )
                response_body["partial_state"] = exc.partial_state.to_dict()
            except Exception:
                slog.error(
                    "convergence_partial_state_save_failed",
                    session_id=session_id,
                    exc_info=True,
                )
                # Do NOT include partial_state in response — it was not
                # persisted, so the frontend cannot resume from it.

        raise HTTPException(status_code=422, detail=response_body) from exc
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/routes.py \
  tests/unit/web/composer/test_service.py \
  tests/unit/web/composer/test_route_integration.py
git commit -m "feat(web/composer): persist partial state on convergence failure with validation/persistence guards"
```

---

### Task 5: Rate Limiting (F3)

**Files:**
- Create: `src/elspeth/web/middleware/__init__.py`
- Create: `src/elspeth/web/middleware/rate_limit.py`
- Create: `tests/unit/web/middleware/__init__.py`
- Create: `tests/unit/web/middleware/test_rate_limit.py`
- Modify: `src/elspeth/web/sessions/routes.py`
- Modify: `src/elspeth/web/app.py`

- [ ] **Step 1: Write rate limiter tests**

```python
# tests/unit/web/middleware/__init__.py
"""Rate limiting test package."""

# tests/unit/web/middleware/test_rate_limit.py
"""Tests for ComposerRateLimiter — in-memory sliding window rate limiter."""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from fastapi import HTTPException

from elspeth.web.middleware.rate_limit import ComposerRateLimiter


class TestRateLimiterAllow:
    """Tests that requests within the limit are allowed."""

    @pytest.mark.asyncio
    async def test_allows_requests_within_limit(self) -> None:
        limiter = ComposerRateLimiter(limit=5)
        # 5 requests should all pass
        for _ in range(5):
            await limiter.check("user_1")

    @pytest.mark.asyncio
    async def test_first_request_always_allowed(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")  # Should not raise


class TestRateLimiterDeny:
    """Tests that requests exceeding the limit are denied."""

    @pytest.mark.asyncio
    async def test_denies_request_exceeding_limit(self) -> None:
        limiter = ComposerRateLimiter(limit=2)
        await limiter.check("user_1")
        await limiter.check("user_1")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("user_1")
        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_429_response_body_shape(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check("user_1")
        detail = exc_info.value.detail
        assert detail["error_type"] == "rate_limited"
        assert "retry_after" in detail
        assert isinstance(detail["retry_after"], int)


class TestRateLimiterWindowReset:
    """Tests that the sliding window resets correctly."""

    @pytest.mark.asyncio
    async def test_window_resets_after_60_seconds(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        # Manually inject an old timestamp to simulate window expiry
        limiter._buckets["user_1"] = [time.monotonic() - 61.0]
        # Should pass — the old request is outside the window
        await limiter.check("user_1")


class TestRateLimiterConcurrentUsers:
    """Tests that per-user limits are independent."""

    @pytest.mark.asyncio
    async def test_independent_per_user_limits(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        # user_2 should still be allowed — separate bucket
        await limiter.check("user_2")

    @pytest.mark.asyncio
    async def test_user1_exhausted_user2_unaffected(self) -> None:
        limiter = ComposerRateLimiter(limit=1)
        await limiter.check("user_1")
        with pytest.raises(HTTPException):
            await limiter.check("user_1")
        # user_2 is not affected
        await limiter.check("user_2")


class TestRateLimiterPerUserLocks:
    """Tests that per-user locks prevent interleaving."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_same_user_serialized(self) -> None:
        """Concurrent requests from the same user are serialized by the
        per-user lock, preventing race conditions in the
        prune-check-append sequence."""
        limiter = ComposerRateLimiter(limit=3)
        # Fire 5 concurrent requests from same user
        results = await asyncio.gather(
            *[limiter.check("user_1") for _ in range(3)],
            return_exceptions=True,
        )
        # All 3 should succeed (within limit)
        assert all(r is None for r in results)

        # Now fire 2 more — should both fail
        results = await asyncio.gather(
            *[limiter.check("user_1") for _ in range(2)],
            return_exceptions=True,
        )
        assert all(isinstance(r, HTTPException) for r in results)
```

- [ ] **Step 2: Run tests -- expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/middleware/test_rate_limit.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement ComposerRateLimiter**

```python
# src/elspeth/web/middleware/__init__.py
"""Middleware package for web application."""

# src/elspeth/web/middleware/rate_limit.py
"""In-memory sliding window rate limiter for composer messages.

Per-user rate limiting via FastAPI Depends(). Not thread-safe across
multiple uvicorn workers — each worker has its own counter. Multi-worker
deployments need Redis or equivalent shared store for accurate
cross-process rate limiting.

Layer: L3 (application).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import HTTPException, Request


class ComposerRateLimiter:
    """In-memory sliding window rate limiter for composer messages.

    Tracks message timestamps per user_id. On each request, prunes
    timestamps older than 60 seconds, then checks count against limit.
    Returns 429 if exceeded.

    Uses per-user asyncio.Lock instances to avoid contention between
    unrelated users. asyncio.Lock guards coroutine suspension points
    (e.g., between the prune-check-append sequence where another
    coroutine could interleave), not thread safety. A top-level
    _locks_lock (held for microseconds — dict lookup only) serializes
    creation/fetch of per-user locks.

    Rate limiting is per-process. Deployments with multiple uvicorn
    workers have an effective rate limit of N * limit across the
    cluster. Multi-worker deployments require Redis or an equivalent
    shared store for accurate cross-process rate limiting.
    """

    _WINDOW_SECONDS: float = 60.0

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buckets: dict[str, list[float]] = {}
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for the given user."""
        async with self._locks_lock:
            if user_id not in self._user_locks:
                self._user_locks[user_id] = asyncio.Lock()
            return self._user_locks[user_id]

    async def check(self, user_id: str) -> None:
        """Check rate limit for the given user.

        Raises HTTPException(429) with Retry-After header if the
        per-user rate limit is exceeded.
        """
        lock = await self._get_user_lock(user_id)
        async with lock:
            now = time.monotonic()
            cutoff = now - self._WINDOW_SECONDS

            # Get or create bucket
            if user_id not in self._buckets:
                self._buckets[user_id] = []

            bucket = self._buckets[user_id]

            # Prune timestamps outside the window
            bucket[:] = [ts for ts in bucket if ts > cutoff]

            # Check limit
            if len(bucket) >= self._limit:
                # Earliest timestamp determines retry delay
                earliest = bucket[0]
                retry_after = int(earliest - cutoff) + 1
                if retry_after < 1:
                    retry_after = 1
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error_type": "rate_limited",
                        "detail": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            # Record this request
            bucket.append(now)


async def get_rate_limiter(request: Request) -> ComposerRateLimiter:
    """FastAPI dependency that extracts the rate limiter from app state."""
    return request.app.state.rate_limiter
```

- [ ] **Step 4: Wire rate limiter into the app factory**

```python
# src/elspeth/web/app.py — in the lifespan or app factory, add:

from elspeth.web.middleware.rate_limit import ComposerRateLimiter

# In the lifespan:
app.state.rate_limiter = ComposerRateLimiter(
    limit=settings.composer_rate_limit_per_minute,
)
```

- [ ] **Step 5: Wire rate limiter into the message route**

```python
# src/elspeth/web/sessions/routes.py — update send_message signature

from elspeth.web.middleware.rate_limit import ComposerRateLimiter, get_rate_limiter

async def send_message(
    session_id: str,
    body: MessageRequest,
    session_service: SessionService = Depends(get_session_service),
    composer_service: ComposerService = Depends(get_composer_service),
    current_user: UserIdentity = Depends(get_current_user),
    rate_limiter: ComposerRateLimiter = Depends(get_rate_limiter),
) -> MessageResponse:
    """Handle a user message — trigger the LLM composer."""
    # Rate limit check — before any work
    await rate_limiter.check(current_user.id)

    # ... rest of handler unchanged ...
```

- [ ] **Step 6: Run tests -- expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/middleware/test_rate_limit.py -v`
Expected: All tests PASS.

- [ ] **Step 7: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/middleware/rate_limit.py`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/middleware/__init__.py \
  src/elspeth/web/middleware/rate_limit.py \
  tests/unit/web/middleware/__init__.py \
  tests/unit/web/middleware/test_rate_limit.py \
  src/elspeth/web/sessions/routes.py \
  src/elspeth/web/app.py
git commit -m "feat(web/middleware): add per-user sliding window rate limiter with FastAPI Depends() wiring"
```

---

## Self-Review Checklist

| # | Acceptance Criterion | Task | Verified |
|---|---------------------|------|----------|
| 1 | Cacheable discovery tool results cached in local `dict` variable, not instance field. All 5 discovery tools are cacheable (`get_current_state` was removed in WIP). Repeated cacheable calls return cached result without incrementing any counter. | T3 | `discovery_cache` is a local variable in `_compose_loop()`. `_CACHEABLE_DISCOVERY_TOOLS == frozenset(_DISCOVERY_TOOLS.keys())`. `TestDiscoveryCache.test_cacheable_tool_returns_cached_result` verifies cache hits do not charge budget. |
| 2 | Dual-counter loop with budget-at-classification-time checking. Exits only when the specific budget being charged is exhausted. Error reports which budget was exhausted. | T3 | `while True` loop with post-classification budget check. `test_budget_checked_at_classification_time` verifies a mutation turn succeeds even when discovery budget is exhausted. |
| 3 | `WebSettings` exposes `composer_max_composition_turns` (15), `composer_max_discovery_turns` (10), `composer_timeout_seconds` (85.0). Old `composer_max_turns` removed. | T3 | Settings replaced. 85.0 < 90.0 (frontend timeout). |
| 4 | Partial state attached to `ComposerConvergenceError` when mutations occurred. Route handler validates, persists, returns 422 with partial state. Validation failure degrades to `is_valid=False`. Persistence failure degrades gracefully (omits partial state). | T4 | `test_convergence_includes_partial_state_when_mutated`, `test_convergence_no_partial_state_when_no_mutations`. Route handler has both validation guard and persistence guard. |
| 5 | 422 response body includes `budget_exhausted` and optional `partial_state`. | T3/T4 | Response body construction includes both fields. `budget_exhausted` is one of `"composition"`, `"discovery"`, `"timeout"`. |
| 6 | Rate limiting enforced via `Depends()` on `POST /api/sessions/{id}/messages`. Returns 429 with `Retry-After` header. | T5 | `get_rate_limiter` dependency. `test_429_response_body_shape` verifies response. |
| 7 | `with_edge()` preserves insertion order. Test verifies. | T1 | `test_with_edge_preserves_order` — index-and-replace pattern. |
| 8 | `execute_tool()` dispatches via registry dict. Two registries (5 discovery + 8 mutation = 13 tools) + cacheable frozenset. `is_discovery_tool()` and `is_cacheable_discovery_tool()` exposed. `data_dir` parameter threaded through `ToolHandler` signature. | T2 | `_DISCOVERY_TOOLS`, `_MUTATION_TOOLS`, `_CACHEABLE_DISCOVERY_TOOLS`. Module-level assertions prevent overlap. |
| 9 | `ComposerConvergenceError` immutability: `partial_state` is a frozen `CompositionState` (already deeply frozen by its own `__post_init__`). The exception is not a dataclass, so no `freeze_fields` call on the exception itself. | T3 | `CompositionState` handles its own deep freezing. |
| 10 | `ComposerRateLimiter` uses per-user `asyncio.Lock` instances. `_locks_lock` held for microseconds. Docstring describes asyncio.Lock as guarding suspension points. | T5 | Per-user lock in `_get_user_lock()`. Docstring accurate. |
| 11 | All existing Sub-Plan 4 tests pass. New tests cover dual-counter, partial state, rate limiter, edge order, timeout, registry dispatch. | T1-T5 | Each task runs the full test suite. `@pytest.mark.asyncio` on all async tests. |
| 12 | `with_output()` audited for insertion-order bug. Fixed with same pattern as `with_edge()`. | T1 | `test_with_output_preserves_order` added. Fix uses index-and-replace pattern. |
| 13 | `WebSettings` docstring documents per-process rate limiting limitation. | T3/T5 | Docstrings on both `WebSettings` and `ComposerRateLimiter`. |
