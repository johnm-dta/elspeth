# RoutingReason Discriminated Union Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace loose `Mapping[str, Any]` typing on `RoutingAction.reason` with a 2-variant discriminated union, enabling compile-time type safety for audit reliability.

**Architecture:** Two TypedDict variants matching actual production code paths: ConfigGateReason (for config-driven gates) and PluginGateReason (for plugin gates). Field presence distinguishes variants - no runtime discriminator needed.

**Tech Stack:** Python TypedDict with `NotRequired`, frozen dataclass, `copy.deepcopy` for mutation protection.

**Bead:** elspeth-rapid-5vc

---

## Implementation Summary

- Introduced `ConfigGateReason`, `PluginGateReason`, and `RoutingReason` union (`src/elspeth/contracts/errors.py`).
- `RoutingAction` now uses typed `RoutingReason` with defensive copy (`src/elspeth/contracts/routing.py`).
- Property tests generate valid routing reasons (`tests/property/engine/test_executor_properties.py`, `tests/property/contracts/test_serialization_properties.py`).

## Task 1: Fix Broken routing.py Code

**Files:**
- Modify: `src/elspeth/contracts/routing.py:8, 100, 120, 128, 144`

**Context:** Code is currently broken - `_freeze_dict` was renamed to `_copy_reason` but call sites weren't updated.

**Step 1: Verify code is broken**

Run: `.venv/bin/python -c "from elspeth.contracts import RoutingAction"`
Expected: `NameError: name '_freeze_dict' is not defined` or similar

**Step 2: Fix imports**

In `src/elspeth/contracts/routing.py`, change line 8 from:

```python
from typing import cast
```

To:

```python
from typing import Any, cast
```

**Step 3: Fix route() method**

Change lines 94-121 - update signature and replace `_freeze_dict` with `_copy_reason`:

```python
    @classmethod
    def route(
        cls,
        label: str,
        *,
        mode: RoutingMode = RoutingMode.MOVE,
        reason: RoutingReason | None = None,
    ) -> "RoutingAction":
        """Route to a specific labeled destination.

        Gates return semantic route labels (e.g., "above", "below", "match").
        The executor resolves these labels via the plugin's `routes` config
        to determine the actual destination (sink name or "continue").

        Args:
            label: Route label that will be resolved via routes config
            mode: MOVE (default). COPY mode not supported - use fork_to_paths() instead.
            reason: Audit trail information about why this route was chosen

        Raises:
            ValueError: If mode is COPY (architectural limitation)
        """
        return cls(
            kind=RoutingKind.ROUTE,
            destinations=(label,),
            mode=mode,
            reason=_copy_reason(reason),
        )
```

**Step 4: Fix fork_to_paths() method**

Change lines 123-145 - update signature and replace `_freeze_dict` with `_copy_reason`:

```python
    @classmethod
    def fork_to_paths(
        cls,
        paths: list[str],
        *,
        reason: RoutingReason | None = None,
    ) -> "RoutingAction":
        """Fork token to multiple parallel paths (always copy mode).

        Raises:
            ValueError: If paths is empty or contains duplicates.
        """
        if not paths:
            raise ValueError("fork_to_paths requires at least one destination path")
        if len(paths) != len(set(paths)):
            duplicates = [p for p in paths if paths.count(p) > 1]
            raise ValueError(f"fork_to_paths requires unique path names (duplicates: {sorted(set(duplicates))})")
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            mode=RoutingMode.COPY,
            reason=_copy_reason(reason),
        )
```

**Step 5: Verify import works**

Run: `.venv/bin/python -c "from elspeth.contracts import RoutingAction; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/elspeth/contracts/routing.py
git commit -m "$(cat <<'EOF'
fix(routing): repair broken factory methods after _freeze_dict rename

- Add missing 'Any' import
- Update route() to use _copy_reason() instead of _freeze_dict()
- Update fork_to_paths() to use _copy_reason() instead of _freeze_dict()
- Change signatures to accept RoutingReason | None

Fixes NameError that was blocking imports.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Define 2-Variant Discriminated Union

**Files:**
- Modify: `src/elspeth/contracts/errors.py:21-38`
- Test: `tests/contracts/test_errors.py`

**Step 1: Write failing tests for new TypedDict variants**

Add to `tests/contracts/test_errors.py`:

```python
class TestRoutingReasonVariants:
    """Tests for RoutingReason 2-variant discriminated union."""

    def test_config_gate_reason_required_keys(self) -> None:
        """ConfigGateReason has condition and result as required."""
        from elspeth.contracts import ConfigGateReason

        assert ConfigGateReason.__required_keys__ == frozenset({"condition", "result"})

    def test_plugin_gate_reason_required_keys(self) -> None:
        """PluginGateReason has rule and matched_value as required."""
        from elspeth.contracts import PluginGateReason

        assert PluginGateReason.__required_keys__ == frozenset({"rule", "matched_value"})

    def test_plugin_gate_reason_optional_keys(self) -> None:
        """PluginGateReason has threshold, field, comparison as optional."""
        from elspeth.contracts import PluginGateReason

        assert PluginGateReason.__optional_keys__ == frozenset({"threshold", "field", "comparison"})


class TestRoutingReasonUsage:
    """Tests for constructing valid RoutingReason variants."""

    def test_config_gate_reason_construction(self) -> None:
        """ConfigGateReason can be constructed with required fields."""
        from elspeth.contracts import ConfigGateReason

        reason: ConfigGateReason = {
            "condition": "row['score'] > 100",
            "result": "true",
        }
        assert reason["condition"] == "row['score'] > 100"
        assert reason["result"] == "true"

    def test_plugin_gate_reason_minimal(self) -> None:
        """PluginGateReason works with only required fields."""
        from elspeth.contracts import PluginGateReason

        reason: PluginGateReason = {
            "rule": "threshold_exceeded",
            "matched_value": 150,
        }
        assert reason["rule"] == "threshold_exceeded"
        assert reason["matched_value"] == 150

    def test_plugin_gate_reason_with_optional_fields(self) -> None:
        """PluginGateReason accepts optional threshold fields."""
        from elspeth.contracts import PluginGateReason

        reason: PluginGateReason = {
            "rule": "value exceeds threshold",
            "matched_value": 150,
            "threshold": 100.0,
            "field": "score",
            "comparison": ">",
        }
        assert reason["threshold"] == 100.0
        assert reason["field"] == "score"
        assert reason["comparison"] == ">"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/contracts/test_errors.py::TestRoutingReasonVariants -v`
Expected: FAIL with `ImportError: cannot import name 'ConfigGateReason'`

**Step 3: Implement 2-variant discriminated union**

Replace the `RoutingReason` section in `src/elspeth/contracts/errors.py` (lines 21-38):

```python
class ConfigGateReason(TypedDict):
    """Reason from config-driven gate (expression evaluation).

    Used by gates defined via GateSettings with condition expressions.
    The executor auto-generates this reason structure at executors.py:739.

    Fields:
        condition: The expression that was evaluated (e.g., "row['score'] > 100")
        result: The route label that matched (e.g., "true", "false")
    """

    condition: str
    result: str


class PluginGateReason(TypedDict):
    """Reason from plugin-based gate.

    Used by custom gate plugins implementing GateProtocol.
    Enforces minimum auditability: every routing decision MUST have
    a rule description and the value that triggered it.

    Required fields:
        rule: Human-readable description of what logic fired
        matched_value: The value that triggered the routing decision

    Optional fields (for threshold-style gates):
        threshold: The threshold value compared against
        field: The field name that was compared
        comparison: The comparison operator used (">", "<", ">=", etc.)
    """

    rule: str
    matched_value: Any
    threshold: NotRequired[float]
    field: NotRequired[str]
    comparison: NotRequired[str]


# Discriminated union - field presence distinguishes variants:
# - ConfigGateReason has "condition" and "result"
# - PluginGateReason has "rule" and "matched_value"
RoutingReason = ConfigGateReason | PluginGateReason
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/test_errors.py::TestRoutingReasonVariants -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/contracts/errors.py tests/contracts/test_errors.py
git commit -m "$(cat <<'EOF'
feat(contracts): add 2-variant discriminated union for RoutingReason

Replace single RoutingReason TypedDict with 2-variant union:
- ConfigGateReason: for config-driven expression gates (condition + result)
- PluginGateReason: for plugin gates (rule + matched_value + optional threshold fields)

Field presence distinguishes variants - no runtime discriminator needed.
Enforces minimum auditability while matching actual production code paths.

Part of elspeth-rapid-5vc type safety audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update contracts/__init__.py Exports

**Files:**
- Modify: `src/elspeth/contracts/__init__.py:101-106, 151-154`

**Step 1: Update imports**

Change lines 101-106 from:

```python
from elspeth.contracts.errors import (
    BatchPendingError,
    ExecutionError,
    RoutingReason,
    TransformReason,
)
```

To:

```python
from elspeth.contracts.errors import (
    BatchPendingError,
    ConfigGateReason,
    ExecutionError,
    PluginGateReason,
    RoutingReason,
    TransformReason,
)
```

**Step 2: Update __all__ list**

Change the errors section in `__all__` (around line 151-154) from:

```python
    # errors
    "BatchPendingError",
    "ExecutionError",
    "RoutingReason",
    "TransformReason",
```

To:

```python
    # errors
    "BatchPendingError",
    "ConfigGateReason",
    "ExecutionError",
    "PluginGateReason",
    "RoutingReason",
    "TransformReason",
```

**Step 3: Verify exports**

Run: `.venv/bin/python -c "from elspeth.contracts import ConfigGateReason, PluginGateReason, RoutingReason; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/elspeth/contracts/__init__.py
git commit -m "$(cat <<'EOF'
chore(contracts): export ConfigGateReason and PluginGateReason

Add new RoutingReason variant types to public exports.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update Config Gate Executor

**Files:**
- Modify: `src/elspeth/engine/executors.py:739`

**Step 1: Add import**

Add to imports at top of `executors.py`:

```python
from elspeth.contracts.errors import ConfigGateReason
```

**Step 2: Update reason construction**

Change line 739 from:

```python
        reason = {"condition": gate_config.condition, "result": route_label}
```

To:

```python
        reason: ConfigGateReason = {
            "condition": gate_config.condition,
            "result": route_label,
        }
```

**Step 3: Run config gate tests**

Run: `.venv/bin/python -m pytest tests/engine/test_gate_executor.py -k "config" -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/engine/executors.py
git commit -m "$(cat <<'EOF'
feat(executor): use ConfigGateReason in config gate

Config gates now produce properly typed ConfigGateReason.
Type annotation ensures condition and result fields are present.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Remove cast() Calls in Executors

**Files:**
- Modify: `src/elspeth/engine/executors.py:858, 872`

**Step 1: Remove casts**

Change line 858 from:
```python
                reason=cast(RoutingReason, dict(action.reason)) if action.reason else None,
```
To:
```python
                reason=action.reason,
```

Change line 872 from:
```python
                reason=cast(RoutingReason, dict(action.reason)) if action.reason else None,
```
To:
```python
                reason=action.reason,
```

**Step 2: Check if cast import can be removed**

Run: `grep -n "cast(" src/elspeth/engine/executors.py`

If no other uses, remove `cast` from the typing imports.

**Step 3: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/engine/executors.py --no-error-summary`
Expected: No errors related to RoutingReason

**Step 4: Run gate executor tests**

Run: `.venv/bin/python -m pytest tests/engine/test_gate_executor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/executors.py
git commit -m "$(cat <<'EOF'
refactor(executor): remove redundant cast() for RoutingReason

Now that RoutingAction.reason is typed as RoutingReason | None,
the cast(RoutingReason, dict(action.reason)) is unnecessary.

Part of elspeth-rapid-5vc type safety audit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update Old RoutingReason Tests

**Files:**
- Modify: `tests/contracts/test_errors.py:33-47, 93-117`

**Step 1: Update schema introspection tests**

Replace `TestRoutingReasonSchema` class (lines 33-47):

```python
class TestRoutingReasonSchema:
    """Tests for RoutingReason union type.

    RoutingReason is a 2-variant discriminated union:
    - ConfigGateReason: condition + result
    - PluginGateReason: rule + matched_value + optional fields
    """

    def test_routing_reason_is_union_of_two_variants(self) -> None:
        """RoutingReason is a Union of ConfigGateReason and PluginGateReason."""
        from typing import get_args

        from elspeth.contracts import RoutingReason

        args = get_args(RoutingReason)
        assert len(args) == 2, f"Expected 2 variants, got {len(args)}: {args}"

    def test_variants_have_distinct_required_fields(self) -> None:
        """ConfigGateReason and PluginGateReason have different required fields."""
        from elspeth.contracts import ConfigGateReason, PluginGateReason

        # ConfigGateReason: condition, result
        assert "condition" in ConfigGateReason.__required_keys__
        assert "result" in ConfigGateReason.__required_keys__
        assert "rule" not in ConfigGateReason.__required_keys__

        # PluginGateReason: rule, matched_value
        assert "rule" in PluginGateReason.__required_keys__
        assert "matched_value" in PluginGateReason.__required_keys__
        assert "condition" not in PluginGateReason.__required_keys__
```

**Step 2: Update TestRoutingReason usage tests**

Replace `TestRoutingReason` class (lines 93-117):

```python
class TestRoutingReason:
    """Tests for RoutingReason variant construction."""

    def test_config_gate_reason_for_expression_gates(self) -> None:
        """ConfigGateReason captures expression evaluation."""
        from elspeth.contracts import ConfigGateReason

        reason: ConfigGateReason = {
            "condition": "row['score'] > 100",
            "result": "true",
        }
        assert reason["condition"] == "row['score'] > 100"
        assert reason["result"] == "true"

    def test_plugin_gate_reason_minimal(self) -> None:
        """PluginGateReason with only required fields."""
        from elspeth.contracts import PluginGateReason

        reason: PluginGateReason = {
            "rule": "value > threshold",
            "matched_value": 42,
        }
        assert reason["rule"] == "value > threshold"
        assert reason["matched_value"] == 42

    def test_plugin_gate_reason_with_threshold_fields(self) -> None:
        """PluginGateReason with optional threshold fields."""
        from elspeth.contracts import PluginGateReason

        reason: PluginGateReason = {
            "rule": "value > threshold",
            "matched_value": 42,
            "threshold": 10.0,
            "field": "score",
            "comparison": ">",
        }
        assert reason["threshold"] == 10.0
        assert reason["field"] == "score"
        assert reason["comparison"] == ">"
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/contracts/test_errors.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/contracts/test_errors.py
git commit -m "$(cat <<'EOF'
test(errors): update RoutingReason tests for 2-variant union

- Update schema tests for ConfigGateReason + PluginGateReason
- Verify variants have distinct required fields
- Add usage tests for both variants

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update Routing Tests

**Files:**
- Modify: `tests/contracts/test_routing.py`

**Step 1: Add typing import**

Add at top of file:

```python
from typing import Any
```

**Step 2: Replace MappingProxyType test with behavioral test**

Replace `test_reason_is_immutable` (lines 88-98):

```python
    def test_reason_mutation_prevented_by_deep_copy(self) -> None:
        """Mutating original dict should not affect stored reason (deep copy)."""
        from elspeth.contracts import RoutingAction

        original: dict[str, Any] = {"rule": "test", "matched_value": 42}
        action = RoutingAction.continue_(reason=original)  # type: ignore[arg-type]

        # Mutate original - should not affect action.reason
        original["rule"] = "mutated"
        assert action.reason["rule"] == "test"  # type: ignore[index]
```

**Step 3: Update test_continue_with_reason**

Change line 35 from:

```python
        action = RoutingAction.continue_(reason={"rule": "passed"})
```

To:

```python
        action = RoutingAction.continue_(reason={"rule": "passed", "matched_value": True})
```

And update assertion:

```python
        assert action.reason["rule"] == "passed"  # type: ignore[index]
```

**Step 4: Update test_route_with_reason**

Change lines 69-70 from:

```python
        action = RoutingAction.route("below", reason={"value": 500})
        assert dict(action.reason) == {"value": 500}
```

To:

```python
        action = RoutingAction.route("below", reason={
            "rule": "value below threshold",
            "matched_value": 500,
        })
        assert action.reason["matched_value"] == 500  # type: ignore[index]
```

**Step 5: Update test_fork_with_reason**

Change lines 85-86 from:

```python
        action = RoutingAction.fork_to_paths(["a", "b"], reason={"strategy": "parallel"})
        assert dict(action.reason) == {"strategy": "parallel"}
```

To:

```python
        action = RoutingAction.fork_to_paths(["a", "b"], reason={
            "rule": "parallel_strategy",
            "matched_value": "split",
        })
        assert action.reason["rule"] == "parallel_strategy"  # type: ignore[index]
```

**Step 6: Update test_reason_deep_copied**

Change lines 100-111:

```python
    def test_reason_deep_copied(self) -> None:
        """Mutating original nested dict should not affect frozen reason."""
        from elspeth.contracts import RoutingAction

        original: dict[str, Any] = {
            "rule": "test",
            "matched_value": {"nested": {"key": "value"}},
        }
        action = RoutingAction.continue_(reason=original)  # type: ignore[arg-type]

        # Mutate nested dict in original
        original["matched_value"]["nested"]["key"] = "modified"

        # Frozen reason should be unchanged (deep copy)
        assert action.reason["matched_value"]["nested"]["key"] == "value"  # type: ignore[index]
```

**Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/contracts/test_routing.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add tests/contracts/test_routing.py
git commit -m "$(cat <<'EOF'
test(routing): update tests for RoutingReason union type

- Replace MappingProxyType isinstance check with behavioral mutation test
- Update test data to use valid PluginGateReason (rule + matched_value)
- Add typing import for dict[str, Any]

Tests now verify behavior (mutation protection) not implementation detail.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update Test Fixtures

**Files:**
- Multiple test files with `reason={...}` patterns

**Step 1: Find all reason dict literals**

Run: `grep -rn 'reason={' tests/ --include='*.py'`

**Step 2: Update each to use valid PluginGateReason**

For each match, ensure the dict has `rule` and `matched_value`:

Change patterns like:
```python
reason={"threshold_exceeded": True, "value": row["value"]}
```
To:
```python
reason={"rule": "threshold_exceeded", "matched_value": row["value"]}
```

Change patterns like:
```python
reason={"confidence": 0.95}
```
To:
```python
reason={"rule": "confidence_check", "matched_value": 0.95}
```

Change patterns like:
```python
reason={"split_reason": "parallel processing"}
```
To:
```python
reason={"rule": "parallel processing", "matched_value": "split"}
```

**Step 3: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
test: update test fixtures to use valid RoutingReason variants

All test reason dicts now have required fields:
- rule: description of what logic fired
- matched_value: value that triggered routing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS

**Step 2: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/`
Expected: No new errors

**Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/`
Expected: No errors

**Step 4: Update bead**

```bash
bd close elspeth-rapid-5vc --reason="Implemented 2-variant discriminated union for RoutingReason (ConfigGateReason + PluginGateReason). Removed cast() calls. All tests pass."
```

---

## Summary

| Variant | Required Fields | Optional Fields | Used By |
|---------|-----------------|-----------------|---------|
| ConfigGateReason | condition, result | - | Config-driven gates (execute_config_gate) |
| PluginGateReason | rule, matched_value | threshold, field, comparison | Plugin gates (execute_gate) |

**Key Design Decisions:**
- **2 variants, not 5** - Matches actual production code paths
- **No reason_type discriminator** - Field presence distinguishes variants
- **PluginGateReason has optional threshold fields** - Accommodates common patterns without requiring them
- **No migration needed** - Pre-release, no production audit data exists

**Benefits:**
- Empty dict `{}` is INVALID (must have required fields)
- Each variant enforces its minimum auditability
- Compile-time type safety via mypy
- Cast calls eliminated from executors.py
