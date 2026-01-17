# WP-09: Engine-Level Gates

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace plugin-based gates with config-driven engine-level gate evaluation using a safe expression parser.

**Architecture:** Gates become declarative config entries with condition expressions. The engine evaluates conditions directly - no plugin code executes. This provides better security (no arbitrary code) and simpler configuration (no plugin boilerplate).

**Tech Stack:** Python 3.12, AST module (safe parsing only, NOT eval)

---

## ⚠️ CRITICAL CONSTRAINT

**WP-02 and WP-09 MUST execute back-to-back.**

Before WP-02 deletes plugin gates, WP-09's engine gates must prove they can handle all scenarios the plugin gates handled. Task 0 verifies this equivalence.

---

## Fork Support

**Status:** Deferred to WP-07 (Fork Work Queue)

The plugin-protocol contract (lines 729-799) describes `fork_to` as a gate routing option. This is **intentionally not implemented** in WP-09 Phase 1 because:

1. Fork requires the work queue infrastructure from WP-07
2. Binary gates (`true`/`false`) cover 95% of use cases
3. Fork adds significant complexity to the expression parser

**Integration point for WP-07:**
```python
# In EngineGate.evaluate(), after condition evaluation:
if self._fork_config:
    return GateResult(
        row=row,
        action=RoutingAction.fork_to_paths(
            self._fork_config["paths"],
            reason={"condition": self._condition_source, "fork": True},
        ),
    )
```

This will be added when WP-07 implements fork handling in the processor.

---

## Design Overview

### Current Architecture (Plugin Gates - DELETED in WP-02)

```yaml
row_plugins:
  - plugin: threshold_gate
    type: gate
    options:
      field: score
      threshold: 0.85
    routes:
      above: continue
      below: review_sink
```

- Gate plugins implement `GateProtocol.evaluate(row, ctx) -> GateResult`
- Plugin code decides routing
- Audit records plugin execution

### New Architecture (Engine Gates)

```yaml
row_plugins:
  - plugin: gate                    # Reserved keyword, not a real plugin
    type: gate
    options:
      condition: "row['score'] >= 0.85"  # Expression evaluated by engine
    routes:
      "true": continue              # Condition true -> continue
      "false": review_sink          # Condition false -> route to sink
```

- Engine parses and evaluates condition expression
- NO plugin code involved
- Expression parser is security-hardened (rejects dangerous patterns)
- Audit records condition, input, result

---

## Security Model

**The expression parser MUST reject:**

| Pattern | Why Dangerous |
|---------|---------------|
| `__import__('os')` | Code execution |
| `eval(...)`, `exec(...)` | Arbitrary code |
| `lambda: ...` | Code injection |
| `[x for x in ...]` | Memory exhaustion |
| Attribute access beyond `row[...]` | Object traversal |
| Function calls except `row.get()` | Side effects |
| Assignment (`:=`) | State mutation |

**Allowed operations:**

| Category | Allowed |
|----------|---------|
| Field access | `row['field']`, `row.get('field', default)` |
| Comparisons | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Boolean | `and`, `or`, `not` |
| Membership | `in`, `not in` |
| Literals | `str`, `int`, `float`, `bool`, `None`, `list`, `dict` |
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%` (on literals/fields) |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/elspeth/engine/expression_parser.py` | Safe expression evaluation |
| `src/elspeth/engine/engine_gate.py` | EngineGate class (replaces plugin) |
| `tests/engine/test_expression_parser.py` | Security and functionality tests |
| `tests/engine/test_engine_gate.py` | EngineGate unit tests |
| `tests/engine/test_engine_gates_integration.py` | Integration tests |
| `tests/engine/test_expression_fuzz.py` | Security fuzz tests |
| `tests/core/test_gate_config.py` | Config validation tests |

## Files to Modify

| File | Change |
|------|--------|
| `src/elspeth/core/config.py` | Add engine gate validation to `RowPluginSettings` |
| `src/elspeth/core/dag.py` | Handle engine gate nodes with condition config |
| `src/elspeth/engine/orchestrator.py` | Instantiate `EngineGate` from settings |
| `src/elspeth/engine/__init__.py` | Export new components |

**Note:** `executors.py` does NOT need modification. `EngineGate` implements enough of the `GateProtocol` interface that the existing `GateExecutor` works unchanged. This is verified in Task 3.

---

## Task 0: Plugin Gate Equivalence Verification

**Goal:** Verify engine gates can handle all scenarios covered by deleted plugin gates.

**Files:**
- Read: `tests/plugins/gates/test_threshold_gate.py`
- Read: `tests/plugins/gates/test_filter_gate.py`
- Read: `tests/plugins/gates/test_field_match_gate.py`
- Create: `tests/engine/test_engine_gate_equivalence.py`

### Step 1: Review deleted plugin gate test scenarios

Read the three gate test files to extract key scenarios:

**ThresholdGate scenarios:**
- Value above threshold → "above" route
- Value below threshold → "below" route
- Value equal to threshold → configurable (inclusive option)
- Nested field access (`metrics.score`)
- Missing field → error
- Type coercion (string "75" → float)
- Audit reason includes threshold details

**FilterGate scenarios:**
- Greater than comparison → "pass"/"discard"
- Less than comparison
- Equals comparison
- Missing field → "discard" by default
- Reason includes field, value, condition

**FieldMatchGate scenarios:**
- Exact match → "match"/"no_match"
- Multiple allowed values
- Case sensitivity option
- Nested field access

### Step 2: Map plugin scenarios to expression syntax

| Plugin Config | Engine Expression |
|---------------|-------------------|
| `threshold_gate: field=score, threshold=50` | `row['score'] > 50` |
| `threshold_gate: field=score, threshold=50, inclusive=True` | `row['score'] >= 50` |
| `filter_gate: field=score, greater_than=0.5` | `row['score'] > 0.5` |
| `filter_gate: field=status, equals='active'` | `row['status'] == 'active'` |
| `field_match_gate: field=type, allowed=['A', 'B']` | `row['type'] in ['A', 'B']` |

### Step 3: Write equivalence test file

Create `tests/engine/test_engine_gate_equivalence.py`:

```python
"""Tests verifying engine gates cover all plugin gate scenarios.

These tests ensure WP-09 engine gates can handle everything the
deleted plugin gates (WP-02) could handle. This is the gate
between WP-02 and WP-09 back-to-back execution.
"""

import pytest

from elspeth.engine.engine_gate import EngineGate
from elspeth.plugins.context import PluginContext


@pytest.fixture
def ctx() -> PluginContext:
    return PluginContext(run_id="test", config={})


class TestThresholdGateEquivalence:
    """Engine gate equivalents of ThresholdGate scenarios."""

    def test_above_threshold(self, ctx: PluginContext) -> None:
        """Equivalent: threshold_gate(field=score, threshold=50) on score=75."""
        gate = EngineGate(
            condition="row['score'] > 50",
            routes={"true": "continue", "false": "review"},
        )
        result = gate.evaluate({"score": 75}, ctx)
        assert result.action.destinations == ("true",)

    def test_below_threshold(self, ctx: PluginContext) -> None:
        """Equivalent: threshold_gate(field=score, threshold=50) on score=25."""
        gate = EngineGate(
            condition="row['score'] > 50",
            routes={"true": "continue", "false": "review"},
        )
        result = gate.evaluate({"score": 25}, ctx)
        assert result.action.destinations == ("false",)

    def test_equal_to_threshold_exclusive(self, ctx: PluginContext) -> None:
        """Equivalent: threshold_gate(field=score, threshold=50) on score=50."""
        gate = EngineGate(
            condition="row['score'] > 50",  # Exclusive
            routes={"true": "above", "false": "below"},
        )
        result = gate.evaluate({"score": 50}, ctx)
        assert result.action.destinations == ("false",)  # Not above

    def test_equal_to_threshold_inclusive(self, ctx: PluginContext) -> None:
        """Equivalent: threshold_gate(inclusive=True) on score=50."""
        gate = EngineGate(
            condition="row['score'] >= 50",  # Inclusive
            routes={"true": "above", "false": "below"},
        )
        result = gate.evaluate({"score": 50}, ctx)
        assert result.action.destinations == ("true",)  # At or above

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Equivalent: threshold_gate(field=metrics.score)."""
        gate = EngineGate(
            condition="row['metrics']['score'] > 50",
            routes={"true": "above", "false": "below"},
        )
        result = gate.evaluate({"metrics": {"score": 75}}, ctx)
        assert result.action.destinations == ("true",)

    def test_reason_includes_condition(self, ctx: PluginContext) -> None:
        """Engine gates record condition and result in reason."""
        gate = EngineGate(
            condition="row['score'] > 50",
            routes={"true": "above", "false": "below"},
        )
        result = gate.evaluate({"score": 75}, ctx)
        assert "condition" in result.action.reason
        assert "result" in result.action.reason
        assert result.action.reason["result"] is True


class TestFilterGateEquivalence:
    """Engine gate equivalents of FilterGate scenarios."""

    def test_greater_than_pass(self, ctx: PluginContext) -> None:
        """Equivalent: filter_gate(field=score, greater_than=0.5) on score=0.8."""
        gate = EngineGate(
            condition="row['score'] > 0.5",
            routes={"true": "continue", "false": "discard"},
        )
        result = gate.evaluate({"score": 0.8}, ctx)
        assert result.action.destinations == ("true",)

    def test_greater_than_fail(self, ctx: PluginContext) -> None:
        """Equivalent: filter_gate(field=score, greater_than=0.5) on score=0.3."""
        gate = EngineGate(
            condition="row['score'] > 0.5",
            routes={"true": "continue", "false": "discard"},
        )
        result = gate.evaluate({"score": 0.3}, ctx)
        assert result.action.destinations == ("false",)

    def test_equals_check(self, ctx: PluginContext) -> None:
        """Equivalent: filter_gate(field=status, equals='active')."""
        gate = EngineGate(
            condition="row['status'] == 'active'",
            routes={"true": "continue", "false": "discard"},
        )
        assert gate.evaluate({"status": "active"}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"status": "inactive"}, ctx).action.destinations == ("false",)

    def test_less_than(self, ctx: PluginContext) -> None:
        """Equivalent: filter_gate(field=age, less_than=18)."""
        gate = EngineGate(
            condition="row['age'] < 18",
            routes={"true": "minor", "false": "adult"},
        )
        assert gate.evaluate({"age": 15}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"age": 25}, ctx).action.destinations == ("false",)


class TestFieldMatchGateEquivalence:
    """Engine gate equivalents of FieldMatchGate scenarios."""

    def test_value_in_allowed_list(self, ctx: PluginContext) -> None:
        """Equivalent: field_match_gate(field=type, allowed=['A', 'B'])."""
        gate = EngineGate(
            condition="row['type'] in ['A', 'B']",
            routes={"true": "match", "false": "no_match"},
        )
        assert gate.evaluate({"type": "A"}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"type": "C"}, ctx).action.destinations == ("false",)

    def test_exact_string_match(self, ctx: PluginContext) -> None:
        """Equivalent: field_match_gate(field=category, match='premium')."""
        gate = EngineGate(
            condition="row['category'] == 'premium'",
            routes={"true": "premium_flow", "false": "standard_flow"},
        )
        assert gate.evaluate({"category": "premium"}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"category": "basic"}, ctx).action.destinations == ("false",)


class TestComplexConditions:
    """Engine gates handle conditions plugin gates couldn't."""

    def test_compound_and_condition(self, ctx: PluginContext) -> None:
        """Engine gates support AND logic (no plugin equivalent)."""
        gate = EngineGate(
            condition="row['score'] > 50 and row['verified'] == True",
            routes={"true": "approved", "false": "review"},
        )
        assert gate.evaluate({"score": 75, "verified": True}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"score": 75, "verified": False}, ctx).action.destinations == ("false",)

    def test_compound_or_condition(self, ctx: PluginContext) -> None:
        """Engine gates support OR logic (no plugin equivalent)."""
        gate = EngineGate(
            condition="row['priority'] == 'high' or row['score'] > 90",
            routes={"true": "fast_track", "false": "normal"},
        )
        assert gate.evaluate({"priority": "high", "score": 50}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"priority": "low", "score": 95}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"priority": "low", "score": 50}, ctx).action.destinations == ("false",)
```

### Step 4: Run equivalence tests (will fail until later tasks)

Run: `pytest tests/engine/test_engine_gate_equivalence.py -v`

Expected: ImportError (EngineGate doesn't exist yet)

### Step 5: Commit test file

```bash
git add tests/engine/test_engine_gate_equivalence.py
git commit -m "$(cat <<'EOF'
test(engine): add engine gate equivalence tests

Verifies engine gates can handle all plugin gate scenarios.
Tests will pass after Task 3 implements EngineGate.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Create Expression Parser (Test First)

**Files:**
- Create: `tests/engine/test_expression_parser.py`
- Create: `src/elspeth/engine/expression_parser.py`

### Step 1: Write failing tests for allowed operations

Create `tests/engine/test_expression_parser.py`:

```python
"""Tests for safe expression parser.

The expression parser MUST:
1. Allow safe operations (comparisons, boolean logic, field access)
2. REJECT dangerous operations (imports, eval, exec, lambdas, comprehensions)
3. Provide clear error messages for rejected expressions
"""

import pytest


class TestExpressionParserAllowed:
    """Test allowed expression patterns."""

    def test_simple_comparison_greater_than(self) -> None:
        """Basic comparison: row['score'] >= 0.85"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['score'] >= 0.85")
        row = {"score": 0.9}
        assert evaluate(expr, row) is True

    def test_simple_comparison_less_than(self) -> None:
        """Basic comparison: row['value'] < 100"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['value'] < 100")
        assert evaluate(expr, {"value": 50}) is True
        assert evaluate(expr, {"value": 150}) is False

    def test_equality_comparison(self) -> None:
        """Equality: row['status'] == 'active'"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['status'] == 'active'")
        assert evaluate(expr, {"status": "active"}) is True
        assert evaluate(expr, {"status": "inactive"}) is False

    def test_boolean_and(self) -> None:
        """Boolean AND: row['a'] > 0 and row['b'] > 0"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['a'] > 0 and row['b'] > 0")
        assert evaluate(expr, {"a": 1, "b": 1}) is True
        assert evaluate(expr, {"a": 1, "b": 0}) is False
        assert evaluate(expr, {"a": 0, "b": 1}) is False

    def test_boolean_or(self) -> None:
        """Boolean OR: row['a'] > 0 or row['b'] > 0"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['a'] > 0 or row['b'] > 0")
        assert evaluate(expr, {"a": 1, "b": 0}) is True
        assert evaluate(expr, {"a": 0, "b": 1}) is True
        assert evaluate(expr, {"a": 0, "b": 0}) is False

    def test_boolean_not(self) -> None:
        """Boolean NOT: not row['blocked']"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("not row['blocked']")
        assert evaluate(expr, {"blocked": False}) is True
        assert evaluate(expr, {"blocked": True}) is False

    def test_membership_in(self) -> None:
        """Membership: row['category'] in ['A', 'B', 'C']"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['category'] in ['A', 'B', 'C']")
        assert evaluate(expr, {"category": "A"}) is True
        assert evaluate(expr, {"category": "D"}) is False

    def test_membership_not_in(self) -> None:
        """Membership: row['status'] not in ['banned', 'suspended']"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['status'] not in ['banned', 'suspended']")
        assert evaluate(expr, {"status": "active"}) is True
        assert evaluate(expr, {"status": "banned"}) is False

    def test_nested_field_access(self) -> None:
        """Nested: row['user']['role'] == 'admin'"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['user']['role'] == 'admin'")
        assert evaluate(expr, {"user": {"role": "admin"}}) is True
        assert evaluate(expr, {"user": {"role": "user"}}) is False

    def test_row_get_with_default(self) -> None:
        """row.get() with default: row.get('optional', 0) > 10"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row.get('optional', 0) > 10")
        assert evaluate(expr, {"optional": 20}) is True
        assert evaluate(expr, {}) is False  # Uses default 0

    def test_arithmetic_on_fields(self) -> None:
        """Arithmetic: row['a'] + row['b'] > 100"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['a'] + row['b'] > 100")
        assert evaluate(expr, {"a": 60, "b": 50}) is True
        assert evaluate(expr, {"a": 40, "b": 50}) is False

    def test_none_comparison(self) -> None:
        """None comparison: row['field'] is not None"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['field'] is not None")
        assert evaluate(expr, {"field": "value"}) is True
        assert evaluate(expr, {"field": None}) is False

    def test_complex_expression(self) -> None:
        """Complex: (row['score'] >= 0.8 and row['verified']) or row['override']"""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression(
            "(row['score'] >= 0.8 and row['verified']) or row['override']"
        )
        assert evaluate(expr, {"score": 0.9, "verified": True, "override": False}) is True
        assert evaluate(expr, {"score": 0.5, "verified": True, "override": True}) is True
        assert evaluate(expr, {"score": 0.5, "verified": True, "override": False}) is False


class TestExpressionParserSecurity:
    """Security tests - MUST ALL PASS to prevent code injection."""

    def test_reject_import(self) -> None:
        """SECURITY: __import__ must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="import"):
            parse_expression("__import__('os').system('rm -rf /')")

    def test_reject_eval(self) -> None:
        """SECURITY: eval() must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="eval"):
            parse_expression("eval('malicious')")

    def test_reject_exec(self) -> None:
        """SECURITY: exec() must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="exec"):
            parse_expression("exec('code')")

    def test_reject_lambda(self) -> None:
        """SECURITY: lambda must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="lambda"):
            parse_expression("(lambda: row['x'])()")

    def test_reject_list_comprehension(self) -> None:
        """SECURITY: list comprehensions must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="comprehension"):
            parse_expression("[x for x in row['items']]")

    def test_reject_dict_comprehension(self) -> None:
        """SECURITY: dict comprehensions must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="comprehension"):
            parse_expression("{k: v for k, v in row['items']}")

    def test_reject_generator_expression(self) -> None:
        """SECURITY: generator expressions must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="generator"):
            parse_expression("sum(x for x in row['items'])")

    def test_reject_walrus_operator(self) -> None:
        """SECURITY: assignment expressions (:=) must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="assignment"):
            parse_expression("(x := row['value']) > 0")

    def test_reject_attribute_access_beyond_row(self) -> None:
        """SECURITY: attribute access on non-row objects must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="attribute"):
            parse_expression("row['obj'].__class__.__bases__")

    def test_reject_arbitrary_function_calls(self) -> None:
        """SECURITY: arbitrary function calls must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="call"):
            parse_expression("len(row['items'])")

    def test_reject_open(self) -> None:
        """SECURITY: open() must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="open"):
            parse_expression("open('/etc/passwd').read()")

    def test_reject_globals_access(self) -> None:
        """SECURITY: globals()/locals() must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError):
            parse_expression("globals()['__builtins__']")

    def test_reject_dunder_methods(self) -> None:
        """SECURITY: __method__ access must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError, match="dunder"):
            parse_expression("row.__class__")

    def test_reject_getattr(self) -> None:
        """SECURITY: getattr() must be rejected."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError):
            parse_expression("getattr(row, 'secret')")


class TestExpressionParserErrorMessages:
    """Test that error messages are helpful for users."""

    def test_missing_field_error_message(self) -> None:
        """Missing field gives clear error with field name."""
        from elspeth.engine.expression_parser import (
            ExpressionEvaluationError,
            evaluate,
            parse_expression,
        )

        expr = parse_expression("row['nonexistent'] > 0")
        with pytest.raises(ExpressionEvaluationError, match="nonexistent"):
            evaluate(expr, {"other_field": 1})

    def test_syntax_error_message(self) -> None:
        """Syntax errors give position information."""
        from elspeth.engine.expression_parser import (
            ExpressionSyntaxError,
            parse_expression,
        )

        with pytest.raises(ExpressionSyntaxError, match="syntax"):
            parse_expression("row['field'] >")  # Incomplete expression


class TestExpressionParserEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string_field(self) -> None:
        """Empty string is falsy but not None."""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['name'] == ''")
        assert evaluate(expr, {"name": ""}) is True
        assert evaluate(expr, {"name": "Alice"}) is False

    def test_zero_is_falsy(self) -> None:
        """Zero is falsy in boolean context."""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['count'] or row['default']")
        assert evaluate(expr, {"count": 0, "default": 10}) == 10
        assert evaluate(expr, {"count": 5, "default": 10}) == 5

    def test_deeply_nested_field(self) -> None:
        """Five levels of nesting works."""
        from elspeth.engine.expression_parser import parse_expression, evaluate

        expr = parse_expression("row['a']['b']['c']['d']['e'] == 'deep'")
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        assert evaluate(expr, data) is True
```

### Step 2: Run tests to verify they fail

Run: `pytest tests/engine/test_expression_parser.py -v`

Expected: ImportError (module doesn't exist)

### Step 3: Write minimal implementation

Create `src/elspeth/engine/expression_parser.py`:

```python
"""Safe expression parser for engine-level gates.

SECURITY CRITICAL: This module parses and evaluates user-provided expressions.
It uses Python's ast module to parse expressions and a custom evaluator to
execute them safely WITHOUT using eval() or exec().

Allowed operations:
- Field access: row['field'], row.get('field', default)
- Comparisons: ==, !=, <, >, <=, >=, is, is not, in, not in
- Boolean: and, or, not
- Arithmetic: +, -, *, /, //, %
- Literals: str, int, float, bool, None, list, dict

REJECTED (raises ExpressionSecurityError at parse time):
- Imports: __import__
- Code execution: eval, exec, compile
- Lambdas and comprehensions
- Assignment expressions (:=)
- Attribute access beyond row[...] and row.get(...)
- Function calls except row.get()
"""

import ast
import operator
from dataclasses import dataclass
from typing import Any


class ExpressionSecurityError(Exception):
    """Raised when an expression contains disallowed constructs."""

    pass


class ExpressionSyntaxError(Exception):
    """Raised when an expression has invalid syntax."""

    pass


class ExpressionEvaluationError(Exception):
    """Raised when expression evaluation fails."""

    pass


@dataclass(frozen=True)
class ParsedExpression:
    """A parsed, validated expression ready for evaluation.

    The AST has been validated for security. Do not construct directly;
    use parse_expression().
    """

    source: str
    _ast: ast.Expression

    def __repr__(self) -> str:
        return f"ParsedExpression({self.source!r})"


# Dangerous names that must be rejected
_FORBIDDEN_NAMES = frozenset({
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "type",
    "isinstance",
    "issubclass",
    "callable",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "breakpoint",
    "memoryview",
    "help",
    "exit",
    "quit",
})

# Allowed comparison operators
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# Allowed binary operators
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

# Allowed unary operators
_UNARY_OPS = {
    ast.Not: operator.not_,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _SecurityValidator(ast.NodeVisitor):
    """AST visitor that validates expressions for security."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.errors.append(f"import statements not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.errors.append(f"import statements not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.errors.append("lambda expressions not allowed")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.errors.append("list comprehensions not allowed")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.errors.append("set comprehensions not allowed")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.errors.append("dict comprehensions not allowed")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.errors.append("generator expressions not allowed")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.errors.append("assignment expressions (:=) not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Only allow row.get
        if isinstance(node.value, ast.Name) and node.value.id == "row":
            if node.attr == "get":
                return  # Allowed
            elif node.attr.startswith("__"):
                self.errors.append(f"dunder attribute access not allowed: {node.attr}")
            else:
                self.errors.append(
                    f"attribute access on row not allowed: row.{node.attr} "
                    f"(use row['{node.attr}'] or row.get('{node.attr}', default))"
                )
        elif isinstance(node.value, ast.Attribute):
            # Nested attribute like row.x.y - only row.get() allowed
            self.errors.append("chained attribute access not allowed")
        else:
            self.errors.append(f"attribute access not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Only allow row.get()
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "row"
                and node.func.attr == "get"
            ):
                # Validate row.get() arguments
                if len(node.args) < 1 or len(node.args) > 2:
                    self.errors.append(
                        "row.get() requires 1 or 2 arguments: row.get('field') or row.get('field', default)"
                    )
                else:
                    # Visit arguments
                    for arg in node.args:
                        self.visit(arg)
                return

        # Check for forbidden function names
        if isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_NAMES:
                self.errors.append(f"call to '{node.func.id}' not allowed")
            else:
                self.errors.append(f"function call not allowed: {node.func.id}()")
        else:
            self.errors.append("function call not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _FORBIDDEN_NAMES:
            self.errors.append(f"forbidden name: {node.id}")
        elif node.id.startswith("__") and node.id.endswith("__"):
            self.errors.append(f"dunder name not allowed: {node.id}")
        elif node.id not in ("row", "True", "False", "None"):
            self.errors.append(
                f"unknown variable: {node.id} (only 'row' is available)"
            )


def parse_expression(source: str) -> ParsedExpression:
    """Parse and validate an expression.

    Args:
        source: Expression string like "row['score'] >= 0.85"

    Returns:
        ParsedExpression ready for evaluation

    Raises:
        ExpressionSyntaxError: If expression has invalid syntax
        ExpressionSecurityError: If expression contains disallowed constructs
    """
    if not source or not source.strip():
        raise ExpressionSyntaxError("empty expression")

    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ExpressionSyntaxError(
            f"syntax error in expression: {e.msg} at position {e.offset}"
        ) from e

    # Validate security
    validator = _SecurityValidator(source)
    validator.visit(tree)

    if validator.errors:
        raise ExpressionSecurityError(
            f"expression rejected: {'; '.join(validator.errors)}"
        )

    return ParsedExpression(source=source, _ast=tree)


def evaluate(expr: ParsedExpression, row: dict[str, Any]) -> Any:
    """Evaluate a parsed expression against a row.

    Args:
        expr: Parsed expression from parse_expression()
        row: Row data dict

    Returns:
        Evaluation result (typically bool for gate conditions)

    Raises:
        ExpressionEvaluationError: If evaluation fails (e.g., missing field)
    """
    return _Evaluator(row).visit(expr._ast.body)


class _Evaluator(ast.NodeVisitor):
    """Safe expression evaluator."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row

    def visit(self, node: ast.AST) -> Any:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ExpressionEvaluationError(
            f"unsupported expression type: {type(node).__name__}"
        )

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == "row":
            return self.row
        elif node.id == "True":
            return True
        elif node.id == "False":
            return False
        elif node.id == "None":
            return None
        else:
            raise ExpressionEvaluationError(f"unknown variable: {node.id}")

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Dict(self, node: ast.Dict) -> dict[Any, Any]:
        return {
            self.visit(k): self.visit(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        key = self.visit(node.slice)

        if not isinstance(value, dict):
            raise ExpressionEvaluationError(
                f"subscript access requires dict, got {type(value).__name__}"
            )

        try:
            return value[key]
        except KeyError:
            raise ExpressionEvaluationError(
                f"field '{key}' not found in row. Available fields: {list(value.keys())}"
            )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Only row.get is allowed (validated at parse time)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "row"
            and node.attr == "get"
        ):
            # Return a callable that performs dict.get
            return self.row.get
        raise ExpressionEvaluationError(f"attribute access not allowed: {node.attr}")

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)

        # func should be row.get (a bound method)
        if func != self.row.get:
            raise ExpressionEvaluationError("only row.get() calls allowed")

        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)

        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_func = _COMPARE_OPS.get(type(op))
            if op_func is None:
                raise ExpressionEvaluationError(
                    f"unsupported comparison: {type(op).__name__}"
                )
            if not op_func(left, right):
                return False
            left = right

        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                result = self.visit(value)
                if result:
                    return result
            return result
        else:
            raise ExpressionEvaluationError(
                f"unsupported boolean operator: {type(node.op).__name__}"
            )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ExpressionEvaluationError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        return op_func(operand)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise ExpressionEvaluationError(
                f"unsupported binary operator: {type(node.op).__name__}"
            )
        return op_func(left, right)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Ternary: value_if_true if condition else value_if_false"""
        if self.visit(node.test):
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_expression_parser.py -v`

Expected: All tests pass

### Step 5: Commit

```bash
git add src/elspeth/engine/expression_parser.py tests/engine/test_expression_parser.py
git commit -m "$(cat <<'EOF'
feat(engine): add safe expression parser for engine-level gates

Implements AST-based expression parsing without eval()/exec().
Rejects dangerous patterns: imports, lambdas, comprehensions, etc.
Allows safe operations: comparisons, boolean logic, field access.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add GateSettings to Config

**Files:**
- Modify: `src/elspeth/core/config.py`
- Create: `tests/core/test_gate_config.py`

### Step 1: Write failing test

Create `tests/core/test_gate_config.py`:

```python
"""Tests for engine gate configuration."""

import pytest


class TestGateSettings:
    """Test gate configuration validation."""

    def test_gate_config_with_condition(self) -> None:
        """Gate config requires condition expression."""
        from elspeth.core.config import RowPluginSettings

        config = RowPluginSettings(
            plugin="gate",
            type="gate",
            options={"condition": "row['score'] >= 0.85"},
            routes={"true": "continue", "false": "review"},
        )
        assert config.plugin == "gate"
        assert config.options["condition"] == "row['score'] >= 0.85"

    def test_gate_config_validates_condition_at_parse(self) -> None:
        """Gate condition is validated for security at config load."""
        from pydantic import ValidationError

        from elspeth.core.config import RowPluginSettings

        with pytest.raises(ValidationError, match="rejected"):
            RowPluginSettings(
                plugin="gate",
                type="gate",
                options={"condition": "eval('malicious')"},
                routes={"true": "continue", "false": "review"},
            )

    def test_gate_routes_require_true_and_false(self) -> None:
        """Gate routes must define 'true' and 'false' destinations."""
        from pydantic import ValidationError

        from elspeth.core.config import RowPluginSettings

        with pytest.raises(ValidationError, match="true"):
            RowPluginSettings(
                plugin="gate",
                type="gate",
                options={"condition": "row['x'] > 0"},
                routes={"yes": "continue"},  # Missing 'true' and 'false'
            )

    def test_gate_missing_condition_rejected(self) -> None:
        """Gate plugin requires condition in options."""
        from pydantic import ValidationError

        from elspeth.core.config import RowPluginSettings

        with pytest.raises(ValidationError, match="condition"):
            RowPluginSettings(
                plugin="gate",
                type="gate",
                options={},  # Missing condition
                routes={"true": "continue", "false": "review"},
            )
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_gate_config.py -v`

Expected: Fails (no validation exists yet)

### Step 3: Update RowPluginSettings with gate validation

Modify `src/elspeth/core/config.py`. Add a validator to `RowPluginSettings`:

```python
from elspeth.engine.expression_parser import (
    ExpressionSecurityError,
    parse_expression,
)

class RowPluginSettings(BaseModel):
    """Transform or gate plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name")
    type: Literal["transform", "gate"] = Field(
        default="transform",
        description="Plugin type: transform (pass-through) or gate (routing)",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    routes: dict[str, str] | None = Field(
        default=None,
        description="Gate routing map: 'true'/'false' -> sink_name or 'continue'",
    )

    @model_validator(mode="after")
    def validate_gate_config(self) -> "RowPluginSettings":
        """Validate gate-specific configuration."""
        if self.type != "gate":
            return self

        # Gate plugin name must be "gate" (reserved keyword)
        if self.plugin != "gate":
            raise ValueError(
                f"Engine gates must use plugin='gate', got '{self.plugin}'. "
                "Plugin-based gates were removed in WP-02."
            )

        # Condition is required
        condition = self.options.get("condition")
        if not condition:
            raise ValueError(
                "Engine gate requires 'condition' in options: "
                "options: {condition: \"row['field'] > value\"}"
            )

        # Validate condition expression at config time
        try:
            parse_expression(condition)
        except ExpressionSecurityError as e:
            raise ValueError(f"Gate condition rejected: {e}") from e

        # Routes must define 'true' and 'false'
        if not self.routes:
            raise ValueError(
                "Engine gate requires routes with 'true' and 'false' keys"
            )
        if "true" not in self.routes:
            raise ValueError("Engine gate routes must define 'true' destination")
        if "false" not in self.routes:
            raise ValueError("Engine gate routes must define 'false' destination")

        return self
```

### Step 4: Run tests

Run: `pytest tests/core/test_gate_config.py -v`

Expected: All pass

### Step 5: Commit

```bash
git add src/elspeth/core/config.py tests/core/test_gate_config.py
git commit -m "$(cat <<'EOF'
feat(config): add engine gate validation

Engine gates require:
- plugin="gate" (reserved keyword)
- condition expression (validated for security)
- routes with 'true' and 'false' keys

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create EngineGate Class

**Files:**
- Create: `src/elspeth/engine/engine_gate.py`
- Create: `tests/engine/test_engine_gate.py`

### Step 1: Write failing test

Create `tests/engine/test_engine_gate.py`:

```python
"""Tests for EngineGate - config-driven gate evaluation."""

import pytest


class TestEngineGate:
    """Test engine gate evaluation."""

    def test_engine_gate_evaluate_true(self) -> None:
        """Condition true -> 'true' route."""
        from elspeth.engine.engine_gate import EngineGate
        from elspeth.plugins.context import PluginContext

        gate = EngineGate(
            condition="row['score'] >= 0.85",
            routes={"true": "continue", "false": "review"},
        )
        ctx = PluginContext(run_id="test", config={})

        result = gate.evaluate({"score": 0.9}, ctx)

        assert result.action.destinations == ("true",)

    def test_engine_gate_evaluate_false(self) -> None:
        """Condition false -> 'false' route."""
        from elspeth.engine.engine_gate import EngineGate
        from elspeth.plugins.context import PluginContext

        gate = EngineGate(
            condition="row['score'] >= 0.85",
            routes={"true": "continue", "false": "review"},
        )
        ctx = PluginContext(run_id="test", config={})

        result = gate.evaluate({"score": 0.5}, ctx)

        assert result.action.destinations == ("false",)

    def test_engine_gate_complex_condition(self) -> None:
        """Complex boolean condition works."""
        from elspeth.engine.engine_gate import EngineGate
        from elspeth.plugins.context import PluginContext

        gate = EngineGate(
            condition="row['a'] > 0 and row['b'] > 0",
            routes={"true": "continue", "false": "reject"},
        )
        ctx = PluginContext(run_id="test", config={})

        assert gate.evaluate({"a": 1, "b": 1}, ctx).action.destinations == ("true",)
        assert gate.evaluate({"a": 0, "b": 1}, ctx).action.destinations == ("false",)

    def test_engine_gate_preserves_row(self) -> None:
        """Gate returns row unchanged."""
        from elspeth.engine.engine_gate import EngineGate
        from elspeth.plugins.context import PluginContext

        gate = EngineGate(
            condition="row['x'] > 0",
            routes={"true": "continue", "false": "reject"},
        )
        ctx = PluginContext(run_id="test", config={})
        row = {"x": 5, "y": "preserved"}

        result = gate.evaluate(row, ctx)

        assert result.row == row
        assert result.row["y"] == "preserved"

    def test_engine_gate_has_required_attributes(self) -> None:
        """EngineGate has attributes needed by GateExecutor."""
        from elspeth.contracts import Determinism
        from elspeth.engine.engine_gate import EngineGate

        gate = EngineGate(
            condition="row['x'] > 0",
            routes={"true": "continue", "false": "reject"},
        )

        assert gate.name == "engine_gate"
        assert gate.determinism == Determinism.DETERMINISTIC
        assert gate.plugin_version == "1.0.0"
        assert gate.node_id is None  # Set by orchestrator

    def test_engine_gate_evaluation_error_on_missing_field(self) -> None:
        """Missing field raises clear error."""
        from elspeth.engine.engine_gate import EngineGate
        from elspeth.engine.expression_parser import ExpressionEvaluationError
        from elspeth.plugins.context import PluginContext

        gate = EngineGate(
            condition="row['missing'] > 0",
            routes={"true": "continue", "false": "reject"},
        )
        ctx = PluginContext(run_id="test", config={})

        with pytest.raises(ExpressionEvaluationError, match="missing"):
            gate.evaluate({"other": 1}, ctx)
```

### Step 2: Create EngineGate implementation

Create `src/elspeth/engine/engine_gate.py`:

```python
"""EngineGate: Config-driven gate evaluation.

This replaces plugin-based gates with expression-evaluated gates.
The gate evaluates a condition expression and routes based on true/false.

Security: Condition expressions are validated at parse time (config load).
At evaluation time, only the pre-validated AST is executed.
"""

from typing import Any

from elspeth.contracts import Determinism
from elspeth.contracts.routing import RoutingAction
from elspeth.engine.expression_parser import (
    ParsedExpression,
    evaluate,
    parse_expression,
)
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult
from elspeth.plugins.schemas import DynamicSchema


class EngineGate:
    """Engine-level gate that evaluates condition expressions.

    Unlike plugin gates, EngineGate:
    - Does not execute arbitrary plugin code
    - Evaluates a pre-validated expression
    - Returns route labels "true" or "false"

    The executor resolves "true"/"false" to actual destinations
    via the routes config (e.g., "true" -> "continue", "false" -> "review_sink").
    """

    name: str = "engine_gate"
    input_schema = DynamicSchema
    output_schema = DynamicSchema
    determinism = Determinism.DETERMINISTIC
    plugin_version: str = "1.0.0"
    node_id: str | None = None  # Set by orchestrator

    def __init__(
        self,
        condition: str,
        routes: dict[str, str],
    ) -> None:
        """Initialize engine gate.

        Args:
            condition: Expression string (already validated at config time)
            routes: Route map with 'true' and 'false' keys
        """
        self._condition_source = condition
        self._condition: ParsedExpression = parse_expression(condition)
        self._routes = routes

    @property
    def condition(self) -> str:
        """The condition expression source (for audit trail)."""
        return self._condition_source

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate condition and return routing decision.

        Args:
            row: Input row data
            ctx: Plugin context (not used by engine gates)

        Returns:
            GateResult with route label "true" or "false"
        """
        result = evaluate(self._condition, row)

        # Convert to bool and select route
        route_label = "true" if result else "false"

        return GateResult(
            row=row,  # Gates pass through row unchanged
            action=RoutingAction.route(
                route_label,
                reason={"condition": self._condition_source, "result": bool(result)},
            ),
        )

    def close(self) -> None:
        """No cleanup needed."""
        pass

    def on_register(self, ctx: PluginContext) -> None:
        """No registration needed."""
        pass

    def on_start(self, ctx: PluginContext) -> None:
        """No start action needed."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """No completion action needed."""
        pass
```

### Step 3: Run tests

Run: `pytest tests/engine/test_engine_gate.py -v`

Expected: All pass

### Step 4: Commit

```bash
git add src/elspeth/engine/engine_gate.py tests/engine/test_engine_gate.py
git commit -m "$(cat <<'EOF'
feat(engine): add EngineGate for config-driven gate evaluation

EngineGate evaluates condition expressions and returns 'true'/'false'
route labels. Replaces plugin-based gates.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update DAG to Create Engine Gates

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Modify: `tests/core/test_dag.py` (add engine gate tests)

### Step 1: Update ExecutionGraph.from_config()

In `src/elspeth/core/dag.py`, modify the gate handling in `from_config()`:

```python
# In the transform chain building loop:
if is_gate:
    # Engine gates use "gate" as plugin name
    if plugin_config.plugin != "gate":
        raise GraphValidationError(
            f"Gate plugins must use plugin='gate', got '{plugin_config.plugin}'. "
            "Plugin-based gates were removed."
        )

    # Store condition in node config for engine gate creation
    condition = plugin_config.options.get("condition")
    if not condition:
        raise GraphValidationError(
            f"Engine gate at position {i} missing 'condition' in options"
        )

    graph.add_node(
        tid,
        node_type="gate",
        plugin_name="engine_gate",  # Mark as engine-handled
        config={
            "condition": condition,
            "routes": plugin_config.routes,
        },
    )
```

### Step 2: Add test for engine gate node creation

Add to `tests/core/test_dag.py`:

```python
def test_engine_gate_node_creation(self) -> None:
    """Engine gates create nodes with condition config."""
    from elspeth.core.config import (
        DatasourceSettings,
        ElspethSettings,
        RowPluginSettings,
        SinkSettings,
    )
    from elspeth.core.dag import ExecutionGraph

    settings = ElspethSettings(
        datasource=DatasourceSettings(plugin="csv"),
        row_plugins=[
            RowPluginSettings(
                plugin="gate",
                type="gate",
                options={"condition": "row['x'] > 0"},
                routes={"true": "continue", "false": "review"},
            ),
        ],
        sinks={"output": SinkSettings(plugin="csv"), "review": SinkSettings(plugin="csv")},
        output_sink="output",
    )

    graph = ExecutionGraph.from_config(settings)

    # Find gate node
    gate_nodes = [
        nid for nid in graph._graph.nodes
        if graph.get_node_info(nid).node_type == "gate"
    ]
    assert len(gate_nodes) == 1

    gate_info = graph.get_node_info(gate_nodes[0])
    assert gate_info.plugin_name == "engine_gate"
    assert gate_info.config["condition"] == "row['x'] > 0"
    assert gate_info.config["routes"]["true"] == "continue"
    assert gate_info.config["routes"]["false"] == "review"
```

### Step 3: Run tests

Run: `pytest tests/core/test_dag.py -v`

Expected: All pass

### Step 4: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): create engine gate nodes with condition config

ExecutionGraph stores gate condition and routes in node config
for engine-level evaluation.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Orchestrator to Instantiate Engine Gates

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`

### Step 1: Modify orchestrator to create EngineGate instances

In `orchestrator.py`, when building the transforms list from settings:

```python
from elspeth.engine.engine_gate import EngineGate

# In the run() method, when creating plugin instances from config:
def _create_transforms_from_settings(
    self,
    settings: "ElspethSettings",
) -> list[RowPlugin]:
    """Create transform/gate instances from settings."""
    transforms: list[RowPlugin] = []

    for plugin_config in settings.row_plugins:
        if plugin_config.type == "gate":
            # Engine gate - create directly from config
            gate = EngineGate(
                condition=plugin_config.options["condition"],
                routes=plugin_config.routes or {},
            )
            transforms.append(gate)
        else:
            # Regular transform - instantiate from plugin registry
            # (existing logic)
            ...

    return transforms
```

### Step 2: Run existing orchestrator tests

Run: `pytest tests/engine/test_orchestrator.py -v`

Expected: Tests using inline gate classes still pass (they create BaseGate subclasses)

### Step 3: Commit

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): instantiate EngineGate from settings

Creates EngineGate instances for type="gate" plugins.
Engine gates are evaluated directly, not via plugin system.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update Route Resolution for Engine Gates

**Files:**
- Modify: `src/elspeth/core/dag.py`

### Step 1: Update route resolution map for true/false routes

Engine gates use "true" and "false" as route labels. The `_route_resolution_map`
needs to map `(gate_node_id, "true")` and `(gate_node_id, "false")` to destinations.

In `ExecutionGraph.from_config()`:

```python
if is_gate and plugin_config.routes:
    for route_label, target in plugin_config.routes.items():
        # Store route resolution: (gate_node, route_label) -> target
        graph._route_resolution_map[(tid, route_label)] = target

        if target == "continue":
            continue  # Not a sink route
        if target not in sink_ids:
            raise GraphValidationError(
                f"Gate routes '{route_label}' to unknown sink '{target}'"
            )
        # Edge for sink route
        graph.add_edge(tid, sink_ids[target], label=route_label, mode=RoutingMode.MOVE)
```

This already handles arbitrary route labels, including "true" and "false".

### Step 2: Run tests to verify

Run: `pytest tests/core/test_dag.py tests/engine/test_orchestrator.py -v`

Expected: Pass

### Step 3: Commit (if changes made)

```bash
git add src/elspeth/core/dag.py
git commit -m "$(cat <<'EOF'
feat(dag): support 'true'/'false' route labels for engine gates

Route resolution handles engine gate labels correctly.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Integration Tests

**Files:**
- Create: `tests/engine/test_engine_gates_integration.py`

### Step 1: Write integration tests

```python
"""Integration tests for engine-level gates."""


class TestEngineGateIntegration:
    """End-to-end tests for engine gates in pipelines."""

    def test_engine_gate_routes_to_sink(self) -> None:
        """Engine gate routes rows to configured sink."""
        # Setup: Create settings with engine gate
        # Create pipeline
        # Process rows
        # Verify routing to correct sinks

    def test_engine_gate_continues_on_true(self) -> None:
        """Engine gate continues to next transform on 'true' -> 'continue'."""
        # Verify row proceeds to next transform

    def test_engine_gate_records_audit_trail(self) -> None:
        """Engine gate evaluation recorded in audit trail."""
        # Verify node_states records condition evaluation

    def test_multiple_engine_gates_in_pipeline(self) -> None:
        """Multiple engine gates work in sequence."""
        # Two gates filtering rows
```

### Step 2: Implement and run

Run: `pytest tests/engine/test_engine_gates_integration.py -v`

### Step 3: Commit

```bash
git add tests/engine/test_engine_gates_integration.py
git commit -m "$(cat <<'EOF'
test(engine): add engine gate integration tests

Verifies end-to-end gate behavior in pipelines.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Security Fuzz Testing

**Files:**
- Create: `tests/engine/test_expression_fuzz.py`

### Step 1: Write fuzz tests

```python
"""Fuzz tests for expression parser security.

These tests throw random/malicious inputs at the parser to ensure
it never executes arbitrary code.
"""

import pytest


class TestExpressionParserFuzz:
    """Fuzz testing for security."""

    @pytest.mark.parametrize(
        "malicious",
        [
            "__import__('os').system('echo pwned')",
            "eval('print(1)')",
            "exec('import sys')",
            "(lambda: __import__('os'))()",
            "[x for x in __builtins__]",
            "globals()['__builtins__']['eval']('1')",
            "getattr(__builtins__, 'eval')('1')",
            "type('', (), {'__init__': lambda s: None})()",
            "''.__class__.__mro__[1].__subclasses__()",
            "().__class__.__bases__[0].__subclasses__()",
            "open('/etc/passwd').read()",
            "breakpoint()",
            "__builtins__.__dict__['eval']('1')",
            "vars()['__builtins__']",
            "dir()",
        ],
    )
    def test_reject_malicious_expressions(self, malicious: str) -> None:
        """All malicious expressions must be rejected at parse time."""
        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            parse_expression,
        )

        with pytest.raises(ExpressionSecurityError):
            parse_expression(malicious)

    def test_random_garbage_no_crash(self) -> None:
        """Random bytes don't crash the parser."""
        import random
        import string

        from elspeth.engine.expression_parser import (
            ExpressionSecurityError,
            ExpressionSyntaxError,
            parse_expression,
        )

        for _ in range(1000):
            garbage = "".join(
                random.choices(string.printable, k=random.randint(1, 100))
            )
            try:
                parse_expression(garbage)
            except (ExpressionSecurityError, ExpressionSyntaxError):
                pass  # Expected
            except Exception as e:
                pytest.fail(f"Unexpected exception for '{garbage}': {e}")
```

### Step 2: Run fuzz tests

Run: `pytest tests/engine/test_expression_fuzz.py -v`

Expected: All pass (no crashes, no code execution)

### Step 3: Commit

```bash
git add tests/engine/test_expression_fuzz.py
git commit -m "$(cat <<'EOF'
test(security): add fuzz tests for expression parser

Verifies parser rejects malicious inputs and doesn't crash on garbage.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update Exports and Final Cleanup

**Files:**
- Modify: `src/elspeth/engine/__init__.py`

### Step 1: Add exports

```python
from elspeth.engine.engine_gate import EngineGate
from elspeth.engine.expression_parser import (
    ExpressionEvaluationError,
    ExpressionSecurityError,
    ExpressionSyntaxError,
    ParsedExpression,
    evaluate,
    parse_expression,
)

__all__ = [
    # ... existing exports ...
    "EngineGate",
    "ParsedExpression",
    "parse_expression",
    "evaluate",
    "ExpressionSecurityError",
    "ExpressionSyntaxError",
    "ExpressionEvaluationError",
]
```

### Step 2: Run full test suite

Run: `pytest tests/ -v`

Expected: All tests pass

### Step 3: Commit

```bash
git add src/elspeth/engine/__init__.py
git commit -m "$(cat <<'EOF'
feat(engine): export engine gate components

Adds EngineGate and expression parser to public API.

Part of WP-09: Engine-Level Gates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

- [ ] Expression parser passes all security tests
- [ ] Expression parser passes fuzz tests (1000+ random inputs)
- [ ] GateSettings validates condition at config time
- [ ] EngineGate evaluates conditions correctly
- [ ] ExecutionGraph creates engine gate nodes
- [ ] Orchestrator instantiates EngineGate from settings
- [ ] Route resolution handles 'true'/'false' labels
- [ ] Integration tests pass
- [ ] All existing tests still pass
- [ ] No `eval()` or `exec()` anywhere in expression parser
- [ ] mypy passes on all new files

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/engine/expression_parser.py` | CREATE | Safe AST-based expression parser |
| `src/elspeth/engine/engine_gate.py` | CREATE | EngineGate class |
| `tests/engine/test_expression_parser.py` | CREATE | Parser tests (security + functionality) |
| `tests/engine/test_engine_gate.py` | CREATE | EngineGate unit tests |
| `tests/engine/test_engine_gates_integration.py` | CREATE | Integration tests |
| `tests/engine/test_expression_fuzz.py` | CREATE | Security fuzz tests |
| `tests/core/test_gate_config.py` | CREATE | Config validation tests |
| `src/elspeth/core/config.py` | MODIFY | Add gate validation |
| `src/elspeth/core/dag.py` | MODIFY | Handle engine gate nodes |
| `src/elspeth/engine/orchestrator.py` | MODIFY | Instantiate EngineGate |
| `src/elspeth/engine/__init__.py` | MODIFY | Export new components |

---

## Security Guarantee

After WP-09 is complete:

1. **No arbitrary code execution** — Expression parser uses AST, not eval/exec
2. **Parse-time rejection** — Dangerous patterns rejected before storage
3. **Evaluation-time safety** — Only pre-validated AST is executed
4. **Audit trail** — Condition expressions recorded for traceability

---

## Migration Notes

**Config changes required:**

```yaml
# OLD (plugin gates - no longer supported)
row_plugins:
  - plugin: threshold_gate
    type: gate
    options: {field: score, threshold: 0.85}
    routes: {above: continue, below: review}

# NEW (engine gates)
row_plugins:
  - plugin: gate
    type: gate
    options:
      condition: "row['score'] >= 0.85"
    routes:
      "true": continue
      "false": review
```

**Key differences:**
1. `plugin: gate` (reserved keyword, not a plugin name)
2. `condition` expression instead of plugin-specific options
3. Routes use `"true"`/`"false"` instead of plugin-defined labels
