## Summary

ExpressionParser.is_boolean_expression classifies any BoolOp as boolean for config validation, but gate execution routes based on the runtime result type, so non-boolean BoolOps can pass validation yet fail or mis-route at runtime.

## Severity

- Severity: major
- Priority: P1

## Anti-Pattern Classification

[Select one primary category:]

- [ ] Parallel Type Evolution (duplicate definitions of same concept)
- [ ] Impedance Mismatch (complex translation at boundaries)
- [ ] Leaky Abstraction (implementation details cross boundaries)
- [x] Contract Violation (undocumented assumptions)
- [ ] Shared Mutable State (unclear ownership)
- [ ] God Object (excessive coupling through large context)
- [ ] Stringly-Typed Interface (magic strings instead of types)
- [ ] Missing Facade (complex subsystem without simple interface)
- [ ] Protocol Drift (versions incompatible)
- [ ] Callback Hell (complex async chains)
- [ ] Missing Error Translation (low-level errors leak)
- [ ] Implicit State Dependencies (call order matters but not enforced)

## Seam Location

**Boundary:** core (config) ↔ engine (gate execution)

**Integration Point:** boolean condition validation (`ExpressionParser.is_boolean_expression`) ↔ runtime route selection (`ExpressionParser.evaluate` result typing)

## Evidence

[MUST include specific file paths and line numbers showing both sides of the seam]

### Side A: engine/expression_parser.py

```python
# src/elspeth/engine/expression_parser.py:420
420     def _is_boolean_node(self, node: ast.expr) -> bool:
421         """Recursively check if an AST node returns a boolean."""
422         # Comparisons always return bool
423         if isinstance(node, ast.Compare):
424             return True
425
426         # Boolean operators (and, or) always return truthy/falsy value
427         # Note: In Python, `x and y` returns y if x is truthy, not necessarily bool
428         # But for gate routing purposes, we treat this as boolean-ish
429         if isinstance(node, ast.BoolOp):
430             return True
```

### Side B: core/config.py

```python
# src/elspeth/core/config.py:264
264     @model_validator(mode="after")
265     def validate_boolean_routes(self) -> "GateSettings":
266         """Validate route labels match the condition's return type.
267
268         Boolean expressions (comparisons, and/or, not) must use "true"/"false"
269         as route labels. Using labels like "above"/"below" for a condition like
270         `row['amount'] > 1000` is a config error - the expression evaluates to
271         True/False, not "above"/"below".
272         """
273         from elspeth.engine.expression_parser import ExpressionParser
274
275         parser = ExpressionParser(self.condition)
276         if parser.is_boolean_expression():
277             route_labels = set(self.routes.keys())
278             expected_labels = {"true", "false"}
```

### Coupling Evidence: runtime route selection depends on actual result type

```python
# src/elspeth/engine/executors.py:553
553         # Convert evaluation result to route label
554         if isinstance(eval_result, bool):
555             route_label = "true" if eval_result else "false"
556         elif isinstance(eval_result, str):
557             route_label = eval_result
558         else:
559             # Unexpected result type - convert to string
560             route_label = str(eval_result)
```

## Root Cause Hypothesis

ExpressionParser’s boolean classification is intentionally broad (“boolean-ish”) while config validation treats it as a strict boolean contract, and execution uses runtime result typing without coercion; the shared contract between validation and execution is not enforced.

## Recommended Fix

1. Tighten boolean classification in `ExpressionParser._is_boolean_node` so `ast.BoolOp` returns True only when all operands are boolean expressions (recursive check).
2. Alternatively, if “boolean-ish” is intended, enforce it in execution by coercing `eval_result` with `bool()` when `parser.is_boolean_expression()` is True, and update validation text accordingly.
3. Add tests covering `and/or` with non-boolean operands to ensure validation and runtime routing agree.

## Impact Assessment

- **Coupling Level:** Medium - config validation depends on ExpressionParser’s static analysis semantics.
- **Maintainability:** Medium - changes to boolean detection can silently affect routing rules.
- **Type Safety:** Low - runtime routing depends on dynamic types.
- **Breaking Change Risk:** Medium - stricter boolean detection or coercion could change accepted configs or routing behavior.

## Related Seams

`src/elspeth/engine/triggers.py`
---
Template Version: 1.0
