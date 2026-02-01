# Test Defect Report

## Summary

- `test_malicious_patterns` claims malicious inputs should be rejected, but it only asserts “no crash,” so security regressions can pass without failing the test.

## Severity

- Severity: major
- Priority: P1

## Category

- [Weak Assertions]

## Evidence

- `tests/engine/test_expression_parser.py:705` states rejection is required, but `tests/engine/test_expression_parser.py:707` only calls `_assert_safe_parse`, which does not assert rejection:
```python
def test_malicious_patterns(self, expression: str) -> None:
    """Known malicious patterns should be rejected, not crash."""
    self._assert_safe_parse(expression)
```
- `_assert_safe_parse` treats successful parsing/evaluation as a pass and only fails on unexpected exceptions, so malicious patterns can be accepted without failing the test (`tests/engine/test_expression_parser.py:682`, `tests/engine/test_expression_parser.py:683`, `tests/engine/test_expression_parser.py:690`):
```python
try:
    parser = ExpressionParser(expression)
    ...
except self.ALLOWED_EXCEPTIONS:
    pass
```

## Impact

- Security-critical patterns could be silently accepted without test failure, weakening the parser security gate.
- A regression that stops rejecting a known malicious construct would not be caught, creating false confidence in parser hardening.

## Root Cause Hypothesis

- A general-purpose “no crash” fuzz helper was reused for a test that intends strict rejection, but the assertion logic was never tightened for malicious patterns.

## Recommended Fix

- Enforce rejection in `test_malicious_patterns` by asserting `ExpressionSecurityError` at parse time:
```python
with pytest.raises(ExpressionSecurityError):
    ExpressionParser(expression)
```
- If any patterns are intentionally invalid syntax, split them into a separate list and assert `ExpressionSyntaxError` explicitly.
- Priority justification: this is a security boundary test; weak assertions here can allow malicious expressions to slip into production unchecked.
