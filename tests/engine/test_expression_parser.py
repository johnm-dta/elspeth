# tests/engine/test_expression_parser.py
"""Tests for safe expression parser."""

import contextlib
import random
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.engine.expression_parser import (
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)


class TestExpressionParserBasicOperations:
    """Test basic allowed operations."""

    def test_simple_equality(self) -> None:
        parser = ExpressionParser("row['status'] == 'active'")
        assert parser.evaluate({"status": "active"}) is True
        assert parser.evaluate({"status": "inactive"}) is False

    def test_numeric_comparison(self) -> None:
        parser = ExpressionParser("row['confidence'] >= 0.85")
        assert parser.evaluate({"confidence": 0.9}) is True
        assert parser.evaluate({"confidence": 0.85}) is True
        assert parser.evaluate({"confidence": 0.8}) is False

    def test_less_than(self) -> None:
        parser = ExpressionParser("row['count'] < 10")
        assert parser.evaluate({"count": 5}) is True
        assert parser.evaluate({"count": 10}) is False
        assert parser.evaluate({"count": 15}) is False

    def test_greater_than(self) -> None:
        parser = ExpressionParser("row['value'] > 100")
        assert parser.evaluate({"value": 150}) is True
        assert parser.evaluate({"value": 100}) is False
        assert parser.evaluate({"value": 50}) is False

    def test_less_than_or_equal(self) -> None:
        parser = ExpressionParser("row['priority'] <= 3")
        assert parser.evaluate({"priority": 2}) is True
        assert parser.evaluate({"priority": 3}) is True
        assert parser.evaluate({"priority": 4}) is False

    def test_not_equal(self) -> None:
        parser = ExpressionParser("row['status'] != 'deleted'")
        assert parser.evaluate({"status": "active"}) is True
        assert parser.evaluate({"status": "deleted"}) is False


class TestExpressionParserBooleanOperations:
    """Test boolean and/or/not operations."""

    def test_and_operator(self) -> None:
        parser = ExpressionParser("row['status'] == 'active' and row['balance'] > 0")
        assert parser.evaluate({"status": "active", "balance": 100}) is True
        assert parser.evaluate({"status": "active", "balance": 0}) is False
        assert parser.evaluate({"status": "inactive", "balance": 100}) is False
        assert parser.evaluate({"status": "inactive", "balance": 0}) is False

    def test_or_operator(self) -> None:
        parser = ExpressionParser(
            "row['status'] == 'active' or row['override'] == True"
        )
        assert parser.evaluate({"status": "active", "override": False}) is True
        assert parser.evaluate({"status": "inactive", "override": True}) is True
        assert parser.evaluate({"status": "inactive", "override": False}) is False

    def test_not_operator(self) -> None:
        parser = ExpressionParser("not row['disabled']")
        assert parser.evaluate({"disabled": False}) is True
        assert parser.evaluate({"disabled": True}) is False

    def test_complex_boolean_expression(self) -> None:
        parser = ExpressionParser(
            "(row['status'] == 'active' or row['status'] == 'pending') "
            "and row['score'] >= 0.5"
        )
        assert parser.evaluate({"status": "active", "score": 0.7}) is True
        assert parser.evaluate({"status": "pending", "score": 0.6}) is True
        assert parser.evaluate({"status": "active", "score": 0.3}) is False
        assert parser.evaluate({"status": "deleted", "score": 0.9}) is False


class TestExpressionParserMembership:
    """Test membership operations (in, not in)."""

    def test_in_list(self) -> None:
        parser = ExpressionParser("row['status'] in ['active', 'pending']")
        assert parser.evaluate({"status": "active"}) is True
        assert parser.evaluate({"status": "pending"}) is True
        assert parser.evaluate({"status": "deleted"}) is False

    def test_not_in_list(self) -> None:
        parser = ExpressionParser("row['category'] not in ['spam', 'trash']")
        assert parser.evaluate({"category": "inbox"}) is True
        assert parser.evaluate({"category": "spam"}) is False

    def test_in_tuple(self) -> None:
        parser = ExpressionParser("row['code'] in (1, 2, 3)")
        assert parser.evaluate({"code": 2}) is True
        assert parser.evaluate({"code": 5}) is False

    def test_in_set(self) -> None:
        parser = ExpressionParser("row['tag'] in {'a', 'b', 'c'}")
        assert parser.evaluate({"tag": "b"}) is True
        assert parser.evaluate({"tag": "d"}) is False


class TestExpressionParserRowGet:
    """Test row.get() method access."""

    def test_row_get_basic(self) -> None:
        parser = ExpressionParser("row.get('status') == 'active'")
        assert parser.evaluate({"status": "active"}) is True

    def test_row_get_missing_key_returns_none(self) -> None:
        parser = ExpressionParser("row.get('missing') is None")
        assert parser.evaluate({}) is True

    def test_row_get_with_default(self) -> None:
        parser = ExpressionParser("row.get('status', 'unknown') == 'unknown'")
        assert parser.evaluate({}) is True
        assert parser.evaluate({"status": "active"}) is False

    def test_row_get_with_default_when_key_exists(self) -> None:
        parser = ExpressionParser("row.get('status', 'default') == 'active'")
        assert parser.evaluate({"status": "active"}) is True


class TestExpressionParserNoneChecks:
    """Test is/is not for None checks."""

    def test_is_none(self) -> None:
        parser = ExpressionParser("row.get('optional') is None")
        assert parser.evaluate({}) is True
        assert parser.evaluate({"optional": None}) is True
        assert parser.evaluate({"optional": "value"}) is False

    def test_is_not_none(self) -> None:
        parser = ExpressionParser("row.get('required') is not None")
        assert parser.evaluate({"required": "value"}) is True
        assert parser.evaluate({"required": 0}) is True  # 0 is not None
        assert parser.evaluate({}) is False


class TestExpressionParserArithmetic:
    """Test arithmetic operations."""

    def test_addition(self) -> None:
        parser = ExpressionParser("row['a'] + row['b'] > 10")
        assert parser.evaluate({"a": 5, "b": 6}) is True
        assert parser.evaluate({"a": 5, "b": 4}) is False

    def test_subtraction(self) -> None:
        parser = ExpressionParser("row['x'] - row['y'] == 5")
        assert parser.evaluate({"x": 10, "y": 5}) is True

    def test_multiplication(self) -> None:
        parser = ExpressionParser("row['qty'] * row['price'] >= 100")
        assert parser.evaluate({"qty": 5, "price": 25}) is True
        assert parser.evaluate({"qty": 2, "price": 10}) is False

    def test_division(self) -> None:
        parser = ExpressionParser("row['total'] / row['count'] > 5")
        assert parser.evaluate({"total": 30, "count": 5}) is True

    def test_floor_division(self) -> None:
        parser = ExpressionParser("row['value'] // 10 == 4")
        assert parser.evaluate({"value": 45}) is True
        assert parser.evaluate({"value": 49}) is True
        assert parser.evaluate({"value": 50}) is False

    def test_modulo(self) -> None:
        parser = ExpressionParser("row['number'] % 2 == 0")
        assert parser.evaluate({"number": 4}) is True
        assert parser.evaluate({"number": 5}) is False

    def test_unary_minus(self) -> None:
        parser = ExpressionParser("-row['value'] < 0")
        assert parser.evaluate({"value": 5}) is True
        assert parser.evaluate({"value": -5}) is False


class TestExpressionParserTernary:
    """Test ternary (conditional) expressions."""

    def test_ternary_true_branch(self) -> None:
        parser = ExpressionParser("'high' if row['score'] >= 0.8 else 'low'")
        assert parser.evaluate({"score": 0.9}) == "high"
        assert parser.evaluate({"score": 0.5}) == "low"

    def test_ternary_in_comparison(self) -> None:
        parser = ExpressionParser(
            "(row.get('priority', 'normal') if row.get('urgent') else 'low') == 'high'"
        )
        assert parser.evaluate({"urgent": True, "priority": "high"}) is True
        assert parser.evaluate({"urgent": False, "priority": "high"}) is False


class TestExpressionParserComparisonChains:
    """Test comparison chains."""

    def test_chained_comparison(self) -> None:
        parser = ExpressionParser("0 < row['value'] < 100")
        assert parser.evaluate({"value": 50}) is True
        assert parser.evaluate({"value": 0}) is False
        assert parser.evaluate({"value": 100}) is False
        assert parser.evaluate({"value": -5}) is False

    def test_double_equals_chain(self) -> None:
        parser = ExpressionParser("row['a'] == row['b'] == 1")
        assert parser.evaluate({"a": 1, "b": 1}) is True
        assert parser.evaluate({"a": 1, "b": 2}) is False


class TestExpressionParserSecurityRejections:
    """Test that forbidden constructs are rejected at parse time."""

    def test_reject_import(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden name"):
            ExpressionParser("__import__('os')")

    def test_reject_eval(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden"):
            ExpressionParser("eval('malicious')")

    def test_reject_exec(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden"):
            ExpressionParser("exec('malicious')")

    def test_reject_compile(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden"):
            ExpressionParser("compile('code', 'file', 'exec')")

    def test_reject_lambda(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Lambda expressions"):
            ExpressionParser("(lambda: True)()")

    def test_reject_list_comprehension(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="List comprehensions"):
            ExpressionParser("[x for x in range(10)]")

    def test_reject_dict_comprehension(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Dict comprehensions"):
            ExpressionParser("{k: v for k, v in items}")

    def test_reject_set_comprehension(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Set comprehensions"):
            ExpressionParser("{x for x in items}")

    def test_reject_generator_expression(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Generator expressions"):
            ExpressionParser("list(x for x in range(10))")

    def test_reject_attribute_access_dunder(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden row attribute"):
            ExpressionParser("row.__class__")

    def test_reject_attribute_access_arbitrary(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden row attribute"):
            ExpressionParser("row.items()")

    def test_reject_assignment_expression(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Assignment expressions"):
            ExpressionParser("(x := 5)")

    def test_reject_fstring(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="F-string"):
            ExpressionParser("f\"value: {row['x']}\"")

    def test_reject_arbitrary_function_call(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden function call"):
            ExpressionParser("len(row)")

    def test_reject_method_call_not_get(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden row attribute"):
            ExpressionParser("row.keys()")

    def test_reject_builtin_access(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden name"):
            ExpressionParser("open('/etc/passwd')")

    def test_reject_arbitrary_name(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden name"):
            ExpressionParser("some_var == 'value'")

    def test_reject_row_get_too_few_args(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="requires 1 or 2 arguments"):
            ExpressionParser("row.get()")

    def test_reject_row_get_too_many_args(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="requires 1 or 2 arguments"):
            ExpressionParser("row.get('a', 'b', 'c')")

    def test_reject_row_get_with_kwargs(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="keyword arguments"):
            ExpressionParser("row.get(key='field')")

    def test_reject_starred_expression(self) -> None:
        """Starred expressions (*x) must be rejected at parse time."""
        with pytest.raises(ExpressionSecurityError, match="Starred expressions"):
            ExpressionParser("[*row['items']]")

    def test_reject_starred_expression_in_tuple(self) -> None:
        """Starred expressions in tuples must be rejected."""
        with pytest.raises(ExpressionSecurityError, match="Starred expressions"):
            ExpressionParser("(*row['items'],)")

    def test_reject_dict_spread(self) -> None:
        """Dict spread (**x) must be rejected at parse time."""
        with pytest.raises(ExpressionSecurityError, match="Dict spread"):
            ExpressionParser("{**row['data']}")

    def test_reject_dict_spread_mixed(self) -> None:
        """Dict spread mixed with regular keys must be rejected."""
        with pytest.raises(ExpressionSecurityError, match="Dict spread"):
            ExpressionParser("{'key': 1, **row['data']}")


class TestExpressionParserSyntaxErrors:
    """Test that syntax errors are handled correctly."""

    def test_invalid_syntax(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Invalid syntax"):
            ExpressionParser("row['field ==")

    def test_incomplete_expression(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Invalid syntax"):
            ExpressionParser("row['field'] ==")

    def test_mismatched_parens(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Invalid syntax"):
            ExpressionParser("(row['field'] == 'value'")


class TestExpressionParserEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_comparison(self) -> None:
        parser = ExpressionParser("row['name'] == ''")
        assert parser.evaluate({"name": ""}) is True
        assert parser.evaluate({"name": "value"}) is False

    def test_zero_comparison(self) -> None:
        parser = ExpressionParser("row['count'] == 0")
        assert parser.evaluate({"count": 0}) is True
        assert parser.evaluate({"count": 1}) is False

    def test_false_boolean_comparison(self) -> None:
        parser = ExpressionParser("row['flag'] == False")
        assert parser.evaluate({"flag": False}) is True
        assert parser.evaluate({"flag": True}) is False

    def test_nested_subscript(self) -> None:
        parser = ExpressionParser("row['data']['nested'] == 'value'")
        assert parser.evaluate({"data": {"nested": "value"}}) is True

    def test_expression_property(self) -> None:
        parser = ExpressionParser("row['x'] == 1")
        assert parser.expression == "row['x'] == 1"

    def test_repr(self) -> None:
        parser = ExpressionParser("row['x'] == 1")
        assert repr(parser) == "ExpressionParser(\"row['x'] == 1\")"

    def test_dict_literal_in_expression(self) -> None:
        parser = ExpressionParser("row['key'] in {'a': 1, 'b': 2}")
        assert parser.evaluate({"key": "a"}) is True
        assert parser.evaluate({"key": "c"}) is False

    def test_negative_number_literal(self) -> None:
        parser = ExpressionParser("row['value'] > -10")
        assert parser.evaluate({"value": 0}) is True
        assert parser.evaluate({"value": -20}) is False

    def test_float_literal(self) -> None:
        parser = ExpressionParser("row['ratio'] < 0.5")
        assert parser.evaluate({"ratio": 0.3}) is True
        assert parser.evaluate({"ratio": 0.7}) is False

    def test_multiple_and_conditions(self) -> None:
        parser = ExpressionParser("row['a'] == 1 and row['b'] == 2 and row['c'] == 3")
        assert parser.evaluate({"a": 1, "b": 2, "c": 3}) is True
        assert parser.evaluate({"a": 1, "b": 2, "c": 4}) is False

    def test_multiple_or_conditions(self) -> None:
        parser = ExpressionParser(
            "row['status'] == 'a' or row['status'] == 'b' or row['status'] == 'c'"
        )
        assert parser.evaluate({"status": "b"}) is True
        assert parser.evaluate({"status": "d"}) is False


class TestExpressionParserRealWorldExamples:
    """Test real-world gate condition examples."""

    def test_confidence_threshold_gate(self) -> None:
        """Classic confidence threshold routing."""
        parser = ExpressionParser("row['confidence'] >= 0.85")
        assert parser.evaluate({"confidence": 0.9, "label": "positive"}) is True
        assert parser.evaluate({"confidence": 0.7, "label": "positive"}) is False

    def test_status_routing_gate(self) -> None:
        """Route based on status field."""
        parser = ExpressionParser(
            "row['status'] in ['approved', 'verified'] and row.get('errors') is None"
        )
        assert parser.evaluate({"status": "approved", "data": "..."}) is True
        assert parser.evaluate({"status": "approved", "errors": ["err"]}) is False
        assert parser.evaluate({"status": "pending", "data": "..."}) is False

    def test_multi_field_validation_gate(self) -> None:
        """Validate multiple required fields present."""
        parser = ExpressionParser(
            "row.get('name') is not None and row.get('email') is not None"
        )
        assert parser.evaluate({"name": "John", "email": "john@example.com"}) is True
        assert parser.evaluate({"name": "John"}) is False

    def test_amount_range_gate(self) -> None:
        """Check if amount is within acceptable range."""
        parser = ExpressionParser("row['amount'] > 0 and row['amount'] <= 10000")
        assert parser.evaluate({"amount": 5000}) is True
        assert parser.evaluate({"amount": 0}) is False
        assert parser.evaluate({"amount": 15000}) is False

    def test_category_with_score_gate(self) -> None:
        """Route high-confidence items in specific categories."""
        parser = ExpressionParser(
            "(row['category'] == 'urgent' and row['score'] >= 0.9) or "
            "(row['category'] == 'normal' and row['score'] >= 0.7)"
        )
        assert parser.evaluate({"category": "urgent", "score": 0.95}) is True
        assert parser.evaluate({"category": "urgent", "score": 0.8}) is False
        assert parser.evaluate({"category": "normal", "score": 0.75}) is True
        assert parser.evaluate({"category": "normal", "score": 0.5}) is False


# =============================================================================
# FUZZ TESTING
# =============================================================================
# Property-based fuzz tests using Hypothesis to verify parser security.
# These tests ensure the parser handles arbitrary malformed input without:
# 1. Crashing with unhandled exceptions
# 2. Executing malicious code
# 3. Leaking internal state
# =============================================================================

# Strategy for generating random character strings
random_chars = st.text(
    alphabet=st.sampled_from(
        string.ascii_letters + string.digits + string.punctuation + " \t\n\r"
    ),
    min_size=0,
    max_size=500,
)

# Strategy for Python keywords that might cause issues
python_keywords = st.sampled_from(
    [
        "import",
        "from",
        "eval",
        "exec",
        "compile",
        "open",
        "lambda",
        "def",
        "class",
        "for",
        "while",
        "if",
        "else",
        "elif",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "raise",
        "assert",
        "return",
        "yield",
        "await",
        "async",
        "global",
        "nonlocal",
        "pass",
        "break",
        "continue",
        "del",
        "__import__",
        "__builtins__",
        "__class__",
        "__dict__",
        "__globals__",
        "__code__",
        "__name__",
        "__doc__",
        "__module__",
        "os",
        "sys",
        "subprocess",
        "socket",
        "pickle",
        "marshal",
    ]
)

# Strategy for common attack patterns
malicious_patterns = st.sampled_from(
    [
        "__import__('os').system('echo pwned')",
        "eval('1+1')",
        "exec('print(1)')",
        "compile('1', '', 'eval')",
        "open('/etc/passwd').read()",
        "lambda: 1",
        "(lambda x: x)(1)",
        "[x for x in [1,2,3]]",
        "{x: x for x in [1,2,3]}",
        "{x for x in [1,2,3]}",
        "(x for x in [1,2,3])",
        "(x := 1)",
        "row.__class__.__bases__[0].__subclasses__()",
        "row.__dict__",
        "row.__getattribute__('keys')()",
        "getattr(row, 'get')",
        "hasattr(row, 'get')",
        "type(row)",
        "dir(row)",
        "vars(row)",
        "globals()",
        "locals()",
        "breakpoint()",
        "help(row)",
        "input('>')",
        "print('pwned')",
        "f'{__import__(\"os\")}'",
        "f'{1+1}'",
        "row['x'].__class__",
        "row.items()",
        "row.keys()",
        "row.values()",
        "row.pop('x')",
        "row.update({})",
        "row.clear()",
        "[*row]",
        "{**row}",
    ]
)

# Strategy for operators and punctuation
operators = st.sampled_from(
    [
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "**",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "and",
        "or",
        "not",
        "in",
        "is",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
        "@",
        ":=",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
    ]
)

# Strategy for bracket types
brackets = st.sampled_from(["(", ")", "[", "]", "{", "}", "'", '"', "'''", '"""'])

# Strategy for unicode characters that might cause issues
unicode_chars = st.text(
    alphabet=st.sampled_from(
        "\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008\u0009\u000a\u000b"
        "\u000c\u000d\u000e\u000f\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017"
        "\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f"  # Control chars
        "\u200b\u200c\u200d\ufeff"  # Zero-width chars
        "\u202a\u202b\u202c\u202d\u202e"  # Bidi overrides
        "\uff01\uff02\uff03"  # Fullwidth punctuation
        "\U0001f600\U0001f601\U0001f602"  # Emoji
    ),
    min_size=0,
    max_size=50,
)


# Composite strategy: build random expression-like strings
@st.composite
def expression_like_input(draw: st.DrawFn) -> str:
    """Generate strings that vaguely resemble expressions."""
    parts = []
    num_parts = draw(st.integers(min_value=0, max_value=10))

    for _ in range(num_parts):
        choice = draw(
            st.sampled_from(
                [
                    "keyword",
                    "operator",
                    "bracket",
                    "text",
                    "number",
                    "row_access",
                    "unicode",
                ]
            )
        )

        if choice == "keyword":
            parts.append(draw(python_keywords))
        elif choice == "operator":
            parts.append(draw(operators))
        elif choice == "bracket":
            parts.append(draw(brackets))
        elif choice == "text":
            parts.append(draw(st.text(max_size=20)))
        elif choice == "number":
            parts.append(str(draw(st.integers() | st.floats(allow_nan=True))))
        elif choice == "row_access":
            field = draw(st.text(min_size=1, max_size=10))
            parts.append(f"row['{field}']")
        elif choice == "unicode":
            parts.append(draw(unicode_chars))

    return " ".join(parts)


@st.composite
def nested_expression(draw: st.DrawFn) -> str:
    """Generate deeply nested bracket expressions."""
    depth = draw(st.integers(min_value=1, max_value=50))
    inner = draw(st.sampled_from(["row['x']", "1", "'str'", "True", "x"]))

    expr = inner
    for _ in range(depth):
        bracket_type = draw(st.sampled_from(["paren", "bracket", "brace"]))
        if bracket_type == "paren":
            expr = f"({expr})"
        elif bracket_type == "bracket":
            expr = f"[{expr}]"
        else:
            expr = f"{{{expr}}}"

    return expr


class TestExpressionParserFuzz:
    """Fuzz tests for expression parser security.

    These tests verify that the parser handles arbitrary malformed input
    without crashing or executing code. Defense-in-depth testing for
    config expressions that might come from tampered YAML files.
    """

    # Expected exception types - anything else is a parser bug
    ALLOWED_EXCEPTIONS = (
        ExpressionSecurityError,
        ExpressionSyntaxError,
    )

    def _assert_safe_parse(self, expression: str) -> None:
        """Assert that parsing either succeeds or fails with expected error type.

        The parser MUST NOT:
        1. Raise an unexpected exception type (unhandled edge case)
        2. Execute any code during parsing (no side effects)
        3. Crash Python
        """
        try:
            parser = ExpressionParser(expression)
            # If parsing succeeds, the expression must be in the safe subset.
            # Try evaluating with empty row to ensure evaluator also handles it.
            # Expected runtime errors (KeyError, TypeError, ZeroDivisionError) are
            # not parser bugs - they're valid expressions failing on empty row data.
            with contextlib.suppress(KeyError, TypeError, ZeroDivisionError):
                parser.evaluate({})
        except self.ALLOWED_EXCEPTIONS:
            # Security or syntax error - this is expected for malformed input
            pass
        except Exception as e:
            # Any other exception is a parser bug - fail the test
            pytest.fail(
                f"Unexpected exception {type(e).__name__} for input {expression!r}: {e}"
            )

    @given(expression=random_chars)
    @settings(max_examples=200, deadline=None)
    def test_random_characters(self, expression: str) -> None:
        """Random character strings should not crash the parser."""
        self._assert_safe_parse(expression)

    @given(expression=malicious_patterns)
    @settings(max_examples=100, deadline=None)
    def test_malicious_patterns(self, expression: str) -> None:
        """Known malicious patterns should be rejected, not crash."""
        self._assert_safe_parse(expression)

    @given(expression=expression_like_input())
    @settings(max_examples=300, deadline=None)
    def test_expression_like_input(self, expression: str) -> None:
        """Random expression-like strings should not crash the parser."""
        self._assert_safe_parse(expression)

    @given(expression=nested_expression())
    @settings(max_examples=100, deadline=None)
    def test_deeply_nested_brackets(self, expression: str) -> None:
        """Deeply nested brackets should not crash or cause stack overflow."""
        self._assert_safe_parse(expression)

    @given(expression=unicode_chars)
    @settings(max_examples=100, deadline=None)
    def test_unicode_and_control_characters(self, expression: str) -> None:
        """Unicode and control characters should not crash the parser."""
        self._assert_safe_parse(expression)

    @given(
        prefix=random_chars,
        valid_expr=st.sampled_from(
            [
                "row['x'] == 1",
                "row.get('y') is None",
                "row['a'] > 0 and row['b'] < 10",
            ]
        ),
        suffix=random_chars,
    )
    @settings(max_examples=200, deadline=None)
    def test_valid_expression_with_garbage(
        self, prefix: str, valid_expr: str, suffix: str
    ) -> None:
        """Valid expressions surrounded by garbage should not crash."""
        expression = prefix + valid_expr + suffix
        self._assert_safe_parse(expression)

    def test_deterministic_fuzz_with_seed(self) -> None:
        """Deterministic fuzz test with 1000+ inputs for reproducibility.

        Uses a fixed seed so failures are reproducible. This supplements
        the Hypothesis tests with a simple random approach.
        """
        rng = random.Random(42)  # Fixed seed for reproducibility

        # Character pools for generating inputs
        all_chars = (
            string.ascii_letters
            + string.digits
            + string.punctuation
            + " \t\n\r"
            + "\x00\x01\x02\x03"  # Null and control chars
        )

        dangerous_fragments = [
            "__import__",
            "os.system",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "lambda",
            "lambda:",
            "[x for x",
            "{x: x for",
            "{x for x",
            "(x for x",
            ":=",
            ".__class__",
            ".__dict__",
            ".__globals__",
            ".__code__",
            ".mro(",
            ".__subclasses__(",
            "breakpoint(",
            "globals(",
            "locals(",
            "getattr(",
            "setattr(",
            "delattr(",
            "hasattr(",
            "f'",
            'f"',
            "f'''",
            'f"""',
            "*row",
            "**row",
            "[*",
            "{**",
        ]

        inputs_tested = 0

        # Test 500 random character strings
        for _ in range(500):
            length = rng.randint(0, 200)
            expression = "".join(rng.choice(all_chars) for _ in range(length))
            self._assert_safe_parse(expression)
            inputs_tested += 1

        # Test 300 combinations of dangerous fragments
        for _ in range(300):
            num_fragments = rng.randint(1, 5)
            fragments = [rng.choice(dangerous_fragments) for _ in range(num_fragments)]
            # Intersperse with random chars
            parts = []
            for frag in fragments:
                parts.append(
                    "".join(rng.choice(all_chars) for _ in range(rng.randint(0, 10)))
                )
                parts.append(frag)
            expression = "".join(parts)
            self._assert_safe_parse(expression)
            inputs_tested += 1

        # Test 200 deeply nested expressions
        for _ in range(200):
            depth = rng.randint(1, 100)
            expr = "row['x']"
            for _ in range(depth):
                bracket = rng.choice(["(", "[", "{"])
                close = {"(": ")", "[": "]", "{": "}"}[bracket]
                expr = bracket + expr + close
            self._assert_safe_parse(expr)
            inputs_tested += 1

        # Test 100 very long expressions
        for _ in range(100):
            length = rng.randint(1000, 5000)
            expression = "".join(rng.choice(all_chars) for _ in range(length))
            self._assert_safe_parse(expression)
            inputs_tested += 1

        assert inputs_tested >= 1000, f"Expected 1000+ inputs, got {inputs_tested}"

    def test_null_byte_injection(self) -> None:
        """Null bytes in expressions should not cause crashes."""
        test_cases = [
            "\x00",
            "row['x']\x00",
            "\x00row['x']",
            "row['\x00x']",
            "row['x']\x00 == 1",
            "\x00\x00\x00",
            "row['x'] == '\x00value'",
        ]
        for expr in test_cases:
            self._assert_safe_parse(expr)

    def test_very_long_expressions(self) -> None:
        """Very long expressions should not cause memory issues."""
        test_cases = [
            "a" * 10000,
            "row['x'] " * 1000,
            "(" * 500 + "row['x']" + ")" * 500,
            "row['x'] == 1 and " * 500 + "True",
        ]
        for expr in test_cases:
            self._assert_safe_parse(expr)

    def test_encoding_edge_cases(self) -> None:
        """Various encoding edge cases should not crash."""
        test_cases = [
            "",  # Empty string
            " ",  # Just whitespace
            "\t\n\r",  # Just control chars
            "\u200b",  # Zero-width space
            "\ufeff",  # BOM
            "\u202e",  # Right-to-left override
            "row['x'] == '\u202eevil\u202c'",  # Bidi override in string
            "\U0001f600",  # Emoji
            "row['\U0001f600']",  # Emoji as field name
        ]
        for expr in test_cases:
            self._assert_safe_parse(expr)

    def test_incomplete_expressions(self) -> None:
        """Incomplete/truncated expressions should not crash."""
        valid_expr = "row['status'] == 'active' and row['count'] > 0"
        # Test all truncation points
        for i in range(len(valid_expr)):
            truncated = valid_expr[:i]
            self._assert_safe_parse(truncated)

    def test_operator_combinations(self) -> None:
        """Unusual operator combinations should not crash."""
        operators = [
            "+",
            "-",
            "*",
            "/",
            "//",
            "%",
            "**",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "and",
            "or",
            "not",
            "in",
            "is",
            "&",
            "|",
            "^",
            "~",
            "<<",
            ">>",
            ":=",
        ]
        rng = random.Random(123)
        for _ in range(100):
            num_ops = rng.randint(2, 10)
            expr = " ".join(rng.choice(operators) for _ in range(num_ops))
            self._assert_safe_parse(expr)

    def test_mixed_quotes_and_brackets(self) -> None:
        """Mixed quote and bracket styles should not crash."""
        test_cases = [
            "row['x\"]",
            "row[\"x']",
            "row[\"x']",
            "row['x\"][",
            "[('{",
            "})]'\"",
            "'''row'''",
            '"""row"""',
            "row['''x''']",
            'row["""x"""]',
        ]
        for expr in test_cases:
            self._assert_safe_parse(expr)
